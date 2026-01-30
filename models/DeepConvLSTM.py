from typing import Sequence

from torch import nn
import torch
import torch.nn.functional as F

from models.gamma_quant import gammaFunction
from models.gamma_quant import UniformQuantizerSTE
from models.continuous_conv import ContinuousConv2d


class DeepConvLSTM(nn.Module):
    """
    Single checkpoint contains 4 frequency-specific ContinuousConv branches:
      - branches["50"], branches["25"], branches["12"], branches["6"]

    Training (LOSO loop unchanged):
      - If conv_type == "continuous", we do multi-rate training inside forward():
        pick one of {50,25,12,6}, resample window to that length, and train that branch.

    Inference:
      - chooses branch based on input length T (snapped to nearest supported rate).

    Debug:
      - self.last_sample_rate is set to the Hz branch used for the last forward().
        The training loop can read this to print branch usage stats.
    """

    def __init__(
        self,
        channels,
        classes,
        window_size,
        conv_kernels=64,
        conv_kernel_size=5,
        lstm_units=128,
        lstm_layers=2,
        dropout=0.5,
        gamma_quant=None,
        quant_bits=4,
        apply_gamma="global",
        gamma_type="id",
        conv_type="standard",
        conv_rank=8,
        conv_mlp_hidden_dim=32,
        supported_sample_rates: Sequence[int] = (50, 25, 12, 6),
        multirate_training: bool = True,
    ):
        super(DeepConvLSTM, self).__init__()

        self.gamma_quant = gamma_quant
        self.channels = channels
        self.classes = classes
        self.window_size = window_size
        self.conv_kernels = conv_kernels
        self.conv_kernel_size = conv_kernel_size
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dropout_p = dropout
        self.conv_type = conv_type

        self.supported_sample_rates = [int(x) for x in supported_sample_rates]
        self.multirate_training = bool(multirate_training) and (conv_type == "continuous")

        # Debug: used by train.py to print branch usage stats
        self.last_sample_rate = None  # int | None

        # In WEAR configs here: 1-second windows, so window_size == 50 for the base config.
        base_fs = float(window_size) if float(window_size) > 0 else 50.0
        self.kernel_support_s = float(conv_kernel_size) / base_fs

        if conv_type == "standard":
            self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
            self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.final_seq_len = window_size - (conv_kernel_size - 1) * 4

        elif conv_type == "continuous":
            # 4 separate branches inside one model checkpoint
            self.branches = nn.ModuleDict()
            for fs in self.supported_sample_rates:
                self.branches[str(int(fs))] = nn.ModuleList(
                    [
                        ContinuousConv2d(
                            1,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=conv_rank,
                            mlp_hidden_dim=conv_mlp_hidden_dim,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                        ContinuousConv2d(
                            conv_kernels,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=conv_rank,
                            mlp_hidden_dim=conv_mlp_hidden_dim,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                        ContinuousConv2d(
                            conv_kernels,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=conv_rank,
                            mlp_hidden_dim=conv_mlp_hidden_dim,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                        ContinuousConv2d(
                            conv_kernels,
                            conv_kernels,
                            (conv_kernel_size, 1),
                            rank=conv_rank,
                            mlp_hidden_dim=conv_mlp_hidden_dim,
                            padding="same",
                            kernel_support_s=self.kernel_support_s,
                        ),
                    ]
                )
            self.final_seq_len = window_size

        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.lstm = nn.LSTM(
            channels * conv_kernels,
            lstm_units,
            num_layers=lstm_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()

        self.apply_gamma = apply_gamma
        if self.gamma_quant is not None:
            if apply_gamma == "global":
                self.gamma_function = gammaFunction(init=gamma_type, offset=0.0)
            elif apply_gamma == "local":
                self.gamma_functions = nn.ModuleList(
                    [gammaFunction(init=gamma_type, offset=0.0) for _ in range(channels)]
                )
            else:
                raise ValueError(f"Unknown apply_gamma mode: {apply_gamma}")

            self.quantizer = UniformQuantizerSTE(n_bits=quant_bits)

    def _snap_sample_rate(self, T: int) -> int:
        return int(min(self.supported_sample_rates, key=lambda fs: abs(int(fs) - int(T))))

    def _resample_time(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: [B, T, C] -> [B, target_len, C]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,C], got {tuple(x.shape)}")
        B, T, C = x.shape
        if int(T) == int(target_len):
            return x

        xt = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)
        xt = F.interpolate(xt, size=int(target_len), mode="linear", align_corners=False)
        xt = xt.view(B, C, int(target_len)).permute(0, 2, 1).contiguous()
        return xt

    def forward(self, x):
        # x: [B, T, C]
        if x.ndim != 3:
            raise ValueError(f"DeepConvLSTM expected [B,T,C], got shape {tuple(x.shape)}")

        self.last_sample_rate = None  # reset every forward

        B, T, C = x.shape

        sample_rate = None
        if self.conv_type == "continuous":
            if self.training and self.multirate_training:
                idx = torch.randint(low=0, high=len(self.supported_sample_rates), size=(1,), device=x.device).item()
                fs = int(self.supported_sample_rates[idx])
                x = self._resample_time(x, target_len=fs)
                sample_rate = fs
            else:
                fs = self._snap_sample_rate(T)
                if int(T) != int(fs):
                    x = self._resample_time(x, target_len=int(fs))
                sample_rate = int(fs)

            self.last_sample_rate = int(sample_rate)

        x = x.unsqueeze(1)  # [B, 1, T, C]

        if self.gamma_quant == "quantize":
            x = self.quantizer(x)
        elif self.gamma_quant == "quantize_and_gamma":
            if hasattr(self, "gamma_functions"):
                x_split = torch.split(x, 1, dim=-1)
                x = torch.cat([self.gamma_functions[i](x_split[i]) for i in range(len(x_split))], dim=-1)
                x = self.quantizer(x)
            else:
                x = self.quantizer(self.gamma_function(x))

        if self.conv_type == "standard":
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))

        elif self.conv_type == "continuous":
            branch = self.branches[str(int(sample_rate))]
            x = self.activation(branch[0](x, sample_rate=sample_rate))
            x = self.activation(branch[1](x, sample_rate=sample_rate))
            x = self.activation(branch[2](x, sample_rate=sample_rate))
            x = self.activation(branch[3](x, sample_rate=sample_rate))

        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

        # [B, K, T, C] -> [T, B, C, K] -> [T, B, C*K]
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x, _ = self.lstm(x)
        x = x[-1, :, :]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)
