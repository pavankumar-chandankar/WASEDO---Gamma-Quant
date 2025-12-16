from torch import nn
import torch

from models.gamma_quant import gammaFunction
from models.gamma_quant import UniformQuantizerSTE
from models.continuous_conv import ContinuousConv2d


class DeepConvLSTM(nn.Module):
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
        if conv_type == "standard":
            self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
            self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        elif conv_type == "continuous":
            self.conv1 = ContinuousConv2d(
                1,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.conv2 = ContinuousConv2d(
                conv_kernels,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.conv3 = ContinuousConv2d(
                conv_kernels,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.conv4 = ContinuousConv2d(
                conv_kernels,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
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
                    [
                        gammaFunction(init=gamma_type, offset=0.0)
                        for _ in range(channels)
                    ]
                )
            else:
                raise ValueError(f"Unknown apply_gamma mode: {apply_gamma}")

            self.quantizer = UniformQuantizerSTE(n_bits=quant_bits)

    def forward(self, x):
        x = x.unsqueeze(1)

        if self.gamma_quant == "quantize":
            x = self.quantizer(x)
        elif self.gamma_quant == "quantize_and_gamma":
            if hasattr(self, "gamma_functions"):
                x_split = torch.split(x, 1, dim=-1)
                x = torch.cat(
                    [
                        self.gamma_functions[i](x_split[i])
                        for i in range(len(x_split))
                    ],
                    dim=-1,
                )
                x = self.quantizer(x)
            else:
                x = self.quantizer(self.gamma_function(x))

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x, _ = self.lstm(x)
        x = x[-1, :, :]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)
