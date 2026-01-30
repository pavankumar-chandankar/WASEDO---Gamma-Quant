from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousConv2d(nn.Module):
    """
    Continuous-time (coordinate-conditioned) 2D convolution with a time-only kernel (k_t, 1).

    Backward compatible:
      - If kernel_support_s is None OR forward() is called without sample_rate,
        uses the fixed discrete kernel size given at init.

    Multi-rate extension:
      - If kernel_support_s is provided and forward() receives sample_rate (Hz),
        the effective kernel length is computed per-forward as:
            k_t = round(kernel_support_s * sample_rate)
        then forced to be odd and >= 1 (stable 'same' padding).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        rank: int = 8,
        mlp_hidden_dim: int = 32,
        bias: bool = True,
        padding: str = "valid",
        *,
        kernel_support_s: Optional[float] = None,
        coord_min: float = -1.0,
        coord_max: float = 1.0,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "kernel_size must be int or (k_t, 1)"
            assert kernel_size[1] == 1, "ContinuousConv2d only supports kernel_size=(k_t, 1)"
            self.kernel_size_time = int(kernel_size[0])
        else:
            self.kernel_size_time = int(kernel_size)

        if self.kernel_size_time < 1:
            raise ValueError("kernel_size_time must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if mlp_hidden_dim < 1:
            raise ValueError("mlp_hidden_dim must be >= 1")

        assert padding in ("valid", "same"), 'padding must be "valid" or "same"'
        self.padding_mode = padding

        if kernel_support_s is not None and float(kernel_support_s) <= 0.0:
            raise ValueError("kernel_support_s must be > 0")
        if float(coord_max) <= float(coord_min):
            raise ValueError("coord_max must be > coord_min")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.rank = int(rank)
        self.mlp_hidden_dim = int(mlp_hidden_dim)

        self.kernel_support_s = float(kernel_support_s) if kernel_support_s is not None else None
        self.coord_min = float(coord_min)
        self.coord_max = float(coord_max)

        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, rank),
        )

        # [O, I, R]
        self.channel_weights = nn.Parameter(torch.randn(out_channels, in_channels, rank) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @staticmethod
    def _make_odd(k: int) -> int:
        if k <= 1:
            return 1
        return k if (k % 2 == 1) else (k + 1)

    def _effective_kernel_size_time(self, sample_rate: Optional[float]) -> int:
        if self.kernel_support_s is None or sample_rate is None:
            return self._make_odd(self.kernel_size_time)

        k = int(round(float(self.kernel_support_s) * float(sample_rate)))
        k = max(1, k)
        return self._make_odd(k)

    def _build_kernel(self, k_t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # positions: [K, 1]
        positions = torch.linspace(self.coord_min, self.coord_max, k_t, device=device, dtype=dtype).unsqueeze(-1)
        # g: [K, R]
        g = self.mlp(positions)
        # K_t: [O, I, K]
        K_t = torch.einsum("oir,kr->oik", self.channel_weights, g)
        # K: [O, I, K, 1]
        return K_t.unsqueeze(-1)

    def forward(self, x: torch.Tensor, sample_rate: Optional[float] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"ContinuousConv2d expected 4D input [B, C_in, T, W], got shape {tuple(x.shape)}"
            )

        k_t = self._effective_kernel_size_time(sample_rate)
        K = self._build_kernel(k_t, device=x.device, dtype=x.dtype)

        pad_t = (k_t // 2) if (self.padding_mode == "same") else 0
        return F.conv2d(x, K, bias=self.bias, padding=(pad_t, 0))
