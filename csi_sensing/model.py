"""Model definitions for CSI sensing."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, pool: bool = True):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class CSISensingModel(nn.Module):
    """Shared encoder with dual heads for presence and distance."""

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4, pool=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_dim = base_channels * 4
        self.head_empty = nn.Linear(hidden_dim, 1)
        self.head_dist = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expect x of shape [B, 1, T, S]
        features = self.encoder(x)
        pooled = self.global_pool(features).flatten(1)
        logits_empty = self.head_empty(pooled).squeeze(-1)
        dist_pred = self.softplus(self.head_dist(pooled).squeeze(-1))
        return logits_empty, dist_pred

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
