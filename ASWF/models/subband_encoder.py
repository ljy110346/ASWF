from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScaleSharedEncoder(nn.Module):
    """Each scale owns one encoder, and IR/Raman share that scale's full parameter set."""

    def __init__(
        self,
        hidden_channels: int,
        projection_dim: int,
        num_layers: int = 2,
        kernel_size: int = 5,
        pooled_length: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(num_layers):
            layers.append(ConvEncoderBlock(in_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout))
            in_channels = hidden_channels
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooled_length, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.projection(pooled)

