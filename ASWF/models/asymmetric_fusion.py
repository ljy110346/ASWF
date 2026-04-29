from __future__ import annotations

import torch
import torch.nn as nn


class ConservativeSharedFusion(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, u_ir: torch.Tensor, u_raman: torch.Tensor) -> torch.Tensor:
        pair_features = torch.cat([u_ir * u_raman, torch.abs(u_ir - u_raman)], dim=-1)
        gate = torch.sigmoid(self.gate(pair_features))
        consensus = 0.5 * (u_ir + u_raman)
        return gate * consensus


class DifferenceAwarePrivateFusion(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.feature = nn.Sequential(
            nn.Linear(feat_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.ir_value = nn.Linear(feat_dim, feat_dim)
        self.raman_value = nn.Linear(feat_dim, feat_dim)

    def forward(self, z_ir: torch.Tensor, z_raman: torch.Tensor) -> torch.Tensor:
        delta = z_ir - z_raman
        summary = torch.cat([z_ir, z_raman, delta, torch.abs(delta)], dim=-1)
        gate = torch.sigmoid(self.gate(summary))
        private_state = self.feature(summary)
        residual = self.ir_value(z_ir) + self.raman_value(z_raman)
        return gate * private_state + (1.0 - gate) * residual


class UnifiedFusion(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = out_dim or feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([left, right, torch.abs(left - right)], dim=-1))


class AverageFusion(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (left + right)
