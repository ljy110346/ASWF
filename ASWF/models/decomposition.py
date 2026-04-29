from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _build_projector(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    hidden_dim = max(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class SharedPrivateDecomposer(nn.Module):
    def __init__(self, in_dim: int, shared_dim: int, private_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.shared_projector = _build_projector(in_dim, shared_dim, dropout)
        self.ir_private_projector = _build_projector(in_dim, private_dim, dropout)
        self.raman_private_projector = _build_projector(in_dim, private_dim, dropout)

    def forward_clean(self, h_ir: torch.Tensor, h_raman: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "u_ir": self.shared_projector(h_ir),
            "u_ra": self.shared_projector(h_raman),
            "z_ir": self.ir_private_projector(h_ir),
            "z_ra": self.raman_private_projector(h_raman),
        }

    def forward_perturbed(
        self,
        h_ir: torch.Tensor,
        h_raman: torch.Tensor,
        noise_std: float,
    ) -> Dict[str, torch.Tensor]:
        noise_ir = torch.randn_like(h_ir) * noise_std
        noise_ra = torch.randn_like(h_raman) * noise_std
        perturbed = self.forward_clean(h_ir + noise_ir, h_raman + noise_ra)
        return {
            "u_ir_tilde": perturbed["u_ir"],
            "u_ra_tilde": perturbed["u_ra"],
            "z_ir_tilde": perturbed["z_ir"],
            "z_ra_tilde": perturbed["z_ra"],
        }

    def forward(
        self,
        h_ir: torch.Tensor,
        h_raman: torch.Tensor,
        use_stability: bool = True,
        noise_std: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward_clean(h_ir=h_ir, h_raman=h_raman)
        if use_stability:
            outputs.update(self.forward_perturbed(h_ir=h_ir, h_raman=h_raman, noise_std=noise_std))
        return outputs

