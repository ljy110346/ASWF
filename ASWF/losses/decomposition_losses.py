from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F


def _normalize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(features, dim=-1, eps=eps)


def _mean_or_zero(values: Iterable[torch.Tensor], device: torch.device) -> torch.Tensor:
    materialized = list(values)
    if not materialized:
        return torch.zeros((), device=device)
    return torch.stack(materialized).mean()


def shared_consistency_loss(shared_ir, shared_raman) -> torch.Tensor:
    device = shared_ir[0].device if shared_ir else torch.device("cpu")
    values = [
        torch.mean((_normalize(ir_feat) - _normalize(ra_feat)) ** 2)
        for ir_feat, ra_feat in zip(shared_ir, shared_raman)
    ]
    return _mean_or_zero(values, device=device)


def orthogonality_loss(shared_ir, shared_raman, private_ir, private_raman) -> torch.Tensor:
    device = shared_ir[0].device if shared_ir else torch.device("cpu")
    values = []
    for u_ir, u_ra, z_ir, z_ra in zip(shared_ir, shared_raman, private_ir, private_raman):
        values.append(torch.mean(torch.sum(_normalize(u_ir) * _normalize(z_ir), dim=-1) ** 2))
        values.append(torch.mean(torch.sum(_normalize(u_ra) * _normalize(z_ra), dim=-1) ** 2))
    return _mean_or_zero(values, device=device)


def stability_loss(clean_shared_ir, clean_shared_raman, clean_private_ir, clean_private_raman, outputs) -> torch.Tensor:
    if not outputs["perturbed_shared_ir"]:
        device = clean_shared_ir[0].device if clean_shared_ir else torch.device("cpu")
        return torch.zeros((), device=device)

    device = clean_shared_ir[0].device
    values = []
    for clean, perturbed in zip(clean_shared_ir, outputs["perturbed_shared_ir"]):
        values.append(torch.mean((_normalize(clean) - _normalize(perturbed)) ** 2))
    for clean, perturbed in zip(clean_shared_raman, outputs["perturbed_shared_raman"]):
        values.append(torch.mean((_normalize(clean) - _normalize(perturbed)) ** 2))
    for clean, perturbed in zip(clean_private_ir, outputs["perturbed_private_ir"]):
        values.append(torch.mean((_normalize(clean) - _normalize(perturbed)) ** 2))
    for clean, perturbed in zip(clean_private_raman, outputs["perturbed_private_raman"]):
        values.append(torch.mean((_normalize(clean) - _normalize(perturbed)) ** 2))
    return _mean_or_zero(values, device=device)


def compute_decomposition_loss(outputs: Dict[str, object], loss_config) -> Dict[str, torch.Tensor]:
    shared_ir = outputs["shared_ir"]
    shared_raman = outputs["shared_raman"]
    private_ir = outputs["private_ir"]
    private_raman = outputs["private_raman"]

    if not shared_ir:
        zero = outputs["encoded_ir"][0].new_zeros(())
        return {"L_cons": zero, "L_orth": zero, "L_stab": zero, "L_decomp": zero}

    l_cons = shared_consistency_loss(shared_ir, shared_raman)
    l_orth = orthogonality_loss(shared_ir, shared_raman, private_ir, private_raman)
    l_stab = stability_loss(shared_ir, shared_raman, private_ir, private_raman, outputs)
    l_decomp = (
        loss_config.lambda_cons * l_cons
        + loss_config.lambda_orth * l_orth
        + loss_config.lambda_stab * l_stab
    )
    return {"L_cons": l_cons, "L_orth": l_orth, "L_stab": l_stab, "L_decomp": l_decomp}
