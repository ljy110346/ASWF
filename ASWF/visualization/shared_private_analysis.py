from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import write_json


def _mean_norm_per_scale(array: np.ndarray | None) -> list[float]:
    if array is None:
        return []
    return [float(np.mean(np.linalg.norm(array[:, scale_idx, :], axis=-1))) for scale_idx in range(array.shape[1])]


def analyze_shared_private_ratio(
    fused_shared: np.ndarray | None,
    fused_private: np.ndarray | None,
    private_ir: np.ndarray | None,
    private_raman: np.ndarray | None,
    output_path: Path | str,
) -> None:
    shared_norms = _mean_norm_per_scale(fused_shared)
    private_norms = _mean_norm_per_scale(fused_private)
    ir_private_norms = _mean_norm_per_scale(private_ir)
    raman_private_norms = _mean_norm_per_scale(private_raman)
    payload = {
        "shared_norm_per_scale": shared_norms,
        "private_norm_per_scale": private_norms,
        "private_ir_norm_per_scale": ir_private_norms,
        "private_raman_norm_per_scale": raman_private_norms,
    }
    write_json(payload, output_path)

