from __future__ import annotations

from pathlib import Path
from typing import Dict

from ..utils.io import write_json


def build_wavenumber_backmap(wave_numbers: Dict[str, object], wavelet_info, output_path: Path | str) -> None:
    payload = {
        "wavelet_name": wavelet_info.wavelet_name,
        "wavelet_mode": wavelet_info.wavelet_mode,
        "J_common": wavelet_info.J_common,
        "notes": (
            "First-version placeholder interface. It records axis availability and fold-level wavelet metadata so "
            "later work can map scale importance back to disease-specific wavenumber coordinates."
        ),
        "modalities": {},
    }
    for modality, axis in wave_numbers.items():
        payload["modalities"][modality] = {
            "path": str(axis.path) if axis.path is not None else None,
            "sheet_name": axis.sheet_name,
            "is_valid": axis.is_valid,
            "axis_length": int(axis.values.shape[0]) if axis.values is not None else None,
            "warning": axis.warning,
        }
    write_json(payload, output_path)
