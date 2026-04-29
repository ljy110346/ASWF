from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pywt
import torch
import torch.nn as nn


@dataclass(frozen=True)
class WaveletLevelInfo:
    wavelet_name: str
    wavelet_mode: str
    J_config: int
    J_ir_max: int
    J_raman_max: int
    J_common: int

    def to_dict(self) -> Dict[str, int | str]:
        return asdict(self)


def compute_common_wavelet_level(
    ir_len: int,
    raman_len: int,
    wavelet_name: str,
    mode: str,
    J_config: int,
) -> WaveletLevelInfo:
    wavelet = pywt.Wavelet(wavelet_name)
    j_ir_max = pywt.dwt_max_level(ir_len, wavelet.dec_len)
    j_raman_max = pywt.dwt_max_level(raman_len, wavelet.dec_len)
    j_common = min(J_config, j_ir_max, j_raman_max)
    if j_common < 1:
        raise ValueError(
            f"J_common must be at least 1 for wavelet='{wavelet_name}', got {j_common} "
            f"(J_ir_max={j_ir_max}, J_raman_max={j_raman_max}, J_config={J_config})."
        )
    return WaveletLevelInfo(
        wavelet_name=wavelet_name,
        wavelet_mode=mode,
        J_config=J_config,
        J_ir_max=j_ir_max,
        J_raman_max=j_raman_max,
        J_common=j_common,
    )


class WaveletDecomposer1D(nn.Module):
    """Fixed 1D DWT used as candidate evidence organization, not as generic preprocessing."""

    def __init__(self, wavelet: str = "db1", level: int = 3, mode: str = "zero", use_wavelet: bool = True) -> None:
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.use_wavelet = use_wavelet

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"Expected x to have shape [B, L] or [B, 1, L], got {tuple(x.shape)}")

        if not self.use_wavelet:
            return [x]

        x_np = x.detach().cpu().numpy()
        coeff_per_sample = [
            pywt.wavedec(sample[0], wavelet=self.wavelet, mode=self.mode, level=self.level) for sample in x_np
        ]
        num_bands = len(coeff_per_sample[0])
        subbands: List[torch.Tensor] = []

        for band_index in range(num_bands):
            # pywt order is [cA_J, cD_J, cD_{J-1}, ..., cD_1], which already matches
            # our s_0 ... s_J convention if band 0 is treated as the approximation band.
            stacked = np.stack([coeffs[band_index] for coeffs in coeff_per_sample], axis=0).astype(np.float32)
            subbands.append(torch.from_numpy(stacked).to(x.device).unsqueeze(1))
        return subbands
