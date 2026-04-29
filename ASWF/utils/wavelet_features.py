from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pywt


DEFAULT_WAVELET_STATS = ("mean", "std", "l1", "energy", "max_abs")
_SUPPORTED_STATS = frozenset(DEFAULT_WAVELET_STATS)


def extract_wavelet_stat_features(
    spectra: np.ndarray,
    wavelet_name: str = "db1",
    levels: int = 3,
    stats: Sequence[str] = DEFAULT_WAVELET_STATS,
    padding_mode: str = "symmetric",
) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError(f"Expected a 2D spectra matrix, got shape {spectra.shape}.")

    resolved_stats = _validate_stats(stats)
    signal_length = int(spectra.shape[1])
    resolved_levels = resolve_wavelet_levels(signal_length, wavelet_name, levels)
    feature_dim = (resolved_levels + 1) * len(resolved_stats)
    features = np.empty((spectra.shape[0], feature_dim), dtype=np.float32)

    for index, signal in enumerate(spectra):
        coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=resolved_levels, mode=padding_mode)
        feature_row: List[float] = []
        for coeff in coeffs:
            coeff = np.asarray(coeff, dtype=np.float32)
            abs_coeff = np.abs(coeff)
            for stat_name in resolved_stats:
                feature_row.append(_stat_value(stat_name, coeff, abs_coeff))
        features[index] = np.asarray(feature_row, dtype=np.float32)

    return features


def wavelet_stat_feature_names(
    signal_length: int,
    wavelet_name: str = "db1",
    levels: int = 3,
    stats: Sequence[str] = DEFAULT_WAVELET_STATS,
    prefix: str | None = None,
) -> List[str]:
    resolved_stats = _validate_stats(stats)
    resolved_levels = resolve_wavelet_levels(signal_length, wavelet_name, levels)

    band_names = [f"A{resolved_levels}"] + [f"D{level}" for level in range(resolved_levels, 0, -1)]
    prefix_text = f"{prefix}_" if prefix else ""
    names: List[str] = []
    for band_name in band_names:
        for stat_name in resolved_stats:
            names.append(f"{prefix_text}{band_name}_{stat_name}")
    return names


def build_pairwise_feature_names(
    left_names: Iterable[str],
    right_names: Iterable[str],
    prefix: str = "paired",
) -> List[str]:
    left = list(left_names)
    right = list(right_names)
    if len(left) != len(right):
        raise ValueError(f"Pairwise features require equal feature counts, got {len(left)} and {len(right)}.")

    pairwise_names: List[str] = []
    for name in left:
        pairwise_names.append(f"{prefix}_left_{name}")
    for name in right:
        pairwise_names.append(f"{prefix}_right_{name}")
    for left_name, right_name in zip(left, right):
        pairwise_names.append(f"{prefix}_delta_{left_name}__minus__{right_name}")
    for left_name, right_name in zip(left, right):
        pairwise_names.append(f"{prefix}_product_{left_name}__times__{right_name}")
    return pairwise_names


def resolve_wavelet_levels(signal_length: int, wavelet_name: str, levels: int) -> int:
    if signal_length <= 0:
        raise ValueError(f"Signal length must be positive, got {signal_length}.")
    if levels <= 0:
        raise ValueError(f"Wavelet levels must be positive, got {levels}.")

    wavelet = pywt.Wavelet(wavelet_name)
    max_levels = pywt.dwt_max_level(signal_length, wavelet.dec_len)
    if levels > max_levels:
        raise ValueError(
            f"Wavelet '{wavelet_name}' supports at most {max_levels} levels for signal length {signal_length}, "
            f"but got {levels}."
        )
    return levels


def _validate_stats(stats: Sequence[str]) -> tuple[str, ...]:
    if not stats:
        raise ValueError("At least one wavelet statistic is required.")

    resolved = tuple(stats)
    unsupported = sorted(set(resolved) - _SUPPORTED_STATS)
    if unsupported:
        raise ValueError(f"Unsupported wavelet statistics: {unsupported}. Supported stats: {sorted(_SUPPORTED_STATS)}.")
    return resolved


def _stat_value(stat_name: str, coeff: np.ndarray, abs_coeff: np.ndarray) -> float:
    if stat_name == "mean":
        return float(np.mean(coeff))
    if stat_name == "std":
        return float(np.std(coeff))
    if stat_name == "l1":
        return float(np.mean(abs_coeff))
    if stat_name == "energy":
        return float(np.mean(np.square(coeff)))
    if stat_name == "max_abs":
        return float(np.max(abs_coeff))
    raise ValueError(f"Unsupported wavelet statistic: {stat_name}")
