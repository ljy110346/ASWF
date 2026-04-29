from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from ..datasets.transforms import fit_feature_zscore


@dataclass(frozen=True)
class PreprocessingArtifacts:
    ir: np.ndarray
    raman: np.ndarray
    metadata: Dict[str, object]


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_fold_split(
    labels: np.ndarray,
    train_val_indices: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    local_labels = labels[train_val_indices]
    train_local, val_local = next(splitter.split(np.zeros_like(local_labels), local_labels))
    return train_val_indices[train_local], train_val_indices[val_local]


def subsample_indices_stratified(
    labels: np.ndarray,
    indices: np.ndarray,
    ratio: float | None,
    seed: int,
) -> np.ndarray:
    if ratio is None or ratio >= 1.0:
        return np.asarray(indices, dtype=np.int64)
    if ratio <= 0.0:
        raise ValueError(f"train_subset_ratio must be in (0, 1], got {ratio}.")

    indices = np.asarray(indices, dtype=np.int64)
    local_labels = labels[indices]
    unique_labels = np.unique(local_labels)
    if indices.shape[0] <= unique_labels.shape[0]:
        return indices

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=ratio, random_state=seed)
    try:
        sampled_local, _ = next(splitter.split(np.zeros_like(local_labels), local_labels))
        return np.sort(indices[sampled_local])
    except ValueError:
        rng = np.random.default_rng(seed)
        mandatory = []
        remaining = []
        for label in unique_labels:
            label_indices = indices[local_labels == label]
            rng.shuffle(label_indices)
            mandatory.append(label_indices[0])
            remaining.extend(label_indices[1:].tolist())

        target_size = max(len(mandatory), int(round(indices.shape[0] * ratio)))
        target_size = min(target_size, indices.shape[0])
        if target_size == len(mandatory):
            return np.sort(np.asarray(mandatory, dtype=np.int64))

        remaining_array = np.asarray(remaining, dtype=np.int64)
        rng.shuffle(remaining_array)
        extra = remaining_array[: target_size - len(mandatory)]
        return np.sort(np.concatenate([np.asarray(mandatory, dtype=np.int64), extra], axis=0))


def apply_preprocessing(ir: np.ndarray, raman: np.ndarray, train_indices: np.ndarray, config) -> PreprocessingArtifacts:
    if config.spectral_normalization == "none":
        return PreprocessingArtifacts(ir=ir, raman=raman, metadata={"spectral_normalization": "none"})

    if config.spectral_normalization != "feature_zscore":
        raise ValueError(f"Unsupported normalization mode: {config.spectral_normalization}")

    ir_transform = fit_feature_zscore(ir, train_indices=train_indices, eps=config.normalization_eps)
    raman_transform = fit_feature_zscore(raman, train_indices=train_indices, eps=config.normalization_eps)
    return PreprocessingArtifacts(
        ir=ir_transform.apply(ir),
        raman=raman_transform.apply(raman),
        metadata={"spectral_normalization": "feature_zscore"},
    )


def class_counts(labels: np.ndarray, indices: Iterable[int]) -> Dict[str, int]:
    selected = labels[np.asarray(list(indices), dtype=np.int64)]
    values, counts = np.unique(selected, return_counts=True)
    mapping = {str(int(value)): int(count) for value, count in zip(values, counts)}
    mapping.setdefault("0", 0)
    mapping.setdefault("1", 0)
    return mapping


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
