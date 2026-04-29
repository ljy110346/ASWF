from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZScoreTransform:
    mean: np.ndarray
    std: np.ndarray
    eps: float

    def apply(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.mean) / (self.std + self.eps)).astype(np.float32, copy=False)


def fit_feature_zscore(array: np.ndarray, train_indices: np.ndarray, eps: float) -> ZScoreTransform:
    train_slice = array[train_indices]
    mean = train_slice.mean(axis=0, keepdims=True)
    std = train_slice.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return ZScoreTransform(mean=mean.astype(np.float32), std=std.astype(np.float32), eps=eps)

