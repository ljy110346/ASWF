from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    acc: float
    bacc: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    f1: float
    auc: float
    mcc: float
    threshold: float | None
    tn: int
    fp: int
    fn: int
    tp: int
    loss: float | None = None

    def to_dict(self) -> Dict[str, float | int | None]:
        return asdict(self)


def safe_auc(y_true: np.ndarray, positive_prob: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, positive_prob))


def select_decision_threshold(
    y_true: Iterable[int] | np.ndarray,
    positive_prob: Iterable[float] | np.ndarray,
    metric: str = "bacc",
) -> float:
    y_true = np.asarray(list(y_true), dtype=np.int64)
    positive_prob = np.asarray(list(positive_prob), dtype=np.float64)

    candidates = np.unique(np.concatenate(([0.05], positive_prob, [0.95])))
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in candidates:
        pred = (positive_prob >= threshold).astype(np.int64)
        if metric == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, pred)
        else:
            score = balanced_accuracy_score(y_true, pred)

        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold


def compute_binary_metrics(
    y_true: Iterable[int] | np.ndarray,
    positive_prob: Iterable[float] | np.ndarray,
    pred: Iterable[int] | np.ndarray,
    threshold: float | None,
    loss: float | None = None,
) -> MetricBundle:
    y_true = np.asarray(list(y_true), dtype=np.int64)
    positive_prob = np.asarray(list(positive_prob), dtype=np.float64)
    pred = np.asarray(list(pred), dtype=np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return MetricBundle(
        acc=float(accuracy_score(y_true, pred)),
        bacc=float(balanced_accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        auc=safe_auc(y_true, positive_prob),
        mcc=float(matthews_corrcoef(y_true, pred)),
        threshold=threshold,
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        loss=loss,
    )


def summarize_metric_series(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "ci95_low": float(np.percentile(valid, 2.5)),
        "ci95_high": float(np.percentile(valid, 97.5)),
    }

