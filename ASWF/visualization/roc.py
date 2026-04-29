from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from ..utils.io import ensure_dir


def plot_roc_curve(y_true, positive_prob, output_path: Path | str) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fpr, tpr, _ = roc_curve(y_true, positive_prob)
    score = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC={score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pr_curve(y_true, positive_prob, output_path: Path | str) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    precision, recall, _ = precision_recall_curve(y_true, positive_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

