from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from ..utils.io import ensure_dir


def plot_confusion_matrix(y_true, pred, output_path: Path | str) -> None:
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["normal", "abnormal"])
    ax.set_yticks([0, 1], labels=["normal", "abnormal"])
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

