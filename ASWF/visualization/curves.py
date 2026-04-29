from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.io import ensure_dir


def plot_stage_history(history: List[dict], output_path: Path | str, stage_name: str) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metric_cols = [column for column in frame.columns if column.startswith("train_")]
    for column in metric_cols:
        axes[0].plot(frame["epoch"], frame[column], label=column)
    axes[0].set_title(f"{stage_name} train")
    axes[0].legend(fontsize=8)

    val_cols = [column for column in frame.columns if column.startswith("val_")]
    for column in val_cols:
        axes[1].plot(frame["epoch"], frame[column], label=column)
    axes[1].set_title(f"{stage_name} val")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

