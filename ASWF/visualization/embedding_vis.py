from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..utils.io import ensure_dir


def _plot_projection(points: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], label=f"class_{int(label)}", s=18)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, output_dir: Path | str, prefix: str) -> None:
    if embeddings.size == 0 or embeddings.shape[0] < 2:
        return
    output_dir = ensure_dir(output_dir)

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(embeddings)
    _plot_projection(pca_points, labels, Path(output_dir) / f"{prefix}_pca.png", f"{prefix} PCA")

    if embeddings.shape[0] >= 5:
        perplexity = max(2, min(30, embeddings.shape[0] // 3))
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity)
        tsne_points = tsne.fit_transform(embeddings)
        _plot_projection(tsne_points, labels, Path(output_dir) / f"{prefix}_tsne.png", f"{prefix} t-SNE")

