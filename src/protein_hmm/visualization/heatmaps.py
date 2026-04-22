"""Heatmap plotting utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


def plot_matrix(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: str | Path | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(np.asarray(matrix, dtype=float), aspect="auto", cmap="viridis")
    axis.set_title(title)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xticks(range(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=45, ha="right")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)
