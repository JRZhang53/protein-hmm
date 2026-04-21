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
    colorbar_label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate: bool = False,
    value_format: str = ".2f",
) -> None:
    plt = _require_matplotlib()
    values = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_xticks(range(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=45, ha="right")
    colorbar = figure.colorbar(image, ax=axis)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label)
    if annotate:
        threshold = float((np.nanmax(values) + np.nanmin(values)) / 2.0)
        for row_index in range(values.shape[0]):
            for col_index in range(values.shape[1]):
                color = "black" if values[row_index, col_index] > threshold else "white"
                axis.text(
                    col_index,
                    row_index,
                    format(values[row_index, col_index], value_format),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)
