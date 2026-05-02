"""Heatmap plotting utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from protein_hmm.visualization.style import apply_style


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    apply_style()
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
    cmap: str = "viridis",
    diverging: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    plt = _require_matplotlib()
    values = np.asarray(matrix, dtype=float)
    n_rows, n_cols = values.shape
    if figsize is None:
        figsize = (max(8.0, 0.7 * n_cols + 4.0), max(6.0, 0.55 * n_rows + 3.5))

    if diverging:
        center = float(np.nanmean(values))
        spread = float(np.nanmax(np.abs(values - center))) or 1.0
        vmin, vmax = center - spread, center + spread
        cmap = "RdBu_r" if cmap == "viridis" else cmap
    else:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))

    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    axis.set_title(title)
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    axis.set_yticks(range(n_rows))
    axis.set_yticklabels(row_labels)
    axis.set_xticks(range(n_cols))
    axis.set_xticklabels(col_labels, rotation=45, ha="right")

    # Subtle white separators between cells
    axis.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.025)
    colorbar.ax.tick_params(labelsize=13)
    if colorbar_label is not None:
        colorbar.set_label(colorbar_label, fontsize=15, fontweight="bold")

    if annotate:
        cmap_obj = image.get_cmap()
        norm = image.norm
        for row_index in range(n_rows):
            for col_index in range(n_cols):
                value = values[row_index, col_index]
                rgba = cmap_obj(norm(value))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.55 else "#222222"
                axis.text(
                    col_index,
                    row_index,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=14,
                    fontweight="bold",
                )

    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
