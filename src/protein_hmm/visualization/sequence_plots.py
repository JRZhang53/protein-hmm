"""Sequence-level plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


def plot_state_path(
    states: list[int],
    labels: str | None = None,
    title: str = "Decoded State Path",
    path: str | Path | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(10, 3))
    positions = list(range(len(states)))
    axis.step(positions, states, where="mid", label="latent state", linewidth=2)
    if labels is not None:
        label_map: dict[str, int] = {}
        label_track = []
        for label in labels:
            if label not in label_map:
                label_map[label] = len(label_map)
            label_track.append(label_map[label])
        axis.plot(positions, label_track, label="labels", alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel("Residue position")
    axis.set_ylabel("State")
    axis.legend()
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)


def plot_state_path_with_labels(
    states: list[int],
    labels: str,
    label_order: tuple[str, ...] = ("H", "E", "C"),
    num_states: int | None = None,
    title: str = "Decoded state path vs DSSP labels",
    path: str | Path | None = None,
) -> None:
    """Two-track plot: latent state on top, DSSP labels below as a colour band."""
    plt = _require_matplotlib()
    if len(states) != len(labels):
        raise ValueError("states and labels must have equal length.")
    num_states = num_states if num_states is not None else (max(states) + 1 if states else 1)
    label_index = {label: index for index, label in enumerate(label_order)}
    label_track = np.asarray([label_index.get(label, len(label_order)) for label in labels])

    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(11, 3.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    axis_top.step(range(len(states)), states, where="mid", linewidth=1.5, color="tab:blue")
    axis_top.set_ylabel("Latent state")
    axis_top.set_yticks(range(num_states))
    axis_top.set_title(title)
    axis_top.set_ylim(-0.5, num_states - 0.5)

    n_labels = len(label_order)
    image = axis_bottom.imshow(
        label_track[None, :],
        aspect="auto",
        cmap="viridis",
        vmin=-0.5,
        vmax=n_labels - 0.5,
        extent=(0, len(labels), 0, 1),
    )
    axis_bottom.set_yticks([])
    axis_bottom.set_ylabel("DSSP")
    axis_bottom.set_xlabel("Residue position")
    colorbar = figure.colorbar(image, ax=axis_bottom, ticks=list(range(n_labels)), orientation="vertical", fraction=0.025, pad=0.02)
    colorbar.ax.set_yticklabels(list(label_order))
    colorbar.ax.tick_params(labelsize=9)
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)
