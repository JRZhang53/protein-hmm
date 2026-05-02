"""Sequence-level plotting helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from protein_hmm.visualization.style import STATE_PALETTE, apply_style


# Discrete DSSP class colours (helix red, strand teal, coil grey).
DSSP_COLORS = {
    "H": "#c1432a",
    "E": "#0d8c8c",
    "C": "#9ea3a8",
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    apply_style()
    return plt


def plot_state_path(
    states: list[int],
    labels: str | None = None,
    title: str = "Decoded State Path",
    path: str | Path | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(12, 4))
    positions = list(range(len(states)))
    axis.step(positions, states, where="mid", label="latent state", linewidth=2.5, color=STATE_PALETTE[0])
    if labels is not None:
        label_map: dict[str, int] = {}
        label_track = []
        for label in labels:
            if label not in label_map:
                label_map[label] = len(label_map)
            label_track.append(label_map[label])
        axis.plot(positions, label_track, label="labels", alpha=0.5, linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Residue position")
    axis.set_ylabel("State")
    axis.legend()
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
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
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    if len(states) != len(labels):
        raise ValueError("states and labels must have equal length.")
    num_states = num_states if num_states is not None else (max(states) + 1 if states else 1)
    label_index = {label: index for index, label in enumerate(label_order)}
    label_track = np.asarray([label_index.get(label, len(label_order)) for label in labels])

    figure, (axis_top, axis_bottom) = plt.subplots(
        2,
        1,
        figsize=(13, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )

    axis_top.step(
        range(len(states)),
        states,
        where="mid",
        linewidth=2.4,
        color=STATE_PALETTE[0],
    )
    axis_top.fill_between(
        range(len(states)),
        states,
        step="mid",
        alpha=0.18,
        color=STATE_PALETTE[0],
    )
    axis_top.set_ylabel("Latent state")
    axis_top.set_yticks(range(num_states))
    axis_top.set_yticklabels([f"S{i}" for i in range(num_states)])
    axis_top.set_title(title)
    axis_top.set_ylim(-0.5, num_states - 0.5)
    axis_top.grid(axis="y", linestyle="--", alpha=0.4)

    n_labels = len(label_order)
    cmap = ListedColormap([DSSP_COLORS.get(label, "#cccccc") for label in label_order])
    axis_bottom.imshow(
        label_track[None, :],
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=n_labels - 0.5,
        extent=(0, len(labels), 0, 1),
        interpolation="nearest",
    )
    axis_bottom.set_yticks([])
    axis_bottom.set_ylabel("DSSP")
    axis_bottom.set_xlabel("Residue position")
    for spine in axis_bottom.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=DSSP_COLORS.get(label, "#cccccc"), edgecolor="white", label=label)
        for label in label_order
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.02),
        ncol=len(label_order),
        frameon=False,
        fontsize=14,
        title=None,
    )

    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path)
        plt.close(figure)
