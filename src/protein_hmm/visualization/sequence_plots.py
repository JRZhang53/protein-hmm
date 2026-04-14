"""Sequence-level plotting helpers."""

from __future__ import annotations

from pathlib import Path


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
