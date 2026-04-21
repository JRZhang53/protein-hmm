"""Summary figure helpers for model selection and family comparisons."""

from __future__ import annotations

from pathlib import Path


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


def plot_likelihood_curve(
    state_counts: list[int],
    scores: list[float],
    title: str,
    path: str | Path | None = None,
    ylabel: str = "Score",
    note: str | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(state_counts, scores, marker="o")
    axis.set_title(title)
    axis.set_xlabel("Number of latent states")
    axis.set_ylabel(ylabel)
    if note is not None:
        axis.text(
            0.02,
            0.02,
            note,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
        plt.close(figure)
