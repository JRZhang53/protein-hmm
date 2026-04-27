"""Summary figure helpers for model selection and family comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    return plt


def _save(figure, path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt = _require_matplotlib()
    plt.close(figure)


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
    _save(figure, path)


def plot_em_convergence(
    histories: dict[str, Iterable[float]],
    title: str = "EM convergence",
    path: str | Path | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, history in histories.items():
        history = list(history)
        if not history:
            continue
        axis.plot(range(1, len(history) + 1), history, marker=".", label=label)
    axis.set_xlabel("EM iteration")
    axis.set_ylabel("Total training log-likelihood")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    _save(figure, path)


def plot_state_property_bars(
    state_labels: list[str],
    series: dict[str, Iterable[float]],
    title: str,
    ylabel: str,
    path: str | Path | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(max(6, len(state_labels) * 0.9), 4))
    series_keys = list(series.keys())
    n_groups = len(series_keys)
    width = 0.8 / max(n_groups, 1)
    indices = np.arange(len(state_labels))
    for index, key in enumerate(series_keys):
        values = list(series[key])
        offset = (index - (n_groups - 1) / 2) * width
        axis.bar(indices + offset, values, width=width, label=key)
    axis.set_xticks(indices)
    axis.set_xticklabels(state_labels)
    axis.axhline(0.0, color="grey", linewidth=0.5)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if n_groups > 1:
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    _save(figure, path)


def plot_grouped_bars(
    group_labels: list[str],
    bar_groups: dict[str, Iterable[float]],
    title: str,
    ylabel: str,
    path: str | Path | None = None,
) -> None:
    plot_state_property_bars(
        state_labels=group_labels,
        series=bar_groups,
        title=title,
        ylabel=ylabel,
        path=path,
    )
