"""Summary figure helpers for model selection and family comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from protein_hmm.visualization.style import (
    DIVERGING_NEG,
    DIVERGING_POS,
    STATE_PALETTE,
    apply_style,
)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting.") from exc
    apply_style()
    return plt


def _save(figure, path: str | Path | None) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
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
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.plot(
        state_counts,
        scores,
        marker="o",
        markersize=12,
        linewidth=2.8,
        color=STATE_PALETTE[0],
        markerfacecolor="white",
        markeredgewidth=2.5,
    )
    axis.set_title(title)
    axis.set_xlabel("Number of latent states")
    axis.set_ylabel(ylabel)
    axis.set_xticks(state_counts)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    if note is not None:
        axis.text(
            0.02,
            0.04,
            note,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            color="#666666",
            style="italic",
        )
    figure.tight_layout()
    _save(figure, path)


def plot_em_convergence(
    histories: dict[str, Iterable[float]],
    title: str = "EM convergence",
    path: str | Path | None = None,
) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    items = list(histories.items())
    for index, (label, history) in enumerate(items):
        history = list(history)
        if not history:
            continue
        color = STATE_PALETTE[index % len(STATE_PALETTE)]
        axis.plot(
            range(1, len(history) + 1),
            history,
            marker="o",
            markersize=5,
            linewidth=2.4,
            label=label,
            color=color,
        )
    axis.set_xlabel("EM iteration")
    axis.set_ylabel("Total training log-likelihood")
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    axis.legend(loc="lower right", title="Model", title_fontsize=14)
    figure.tight_layout()
    _save(figure, path)


def plot_state_property_bars(
    state_labels: list[str],
    series: dict[str, Iterable[float]],
    title: str,
    ylabel: str,
    path: str | Path | None = None,
    diverging: bool = False,
    annotate: bool = True,
) -> None:
    plt = _require_matplotlib()
    n_states = len(state_labels)
    series_keys = list(series.keys())
    n_groups = len(series_keys)
    figure, axis = plt.subplots(figsize=(max(7.0, 1.2 * n_states + 3.0), 5.5))
    width = 0.8 / max(n_groups, 1)
    indices = np.arange(n_states)

    all_values: list[float] = []
    for values in series.values():
        all_values.extend(values)
    span = max(abs(min(all_values, default=0.0)), abs(max(all_values, default=0.0)), 1.0)

    for group_index, key in enumerate(series_keys):
        values = list(series[key])
        offset = (group_index - (n_groups - 1) / 2) * width
        if diverging and n_groups == 1:
            colors = [DIVERGING_POS if v >= 0 else DIVERGING_NEG for v in values]
        elif n_groups == 1:
            colors = [STATE_PALETTE[i % len(STATE_PALETTE)] for i in range(n_states)]
        else:
            colors = STATE_PALETTE[group_index % len(STATE_PALETTE)]
        bars = axis.bar(
            indices + offset,
            values,
            width=width * 0.92,
            label=key,
            color=colors,
            edgecolor="white",
            linewidth=1.2,
        )
        if annotate:
            for bar, value in zip(bars, values):
                bar_height = bar.get_height()
                va = "bottom" if bar_height >= 0 else "top"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_height + (0.015 if bar_height >= 0 else -0.015) * span,
                    f"{value:.2f}",
                    ha="center",
                    va=va,
                    fontsize=13,
                    fontweight="bold",
                    color="#222222",
                )

    axis.set_xticks(indices)
    axis.set_xticklabels(state_labels)
    axis.axhline(0.0, color="#888888", linewidth=0.8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", linestyle="--", alpha=0.5)
    if n_groups > 1:
        axis.legend(loc="best", title_fontsize=14)
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
