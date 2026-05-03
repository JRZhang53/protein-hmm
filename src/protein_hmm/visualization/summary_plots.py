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


def plot_bic_and_test_ll(
    state_counts: list[int],
    bic_scores: list[float],
    test_ll_per_residue: list[float],
    selected_K: int | None = None,
    title: str = "Model selection: BIC and held-out log-likelihood vs K",
    path: str | Path | None = None,
) -> None:
    """Dual-axis plot: BIC (left, lower=better) and test LL/residue (right, higher=better)."""
    plt = _require_matplotlib()
    figure, ax1 = plt.subplots(figsize=(11, 7.5))

    bic_color = STATE_PALETTE[0]   # navy
    ll_color = STATE_PALETTE[1]    # red

    # Left axis: BIC
    line1 = ax1.plot(
        state_counts, bic_scores,
        marker="o", markersize=18, linewidth=3.5,
        color=bic_color, markerfacecolor="white", markeredgewidth=3.5,
        label="BIC (lower = better)",
        zorder=3,
    )
    ax1.set_xlabel("Number of latent states K", fontsize=22, fontweight="bold")
    ax1.set_ylabel("BIC ↓", fontsize=22, fontweight="bold", color=bic_color)
    ax1.tick_params(axis="y", labelcolor=bic_color, labelsize=18)
    ax1.tick_params(axis="x", labelsize=18)
    ax1.set_xticks(state_counts)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, color=bic_color)

    # Right axis: test LL/residue
    ax2 = ax1.twinx()
    line2 = ax2.plot(
        state_counts, test_ll_per_residue,
        marker="s", markersize=18, linewidth=3.5,
        color=ll_color, markerfacecolor="white", markeredgewidth=3.5,
        label="Test LL / residue (higher = better)",
        zorder=3,
    )
    ax2.set_ylabel("Test LL / residue ↑", fontsize=22, fontweight="bold", color=ll_color)
    ax2.tick_params(axis="y", labelcolor=ll_color, labelsize=18)

    # Highlight selected K
    if selected_K is not None and selected_K in state_counts:
        idx = state_counts.index(selected_K)
        ax1.axvline(selected_K, color="#777777", linewidth=1.5, linestyle=":", zorder=1)
        ax1.scatter([selected_K], [bic_scores[idx]],
                    s=900, facecolors="none", edgecolors=bic_color, linewidths=5, zorder=4)
        ax2.scatter([selected_K], [test_ll_per_residue[idx]],
                    s=900, facecolors="none", edgecolors=ll_color, linewidths=5, zorder=4)
        # Place the "selected K=N" label inside the plot area, just above
        # the BIC curve at the selected K, so it never collides with the
        # x-axis label.
        ax1.annotate(
            f"selected K={selected_K}",
            xy=(selected_K, bic_scores[idx]),
            xytext=(14, 14),
            textcoords="offset points",
            ha="left", va="bottom",
            fontsize=18, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#777777", lw=1.2),
        )

    ax1.set_title(title, fontsize=22, fontweight="bold", pad=18)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=16,
        framealpha=0.95,
        frameon=True,
        borderpad=0.7,
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
