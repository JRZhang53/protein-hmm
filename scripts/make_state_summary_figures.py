"""Generate poster-ready summary figures from the saved K=4 unsupervised HMM.

Outputs (under reports/figures/):
  - state_kd_vs_rsa_scatter.png   — 2D scatter of states in (KD, RSA) space
  - state_transition_graph.png    — node-and-arrow transition diagram
  - state_dwell_times.png         — geometric expected dwell per state
  - state_top_residues.png        — top-residue mini-bars per state
  - state_fingerprint_cards.png   — composite "card" view for each state
"""

from __future__ import annotations

from pathlib import Path

import json
import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from protein_hmm.constants import (
    AMINO_ACIDS,
    KYTE_DOOLITTLE_HYDROPHOBICITY,
)
from protein_hmm.config import load_project_config
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.utils.io import read_json
from protein_hmm.utils.paths import by_K_dir, resolve_project_path
from protein_hmm.visualization.style import STATE_PALETTE, apply_style


DSSP_LABELS = ("H", "E", "C")
DSSP_COLORS = {"H": "#1f4e79", "E": "#c1432a", "C": "#7f7f7f"}
STATE_INTERPRETATION = {
    0: "turn-rich coil",
    1: "amphipathic helix",
    2: "acidic surface loop",
    3: "buried hydrophobic core",
}


def _state_summary(model: DiscreteHMM, rsa_means: list[float], dssp_enrichment: np.ndarray) -> list[dict]:
    if model.params is None:
        raise RuntimeError("Model has no params")
    emit = model.params.emission_probs
    trans = model.params.transition_probs
    kd = np.asarray([KYTE_DOOLITTLE_HYDROPHOBICITY[a] for a in AMINO_ACIDS])
    stat = np.linalg.matrix_power(trans, 500)[0]

    summaries = []
    for k in range(model.num_states):
        e = emit[k]
        top_indices = np.argsort(e)[::-1][:5]
        dssp_probs = dssp_enrichment[k]
        dom_idx = int(np.argmax(dssp_probs))
        summaries.append({
            "state": k,
            "kd": float(e @ kd),
            "rsa": float(rsa_means[k]),
            "dssp_probs": dssp_probs.tolist(),
            "dominant_dssp": DSSP_LABELS[dom_idx],
            "top_residues": [(AMINO_ACIDS[i], float(e[i])) for i in top_indices],
            "stationary": float(stat[k]),
            "self_transition": float(trans[k, k]),
            "dwell": float(1.0 / max(1.0 - trans[k, k], 1e-9)),
            "interpretation": STATE_INTERPRETATION.get(k, ""),
        })
    return summaries


def plot_kd_vs_rsa(summaries, fig_dir: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    for s in summaries:
        color = DSSP_COLORS[s["dominant_dssp"]]
        size = 2200 * s["stationary"]
        ax.scatter(
            s["kd"], s["rsa"],
            s=size,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=2.5,
            zorder=3,
        )
        ax.annotate(
            f"S{s['state']}\n{s['interpretation']}",
            xy=(s["kd"], s["rsa"]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=14,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
        )
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, zorder=1)
    avg_rsa = float(np.mean([s["rsa"] for s in summaries]))
    ax.axhline(avg_rsa, color="#bbbbbb", linewidth=0.8, zorder=1, linestyle="--")
    ax.set_xlabel("Mean Kyte-Doolittle hydrophobicity\n(← polar / charged       hydrophobic →)", fontsize=14)
    ax.set_ylabel("Mean RSA per state\n(← buried        exposed →)", fontsize=14)
    ax.set_title("States separate in biochemistry × geometry space", fontsize=18, fontweight="bold", pad=14)
    ax.grid(axis="both", linestyle=":", alpha=0.5)

    handles = [
        mpatches.Patch(color=DSSP_COLORS["H"], label="Dominant: helix (H)"),
        mpatches.Patch(color=DSSP_COLORS["E"], label="Dominant: strand (E)"),
        mpatches.Patch(color=DSSP_COLORS["C"], label="Dominant: coil (C)"),
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=12, framealpha=0.95)
    legend.set_title("DSSP enrichment", prop={"size": 12, "weight": "bold"})

    ax.text(
        0.02, 0.02,
        "marker size ∝ stationary state occupancy",
        transform=ax.transAxes,
        fontsize=11, color="#666", style="italic",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "state_kd_vs_rsa_scatter.png", dpi=200)
    plt.close(fig)


def plot_transition_graph(model: DiscreteHMM, summaries, fig_dir: Path) -> None:
    apply_style()
    if model.params is None:
        raise RuntimeError
    trans = np.asarray(model.params.transition_probs)
    K = model.num_states

    angles = np.linspace(0, 2 * np.pi, K, endpoint=False) + np.pi / 2
    radius = 1.0
    coords = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    fig, ax = plt.subplots(figsize=(11, 9))

    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            p = trans[i, j]
            if p < 0.05:
                continue
            x_start, y_start = coords[i]
            x_end, y_end = coords[j]
            dx = x_end - x_start
            dy = y_end - y_start
            length = np.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            shrink = 0.18
            x_start_s = x_start + ux * shrink
            y_start_s = y_start + uy * shrink
            x_end_s = x_end - ux * shrink
            y_end_s = y_end - uy * shrink

            ax.annotate(
                "",
                xy=(x_end_s, y_end_s),
                xytext=(x_start_s, y_start_s),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#666",
                    lw=1 + p * 6,
                    alpha=0.5 + 0.5 * p,
                    connectionstyle="arc3,rad=0.18",
                ),
            )

    for s in summaries:
        i = s["state"]
        x, y = coords[i]
        color = DSSP_COLORS[s["dominant_dssp"]]
        size = 5500 * (0.5 + s["stationary"])
        ax.scatter(x, y, s=size, c=color, edgecolors="white", linewidths=3, zorder=5)
        ax.text(
            x, y, f"S{i}",
            ha="center", va="center",
            fontsize=22, fontweight="bold", color="white", zorder=6,
        )
        # Place every label above or below its node so left/right text never
        # spills off the canvas.
        a = angles[i]
        is_top = np.sin(a) > 0
        ly = y + (0.42 if is_top else -0.42)
        ax.text(
            x, ly,
            f"{s['interpretation']}\ndwell ≈ {s['dwell']:.1f} res",
            ha="center", va="bottom" if is_top else "top",
            fontsize=12, color=color, fontweight="bold",
        )
        # Self-loop indicator
        ax.text(
            x, y - 0.12,
            f"↺ {s['self_transition']:.2f}",
            ha="center", va="center",
            fontsize=11, color="white", fontweight="bold", zorder=6,
        )

    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Latent transition graph (K=4)", fontsize=18, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(fig_dir / "state_transition_graph.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dwell_times(summaries, fig_dir: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    states = [s["state"] for s in summaries]
    # Newline-wrap the multi-word interpretations so the x-tick labels don't collide.
    labels = [f"S{s['state']}\n" + "\n".join(s["interpretation"].split()) for s in summaries]
    dwells = [s["dwell"] for s in summaries]
    colors = [DSSP_COLORS[s["dominant_dssp"]] for s in summaries]
    bars = ax.bar(range(len(states)), dwells, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, d in zip(bars, dwells):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{d:.1f}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold",
        )
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Expected dwell time (residues)", fontsize=13)
    ax.set_title("Helix and strand states persist; loops do not", fontsize=16, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(dwells) * 1.18)
    fig.tight_layout()
    fig.savefig(fig_dir / "state_dwell_times.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_top_residues_per_state(summaries, fig_dir: Path) -> None:
    apply_style()
    K = len(summaries)
    fig, axes = plt.subplots(1, K, figsize=(2.6 * K, 4.6), sharey=False)
    if K == 1:
        axes = [axes]
    for ax, s in zip(axes, summaries):
        residues = [r for r, _ in s["top_residues"]]
        probs = [p for _, p in s["top_residues"]]
        color = DSSP_COLORS[s["dominant_dssp"]]
        bars = ax.barh(range(len(residues))[::-1], probs, color=color, edgecolor="white", linewidth=1.2)
        ax.set_yticks(range(len(residues))[::-1])
        ax.set_yticklabels(residues, fontsize=14, fontweight="bold")
        ax.set_xlim(0, max(probs) * 1.3)
        ax.set_xticks([])
        ax.set_title(f"S{s['state']}\n{s['interpretation']}", fontsize=12, fontweight="bold", color=color)
        for bar, p in zip(bars, probs):
            ax.text(p + max(probs) * 0.04, bar.get_y() + bar.get_height() / 2, f"{p:.2f}", va="center", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("Top emitted residues per latent state", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "state_top_residues.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_fingerprint_cards(summaries, fig_dir: Path) -> None:
    apply_style()
    K = len(summaries)
    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 5.5))
    if K == 1:
        axes = [axes]

    for ax, s in zip(axes, summaries):
        color = DSSP_COLORS[s["dominant_dssp"]]
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            transform=ax.transAxes,
            linewidth=2,
            edgecolor=color,
            facecolor="white",
            zorder=0,
        ))

        # Header
        ax.text(0.5, 0.94, f"S{s['state']}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=28, fontweight="bold", color=color)
        ax.text(0.5, 0.85, s["interpretation"],
                transform=ax.transAxes, ha="center", va="top",
                fontsize=13, color=color, style="italic")

        # Top residues row
        top_row = "  ".join(r for r, _ in s["top_residues"][:5])
        ax.text(0.5, 0.74, top_row,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=18, fontweight="bold", color="#222")
        ax.text(0.5, 0.69, "top emitted residues",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, color="#666", style="italic")

        # Numbers grid
        rows = [
            ("Hydrophobicity (KD)", f"{s['kd']:+.2f}"),
            ("Mean RSA", f"{s['rsa']:.2f}"),
            (f"P({s['dominant_dssp']} | state)", f"{s['dssp_probs'][DSSP_LABELS.index(s['dominant_dssp'])]:.2f}"),
            ("Expected dwell (res)", f"{s['dwell']:.1f}"),
            ("State occupancy", f"{s['stationary']:.2f}"),
        ]
        y_top = 0.58
        dy = 0.085
        for i, (label, value) in enumerate(rows):
            y = y_top - i * dy
            ax.text(0.08, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=11, color="#444")
            ax.text(0.92, y, value, transform=ax.transAxes, ha="right", va="center", fontsize=13, fontweight="bold", color="#111")

    fig.suptitle("Latent state fingerprints (K=4 unsupervised)", fontsize=18, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "state_fingerprint_cards.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_project_config(ROOT)
    K = int(config.models["unsupervised"]["num_states"])
    metrics_dir = by_K_dir(K, "metrics", ROOT)
    fig_dir = by_K_dir(K, "figures", ROOT)

    model = DiscreteHMM.load(by_K_dir(K, "models", ROOT) / "unsupervised_hmm.json")
    rsa_payload = read_json(metrics_dir / "state_rsa.json")
    rsa_means = rsa_payload["mean_rsa_per_state"]
    annotation = read_json(metrics_dir / "annotation_evaluation.json")
    dssp_enrichment = np.asarray(annotation["state_label_enrichment"])

    summaries = _state_summary(model, rsa_means, dssp_enrichment)

    # Persist a single JSON for downstream consumers
    payload = {"summaries": summaries, "dssp_labels": list(DSSP_LABELS)}
    (metrics_dir / "state_summaries.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    plot_kd_vs_rsa(summaries, fig_dir)
    plot_transition_graph(model, summaries, fig_dir)
    plot_dwell_times(summaries, fig_dir)
    plot_top_residues_per_state(summaries, fig_dir)
    plot_fingerprint_cards(summaries, fig_dir)
    print("Wrote state summary figures and JSON")


if __name__ == "__main__":
    main()
