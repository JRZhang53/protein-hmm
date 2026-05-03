"""Compact flowchart explaining how cross-family transition matrices are compared.

Pipeline visualised:
  arbitrary state labels → Hungarian-align by JS on emissions → aligned matrices → Frobenius

Output: reports/figures/alignment_flowchart.png  (≈ 12" × 3" poster strip).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from _bootstrap import bootstrap

ROOT = bootstrap()

from protein_hmm.config import load_project_config
from protein_hmm.utils.paths import resolve_project_path
from protein_hmm.visualization.style import apply_style


PALETTE_A = ["#1f4e79", "#c1432a", "#3f7d20", "#d4a017"]   # Family A: states in some order
PALETTE_B = ["#3f7d20", "#1f4e79", "#d4a017", "#c1432a"]   # Family B: same states, different label permutation
PALETTE_B_ALIGNED = PALETTE_A                               # After Hungarian alignment


def _draw_state_strip(ax, x0, y0, palette, width=0.9, height=0.6, gap=0.1):
    n = len(palette)
    box_w = (width - gap * (n - 1)) / n
    for i, color in enumerate(palette):
        x = x0 + i * (box_w + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y0), box_w, height,
            boxstyle="round,pad=0.005,rounding_size=0.04",
            linewidth=0,
            facecolor=color,
        ))


def _draw_matrix(ax, x0, y0, size, palette, title=None, cell_pattern=None):
    """Draw a small K×K matrix grid with diagonal-strong colouring."""
    n = len(palette)
    cell = size / n
    if cell_pattern is None:
        cell_pattern = np.zeros((n, n))
        for i in range(n):
            cell_pattern[i, i] = 0.85
            for j in range(n):
                if i != j:
                    cell_pattern[i, j] = 0.18 + 0.05 * ((i + j) % 3)
    for i in range(n):
        for j in range(n):
            x = x0 + j * cell
            y = y0 + (n - 1 - i) * cell
            color = palette[i]
            alpha = cell_pattern[i, j]
            ax.add_patch(mpatches.Rectangle((x, y), cell * 0.92, cell * 0.92,
                                            facecolor=color, alpha=alpha,
                                            linewidth=0))
    # Outer frame
    ax.add_patch(mpatches.Rectangle((x0, y0), size, size, fill=False,
                                    edgecolor="#bbbbbb", linewidth=1.0))
    if title:
        ax.text(x0 + size / 2, y0 + size + 0.06, title,
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")


def _draw_arrow(ax, x_start, x_end, y, label_top=None, label_bot=None):
    ax.annotate(
        "",
        xy=(x_end, y), xytext=(x_start, y),
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=2.2),
    )
    mid = (x_start + x_end) / 2
    if label_top:
        ax.text(mid, y + 0.12, label_top, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#1f4e79")
    if label_bot:
        ax.text(mid, y - 0.16, label_bot, ha="center", va="top",
                fontsize=10, color="#666", style="italic")


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    rng = np.random.default_rng(7)

    matrix_size = 1.4
    arrow_y = 1.2

    # Stage 1: two unaligned matrices (Family A and Family B with permuted state labels)
    pat_a = np.diag(rng.uniform(0.65, 0.85, 4)) + rng.uniform(0.05, 0.25, (4, 4)) * (1 - np.eye(4))
    perm = np.array([1, 3, 0, 2])  # B's labels are permuted
    pat_b = pat_a[np.ix_(perm, perm)]

    _draw_matrix(ax, 0.4, 1.30, matrix_size, PALETTE_A, title="Family A", cell_pattern=pat_a)
    _draw_state_strip(ax, 0.4, 1.04, PALETTE_A, width=matrix_size, height=0.18, gap=0.05)

    _draw_matrix(ax, 0.4, -0.45 + 0.65, matrix_size * 0.0, PALETTE_A)  # spacer (no-op)

    _draw_matrix(ax, 0.4, 0.05, matrix_size, PALETTE_B, title="Family B (raw labels)", cell_pattern=pat_b)
    _draw_state_strip(ax, 0.4, -0.21, PALETTE_B, width=matrix_size, height=0.18, gap=0.05)

    ax.text(1.1, 2.95, "Two HMMs trained\nper family",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#1f4e79")
    ax.text(1.1, -0.45, "State indices are\narbitrary across models",
            ha="center", va="top", fontsize=9, color="#666", style="italic")

    # Arrow 1 → Hungarian + JS on emissions
    _draw_arrow(ax, 2.2, 4.2, arrow_y,
                label_top="Hungarian on JS(emit)",
                label_bot="match states by closest emission distribution")

    # Stage 2: Family B re-permuted to match A
    pat_b_aligned = pat_a  # after Hungarian, the off-diagonal noise should match A's
    pat_b_aligned = pat_b[np.ix_(np.argsort(perm), np.argsort(perm))]
    _draw_matrix(ax, 4.5, 0.7, matrix_size, PALETTE_B_ALIGNED,
                 title="Family B (aligned)", cell_pattern=pat_b_aligned)
    _draw_state_strip(ax, 4.5, 0.44, PALETTE_B_ALIGNED, width=matrix_size, height=0.18, gap=0.05)

    ax.text(5.2, 2.45, "Same residue\nenvironments,\nrelabeled",
            ha="center", va="center", fontsize=10, color="#444")

    # Arrow 2 → Frobenius norm
    _draw_arrow(ax, 6.3, 8.3, arrow_y,
                label_top="‖A − B_aligned‖_F",
                label_bot="Frobenius norm on transition matrices")

    # Stage 3: scalar distance (minimal numeric panel)
    ax.add_patch(mpatches.FancyBboxPatch(
        (8.5, 0.6), 4.1, 1.5,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=2, edgecolor="#1f4e79",
        facecolor="white",
    ))
    ax.text(10.55, 1.6, "d(Family A, Family B)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#1f4e79")
    ax.text(10.55, 1.0, "single scalar  ·  comparable across families",
            ha="center", va="center", fontsize=10, color="#444", style="italic")

    ax.set_title("Cross-family transition-matrix comparison",
                 fontsize=15, fontweight="bold", pad=8, loc="left")

    fig.tight_layout()

    config = load_project_config(ROOT)
    fig_dir = resolve_project_path(config.experiments["outputs"]["figure_dir"], ROOT)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "alignment_flowchart.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
