"""Shared figure styling for poster-quality output."""

from __future__ import annotations


# Categorical palette used for per-state bars, EM convergence lines, etc.
STATE_PALETTE: tuple[str, ...] = (
    "#1f4e79",  # deep blue
    "#c1432a",  # warm red
    "#3f7d20",  # forest green
    "#d4a017",  # mustard
    "#5b3a8c",  # purple
    "#0d8c8c",  # teal
    "#8c4a2b",  # earthy brown
    "#2e2e2e",  # near-black for K=8
)

# Diverging colour for signed quantities (e.g. hydrophobicity).
DIVERGING_NEG = "#1f4e79"
DIVERGING_POS = "#c1432a"


def apply_style() -> None:
    """Apply project-wide matplotlib rcParams for poster-quality figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 18,
            "axes.labelweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.4,
            "axes.edgecolor": "#333333",
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.titlepad": 16,
            "axes.labelpad": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
