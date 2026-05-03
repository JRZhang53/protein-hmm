"""Path utilities for project-aware scripts."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return project_root() / "data"


def results_dir() -> Path:
    return project_root() / "results"


def reports_dir() -> Path:
    return project_root() / "reports"


def resolve_project_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or project_root()) / candidate


_BY_K_BASES = {
    "models": "results/models/by_K",
    "metrics": "results/metrics/by_K",
    "figures": "reports/figures/by_K",
}


def by_K_dir(K: int, kind: str, root: Path | None = None) -> Path:
    """Return the per-K output directory for a given output ``kind``.

    ``kind`` is one of "models", "metrics", or "figures". Creates the directory
    if it doesn't exist so callers can write to it without extra ceremony.
    """
    if kind not in _BY_K_BASES:
        raise ValueError(f"unknown kind {kind!r}; expected one of {list(_BY_K_BASES)}")
    target = resolve_project_path(f"{_BY_K_BASES[kind]}/K{K}", root)
    target.mkdir(parents=True, exist_ok=True)
    return target
