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
