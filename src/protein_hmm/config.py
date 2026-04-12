"""Configuration helpers for scripts and experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ProjectConfig:
    data: dict[str, Any]
    models: dict[str, Any]
    experiments: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return getattr(self, name)


def _read_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Could not parse config at {config_path}. Install PyYAML or "
                "keep the config JSON-compatible."
            ) from exc
        loaded = yaml.safe_load(text)
        return loaded or {}


def load_project_config(root: str | Path | None = None) -> ProjectConfig:
    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    config_dir = project_root / "configs"
    return ProjectConfig(
        data=_read_mapping(config_dir / "data.yaml"),
        models=_read_mapping(config_dir / "models.yaml"),
        experiments=_read_mapping(config_dir / "experiments.yaml"),
    )
