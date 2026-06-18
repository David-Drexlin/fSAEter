"""Config and path helpers for fSAEter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    cursor = (start or Path(__file__)).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "fsaeter").is_dir():
            return candidate
    raise RuntimeError(f"Could not find fSAEter project root from {cursor}")


def runtime_base_root(start: Path | None = None) -> Path:
    override = os.environ.get("FSAETER_BASE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    try:
        return find_project_root(start)
    except RuntimeError:
        return Path.cwd().resolve()


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base or find_project_root()).resolve() / path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping config at {path}, got {type(data).__name__}")
    return data


def save_yaml_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
