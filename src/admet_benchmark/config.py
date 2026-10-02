"""Configuration loading with portable, environment-aware paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and retain its location for resolving relative data paths."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    config = _expand_environment(config)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_path(value: str | Path, config: dict[str, Any]) -> Path:
    """Resolve a user path relative to the YAML file that contains it."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(config["_config_dir"]) / candidate
