"""Pathology class names — single source: config/pathologies.json."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config" / "pathologies.json"


@lru_cache(maxsize=1)
def load_pathologies() -> tuple[str, ...]:
    path = _config_path()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data or not all(isinstance(name, str) for name in data):
        raise ValueError(f"{path} must be a non-empty JSON array of strings")
    return tuple(data)


PATHOLOGIES: tuple[str, ...] = load_pathologies()
