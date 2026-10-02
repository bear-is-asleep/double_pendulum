"""YAML mapping read helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_mapping(path: Path | str) -> dict[str, Any]:
  """Load YAML file; root must be a mapping (empty file -> {})."""
  with Path(path).open(encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if data is None:
    return {}
  if not isinstance(data, dict):
    raise TypeError(f"Expected mapping at root of {path}, got {type(data).__name__}")
  return data
