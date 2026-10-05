"""YAML mapping read helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def yaml_safe_value(obj: Any) -> Any:
  """Recursively coerce values for ``yaml.safe_dump`` (e.g. ``Path`` -> str)."""
  if isinstance(obj, Path):
    return str(obj)
  if isinstance(obj, dict):
    return {k: yaml_safe_value(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [yaml_safe_value(v) for v in obj]
  return obj


def print_mapping_yaml(title: str, mapping: dict[str, Any]) -> None:
  """Stdout banner + YAML dump of resolved run parameters."""
  print(f"--- {title} ---")
  body = yaml.safe_dump(
    yaml_safe_value(mapping),
    sort_keys=False,
    default_flow_style=False,
  ).rstrip()
  print(body)
  print("---")


def read_mapping(path: Path | str) -> dict[str, Any]:
  """Load YAML file; root must be a mapping (empty file -> {})."""
  with Path(path).open(encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if data is None:
    return {}
  if not isinstance(data, dict):
    raise TypeError(f"Expected mapping at root of {path}, got {type(data).__name__}")
  return data
