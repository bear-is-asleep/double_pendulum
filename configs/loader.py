"""Load sampler + model YAML. Edit YAML only; do not hardcode stage bounds in train code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parent
_MODELS_DIR = _CONFIG_ROOT / "models"


def _read_yaml(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if data is None:
    return {}
  if not isinstance(data, dict):
    raise TypeError(f"Expected mapping at root of {path}, got {type(data).__name__}")
  return data


def load_sampler_config(path: Path | None = None) -> dict[str, Any]:
  """Shared stage bounds, safety, IC boxes, pool sizes."""
  return _read_yaml(path or (_CONFIG_ROOT / "sampler.yaml"))


def load_model_config(
  model_name: str,
  models_dir: Path | None = None,
) -> dict[str, Any]:
  """
  Merge configs/models/base.yaml with configs/models/<model_name>.yaml.
  model_name: baseline | curriculum | active | progressive (no .yaml).
  """
  root = models_dir or _MODELS_DIR
  base = _read_yaml(root / "base.yaml")
  overlay = _read_yaml(root / f"{model_name}.yaml")
  merged = {**base, **overlay}
  if "name" not in merged:
    raise KeyError(f"Model config {model_name} missing 'name' after merge")
  return merged


def list_model_configs(models_dir: Path | None = None) -> list[str]:
  """Strategy overlay stems (excludes base)."""
  root = models_dir or _MODELS_DIR
  return sorted(
    p.stem for p in root.glob("*.yaml") if p.stem != "base"
  )
