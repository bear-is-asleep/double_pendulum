"""Filesystem paths: create dirs, resolve checkpoints/runs, read paths from JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from srcs.utils.yaml_io import read_mapping


def ensure_dir(path: Path | str) -> Path:
  """Create directory (and parents) if missing; return normalized path."""
  out = Path(path)
  out.mkdir(parents=True, exist_ok=True)
  return out


def ensure_parent_dir(path: Path | str) -> Path:
  """Ensure parent of a file path exists; return normalized file path."""
  out = Path(path)
  if out.parent != out:
    out.parent.mkdir(parents=True, exist_ok=True)
  return out


def resolve_run_dir(checkpoint_path: Path | str) -> Path:
  """Run root from run folder, ``checkpoints/*.pt``, or a ``.pt`` file."""
  path = Path(checkpoint_path).resolve()
  if path.is_file():
    if path.parent.name == "checkpoints":
      return path.parent.parent
    return path.parent
  if (path / "checkpoints" / "best.pt").is_file():
    return path
  if (path / "config.yaml").is_file():
    return path
  raise FileNotFoundError(f"cannot resolve run dir from {checkpoint_path!r}")


def resolve_checkpoint_file(checkpoint_path: Path | str) -> Path:
  """``.pt`` file from explicit path, run dir, or ``checkpoints/best.pt`` / ``last.pt``."""
  path = Path(checkpoint_path).resolve()
  if path.is_file() and path.suffix == ".pt":
    return path
  run = resolve_run_dir(path)
  best = run / "checkpoints" / "best.pt"
  if best.is_file():
    return best
  last = run / "checkpoints" / "last.pt"
  if last.is_file():
    return last
  raise FileNotFoundError(f"no .pt checkpoint under {run}")


def load_run_config(run_dir: Path, ckpt_cfg: Any) -> dict[str, Any]:
  """Hyperparameter dict from checkpoint payload or ``config.yaml`` under ``run_dir``."""
  if isinstance(ckpt_cfg, dict) and ckpt_cfg.get("input_dim"):
    return dict(ckpt_cfg)
  yaml_path = run_dir / "config.yaml"
  if yaml_path.is_file():
    return read_mapping(yaml_path)
  raise ValueError(f"no rebuildable config in checkpoint or {yaml_path}")
