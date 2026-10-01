"""Run folder layout for baseline trials (Step 5; extended in Step 8/9)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim


def baseline_run_id(cfg: dict[str, Any], *, tag: str | None = None) -> str:
  """Filesystem-safe id encoding architecture + train knobs that differ."""
  parts = [
    "baseline",
    f"w{int(cfg['hidden_width'])}",
    f"d{int(cfg['hidden_depth'])}",
    f"k{int(cfg['subsample_stride_k'])}",
    f"seed{int(cfg['seed'])}",
  ]
  if tag:
    parts.append(tag)
  return "_".join(parts)


def init_run_dir(runs_root: Path | str, run_id: str, cfg: dict[str, Any]) -> Path:
  root = Path(runs_root) / run_id
  if root.exists():
    raise FileExistsError(f"run dir already exists: {root}")
  (root / "checkpoints").mkdir(parents=True)
  with (root / "config.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(dict(cfg), f, sort_keys=False)
  return root


def append_metrics_jsonl(run_dir: Path, record: dict[str, Any]) -> None:
  path = run_dir / "metrics.jsonl"
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(
  path: Path,
  *,
  model: nn.Module,
  optimizer: optim.Optimizer | None,
  epoch: int,
  global_step: int,
  val_metric: float,
  cfg: dict[str, Any],
) -> None:
  """Portable dict for rebuild-from-YAML + ``load_state_dict`` (see project.md)."""
  payload: dict[str, Any] = {
    "model_state_dict": model.state_dict(),
    "epoch": int(epoch),
    "global_step": int(global_step),
    "val_metric": float(val_metric),
    "config": dict(cfg),
  }
  if optimizer is not None:
    payload["optimizer_state_dict"] = optimizer.state_dict()
  torch.save(payload, path)


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
  with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")


def utc_now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
