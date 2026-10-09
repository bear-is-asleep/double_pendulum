"""Run folder layout for baseline trials (Step 5; extended in Step 8/9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn, optim

from srcs.utils.json_io import append_jsonl_line, write_json
from srcs.utils.time import utc_now_iso

__all__ = [
  "append_metrics_jsonl",
  "baseline_run_id",
  "curriculum_run_id",
  "init_run_dir",
  "save_checkpoint",
  "strategy_run_id",
  "utc_now_iso",
  "write_summary",
]


def _hidden_width_run_tag(cfg: dict[str, Any]) -> str:
  raw = cfg["hidden_width"]
  if isinstance(raw, (list, tuple)):
    return "w" + "x".join(str(int(x)) for x in raw)
  return f"w{int(raw)}"


def strategy_run_id(
  cfg: dict[str, Any],
  strategy: str,
  *,
  tag: str | None = None,
) -> str:
  """Filesystem-safe id: strategy + locked architecture / train knobs."""
  parts = [
    strategy,
    _hidden_width_run_tag(cfg),
    f"d{int(cfg['hidden_depth'])}",
    f"k{int(cfg['subsample_stride_k'])}",
    f"seed{int(cfg['seed'])}",
  ]
  if tag:
    parts.append(tag)
  return "_".join(parts)


def baseline_run_id(cfg: dict[str, Any], *, tag: str | None = None) -> str:
  return strategy_run_id(cfg, "baseline", tag=tag)


def curriculum_run_id(cfg: dict[str, Any], *, tag: str | None = None) -> str:
  return strategy_run_id(cfg, "curriculum", tag=tag)


def progressive_run_id(cfg: dict[str, Any], *, tag: str | None = None) -> str:
  return strategy_run_id(cfg, "progressive", tag=tag)


def init_run_dir(runs_root: Path | str, run_id: str, cfg: dict[str, Any]) -> Path:
  root = Path(runs_root) / run_id
  if (root / "config.yaml").is_file():
    raise FileExistsError(f"run dir already exists: {root}")
  root.mkdir(parents=True, exist_ok=True)
  (root / "checkpoints").mkdir(parents=True)
  with (root / "config.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(dict(cfg), f, sort_keys=False)
  return root


def append_metrics_jsonl(run_dir: Path, record: dict[str, Any]) -> None:
  append_jsonl_line(run_dir / "metrics.jsonl", record, sort_keys=True)


def save_checkpoint(
  path: Path,
  *,
  model: nn.Module,
  optimizer: optim.Optimizer | None,
  epoch: int,
  global_step: int,
  val_metric: float,
  cfg: dict[str, Any],
  resume_state: dict[str, Any] | None = None,
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
  if resume_state is not None:
    payload["resume_state"] = dict(resume_state)
  torch.save(payload, path)


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
  write_json(run_dir / "summary.json", summary, sort_keys=True)
