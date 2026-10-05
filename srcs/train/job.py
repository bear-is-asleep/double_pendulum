"""Resolve CLI flags or ``configs/train`` YAML into a single train job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srcs.loader import load_model_config, load_train_config, merge_model_train_cfg
from srcs.train.base import parse_stage_list


@dataclass(frozen=True)
class TrainJob:
  model_name: str
  cfg: dict[str, Any]
  data_root: Path
  runs_root: Path
  run_id: str
  stages: list[int]
  device: str | None
  max_epochs: int


def _model_name_from_yaml(
  train_job: dict[str, Any],
  cli_model: str | None,
) -> str:
  if cli_model is not None:
    return cli_model
  return str(train_job["train"].get("model", "baseline"))


def _train_job_from_yaml(
  name: str,
  *,
  model: str | None,
  data_root: Path | None,
  runs_root: Path | None,
  run_id: str | None,
  stages_text: str | None,
  seed: int | None,
  max_epochs: int | None,
) -> TrainJob:
  train_job = load_train_config(name)
  model_name = _model_name_from_yaml(train_job, model)
  cfg, yaml_device = merge_model_train_cfg(
    train_job,
    model_name=model_name,
    seed=seed,
  )
  stages = (
    parse_stage_list(stages_text)
    if stages_text
    else [int(s) for s in train_job["stages"]]
  )
  epochs = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
  cfg["max_epochs"] = epochs
  return TrainJob(
    model_name=model_name,
    cfg=cfg,
    data_root=data_root or Path(train_job["data_root"]),
    runs_root=runs_root or Path(train_job["runs_root"]),
    run_id=run_id or str(train_job["run_id"]),
    stages=stages,
    device=yaml_device,
    max_epochs=epochs,
  )


def _train_job_from_flags(
  *,
  model_name: str,
  data_root: Path,
  runs_root: Path,
  run_id: str,
  stages_text: str,
  seed: int | None,
  max_epochs: int | None,
) -> TrainJob:
  cfg = load_model_config(model_name)
  if seed is not None:
    cfg["seed"] = int(seed)
  epochs = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
  cfg["max_epochs"] = epochs
  return TrainJob(
    model_name=model_name,
    cfg=cfg,
    data_root=data_root,
    runs_root=runs_root,
    run_id=run_id,
    stages=parse_stage_list(stages_text),
    device=None,
    max_epochs=epochs,
  )


def resolve_train_job(
  *,
  config: str | None,
  model: str | None,
  data_root: Path | None,
  runs_root: Path | None,
  run_id: str | None,
  stages_text: str | None,
  seed: int | None,
  max_epochs: int | None,
) -> TrainJob:
  """Build a ``TrainJob`` from ``python -m srcs.train`` flags."""
  if config is not None:
    return _train_job_from_yaml(
      config,
      model=model,
      data_root=data_root,
      runs_root=runs_root,
      run_id=run_id,
      stages_text=stages_text,
      seed=seed,
      max_epochs=max_epochs,
    )
  if data_root is None or runs_root is None:
    raise ValueError("--data-root and --runs-root required without --config")
  if stages_text is None:
    raise ValueError("--stages required without --config")
  if run_id is None:
    raise ValueError("--run-id required without --config")
  model_name = model or "baseline"
  return _train_job_from_flags(
    model_name=model_name,
    data_root=data_root,
    runs_root=runs_root,
    run_id=run_id,
    stages_text=stages_text,
    seed=seed,
    max_epochs=max_epochs,
  )
