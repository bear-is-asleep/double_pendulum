"""Resolve CLI / smoke preset into a single train job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srcs.loader import load_model_config, load_smoke_preset
from srcs.train.base import parse_stage_list, train_cfg_from_smoke


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


def resolve_train_job(
  *,
  smoke: str | None,
  model: str | None,
  data_root: Path | None,
  runs_root: Path | None,
  run_id: str | None,
  stages_text: str | None,
  seed: int | None,
  max_epochs: int | None,
) -> TrainJob:
  """
  Build a ``TrainJob`` from ``python -m srcs.train`` flags.

  Raises ``ValueError`` when required fields are missing.
  """
  preset = load_smoke_preset(smoke) if smoke else None
  model_name = model
  if model_name is None:
    model_name = (
      str(preset["train"].get("model", "baseline"))
      if preset is not None
      else "baseline"
    )

  if preset is not None:
    root = data_root or Path(preset["data_root"])
    runs = runs_root or Path(preset["runs_root"])
    rid = run_id or str(preset["run_id"])
    stage_list = (
      parse_stage_list(stages_text) if stages_text else list(preset["stages"])
    )
    cfg, preset_device = train_cfg_from_smoke(
      preset,
      model_name=model_name,
      seed=seed,
    )
    device = preset_device
  else:
    if data_root is None or runs_root is None:
      raise ValueError("--data-root and --runs-root required without --smoke")
    if stages_text is None:
      raise ValueError("--stages required without --smoke")
    if run_id is None:
      raise ValueError("--run-id required without --smoke")
    root = data_root
    runs = runs_root
    rid = run_id
    stage_list = parse_stage_list(stages_text)
    cfg = load_model_config(model_name)
    if seed is not None:
      cfg["seed"] = int(seed)
    device = None

  epochs = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
  cfg["max_epochs"] = epochs

  return TrainJob(
    model_name=model_name,
    cfg=cfg,
    data_root=root,
    runs_root=runs,
    run_id=rid,
    stages=stage_list,
    device=device,
    max_epochs=epochs,
  )
