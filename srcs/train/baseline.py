"""Baseline establishment: train mixed pools, per-stage val, pass bar (Step 5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn, optim
from tqdm.auto import tqdm

from srcs.model.mlp import build_mlp, count_parameters
from srcs.model.train_data import (
  load_mixed_xy,
  load_stage_split_xy,
  make_loader,
)
from srcs.train.epoch import (
  BestCheckpointTracker,
  LossWeights,
  eval_loader_mse,
  loss_weights_from_cfg,
  pick_device,
  run_train_epoch,
  save_best_if_improved,
)
from srcs.train.run_dir import (
  append_metrics_jsonl,
  baseline_run_id,
  init_run_dir,
  save_checkpoint,
  utc_now_iso,
  write_summary,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageValLoss:
  stage: int
  mse: float


@dataclass(frozen=True)
class BaselineTrialResult:
  run_id: str
  run_dir: Path
  mean_val_mse: float
  stage_val: tuple[StageValLoss, ...]
  passes_bar: bool
  n_params: int
  epochs_run: int


def _stage_val_dict(stage_val: list[StageValLoss]) -> dict[str, float]:
  return {str(row.stage): row.mse for row in stage_val}


def _mean_finite_mse(stage_val: list[StageValLoss]) -> float:
  return float(np.mean([row.mse for row in stage_val]))


def per_stage_val_losses(
  model: nn.Module,
  data_root: Path | str,
  stages: list[int],
  cfg: dict[str, Any],
  device: torch.device,
  weights: LossWeights,
) -> list[StageValLoss]:
  k = int(cfg["subsample_stride_k"])
  bs = int(cfg["batch_size"])
  out: list[StageValLoss] = []
  for stage in stages:
    x, y = load_stage_split_xy(data_root, stage, "val", k)
    if x.shape[0] == 0:
      out.append(StageValLoss(stage=stage, mse=float("nan")))
      continue
    loader = make_loader(x, y, batch_size=bs, shuffle=False, seed=int(cfg["seed"]))
    mse = eval_loader_mse(model, loader, device, weights)
    out.append(StageValLoss(stage=stage, mse=mse))
  return out


def clears_pass_bar(stage_losses: list[StageValLoss], threshold: float) -> bool:
  """True when every stage val MSE is finite and at or below ``threshold``."""
  if not stage_losses:
    return False
  for row in stage_losses:
    if not np.isfinite(row.mse) or row.mse > threshold:
      return False
  return True


def _trial_summary(
  *,
  rid: str,
  cfg: dict[str, Any],
  stages: list[int],
  stage_val: list[StageValLoss],
  passes: bool,
  n_params: int,
  epochs_run: int,
  global_step: int,
  k: int,
) -> dict[str, Any]:
  mean_val = _mean_finite_mse(stage_val)
  return {
    "run_id": rid,
    "experiment": "baseline_establishment",
    "finished_at": utc_now_iso(),
    "mean_val_mse": mean_val,
    "stage_val_mse": _stage_val_dict(stage_val),
    "passes_stage_pass_mse": passes,
    "stage_pass_mse": float(cfg["stage_pass_mse"]),
    "n_params": n_params,
    "epochs_run": epochs_run,
    "global_step": global_step,
    "hidden_width": int(cfg["hidden_width"]),
    "hidden_depth": int(cfg["hidden_depth"]),
    "subsample_stride_k": k,
    "lr": float(cfg["lr"]),
    "batch_size": int(cfg["batch_size"]),
    "stages": stages,
  }


def train_baseline_trial(
  cfg: dict[str, Any],
  *,
  data_root: Path | str,
  stages: list[int],
  runs_root: Path | str,
  run_id: str | None = None,
  device_name: str | None = None,
  max_epochs: int | None = None,
) -> BaselineTrialResult:
  """
  Uniform mix of stage train pools; early stop on mean val MSE across ``stages``.

  Writes ``runs/<run_id>/`` with config, metrics.jsonl, checkpoints, summary.json.
  """
  device = pick_device(device_name)
  rid = run_id or baseline_run_id(cfg)
  run_dir = init_run_dir(runs_root, rid, cfg)
  k = int(cfg["subsample_stride_k"])
  weights = loss_weights_from_cfg(cfg)
  x_train, y_train = load_mixed_xy(data_root, stages, "train", k)
  train_loader = make_loader(
    x_train,
    y_train,
    batch_size=int(cfg["batch_size"]),
    shuffle=True,
    seed=int(cfg["seed"]),
  )

  model = build_mlp(cfg).to(device)
  n_params = count_parameters(model)
  optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
  patience = int(cfg["early_stop_patience"])
  epochs_cap = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
  pass_mse = float(cfg["stage_pass_mse"])

  tracker = BestCheckpointTracker()
  global_step = 0
  epochs_run = 0
  ckpt_dir = run_dir / "checkpoints"

  epoch_bar = tqdm(range(epochs_cap), desc="epoch", unit="epoch")
  for epoch in epoch_bar:
    epochs_run = epoch + 1
    global_step += run_train_epoch(model, train_loader, device, optimizer, weights)

    stage_val = per_stage_val_losses(model, data_root, stages, cfg, device, weights)
    mean_val = _mean_finite_mse(stage_val)
    append_metrics_jsonl(
      run_dir,
      {
        "epoch": epoch + 1,
        "mean_val_mse": mean_val,
        "stage_val_mse": _stage_val_dict(stage_val),
        "global_step": global_step,
      },
    )

    save_checkpoint(
      ckpt_dir / "last.pt",
      model=model,
      optimizer=optimizer,
      epoch=epoch + 1,
      global_step=global_step,
      val_metric=mean_val,
      cfg=cfg,
    )

    save_best_if_improved(
      tracker,
      mean_val,
      model=model,
      optimizer=optimizer,
      ckpt_dir=ckpt_dir,
      epoch=epoch + 1,
      global_step=global_step,
      cfg=cfg,
      save_fn=save_checkpoint,
    )
    epoch_bar.set_postfix(
      val_mse=f"{mean_val:.4f}",
      best=f"{tracker.best_mean:.4f}",
      stale=tracker.stale_epochs,
    )

    if tracker.should_stop(patience):
      logger.info("early stop epoch %s mean_val=%.6f", epoch + 1, mean_val)
      break

  tracker.restore_best(model)

  final_stage_val = per_stage_val_losses(model, data_root, stages, cfg, device, weights)
  final_mean = _mean_finite_mse(final_stage_val)
  passes = clears_pass_bar(final_stage_val, pass_mse)
  write_summary(
    run_dir,
    _trial_summary(
      rid=rid,
      cfg=cfg,
      stages=stages,
      stage_val=final_stage_val,
      passes=passes,
      n_params=n_params,
      epochs_run=epochs_run,
      global_step=global_step,
      k=k,
    ),
  )

  return BaselineTrialResult(
    run_id=rid,
    run_dir=run_dir,
    mean_val_mse=final_mean,
    stage_val=tuple(final_stage_val),
    passes_bar=passes,
    n_params=n_params,
    epochs_run=epochs_run,
  )


def iter_search_grid(model_cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
  """Cartesian product of ``baseline_search`` lists in the baseline model YAML."""
  grid = model_cfg.get("baseline_search") or {}
  widths = grid.get("hidden_widths") or [int(model_cfg["hidden_width"])]
  depths = grid.get("hidden_depths") or [int(model_cfg["hidden_depth"])]
  ks = grid.get("subsample_stride_k") or [int(model_cfg["subsample_stride_k"])]
  for w in widths:
    for d in depths:
      for k in ks:
        trial = dict(model_cfg)
        trial["hidden_width"] = int(w)
        trial["hidden_depth"] = int(d)
        trial["subsample_stride_k"] = int(k)
        yield trial


def pick_smallest_passing(trials: list[BaselineTrialResult]) -> BaselineTrialResult | None:
  passing = [t for t in trials if t.passes_bar]
  if not passing:
    return None
  passing.sort(key=lambda t: (t.n_params, t.mean_val_mse))
  return passing[0]
