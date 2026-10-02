"""Curriculum strategy: stages in order, train on current stage only (Step 7)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from srcs.model.train_data import load_stage_split_xy, make_loader
from srcs.train.base import (
  MixedPoolTrialResult,
  StrategyTrainer,
  complete_trial,
  log_epoch_metrics,
  persist_epoch_artifacts,
  stage_val_mse,
  train_eval_epoch,
)
from srcs.train.epoch import (
  BestCheckpointTracker,
  LossWeights,
  loss_weights_from_cfg,
  pick_device,
)
from srcs.train.mlp_trainer import MlpTrainerMixin
from srcs.train.run_dir import curriculum_run_id, init_run_dir

logger = logging.getLogger(__name__)

CurriculumTrialResult = MixedPoolTrialResult


def epochs_per_stage_cap(cfg: dict[str, Any]) -> int:
  return int(cfg.get("max_epochs_per_stage") or cfg["max_epochs"])


def should_advance_curriculum_stage(
  stage_mse: float,
  pass_mse: float,
  stale_epochs: int,
  patience: int,
  epoch_in_stage: int,
  epochs_cap: int,
) -> bool:
  """Advance when stage clears bar, early-stops on stage val, or hits stage epoch cap."""
  if np.isfinite(stage_mse) and stage_mse <= pass_mse:
    return True
  if stale_epochs >= patience:
    return True
  return epoch_in_stage >= epochs_cap


def bump_stage_staleness(
  stage_mse: float,
  best_stage_val: float,
  stage_stale: int,
) -> tuple[float, int]:
  if np.isfinite(stage_mse) and stage_mse < best_stage_val:
    return float(stage_mse), 0
  return best_stage_val, stage_stale + 1


@dataclass
class CurriculumProgress:
  epochs_run: int = 0
  global_step: int = 0


def train_curriculum_stage(
  *,
  stage: int,
  n_train_rows: int,
  ordered_stages: list[int],
  data_root: Path | str,
  cfg: dict[str, Any],
  weights: LossWeights,
  model: nn.Module,
  train_loader: DataLoader,
  device: torch.device,
  optimizer: optim.Optimizer,
  run_dir: Path,
  ckpt_dir: Path,
  tracker: BestCheckpointTracker,
  progress: CurriculumProgress,
  epochs_per_stage: int,
  pass_mse: float,
  patience: int,
  log_epochs_cap: int,
) -> None:
  """Train on one stage pool until advance rule fires; updates ``progress`` in place."""
  stage_stale = 0
  best_stage_val = float("inf")
  logger.info("curriculum stage %s train rows=%s", stage, n_train_rows)

  for epoch_in_stage in range(1, epochs_per_stage + 1):
    epoch_t0 = time.perf_counter()
    metrics = train_eval_epoch(
      model,
      train_loader,
      device,
      optimizer,
      weights,
      data_root,
      ordered_stages,
      cfg,
    )
    progress.epochs_run += 1
    progress.global_step += metrics.steps
    stage_mse = stage_val_mse(metrics.stage_val, stage)
    best_stage_val, stage_stale = bump_stage_staleness(
      stage_mse,
      best_stage_val,
      stage_stale,
    )

    persist_epoch_artifacts(
      run_dir=run_dir,
      ckpt_dir=ckpt_dir,
      model=model,
      optimizer=optimizer,
      tracker=tracker,
      cfg=cfg,
      epoch=progress.epochs_run,
      global_step=progress.global_step,
      train_loss=metrics.train_loss,
      mean_val=metrics.mean_val,
      stage_val=metrics.stage_val,
      metrics_extra={
        "curriculum_stage": stage,
        "stage_epoch": epoch_in_stage,
        "stage_train_mse": stage_mse,
      },
    )
    log_epoch_metrics(
      epoch=progress.epochs_run,
      epochs_cap=log_epochs_cap,
      train_loss=metrics.train_loss,
      mean_val=metrics.mean_val,
      best_val=tracker.best_mean,
      stale_epochs=stage_stale,
      stage_val=metrics.stage_val,
      epoch_seconds=time.perf_counter() - epoch_t0,
    )

    if should_advance_curriculum_stage(
      stage_mse,
      pass_mse,
      stage_stale,
      patience,
      epoch_in_stage,
      epochs_per_stage,
    ):
      logger.info(
        "curriculum advance stage %s stage_mse=%.6f stale=%s",
        stage,
        stage_mse,
        stage_stale,
      )
      return


class CurriculumTrainer(MlpTrainerMixin, StrategyTrainer):
  """Sequential stages 0..N; each segment uses that stage's train pool only."""

  def experiment_name(self) -> str:
    return "curriculum"

  def train_trial(
    self,
    cfg: dict[str, Any],
    *,
    data_root: Path | str,
    stages: list[int],
    runs_root: Path | str,
    run_id: str | None = None,
    device_name: str | None = None,
    max_epochs: int | None = None,
  ) -> CurriculumTrialResult:
    if not stages:
      raise ValueError("curriculum requires at least one stage id")

    ordered = sorted(stages)
    device = pick_device(device_name)
    rid = run_id or curriculum_run_id(cfg)
    run_dir = init_run_dir(runs_root, rid, cfg)
    k = int(cfg["subsample_stride_k"])
    weights = loss_weights_from_cfg(cfg)
    pass_mse = float(cfg["stage_pass_mse"])
    patience = int(cfg["early_stop_patience"])
    epochs_per_stage = int(
      max_epochs if max_epochs is not None else epochs_per_stage_cap(cfg)
    )

    model = self.build_model(cfg).to(device)
    n_params = self.count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    tracker = BestCheckpointTracker()
    ckpt_dir = run_dir / "checkpoints"
    progress = CurriculumProgress()
    last_stage = ordered[0]
    log_epochs_cap = epochs_per_stage * len(ordered)

    for stage in ordered:
      last_stage = stage
      x_train, y_train = load_stage_split_xy(data_root, stage, "train", k)
      train_loader = make_loader(
        x_train,
        y_train,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        seed=int(cfg["seed"]) + stage,
      )
      train_curriculum_stage(
        stage=stage,
        n_train_rows=int(x_train.shape[0]),
        ordered_stages=ordered,
        data_root=data_root,
        cfg=cfg,
        weights=weights,
        model=model,
        train_loader=train_loader,
        device=device,
        optimizer=optimizer,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        tracker=tracker,
        progress=progress,
        epochs_per_stage=epochs_per_stage,
        pass_mse=pass_mse,
        patience=patience,
        log_epochs_cap=log_epochs_cap,
      )

    return complete_trial(
      self,
      tracker=tracker,
      model=model,
      run_dir=run_dir,
      rid=rid,
      cfg=cfg,
      stages=ordered,
      data_root=data_root,
      device=device,
      weights=weights,
      n_params=n_params,
      epochs_run=progress.epochs_run,
      global_step=progress.global_step,
      k=k,
      pass_mse=pass_mse,
      summary_extra={
        "curriculum_last_stage": int(last_stage),
        "max_epochs_per_stage": epochs_per_stage,
      },
    )
