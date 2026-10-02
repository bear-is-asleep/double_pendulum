"""Shared training loop and run-folder helpers (Step 5+)."""

from __future__ import annotations

import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from srcs.loader import load_model_config
from srcs.model.train_data import (
  load_mixed_xy,
  load_stage_split_xy,
  make_loader,
)
from srcs.simulation.data import pool_path
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
class EpochMetrics:
  """One optimizer pass plus train/val MSE for logging and checkpoints."""

  train_loss: float
  stage_val: list[StageValLoss]
  mean_val: float
  steps: int


@dataclass(frozen=True)
class MixedPoolTrialResult:
  run_id: str
  run_dir: Path
  mean_val_mse: float
  stage_val: tuple[StageValLoss, ...]
  passes_bar: bool
  n_params: int
  epochs_run: int


def stage_val_dict(stage_val: list[StageValLoss]) -> dict[str, float]:
  return {str(row.stage): row.mse for row in stage_val}


def mean_finite_mse(stage_val: list[StageValLoss]) -> float:
  return float(np.mean([row.mse for row in stage_val]))


def stage_val_mse(stage_val: list[StageValLoss], stage: int) -> float:
  for row in stage_val:
    if row.stage == stage:
      return row.mse
  return float("nan")


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


def format_stage_val_line(stage_val: list[StageValLoss]) -> str:
  parts = [f"s{row.stage}={row.mse:.4f}" for row in stage_val]
  return " ".join(parts) if parts else "stage_val=-"


def log_epoch_metrics(
  *,
  epoch: int,
  epochs_cap: int,
  train_loss: float,
  mean_val: float,
  best_val: float,
  stale_epochs: int,
  stage_val: list[StageValLoss],
  epoch_seconds: float,
) -> None:
  logger.info(
    "epoch %s/%s train=%.6f val_mse=%.6f best=%.6f stale=%s %s (%.1fs)",
    epoch,
    epochs_cap,
    train_loss,
    mean_val,
    best_val,
    stale_epochs,
    format_stage_val_line(stage_val),
    epoch_seconds,
  )


def persist_epoch_artifacts(
  *,
  run_dir: Path,
  ckpt_dir: Path,
  model: nn.Module,
  optimizer: optim.Optimizer,
  tracker: BestCheckpointTracker,
  cfg: dict[str, Any],
  epoch: int,
  global_step: int,
  train_loss: float,
  mean_val: float,
  stage_val: list[StageValLoss],
  metrics_extra: dict[str, Any] | None = None,
) -> None:
  """Write metrics.jsonl row, last.pt, and best.pt when val improves."""
  record: dict[str, Any] = {
    "epoch": epoch,
    "train_loss": train_loss,
    "mean_val_mse": mean_val,
    "stage_val_mse": stage_val_dict(stage_val),
    "global_step": global_step,
  }
  if metrics_extra:
    record.update(metrics_extra)
  append_metrics_jsonl(run_dir, record)
  save_checkpoint(
    ckpt_dir / "last.pt",
    model=model,
    optimizer=optimizer,
    epoch=epoch,
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
    epoch=epoch,
    global_step=global_step,
    cfg=cfg,
    save_fn=save_checkpoint,
  )


def train_eval_epoch(
  model: nn.Module,
  train_loader: DataLoader,
  device: torch.device,
  optimizer: optim.Optimizer,
  weights: LossWeights,
  data_root: Path | str,
  val_stages: list[int],
  cfg: dict[str, Any],
) -> EpochMetrics:
  """Run one training epoch and per-stage validation MSE."""
  steps = run_train_epoch(model, train_loader, device, optimizer, weights)
  train_loss = eval_loader_mse(model, train_loader, device, weights)
  stage_val = per_stage_val_losses(
    model, data_root, val_stages, cfg, device, weights
  )
  return EpochMetrics(
    train_loss=train_loss,
    stage_val=stage_val,
    mean_val=mean_finite_mse(stage_val),
    steps=steps,
  )


def complete_trial(
  trainer: StrategyTrainer,
  *,
  tracker: BestCheckpointTracker,
  model: nn.Module,
  run_dir: Path,
  rid: str,
  cfg: dict[str, Any],
  stages: list[int],
  data_root: Path | str,
  device: torch.device,
  weights: LossWeights,
  n_params: int,
  epochs_run: int,
  global_step: int,
  k: int,
  pass_mse: float,
  summary_extra: dict[str, Any] | None = None,
) -> MixedPoolTrialResult:
  """Restore best weights, write summary.json, return trial result."""
  tracker.restore_best(model)
  final_stage_val = per_stage_val_losses(
    model, data_root, stages, cfg, device, weights
  )
  final_mean = mean_finite_mse(final_stage_val)
  passes = clears_pass_bar(final_stage_val, pass_mse)
  summary = trainer.trial_summary(
    rid=rid,
    cfg=cfg,
    stages=stages,
    stage_val=final_stage_val,
    passes=passes,
    n_params=n_params,
    epochs_run=epochs_run,
    global_step=global_step,
    k=k,
    data_root=data_root,
  )
  if summary_extra:
    summary.update(summary_extra)
  write_summary(run_dir, summary)
  return MixedPoolTrialResult(
    run_id=rid,
    run_dir=run_dir,
    mean_val_mse=final_mean,
    stage_val=tuple(final_stage_val),
    passes_bar=passes,
    n_params=n_params,
    epochs_run=epochs_run,
  )


def clears_pass_bar(stage_losses: list[StageValLoss], threshold: float) -> bool:
  """True when every stage val MSE is finite and at or below ``threshold``."""
  if not stage_losses:
    return False
  for row in stage_losses:
    if not np.isfinite(row.mse) or row.mse > threshold:
      return False
  return True


class StrategyTrainer(ABC):
  """
  Training strategy hook: build net, name experiment, optional custom ``train_trial``.

  Baseline uses ``MixedPoolTrainer``; curriculum overrides ``train_trial`` only.
  """

  @abstractmethod
  def build_model(self, cfg: dict[str, Any]) -> nn.Module:
    ...

  @abstractmethod
  def count_parameters(self, model: nn.Module) -> int:
    ...

  @abstractmethod
  def experiment_name(self) -> str:
    ...

  def trial_summary(
    self,
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
    data_root: Path | str,
  ) -> dict[str, Any]:
    mean_val = mean_finite_mse(stage_val)
    summary: dict[str, Any] = {
      "run_id": rid,
      "data_root": str(Path(data_root).resolve()),
      "experiment": self.experiment_name(),
      "finished_at": utc_now_iso(),
      "mean_val_mse": mean_val,
      "stage_val_mse": stage_val_dict(stage_val),
      "passes_stage_pass_mse": passes,
      "stage_pass_mse": float(cfg["stage_pass_mse"]),
      "n_params": n_params,
      "epochs_run": epochs_run,
      "global_step": global_step,
      "subsample_stride_k": k,
      "lr": float(cfg["lr"]),
      "batch_size": int(cfg["batch_size"]),
      "stages": stages,
      "seed": int(cfg["seed"]),
    }
    summary.update(self.extra_summary_fields(cfg))
    return summary

  def extra_summary_fields(self, cfg: dict[str, Any]) -> dict[str, Any]:
    """Architecture-specific keys for ``summary.json``."""
    return {}

  @abstractmethod
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
  ) -> MixedPoolTrialResult:
    ...


class MixedPoolTrainer(StrategyTrainer):
  """Uniform mix of stage train pools; early stop on mean val MSE across stages."""

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
  ) -> MixedPoolTrialResult:
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

    model = self.build_model(cfg).to(device)
    n_params = self.count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    patience = int(cfg["early_stop_patience"])
    epochs_cap = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
    pass_mse = float(cfg["stage_pass_mse"])

    tracker = BestCheckpointTracker()
    global_step = 0
    epochs_run = 0
    ckpt_dir = run_dir / "checkpoints"

    for epoch in range(epochs_cap):
      epoch_t0 = time.perf_counter()
      metrics = train_eval_epoch(
        model,
        train_loader,
        device,
        optimizer,
        weights,
        data_root,
        stages,
        cfg,
      )
      epochs_run = epoch + 1
      global_step += metrics.steps
      persist_epoch_artifacts(
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        cfg=cfg,
        epoch=epochs_run,
        global_step=global_step,
        train_loss=metrics.train_loss,
        mean_val=metrics.mean_val,
        stage_val=metrics.stage_val,
      )
      log_epoch_metrics(
        epoch=epochs_run,
        epochs_cap=epochs_cap,
        train_loss=metrics.train_loss,
        mean_val=metrics.mean_val,
        best_val=tracker.best_mean,
        stale_epochs=tracker.stale_epochs,
        stage_val=metrics.stage_val,
        epoch_seconds=time.perf_counter() - epoch_t0,
      )

      if tracker.should_stop(patience):
        logger.info("early stop epoch %s mean_val=%.6f", epochs_run, metrics.mean_val)
        break

    return complete_trial(
      self,
      tracker=tracker,
      model=model,
      run_dir=run_dir,
      rid=rid,
      cfg=cfg,
      stages=stages,
      data_root=data_root,
      device=device,
      weights=weights,
      n_params=n_params,
      epochs_run=epochs_run,
      global_step=global_step,
      k=k,
      pass_mse=pass_mse,
    )


def parse_stage_list(text: str) -> list[int]:
  return [int(p.strip()) for p in text.split(",") if p.strip()]


def train_cfg_from_smoke(
  preset: dict[str, Any],
  *,
  model_name: str,
  seed: int | None = None,
) -> tuple[dict[str, Any], str | None]:
  """Merge ``configs/smoke`` train block onto ``configs/models/<model_name>.yaml``."""
  train = dict(preset["train"])
  train.pop("model", None)
  device = train.pop("device", None)
  if device is not None and not isinstance(device, str):
    device = None
  cfg = load_model_config(model_name)
  cfg.update(train)
  if seed is not None:
    cfg["seed"] = int(seed)
  return cfg, device


def require_stage_pools(data_root: Path, stages: Sequence[int]) -> None:
  missing: list[str] = []
  for stage in stages:
    for split in ("train", "val"):
      path = pool_path(data_root, stage, split)
      if not path.is_file():
        missing.append(str(path))
  if missing:
    raise FileNotFoundError(
      "missing pool files (run generate_smoke_data first):\n  "
      + "\n  ".join(missing)
    )


def prepare_run_dir(runs_root: Path, run_id: str, force: bool) -> None:
  run_dir = runs_root / run_id
  if run_dir.exists():
    if not force:
      raise FileExistsError(
        f"run dir already exists: {run_dir} (pass --force to replace)"
      )
    shutil.rmtree(run_dir)
