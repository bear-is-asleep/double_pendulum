"""Shared training loop and run-folder helpers (Step 5+)."""

from __future__ import annotations

import logging
import math
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from srcs.loader import load_model_config
from srcs.model.train_data import (
  BASELINE_STAGE_MIX_EQUAL,
  baseline_stage_mix_fractions,
  concat_xy,
  load_stage_split_xy,
  make_loader,
  make_weighted_stage_loader,
  resolve_baseline_stage_mix,
  subsampled_points_per_trajectory,
)
from srcs.train.curriculum_mix import (
  fraction_dict_for_log,
  stage_mses_from_val_rows,
  weighted_mean_stage_mse,
)
from srcs.simulation.data import open_pool, pool_path
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
from srcs.train.summary_fields import strategy_summary_fields
from srcs.utils.paths import load_run_config, resolve_checkpoint_file, resolve_run_dir
from srcs.utils.run_logging import attach_training_terminal_log

logger = logging.getLogger(__name__)

_CHECKPOINT_CFG_KEYS = (
  "model_type",
  "hidden_width",
  "hidden_depth",
  "input_dim",
  "output_dim",
  "n_columns",
)


def assert_checkpoint_cfg_compatible(
  saved_cfg: dict[str, Any],
  job_cfg: dict[str, Any],
) -> None:
  for key in _CHECKPOINT_CFG_KEYS:
    if key not in saved_cfg and key not in job_cfg:
      continue
    if saved_cfg.get(key) != job_cfg.get(key):
      raise ValueError(
        f"resume checkpoint config mismatch on {key!r}: "
        f"checkpoint={saved_cfg.get(key)!r} job={job_cfg.get(key)!r}"
      )


def pack_checkpoint_resume_state(
  tracker: BestCheckpointTracker,
  *,
  inlet_state: Any = None,
  column_unlock_epoch: dict[int, int] | None = None,
) -> dict[str, Any]:
  state: dict[str, Any] = {
    "tracker": {
      "best_mean": float(tracker.best_mean),
      "stale_epochs": int(tracker.stale_epochs),
    },
  }
  if inlet_state is not None:
    from srcs.train.curriculum_inlet_loop import pack_inlet_state

    state["inlet"] = pack_inlet_state(inlet_state)
  if column_unlock_epoch is not None:
    state["column_unlock_epoch"] = {str(k): int(v) for k, v in column_unlock_epoch.items()}
  return state


@dataclass
class TrialRun:
  """Model, optimizer, and run paths for one training trial (fresh or resumed)."""

  run_dir: Path
  run_id: str
  model: nn.Module
  optimizer: optim.Optimizer
  tracker: BestCheckpointTracker
  ckpt_dir: Path
  epochs_done: int
  global_step: int
  saved_resume_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageValLoss:
  stage: int
  mse: float


StageValFn = Callable[
  [
    nn.Module,
    Path | str,
    list[int],
    dict[str, Any],
    torch.device,
    LossWeights,
  ],
  list[StageValLoss],
]


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
  *,
  split: str = "val",
  stage_ready: Callable[[nn.Module, int], bool] | None = None,
  stage_predict: Callable[[nn.Module, torch.Tensor, int], torch.Tensor] | None = None,
) -> list[StageValLoss]:
  """
  Per-stage surrogate MSE on a pool split (default ``val``).

  Default forward: ``model(x)``. Pass ``stage_predict`` for PNN ``column_index=stage``.
  """
  from srcs.train.epoch import eval_numpy_xy_mse

  k = int(cfg["subsample_stride_k"])
  bs = int(cfg["batch_size"])
  out: list[StageValLoss] = []
  for stage in stages:
    if stage_ready is not None and not stage_ready(model, stage):
      out.append(StageValLoss(stage=stage, mse=float("nan")))
      continue
    x, y = load_stage_split_xy(data_root, stage, split, k)
    if x.shape[0] == 0:
      out.append(StageValLoss(stage=stage, mse=float("nan")))
      continue
    if stage_predict is None:
      loader = make_loader(x, y, batch_size=bs, shuffle=False, seed=int(cfg["seed"]))
      mse = eval_loader_mse(model, loader, device, weights)
    else:
      stage_i = int(stage)
      mse = eval_numpy_xy_mse(
        model,
        x,
        y,
        device,
        weights,
        bs,
        lambda xb, m=model, s=stage_i: stage_predict(m, xb, s),
      )
    out.append(StageValLoss(stage=stage, mse=mse))
  return out


def format_stage_val_line(stage_val: list[StageValLoss]) -> str:
  parts = [f"val_s{row.stage}={row.mse:.4f}" for row in stage_val]
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
  weighted_val: float | None = None,
) -> None:
  wval = weighted_val if weighted_val is not None else mean_val
  logger.info(
    "epoch %s/%s train=%.6f val_mse=%.6f wval=%.6f best=%.6f stale=%s %s (%.1fs)",
    epoch,
    epochs_cap,
    train_loss,
    mean_val,
    wval,
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
  epoch_seconds: float | None = None,
  train_steps: int | None = None,
  resume_state: dict[str, Any] | None = None,
) -> None:
  """Write metrics.jsonl row, last.pt, and best.pt when val improves."""
  record: dict[str, Any] = {
    "epoch": epoch,
    "train_loss": train_loss,
    "mean_val_mse": mean_val,
    "stage_val_mse": stage_val_dict(stage_val),
    "global_step": global_step,
    "batch_size": int(cfg["batch_size"]),
  }
  if train_steps is not None:
    record["train_steps"] = int(train_steps)
  if epoch_seconds is not None:
    record["epoch_seconds"] = float(epoch_seconds)
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
    resume_state=resume_state,
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
    resume_state=resume_state,
  )


def finish_logged_epoch(
  *,
  epoch_t0: float,
  log_fn: Callable[[float], None],
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
  train_steps: int | None = None,
  resume_state: dict[str, Any] | None = None,
) -> float:
  """One wall-clock sample: write metrics.jsonl once, then log it."""
  epoch_seconds = time.perf_counter() - epoch_t0
  persist_epoch_artifacts(
    run_dir=run_dir,
    ckpt_dir=ckpt_dir,
    model=model,
    optimizer=optimizer,
    tracker=tracker,
    cfg=cfg,
    epoch=epoch,
    global_step=global_step,
    train_loss=train_loss,
    mean_val=mean_val,
    stage_val=stage_val,
    metrics_extra=metrics_extra,
    epoch_seconds=epoch_seconds,
    train_steps=train_steps,
    resume_state=resume_state,
  )
  log_fn(epoch_seconds)
  return epoch_seconds


def baseline_epoch_losses(
  metrics: EpochMetrics,
  *,
  model: nn.Module,
  mix_mode: str,
  mix_fractions: dict[int, float],
  data_root: Path | str,
  stages: list[int],
  cfg: dict[str, Any],
  device: torch.device,
  weights: LossWeights,
) -> tuple[float, float]:
  """Train loss and val MSE using the same stage mix weights as training."""
  stage_val_mses = stage_mses_from_val_rows(metrics.stage_val)
  weighted_val = weighted_mean_stage_mse(stage_val_mses, mix_fractions)
  if mix_mode == BASELINE_STAGE_MIX_EQUAL:
    stage_train = per_stage_val_losses(
      model,
      data_root,
      stages,
      cfg,
      device,
      weights,
      split="train",
    )
    train_loss = weighted_mean_stage_mse(
      stage_mses_from_val_rows(stage_train),
      mix_fractions,
    )
  else:
    train_loss = metrics.train_loss
  return train_loss, weighted_val


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
  pass_mse: float | Mapping[int, float],
  summary_extra: dict[str, Any] | None = None,
  per_stage_val_fn: StageValFn | None = None,
) -> MixedPoolTrialResult:
  """Restore best weights, write summary.json, return trial result."""
  tracker.restore_best(model)
  val_fn = per_stage_val_fn or per_stage_val_losses
  final_stage_val = val_fn(model, data_root, stages, cfg, device, weights)
  final_mean = mean_finite_mse(final_stage_val)
  passes = clears_pass_bar(
    final_stage_val,
    resolve_pass_thresholds(pass_mse, final_stage_val),
  )
  if "_n_params_trainable" not in cfg:
    cfg["_n_params_trainable"] = n_params
  if "_n_params_total" not in cfg:
    cfg["_n_params_total"] = n_params
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
    run_outcome=summary_extra,
  )
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


def resolve_pass_thresholds(
  pass_mse: float | Mapping[int, float],
  stage_losses: list[StageValLoss],
) -> dict[int, float]:
  """Scalar pass bar broadcast to each stage row; mapping used as-is."""
  if isinstance(pass_mse, Mapping):
    return {int(k): float(v) for k, v in pass_mse.items()}
  th = float(pass_mse)
  return {int(row.stage): th for row in stage_losses}


def clears_pass_bar(
  stage_losses: list[StageValLoss],
  thresholds: Mapping[int, float],
) -> bool:
  """True when each stage val MSE is finite and at or below its threshold."""
  if not stage_losses:
    return False
  for row in stage_losses:
    th = thresholds.get(int(row.stage))
    if th is None:
      return False
    if not np.isfinite(row.mse) or row.mse > float(th):
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

  def make_optimizer(self, model: nn.Module, cfg: dict[str, Any]) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=float(cfg["lr"]))

  def setup_trial(
    self,
    cfg: dict[str, Any],
    *,
    runs_root: Path | str,
    run_id: str,
    device: torch.device,
    max_epochs: int | None,
    resume_from: Path | None,
  ) -> tuple[TrialRun, int]:
    """Create or resume a run dir, load weights, tee ``train.log``."""
    epochs_cap = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
    if resume_from is None:
      run_dir = init_run_dir(runs_root, run_id, cfg)
      model = self.build_model(cfg).to(device)
      trial = TrialRun(
        run_dir=run_dir,
        run_id=run_id,
        model=model,
        optimizer=self.make_optimizer(model, cfg),
        tracker=BestCheckpointTracker(),
        ckpt_dir=run_dir / "checkpoints",
        epochs_done=0,
        global_step=0,
      )
    else:
      trial = self._trial_from_checkpoint(resume_from, cfg, device)
    attach_training_terminal_log(trial.run_dir)
    if resume_from is not None:
      done = trial.epochs_done
      if done >= epochs_cap:
        logger.info(
          "resume epoch %s already at max_epochs %s; skipping training loop",
          done,
          epochs_cap,
        )
      else:
        logger.info(
          "resume training from epoch %s (next=%s) through %s",
          done,
          done + 1,
          epochs_cap,
        )
    return trial, epochs_cap

  def _trial_from_checkpoint(
    self,
    checkpoint_path: Path | str,
    cfg: dict[str, Any],
    device: torch.device,
  ) -> TrialRun:
    ckpt_path = resolve_checkpoint_file(checkpoint_path)
    run_dir = resolve_run_dir(ckpt_path).resolve()
    if not (run_dir / "config.yaml").is_file():
      raise FileNotFoundError(f"resume run missing config.yaml: {run_dir}")
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
      raise KeyError(f"{ckpt_path}: expected dict with model_state_dict")
    saved_cfg = load_run_config(run_dir, raw.get("config"))
    assert_checkpoint_cfg_compatible(saved_cfg, cfg)
    saved_resume = raw.get("resume_state")
    if saved_resume is not None and not isinstance(saved_resume, dict):
      saved_resume = None

    model = self.build_model(cfg).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.train()

    optimizer = self.make_optimizer(model, cfg)
    opt_state = raw.get("optimizer_state_dict")
    if opt_state is not None:
      optimizer.load_state_dict(opt_state)
    else:
      logger.warning("checkpoint missing optimizer state; starting fresh optimizer")

    ckpt_dir = run_dir / "checkpoints"
    tracker = BestCheckpointTracker()
    best_path = ckpt_dir / "best.pt"
    best_state: dict[str, Any] | None = None
    best_val: float | None = None
    if best_path.is_file():
      best_raw = torch.load(best_path, map_location=device, weights_only=True)
      if isinstance(best_raw, dict):
        sd = best_raw.get("model_state_dict")
        if isinstance(sd, dict):
          best_state = sd
        if best_raw.get("val_metric") is not None:
          best_val = float(best_raw["val_metric"])
    tracker_data = (
      saved_resume.get("tracker") if isinstance(saved_resume, dict) else None
    )
    if isinstance(tracker_data, dict):
      tracker.best_mean = float(tracker_data.get("best_mean", tracker.best_mean))
      tracker.stale_epochs = int(tracker_data.get("stale_epochs", 0))
    else:
      fallback = best_val if best_val is not None else raw.get("val_metric")
      if fallback is not None and math.isfinite(float(fallback)):
        tracker.best_mean = float(fallback)
    if best_state is not None:
      tracker.best_state = {
        k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
        for k, v in best_state.items()
      }

    return TrialRun(
      run_dir=run_dir,
      run_id=run_dir.name,
      model=model,
      optimizer=optimizer,
      tracker=tracker,
      ckpt_dir=ckpt_dir,
      epochs_done=int(raw.get("epoch", 0)),
      global_step=int(raw.get("global_step", 0)),
      saved_resume_state=saved_resume,
    )

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
    run_outcome: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Build ``summary.json`` payload (core metrics + unified strategy fields)."""
    mean_val = mean_finite_mse(stage_val)
    pptr = subsampled_points_per_trajectory(
      open_pool(data_root, int(stages[0]), "train"),
      k,
    )
    summary: dict[str, Any] = {
      "run_id": rid,
      "data_root": str(Path(data_root).resolve()),
      "experiment": self.experiment_name(),
      "finished_at": utc_now_iso(),
      "mean_val_mse": mean_val,
      "stage_val_mse": stage_val_dict(stage_val),
      "passes_stage_pass_mse": passes,
      "stage_pass_mse": cfg["stage_pass_mse"],
      "n_params": n_params,
      "epochs_run": epochs_run,
      "global_step": global_step,
      "subsample_stride_k": k,
      "lr": float(cfg["lr"]),
      "batch_size": int(cfg["batch_size"]),
      "points_per_trajectory": pptr,
      "stages": stages,
      "seed": int(cfg["seed"]),
    }
    summary.update(
      strategy_summary_fields(
        cfg,
        stages=stages,
        n_params_trainable=n_params,
        run_outcome=run_outcome,
      )
    )
    return summary

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
    resume_from: Path | None = None,
  ) -> MixedPoolTrialResult:
    ...


class MixedPoolTrainer(StrategyTrainer):
  """Mixed stage train pools; early stop on unweighted mean val MSE across stages."""

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
    resume_from: Path | None = None,
  ) -> MixedPoolTrialResult:
    device = pick_device(device_name)
    rid = run_id or baseline_run_id(cfg)
    trial, epochs_cap = self.setup_trial(
      cfg,
      runs_root=runs_root,
      run_id=rid,
      device=device,
      max_epochs=max_epochs,
      resume_from=resume_from,
    )
    run_dir = trial.run_dir
    rid = trial.run_id
    model = trial.model
    optimizer = trial.optimizer
    tracker = trial.tracker
    ckpt_dir = trial.ckpt_dir
    global_step = trial.global_step
    epochs_run = trial.epochs_done
    k = int(cfg["subsample_stride_k"])
    weights = loss_weights_from_cfg(cfg)
    pairs = [load_stage_split_xy(data_root, s, "train", k) for s in stages]
    row_counts = {stages[i]: int(pairs[i][0].shape[0]) for i in range(len(stages))}
    mix_mode = resolve_baseline_stage_mix(cfg)
    mix_fractions = baseline_stage_mix_fractions(stages, row_counts, mix_mode)
    train_stage_fraction = fraction_dict_for_log(mix_fractions)
    stage_xy = {stages[i]: pairs[i] for i in range(len(stages))}
    bs = int(cfg["batch_size"])
    seed = int(cfg["seed"])
    if mix_mode == BASELINE_STAGE_MIX_EQUAL:
      train_loader = make_weighted_stage_loader(
        stage_xy,
        mix_fractions,
        batch_size=bs,
        seed=seed,
      )
    else:
      x_train, y_train = concat_xy(pairs)
      train_loader = make_loader(
        x_train,
        y_train,
        batch_size=bs,
        shuffle=True,
        seed=seed,
      )

    n_params = self.count_parameters(model)
    patience = int(cfg["early_stop_patience"])
    pass_mse = float(cfg["stage_pass_mse"])

    for epoch in range(epochs_run, epochs_cap):
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
      train_loss, weighted_val = baseline_epoch_losses(
        metrics,
        model=model,
        mix_mode=mix_mode,
        mix_fractions=mix_fractions,
        data_root=data_root,
        stages=stages,
        cfg=cfg,
        device=device,
        weights=weights,
      )
      epochs_run = epoch + 1
      global_step += metrics.steps
      finish_logged_epoch(
        epoch_t0=epoch_t0,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        cfg=cfg,
        epoch=epochs_run,
        global_step=global_step,
        train_loss=train_loss,
        mean_val=metrics.mean_val,
        stage_val=metrics.stage_val,
        log_fn=lambda secs: log_epoch_metrics(
          epoch=epochs_run,
          epochs_cap=epochs_cap,
          train_loss=train_loss,
          mean_val=metrics.mean_val,
          best_val=tracker.best_mean,
          stale_epochs=tracker.stale_epochs,
          stage_val=metrics.stage_val,
          epoch_seconds=secs,
          weighted_val=weighted_val,
        ),
        train_steps=metrics.steps,
        metrics_extra={
          "baseline_stage_mix": mix_mode,
          "train_stage_fraction": train_stage_fraction,
          "weighted_val_mse": weighted_val,
        },
        resume_state=pack_checkpoint_resume_state(tracker),
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


def require_stage_pools(data_root: Path, stages: Sequence[int]) -> None:
  missing: list[str] = []
  for stage in stages:
    for split in ("train", "val"):
      path = pool_path(data_root, stage, split)
      if not path.is_file():
        missing.append(str(path))
  if missing:
    raise FileNotFoundError(
      "missing pool files (run generate_data <preset> first):\n  "
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


