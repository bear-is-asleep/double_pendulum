"""Curriculum: val-delta adaptive inlet across stages (single training loop)."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import optim

from srcs.model.train_data import (
  load_stage_split_xy,
  make_weighted_stage_loader,
)
from srcs.train.base import (
  MixedPoolTrialResult,
  StageValLoss,
  StrategyTrainer,
  complete_trial,
  pack_checkpoint_resume_state,
  per_stage_val_losses,
  finish_logged_epoch,
)
from srcs.train.curriculum_inlet_loop import (
  InletRunState,
  inlet_mix_metrics_extra,
  restore_inlet_state,
  step_inlet_epoch,
)
from srcs.train.curriculum_mix import (
  InletReason,
  active_val_stages,
  fraction_dict_for_log,
  inlet_knobs_from_cfg,
  validate_curriculum_cfg,
)
from srcs.train.epoch import (
  LossWeights,
  eval_loader_mse,
  loss_weights_from_cfg,
  pick_device,
  run_train_epoch,
)
from srcs.train.mlp_trainer import MlpTrainerMixin
from srcs.train.run_dir import curriculum_run_id

logger = logging.getLogger(__name__)

CurriculumTrialResult = MixedPoolTrialResult


def format_curriculum_stage_line(
  active_max: int,
  stage_val: list[StageValLoss],
  train_fractions: dict[int, float],
) -> str:
  mses = {int(row.stage): float(row.mse) for row in stage_val}
  parts: list[str] = []
  for s in range(0, active_max + 1):
    mse = mses.get(s, float("nan"))
    frac = float(train_fractions.get(s, 0.0))
    if math.isfinite(mse):
      mse_text = f"{mse:.3f}"
    else:
      mse_text = "nan"
    parts.append(f"stage {s} (mse = {mse_text}, f = {frac:.2f})")
  return " | ".join(parts)


def log_curriculum_inlet_injection(
  *,
  inlet_reason: InletReason,
  inlet_chunk: float,
  val_delta: float,
  active_max_before: int,
  active_max_after: int,
  fractions_before: dict[int, float],
  fractions_after: dict[int, float],
  stagnation_flat_at_fire: int,
  log: logging.Logger | None = None,
) -> None:
  """Emit only on stagnation-driven mix injection (val-drop inlets are on the epoch line)."""
  if inlet_reason != "stagnation":
    return
  out = log if log is not None else logger
  fr_before = fraction_dict_for_log(fractions_before)
  fr_after = fraction_dict_for_log(fractions_after)
  out.info(
    "stagnation inlet: flat_epochs=%s chunk=%.4f wval_delta=%.6f "
    "active_max %s->%s mix %s -> %s",
    stagnation_flat_at_fire,
    inlet_chunk,
    val_delta,
    active_max_before,
    active_max_after,
    fr_before,
    fr_after,
  )


def log_curriculum_epoch(
  *,
  epoch: int,
  epochs_cap: int,
  train_loss: float,
  mean_val: float,
  best_val: float,
  epoch_seconds: float,
  active_max: int,
  val_delta: float,
  inlet_chunk: float,
  inlet_reason: InletReason,
  stagnation_epochs: int,
  stagnation_flat_at_fire: int,
  inlet_terminal: bool,
  stage_val: list[StageValLoss],
  train_fractions: dict[int, float],
  log: logging.Logger | None = None,
) -> None:
  out = log if log is not None else logger
  epoch_line = (
    f"epoch {epoch}/{epochs_cap} train={train_loss:.6f} "
    f"wval={mean_val:.6f} best={best_val:.6f} "
    f"delta={val_delta:.6f} chunk={inlet_chunk:.4f} reason={inlet_reason} "
    f"stag={stagnation_epochs} terminal={inlet_terminal} active_max={active_max} "
    f"({epoch_seconds:.1f}s)"
  )
  out.info("-" * len(epoch_line))
  out.info(epoch_line)
  out.info(format_curriculum_stage_line(active_max, stage_val, train_fractions))


@dataclass
class CurriculumProgress:
  epochs_run: int = 0
  global_step: int = 0


class CurriculumTrainer(MlpTrainerMixin, StrategyTrainer):
  """Val-drop adaptive inlet; flat-band stagnation inlet (counters independent)."""

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
    resume_from: Path | None = None,
  ) -> CurriculumTrialResult:
    if not stages:
      raise ValueError("curriculum requires at least one stage id")

    validate_curriculum_cfg(cfg)
    ordered = sorted(stages)
    last_stage = int(max(ordered))
    device = pick_device(device_name)
    rid = run_id or curriculum_run_id(cfg)
    k = int(cfg["subsample_stride_k"])
    weights = loss_weights_from_cfg(cfg)
    patience = int(cfg["early_stop_patience"])
    knobs = inlet_knobs_from_cfg(cfg)
    pass_mse = float(cfg["stage_pass_mse"])

    stage_train_xy: dict[int, tuple[Any, Any]] = {}
    for s in ordered:
      stage_train_xy[s] = load_stage_split_xy(data_root, s, "train", k)

    trial, epochs_cap = self.setup_trial(
      cfg,
      runs_root=runs_root,
      run_id=rid,
      device=device,
      max_epochs=max_epochs,
      resume_from=resume_from,
    )
    if resume_from is not None:
      progress = CurriculumProgress(
        epochs_run=trial.epochs_done,
        global_step=trial.global_step,
      )
      inlet_state = restore_inlet_state(
        trial.run_dir,
        trial.saved_resume_state,
        first_stage=ordered[0],
      )
    else:
      progress = CurriculumProgress()
      inlet_state = InletRunState(train_fractions={ordered[0]: 1.0})
    run_dir = trial.run_dir
    rid = trial.run_id
    model = trial.model
    optimizer = trial.optimizer
    tracker = trial.tracker
    ckpt_dir = trial.ckpt_dir
    n_params = self.count_parameters(model)
    seed_base = int(cfg["seed"])

    logger.info(
      "curriculum adaptive inlet stages=%s epochs=%s last_stage=%s",
      ordered,
      epochs_cap,
      last_stage,
    )

    while progress.epochs_run < epochs_cap:
      if inlet_state.stale_epochs >= patience:
        logger.info("curriculum early stop stale_epochs=%s", inlet_state.stale_epochs)
        break

      epoch_in_run = progress.epochs_run + 1
      epoch_t0 = time.perf_counter()
      val_stages = active_val_stages(inlet_state.active_max)
      stage_val = per_stage_val_losses(
        model, data_root, val_stages, cfg, device, weights
      )
      step = step_inlet_epoch(
        inlet_state,
        stage_val=stage_val,
        knobs=knobs,
        last_stage=last_stage,
      )
      if step.inlet_chunk > 0.0:
        log_curriculum_inlet_injection(
          inlet_reason=step.inlet_reason,
          inlet_chunk=step.inlet_chunk,
          val_delta=step.val_delta,
          active_max_before=step.active_max_before,
          active_max_after=step.active_max,
          fractions_before=step.fractions_before,
          fractions_after=step.train_fractions,
          stagnation_flat_at_fire=step.stagnation_flat_at_fire,
        )

      train_loader = make_weighted_stage_loader(
        stage_train_xy,
        step.train_fractions,
        batch_size=int(cfg["batch_size"]),
        seed=seed_base + progress.epochs_run,
      )
      steps = run_train_epoch(model, train_loader, device, optimizer, weights)
      train_loss = eval_loader_mse(model, train_loader, device, weights)

      progress.epochs_run += 1
      progress.global_step += steps

      finish_logged_epoch(
        epoch_t0=epoch_t0,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        cfg=cfg,
        epoch=progress.epochs_run,
        global_step=progress.global_step,
        train_loss=train_loss,
        mean_val=step.wval,
        stage_val=stage_val,
        metrics_extra=inlet_mix_metrics_extra(step),
        log_fn=lambda secs: log_curriculum_epoch(
          epoch=epoch_in_run,
          epochs_cap=epochs_cap,
          train_loss=train_loss,
          mean_val=step.wval,
          best_val=tracker.best_mean,
          epoch_seconds=secs,
          active_max=step.active_max,
          val_delta=step.val_delta,
          inlet_chunk=step.inlet_chunk,
          inlet_reason=step.inlet_reason,
          stagnation_epochs=step.stagnation_epochs,
          stagnation_flat_at_fire=step.stagnation_flat_at_fire,
          inlet_terminal=step.inlet_terminal,
          stage_val=stage_val,
          train_fractions=step.train_fractions,
        ),
        train_steps=steps,
        resume_state=pack_checkpoint_resume_state(tracker, inlet_state=inlet_state),
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
      summary_extra={"final_active_max_stage": inlet_state.active_max},
    )
