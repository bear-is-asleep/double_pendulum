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
  per_stage_val_losses,
  finish_logged_epoch,
)
from srcs.train.curriculum_mix import (
  InletReason,
  active_val_stages,
  decide_epoch_inlet,
  fraction_dict_for_log,
  inlet_knobs_from_cfg,
  inlet_mix_at_terminal,
  stage_mses_from_val_rows,
  update_early_stop_staleness,
  validate_curriculum_cfg,
  weighted_mean_stage_mse,
  weights_for_ramp_wval,
)
from srcs.train.epoch import (
  BestCheckpointTracker,
  LossWeights,
  eval_loader_mse,
  loss_weights_from_cfg,
  pick_device,
  run_train_epoch,
)
from srcs.train.mlp_trainer import MlpTrainerMixin
from srcs.train.run_dir import curriculum_run_id, init_run_dir
from srcs.utils.run_logging import attach_training_terminal_log

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
) -> None:
  """Emit only on stagnation-driven mix injection (val-drop inlets are on the epoch line)."""
  if inlet_reason != "stagnation":
    return
  fr_before = fraction_dict_for_log(fractions_before)
  fr_after = fraction_dict_for_log(fractions_after)
  logger.info(
    "curriculum stagnation inlet: flat_epochs=%s chunk=%.4f wval_delta=%.6f "
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
) -> None:
  epoch_line = (
    f"epoch {epoch}/{epochs_cap} train={train_loss:.6f} "
    f"wval={mean_val:.6f} best={best_val:.6f} "
    f"delta={val_delta:.6f} chunk={inlet_chunk:.4f} reason={inlet_reason} "
    f"stag={stagnation_epochs} terminal={inlet_terminal} active_max={active_max} "
    f"({epoch_seconds:.1f}s)"
  )
  logger.info("-" * len(epoch_line))
  logger.info(epoch_line)
  logger.info(format_curriculum_stage_line(active_max, stage_val, train_fractions))


@dataclass
class CurriculumProgress:
  epochs_run: int = 0
  global_step: int = 0


class CurriculumTrainer(MlpTrainerMixin, StrategyTrainer):
  """Val-drop adaptive inlet; flat-band stagnation inlet (counters independent)."""

  def experiment_name(self) -> str:
    return "curriculum"

  def extra_summary_fields(self, cfg: dict[str, Any]) -> dict[str, Any]:
    knobs = inlet_knobs_from_cfg(cfg)
    return {
      "strategy": "curriculum",
      "passed_stage_min_fraction": knobs.prior_floor,
      "mix_inlet_gain": knobs.gain,
      "mix_inlet_max_chunk": knobs.max_chunk,
      "mix_min_val_delta": knobs.min_delta,
      "mix_decay_lambda": knobs.decay_lambda,
      "mix_unlock_fraction": knobs.unlock_fraction,
      "mix_stagnation_wval_band": knobs.stagnation_band,
      "mix_stagnation_patience": knobs.stagnation_patience,
      "mix_stagnation_chunk": knobs.stagnation_chunk,
      "mix_inlet_terminal_frac_tol": knobs.terminal_frac_tol,
    }

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

    validate_curriculum_cfg(cfg)
    ordered = sorted(stages)
    last_stage = int(max(ordered))
    device = pick_device(device_name)
    rid = run_id or curriculum_run_id(cfg)
    run_dir = init_run_dir(runs_root, rid, cfg)
    attach_training_terminal_log(run_dir)
    k = int(cfg["subsample_stride_k"])
    weights = loss_weights_from_cfg(cfg)
    patience = int(cfg["early_stop_patience"])
    epochs_cap = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
    knobs = inlet_knobs_from_cfg(cfg)
    pass_mse = float(cfg["stage_pass_mse"])

    stage_train_xy: dict[int, tuple[Any, Any]] = {}
    for s in ordered:
      stage_train_xy[s] = load_stage_split_xy(data_root, s, "train", k)

    model = self.build_model(cfg).to(device)
    n_params = self.count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    tracker = BestCheckpointTracker()
    ckpt_dir = run_dir / "checkpoints"
    progress = CurriculumProgress()

    train_fractions: dict[int, float] = {ordered[0]: 1.0}
    active_max = 0
    prev_wval: float | None = None
    stale_epochs = 0
    stagnation_epochs = 0
    best_wval = float("inf")
    was_inlet_terminal = False
    seed_base = int(cfg["seed"])

    logger.info(
      "curriculum adaptive inlet stages=%s epochs=%s last_stage=%s",
      ordered,
      epochs_cap,
      last_stage,
    )

    for epoch_in_run in range(1, epochs_cap + 1):
      if stale_epochs >= patience:
        logger.info("curriculum early stop stale_epochs=%s", stale_epochs)
        break

      epoch_t0 = time.perf_counter()
      val_stages = active_val_stages(active_max)
      stage_val = per_stage_val_losses(
        model, data_root, val_stages, cfg, device, weights
      )
      mses = stage_mses_from_val_rows(stage_val)
      wval_weights = weights_for_ramp_wval(active_max, train_fractions)
      wval = weighted_mean_stage_mse(mses, wval_weights)

      has_prev_wval = prev_wval is not None and _finite_wval_pair(prev_wval, wval)
      val_delta = float(prev_wval - wval) if has_prev_wval else 0.0
      active_max_before = active_max
      fractions_before = dict(train_fractions)
      inlet_epoch = decide_epoch_inlet(
        train_fractions,
        active_max,
        last_stage,
        val_delta,
        knobs,
        stagnation_epochs,
        has_prev_wval=has_prev_wval,
      )
      stagnation_epochs = inlet_epoch.stagnation_epochs
      inlet_reason = inlet_epoch.inlet_reason
      stagnation_flat_at_fire = inlet_epoch.stagnation_flat_at_fire
      inlet_out = inlet_epoch.result
      train_fractions = inlet_out.fractions
      active_max = inlet_out.active_max_stage
      inlet_chunk = inlet_out.inlet_chunk
      if inlet_chunk > 0.0:
        log_curriculum_inlet_injection(
          inlet_reason=inlet_reason,
          inlet_chunk=inlet_chunk,
          val_delta=val_delta,
          active_max_before=active_max_before,
          active_max_after=active_max,
          fractions_before=fractions_before,
          fractions_after=train_fractions,
          stagnation_flat_at_fire=stagnation_flat_at_fire,
        )

      inlet_terminal = inlet_mix_at_terminal(
        active_max,
        last_stage,
        train_fractions,
        knobs,
      )
      if math.isfinite(wval):
        prev_wval = float(wval)
      best_wval, stale_epochs, was_inlet_terminal = update_early_stop_staleness(
        wval,
        inlet_terminal=inlet_terminal,
        best_wval=best_wval,
        stale_epochs=stale_epochs,
        was_inlet_terminal=was_inlet_terminal,
      )

      train_loader = make_weighted_stage_loader(
        stage_train_xy,
        train_fractions,
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
        mean_val=wval,
        stage_val=stage_val,
        metrics_extra={
          "mix_weighted_val_mse": wval,
          "mix_val_delta": val_delta,
          "mix_inlet_chunk": inlet_chunk,
          "mix_inlet_reason": inlet_reason,
          "mix_stagnation_epochs": stagnation_epochs,
          "mix_stagnation_flat_at_fire": stagnation_flat_at_fire,
          "mix_inlet_terminal": inlet_terminal,
          "active_max_stage": active_max,
          "train_stage_fraction": fraction_dict_for_log(train_fractions),
        },
        log_fn=lambda secs: log_curriculum_epoch(
          epoch=epoch_in_run,
          epochs_cap=epochs_cap,
          train_loss=train_loss,
          mean_val=wval,
          best_val=tracker.best_mean,
          epoch_seconds=secs,
          active_max=active_max,
          val_delta=val_delta,
          inlet_chunk=inlet_chunk,
          inlet_reason=inlet_reason,
          stagnation_epochs=stagnation_epochs,
          stagnation_flat_at_fire=stagnation_flat_at_fire,
          inlet_terminal=inlet_terminal,
          stage_val=stage_val,
          train_fractions=train_fractions,
        ),
        train_steps=steps,
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
        "curriculum_last_stage": last_stage,
        "final_active_max_stage": active_max,
      },
    )


def _finite_wval_pair(prev: float, curr: float) -> bool:
  return math.isfinite(prev) and math.isfinite(curr)
