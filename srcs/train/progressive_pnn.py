"""Progressive PNN trainer: curriculum inlet unlocks columns (experiment #4)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn, optim

from srcs.model.pnn_width import apply_pnn_column_width_scale
from srcs.model.progressive_net import (
  ProgressiveNet,
  build_progressive_net,
  count_parameters as count_pnn_parameters,
  pnn_stage_predict,
  pnn_stage_ready,
  pnn_tagged_predict,
  refresh_pnn_param_counts,
  sync_progressive_cfg,
)
from srcs.model.train_data import (
  load_stage_split_xy,
  make_weighted_stage_loader_tagged,
)
from srcs.train.base import (
  MixedPoolTrialResult,
  StrategyTrainer,
  finish_logged_epoch,
  pack_checkpoint_resume_state,
  per_stage_val_losses,
  complete_trial,
)
from srcs.train.curriculum import log_curriculum_epoch, log_curriculum_inlet_injection
from srcs.train.curriculum_inlet_loop import (
  InletRunState,
  column_unlock_from_checkpoint_state,
  inlet_mix_metrics_extra,
  mix_weighted_val_mse,
  restore_inlet_state,
  step_inlet_epoch,
)
from srcs.train.curriculum_mix import (
  active_val_stages,
  inlet_knobs_from_cfg,
  validate_curriculum_cfg,
)
from srcs.train.epoch import (
  adam_on_trainable,
  eval_tagged_loader_mse,
  loss_weights_from_cfg,
  pick_device,
  run_tagged_train_epoch,
)
from srcs.train.pnn_train_mix import pnn_train_stage_fractions
from srcs.train.run_dir import progressive_run_id

logger = logging.getLogger(__name__)

ProgressiveTrialResult = MixedPoolTrialResult


@dataclass
class _Progress:
  epochs_run: int = 0
  global_step: int = 0


def _format_active_hidden_nodes_line(
  per_column_widths: list[int],
  *,
  n_columns: int,
) -> str:
  scale = max(1, int(n_columns))
  nodes = [int(w) * scale for w in per_column_widths]
  listed = ", ".join(str(n) for n in nodes)
  return f"active hidden nodes: [{listed}] ({int(n_columns)} columns)"


def _log_progressive_epoch(
  *,
  model: ProgressiveNet,
  log: logging.Logger,
  **epoch_kwargs: Any,
) -> None:
  log_curriculum_epoch(**epoch_kwargs, log=log)
  log.info(_format_active_hidden_nodes_line(model.widths, n_columns=model.n_columns))


def _per_stage_losses_pnn(
  model: nn.Module,
  data_root: Path | str,
  stages: list[int],
  cfg: dict[str, Any],
  device,
  weights,
  *,
  split: str = "val",
):
  return per_stage_val_losses(
    model,
    data_root,
    stages,
    cfg,
    device,
    weights,
    split=split,
    stage_ready=pnn_stage_ready,
    stage_predict=pnn_stage_predict,
  )


class ProgressivePnnTrainer(StrategyTrainer):
  """Curriculum inlet + one PNN column per unlocked stage."""

  def build_model(self, cfg: dict[str, Any]) -> nn.Module:
    trial_cfg = dict(cfg)
    trial_cfg.setdefault("n_columns", 1)
    return build_progressive_net(trial_cfg)

  def count_parameters(self, model: nn.Module) -> int:
    return count_pnn_parameters(model, trainable_only=True)

  def make_optimizer(self, model: nn.Module, cfg: dict[str, Any]) -> optim.Optimizer:
    return adam_on_trainable(model, float(cfg["lr"]))

  def experiment_name(self) -> str:
    return "progressive"

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
  ) -> ProgressiveTrialResult:
    if not stages:
      raise ValueError("progressive PNN requires at least one stage id")

    validate_curriculum_cfg(cfg)
    ordered = sorted(stages)
    last_stage = int(max(ordered))
    apply_pnn_column_width_scale(cfg, n_planned_stages=len(ordered))
    device = pick_device(device_name)
    rid = run_id or progressive_run_id(cfg)
    k = int(cfg["subsample_stride_k"])
    weights = loss_weights_from_cfg(cfg)
    patience = int(cfg["early_stop_patience"])
    knobs = inlet_knobs_from_cfg(cfg)
    pass_mse = float(cfg["stage_pass_mse"])

    stage_train_xy = {
      s: load_stage_split_xy(data_root, s, "train", k) for s in ordered
    }

    trial, epochs_cap = self.setup_trial(
      cfg,
      runs_root=runs_root,
      run_id=rid,
      device=device,
      max_epochs=max_epochs,
      resume_from=resume_from,
    )
    model = trial.model
    assert isinstance(model, ProgressiveNet)
    sync_progressive_cfg(cfg, model)
    if resume_from is not None:
      progress = _Progress(
        epochs_run=trial.epochs_done,
        global_step=trial.global_step,
      )
      inlet_state = restore_inlet_state(
        trial.run_dir,
        trial.saved_resume_state,
        first_stage=ordered[0],
      )
      column_unlock_epoch = column_unlock_from_checkpoint_state(trial.saved_resume_state)
    else:
      progress = _Progress()
      inlet_state = InletRunState(train_fractions={ordered[0]: 1.0})
      column_unlock_epoch = {0: 0}
    run_dir = trial.run_dir
    rid = trial.run_id
    optimizer = trial.optimizer
    tracker = trial.tracker
    ckpt_dir = trial.ckpt_dir
    n_params_trainable, n_params_total = refresh_pnn_param_counts(cfg, model)
    seed_base = int(cfg["seed"])

    logger.info(
      "progressive PNN inlet stages=%s epochs=%s last_stage=%s n_columns=%s",
      ordered,
      epochs_cap,
      last_stage,
      model.n_columns,
    )

    while progress.epochs_run < epochs_cap:
      if inlet_state.stale_epochs >= patience:
        logger.info("progressive early stop stale_epochs=%s", inlet_state.stale_epochs)
        break

      epoch_in_run = progress.epochs_run + 1
      epoch_t0 = time.perf_counter()
      val_stages = active_val_stages(inlet_state.active_max)
      stage_val = _per_stage_losses_pnn(
        model, data_root, val_stages, cfg, device, weights, split="val"
      )
      step = step_inlet_epoch(
        inlet_state,
        stage_val=stage_val,
        knobs=knobs,
        last_stage=last_stage,
      )

      optimizer_reset = False
      if step.active_max > step.active_max_before:
        for new_col in range(step.active_max_before + 1, step.active_max + 1):
          if new_col >= model.n_columns:
            model.add_column(new_col)
            model.freeze_columns_before(new_col)
            column_unlock_epoch[new_col] = progress.epochs_run + 1
        sync_progressive_cfg(cfg, model)
        optimizer = adam_on_trainable(model, float(cfg["lr"]))
        optimizer_reset = True
        n_params_trainable, n_params_total = refresh_pnn_param_counts(cfg, model)

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
          log=logger,
        )

      train_fractions = pnn_train_stage_fractions(
        step.train_fractions,
        step.active_max,
      )
      train_loader = make_weighted_stage_loader_tagged(
        stage_train_xy,
        train_fractions,
        batch_size=int(cfg["batch_size"]),
        seed=seed_base + progress.epochs_run,
      )
      steps = run_tagged_train_epoch(
        model,
        train_loader,
        device,
        optimizer,
        weights,
        pnn_tagged_predict,
      )
      stage_train = _per_stage_losses_pnn(
        model,
        data_root,
        val_stages,
        cfg,
        device,
        weights,
        split="train",
      )
      # Match wval: inlet-weighted mix over stages 0..active_max (column s on pool s).
      train_loss = mix_weighted_val_mse(
        step.active_max,
        step.train_fractions,
        stage_train,
      )
      train_loss_active = eval_tagged_loader_mse(
        model, train_loader, device, weights, pnn_tagged_predict
      )

      progress.epochs_run += 1
      progress.global_step += steps

      metrics_extra = inlet_mix_metrics_extra(
        step,
        n_columns=model.n_columns,
        n_params_trainable=n_params_trainable,
        n_params_total=n_params_total,
        column_unlock_epoch={str(k): v for k, v in column_unlock_epoch.items()},
        train_loss_active_stage_mse=train_loss_active,
        stage_train_mse={
          str(row.stage): float(row.mse) for row in stage_train
        },
      )
      if optimizer_reset:
        metrics_extra["optimizer_reset"] = True

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
        metrics_extra=metrics_extra,
        log_fn=lambda secs: _log_progressive_epoch(
          model=model,
          log=logger,
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
        resume_state=pack_checkpoint_resume_state(
          tracker,
          inlet_state=inlet_state,
          column_unlock_epoch=column_unlock_epoch,
        ),
      )

    refresh_pnn_param_counts(cfg, model)
    sync_progressive_cfg(cfg, model)

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
      n_params=n_params_trainable,
      epochs_run=progress.epochs_run,
      global_step=progress.global_step,
      k=k,
      pass_mse=pass_mse,
      per_stage_val_fn=_per_stage_losses_pnn,
      summary_extra={"final_active_max_stage": inlet_state.active_max},
    )
