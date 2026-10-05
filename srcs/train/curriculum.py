"""Curriculum: cumulative stage mix, val-scoped ramp, per-stage pass thresholds."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn, optim

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
  CurriculumMixKnobs,
  bump_staleness,
  cumulative_val_stages,
  fraction_dict_for_log,
  mix_knobs_from_cfg,
  pair_fraction_from_val,
  resolve_stage_pass_mse,
  stage_mses_from_val_rows,
  stage_train_fractions,
  validate_curriculum_mix_cfg,
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

logger = logging.getLogger(__name__)

CurriculumTrialResult = MixedPoolTrialResult


def epochs_per_stage_cap(cfg: dict[str, Any]) -> int:
  return int(cfg.get("max_epochs_per_stage") or cfg["max_epochs"])


def format_curriculum_stage_line(
  target_stage: int,
  stage_val: list[StageValLoss],
  train_fractions: dict[int, float],
) -> str:
  """One line: stage N (mse = ..., f = ...) for stages 0..target_stage."""
  mses = {int(row.stage): float(row.mse) for row in stage_val}
  parts: list[str] = []
  for s in range(0, target_stage + 1):
    mse = mses.get(s, float("nan"))
    frac = float(train_fractions.get(s, 0.0))
    if math.isfinite(mse):
      mse_text = f"{mse:.3f}"
    else:
      mse_text = "nan"
    parts.append(f"stage {s} (mse = {mse_text}, f = {frac:.2f})")
  return " | ".join(parts)


def log_curriculum_epoch(
  *,
  epoch: int,
  epochs_cap: int,
  train_loss: float,
  mean_val: float,
  best_val: float,
  epoch_seconds: float,
  target_stage: int,
  stage_val: list[StageValLoss],
  train_fractions: dict[int, float],
) -> None:
  epoch_line = (
    f"segment {target_stage} epoch {epoch}/{epochs_cap} "
    f"train={train_loss:.6f} val_mse={mean_val:.6f} best={best_val:.6f} ({epoch_seconds:.1f}s)"
  )
  logger.info("-"*len(epoch_line))
  logger.info(epoch_line)
  logger.info(format_curriculum_stage_line(target_stage, stage_val, train_fractions))



def should_advance_curriculum_stage(
  stale_epochs: int,
  patience: int,
  epoch_in_segment: int,
  epochs_cap: int,
  *,
  stage_val: list[StageValLoss] | None = None,
  target_stage: int | None = None,
  pass_mse_for_target: float | None = None,
  allow_pass_mse_advance: bool = True,
) -> bool:
  """
  Advance segment on patience, ``max_epochs_per_stage``, or when target stage val
  MSE clears ``stage_pass_mse`` (skipped on the final curriculum stage).
  """
  if stale_epochs >= patience:
    return True
  if epoch_in_segment >= epochs_cap:
    return True
  if (
    allow_pass_mse_advance
    and stage_val is not None
    and target_stage is not None
    and pass_mse_for_target is not None
  ):
    mses = stage_mses_from_val_rows(stage_val)
    mse = mses.get(int(target_stage))
    if mse is not None and math.isfinite(mse) and mse <= pass_mse_for_target:
      return True
  return False


@dataclass
class CurriculumProgress:
  epochs_run: int = 0
  global_step: int = 0


@dataclass
class _SegmentEpochOut:
  wval: float
  train_loss: float
  steps: int
  stage_val: list[StageValLoss]
  fractions: dict[int, float]
  pair_frac: float


def _run_segment_epoch(
  *,
  target_stage: int,
  pair_frac: float,
  mix: CurriculumMixKnobs,
  data_root: Path | str,
  stage_train_xy: dict[int, tuple[Any, Any]],
  cfg: dict[str, Any],
  weights: LossWeights,
  model: nn.Module,
  device: Any,
  optimizer: optim.Optimizer,
  loader_seed: int,
) -> _SegmentEpochOut:
  """
  Val on stages 0..target_stage, update pair_frac from weighted val, then train.
  """
  pre_mix = stage_train_fractions(target_stage, pair_frac, mix.prior_floor)
  val_stages = cumulative_val_stages(target_stage)
  stage_val = per_stage_val_losses(
    model, data_root, val_stages, cfg, device, weights
  )
  mses = stage_mses_from_val_rows(stage_val)
  wval_weights = weights_for_ramp_wval(target_stage, pre_mix)
  wval = weighted_mean_stage_mse(mses, wval_weights)
  new_pair = pair_fraction_from_val(wval, mix.vmin, mix.vmax)
  train_fractions = stage_train_fractions(target_stage, new_pair, mix.prior_floor)

  train_loader = make_weighted_stage_loader(
    stage_train_xy,
    train_fractions,
    batch_size=int(cfg["batch_size"]),
    seed=loader_seed,
  )
  steps = run_train_epoch(model, train_loader, device, optimizer, weights)
  train_loss = eval_loader_mse(model, train_loader, device, weights)

  return _SegmentEpochOut(
    wval=wval,
    train_loss=train_loss,
    steps=steps,
    stage_val=stage_val,
    fractions=train_fractions,
    pair_frac=new_pair,
  )


def train_curriculum_segment(
  *,
  target_stage: int,
  data_root: Path | str,
  stage_train_xy: dict[int, tuple[Any, Any]],
  cfg: dict[str, Any],
  mix: CurriculumMixKnobs,
  weights: LossWeights,
  model: nn.Module,
  device: Any,
  optimizer: optim.Optimizer,
  run_dir: Path,
  ckpt_dir: Path,
  tracker: BestCheckpointTracker,
  progress: CurriculumProgress,
  patience: int,
  pass_mse_for_target: float,
  allow_pass_mse_advance: bool,
) -> None:
  """Train until segment advance rule fires for ``target_stage``."""
  segment_stale = 0
  best_wval = float("inf")
  pair_frac = 0.0
  seed_base = int(cfg["seed"]) + target_stage * 1000

  logger.info("curriculum segment target_stage=%s", target_stage)

  for epoch_in_segment in range(1, mix.epochs_per_segment + 1):
    epoch_t0 = time.perf_counter()
    out = _run_segment_epoch(
      target_stage=target_stage,
      pair_frac=pair_frac,
      mix=mix,
      data_root=data_root,
      stage_train_xy=stage_train_xy,
      cfg=cfg,
      weights=weights,
      model=model,
      device=device,
      optimizer=optimizer,
      loader_seed=seed_base + progress.epochs_run,
    )
    pair_frac = out.pair_frac

    progress.epochs_run += 1
    progress.global_step += out.steps
    best_wval, segment_stale = bump_staleness(out.wval, best_wval, segment_stale)
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
      train_loss=out.train_loss,
      mean_val=out.wval,
      stage_val=out.stage_val,
      metrics_extra={
        "curriculum_target_stage": target_stage,
        "segment_epoch": epoch_in_segment,
        "mix_weighted_val_mse": out.wval,
        "mix_pair_fraction": pair_frac,
        "train_stage_fraction": fraction_dict_for_log(out.fractions),
      },
      log_fn=lambda secs: log_curriculum_epoch(
        epoch=epoch_in_segment,
        epochs_cap=mix.epochs_per_segment,
        train_loss=out.train_loss,
        mean_val=out.wval,
        best_val=tracker.best_mean,
        epoch_seconds=secs,
        target_stage=target_stage,
        stage_val=out.stage_val,
        train_fractions=out.fractions,
      ),
    )

    if should_advance_curriculum_stage(
      segment_stale,
      patience,
      epoch_in_segment,
      mix.epochs_per_segment,
      stage_val=out.stage_val,
      target_stage=target_stage,
      pass_mse_for_target=pass_mse_for_target,
      allow_pass_mse_advance=allow_pass_mse_advance,
    ):
      logger.info(
        "curriculum advance target_stage=%s wval=%.6f stale=%s",
        target_stage,
        out.wval,
        segment_stale,
      )
      return


class CurriculumTrainer(MlpTrainerMixin, StrategyTrainer):
  """Cumulative stages 0..s with val-driven mix and scoped validation."""

  def experiment_name(self) -> str:
    return "curriculum"

  def extra_summary_fields(self, cfg: dict[str, Any]) -> dict[str, Any]:
    mix = mix_knobs_from_cfg(cfg, epochs_per_stage_cap(cfg))
    return {
      "strategy": "curriculum",
      "mix_val_mse_min": mix.vmin,
      "mix_val_mse_max": mix.vmax,
      "passed_stage_min_fraction": mix.prior_floor,
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

    validate_curriculum_mix_cfg(cfg)
    ordered = sorted(stages)
    pass_by_stage = resolve_stage_pass_mse(cfg, ordered)
    device = pick_device(device_name)
    rid = run_id or curriculum_run_id(cfg)
    run_dir = init_run_dir(runs_root, rid, cfg)
    k = int(cfg["subsample_stride_k"])
    weights = loss_weights_from_cfg(cfg)
    patience = int(cfg["early_stop_patience"])
    epochs_per_segment = int(
      max_epochs if max_epochs is not None else epochs_per_stage_cap(cfg)
    )
    mix = mix_knobs_from_cfg(cfg, epochs_per_segment)

    stage_train_xy: dict[int, tuple[Any, Any]] = {}

    def ensure_train_stage(s: int) -> None:
      if s not in stage_train_xy:
        stage_train_xy[s] = load_stage_split_xy(data_root, s, "train", k)

    model = self.build_model(cfg).to(device)
    n_params = self.count_parameters(model)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    tracker = BestCheckpointTracker()
    ckpt_dir = run_dir / "checkpoints"
    progress = CurriculumProgress()
    last_stage = int(max(ordered))

    for target_stage in ordered:
      for s in range(0, target_stage + 1):
        ensure_train_stage(s)

      train_curriculum_segment(
        target_stage=target_stage,
        data_root=data_root,
        stage_train_xy=stage_train_xy,
        cfg=cfg,
        mix=mix,
        weights=weights,
        model=model,
        device=device,
        optimizer=optimizer,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        tracker=tracker,
        progress=progress,
        patience=patience,
        pass_mse_for_target=pass_by_stage[int(target_stage)],
        allow_pass_mse_advance=int(target_stage) != last_stage,
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
      pass_mse=pass_by_stage,
      summary_extra={
        "curriculum_last_stage": int(max(ordered)),
        "max_epochs_per_stage": epochs_per_segment,
      },
    )
