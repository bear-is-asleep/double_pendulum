"""Shared val + adaptive inlet step for curriculum and progressive PNN trainers."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from srcs.train.base import StageValLoss
from srcs.train.curriculum_mix import (
  AdaptiveInletKnobs,
  InletReason,
  decide_epoch_inlet,
  fraction_dict_for_log,
  inlet_mix_at_terminal,
  stage_mses_from_val_rows,
  update_early_stop_staleness,
  weighted_mean_stage_mse,
  weights_for_ramp_wval,
)


def finite_wval_pair(prev: float, curr: float) -> bool:
  return math.isfinite(prev) and math.isfinite(curr)


def mix_weighted_val_mse(
  active_max: int,
  train_fractions: dict[int, float],
  stage_val: list[StageValLoss],
) -> float:
  mses = stage_mses_from_val_rows(stage_val)
  wval_weights = weights_for_ramp_wval(active_max, train_fractions)
  return weighted_mean_stage_mse(mses, wval_weights)


@dataclass
class InletRunState:
  """Mutable counters carried across inlet training epochs."""

  train_fractions: dict[int, float]
  active_max: int = 0
  prev_wval: float | None = None
  stale_epochs: int = 0
  stagnation_epochs: int = 0
  epochs_at_active_max: int = 0
  best_wval: float = field(default_factory=lambda: float("inf"))
  was_inlet_terminal: bool = False


@dataclass(frozen=True)
class InletStepResult:
  """One inlet epoch: val mix, inlet decision, early-stop staleness update."""

  stage_val: list[StageValLoss]
  wval: float
  val_delta: float
  active_max_before: int
  fractions_before: dict[int, float]
  train_fractions: dict[int, float]
  active_max: int
  inlet_chunk: float
  inlet_reason: InletReason
  stagnation_epochs: int
  stagnation_flat_at_fire: int
  inlet_terminal: bool
  epochs_at_active_max_after: int


def step_inlet_epoch(
  state: InletRunState,
  *,
  stage_val: list[StageValLoss],
  knobs: AdaptiveInletKnobs,
  last_stage: int,
) -> InletStepResult:
  """
  Apply val-delta / stagnation inlet rules and update ``state`` in place.

  Call after per-stage validation; does not train or touch the model.
  """
  wval = mix_weighted_val_mse(state.active_max, state.train_fractions, stage_val)
  has_prev_wval = state.prev_wval is not None and finite_wval_pair(state.prev_wval, wval)
  val_delta = float(state.prev_wval - wval) if has_prev_wval else 0.0
  active_max_before = state.active_max
  fractions_before = dict(state.train_fractions)
  inlet_epoch = decide_epoch_inlet(
    state.train_fractions,
    state.active_max,
    last_stage,
    val_delta,
    knobs,
    state.stagnation_epochs,
    has_prev_wval=has_prev_wval,
    epochs_at_active_max=state.epochs_at_active_max,
  )
  state.stagnation_epochs = inlet_epoch.stagnation_epochs
  inlet_out = inlet_epoch.result
  state.train_fractions = inlet_out.fractions
  if inlet_out.active_max_stage > active_max_before:
    state.epochs_at_active_max = 0
  else:
    state.epochs_at_active_max += 1
  state.active_max = inlet_out.active_max_stage
  inlet_terminal = inlet_mix_at_terminal(
    state.active_max,
    last_stage,
    state.train_fractions,
    knobs,
  )
  if math.isfinite(wval):
    state.prev_wval = float(wval)
  state.best_wval, state.stale_epochs, state.was_inlet_terminal = update_early_stop_staleness(
    wval,
    inlet_terminal=inlet_terminal,
    best_wval=state.best_wval,
    stale_epochs=state.stale_epochs,
    was_inlet_terminal=state.was_inlet_terminal,
  )
  return InletStepResult(
    stage_val=stage_val,
    wval=wval,
    val_delta=val_delta,
    active_max_before=active_max_before,
    fractions_before=fractions_before,
    train_fractions=dict(state.train_fractions),
    active_max=state.active_max,
    inlet_chunk=inlet_out.inlet_chunk,
    inlet_reason=inlet_epoch.inlet_reason,
    stagnation_epochs=state.stagnation_epochs,
    stagnation_flat_at_fire=inlet_epoch.stagnation_flat_at_fire,
    inlet_terminal=inlet_terminal,
    epochs_at_active_max_after=state.epochs_at_active_max,
  )


def inlet_mix_metrics_extra(step: InletStepResult, **extra: Any) -> dict[str, Any]:
  """Standard ``metrics.jsonl`` keys for curriculum-style inlet runs."""
  record: dict[str, Any] = {
    "mix_weighted_val_mse": step.wval,
    "mix_val_delta": step.val_delta,
    "mix_inlet_chunk": step.inlet_chunk,
    "mix_inlet_reason": step.inlet_reason,
    "mix_stagnation_epochs": step.stagnation_epochs,
    "mix_stagnation_flat_at_fire": step.stagnation_flat_at_fire,
    "mix_inlet_terminal": step.inlet_terminal,
    "mix_epochs_at_active_max": step.epochs_at_active_max_after,
    "active_max_stage": step.active_max,
    "train_stage_fraction": fraction_dict_for_log(step.train_fractions),
  }
  record.update(extra)
  return record


def pack_inlet_state(state: InletRunState) -> dict[str, Any]:
  return {
    "train_fractions": {int(k): float(v) for k, v in state.train_fractions.items()},
    "active_max": int(state.active_max),
    "prev_wval": state.prev_wval,
    "stale_epochs": int(state.stale_epochs),
    "stagnation_epochs": int(state.stagnation_epochs),
    "epochs_at_active_max": int(state.epochs_at_active_max),
    "best_wval": float(state.best_wval),
    "was_inlet_terminal": bool(state.was_inlet_terminal),
  }


def unpack_inlet_state(data: dict[str, Any]) -> InletRunState:
  fractions = data.get("train_fractions") or {}
  train_fractions = {int(k): float(v) for k, v in fractions.items()}
  prev = data.get("prev_wval")
  prev_wval = float(prev) if prev is not None else None
  return InletRunState(
    train_fractions=train_fractions,
    active_max=int(data.get("active_max", 0)),
    prev_wval=prev_wval,
    stale_epochs=int(data.get("stale_epochs", 0)),
    stagnation_epochs=int(data.get("stagnation_epochs", 0)),
    epochs_at_active_max=int(data.get("epochs_at_active_max", 0)),
    best_wval=float(data.get("best_wval", float("inf"))),
    was_inlet_terminal=bool(data.get("was_inlet_terminal", False)),
  )


def _last_metrics_row(run_dir: Path) -> dict[str, Any] | None:
  path = run_dir / "metrics.jsonl"
  if not path.is_file():
    return None
  last: str | None = None
  with path.open(encoding="utf-8") as f:
    for line in f:
      if line.strip():
        last = line.strip()
  if last is None:
    return None
  row = json.loads(last)
  return row if isinstance(row, dict) else None


def restore_inlet_state(
  run_dir: Path,
  checkpoint_resume_state: dict[str, Any] | None,
  *,
  first_stage: int,
) -> InletRunState:
  if checkpoint_resume_state is not None:
    inlet = checkpoint_resume_state.get("inlet")
    if isinstance(inlet, dict):
      return unpack_inlet_state(inlet)
  row = _last_metrics_row(run_dir)
  if row is not None:
    logger.warning("checkpoint missing inlet state; using last metrics.jsonl row")
    fractions_raw = row.get("train_stage_fraction") or {}
    train_fractions = {int(k): float(v) for k, v in fractions_raw.items()} if isinstance(
      fractions_raw, dict
    ) else {}
    if not train_fractions:
      train_fractions = {first_stage: 1.0}
    return unpack_inlet_state(
      {
        "train_fractions": train_fractions,
        "active_max": int(row.get("active_max_stage", 0)),
        "stagnation_epochs": int(row.get("mix_stagnation_epochs", 0)),
        "epochs_at_active_max": int(row.get("mix_epochs_at_active_max", 0)),
        "was_inlet_terminal": bool(row.get("mix_inlet_terminal", False)),
      }
    )
  return InletRunState(train_fractions={first_stage: 1.0})


def column_unlock_from_checkpoint_state(
  checkpoint_resume_state: dict[str, Any] | None,
) -> dict[int, int]:
  if checkpoint_resume_state is None:
    return {0: 0}
  raw = checkpoint_resume_state.get("column_unlock_epoch")
  if not isinstance(raw, dict):
    return {0: 0}
  return {int(k): int(v) for k, v in raw.items()}
