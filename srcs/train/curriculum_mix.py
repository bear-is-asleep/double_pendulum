"""Val-delta adaptive curriculum mix (no torch)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

InletReason = Literal["delta", "stagnation", "none"]


@dataclass(frozen=True)
class AdaptiveInletKnobs:
  """YAML-backed inlet and exp template."""

  gain: float
  max_chunk: float
  min_delta: float
  decay_lambda: float
  unlock_fraction: float
  prior_floor: float
  stagnation_band: float
  stagnation_patience: int
  stagnation_chunk: float
  terminal_frac_tol: float
  min_epochs_per_stage: int


@dataclass(frozen=True)
class InletStepResult:
  fractions: dict[int, float]
  active_max_stage: int
  inlet_chunk: float
  val_delta: float


_INLET_CFG_REQUIRED_KEYS = (
  "passed_stage_min_fraction",
  "mix_inlet_gain",
  "mix_inlet_max_chunk",
  "mix_min_val_delta",
  "mix_decay_lambda",
  "mix_unlock_fraction",
)
_INLET_CFG_OPTIONAL_KEYS = (
  "mix_stagnation_wval_band",
  "mix_stagnation_patience",
  "mix_stagnation_chunk",
  "mix_inlet_terminal_frac_tol",
  "mix_min_epochs_per_stage",
)


def inlet_cfg_slice_from_mapping(data: Mapping[str, Any]) -> dict[str, Any] | None:
  """Inlet subset for ``validate_curriculum_cfg`` / theory plots, or None if incomplete."""
  if not all(k in data for k in _INLET_CFG_REQUIRED_KEYS):
    return None
  out = {k: data[k] for k in _INLET_CFG_REQUIRED_KEYS}
  for key in _INLET_CFG_OPTIONAL_KEYS:
    if key in data:
      out[key] = data[key]
  return out


def inlet_knobs_from_cfg(cfg: dict[str, Any]) -> AdaptiveInletKnobs:
  return AdaptiveInletKnobs(
    gain=float(cfg["mix_inlet_gain"]),
    max_chunk=float(cfg["mix_inlet_max_chunk"]),
    min_delta=float(cfg["mix_min_val_delta"]),
    decay_lambda=float(cfg["mix_decay_lambda"]),
    unlock_fraction=float(cfg["mix_unlock_fraction"]),
    prior_floor=float(cfg["passed_stage_min_fraction"]),
    stagnation_band=float(cfg.get("mix_stagnation_wval_band", 0.0)),
    stagnation_patience=int(cfg.get("mix_stagnation_patience", 0)),
    stagnation_chunk=float(cfg.get("mix_stagnation_chunk", 0.02)),
    terminal_frac_tol=float(cfg.get("mix_inlet_terminal_frac_tol", 0.02)),
    min_epochs_per_stage=int(cfg.get("mix_min_epochs_per_stage", 0)),
  )


def normalize_stage_fractions(fracs: Mapping[int, float]) -> dict[int, float]:
  total = sum(float(v) for v in fracs.values())
  if total <= 0.0:
    raise ValueError("stage fractions sum to zero")
  return {int(k): float(v) / total for k, v in fracs.items()}


def prior_stage_min_each(active_max: int, prior_floor: float) -> float:
  """Even share of ``passed_stage_min_fraction`` per active stage slot."""
  if active_max < 1:
    return 0.0
  if not 0.0 <= prior_floor < 1.0:
    raise ValueError(f"prior_floor must be in [0, 1), got {prior_floor}")
  return prior_floor / float(active_max)


def passed_stage_min_bounds(active_max: int, prior_floor: float) -> dict[int, float]:
  """Lower bounds on stages ``0..active_max`` when multiple stages are active."""
  each = prior_stage_min_each(active_max, prior_floor)
  if each <= 0.0:
    return {}
  return {i: each for i in range(0, active_max + 1)}


def project_slack_weighted_mins(
  proposed: Mapping[int, float],
  mins: Mapping[int, float],
  *,
  eps: float = 1.0e-12,
) -> dict[int, float]:
  """Raise shortfalls; pay deficit from slack proportional to ``f - min``."""
  fr: dict[int, float] = {int(k): float(v) for k, v in proposed.items()}
  for stage, _lo in mins.items():
    fr.setdefault(int(stage), 0.0)

  deficit = 0.0
  for stage, lo in mins.items():
    sid = int(stage)
    need = float(lo) - fr[sid]
    if need > eps:
      deficit += need
      fr[sid] = float(lo)

  if deficit <= eps:
    return normalize_stage_fractions(fr)

  slack: dict[int, float] = {}
  for sid, val in fr.items():
    headroom = val - float(mins.get(sid, 0.0))
    if headroom > eps:
      slack[sid] = headroom
  total_slack = sum(slack.values())
  if total_slack + eps < deficit:
    raise ValueError(
      f"passed_stage mins infeasible: deficit={deficit}, slack={total_slack}, "
      f"mins={dict(mins)}, proposed={dict(proposed)}"
    )
  for sid, headroom in slack.items():
    fr[sid] -= deficit * (headroom / total_slack)
  return normalize_stage_fractions(fr)


def apply_passed_stage_min_projection(
  proposed: Mapping[int, float],
  active_max: int,
  prior_floor: float,
) -> dict[int, float]:
  mins = passed_stage_min_bounds(active_max, prior_floor)
  if not mins:
    return normalize_stage_fractions(proposed)
  return project_slack_weighted_mins(proposed, mins)


def exp_template_fractions(active_max: int, decay_lambda: float) -> dict[int, float]:
  """Normalized exp weights on ``0..active_max`` (more mass on harder active stage)."""
  if active_max < 0:
    raise ValueError(f"active_max must be >= 0, got {active_max}")
  if decay_lambda < 0.0:
    raise ValueError(f"mix_decay_lambda must be >= 0, got {decay_lambda}")
  raw = {
    s: math.exp(-float(decay_lambda) * float(active_max - s))
    for s in range(0, active_max + 1)
  }
  return normalize_stage_fractions(raw)


def active_val_stages(active_max: int) -> list[int]:
  if active_max < 0:
    raise ValueError(f"active_max must be >= 0, got {active_max}")
  return list(range(0, active_max + 1))


def weights_for_ramp_wval(
  active_max: int,
  train_fractions: Mapping[int, float],
) -> dict[int, float]:
  stages = active_val_stages(active_max)
  if len(stages) == 1:
    return {stages[0]: 1.0}
  raw = {s: max(float(train_fractions.get(s, 0.0)), 1.0e-6) for s in stages}
  total = sum(raw.values())
  return {s: raw[s] / total for s in stages}


def weighted_mean_stage_mse(
  stage_mses: Mapping[int, float],
  fractions: Mapping[int, float],
) -> float:
  num = 0.0
  den = 0.0
  for stage, frac in fractions.items():
    if frac <= 0.0:
      continue
    mse = stage_mses.get(stage, float("nan"))
    if not np.isfinite(mse):
      continue
    num += frac * float(mse)
    den += frac
  if den <= 0.0:
    return float("nan")
  return num / den


def stage_mses_from_val_rows(stage_val: list[Any]) -> dict[int, float]:
  out: dict[int, float] = {}
  for row in stage_val:
    out[int(row.stage)] = float(row.mse)
  return out


def bump_staleness(
  metric: float,
  best_metric: float,
  stale_epochs: int,
) -> tuple[float, int]:
  if np.isfinite(metric) and metric < best_metric:
    return float(metric), 0
  return best_metric, stale_epochs + 1


def terminal_inlet_target_fractions(
  last_stage: int,
  knobs: AdaptiveInletKnobs,
) -> dict[int, float]:
  """Projected exp template at ``last_stage`` (inlet fixed point with prior floors)."""
  raw = exp_template_fractions(last_stage, knobs.decay_lambda)
  return apply_passed_stage_min_projection(raw, last_stage, knobs.prior_floor)


def inlet_mix_at_terminal(
  active_max: int,
  last_stage: int,
  fractions: Mapping[int, float],
  knobs: AdaptiveInletKnobs,
) -> bool:
  """
  True when all stages are active and train mix matches the terminal inlet target.

  Early stopping on wval should only run in this state.
  """
  if active_max < last_stage:
    return False
  target = terminal_inlet_target_fractions(last_stage, knobs)
  fr = normalize_stage_fractions(
    {s: float(fractions.get(s, 0.0)) for s in range(0, last_stage + 1)}
  )
  tol = knobs.terminal_frac_tol
  return all(abs(fr[s] - target[s]) <= tol for s in range(0, last_stage + 1))


def update_early_stop_staleness(
  wval: float,
  *,
  inlet_terminal: bool,
  best_wval: float,
  stale_epochs: int,
  was_inlet_terminal: bool,
) -> tuple[float, int, bool]:
  """
  Count stale epochs only after inlet reaches terminal mix.

  Resets the patience counter when leaving terminal; re-arms best wval on entry.
  """
  if not inlet_terminal:
    return best_wval, 0, False
  if not was_inlet_terminal:
    seed = float(wval) if np.isfinite(wval) else float("inf")
    return seed, 0, True
  best, stale = bump_staleness(wval, best_wval, stale_epochs)
  return best, stale, True


def maybe_raise_active_max(
  active_max: int,
  last_stage: int,
  fractions: Mapping[int, float],
  unlock_fraction: float,
) -> int:
  """Raise ceiling by one when frontier stage mass meets unlock threshold."""
  if active_max >= last_stage:
    return active_max
  fr = normalize_stage_fractions(fractions)
  if float(fr.get(active_max, 0.0)) >= unlock_fraction:
    return active_max + 1
  return active_max


def cap_active_max_raise(
  active_max: int,
  proposed: int,
  last_stage: int,
  knobs: AdaptiveInletKnobs,
  epochs_at_active_max: int,
) -> int:
  """Block raising ``active_max_stage`` until min epoch dwell at current ceiling."""
  if proposed <= active_max:
    return proposed
  if knobs.min_epochs_per_stage <= 0:
    return min(proposed, last_stage)
  if epochs_at_active_max < knobs.min_epochs_per_stage:
    return active_max
  return min(proposed, last_stage)


def is_wval_flat(val_delta: float, band: float) -> bool:
  """True when epoch-to-epoch wval move is within ``band``."""
  return abs(float(val_delta)) <= float(band)


def frozen_inlet_step(
  prev_fr: Mapping[int, float],
  active_max: int,
  val_delta: float,
) -> InletStepResult:
  cur = normalize_stage_fractions(prev_fr)
  return InletStepResult(dict(cur), active_max, 0.0, val_delta)


def blend_inlet_fractions(
  prev_fr: Mapping[int, float],
  active_max: int,
  last_stage: int,
  chunk: float,
  knobs: AdaptiveInletKnobs,
  wval_delta: float,
  *,
  epochs_at_active_max: int = 0,
) -> InletStepResult:
  """Blend mix toward exp template by ``chunk`` (not capped by ``max_chunk``)."""
  if not 0.0 < chunk <= 1.0:
    raise ValueError(f"blend chunk must be in (0, 1], got {chunk}")
  cur = normalize_stage_fractions(prev_fr)
  new_active = active_max
  if new_active == 0:
    new_active = min(1, last_stage)

  new_active = maybe_raise_active_max(
    new_active,
    last_stage,
    cur,
    knobs.unlock_fraction,
  )
  new_active = cap_active_max_raise(
    active_max,
    new_active,
    last_stage,
    knobs,
    epochs_at_active_max,
  )

  template = exp_template_fractions(new_active, knobs.decay_lambda)
  active_fr = {s: float(cur.get(s, 0.0)) for s in range(0, new_active + 1)}
  active_fr = normalize_stage_fractions(active_fr)
  blended: dict[int, float] = {}
  for s in range(0, new_active + 1):
    blended[s] = (1.0 - chunk) * active_fr[s] + chunk * template[s]

  out_fr = apply_passed_stage_min_projection(
    blended,
    new_active,
    knobs.prior_floor,
  )
  return InletStepResult(out_fr, new_active, float(chunk), wval_delta)


def step_adaptive_inlet_fractions(
  prev_fr: Mapping[int, float],
  active_max: int,
  last_stage: int,
  wval_delta: float,
  knobs: AdaptiveInletKnobs,
  *,
  epochs_at_active_max: int = 0,
) -> InletStepResult:
  """
  On wval drop: blend toward exp template by chunk; else freeze fractions.

  First qualifying drop sets ``active_max >= 1``.
  """
  if wval_delta <= knobs.min_delta:
    return frozen_inlet_step(prev_fr, active_max, wval_delta)

  chunk = min(knobs.max_chunk, knobs.gain * wval_delta)
  return blend_inlet_fractions(
    prev_fr,
    active_max,
    last_stage,
    chunk,
    knobs,
    wval_delta,
    epochs_at_active_max=epochs_at_active_max,
  )


def step_stagnation_inlet_fractions(
  prev_fr: Mapping[int, float],
  active_max: int,
  last_stage: int,
  wval_delta: float,
  knobs: AdaptiveInletKnobs,
  *,
  epochs_at_active_max: int = 0,
) -> InletStepResult:
  """Force blend by ``stagnation_chunk`` (may exceed ``max_chunk``)."""
  return blend_inlet_fractions(
    prev_fr,
    active_max,
    last_stage,
    knobs.stagnation_chunk,
    knobs,
    wval_delta,
    epochs_at_active_max=epochs_at_active_max,
  )


@dataclass(frozen=True)
class InletEpochOutcome:
  result: InletStepResult
  stagnation_epochs: int
  inlet_reason: InletReason
  # Flat epochs counted before a stagnation-fired blend (0 unless reason stagnation).
  stagnation_flat_at_fire: int = 0


def decide_epoch_inlet(
  prev_fr: Mapping[int, float],
  active_max: int,
  last_stage: int,
  val_delta: float,
  knobs: AdaptiveInletKnobs,
  stagnation_epochs: int,
  *,
  has_prev_wval: bool,
  epochs_at_active_max: int = 0,
) -> InletEpochOutcome:
  """
  Qualifying wval drop may blend adaptively; flat-band patience is independent.

  ``|val_delta| <= stagnation_band`` increments flat epochs even when a val-drop
  inlet also fired. Non-flat epochs reset the counter. After patience, stagnation
  blends from the post-adaptive mix when both apply on the same epoch.
  """
  frozen = frozen_inlet_step(prev_fr, active_max, val_delta)
  if not has_prev_wval:
    return InletEpochOutcome(frozen, stagnation_epochs, "none")

  adaptive = step_adaptive_inlet_fractions(
    prev_fr,
    active_max,
    last_stage,
    val_delta,
    knobs,
    epochs_at_active_max=epochs_at_active_max,
  )
  if adaptive.inlet_chunk > 0.0:
    step = adaptive
    mix_fr = adaptive.fractions
    mix_active = adaptive.active_max_stage
    inlet_reason: InletReason = "delta"
  else:
    step = frozen
    mix_fr = prev_fr
    mix_active = active_max
    inlet_reason = "none"

  if not is_wval_flat(val_delta, knobs.stagnation_band):
    return InletEpochOutcome(step, 0, inlet_reason)

  patience = knobs.stagnation_patience
  if patience <= 0:
    return InletEpochOutcome(step, stagnation_epochs, inlet_reason)

  flat_epochs = stagnation_epochs + 1
  if flat_epochs < patience:
    return InletEpochOutcome(step, flat_epochs, inlet_reason)

  stim = step_stagnation_inlet_fractions(
    mix_fr,
    mix_active,
    last_stage,
    val_delta,
    knobs,
    epochs_at_active_max=epochs_at_active_max,
  )
  return InletEpochOutcome(
    stim,
    0,
    "stagnation",
    stagnation_flat_at_fire=flat_epochs,
  )


def validate_curriculum_cfg(cfg: dict[str, Any]) -> None:
  pf = float(cfg["passed_stage_min_fraction"])
  if not 0.0 <= pf < 1.0:
    raise ValueError(f"passed_stage_min_fraction must be in [0, 1), got {pf}")
  gain = float(cfg["mix_inlet_gain"])
  if gain <= 0.0:
    raise ValueError(f"mix_inlet_gain must be > 0, got {gain}")
  max_chunk = float(cfg["mix_inlet_max_chunk"])
  if not 0.0 < max_chunk <= 1.0:
    raise ValueError(f"mix_inlet_max_chunk must be in (0, 1], got {max_chunk}")
  min_delta = float(cfg["mix_min_val_delta"])
  if min_delta < 0.0:
    raise ValueError(f"mix_min_val_delta must be >= 0, got {min_delta}")
  lam = float(cfg["mix_decay_lambda"])
  if lam < 0.0:
    raise ValueError(f"mix_decay_lambda must be >= 0, got {lam}")
  unlock = float(cfg["mix_unlock_fraction"])
  if not 0.0 < unlock <= 1.0:
    raise ValueError(f"mix_unlock_fraction must be in (0, 1], got {unlock}")
  band = float(cfg.get("mix_stagnation_wval_band", 0.0))
  if band < 0.0:
    raise ValueError(f"mix_stagnation_wval_band must be >= 0, got {band}")
  stag_pat = int(cfg.get("mix_stagnation_patience", 0))
  if stag_pat < 0:
    raise ValueError(f"mix_stagnation_patience must be >= 0, got {stag_pat}")
  if stag_pat > 0:
    stag_chunk = float(cfg.get("mix_stagnation_chunk", 0.02))
    if not 0.0 < stag_chunk <= 1.0:
      raise ValueError(
        f"mix_stagnation_chunk must be in (0, 1] when patience > 0, got {stag_chunk}"
      )
    if band == 0.0:
      raise ValueError(
        "mix_stagnation_wval_band must be > 0 when mix_stagnation_patience > 0"
      )
  term_tol = float(cfg.get("mix_inlet_terminal_frac_tol", 0.02))
  if term_tol <= 0.0:
    raise ValueError(f"mix_inlet_terminal_frac_tol must be > 0, got {term_tol}")
  min_ep = int(cfg.get("mix_min_epochs_per_stage", 0))
  if min_ep < 0:
    raise ValueError(f"mix_min_epochs_per_stage must be >= 0, got {min_ep}")


def fraction_dict_for_log(fractions: Mapping[int, float]) -> dict[str, float]:
  return {str(k): float(v) for k, v in sorted(fractions.items())}
