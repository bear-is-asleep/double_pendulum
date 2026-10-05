"""Cumulative curriculum mix fractions and val-weighted ramp (no torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CurriculumMixKnobs:
  """YAML-backed mix ramp and per-segment epoch cap."""

  vmin: float
  vmax: float
  prior_floor: float
  epochs_per_segment: int


def mix_knobs_from_cfg(cfg: dict[str, Any], epochs_per_segment: int) -> CurriculumMixKnobs:
  return CurriculumMixKnobs(
    vmin=float(cfg["mix_val_mse_min"]),
    vmax=float(cfg["mix_val_mse_max"]),
    prior_floor=float(cfg["passed_stage_min_fraction"]),
    epochs_per_segment=int(epochs_per_segment),
  )


def pair_fraction_from_val(wval: float, vmin: float, vmax: float) -> float:
  """
  Map weighted val MSE to upper-stage share of the (s-1, s) remainder.

  Lower val MSE means the model is ready for stage s: wval <= vmin -> 1 (all on s);
  wval >= vmax -> 0 (all remainder on s-1).
  """
  if vmax <= vmin:
    raise ValueError(f"mix_val_mse_max must exceed mix_val_mse_min, got {vmin} {vmax}")
  if not np.isfinite(wval):
    return 0.0
  t = (float(vmax) - float(wval)) / (float(vmax) - float(vmin))
  return float(np.clip(t, 0.0, 1.0))


def stage_train_fractions(
  target_s: int,
  pair_frac: float,
  prior_floor: float,
) -> dict[int, float]:
  """
  Train mass on stages 0..target_s.

  ``prior_floor`` split evenly across 0..target_s-2; remainder splits s-1 vs s.
  """
  if target_s < 0:
    raise ValueError(f"target_s must be >= 0, got {target_s}")
  if not 0.0 <= prior_floor < 1.0:
    raise ValueError(f"prior_floor must be in [0, 1), got {prior_floor}")
  if not 0.0 <= pair_frac <= 1.0:
    raise ValueError(f"pair_frac must be in [0, 1], got {pair_frac}")

  fracs: dict[int, float] = {}
  if target_s == 0:
    fracs[0] = 1.0
    return fracs

  n_prior_stages = target_s - 1
  prior_used = prior_floor if n_prior_stages > 0 else 0.0
  if prior_used > 0.0:
    each = prior_used / n_prior_stages
    for i in range(0, target_s - 1):
      fracs[i] = each

  remainder = 1.0 - prior_used
  lo = target_s - 1
  hi = target_s
  fracs[lo] = fracs.get(lo, 0.0) + remainder * (1.0 - pair_frac)
  fracs[hi] = fracs.get(hi, 0.0) + remainder * pair_frac

  total = sum(fracs.values())
  if total <= 0.0:
    raise ValueError("stage fractions sum to zero")
  if abs(total - 1.0) > 1.0e-9:
    for k in fracs:
      fracs[k] /= total
  return fracs


def cumulative_val_stages(target_s: int) -> list[int]:
  """Val pools for segment introducing ``target_s`` (stages 0..s)."""
  if target_s < 0:
    raise ValueError(f"target_s must be >= 0, got {target_s}")
  return list(range(0, target_s + 1))


def weights_for_ramp_wval(
  target_s: int,
  train_fractions: Mapping[int, float],
) -> dict[int, float]:
  """
  Weights for mix_weighted_val_mse / pair_frac.

  Every cumulative stage gets positive weight so upper-stage val is seen even
  when its train fraction is still 0.
  """
  stages = cumulative_val_stages(target_s)
  if len(stages) == 1:
    return {stages[0]: 1.0}
  raw = {s: max(float(train_fractions.get(s, 0.0)), 1.0e-6) for s in stages}
  total = sum(raw.values())
  return {s: raw[s] / total for s in stages}


def active_stages(fractions: Mapping[int, float], eps: float = 1.0e-12) -> list[int]:
  return sorted(s for s, f in fractions.items() if f > eps)


def weighted_mean_stage_mse(
  stage_mses: Mapping[int, float],
  fractions: Mapping[int, float],
) -> float:
  """Weighted mean over stages present in both maps with positive fraction."""
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


def stage_mses_from_val_rows(
  stage_val: list[Any],
) -> dict[int, float]:
  """``StageValLoss`` rows -> {stage: mse}."""
  out: dict[int, float] = {}
  for row in stage_val:
    out[int(row.stage)] = float(row.mse)
  return out


def bump_staleness(
  metric: float,
  best_metric: float,
  stale_epochs: int,
) -> tuple[float, int]:
  """Track best finite metric; increment stale counter when metric does not improve."""
  if np.isfinite(metric) and metric < best_metric:
    return float(metric), 0
  return best_metric, stale_epochs + 1


def resolve_stage_pass_mse(
  cfg: dict[str, Any],
  stages: list[int],
) -> dict[int, float]:
  """Scalar or list ``stage_pass_mse`` -> threshold per stage id in ``stages``."""
  raw = cfg["stage_pass_mse"]
  if isinstance(raw, (int, float)):
    val = float(raw)
    return {int(s): val for s in stages}
  if isinstance(raw, list):
    out: dict[int, float] = {}
    for s in stages:
      if s < 0 or s >= len(raw):
        raise ValueError(
          f"stage_pass_mse list length {len(raw)} missing index for stage {s}"
        )
      out[int(s)] = float(raw[s])
    return out
  raise TypeError(f"stage_pass_mse must be scalar or list, got {type(raw)}")


def validate_curriculum_mix_cfg(cfg: dict[str, Any]) -> None:
  vmin = float(cfg["mix_val_mse_min"])
  vmax = float(cfg["mix_val_mse_max"])
  if vmax <= vmin:
    raise ValueError(f"mix_val_mse_max must exceed mix_val_mse_min, got {vmin} {vmax}")
  pf = float(cfg["passed_stage_min_fraction"])
  if not 0.0 <= pf < 1.0:
    raise ValueError(f"passed_stage_min_fraction must be in [0, 1), got {pf}")


def fraction_dict_for_log(fractions: Mapping[int, float]) -> dict[str, float]:
  return {str(k): float(v) for k, v in sorted(fractions.items())}
