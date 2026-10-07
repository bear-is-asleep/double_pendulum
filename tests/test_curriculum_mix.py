"""Unit tests for adaptive curriculum mix (no torch)."""

from __future__ import annotations

import math

import pytest

from srcs.train.curriculum_mix import (
  AdaptiveInletKnobs,
  apply_passed_stage_min_projection,
  bump_staleness,
  decide_epoch_inlet,
  inlet_cfg_slice_from_mapping,
  inlet_mix_at_terminal,
  exp_template_fractions,
  is_wval_flat,
  terminal_inlet_target_fractions,
  update_early_stop_staleness,
  maybe_raise_active_max,
  normalize_stage_fractions,
  project_slack_weighted_mins,
  step_adaptive_inlet_fractions,
  step_stagnation_inlet_fractions,
  validate_curriculum_cfg,
  weighted_mean_stage_mse,
  weights_for_ramp_wval,
)


def _knobs(**over: float) -> AdaptiveInletKnobs:
  base = dict(
    gain=0.5,
    max_chunk=0.05,
    min_delta=1.0e-4,
    decay_lambda=1.0,
    unlock_fraction=0.15,
    prior_floor=0.05,
    stagnation_band=1.0e-5,
    stagnation_patience=0,
    stagnation_chunk=0.02,
    terminal_frac_tol=0.02,
  )
  base.update(over)
  return AdaptiveInletKnobs(**base)


def test_exp_template_favors_frontier() -> None:
  fr = exp_template_fractions(2, 1.0)
  assert fr[2] > fr[1] > fr[0]
  assert math.isclose(sum(fr.values()), 1.0)


def test_step_inlet_freezes_on_no_delta() -> None:
  fr = {0: 1.0}
  out = step_adaptive_inlet_fractions(fr, 0, 3, 0.0, _knobs())
  assert out.inlet_chunk == 0.0
  assert math.isclose(out.fractions[0], 1.0)


def test_step_inlet_first_drop_unlocks_stage_one() -> None:
  fr = {0: 1.0}
  out = step_adaptive_inlet_fractions(fr, 0, 3, 0.1, _knobs())
  assert out.active_max_stage >= 1
  assert out.inlet_chunk > 0.0
  assert 1 in out.fractions
  assert math.isclose(sum(out.fractions.values()), 1.0)


def test_maybe_raise_active_max() -> None:
  fr = {0: 0.5, 1: 0.2, 2: 0.3}
  assert maybe_raise_active_max(1, 3, fr, 0.15) == 2
  assert maybe_raise_active_max(1, 3, {0: 0.9, 1: 0.1}, 0.15) == 1


def test_weighted_mean_stage_mse() -> None:
  mses = {0: 1.0, 1: 3.0}
  fr = {0: 0.25, 1: 0.75}
  assert weighted_mean_stage_mse(mses, fr) == 2.5


def test_weights_for_ramp_wval() -> None:
  w = weights_for_ramp_wval(1, {0: 0.9, 1: 0.1})
  assert math.isclose(sum(w.values()), 1.0)
  assert w[1] > 0.0


def test_bump_staleness() -> None:
  best, stale = bump_staleness(0.5, float("inf"), 3)
  assert best == 0.5 and stale == 0
  best, stale = bump_staleness(0.6, best, stale)
  assert stale == 1


def test_project_slack_weighted_mins() -> None:
  mins = {0: 0.05, 1: 0.05}
  proposed = {0: 0.5, 1: 0.0, 2: 0.5}
  out = project_slack_weighted_mins(proposed, mins)
  assert math.isclose(out[1], 0.05)
  assert math.isclose(sum(out.values()), 1.0)


def test_apply_passed_stage_min_noop_single_stage() -> None:
  fr = {0: 1.0}
  out = apply_passed_stage_min_projection(fr, 0, 0.05)
  assert math.isclose(out[0], 1.0)


def test_is_wval_flat() -> None:
  assert is_wval_flat(0.0, 1.0e-5)
  assert is_wval_flat(-3.0e-6, 1.0e-5)
  assert not is_wval_flat(0.1, 1.0e-5)


def test_stagnation_chunk_not_capped_by_max_chunk() -> None:
  fr = {0: 0.85, 1: 0.15}
  knobs = _knobs(max_chunk=0.05, stagnation_chunk=0.1)
  out = step_stagnation_inlet_fractions(fr, 1, 3, 0.0, knobs)
  assert out.inlet_chunk == 0.1
  assert math.isclose(sum(out.fractions.values()), 1.0)


def test_inlet_mix_at_terminal() -> None:
  knobs = _knobs()
  target = terminal_inlet_target_fractions(2, knobs)
  assert inlet_mix_at_terminal(1, 2, target, knobs) is False
  assert inlet_mix_at_terminal(2, 2, target, knobs) is True
  assert inlet_mix_at_terminal(2, 2, {0: 1.0}, knobs) is False


def test_update_early_stop_staleness_gated_on_terminal() -> None:
  best, stale, was = update_early_stop_staleness(
    1.0,
    inlet_terminal=False,
    best_wval=float("inf"),
    stale_epochs=5,
    was_inlet_terminal=False,
  )
  assert stale == 0 and not was
  best, stale, was = update_early_stop_staleness(
    1.0,
    inlet_terminal=True,
    best_wval=float("inf"),
    stale_epochs=0,
    was_inlet_terminal=False,
  )
  assert best == 1.0 and stale == 0 and was
  best, stale, was = update_early_stop_staleness(
    1.1,
    inlet_terminal=True,
    best_wval=best,
    stale_epochs=stale,
    was_inlet_terminal=was,
  )
  assert stale == 1


def test_decide_epoch_inlet_resets_stagnation_when_not_flat() -> None:
  fr = {0: 1.0}
  knobs = _knobs(
    min_delta=1.0e-4,
    stagnation_band=1.0e-5,
    stagnation_patience=5,
  )
  out = decide_epoch_inlet(fr, 0, 3, 0.0, knobs, 3, has_prev_wval=True)
  assert out.stagnation_epochs == 4
  # Between band and min_delta: not flat, not adaptive.
  out = decide_epoch_inlet(fr, 0, 3, 5.0e-5, knobs, out.stagnation_epochs, has_prev_wval=True)
  assert out.stagnation_epochs == 0
  assert out.inlet_reason == "none"


def test_decide_epoch_inlet_counts_flat_while_delta_inlet() -> None:
  fr = {0: 1.0}
  knobs = _knobs(
    min_delta=1.0e-4,
    gain=1.0,
    max_chunk=1.0,
    stagnation_band=1.0e-2,
    stagnation_patience=3,
  )
  val_delta = 5.0e-4
  out = decide_epoch_inlet(fr, 0, 3, val_delta, knobs, 0, has_prev_wval=True)
  assert out.inlet_reason == "delta"
  assert out.result.inlet_chunk > 0.0
  assert out.stagnation_epochs == 1


def test_decide_epoch_inlet_stagnation_stacks_after_delta_on_fire() -> None:
  fr = {0: 1.0}
  knobs = _knobs(
    min_delta=1.0e-4,
    gain=1.0,
    max_chunk=1.0,
    stagnation_band=1.0e-2,
    stagnation_patience=3,
    stagnation_chunk=0.04,
  )
  val_delta = 5.0e-4
  stag = 0
  for _ in range(2):
    out = decide_epoch_inlet(fr, 0, 3, val_delta, knobs, stag, has_prev_wval=True)
    assert out.inlet_reason == "delta"
    stag = out.stagnation_epochs
    fr = dict(out.result.fractions)
  out = decide_epoch_inlet(fr, 0, 3, val_delta, knobs, stag, has_prev_wval=True)
  delta_only = step_adaptive_inlet_fractions(fr, 0, 3, val_delta, knobs)
  assert out.inlet_reason == "stagnation"
  assert out.stagnation_flat_at_fire == 3
  assert out.result.fractions != delta_only.fractions


def test_decide_epoch_inlet_stagnation_after_patience() -> None:
  fr = {0: 1.0}
  knobs = _knobs(
    min_delta=1.0e-4,
    stagnation_band=1.0e-3,
    stagnation_patience=3,
    stagnation_chunk=0.04,
  )
  for i in range(2):
    out = decide_epoch_inlet(fr, 0, 3, 0.0, knobs, i, has_prev_wval=True)
    assert out.inlet_reason == "none"
    assert out.stagnation_epochs == i + 1
    assert out.result.inlet_chunk == 0.0
  out = decide_epoch_inlet(fr, 0, 3, 0.0, knobs, 2, has_prev_wval=True)
  assert out.inlet_reason == "stagnation"
  assert out.stagnation_epochs == 0
  assert out.stagnation_flat_at_fire == 3
  assert out.result.inlet_chunk == 0.04
  assert out.result.active_max_stage >= 1


def test_inlet_cfg_slice_from_mapping() -> None:
  full = {
    "passed_stage_min_fraction": 0.05,
    "mix_inlet_gain": 0.5,
    "mix_inlet_max_chunk": 0.05,
    "mix_min_val_delta": 1.0e-4,
    "mix_decay_lambda": 1.0,
    "mix_unlock_fraction": 0.15,
    "mix_stagnation_wval_band": 1.0e-5,
    "mix_stagnation_patience": 5,
  }
  sliced = inlet_cfg_slice_from_mapping(full)
  assert sliced is not None
  validate_curriculum_cfg(sliced)
  assert inlet_cfg_slice_from_mapping({"mix_inlet_gain": 1.0}) is None


def test_validate_curriculum_cfg() -> None:
  validate_curriculum_cfg(
    {
      "passed_stage_min_fraction": 0.05,
      "mix_inlet_gain": 0.5,
      "mix_inlet_max_chunk": 0.05,
      "mix_min_val_delta": 1.0e-4,
      "mix_decay_lambda": 1.0,
      "mix_unlock_fraction": 0.15,
      "mix_stagnation_wval_band": 1.0e-5,
      "mix_stagnation_patience": 5,
      "mix_stagnation_chunk": 0.02,
    }
  )
  with pytest.raises(ValueError):
    validate_curriculum_cfg(
      {
        "passed_stage_min_fraction": 0.05,
        "mix_inlet_gain": 0.0,
        "mix_inlet_max_chunk": 0.05,
        "mix_min_val_delta": 1.0e-4,
        "mix_decay_lambda": 1.0,
        "mix_unlock_fraction": 0.15,
      }
    )
  with pytest.raises(ValueError):
    validate_curriculum_cfg(
      {
        "passed_stage_min_fraction": 0.05,
        "mix_inlet_gain": 0.5,
        "mix_inlet_max_chunk": 0.05,
        "mix_min_val_delta": 1.0e-4,
        "mix_decay_lambda": 1.0,
        "mix_unlock_fraction": 0.15,
        "mix_stagnation_wval_band": 0.0,
        "mix_stagnation_patience": 3,
      }
    )
