"""Unit tests for curriculum mix fractions (no torch)."""

from __future__ import annotations

import math

import pytest

from srcs.train.curriculum_mix import (
  active_stages,
  bump_staleness,
  pair_fraction_from_val,
  resolve_stage_pass_mse,
  stage_train_fractions,
  weighted_mean_stage_mse,
  weights_for_ramp_wval,
)


def test_pair_fraction_from_val_endpoints() -> None:
  assert pair_fraction_from_val(0.0, 0.1, 0.2) == 1.0
  assert pair_fraction_from_val(0.1, 0.1, 0.2) == 1.0
  assert pair_fraction_from_val(0.2, 0.1, 0.2) == 0.0
  assert math.isclose(pair_fraction_from_val(0.15, 0.1, 0.2), 0.5)


def test_stage_train_fractions_s0() -> None:
  fr = stage_train_fractions(0, 0.5, 0.05)
  assert fr == {0: 1.0}


def test_stage_train_fractions_s1() -> None:
  fr = stage_train_fractions(1, 0.0, 0.05)
  assert math.isclose(fr[0], 1.0)
  assert 1 not in fr or fr.get(1, 0.0) == 0.0

  fr_hi = stage_train_fractions(1, 1.0, 0.05)
  assert math.isclose(fr_hi[1], 1.0)


def test_stage_train_fractions_s5_prior_floor() -> None:
  fr = stage_train_fractions(5, 0.0, 0.05)
  prior_sum = sum(fr[i] for i in range(0, 4))
  assert math.isclose(prior_sum, 0.05)
  assert math.isclose(fr[4], 0.95)
  assert fr.get(5, 0.0) == 0.0

  fr_mid = stage_train_fractions(5, 0.5, 0.05)
  assert math.isclose(sum(fr_mid.values()), 1.0)
  assert math.isclose(sum(fr_mid[i] for i in range(0, 4)), 0.05)


def test_weights_for_ramp_wval_includes_upper_at_zero_train_frac() -> None:
  train = stage_train_fractions(1, 0.0, 0.05)
  w = weights_for_ramp_wval(1, train)
  assert 0 in w and 1 in w
  assert w[1] > 0.0
  assert math.isclose(sum(w.values()), 1.0)


def test_weighted_mean_stage_mse() -> None:
  mses = {0: 1.0, 1: 3.0}
  fr = {0: 0.25, 1: 0.75}
  assert weighted_mean_stage_mse(mses, fr) == 2.5


def test_active_stages() -> None:
  assert active_stages({0: 0.5, 1: 0.0, 2: 1e-13}) == [0]


def test_resolve_stage_pass_mse_scalar() -> None:
  th = resolve_stage_pass_mse({"stage_pass_mse": 0.01}, [0, 2])
  assert th == {0: 0.01, 2: 0.01}


def test_resolve_stage_pass_mse_list() -> None:
  th = resolve_stage_pass_mse({"stage_pass_mse": [0.1, 0.2, 0.3]}, [0, 1])
  assert th[0] == 0.1 and th[1] == 0.2


def test_bump_staleness() -> None:
  best, stale = bump_staleness(0.5, float("inf"), 3)
  assert best == 0.5 and stale == 0
  best, stale = bump_staleness(0.6, best, stale)
  assert stale == 1


def test_pair_fraction_bad_range() -> None:
  with pytest.raises(ValueError):
    pair_fraction_from_val(0.5, 0.2, 0.1)
