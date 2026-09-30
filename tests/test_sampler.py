"""Step 3: stage-aware sampler accept/reject."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from configs.loader import load_sampler_config
from double_pendulum.physics import PendulumParams, PendulumState, potential_energy
from double_pendulum.sampler import (
  SampleRow,
  accept_row,
  check_probe_trajectory,
  check_static_constraints,
  draw_candidate,
  sample,
)


def _tiny_cfg() -> dict:
  """Fast probe, loose energy tol for unit tests."""
  cfg = copy.deepcopy(load_sampler_config())
  cfg["probe_time"] = 0.5
  cfg["max_draw_attempts"] = 200
  cfg["seed"] = 42
  return cfg


def test_draw_stage1_zero_g_equal_mass() -> None:
  cfg = _tiny_cfg()
  rng = np.random.default_rng(0)
  for _ in range(20):
    row = draw_candidate(1, cfg, rng)
    assert row.g == cfg["g_stage1"] == 0.0
    assert row.m1 == row.m2 == cfg["m_equal"]


def test_draw_stage2_3_fixed_g() -> None:
  cfg = _tiny_cfg()
  rng = np.random.default_rng(1)
  r2 = draw_candidate(2, cfg, rng)
  r3 = draw_candidate(3, cfg, rng)
  assert r2.g == cfg["g_low"]
  assert r3.g == cfg["g_high"]
  assert r2.m1 == r2.m2
  assert r3.m1 == r3.m2


def test_stage4_rejects_low_pe() -> None:
  cfg = _tiny_cfg()
  ell = cfg["l"]
  row = SampleRow(0.0, 0.0, 0.1, -0.1, 1.0, 1.0, ell, 0.0)
  assert check_static_constraints(row, 4, cfg) == "stage4_pe_below_min"


def test_stage5_mass_gap() -> None:
  cfg = _tiny_cfg()
  ell = cfg["l"]
  g = 9.81
  state = PendulumState(1.0, 1.2, 0.0, 0.0)
  params = PendulumParams(1.0, 1.0, ell, ell, g)
  v0 = potential_energy(state, params)
  assert v0 >= cfg["stage5"]["pe_min"]
  row = SampleRow(1.0, 1.2, 0.0, 0.0, 1.0, 1.05, ell, g)
  assert check_static_constraints(row, 5, cfg) == "stage5_mass_gap"


def test_never_negative_g_in_sample() -> None:
  cfg = _tiny_cfg()
  for stage in range(1, 6):
    res = sample(stage, 3, cfg, rng=np.random.default_rng(10 + stage))
    for row in res.rows:
      assert row.g >= 0.0
      assert row.g >= cfg["g_min"]
      assert row.g <= cfg["g_max"]
      assert row.m2 > 0.0


def test_sample_returns_n_rows() -> None:
  cfg = _tiny_cfg()
  res = sample(2, 5, cfg, rng=np.random.default_rng(99))
  assert len(res.rows) == 5
  assert res.as_array().shape == (5, 8)
  assert res.attempts >= 5


def test_probe_rejects_huge_omega() -> None:
  cfg = _tiny_cfg()
  cfg["omega_traj_max"] = 0.01
  ell = cfg["l"]
  row = SampleRow(2.5, -2.0, 5.0, 5.0, 1.0, 1.0, ell, 9.81)
  assert check_probe_trajectory(row, cfg) == "omega_traj_exceeded"


def test_energy_drift_reject() -> None:
  cfg = _tiny_cfg()
  cfg["energy_abs_tol"] = 0.0
  cfg["energy_rel_tol"] = 0.0
  ell = cfg["l"]
  row = SampleRow(2.9, 2.8, 4.0, 4.0, 1.0, 1.0, ell, 9.81)
  reason = check_probe_trajectory(row, cfg)
  assert reason == "energy_drift"


def test_accept_row_happy_path() -> None:
  cfg = _tiny_cfg()
  ell = cfg["l"]
  row = SampleRow(0.2, -0.3, 0.5, -0.4, 1.0, 1.0, ell, cfg["g_low"])
  assert accept_row(row, 2, cfg) is None


def test_sample_invalid_stage() -> None:
  cfg = _tiny_cfg()
  with pytest.raises(ValueError):
    sample(0, 1, cfg)
