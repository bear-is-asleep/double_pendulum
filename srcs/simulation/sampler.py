"""Stage-aware IC / parameter sampler with probe-sim accept-reject.

Reads bounds from ``configs/sampler.yaml`` (pass the loaded dict as ``cfg``).
Each accepted row is one trajectory's initial conditions plus masses, length, and g.

Row fields (``SampleRow``): theta1, theta2, omega1, omega2, m1, m2, l, g.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from srcs.physics.core import (
  PendulumParams,
  PendulumState,
  eom_denominator,
  integrate_rk4,
  potential_energy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleRow:
  """One accepted trajectory seed (ICs + physical params)."""

  theta1: float
  theta2: float
  omega1: float
  omega2: float
  m1: float
  m2: float
  l: float
  g: float


@dataclass
class SampleResult:
  """Output of ``sample``."""

  stage: int
  rows: list[SampleRow]
  attempts: int = 0
  rejected: int = 0
  reject_reasons: dict[str, int] = field(default_factory=dict)

  @property
  def reject_rate(self) -> float:
    if self.attempts == 0:
      return 0.0
    return self.rejected / self.attempts

  def as_array(self) -> np.ndarray:
    """Shape (n, 8): theta1, theta2, omega1, omega2, m1, m2, l, g."""
    if not self.rows:
      return np.zeros((0, 8), dtype=np.float64)
    data = [
      (
        r.theta1,
        r.theta2,
        r.omega1,
        r.omega2,
        r.m1,
        r.m2,
        r.l,
        r.g,
      )
      for r in self.rows
    ]
    return np.array(data, dtype=np.float64)


class SamplingError(RuntimeError):
  """Could not collect ``n`` accepted rows within per-row attempt budget."""


def stage_id_bounds(cfg: dict) -> tuple[int, int]:
  """Inclusive curriculum stage id range (default ids 0..6)."""
  lo = int(cfg.get("stage_min", 0))
  hi = int(cfg.get("num_stages", 6))
  return lo, hi


def pool_index(stage: int, cfg: dict) -> int:
  """Index into ``pools.train`` / ``pools.test`` for stage id ``stage``."""
  lo, _ = stage_id_bounds(cfg)
  return stage - lo


def _stage_box(cfg: dict, stage: int) -> dict:
  key = f"stage{stage}"
  if key not in cfg:
    raise ValueError(f"unknown stage {stage}")
  return cfg[key]


def _rng_from_cfg(cfg: dict, rng: np.random.Generator | None) -> np.random.Generator:
  if rng is not None:
    return rng
  return np.random.default_rng(int(cfg.get("seed", 0)))


def _draw_uniform(rng: np.random.Generator, low: float, high: float) -> float:
  return float(rng.uniform(low, high))


def draw_candidate(
  stage: int,
  cfg: dict,
  rng: np.random.Generator,
) -> SampleRow:
  """Draw one candidate row following stage rules (no probe yet)."""
  ell = float(cfg["l"])
  m_min = float(cfg["m_min"])
  g_min = float(cfg["g_min"])
  g_max = float(cfg["g_max"])
  omega0_max = float(cfg["omega0_max"])
  box = _stage_box(cfg, stage)

  th_lo, th_hi = box["theta"]
  om_lo, om_hi = box["omega"]
  theta1 = _draw_uniform(rng, th_lo, th_hi)
  theta2 = _draw_uniform(rng, th_lo, th_hi)
  omega1 = _draw_uniform(rng, om_lo, om_hi)
  omega2 = _draw_uniform(rng, om_lo, om_hi)
  if stage == 0:
    # Same IC box as stage 1, but exactly one joint starts at rest.
    if rng.integers(2) == 0:
      omega1 = 0.0
    else:
      omega2 = 0.0

  m_equal = float(cfg["m_equal"])
  if stage in (0, 1):
    g = float(cfg["g_zero"])
    m1 = m_equal
    m2 = 0.0
  elif stage == 2:
    g = float(cfg["g_zero"])
    m1 = m2 = m_equal
  elif stage == 3:
    g = float(cfg["g_low"])
    m1 = m2 = m_equal
  elif stage == 4:
    g = float(cfg["g_high"])
    m1 = m2 = m_equal
  elif stage == 5:
    g = _draw_uniform(rng, g_min, g_max)
    m1 = m2 = m_equal
  elif stage == 6:
    g = _draw_uniform(rng, g_min, g_max)
    m_lo, m_hi = box["m"]
    m1 = _draw_uniform(rng, m_lo, m_hi)
    m2 = _draw_uniform(rng, m_lo, m_hi)
  else:
    lo, hi = stage_id_bounds(cfg)
    raise ValueError(f"stage must be {lo}..{hi}, got {stage}")

  return SampleRow(theta1, theta2, omega1, omega2, m1, m2, ell, g)


def _bump(reasons: dict[str, int], reason: str) -> str:
  reasons[reason] = reasons.get(reason, 0) + 1
  return reason


def check_static_constraints(
  row: SampleRow,
  stage: int,
  cfg: dict,
) -> str | None:
  """Return reject reason string, or None if OK."""
  g_min = float(cfg["g_min"])
  g_max = float(cfg["g_max"])
  m_min = float(cfg["m_min"])
  den_min = float(cfg["den_min"])
  omega0_max = float(cfg["omega0_max"])

  if row.g < 0.0 or row.g < g_min or row.g > g_max:
    return "gravity_out_of_band"

  m_equal = float(cfg["m_equal"])
  if stage in (0, 1):
    if row.m2 != 0.0:
      return "stage1_m2_not_zero"
    if row.m1 != m_equal:
      return "stage1_m1_not_equal"
    if row.m1 < m_min:
      return "mass_below_min"
    if stage == 0:
      one_zero = (row.omega1 == 0.0) ^ (row.omega2 == 0.0)
      if not one_zero:
        return "stage0_omega_not_singular"
  else:
    if row.m1 < m_min or row.m2 < m_min:
      return "mass_below_min"
    if row.m2 <= 0.0:
      return "m2_nonpositive"
  if abs(row.omega1) > omega0_max or abs(row.omega2) > omega0_max:
    return "omega0_exceeds_max"

  den = eom_denominator(row.theta1, row.theta2, row.m1, row.m2)
  if den < den_min:
    return "eom_denominator_too_small"

  state = PendulumState(row.theta1, row.theta2, row.omega1, row.omega2)
  params = PendulumParams(row.m1, row.m2, row.l, row.l, row.g)
  v0 = potential_energy(state, params)

  if stage == 5:
    pe_min = float(_stage_box(cfg, 5)["pe_min"])
    if v0 < pe_min:
      return "stage5_pe_below_min"
  if stage == 6:
    pe_min = float(_stage_box(cfg, 6)["pe_min"])
    mass_gap = float(_stage_box(cfg, 6)["mass_diff_min"])
    if v0 < pe_min:
      return "stage6_pe_below_min"
    if abs(row.m1 - row.m2) < mass_gap:
      return "stage6_mass_gap"

  if not np.isfinite(v0):
    return "nonfinite_pe"

  return None


def check_probe_trajectory(
  row: SampleRow,
  cfg: dict,
) -> str | None:
  """Short RK4 probe; reject unstable or drifting trajectories."""
  dt = float(cfg["dt"])
  probe_time = float(cfg["probe_time"])
  omega_traj_max = float(cfg["omega_traj_max"])
  energy_abs_tol = float(cfg["energy_abs_tol"])
  energy_rel_tol = float(cfg["energy_rel_tol"])
  energy_eps = float(cfg["energy_eps"])
  max_unwrap = cfg.get("max_mean_unwrap_rate")

  state = PendulumState(row.theta1, row.theta2, row.omega1, row.omega2)
  params = PendulumParams(row.m1, row.m2, row.l, row.l, row.g)
  traj = integrate_rk4(state, params, t_end=probe_time, dt=dt)

  if not (
    np.all(np.isfinite(traj.theta1))
    and np.all(np.isfinite(traj.theta2))
    and np.all(np.isfinite(traj.omega1))
    and np.all(np.isfinite(traj.omega2))
    and np.all(np.isfinite(traj.energy))
  ):
    return "nonfinite_state"

  if np.max(np.abs(traj.omega1)) > omega_traj_max:
    return "omega_traj_exceeded"
  if np.max(np.abs(traj.omega2)) > omega_traj_max:
    return "omega_traj_exceeded"

  e0 = traj.energy[0]
  if not np.isfinite(e0):
    return "nonfinite_energy"

  drift = np.max(np.abs(traj.energy - e0))
  rel = drift / (abs(e0) + energy_eps)
  if drift > energy_abs_tol or rel > energy_rel_tol:
    return "energy_drift"

  if max_unwrap is not None:
    rate1 = np.mean(np.abs(np.diff(np.unwrap(traj.theta1)) / dt))
    rate2 = np.mean(np.abs(np.diff(np.unwrap(traj.theta2)) / dt))
    cap = float(max_unwrap)
    if rate1 > cap or rate2 > cap:
      return "unwrap_rate"

  return None


def accept_row(row: SampleRow, stage: int, cfg: dict) -> str | None:
  """Full accept/reject for one candidate."""
  reason = check_static_constraints(row, stage, cfg)
  if reason is not None:
    return reason
  return check_probe_trajectory(row, cfg)


def sample(
  stage: int,
  n: int,
  cfg: dict,
  *,
  rng: np.random.Generator | None = None,
) -> SampleResult:
  """Draw ``n`` accepted IC rows for curriculum stage ``stage``."""
  if n < 0:
    raise ValueError("n must be non-negative")
  lo, hi = stage_id_bounds(cfg)
  if stage < lo or stage > hi:
    raise ValueError(f"stage must be {lo}..{hi}, got {stage}")

  gen = _rng_from_cfg(cfg, rng)
  max_per_row = int(cfg["max_draw_attempts"])
  rows: list[SampleRow] = []
  attempts = 0
  rejected = 0
  reasons: dict[str, int] = {}

  for _ in range(n):
    accepted = False
    for _try in range(max_per_row):
      attempts += 1
      cand = draw_candidate(stage, cfg, gen)
      reason = accept_row(cand, stage, cfg)
      if reason is None:
        rows.append(cand)
        accepted = True
        break
      rejected += 1
      _bump(reasons, reason)
    if not accepted:
      logger.warning(
        "sampler stage=%s failed row after %s attempts; reject_rate=%.3f",
        stage,
        max_per_row,
        rejected / max(attempts, 1),
      )
      raise SamplingError(
        f"stage {stage}: could not accept a row in {max_per_row} tries "
        f"({rejected}/{attempts} rejected so far)"
      )

  result = SampleResult(
    stage=stage,
    rows=rows,
    attempts=attempts,
    rejected=rejected,
    reject_reasons=reasons,
  )
  logger.info(
    "sampler stage=%s accepted=%s attempts=%s reject_rate=%.3f reasons=%s",
    stage,
    len(rows),
    attempts,
    result.reject_rate,
    reasons,
  )
  return result
