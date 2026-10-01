"""Dataset pools on disk as NumPy ``.npz`` files (Step 4).

Layout under ``data_root`` (default ``data/``)::

  stage{S}_{split}.npz   # S in 0..6, split in {train, val, test}

Array keys (this is the schema; keep stable for readers / Step 4b)::

  stage       int32 scalar
  split       unicode scalar  ("train" | "val" | "test")
  frozen      bool scalar     (True for frozen test; refuse overwrite)
  t           float64 (n_t,)  shared time grid (full resolution; k is train-time)
  params      float64 (n, 8)  columns:
                theta10, theta20, omega10, omega20, m1, m2, l, g
  theta1      float64 (n, n_t)  wrapped to (-pi, pi]
  theta2      float64 (n, n_t)  wrapped to (-pi, pi]
  sin_theta1  float64 (n, n_t)
  cos_theta1  float64 (n, n_t)
  sin_theta2  float64 (n, n_t)
  cos_theta2  float64 (n, n_t)
  omega1      float64 (n, n_t)
  omega2      float64 (n, n_t)
  kinetic     float64 (n, n_t)
  potential   float64 (n, n_t)
  energy      float64 (n, n_t)

Pointwise training row (full-res expand; subsample stride k applied later)::

  X: (t, theta10, theta20, omega10, omega20, m1, m2, g)   # dim 8; l omitted
  Y: (sin_theta1, cos_theta1, sin_theta2, cos_theta2, omega1, omega2)  # dim 6

Do not delete frozen test pools without an explicit force flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from srcs.physics.core import (
  PendulumParams,
  PendulumState,
  Trajectory,
  integrate_rk4,
)
from srcs.simulation.sampler import SampleRow, sample

logger = logging.getLogger(__name__)

PARAM_COLS = (
  "theta10",
  "theta20",
  "omega10",
  "omega20",
  "m1",
  "m2",
  "l",
  "g",
)
SPLITS = ("train", "val", "test")
SERIES_KEYS = (
  "theta1",
  "theta2",
  "sin_theta1",
  "cos_theta1",
  "sin_theta2",
  "cos_theta2",
  "omega1",
  "omega2",
  "kinetic",
  "potential",
  "energy",
)


class FrozenPoolError(RuntimeError):
  """Refused to overwrite a frozen test pool."""


@dataclass(frozen=True)
class TrajectoryView:
  """
  Read-only view of one simulated trajectory inside a pool.

  Slices are 1D time series sharing the pool's ``t`` vector. Use
  ``frame_at(k)`` when you need a ``PendulumState`` for drawing or physics;
  prefer sin/cos columns when comparing to network training targets.

  Wrap ``GroundTruthSource`` (in ``sources.py``) for animation and gif export.
  """

  index: int
  stage: int
  split: str
  t: NDArray[np.float64]
  params: NDArray[np.float64]  # length-8 row
  theta1: NDArray[np.float64]
  theta2: NDArray[np.float64]
  sin_theta1: NDArray[np.float64]
  cos_theta1: NDArray[np.float64]
  sin_theta2: NDArray[np.float64]
  cos_theta2: NDArray[np.float64]
  omega1: NDArray[np.float64]
  omega2: NDArray[np.float64]
  kinetic: NDArray[np.float64]
  potential: NDArray[np.float64]
  energy: NDArray[np.float64]

  def initial_state(self) -> PendulumState:
    """IC state from params (angles may differ from wrapped series[0] by 2*pi)."""
    return PendulumState(
      float(self.params[0]),
      float(self.params[1]),
      float(self.params[2]),
      float(self.params[3]),
    )

  def frame_at(self, k: int) -> PendulumState:
    """Decoded state at time index ``k`` via atan2 on stored sin/cos."""
    return PendulumState(
      float(np.arctan2(self.sin_theta1[k], self.cos_theta1[k])),
      float(np.arctan2(self.sin_theta2[k], self.cos_theta2[k])),
      float(self.omega1[k]),
      float(self.omega2[k]),
    )

  def pendulum_params(self) -> PendulumParams:
    """Physical constants for this trajectory (l1 = l2 = params[6])."""
    ell = float(self.params[6])
    return PendulumParams(
      m1=float(self.params[4]),
      m2=float(self.params[5]),
      l1=ell,
      l2=ell,
      g=float(self.params[7]),
    )


@dataclass
class PoolData:
  """In-memory trajectory pool matching the ``.npz`` schema."""

  stage: int
  split: str
  frozen: bool
  t: NDArray[np.float64]
  params: NDArray[np.float64]
  theta1: NDArray[np.float64]
  theta2: NDArray[np.float64]
  sin_theta1: NDArray[np.float64]
  cos_theta1: NDArray[np.float64]
  sin_theta2: NDArray[np.float64]
  cos_theta2: NDArray[np.float64]
  omega1: NDArray[np.float64]
  omega2: NDArray[np.float64]
  kinetic: NDArray[np.float64]
  potential: NDArray[np.float64]
  energy: NDArray[np.float64]

  @property
  def n_traj(self) -> int:
    return int(self.params.shape[0])

  @property
  def n_t(self) -> int:
    return int(self.t.shape[0])

  def get_traj(self, i: int) -> TrajectoryView:
    if i < 0 or i >= self.n_traj:
      raise IndexError(f"traj index {i} out of range [0, {self.n_traj})")
    return TrajectoryView(
      index=i,
      stage=self.stage,
      split=self.split,
      t=self.t,
      params=self.params[i],
      theta1=self.theta1[i],
      theta2=self.theta2[i],
      sin_theta1=self.sin_theta1[i],
      cos_theta1=self.cos_theta1[i],
      sin_theta2=self.sin_theta2[i],
      cos_theta2=self.cos_theta2[i],
      omega1=self.omega1[i],
      omega2=self.omega2[i],
      kinetic=self.kinetic[i],
      potential=self.potential[i],
      energy=self.energy[i],
    )

  def pointwise(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expand to full-resolution pointwise (X, Y). Stride k is train-time only."""
    n, nt = self.n_traj, self.n_t
    if n == 0:
      return (
        np.zeros((0, 8), dtype=np.float64),
        np.zeros((0, 6), dtype=np.float64),
      )
    t_col = np.tile(self.t, n)
    # repeat each IC row once per time sample
    p = np.repeat(self.params, nt, axis=0)
    x = np.column_stack(
      [
        t_col,
        p[:, 0],
        p[:, 1],
        p[:, 2],
        p[:, 3],
        p[:, 4],
        p[:, 5],
        p[:, 7],
      ]
    )
    y = np.column_stack(
      [
        self.sin_theta1.reshape(-1),
        self.cos_theta1.reshape(-1),
        self.sin_theta2.reshape(-1),
        self.cos_theta2.reshape(-1),
        self.omega1.reshape(-1),
        self.omega2.reshape(-1),
      ]
    )
    return x, y

  def select(self, indices: NDArray[np.int64], *, split: str) -> PoolData:
    """Subset trajectories; keep shared time grid."""
    idx = np.asarray(indices, dtype=np.int64)
    return replace(
      self,
      split=split,
      frozen=False,
      t=self.t.copy(),
      params=self.params[idx].copy(),
      theta1=self.theta1[idx].copy(),
      theta2=self.theta2[idx].copy(),
      sin_theta1=self.sin_theta1[idx].copy(),
      cos_theta1=self.cos_theta1[idx].copy(),
      sin_theta2=self.sin_theta2[idx].copy(),
      cos_theta2=self.cos_theta2[idx].copy(),
      omega1=self.omega1[idx].copy(),
      omega2=self.omega2[idx].copy(),
      kinetic=self.kinetic[idx].copy(),
      potential=self.potential[idx].copy(),
      energy=self.energy[idx].copy(),
    )


def pool_path(data_root: Path | str, stage: int, split: str) -> Path:
  """Canonical path: ``data_root/stage{S}_{split}.npz``."""
  if split not in SPLITS:
    raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
  return Path(data_root) / f"stage{stage}_{split}.npz"


def list_pools(data_root: Path | str) -> list[tuple[int, str, Path]]:
  """
  Discover on-disk pools under ``data_root``.

  Returns ``(stage, split, path)`` tuples sorted by stage then split order
  train, then val, then test. Used by ``app_data.py`` and batch plot CLIs.
  """
  root = Path(data_root)
  found: list[tuple[int, str, Path]] = []
  if not root.is_dir():
    return found
  for split in SPLITS:
    for path in sorted(root.glob(f"stage*_{split}.npz")):
      stem = path.stem  # stage{S}_{split}
      prefix = f"_{split}"
      if not stem.endswith(prefix):
        continue
      stage_part = stem[: -len(prefix)]
      if not stage_part.startswith("stage"):
        continue
      try:
        stage = int(stage_part[len("stage") :])
      except ValueError:
        continue
      found.append((stage, split, path))
  found.sort(key=lambda x: (x[0], SPLITS.index(x[1])))
  return found


def _wrap_angles(theta: NDArray[np.float64]) -> NDArray[np.float64]:
  """Vectorized wrap to (-pi, pi] (same rule as ``wrap_angle``)."""
  return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _traj_to_wrapped_arrays(
  traj: Trajectory,
) -> dict[str, NDArray[np.float64]]:
  """Wrap angles, build sin/cos + omega + energy series for one traj."""
  th1 = _wrap_angles(traj.theta1)
  th2 = _wrap_angles(traj.theta2)
  return {
    "theta1": th1,
    "theta2": th2,
    "sin_theta1": np.sin(th1),
    "cos_theta1": np.cos(th1),
    "sin_theta2": np.sin(th2),
    "cos_theta2": np.cos(th2),
    "omega1": traj.omega1.astype(np.float64, copy=True),
    "omega2": traj.omega2.astype(np.float64, copy=True),
    "kinetic": traj.kinetic.astype(np.float64, copy=True),
    "potential": traj.potential.astype(np.float64, copy=True),
    "energy": traj.energy.astype(np.float64, copy=True),
  }


def _full_traj_ok(traj: Trajectory, cfg: dict) -> str | None:
  """Full-horizon safety (same checks as sampler probe, on full T)."""
  omega_traj_max = float(cfg["omega_traj_max"])
  energy_abs_tol = float(cfg["energy_abs_tol"])
  energy_rel_tol = float(cfg["energy_rel_tol"])
  energy_eps = float(cfg["energy_eps"])

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
  return None


def integrate_row(row: SampleRow, cfg: dict) -> Trajectory | None:
  """Full-T RK4 for one accepted IC row; None if full traj fails safety."""
  state = PendulumState(row.theta1, row.theta2, row.omega1, row.omega2)
  params = PendulumParams(row.m1, row.m2, row.l, row.l, row.g)
  traj = integrate_rk4(
    state,
    params,
    t_end=float(cfg["T"]),
    dt=float(cfg["dt"]),
  )
  reason = _full_traj_ok(traj, cfg)
  if reason is not None:
    logger.debug("full traj reject: %s", reason)
    return None
  return traj


def _empty_pool(stage: int, split: str, t: NDArray[np.float64]) -> PoolData:
  nt = t.shape[0]
  z = np.zeros((0, nt), dtype=np.float64)
  return PoolData(
    stage=stage,
    split=split,
    frozen=False,
    t=t.astype(np.float64, copy=True),
    params=np.zeros((0, 8), dtype=np.float64),
    theta1=z.copy(),
    theta2=z.copy(),
    sin_theta1=z.copy(),
    cos_theta1=z.copy(),
    sin_theta2=z.copy(),
    cos_theta2=z.copy(),
    omega1=z.copy(),
    omega2=z.copy(),
    kinetic=z.copy(),
    potential=z.copy(),
    energy=z.copy(),
  )


def build_pool(
  stage: int,
  n: int,
  cfg: dict,
  *,
  split: str = "train",
  rng: np.random.Generator | None = None,
) -> PoolData:
  """Sample ``n`` ICs, integrate full T, return a pool (not yet on disk)."""
  if n < 0:
    raise ValueError("n must be non-negative")
  if split not in SPLITS:
    raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

  t_end = float(cfg["T"])
  dt = float(cfg["dt"])
  n_steps = int(np.round(t_end / dt))
  t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1, dtype=np.float64)

  if n == 0:
    return _empty_pool(stage, split, t_grid)

  # Oversample a bit: full-T may reject rows the probe accepted
  max_batch_rounds = max(n * int(cfg.get("max_draw_attempts", 50)), n)
  gen = rng if rng is not None else np.random.default_rng(int(cfg.get("seed", 0)))

  params_list: list[NDArray[np.float64]] = []
  series: dict[str, list[NDArray[np.float64]]] = {k: [] for k in SERIES_KEYS}
  accepted = 0
  rounds = 0

  while accepted < n and rounds < max_batch_rounds:
    need = n - accepted
    result = sample(stage, need, cfg, rng=gen)
    rounds += need
    for row in result.rows:
      traj = integrate_row(row, cfg)
      if traj is None:
        continue
      if traj.t.shape != t_grid.shape or not np.allclose(traj.t, t_grid):
        raise RuntimeError("trajectory time grid mismatch vs sampler T/dt")
      packed = _traj_to_wrapped_arrays(traj)
      params_list.append(
        np.array(
          [
            row.theta1,
            row.theta2,
            row.omega1,
            row.omega2,
            row.m1,
            row.m2,
            row.l,
            row.g,
          ],
          dtype=np.float64,
        )
      )
      for key in SERIES_KEYS:
        series[key].append(packed[key])
      accepted += 1
      if accepted >= n:
        break

  if accepted < n:
    raise RuntimeError(
      f"stage {stage}: only integrated {accepted}/{n} full trajs "
      f"after {rounds} sample draws"
    )

  return PoolData(
    stage=stage,
    split=split,
    frozen=False,
    t=t_grid,
    params=np.stack(params_list, axis=0),
    theta1=np.stack(series["theta1"], axis=0),
    theta2=np.stack(series["theta2"], axis=0),
    sin_theta1=np.stack(series["sin_theta1"], axis=0),
    cos_theta1=np.stack(series["cos_theta1"], axis=0),
    sin_theta2=np.stack(series["sin_theta2"], axis=0),
    cos_theta2=np.stack(series["cos_theta2"], axis=0),
    omega1=np.stack(series["omega1"], axis=0),
    omega2=np.stack(series["omega2"], axis=0),
    kinetic=np.stack(series["kinetic"], axis=0),
    potential=np.stack(series["potential"], axis=0),
    energy=np.stack(series["energy"], axis=0),
  )


def carve_validation(
  train_pool: PoolData,
  val_fraction: float,
  rng: np.random.Generator,
) -> tuple[PoolData, PoolData]:
  """Split train pool into (train, val). Val never touches frozen test."""
  if not 0.0 <= val_fraction < 1.0:
    raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
  if train_pool.split != "train":
    raise ValueError("carve_validation expects a train pool")
  n = train_pool.n_traj
  if n == 0:
    empty_t = train_pool.t
    return (
      _empty_pool(train_pool.stage, "train", empty_t),
      _empty_pool(train_pool.stage, "val", empty_t),
    )
  n_val = int(np.floor(n * val_fraction))
  perm = rng.permutation(n)
  val_idx = perm[:n_val]
  train_idx = perm[n_val:]
  return (
    train_pool.select(train_idx, split="train"),
    train_pool.select(val_idx, split="val"),
  )


def save_pool(
  path: Path | str,
  pool: PoolData,
  *,
  overwrite_frozen: bool = False,
) -> Path:
  """Write ``.npz``. Blocks overwrite of frozen files unless forced."""
  path = Path(path)
  if path.exists():
    existing = np.load(path, allow_pickle=False)
    try:
      was_frozen = bool(existing["frozen"])
    finally:
      existing.close()
    if was_frozen and not overwrite_frozen:
      raise FrozenPoolError(
        f"refusing to overwrite frozen pool {path}; "
        "pass overwrite_frozen=True only if you mean it"
      )

  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    path,
    stage=np.int32(pool.stage),
    split=np.asarray(pool.split),
    frozen=np.bool_(pool.frozen),
    t=pool.t,
    params=pool.params,
    theta1=pool.theta1,
    theta2=pool.theta2,
    sin_theta1=pool.sin_theta1,
    cos_theta1=pool.cos_theta1,
    sin_theta2=pool.sin_theta2,
    cos_theta2=pool.cos_theta2,
    omega1=pool.omega1,
    omega2=pool.omega2,
    kinetic=pool.kinetic,
    potential=pool.potential,
    energy=pool.energy,
  )
  return path


def load_pool(path: Path | str) -> PoolData:
  """Read a pool ``.npz`` into ``PoolData``."""
  path = Path(path)
  with np.load(path, allow_pickle=False) as data:
    split_val = data["split"]
    split = str(split_val.item() if hasattr(split_val, "item") else split_val)
    return PoolData(
      stage=int(data["stage"]),
      split=split,
      frozen=bool(data["frozen"]),
      t=np.asarray(data["t"], dtype=np.float64),
      params=np.asarray(data["params"], dtype=np.float64),
      theta1=np.asarray(data["theta1"], dtype=np.float64),
      theta2=np.asarray(data["theta2"], dtype=np.float64),
      sin_theta1=np.asarray(data["sin_theta1"], dtype=np.float64),
      cos_theta1=np.asarray(data["cos_theta1"], dtype=np.float64),
      sin_theta2=np.asarray(data["sin_theta2"], dtype=np.float64),
      cos_theta2=np.asarray(data["cos_theta2"], dtype=np.float64),
      omega1=np.asarray(data["omega1"], dtype=np.float64),
      omega2=np.asarray(data["omega2"], dtype=np.float64),
      kinetic=np.asarray(data["kinetic"], dtype=np.float64),
      potential=np.asarray(data["potential"], dtype=np.float64),
      energy=np.asarray(data["energy"], dtype=np.float64),
    )


def open_pool(data_root: Path | str, stage: int, split: str) -> PoolData:
  """
  Load a single pool file into memory.

  Path pattern: ``{data_root}/stage{S}_{split}.npz``. Call ``get_traj(i)`` on
  the returned ``PoolData`` for visualization-friendly ``TrajectoryView`` rows.
  """
  return load_pool(pool_path(data_root, stage, split))


def freeze_pool(pool: PoolData) -> PoolData:
  """Mark pool frozen (typically test)."""
  return replace(pool, frozen=True)


def generate_stage(
  stage: int,
  data_root: Path | str,
  sampler_cfg: dict,
  *,
  train_n: int,
  test_n: int,
  val_fraction: float,
  rng: np.random.Generator,
  overwrite_frozen: bool = False,
) -> dict[str, Path]:
  """Build train/val/test for one stage and write ``.npz`` files.

  Test is frozen on disk. Val carved from the train pool.
  """
  root = Path(data_root)
  raw_train = build_pool(stage, train_n, sampler_cfg, split="train", rng=rng)
  train_pool, val_pool = carve_validation(raw_train, val_fraction, rng)
  test_pool = freeze_pool(
    build_pool(stage, test_n, sampler_cfg, split="test", rng=rng)
  )

  paths = {
    "train": save_pool(
      pool_path(root, stage, "train"),
      train_pool,
      overwrite_frozen=overwrite_frozen,
    ),
    "val": save_pool(
      pool_path(root, stage, "val"),
      val_pool,
      overwrite_frozen=overwrite_frozen,
    ),
    "test": save_pool(
      pool_path(root, stage, "test"),
      test_pool,
      overwrite_frozen=overwrite_frozen,
    ),
  }
  logger.info(
    "stage %s wrote train=%s val=%s test=%s (frozen)",
    stage,
    train_pool.n_traj,
    val_pool.n_traj,
    test_pool.n_traj,
  )
  return paths


def generate_all_stages(
  data_root: Path | str,
  sampler_cfg: dict,
  *,
  val_fraction: float,
  stages: list[int] | None = None,
  train_counts: list[int] | None = None,
  test_counts: list[int] | None = None,
  seed: int | None = None,
  overwrite_frozen: bool = False,
) -> dict[int, dict[str, Path]]:
  """Generate pools for stages 0..6 (or subset). Pool sizes from YAML by default."""
  from srcs.simulation.sampler import pool_index, stage_id_bounds

  lo, hi = stage_id_bounds(sampler_cfg)
  stage_list = stages if stages is not None else list(range(lo, hi + 1))
  n_pools = hi - lo + 1
  pools_cfg = sampler_cfg["pools"]
  train_ns = train_counts if train_counts is not None else list(pools_cfg["train"])
  test_ns = test_counts if test_counts is not None else list(pools_cfg["test"])
  if len(train_ns) < n_pools or len(test_ns) < n_pools:
    raise ValueError("train/test pool size lists must cover all stages")

  gen = np.random.default_rng(
    int(sampler_cfg.get("seed", 0)) if seed is None else seed
  )
  out: dict[int, dict[str, Path]] = {}
  for stage in stage_list:
    out[stage] = generate_stage(
      stage,
      data_root,
      sampler_cfg,
      train_n=int(train_ns[pool_index(stage, sampler_cfg)]),
      test_n=int(test_ns[pool_index(stage, sampler_cfg)]),
      val_fraction=val_fraction,
      rng=gen,
      overwrite_frozen=overwrite_frozen,
    )
  return out
