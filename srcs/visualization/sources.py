"""Trajectory providers for visualization and batch export.

Apps and ``plots.py`` should not read ``.npz`` files directly in the UI layer.
Instead they use a small protocol so the same animator can show:

* ground truth from dataset pools (today), and
* neural-network rollouts via ``SurrogateSource`` in ``surrogate.py``.

See ``docs/visualization.md`` for how the pieces connect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from srcs.simulation.data import TrajectoryView
from srcs.physics.core import PendulumParams, PendulumState, cartesian, integrate_rk4

__all__ = [
  "GroundTruthSource",
  "TrajectorySource",
]


def _max_abs_energy_drift(energy: NDArray[np.float64]) -> float:
  """Largest |E(t) - E(0)|. Empty series is zero drift."""
  if energy.size == 0:
    return 0.0
  return float(np.max(np.abs(energy - energy[0])))


@runtime_checkable
class TrajectorySource(Protocol):
  """
  Time-discrete pendulum motion on a fixed grid.

  Frame index ``k`` runs from ``0`` to ``n_frames() - 1``. All sources share the
  same calling convention so ``build_svg``, gif export, and surrogate
  overlays can treat them interchangeably.
  """

  def n_frames(self) -> int:
    """Number of time samples (includes t = 0)."""
    ...

  def time_at(self, k: int) -> float:
    """Physical time in seconds at index ``k``."""
    ...

  def params(self) -> PendulumParams:
    """Masses, lengths, and gravity for this trajectory (constant in time)."""
    ...

  def frame_state(self, k: int) -> PendulumState:
    """Angles and angular velocities at index ``k``."""
    ...

  def tip_trail(self, k: int) -> list[tuple[float, float]]:
    """Lower-bob (x2, y2) positions from frame 0 through ``k`` inclusive."""
    ...


@dataclass
class GroundTruthSource:
  """
  Ground truth from a saved pool trajectory.

  **stored** (default)
      Use sin/cos and omega arrays from disk. Angles are recovered with ``atan2``
      so they stay consistent with training targets (wrapped circle).

  **reintegrate**
      Ignore stored angles/omega after the IC: run ``integrate_rk4`` from
      ``view.initial_state()`` with the same ``t`` spacing as the pool. Useful
      to confirm the simulator matches what was written during data generation.
  """

  view: TrajectoryView
  mode: Literal["stored", "reintegrate"] = "stored"
  _reint_states: list[PendulumState] | None = None
  # Full tip path, filled on first trail request. View is fixed after init.
  _tips: list[tuple[float, float]] | None = None

  def __post_init__(self) -> None:
    if self.mode == "reintegrate":
      # Precompute the whole series once; playback only indexes into it.
      self._reint_states = self._integrate_series()

  def n_frames(self) -> int:
    return int(self.view.t.shape[0])

  def time_at(self, k: int) -> float:
    return float(self.view.t[k])

  def params(self) -> PendulumParams:
    return self.view.pendulum_params()

  def frame_state(self, k: int) -> PendulumState:
    if self.mode == "stored":
      return self.view.frame_at(k)
    assert self._reint_states is not None
    return self._reint_states[k]

  def tip_trail(self, k: int) -> list[tuple[float, float]]:
    return self._all_tips()[: k + 1]

  def _all_tips(self) -> list[tuple[float, float]]:
    """Lower-bob (x2, y2) at every frame. One cartesian pass, then slice."""
    if self._tips is not None:
      return self._tips
    p = self.params()
    tips: list[tuple[float, float]] = []
    for i in range(self.n_frames()):
      _, _, x2, y2 = cartesian(self.frame_state(i), p)
      tips.append((x2, y2))
    self._tips = tips
    return tips

  def energy_drift_max(self) -> float:
    """Largest absolute deviation of total energy from its initial value on disk."""
    return _max_abs_energy_drift(self.view.energy)

  def ic_meta(self) -> dict[str, float]:
    """
    Human-readable IC and physics scalars for UI labels.

    ``params`` column order matches ``data.PARAM_COLS``: theta10, theta20,
    omega10, omega20, m1, m2, l, g.
    """
    row = self.view.params
    return {
      "theta10": float(row[0]),
      "theta20": float(row[1]),
      "omega10": float(row[2]),
      "omega20": float(row[3]),
      "m1": float(row[4]),
      "m2": float(row[5]),
      "l": float(row[6]),
      "g": float(row[7]),
      "energy_drift_max": self.energy_drift_max(),
    }

  def _integrate_series(self) -> list[PendulumState]:
    p = self.params()
    ic = self.view.initial_state()
    if self.view.t.size < 2:
      return [ic]
    dt = float(self.view.t[1] - self.view.t[0])
    traj = integrate_rk4(ic, p, t_end=float(self.view.t[-1]), dt=dt, t0=float(self.view.t[0]))
    return [
      PendulumState(
        float(traj.theta1[i]),
        float(traj.theta2[i]),
        float(traj.omega1[i]),
        float(traj.omega2[i]),
      )
      for i in range(traj.t.shape[0])
    ]
