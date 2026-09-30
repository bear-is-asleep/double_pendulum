"""Trajectory providers for visualization and batch export.

Apps and ``plots.py`` should not read ``.npz`` files directly in the UI layer.
Instead they use a small protocol so the same animator can show:

* ground truth from dataset pools (today), and
* neural-network rollouts (once checkpoint inference exists).

See ``docs/visualization.md`` for how the pieces connect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from double_pendulum.data import TrajectoryView
from double_pendulum.physics import PendulumParams, PendulumState, cartesian, integrate_rk4


@runtime_checkable
class TrajectorySource(Protocol):
  """
  Time-discrete pendulum motion on a fixed grid.

  Frame index ``k`` runs from ``0`` to ``n_frames() - 1``. All sources share the
  same calling convention so ``build_svg``, gif export, and future surrogate
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
    p = self.params()
    out: list[tuple[float, float]] = []
    for i in range(k + 1):
      _, _, x2, y2 = cartesian(self.frame_state(i), p)
      out.append((x2, y2))
    return out

  def energy_drift_max(self) -> float:
    """Largest absolute deviation of total energy from its initial value on disk."""
    e = self.view.energy
    if e.size == 0:
      return 0.0
    return float(np.max(np.abs(e - e[0])))

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


@dataclass
class SurrogateSource:
  """
  Placeholder for MLP trajectory predictions.

  The dataset app exposes UI hooks (overlay switch, ``run_id`` field) but keeps
  them disabled until ``is_available()`` is true. A future checkpoint loader will
  set ``run_id``, flip ``enabled``, and implement ``predict_series``.
  """

  run_id: str | None = None
  enabled: bool = False

  def is_available(self) -> bool:
    return bool(self.enabled and self.run_id)

  def predict_series(
    self,
    t: NDArray[np.float64],
    ic: PendulumState,
    params: PendulumParams,
  ) -> NDArray[np.float64]:
    """
    Predict the 6-D training target at each time in ``t``.

    Returns an array of shape ``(len(t), 6)`` with columns::

      sin(theta1), cos(theta1), sin(theta2), cos(theta2), omega1, omega2

    Decode scalar angles with ``atan2(sin, cos)`` before cartesian drawing.
    Lengths are fixed in this project and are not model inputs.
    """
    raise NotImplementedError(
      "Surrogate inference is not wired yet. "
      f"run_id={self.run_id!r}; implement predict_series when the checkpoint loader exists."
    )
