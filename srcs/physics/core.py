"""Lagrangian double-pendulum dynamics and RK4 integration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_F = np.float64
# Collinearity / rank-loss guards for coupled EOM solve.
_EOM_DEN_MIN = 1.0e-6
_EOM_NUM_EPS = 1.0e-12


@dataclass(frozen=True)
class PendulumParams:
  """Physical constants for a planar double pendulum."""

  m1: float = 1.0
  m2: float = 1.0
  l1: float = 1.0
  l2: float = 1.0
  g: float = 9.81


@dataclass
class PendulumState:
  """Angles (rad) and angular velocities (rad/s)."""

  theta1: float
  theta2: float
  omega1: float = 0.0
  omega2: float = 0.0

  def as_array(self) -> NDArray[np.float64]:
    return np.array([self.theta1, self.theta2, self.omega1, self.omega2], dtype=_F)

  @classmethod
  def from_array(cls, y: NDArray[np.float64]) -> PendulumState:
    return cls(float(y[0]), float(y[1]), float(y[2]), float(y[3]))


def _prepare_integrator_state(y: NDArray[np.float64], p: PendulumParams) -> NDArray[np.float64]:
  """One-time IC when m1=0: shared omega, fixed link angle (not per RK4 step)."""
  if p.m1 != 0.0 or p.m2 <= 0.0:
    return y
  alpha = float(y[0] - y[1])
  w = 0.5 * (float(y[2]) + float(y[3]))
  return np.array([float(y[1]) + alpha, float(y[1]), w, w], dtype=_F)


def eom_denominator(theta1: float, theta2: float, m1: float, m2: float) -> float:
  """Mass-matrix determinant factor (sampler guard); 0 when m1=0 and rods align."""
  d = theta1 - theta2
  return float(2.0 * m1 + m2 - m2 * np.cos(2.0 * d))


def _eom_numerators(
  theta1: float,
  theta2: float,
  omega1: float,
  omega2: float,
  m1: float,
  m2: float,
  l1: float,
  l2: float,
  g: float,
) -> tuple[float, float, float]:
  """Coupled EL numerators and shared denominator (equivalent to M qdd = b)."""
  d = theta1 - theta2
  sd, cd = np.sin(d), np.cos(d)
  den = eom_denominator(theta1, theta2, m1, m2)
  num1 = (
    -g * (2 * m1 + m2) * np.sin(theta1)
    - m2 * g * np.sin(theta1 - 2 * theta2)
    - 2 * sd * m2 * (omega2**2 * l2 + omega1**2 * l1 * cd)
  )
  num2 = 2 * sd * (
    omega1**2 * l1 * (m1 + m2)
    + g * (m1 + m2) * np.cos(theta1)
    + omega2**2 * l2 * m2 * cd
  )
  return float(num1), float(num2), float(den)


def _coupled_angular_accelerations(
  theta1: float,
  theta2: float,
  omega1: float,
  omega2: float,
  p: PendulumParams,
) -> tuple[float, float]:
  """Solve M qdd = b with b = M@[n1/l1,n2/l2]/den (avoids divide-then-multiply)."""
  m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g
  cd = np.cos(theta1 - theta2)
  cross = m2 * l1 * l2 * cd
  inertia = np.array([[(m1 + m2) * l1**2, cross], [cross, m2 * l2**2]], dtype=_F)
  num1, num2, den = _eom_numerators(theta1, theta2, omega1, omega2, m1, m2, l1, l2, g)
  # Rest equilibrium on collinearity: removable 0/0 -> zero acceleration.
  if abs(den) < _EOM_DEN_MIN and abs(num1) < _EOM_NUM_EPS and abs(num2) < _EOM_NUM_EPS:
    return 0.0, 0.0
  scaled = inertia @ np.array([num1 / l1, num2 / l2], dtype=_F)
  with np.errstate(divide="ignore", invalid="ignore"):
    forcing = scaled / den
  if not np.all(np.isfinite(forcing)) or np.allclose(forcing, 0.0, atol=_EOM_NUM_EPS):
    return 0.0, 0.0
  if abs(float(np.linalg.det(inertia))) < _EOM_DEN_MIN:
    if abs(num1) >= _EOM_NUM_EPS or abs(num2) >= _EOM_NUM_EPS:
      warnings.warn(
        "double-pendulum near collinear: inertia singular; using least-squares accelerations",
        RuntimeWarning,
        stacklevel=2,
      )
    qdd, *_ = np.linalg.lstsq(inertia, forcing, rcond=None)
  else:
    qdd = np.linalg.solve(inertia, forcing)
  return float(qdd[0]), float(qdd[1])


def derivatives(y: NDArray[np.float64], p: PendulumParams) -> NDArray[np.float64]:
  """Return dy/dt for state y = [theta1, theta2, omega1, omega2]."""
  theta1, theta2, omega1, omega2 = y
  m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g
  if m1 == 0.0 and m2 == 0.0:
    return np.array([omega1, omega2, 0.0, 0.0], dtype=_F)
  if m1 == 0.0:
    # Compound pendulum: dtheta1/dt = dtheta2/dt preserves link angle after IC prep.
    alpha = float(theta1 - theta2)
    w = float(omega1)
    inertia = l1**2 + l2**2 + 2.0 * l1 * l2 * np.cos(alpha)
    dd = 0.0 if inertia <= 0.0 else -g * (l1 * np.sin(theta2 + alpha) + l2 * np.sin(theta2)) / inertia
    return np.array([w, w, dd, dd], dtype=_F)
  if m2 == 0.0:
    # Massless second bob: simple pendulum on theta1 only.
    return np.array([omega1, omega2, -g * np.sin(theta1) / l1, 0.0], dtype=_F)
  d1, d2 = _coupled_angular_accelerations(float(theta1), float(theta2), float(omega1), float(omega2), p)
  return np.array([omega1, omega2, d1, d2], dtype=_F)


def rk4_step(y: NDArray[np.float64], p: PendulumParams, dt: float) -> NDArray[np.float64]:
  """Advance state by one RK4 step of size dt."""
  k1 = derivatives(y, p)
  k2 = derivatives(y + 0.5 * dt * k1, p)
  k3 = derivatives(y + 0.5 * dt * k2, p)
  k4 = derivatives(y + dt * k3, p)
  return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap_angle(theta):
  """Wrap scalar angle to (-pi, pi] (project `angle_wrap: negpi_pi`)."""
  return (theta + np.pi) % (2.0 * np.pi) - np.pi

def wrap_state(state: PendulumState) -> PendulumState:
  """Return state with theta1, theta2 wrapped; omegas unchanged."""
  return PendulumState(
    wrap_angle(state.theta1),
    wrap_angle(state.theta2),
    state.omega1,
    state.omega2,
  )


def kinetic_energy(state: PendulumState, p: PendulumParams) -> float:
  """Kinetic energy T (project.md Lagrangian)."""
  w1, w2 = state.omega1, state.omega2
  cd = np.cos(state.theta1 - state.theta2)
  return float(
    0.5 * (p.m1 + p.m2) * p.l1**2 * w1**2
    + 0.5 * p.m2 * p.l2**2 * w2**2
    + p.m2 * p.l1 * p.l2 * w1 * w2 * cd
  )


def potential_energy(state: PendulumState, p: PendulumParams) -> float:
  """Potential V with PE zero at hanging equilibrium (theta=0)."""
  return float(
    (p.m1 + p.m2) * p.g * p.l1 * (1.0 - np.cos(state.theta1))
    + p.m2 * p.g * p.l2 * (1.0 - np.cos(state.theta2))
  )


def total_energy(state: PendulumState, p: PendulumParams) -> float:
  """Total mechanical energy E = T + V."""
  return kinetic_energy(state, p) + potential_energy(state, p)


@dataclass(frozen=True)
class Trajectory:
  """RK4 integration output aligned with sampler / dataset needs."""

  t: NDArray[np.float64]
  theta1: NDArray[np.float64]
  theta2: NDArray[np.float64]
  omega1: NDArray[np.float64]
  omega2: NDArray[np.float64]
  kinetic: NDArray[np.float64]
  potential: NDArray[np.float64]
  energy: NDArray[np.float64]


def integrate_rk4(
  state: PendulumState,
  params: PendulumParams,
  *,
  t_end: float,
  dt: float,
  t0: float = 0.0,
) -> Trajectory:
  """Integrate with fixed-step RK4 from t0 to t_end (inclusive of endpoints)."""
  if dt <= 0:
    raise ValueError("dt must be positive")
  if t_end < t0:
    raise ValueError("t_end must be >= t0")

  y = _prepare_integrator_state(state.as_array(), params)
  n_steps = int(np.round((t_end - t0) / dt))
  n_pts = n_steps + 1
  t = np.linspace(t0, t0 + n_steps * dt, n_pts, dtype=_F)
  theta1 = np.empty(n_pts, dtype=_F)
  theta2 = np.empty(n_pts, dtype=_F)
  omega1 = np.empty(n_pts, dtype=_F)
  omega2 = np.empty(n_pts, dtype=_F)
  kinetic = np.empty(n_pts, dtype=_F)
  potential = np.empty(n_pts, dtype=_F)
  energy = np.empty(n_pts, dtype=_F)

  for i in range(n_pts):
    s = PendulumState.from_array(y)
    theta1[i], theta2[i], omega1[i], omega2[i] = y[0], y[1], y[2], y[3]
    kinetic[i] = kinetic_energy(s, params)
    potential[i] = potential_energy(s, params)
    energy[i] = kinetic[i] + potential[i]
    if i < n_steps:
      y = rk4_step(y, params, dt)

  return Trajectory(t, theta1, theta2, omega1, omega2, kinetic, potential, energy)


def cartesian(state: PendulumState, p: PendulumParams) -> tuple[float, float, float, float]:
  """Return (x1, y1, x2, y2) with pivot at origin and +y upward."""
  x1 = p.l1 * np.sin(state.theta1)
  y1 = -p.l1 * np.cos(state.theta1)
  x2 = x1 + p.l2 * np.sin(state.theta2)
  y2 = y1 - p.l2 * np.cos(state.theta2)
  return float(x1), float(y1), float(x2), float(y2)


class DoublePendulum:
  """Mutable double-pendulum simulator with fixed-step RK4."""

  def __init__(
    self,
    params: PendulumParams | None = None,
    state: PendulumState | None = None,
    dt: float = 1 / 120,
  ) -> None:
    self.params = params or PendulumParams()
    self.dt = dt
    self.time = 0.0
    self.trail: list[tuple[float, float]] = []
    self.max_trail = 400
    self.state = state or PendulumState(theta1=3 * np.pi / 4, theta2=np.pi / 2)
    self.state = PendulumState.from_array(_prepare_integrator_state(self.state.as_array(), self.params))

  def reset(self, state: PendulumState | None = None) -> None:
    """Reset time, trail, and optionally the state."""
    if state is not None:
      self.state = state
    self.state = PendulumState.from_array(_prepare_integrator_state(self.state.as_array(), self.params))
    self.time = 0.0
    self.trail.clear()

  def step(self, n: int = 1, record_trail: bool = True) -> PendulumState:
    """Integrate n RK4 steps; optionally append the tip to the trail."""
    y = self.state.as_array()
    for _ in range(n):
      y = rk4_step(y, self.params, self.dt)
      self.time += self.dt
    self.state = PendulumState.from_array(y)
    if record_trail:
      _, _, x2, y2 = cartesian(self.state, self.params)
      self.trail.append((x2, y2))
      if len(self.trail) > self.max_trail:
        del self.trail[: len(self.trail) - self.max_trail]
    return self.state

  def positions(self) -> tuple[float, float, float, float]:
    """Current bob positions in cartesian coordinates."""
    return cartesian(self.state, self.params)
