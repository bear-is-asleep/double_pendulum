"""Lagrangian double-pendulum dynamics and RK4 integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
    return np.array(
      [self.theta1, self.theta2, self.omega1, self.omega2],
      dtype=np.float64,
    )

  @classmethod
  def from_array(cls, y: NDArray[np.float64]) -> PendulumState:
    return cls(float(y[0]), float(y[1]), float(y[2]), float(y[3]))


def derivatives(y: NDArray[np.float64], p: PendulumParams) -> NDArray[np.float64]:
  """Return dy/dt for state y = [theta1, theta2, omega1, omega2]."""
  theta1, theta2, omega1, omega2 = y
  m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g

  delta = theta1 - theta2
  sin_delta = np.sin(delta)
  cos_delta = np.cos(delta)
  den = 2 * m1 + m2 - m2 * np.cos(2 * delta)

  domega1 = (
    -g * (2 * m1 + m2) * np.sin(theta1)
    - m2 * g * np.sin(theta1 - 2 * theta2)
    - 2 * sin_delta * m2 * (omega2**2 * l2 + omega1**2 * l1 * cos_delta)
  ) / (l1 * den)

  domega2 = (
    2
    * sin_delta
    * (
      omega1**2 * l1 * (m1 + m2)
      + g * (m1 + m2) * np.cos(theta1)
      + omega2**2 * l2 * m2 * cos_delta
    )
  ) / (l2 * den)

  return np.array([omega1, omega2, domega1, domega2], dtype=np.float64)


def rk4_step(
  y: NDArray[np.float64],
  p: PendulumParams,
  dt: float,
) -> NDArray[np.float64]:
  """Advance state by one RK4 step of size dt."""
  k1 = derivatives(y, p)
  k2 = derivatives(y + 0.5 * dt * k1, p)
  k3 = derivatives(y + 0.5 * dt * k2, p)
  k4 = derivatives(y + dt * k3, p)
  return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def wrap_angle(theta: float) -> float:
  """Wrap scalar angle to (-pi, pi] (project `angle_wrap: negpi_pi`)."""
  return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_state(state: PendulumState) -> PendulumState:
  """Return state with theta1, theta2 wrapped; omegas unchanged."""
  return PendulumState(
    wrap_angle(state.theta1),
    wrap_angle(state.theta2),
    state.omega1,
    state.omega2,
  )


def kinetic_energy(state: PendulumState, p: PendulumParams) -> float:
  """Kinetic energy T for the planar double pendulum (project.md formula)."""
  w1, w2 = state.omega1, state.omega2
  m1, m2, l1, l2 = p.m1, p.m2, p.l1, p.l2
  cos_d = np.cos(state.theta1 - state.theta2)
  return float(
    0.5 * (m1 + m2) * l1**2 * w1**2
    + 0.5 * m2 * l2**2 * w2**2
    + m2 * l1 * l2 * w1 * w2 * cos_d
  )


def potential_energy(state: PendulumState, p: PendulumParams) -> float:
  """Potential V with PE zero at hanging equilibrium (theta=0)."""
  th1, th2 = state.theta1, state.theta2
  m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g
  return float(
    (m1 + m2) * g * l1 * (1.0 - np.cos(th1))
    + m2 * g * l2 * (1.0 - np.cos(th2))
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

  n_steps = int(np.round((t_end - t0) / dt))
  t_end_aligned = t0 + n_steps * dt
  n_pts = n_steps + 1

  t = np.linspace(t0, t_end_aligned, n_pts, dtype=np.float64)
  theta1 = np.empty(n_pts, dtype=np.float64)
  theta2 = np.empty(n_pts, dtype=np.float64)
  omega1 = np.empty(n_pts, dtype=np.float64)
  omega2 = np.empty(n_pts, dtype=np.float64)
  kinetic = np.empty(n_pts, dtype=np.float64)
  potential = np.empty(n_pts, dtype=np.float64)
  energy = np.empty(n_pts, dtype=np.float64)

  y = state.as_array()
  for i in range(n_pts):
    s = PendulumState.from_array(y)
    theta1[i] = s.theta1
    theta2[i] = s.theta2
    omega1[i] = s.omega1
    omega2[i] = s.omega2
    kinetic[i] = kinetic_energy(s, params)
    potential[i] = potential_energy(s, params)
    energy[i] = kinetic[i] + potential[i]
    if i < n_steps:
      y = rk4_step(y, params, dt)

  return Trajectory(
    t=t,
    theta1=theta1,
    theta2=theta2,
    omega1=omega1,
    omega2=omega2,
    kinetic=kinetic,
    potential=potential,
    energy=energy,
  )


def cartesian(
  state: PendulumState,
  p: PendulumParams,
) -> tuple[float, float, float, float]:
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
    self.state = state or PendulumState(theta1=3 * np.pi / 4, theta2=np.pi / 2)
    self.dt = dt
    self.time = 0.0
    self.trail: list[tuple[float, float]] = []
    self.max_trail = 400

  def reset(self, state: PendulumState | None = None) -> None:
    """Reset time, trail, and optionally the state."""
    if state is not None:
      self.state = state
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
