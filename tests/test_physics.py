"""Parity and sanity checks for double-pendulum physics."""

from __future__ import annotations

import math

import numpy as np

from double_pendulum.physics import (
  DoublePendulum,
  PendulumParams,
  PendulumState,
  cartesian,
  derivatives,
  integrate_rk4,
  kinetic_energy,
  potential_energy,
  rk4_step,
  total_energy,
  wrap_angle,
  wrap_state,
)


def test_hanging_equilibrium() -> None:
  p = PendulumParams()
  y = np.array([0.0, 0.0, 0.0, 0.0])
  dydt = derivatives(y, p)
  assert np.allclose(dydt, 0.0, atol=1e-12)


def test_cartesian_hanging() -> None:
  p = PendulumParams(l1=1.0, l2=1.5)
  s = PendulumState(0.0, 0.0)
  x1, y1, x2, y2 = cartesian(s, p)
  assert math.isclose(x1, 0.0, abs_tol=1e-12)
  assert math.isclose(y1, -1.0, abs_tol=1e-12)
  assert math.isclose(x2, 0.0, abs_tol=1e-12)
  assert math.isclose(y2, -2.5, abs_tol=1e-12)


def test_potential_zero_at_hanging() -> None:
  p = PendulumParams(m1=1.2, m2=0.8, l1=1.0, l2=0.9, g=9.81)
  s = PendulumState(0.0, 0.0, 0.5, -0.3)
  assert math.isclose(potential_energy(s, p), 0.0, abs_tol=1e-12)


def test_total_energy_is_T_plus_V() -> None:
  p = PendulumParams()
  s = PendulumState(0.7, -1.1, 1.5, -0.4)
  assert math.isclose(
    total_energy(s, p),
    kinetic_energy(s, p) + potential_energy(s, p),
    rel_tol=0.0,
    abs_tol=1e-12,
  )


def test_wrap_angle_negpi_pi() -> None:
  assert math.isclose(wrap_angle(0.0), 0.0, abs_tol=1e-12)
  assert math.isclose(wrap_angle(3.5 * np.pi), wrap_angle(-0.5 * np.pi), abs_tol=1e-10)
  wrapped = wrap_state(PendulumState(4.0 * np.pi, -3.2 * np.pi, 1.0, 2.0))
  assert -np.pi < wrapped.theta1 <= np.pi
  assert -np.pi < wrapped.theta2 <= np.pi
  assert wrapped.omega1 == 1.0 and wrapped.omega2 == 2.0


def test_integrate_rk4_trajectory() -> None:
  p = PendulumParams()
  s = PendulumState(theta1=1.0, theta2=0.5, omega1=0.2, omega2=-0.1)
  traj = integrate_rk4(s, p, t_end=1.0, dt=0.01)
  assert traj.t.shape == traj.energy.shape == (101,)
  assert np.allclose(traj.energy, traj.kinetic + traj.potential, rtol=0, atol=1e-12)
  assert math.isclose(float(traj.t[0]), 0.0, abs_tol=1e-12)
  assert math.isclose(float(traj.t[-1]), 1.0, abs_tol=1e-9)


def test_energy_nearly_conserved() -> None:
  p = PendulumParams()
  s = PendulumState(theta1=2.0, theta2=1.0, omega1=0.3, omega2=-0.2)
  traj = integrate_rk4(s, p, t_end=10.0, dt=1 / 240)
  e0 = float(traj.energy[0])
  e1 = float(traj.energy[-1])
  assert abs(e1 - e0) / max(abs(e0), 1e-9) < 5e-3


def test_energy_conserved_zero_gravity_smoke() -> None:
  p = PendulumParams(g=0.0)
  s = PendulumState(theta1=0.5, theta2=-0.3, omega1=1.0, omega2=0.5)
  traj = integrate_rk4(s, p, t_end=2.0, dt=1 / 120)
  assert np.nanmax(np.abs(traj.energy - traj.energy[0])) < 1e-6


def test_step_advances_time_and_trail() -> None:
  sim = DoublePendulum(dt=1 / 120)
  sim.step(10)
  assert math.isclose(sim.time, 10 / 120, rel_tol=1e-12)
  assert len(sim.trail) == 1
  sim.reset()
  sim.step(5, record_trail=False)
  assert sim.trail == []


if __name__ == "__main__":
  test_hanging_equilibrium()
  test_cartesian_hanging()
  test_potential_zero_at_hanging()
  test_total_energy_is_T_plus_V()
  test_wrap_angle_negpi_pi()
  test_integrate_rk4_trajectory()
  test_energy_nearly_conserved()
  test_energy_conserved_zero_gravity_smoke()
  test_step_advances_time_and_trail()
  print("All physics checks passed.")
