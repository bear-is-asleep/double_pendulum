"""Planar double pendulum dynamics (RK4, energy, wrapping)."""

from srcs.physics.core import (
  DoublePendulum,
  PendulumParams,
  PendulumState,
  Trajectory,
  _eom_numerators,
  cartesian,
  derivatives,
  eom_denominator,
  integrate_rk4,
  kinetic_energy,
  potential_energy,
  rk4_step,
  total_energy,
  wrap_angle,
  wrap_state,
)

__all__ = [
  "DoublePendulum",
  "PendulumParams",
  "PendulumState",
  "Trajectory",
  "_eom_numerators",
  "cartesian",
  "derivatives",
  "eom_denominator",
  "integrate_rk4",
  "kinetic_energy",
  "potential_energy",
  "rk4_step",
  "total_energy",
  "wrap_angle",
  "wrap_state",
]
