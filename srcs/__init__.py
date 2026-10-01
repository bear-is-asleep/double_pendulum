"""Double pendulum: physics, simulation, models, training, visualization."""

from srcs.physics import (
  DoublePendulum,
  PendulumParams,
  PendulumState,
  Trajectory,
  integrate_rk4,
  kinetic_energy,
  potential_energy,
  total_energy,
  wrap_angle,
  wrap_state,
)

__all__ = [
  "DoublePendulum",
  "PendulumParams",
  "PendulumState",
  "Trajectory",
  "integrate_rk4",
  "kinetic_energy",
  "potential_energy",
  "total_energy",
  "wrap_angle",
  "wrap_state",
]
