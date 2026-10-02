"""Output channel names shared by test-pool metrics and figure exports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

CHANNEL_KEYS = (
  "sin_theta1",
  "sin_theta2",
  "omega1",
  "omega2",
  "potential",
  "kinetic",
)

# Subplot row labels for channel MAE figure (keys must be in CHANNEL_KEYS).
CHANNEL_MAE_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
  ("sin theta", ("sin_theta1", "sin_theta2")),
  ("omega", ("omega1", "omega2")),
  ("energy", ("potential", "kinetic")),
)


def per_step_channel_abs_errors(
  view,
  pred: NDArray[np.float64],
  pred_pe: NDArray[np.float64],
  pred_ke: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
  """Mean |error| accumulators per channel; one vector per time index."""
  return {
    "sin_theta1": np.abs(view.sin_theta1 - pred[:, 0]),
    "sin_theta2": np.abs(view.sin_theta2 - pred[:, 2]),
    "omega1": np.abs(view.omega1 - pred[:, 4]),
    "omega2": np.abs(view.omega2 - pred[:, 5]),
    "potential": np.abs(view.potential - pred_pe),
    "kinetic": np.abs(view.kinetic - pred_ke),
  }
