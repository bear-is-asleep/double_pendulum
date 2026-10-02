"""Six-head training targets: sin/cos angles plus omega."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from srcs.physics.core import PendulumState


def decode_pred_row(row: NDArray[np.float64]) -> PendulumState:
  """One model row -> angles via atan2 and omega heads."""
  return PendulumState(
    theta1=float(np.arctan2(row[0], row[1])),
    theta2=float(np.arctan2(row[2], row[3])),
    omega1=float(row[4]),
    omega2=float(row[5]),
  )
