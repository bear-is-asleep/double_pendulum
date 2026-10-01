"""Shared arrays for stored vs surrogate time-series comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from srcs.visualization.sources import GroundTruthSource


@dataclass(frozen=True)
class ReferenceSeries:
  t: NDArray[np.float64]
  sin_theta1: NDArray[np.float64]
  sin_theta2: NDArray[np.float64]
  omega1: NDArray[np.float64]
  omega2: NDArray[np.float64]
  potential: NDArray[np.float64]
  kinetic: NDArray[np.float64]


@dataclass(frozen=True)
class LayerSeries:
  layer_id: str
  label: str
  sin_theta1: NDArray[np.float64]
  sin_theta2: NDArray[np.float64]
  omega1: NDArray[np.float64]
  omega2: NDArray[np.float64]
  potential: NDArray[np.float64]
  kinetic: NDArray[np.float64]
  trace_colors: dict[str, str]

  def color_for(self, field: str) -> str:
    return self.trace_colors[field]


# Stored reference: θ1 cool, θ2 warm; energy purple / green.
CHART_REF_COLORS: dict[str, str] = {
  "sin_theta1": "#1d4ed8",
  "sin_theta2": "#dc2626",
  "omega1": "#2563eb",
  "omega2": "#ea580c",
  "potential": "#7c3aed",
  "kinetic": "#059669",
}

# Per loaded checkpoint (high contrast pairs for θ1 vs θ2).
CHART_LAYER_COLOR_SETS: list[dict[str, str]] = [
  {
    "sin_theta1": "#c2410c",
    "sin_theta2": "#0891b2",
    "omega1": "#c2410c",
    "omega2": "#0891b2",
    "potential": "#a855f7",
    "kinetic": "#10b981",
  },
  {
    "sin_theta1": "#15803d",
    "sin_theta2": "#be185d",
    "omega1": "#15803d",
    "omega2": "#be185d",
    "potential": "#6366f1",
    "kinetic": "#ca8a04",
  },
  {
    "sin_theta1": "#4338ca",
    "sin_theta2": "#b45309",
    "omega1": "#4338ca",
    "omega2": "#b45309",
    "potential": "#0d9488",
    "kinetic": "#db2777",
  },
  {
    "sin_theta1": "#0f766e",
    "sin_theta2": "#7e22ce",
    "omega1": "#0f766e",
    "omega2": "#7e22ce",
    "potential": "#b91c1c",
    "kinetic": "#1d4ed8",
  },
]


def reference_from_ground_truth(src: GroundTruthSource) -> ReferenceSeries:
  v = src.view
  t = np.asarray(v.t, dtype=np.float64)
  return ReferenceSeries(
    t=t,
    sin_theta1=np.asarray(v.sin_theta1, dtype=np.float64),
    sin_theta2=np.asarray(v.sin_theta2, dtype=np.float64),
    omega1=np.asarray(v.omega1, dtype=np.float64),
    omega2=np.asarray(v.omega2, dtype=np.float64),
    potential=np.asarray(v.potential, dtype=np.float64),
    kinetic=np.asarray(v.kinetic, dtype=np.float64),
  )
