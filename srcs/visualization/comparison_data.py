"""Shared arrays for stored vs surrogate time-series comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from srcs.simulation.data import TrajectoryView
from srcs.visualization.sources import GroundTruthSource

# Pool columns copied into chart series. Order matches PlaybackSeries fields.
_VIEW_COLUMNS = (
  "sin_theta1",
  "cos_theta1",
  "sin_theta2",
  "cos_theta2",
  "omega1",
  "omega2",
  "potential",
  "kinetic",
  "energy",
)


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


# Stored reference: theta1 cool, theta2 warm; energy purple / green.
CHART_REF_COLORS: dict[str, str] = {
  "sin_theta1": "#1d4ed8",
  "sin_theta2": "#dc2626",
  "omega1": "#2563eb",
  "omega2": "#ea580c",
  "potential": "#7c3aed",
  "kinetic": "#059669",
}

# Per loaded checkpoint (high contrast pairs for theta1 vs theta2).
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


def pool_series_arrays(view: TrajectoryView) -> dict[str, NDArray[np.float64]]:
  """Float64 time plus labeled pool columns. One copy path for every chart."""
  columns = {"t": np.asarray(view.t, dtype=np.float64)}
  for name in _VIEW_COLUMNS:
    columns[name] = np.asarray(getattr(view, name), dtype=np.float64)
  return columns


def reference_from_ground_truth(src: GroundTruthSource) -> ReferenceSeries:
  """Stored sin, omega, and PE/KE. Cos and total energy stay off this chart."""
  columns = pool_series_arrays(src.view)
  return ReferenceSeries(
    t=columns["t"],
    sin_theta1=columns["sin_theta1"],
    sin_theta2=columns["sin_theta2"],
    omega1=columns["omega1"],
    omega2=columns["omega2"],
    potential=columns["potential"],
    kinetic=columns["kinetic"],
  )
