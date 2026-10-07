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


# Matches ``plots.plot_trajectory_timeseries`` (theta1 steel, theta2 accent, PE purple).
_CHART_FIELDS = (
  "sin_theta1",
  "sin_theta2",
  "omega1",
  "omega2",
  "potential",
  "kinetic",
)

CHART_REF_COLORS: dict[str, str] = {
  "sin_theta1": "#2f6f8f",
  "sin_theta2": "#c45c26",
  "omega1": "#2f6f8f",
  "omega2": "#c45c26",
  "potential": "#6b4c9a",
  "kinetic": "#2f6f8f",
}

# First channel in each subplot row solid; second dashed (PE vs KE, etc.).
CHART_FIELD_DASH: dict[str, str] = {
  "sin_theta1": "solid",
  "sin_theta2": "dash",
  "omega1": "solid",
  "omega2": "dash",
  "potential": "solid",
  "kinetic": "dash",
}

# Pendulum + chart: stored sim gray, all checkpoints share NN purple.
LAYER_COLOR_STORED = "#6b757d"
LAYER_COLOR_NN = "#6b4c9a"
CHART_NN_COLOR = LAYER_COLOR_NN


def nn_trace_colors() -> dict[str, str]:
  return {name: CHART_NN_COLOR for name in _CHART_FIELDS}


def layer_swatch_html(color: str, *, dashed: bool, title: str) -> str:
  """Sidebar chip: solid fill (stored) or dashed stroke segment (NN overlay)."""
  if dashed:
    return (
      f'<svg class="layer-swatch" width="14" height="14" viewBox="0 0 14 14" '
      f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{title}">'
      f'<line x1="1" y1="7" x2="13" y2="7" stroke="{color}" stroke-width="3" '
      f'stroke-linecap="round" stroke-dasharray="3 2"/></svg>'
    )
  border = "rgba(26, 42, 58, 0.22)"
  return (
    f'<span class="layer-swatch" title="{title}" '
    f'style="background:{color}; border:1px solid {border};"></span>'
  )


# Back-compat alias for tests that import CHART_LAYER_COLOR_SETS[0].
CHART_LAYER_COLOR_SETS: list[dict[str, str]] = [nn_trace_colors()]


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
