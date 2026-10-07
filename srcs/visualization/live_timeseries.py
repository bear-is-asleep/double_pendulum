"""Live time-series charts synced with dataset playback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from srcs.visualization.comparison_data import (
  CHART_FIELD_DASH,
  CHART_REF_COLORS,
  LayerSeries,
  ReferenceSeries,
  pool_series_arrays,
)
from srcs.visualization.layer_registry import PoolComparisonContext
from srcs.visualization.sources import GroundTruthSource
from srcs.visualization.surrogate import STORED_LAYER_ID


@dataclass(frozen=True)
class PlaybackSeries:
  """Aligned arrays for sin/cos, omega, and energy plots (GT-only legacy)."""

  t: NDArray[np.float64]
  sin_theta1: NDArray[np.float64]
  cos_theta1: NDArray[np.float64]
  sin_theta2: NDArray[np.float64]
  cos_theta2: NDArray[np.float64]
  omega1: NDArray[np.float64]
  omega2: NDArray[np.float64]
  potential: NDArray[np.float64]
  kinetic: NDArray[np.float64]
  energy: NDArray[np.float64]


def series_from_ground_truth(src: GroundTruthSource) -> PlaybackSeries:
  """Build plot arrays from stored pool labels."""
  return PlaybackSeries(**pool_series_arrays(src.view))


def _ylim_pad(y: NDArray[np.float64], pad_frac: float = 0.05) -> tuple[float, float]:
  ymin, ymax = float(np.min(y)), float(np.max(y))
  if ymin == ymax:
    ymin -= 1.0
    ymax += 1.0
  pad = pad_frac * (ymax - ymin)
  return ymin - pad, ymax + pad


DisplayMode = Literal["values", "errors"]


@dataclass
class ComparisonChartState:
  ref: ReferenceSeries
  layers: list[LayerSeries]
  visible: dict[str, bool] = field(default_factory=dict)
  display_mode: DisplayMode = "values"


@dataclass(frozen=True)
class ChartTrace:
  """One Plotly line. ``y`` is the full series; playback slices it."""

  row: int
  name: str
  layer_key: str
  y: NDArray[np.float64]
  color: str
  dash: str
  visible: bool


# (subplot row, series attribute, legend stem)
_PANELS: tuple[tuple[int, str, str], ...] = (
  (1, "sin_theta1", "sin theta1"),
  (1, "sin_theta2", "sin theta2"),
  (2, "omega1", "omega1"),
  (2, "omega2", "omega2"),
  (3, "potential", "PE"),
  (3, "kinetic", "KE"),
)

_ROW_FIELDS: dict[int, tuple[str, str]] = {
  1: ("sin_theta1", "sin_theta2"),
  2: ("omega1", "omega2"),
  3: ("potential", "kinetic"),
}


def _field_y(
  state: ComparisonChartState,
  field: str,
  layer: LayerSeries | None,
) -> NDArray[np.float64]:
  """Stored series, model series, or stored minus model when mode is errors."""
  ref_y = getattr(state.ref, field)
  if layer is None:
    return ref_y
  pred_y = getattr(layer, field)
  if state.display_mode == "errors":
    return ref_y - pred_y
  return pred_y


def chart_traces(state: ComparisonChartState) -> list[ChartTrace]:
  """Trace order shared by figure build and frame slicing."""
  show_ref = state.visible.get(STORED_LAYER_ID, True)
  values_mode = state.display_mode == "values"
  specs: list[ChartTrace] = []
  for row, field, label in _PANELS:
    if values_mode and show_ref:
      specs.append(ChartTrace(
        row=row,
        name=f"{label} (stored)",
        layer_key="ref",
        y=_field_y(state, field, None),
        color=CHART_REF_COLORS[field],
        dash=CHART_FIELD_DASH[field],
        visible=True,
      ))
    for ly in state.layers:
      if values_mode:
        name = f"{label} ({ly.label})"
        dash = CHART_FIELD_DASH[field]
      else:
        name = f"{label} delta ({ly.label})"
        dash = "solid"
      specs.append(ChartTrace(
        row=row,
        name=name,
        layer_key=ly.layer_id,
        y=_field_y(state, field, ly),
        color=ly.color_for(field),
        dash=dash,
        visible=state.visible.get(ly.layer_id, True),
      ))
  return specs


def _ylim_chunks(state: ComparisonChartState) -> dict[int, list[NDArray[np.float64]]]:
  """Y samples that set each subplot range. Hidden layers stay out of the range."""
  show_ref = state.visible.get(STORED_LAYER_ID, True)
  values_mode = state.display_mode == "values"
  chunks: dict[int, list[NDArray[np.float64]]] = {row: [] for row in _ROW_FIELDS}
  for row, fields in _ROW_FIELDS.items():
    for field in fields:
      if values_mode and show_ref:
        chunks[row].append(_field_y(state, field, None))
      for ly in state.layers:
        if not state.visible.get(ly.layer_id, True):
          continue
        chunks[row].append(_field_y(state, field, ly))
  return chunks


class LiveTimeseriesChart:
  """
  Three stacked Plotly panels: sin theta1/theta2, omega, PE/KE.

  Supports stored reference + optional NN layers (values or residuals).
  """

  def __init__(self, plotly_element) -> None:
    self._el = plotly_element
    self._fig: go.Figure | None = None
    self._state: ComparisonChartState | None = None
    self._last_frame: int | None = None
    self._trace_map: list[ChartTrace] = []

  def load_comparison(self, ctx: PoolComparisonContext) -> None:
    ref = ctx.reference_series()
    layers = ctx.layer_series_list()
    visible = dict(ctx.visible)
    self._state = ComparisonChartState(ref=ref, layers=layers, visible=visible)
    self._last_frame = None
    self._build_figure()

  def load(self, series: PlaybackSeries) -> None:
    """Legacy GT-only chart."""
    ref = ReferenceSeries(
      t=series.t,
      sin_theta1=series.sin_theta1,
      sin_theta2=series.sin_theta2,
      omega1=series.omega1,
      omega2=series.omega2,
      potential=series.potential,
      kinetic=series.kinetic,
    )
    self._state = ComparisonChartState(ref=ref, layers=[], visible={STORED_LAYER_ID: True})
    self._last_frame = None
    self._build_figure()

  def set_display_mode(self, mode: DisplayMode) -> None:
    if self._state is None:
      return
    self._state.display_mode = mode
    self._last_frame = None
    self._build_figure()

  def set_layer_visible(self, layer_id: str, visible: bool) -> None:
    if self._state is None:
      return
    self._state.visible[layer_id] = visible
    if self._fig is None:
      return
    for idx, spec in enumerate(self._trace_map):
      if spec.layer_key == layer_id or (spec.layer_key == "ref" and layer_id == STORED_LAYER_ID):
        self._fig.data[idx].visible = visible
    self._el.update_figure(self._fig)

  def _build_figure(self) -> None:
    state = self._state
    if state is None:
      return
    t = state.ref.t
    fig = make_subplots(
      rows=3,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.08,
      row_heights=[0.34, 0.33, 0.33],
    )
    self._trace_map = []

    if t.size == 0:
      self._fig = fig
      self._el.update_figure(fig)
      return

    t0, t1 = float(t[0]), float(t[-1])
    specs = chart_traces(state)
    self._trace_map = specs
    for spec in specs:
      fig.add_trace(
        go.Scatter(
          x=[],
          y=[],
          mode="lines",
          name=spec.name,
          line={"color": spec.color, "width": 1.5, "dash": spec.dash},
          visible=spec.visible,
        ),
        row=spec.row,
        col=1,
      )

    for row, chunks in _ylim_chunks(state).items():
      if chunks:
        fig.update_yaxes(range=_ylim_pad(np.concatenate(chunks)), row=row, col=1)

    fig.update_xaxes(range=[t0, t1], row=3, col=1)
    fig.update_yaxes(title_text="sin theta", row=1, col=1)
    fig.update_yaxes(title_text="omega (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="energy", row=3, col=1)
    fig.update_layout(
      height=440,
      margin=dict(l=56, r=12, t=16, b=40),
      uirevision="timeseries",
      showlegend=True,
      legend=dict(font=dict(size=8), x=1.02),
    )
    fig.update_xaxes(title_text="t (sec)", row=3, col=1)

    self._fig = fig
    self._el.update_figure(fig)
    self.set_frame(0)

  def set_frame(self, k: int) -> None:
    if self._fig is None or self._state is None:
      return
    t = self._state.ref.t
    if t.size == 0:
      return
    k = int(max(0, min(k, t.size - 1)))
    if k == self._last_frame:
      return
    self._last_frame = k
    end = k + 1
    tx = t[:end]
    for idx, spec in enumerate(self._trace_map):
      self._fig.data[idx].x = tx
      self._fig.data[idx].y = spec.y[:end]
    self._el.update_figure(self._fig)
