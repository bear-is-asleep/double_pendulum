"""Live time-series charts synced with dataset playback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from srcs.physics.core import cartesian
from srcs.visualization.comparison_data import (
  CHART_REF_COLORS,
  LayerSeries,
  ReferenceSeries,
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
  """
  Build plot arrays from stored pool labels.
  """
  v = src.view
  t = np.asarray(v.t, dtype=np.float64)
  return PlaybackSeries(
    t=t,
    sin_theta1=np.asarray(v.sin_theta1, dtype=np.float64),
    cos_theta1=np.asarray(v.cos_theta1, dtype=np.float64),
    sin_theta2=np.asarray(v.sin_theta2, dtype=np.float64),
    cos_theta2=np.asarray(v.cos_theta2, dtype=np.float64),
    omega1=np.asarray(v.omega1, dtype=np.float64),
    omega2=np.asarray(v.omega2, dtype=np.float64),
    potential=np.asarray(v.potential, dtype=np.float64),
    kinetic=np.asarray(v.kinetic, dtype=np.float64),
    energy=np.asarray(v.energy, dtype=np.float64),
  )


def precompute_tip_positions(src: GroundTruthSource) -> list[tuple[float, float]]:
  """Lower-bob (x2, y2) at every frame; built once per trajectory selection."""
  p = src.params()
  n = src.n_frames()
  tips: list[tuple[float, float]] = []
  for i in range(n):
    _, _, x2, y2 = cartesian(src.frame_state(i), p)
    tips.append((x2, y2))
  return tips


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


class LiveTimeseriesChart:
  """
  Three stacked Plotly panels: sin θ1/θ2, ω, PE/KE.

  Supports stored reference + optional NN layers (values or residuals).
  """

  REF_DASH = "solid"

  def _ref_color(self, field: str) -> str:
    return CHART_REF_COLORS[field]

  def _layer_color(self, layer: LayerSeries, field: str) -> str:
    return layer.color_for(field)

  def __init__(self, plotly_element) -> None:
    self._el = plotly_element
    self._fig: go.Figure | None = None
    self._state: ComparisonChartState | None = None
    self._last_frame: int | None = None
    # trace index map: list of (row, name, layer_id | "ref", field)
    self._trace_map: list[tuple[int, str, str, str]] = []

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
    for idx, (_row, _name, lid, _field) in enumerate(self._trace_map):
      if lid == layer_id or (lid == "ref" and layer_id == STORED_LAYER_ID):
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
    mode = state.display_mode
    show_ref = state.visible.get(STORED_LAYER_ID, True)
    compare_layers = [
      ly for ly in state.layers if state.visible.get(ly.layer_id, True)
    ]

    def add_trace(row: int, name: str, layer_key: str, yfull: NDArray[np.float64], *, color: str, dash: str) -> None:
      fig.add_trace(
        go.Scatter(
          x=[],
          y=[],
          mode="lines",
          name=name,
          line={"color": color, "width": 1.5, "dash": dash},
          visible=state.visible.get(layer_key, True) if layer_key != "ref" else show_ref,
        ),
        row=row,
        col=1,
      )
      self._trace_map.append((row, name, layer_key, ""))

    # Panel fields: (row, field_name, short label)
    panels = [
      (1, "sin_theta1", "sin θ1"),
      (1, "sin_theta2", "sin θ2"),
      (2, "omega1", "ω1"),
      (2, "omega2", "ω2"),
      (3, "potential", "PE"),
      (3, "kinetic", "KE"),
    ]

    for row, field, label in panels:
      ref_y = getattr(state.ref, field)
      if mode == "values":
        if show_ref:
          add_trace(
            row,
            f"{label} (stored)",
            "ref",
            ref_y,
            color=self._ref_color(field),
            dash=self.REF_DASH,
          )
        for ly in state.layers:
          ly_y = getattr(ly, field)
          add_trace(
            row,
            f"{label} ({ly.label})",
            ly.layer_id,
            ly_y,
            color=self._layer_color(ly, field),
            dash="dash",
          )
      else:
        for ly in state.layers:
          ly_y = getattr(ly, field)
          err = ref_y - ly_y
          add_trace(
            row,
            f"{label} Δ ({ly.label})",
            ly.layer_id,
            err,
            color=self._layer_color(ly, field),
            dash="solid",
          )

    row_fields = {
      1: ["sin_theta1", "sin_theta2"],
      2: ["omega1", "omega2"],
      3: ["potential", "kinetic"],
    }
    for row, fields in row_fields.items():
      chunks: list[NDArray[np.float64]] = []
      if mode == "values" and show_ref:
        for f in fields:
          chunks.append(getattr(state.ref, f))
      for ly in compare_layers:
        for f in fields:
          ly_y = getattr(ly, f)
          if mode == "values":
            chunks.append(ly_y)
          else:
            chunks.append(getattr(state.ref, f) - ly_y)
      if chunks:
        fig.update_yaxes(range=_ylim_pad(np.concatenate(chunks)), row=row, col=1)

    fig.update_xaxes(range=[t0, t1], row=3, col=1)
    fig.update_yaxes(title_text="sin θ", row=1, col=1)
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

    trace_i = 0
    state = self._state
    mode = state.display_mode
    show_ref = state.visible.get(STORED_LAYER_ID, True)

    panels = [
      ("sin_theta1",),
      ("sin_theta2",),
      ("omega1",),
      ("omega2",),
      ("potential",),
      ("kinetic",),
    ]

    for field_tuple in panels:
      field = field_tuple[0]
      ref_y = getattr(state.ref, field)
      if mode == "values":
        if show_ref:
          self._fig.data[trace_i].x = tx
          self._fig.data[trace_i].y = ref_y[:end]
          trace_i += 1
        for ly in state.layers:
          ly_y = getattr(ly, field)
          self._fig.data[trace_i].x = tx
          self._fig.data[trace_i].y = ly_y[:end]
          trace_i += 1
      else:
        for ly in state.layers:
          ly_y = getattr(ly, field)
          self._fig.data[trace_i].x = tx
          self._fig.data[trace_i].y = (ref_y - ly_y)[:end]
          trace_i += 1

    self._el.update_figure(self._fig)
