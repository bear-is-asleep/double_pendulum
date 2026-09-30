"""Live time-series charts synced with dataset playback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from double_pendulum.physics import cartesian, kinetic_energy, potential_energy
from double_pendulum.sources import GroundTruthSource


@dataclass(frozen=True)
class PlaybackSeries:
  """Aligned arrays for sin/cos, omega, and energy plots."""

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
  Build plot arrays that match the active playback mode.

  ``stored`` reads labels from the pool file. ``reintegrate`` fills series from
  the re-run RK4 states (including PE/KE from physics helpers).
  """
  v = src.view
  t = np.asarray(v.t, dtype=np.float64)
  if src.mode == "stored":
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

  n = src.n_frames()
  p = src.params()
  sin_theta1 = np.empty(n, dtype=np.float64)
  cos_theta1 = np.empty(n, dtype=np.float64)
  sin_theta2 = np.empty(n, dtype=np.float64)
  cos_theta2 = np.empty(n, dtype=np.float64)
  omega1 = np.empty(n, dtype=np.float64)
  omega2 = np.empty(n, dtype=np.float64)
  potential = np.empty(n, dtype=np.float64)
  kinetic = np.empty(n, dtype=np.float64)
  energy = np.empty(n, dtype=np.float64)
  for i in range(n):
    s = src.frame_state(i)
    sin_theta1[i] = np.sin(s.theta1)
    cos_theta1[i] = np.cos(s.theta1)
    sin_theta2[i] = np.sin(s.theta2)
    cos_theta2[i] = np.cos(s.theta2)
    omega1[i] = s.omega1
    omega2[i] = s.omega2
    pe = potential_energy(s, p)
    ke = kinetic_energy(s, p)
    potential[i] = pe
    kinetic[i] = ke
    energy[i] = pe + ke
  return PlaybackSeries(
    t=t,
    sin_theta1=sin_theta1,
    cos_theta1=cos_theta1,
    sin_theta2=sin_theta2,
    cos_theta2=cos_theta2,
    omega1=omega1,
    omega2=omega2,
    potential=potential,
    kinetic=kinetic,
    energy=energy,
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


_TRACE_SPECS: list[tuple[str, str, dict]] = [
  ("sin(theta1)", "sin_theta1", {"line": {"color": "#2f6f8f", "width": 1.5}}),
  ("cos(theta1)", "cos_theta1", {"line": {"color": "#1a5a72", "width": 1.5, "dash": "dash"}}),
  ("sin(theta2)", "sin_theta2", {"line": {"color": "#c45c26", "width": 1.5}}),
  ("cos(theta2)", "cos_theta2", {"line": {"color": "#8a3d18", "width": 1.5, "dash": "dash"}}),
  ("omega1", "omega1", {"line": {"color": "#2f6f8f", "width": 1.5}}),
  ("omega2", "omega2", {"line": {"color": "#c45c26", "width": 1.5}}),
  ("PE", "potential", {"line": {"color": "#6b4c9a", "width": 1.5}}),
  ("KE", "kinetic", {"line": {"color": "#2f6f8f", "width": 1.5}}),
  ("E", "energy", {"line": {"color": "#1a2a3a", "width": 2}}),
]


class LiveTimeseriesChart:
  """
  Three stacked Plotly panels; lines grow from t=0 through the current frame.

  Plotly renders in the browser (no matplotlib SVG refresh per step).
  """

  def __init__(self, plotly_element) -> None:
    self._el = plotly_element
    self._fig: go.Figure | None = None
    self._series: PlaybackSeries | None = None
    self._y_arrays: list[NDArray[np.float64]] = []
    self._last_frame: int | None = None

  def load(self, series: PlaybackSeries) -> None:
    """Reset traces and axis limits from the full trajectory."""
    self._series = series
    self._last_frame = None
    t = series.t
    self._y_arrays = [
      series.sin_theta1,
      series.cos_theta1,
      series.sin_theta2,
      series.cos_theta2,
      series.omega1,
      series.omega2,
      series.potential,
      series.kinetic,
      series.energy,
    ]

    fig = make_subplots(
      rows=3,
      cols=1,
      shared_xaxes=True,
      vertical_spacing=0.08,
      row_heights=[0.34, 0.33, 0.33],
    )

    if t.size == 0:
      self._fig = fig
      self._el.update_figure(fig)
      return

    t0, t1 = float(t[0]), float(t[-1])
    row_for_trace = [1, 1, 1, 1, 2, 2, 3, 3, 3]
    legend_for_row = {1: "legend", 2: "legend2", 3: "legend3"}

    for i, (name, _field, style) in enumerate(_TRACE_SPECS):
      row = row_for_trace[i]
      fig.add_trace(
        go.Scatter(
          x=[],
          y=[],
          mode="lines",
          name=name,
          line=style["line"],
          legend=legend_for_row[row],
          showlegend=True,
        ),
        row=row,
        col=1,
      )

    fig.update_xaxes(range=[t0, t1], row=3, col=1)
    fig.update_yaxes(
      range=_ylim_pad(np.concatenate(self._y_arrays[0:4])),
      title_text="sin / cos",
      row=1,
      col=1,
    )
    fig.update_yaxes(
      range=_ylim_pad(np.concatenate(self._y_arrays[4:6])),
      title_text="omega (rad/s)",
      row=2,
      col=1,
    )
    fig.update_yaxes(
      range=_ylim_pad(np.concatenate(self._y_arrays[6:9])),
      title_text="energy",
      row=3,
      col=1,
    )

    legend_style = dict(
      font=dict(size=8),
      bgcolor="rgba(255,255,255,0.85)",
      bordercolor="rgba(26,42,58,0.12)",
      borderwidth=1,
      x=0.99,
      xanchor="right",
      yanchor="top",
    )
    fig.update_layout(
      height=440,
      margin=dict(l=56, r=12, t=16, b=40),
      uirevision="timeseries",
      legend=dict(y=0.99, **legend_style),
      legend2=dict(y=0.63, **legend_style),
      legend3=dict(y=0.27, **legend_style),
    )
    fig.update_xaxes(title_text="t (sec)", row=3, col=1)

    self._fig = fig
    self._el.update_figure(fig)
    self.set_frame(0)

  def set_frame(self, k: int) -> None:
    """Extend visible polylines through simulation index ``k``."""
    if self._fig is None or self._series is None or not self._y_arrays:
      return
    t = self._series.t
    if t.size == 0:
      return
    k = int(max(0, min(k, t.size - 1)))
    if k == self._last_frame:
      return
    self._last_frame = k
    end = k + 1
    tx = t[:end]
    for i, yfull in enumerate(self._y_arrays):
      self._fig.data[i].x = tx
      self._fig.data[i].y = yfull[:end]
    self._el.update_figure(self._fig)
