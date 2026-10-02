"""Pool trajectory comparison: stored GT + N surrogate checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from srcs.physics.core import PendulumParams
from srcs.simulation.data import TrajectoryView
from srcs.visualization.comparison_data import (
  CHART_LAYER_COLOR_SETS,
  LayerSeries,
  ReferenceSeries,
  reference_from_ground_truth,
)
from srcs.visualization.sources import GroundTruthSource
from srcs.visualization.surrogate import (
  STORED_LAYER_ID,
  OVERLAY_PALETTES,
  SurrogateSource,
  series_from_pred,
)
from srcs.visualization.viz import LayerStyle, PendulumFrame


@dataclass
class PoolComparisonContext:
  """Stored reference + loaded surrogates for one pool trajectory."""

  view: TrajectoryView
  cache_key: tuple
  stored: GroundTruthSource | None = None
  surrogates: list[SurrogateSource] = field(default_factory=list)
  visible: dict[str, bool] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if self.stored is None:
      self.stored = GroundTruthSource(self.view, mode="stored")
    if STORED_LAYER_ID not in self.visible:
      self.visible[STORED_LAYER_ID] = True
    for s in self.surrogates:
      lid = s.layer_id()
      if lid not in self.visible:
        self.visible[lid] = True

  def set_view(self, view: TrajectoryView, cache_key: tuple) -> None:
    self.view = view
    self.cache_key = cache_key
    self.stored = GroundTruthSource(view, mode="stored")
    for s in self.surrogates:
      s.invalidate_cache()

  def params(self) -> PendulumParams:
    return self.stored.params()

  def n_frames(self) -> int:
    return self.stored.n_frames()

  def time_at(self, k: int) -> float:
    return self.stored.time_at(k)

  def set_visible(self, layer_id: str, on: bool) -> None:
    self.visible[layer_id] = on

  def svg_layers(
    self,
    k: int,
    *,
    show_trail: bool,
  ) -> tuple[PendulumFrame | None, list[tuple[PendulumFrame, LayerStyle]]]:
    p = self.params()
    overlays: list[tuple[PendulumFrame, LayerStyle]] = []
    primary: PendulumFrame | None = None

    if self.visible.get(STORED_LAYER_ID, True):
      trail = self.stored.tip_trail(k) if show_trail else ()
      primary = PendulumFrame(
        params=p,
        state=self.stored.frame_state(k),
        trail=trail,
      )

    for i, sur in enumerate(self.surrogates):
      lid = sur.layer_id()
      if not self.visible.get(lid, True):
        continue
      trail = sur.tip_trail(k, self.view, p, self.cache_key) if show_trail else ()
      state = sur.frame_state(k, self.view, self.cache_key)
      pal = OVERLAY_PALETTES[i % len(OVERLAY_PALETTES)]
      style = LayerStyle(dashed=True, **pal)
      overlays.append((PendulumFrame(params=p, state=state, trail=trail), style))

    return primary, overlays

  def reference_series(self) -> ReferenceSeries:
    return reference_from_ground_truth(self.stored)

  def layer_series_list(self) -> list[LayerSeries]:
    ref = self.reference_series()
    t = ref.t
    p = self.params()
    layers: list[LayerSeries] = []
    for i, sur in enumerate(self.surrogates):
      pred = sur.predict_series(self.view, self.cache_key)
      arrays = series_from_pred(t, pred, p)
      trace_colors = CHART_LAYER_COLOR_SETS[i % len(CHART_LAYER_COLOR_SETS)]
      layers.append(
        LayerSeries(
          layer_id=sur.layer_id(),
          label=sur.label,
          sin_theta1=arrays["sin_theta1"],
          sin_theta2=arrays["sin_theta2"],
          omega1=arrays["omega1"],
          omega2=arrays["omega2"],
          potential=arrays["potential"],
          kinetic=arrays["kinetic"],
          trace_colors=trace_colors,
        )
      )
    return layers


# v2: LiveComparisonContext(RK4 reference, rolling buffer, predict_one per tick)
