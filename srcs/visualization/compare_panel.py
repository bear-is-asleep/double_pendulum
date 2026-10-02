"""Shared NiceGUI controls for checkpoint comparison (pool mode v1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nicegui import ui

from srcs.visualization.layer_registry import PoolComparisonContext
from srcs.visualization.live_timeseries import (
  DisplayMode,
  LiveTimeseriesChart,
  series_from_ground_truth,
)
from srcs.visualization.surrogate import STORED_LAYER_ID, SurrogateSource


@dataclass
class PoolComparisonUI:
  """Handles loaded checkpoints and layer visibility for ``app_data.py``."""

  ctx: PoolComparisonContext
  chart: LiveTimeseriesChart
  on_change: Callable[[], None]
  _layer_switch_map: dict[str, ui.switch] = field(default_factory=dict)
  display_mode: DisplayMode = "values"
  status_label: ui.label | None = None
  layer_column: ui.column | None = None
  _mode_sel: ui.select | None = field(default=None, repr=False)

  def reload_chart(self) -> None:
    if self.ctx.surrogates:
      self.chart.load_comparison(self.ctx)
    else:
      self.chart.load(series_from_ground_truth(self.ctx.stored))
    mode = self.display_mode
    if mode == "errors" and not self.ctx.surrogates:
      mode = "values"
      self.display_mode = "values"
      if self._mode_sel is not None:
        self._mode_sel.value = "values"
    self.chart.set_display_mode(mode)

  def sync_layer_switches(self) -> None:
    if self.layer_column is None:
      return
    self.layer_column.clear()
    self._layer_switch_map.clear()
    with self.layer_column:
      stored_sw = ui.switch("Stored (disk)", value=self.ctx.visible.get(STORED_LAYER_ID, True))
      stored_sw.on_value_change(lambda e: self._on_layer(STORED_LAYER_ID, bool(e.value)))
      self._layer_switch_map[STORED_LAYER_ID] = stored_sw
      for sur in self.ctx.surrogates:
        lid = sur.layer_id()
        sw = ui.switch(f"NN: {sur.label}", value=self.ctx.visible.get(lid, True))
        sw.on_value_change(lambda e, layer=lid: self._on_layer(layer, bool(e.value)))
        self._layer_switch_map[lid] = sw

  def _on_layer(self, layer_id: str, on: bool) -> None:
    self.ctx.set_visible(layer_id, on)
    self.chart.set_layer_visible(layer_id, on)
    self.on_change()

  def add_checkpoint(self, path: Path) -> None:
    path = Path(path)
    for s in self.ctx.surrogates:
      if s.checkpoint_path.resolve() == path.resolve():
        ui.notify("Model already loaded")
        return
    sur = SurrogateSource(checkpoint_path=path)
    try:
      meta = sur.meta()
    except Exception as exc:
      ui.notify(f"Load failed: {exc}", type="negative")
      return
    self.ctx.surrogates.append(sur)
    self.ctx.visible[sur.layer_id()] = True
    vm = meta.get("val_metric")
    msg = f"Loaded {sur.label}"
    if vm is not None:
      msg += f"  val={float(vm):.4g}"
    if self.status_label is not None:
      self.status_label.text = msg
    ui.notify(msg)
    self.sync_layer_switches()
    self.reload_chart()
    self.on_change()

  def remove_checkpoint(self, index: int) -> None:
    if index < 0 or index >= len(self.ctx.surrogates):
      return
    removed = self.ctx.surrogates.pop(index)
    self.ctx.visible.pop(removed.layer_id(), None)
    self.sync_layer_switches()
    self.reload_chart()
    self.on_change()


def attach_pool_comparison_panel(
  panel: ui.column,
  *,
  ctx: PoolComparisonContext,
  chart: LiveTimeseriesChart,
  on_change: Callable[[], None],
) -> PoolComparisonUI:
  """
  Add checkpoint + layer + chart mode controls to an existing NiceGUI column.

  v2: ``attach_live_comparison_panel`` for ``app.py``.
  """
  comp_ui = PoolComparisonUI(
    ctx=ctx,
    chart=chart,
    on_change=on_change,
  )

  with panel:
    ui.label("Models").classes("slider-caption")
    status = ui.label("No checkpoint loaded").classes("meta")
    comp_ui.status_label = status
    ckpt_input = ui.input(
      label="Checkpoint path",
      placeholder="runs/.../checkpoints/best.pt",
    ).classes("w-full")

    def on_add() -> None:
      raw = (ckpt_input.value or "").strip()
      if not raw:
        ui.notify("Enter a checkpoint path", type="warning")
        return
      comp_ui.add_checkpoint(Path(raw))

    with ui.row().style("gap: 0.5rem; flex-wrap: wrap;"):
      ui.button("Add model", on_click=on_add)

    ui.label("Animation layers").classes("slider-caption")
    layer_column = ui.column().classes("w-full")
    comp_ui.layer_column = layer_column
    comp_ui.sync_layer_switches()

    mode_sel = ui.select(
      {"values": "Chart: values", "errors": "Chart: errors (stored - NN)"},
      value="values",
      label="Time series mode",
    )

    def on_mode(_: object) -> None:
      mode = str(mode_sel.value)
      if mode not in ("values", "errors"):
        return
      if mode == "errors" and not ctx.surrogates:
        ui.notify("Load a model for error view", type="warning")
        mode_sel.value = "values"
        return
      comp_ui.display_mode = mode  # type: ignore[assignment]
      chart.set_display_mode(mode)  # type: ignore[arg-type]
      on_change()

    mode_sel.on_value_change(on_mode)
    comp_ui._mode_sel = mode_sel

  return comp_ui


def attach_live_comparison_panel(*_args, **_kwargs) -> None:
  """Reserved for v2 ``app.py`` wiring."""
  raise NotImplementedError("live comparison panel ships in v2 (app.py)")
