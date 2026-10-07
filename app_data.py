"""NiceGUI app for browsing pre-generated trajectory pools.

Run::

  python app_data.py

Open http://localhost:8766, enter **data_root**, then **Load pools**.
Optional: ``create_app(Path("data/v1_small"))`` skips the picker when pools exist.

See ``docs/visualization.md`` for pool file layout and NN overlay options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from nicegui import ui

from srcs.simulation.data import list_pools, open_pool
from srcs.visualization.compare_panel import attach_pool_comparison_panel
from srcs.visualization.layer_registry import PoolComparisonContext
from srcs.visualization.live_timeseries import LiveTimeseriesChart
from srcs.visualization.viz import SHARED_HEAD_HTML, build_svg

PORT = 8766
UI_HZ = 60
_live_timers: list = []


def _stop_live_timers() -> None:
  for timer in _live_timers:
    timer.deactivate()
  _live_timers.clear()


def parse_data_root(raw: str) -> Path | None:
  """Validate user path; notify and return None on failure."""
  text = (raw or "").strip()
  if not text:
    ui.notify("Enter a data_root path", type="warning")
    return None
  path = Path(text).expanduser()
  if not path.is_dir():
    ui.notify(f"Not a directory: {path}", type="negative")
    return None
  if not list_pools(path):
    ui.notify(f"No stage*_<split>.npz pools under {path}", type="negative")
    return None
  return path.resolve()


def mount_picker(host: ui.column, initial: str, on_load: Callable[[Path], None]) -> None:
  """First screen: user supplies a pool directory."""
  _stop_live_timers()
  host.clear()
  with host:
    ui.label("Double pendulum dataset browser").classes("text-h6")
    ui.label(
      "Folder must contain stage{S}_{split}.npz files (e.g. data/v1_small)."
    ).classes("meta")
    data_root_input = ui.input(
      label="data_root",
      value=initial,
      placeholder="data/v1_small",
    ).classes("w-full")

    def apply() -> None:
      root = parse_data_root(data_root_input.value or "")
      if root is not None:
        on_load(root)

    ui.button("Load pools", on_click=apply).props("outline")


def mount_browser(host: ui.column, root: Path) -> None:
  """Trajectory viewer and sidebar controls for one data_root."""
  _stop_live_timers()
  pools = list_pools(root)
  if not pools:
    mount_picker(host, str(root), lambda r: mount_browser(host, r))
    return

  host.clear()

  stage_ids = sorted({stage for stage, _, _ in pools})
  stage_options = {st: f"Stage {st}" for st in stage_ids}
  default_stage, default_split, _ = pools[0]

  pool = open_pool(root, default_stage, default_split)
  view0 = pool.get_traj(0)
  cache_key0 = (default_stage, default_split, 0)

  ctx = PoolComparisonContext(view=view0, cache_key=cache_key0)

  state = {
    "frame": 0,
    "frame_accum": 0.0,
    "running": True,
    "show_trail": True,
    "playback_speed": 1.0,
    "data_root": root,
    "stage": default_stage,
    "split": default_split,
    "traj": 0,
    "show_timeseries": True,
    "last_rendered_frame": -1,
  }

  def cache_key() -> tuple:
    return (state["stage"], state["split"], state["traj"])

  def refresh_trajectory() -> None:
    pool = open_pool(state["data_root"], state["stage"], state["split"])
    idx = min(state["traj"], max(0, pool.n_traj - 1))
    key = cache_key()
    ctx.set_view(pool.get_traj(idx), key)

  def render_pendulum(k: int) -> None:
    primary, overlays = ctx.svg_layers(k, show_trail=state["show_trail"])
    stage_el.set_content(
      build_svg(primary, show_trail=state["show_trail"], overlays=overlays)
    )
    time_label.text = (
      f"t = {ctx.time_at(k):.3f} sec  "
      f"frame {k}/{ctx.n_frames() - 1}"
    )

  def sync_scrub_position() -> None:
    if ctx.n_frames() == 0:
      return
    k = min(state["frame"], ctx.n_frames() - 1)
    scrub_time_val.text = f"t = {ctx.time_at(k):.3f} sec"
    cur = int(round(frame_scrub.value or 0))
    if cur != k:
      frame_scrub.set_value(float(k))

  def sync_scrub_bounds() -> None:
    last = max(0, ctx.n_frames() - 1)
    with frame_scrub._props.suspend_updates():
      frame_scrub._props["min"] = 0
      frame_scrub._props["max"] = last
    frame_scrub.update()
    if state["frame"] > last:
      state["frame"] = last
    sync_scrub_position()

  def on_visual_change() -> None:
    state["last_rendered_frame"] = -1
    render_frame()

  def render_frame() -> None:
    k = min(state["frame"], max(0, ctx.n_frames() - 1))
    if k != state["last_rendered_frame"]:
      render_pendulum(k)
      state["last_rendered_frame"] = k
      if state["show_timeseries"]:
        ts_chart.set_frame(k)
    sync_scrub_position()

  def refresh_meta(meta_label: ui.label) -> None:
    try:
      m = ctx.stored.ic_meta()
      meta_label.text = (
        f"g={m['g']:.3g}  m1={m['m1']:.3g}  m2={m['m2']:.3g}  "
        f"|dE|_max={m['energy_drift_max']:.2e}"
      )
    except Exception as exc:
      meta_label.text = str(exc)

  def on_pool_change() -> None:
    state["stage"] = int(stage_sel.value)
    state["split"] = str(split_sel.value)
    pool = open_pool(state["data_root"], state["stage"], state["split"])
    state["traj"] = min(int(traj_sel.value or 0), max(0, pool.n_traj - 1))
    traj_sel.value = state["traj"]
    state["frame"] = 0
    state["frame_accum"] = 0.0
    state["last_rendered_frame"] = -1
    refresh_trajectory()
    refresh_meta(meta_label)
    comp_ui.reload_chart()
    sync_scrub_bounds()
    render_frame()

  def apply_data_root() -> None:
    new_root = parse_data_root(data_root_input.value or "")
    if new_root is None:
      return
    found = list_pools(new_root)
    state["data_root"] = new_root
    data_root_input.value = str(new_root)
    stages = sorted({stage for stage, _, _ in found})
    opts = {st: f"Stage {st}" for st in stages}
    stage_sel.options = opts
    stage_sel.update()
    state["stage"] = stages[0]
    stage_sel.value = stages[0]
    state["split"] = str(split_sel.value or "test")
    ui.notify(f"Using data_root {new_root}")
    on_pool_change()

  def change_data_root() -> None:
    mount_picker(host, str(state["data_root"]), lambda r: mount_browser(host, r))

  with host:
    with ui.column().classes("w-full").style("gap: 1.25rem;"):
      with ui.row().classes("w-full items-start").style(
        "gap: 1.25rem; flex-wrap: wrap; align-items: flex-start;"
      ):
        with ui.column().style(
          "flex: 1 1 520px; min-width: 280px; gap: 1rem; align-items: center;"
        ):
          stage_el = ui.html("", sanitize=False).classes("stage")
          ts_plot = ui.plotly({}).classes("w-full").style(
            "max-width: 640px; min-height: 420px; border-radius: 16px; "
            "border: 1px solid rgba(26, 42, 58, 0.08); "
            "box-shadow: 0 12px 28px rgba(26, 42, 58, 0.1);"
          )
          ts_chart = LiveTimeseriesChart(ts_plot)

        with ui.column().classes("panel").style(
          "flex: 0 0 300px; width: min(300px, 100%); gap: 0.55rem;"
        ):
          status = ui.label("Running").classes("meta")
          time_label = ui.label("t = 0.00 sec").classes("meta")
          meta_label = ui.label("").classes("meta")

          data_root_input = ui.input(
            label="data_root",
            value=str(root.resolve()),
          ).classes("w-full")
          with ui.row().style("gap: 0.5rem; flex-wrap: wrap;"):
            ui.button("Apply data root", on_click=apply_data_root).props("outline")
            ui.button("Change folder", on_click=change_data_root).props("flat")

          stage_sel = ui.select(
            stage_options,
            value=state["stage"],
            label="Curriculum stage",
          )
          split_sel = ui.select(
            ["train", "val", "test"],
            value=state["split"],
            label="split",
          )
          traj_sel = ui.number(
            label="Trajectory index",
            value=0,
            min=0,
            step=1,
            precision=0,
          )
          with ui.column().classes("slider-block"):
            with ui.row().classes("slider-head"):
              ui.label("Playback speed").classes("slider-caption")
              speed_val = ui.label("1x").classes("meta slider-value")
            speed_slider = ui.slider(min=1, max=10, value=1, step=1)
          with ui.column().classes("slider-block"):
            with ui.row().classes("slider-head"):
              ui.label("Timeline").classes("slider-caption")
              scrub_time_val = ui.label("t = 0.000 sec").classes("meta slider-value")
            frame_scrub = ui.slider(min=0, max=1, value=0, step=1)
          trail_toggle = ui.switch("Show trail", value=True)
          timeseries_toggle = ui.switch("Show time series", value=True)

          compare_host = ui.column().classes("w-full")
          comp_ui = attach_pool_comparison_panel(
            compare_host,
            ctx=ctx,
            chart=ts_chart,
            on_change=on_visual_change,
          )

          with ui.row().style("gap: 0.5rem; flex-wrap: wrap;"):
            play_btn = ui.button("Pause")
            reset_btn = ui.button("Reset", color="secondary")
            gif_btn = ui.button("Export gif", color="secondary")

  def on_scrub_change() -> None:
    if ctx.n_frames() == 0:
      return
    new_k = int(round(frame_scrub.value or 0))
    new_k = min(max(0, new_k), ctx.n_frames() - 1)
    if new_k == state["frame"]:
      return
    state["frame"] = new_k
    state["last_rendered_frame"] = -1
    scrub_time_val.text = f"t = {ctx.time_at(new_k):.3f} sec"
    render_frame()

  def toggle_play() -> None:
    state["running"] = not state["running"]
    play_btn.text = "Pause" if state["running"] else "Play"
    status.text = "Running" if state["running"] else "Paused"

  def on_reset() -> None:
    state["frame"] = 0
    state["frame_accum"] = 0.0
    state["last_rendered_frame"] = -1
    sync_scrub_position()
    render_frame()

  def on_speed_change() -> None:
    state["playback_speed"] = float(speed_slider.value)
    speed_val.text = f"{int(state['playback_speed'])}x"

  def on_trail() -> None:
    state["show_trail"] = bool(trail_toggle.value)
    state["last_rendered_frame"] = -1
    render_frame()

  def sim_dt() -> float:
    if ctx.n_frames() < 2:
      return 1.0 / UI_HZ
    return max(float(ctx.time_at(1) - ctx.time_at(0)), 1e-9)

  def on_timeseries_toggle() -> None:
    state["show_timeseries"] = bool(timeseries_toggle.value)
    ts_plot.set_visibility(state["show_timeseries"])
    if state["show_timeseries"]:
      comp_ui.reload_chart()
      state["last_rendered_frame"] = -1
      render_frame()

  def on_export_gif() -> None:
    from srcs.visualization.plots import export_gif_batch

    idx = int(traj_sel.value or 0)
    paths = export_gif_batch(
      state["data_root"],
      Path("figures/gifs"),
      stage=state["stage"],
      split=state["split"],
      indices=[idx],
      stride=max(1, ctx.n_frames() // 60),
    )
    ui.notify(f"Wrote {paths[0]}" if paths else "Nothing exported")

  stage_sel.on_value_change(lambda _: on_pool_change())
  split_sel.on_value_change(lambda _: on_pool_change())
  traj_sel.on_value_change(lambda _: on_pool_change())
  play_btn.on_click(toggle_play)
  reset_btn.on_click(on_reset)
  trail_toggle.on_value_change(lambda _: on_trail())
  timeseries_toggle.on_value_change(lambda _: on_timeseries_toggle())
  speed_slider.on_value_change(lambda _: on_speed_change())
  frame_scrub.on_value_change(lambda _: on_scrub_change())
  on_speed_change()

  gif_btn.on_click(on_export_gif)

  refresh_meta(meta_label)
  comp_ui.reload_chart()
  sync_scrub_bounds()
  render_frame()

  def tick() -> None:
    if state["running"]:
      n = ctx.n_frames()
      if n > 1:
        dt = sim_dt()
        state["frame_accum"] += (1.0 / UI_HZ) / dt * state["playback_speed"]
        step = int(state["frame_accum"])
        if step > 0:
          state["frame_accum"] -= step
          state["frame"] = (state["frame"] + step) % n
    render_frame()

  _live_timers.append(ui.timer(1 / UI_HZ, tick))


def create_app(data_root: Path | None = None) -> None:
  ui.add_head_html(SHARED_HEAD_HTML)
  host = ui.column().classes("w-full q-pa-md").style(
    "max-width: 1100px; margin: 0 auto;"
  )
  if data_root is not None and list_pools(data_root):
    mount_browser(host, Path(data_root).expanduser().resolve())
  else:
    initial = str(data_root) if data_root is not None else ""
    mount_picker(host, initial, lambda root: mount_browser(host, root))


def main() -> None:
  create_app()
  ui.run(
    title="Double Pendulum - data",
    host="0.0.0.0",
    port=PORT,
    reload=False,
    show=False,
  )


if __name__ in {"__main__", "__mp_main__"}:
  main()
