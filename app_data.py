"""NiceGUI app for browsing pre-generated trajectory pools.

Run::

  python app_data.py

Then open http://localhost:8766 (default). Pools are read from ``data/`` unless
you change ``DEFAULT_DATA_ROOT`` or pass ``data_root`` into ``create_app``.

This complements ``app.py`` (port 8765), which integrates the ODE live from
sliders. Here every frame comes from disk or from a re-integration sanity check.

See ``docs/visualization.md`` for pool file layout and CLI plotting options.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from double_pendulum.data import list_pools, open_pool
from double_pendulum.live_timeseries import (
  LiveTimeseriesChart,
  precompute_tip_positions,
  series_from_ground_truth,
)
from double_pendulum.sources import GroundTruthSource, SurrogateSource
from double_pendulum.viz import SHARED_HEAD_HTML, PendulumFrame, build_svg

PORT = 8766
DEFAULT_DATA_ROOT = Path("data")
UI_HZ = 60


def create_app(data_root: Path = DEFAULT_DATA_ROOT) -> None:
  """
  Build the dataset browser page.

  Parameters
  ----------
  data_root
      Directory containing ``stage{S}_{split}.npz`` files (see ``data.pool_path``).
  """
  root = Path(data_root)
  pools = list_pools(root)
  if not pools:
    ui.label(
      f"No pools under {root}. Generate data first "
      "(``python -m double_pendulum.generate_data`` or your project script)."
    )
    return

  stage_ids = sorted({stage for stage, _, _ in pools})
  stage_options = {st: f"Stage {st}" for st in stage_ids}
  default_stage, default_split, _ = pools[0]

  state = {
    "frame": 0,
    "frame_accum": 0.0,
    "running": True,
    "show_trail": True,
    "playback_speed": 1.0,
    "mode": "stored",
    "data_root": root,
    "stage": default_stage,
    "split": default_split,
    "traj": 0,
    "show_timeseries": True,
    "last_rendered_frame": -1,
    "source_key": None,
    "source": None,
    "tips": [],
    "playback_series": None,
  }
  surrogate = SurrogateSource(run_id=None, enabled=False)

  ui.add_head_html(SHARED_HEAD_HTML)

  def source_key() -> tuple:
    return (state["stage"], state["split"], state["traj"], state["mode"])

  def invalidate_playback_cache() -> None:
    state["source_key"] = None
    state["source"] = None
    state["tips"] = []
    state["playback_series"] = None

  def current_source() -> GroundTruthSource:
    key = source_key()
    if state["source_key"] != key or state["source"] is None:
      pool = open_pool(state["data_root"], state["stage"], state["split"])
      idx = min(state["traj"], max(0, pool.n_traj - 1))
      src = GroundTruthSource(pool.get_traj(idx), mode=state["mode"])
      state["source_key"] = key
      state["source"] = src
      state["tips"] = precompute_tip_positions(src)
      state["playback_series"] = series_from_ground_truth(src)
    return state["source"]

  def refresh_meta(meta_label: ui.label) -> None:
    try:
      src = current_source()
      m = src.ic_meta()
      meta_label.text = (
        f"g={m['g']:.3g}  m1={m['m1']:.3g}  m2={m['m2']:.3g}  "
        f"|dE|_max={m['energy_drift_max']:.2e}"
      )
    except Exception as exc:
      meta_label.text = str(exc)

  with ui.column().classes("w-full q-pa-md").style(
    "max-width: 1100px; margin: 0 auto; gap: 1.25rem;"
  ):
    ui.html('<p class="brand">Dataset trajectories</p>', sanitize=False)
    ui.html(
      '<p class="lede">Inspect saved simulations: choose curriculum stage, '
      "data split, and trajectory index.</p>",
      sanitize=False,
    )

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

        ui.label(f"data_root: {root}").classes("meta")

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
        playback_options = {
          "stored": "From disk (saved sin/cos, omega)",
          "reintegrate": "Re-run RK4 from initial condition",
        }
        mode_sel = ui.select(
          playback_options,
          value="stored",
          label="Playback mode",
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
        overlay_toggle = ui.switch("ANN overlay", value=False)
        overlay_toggle.disable()
        run_id_input = ui.input(
          label="run_id (optional)",
          placeholder="baseline_ann_...",
        )
        run_id_input.disable()

        with ui.row().style("gap: 0.5rem; flex-wrap: wrap;"):
          play_btn = ui.button("Pause")
          reset_btn = ui.button("Reset", color="secondary")
          gif_btn = ui.button("Export gif", color="secondary")

  def reload_timeseries() -> None:
    if not state["show_timeseries"]:
      return
    src = current_source()
    series = state["playback_series"]
    if series is None:
      series = series_from_ground_truth(src)
      state["playback_series"] = series
    ts_chart.load(series)

  def render_pendulum(k: int) -> None:
    src = current_source()
    trail = ()
    if state["show_trail"] and state["tips"]:
      trail = state["tips"][: k + 1]
    frame = PendulumFrame(
      params=src.params(),
      state=src.frame_state(k),
      trail=trail,
    )
    overlay = None
    if surrogate.is_available():
      pass
    stage_el.set_content(build_svg(frame, show_trail=state["show_trail"], overlay=overlay))
    time_label.text = (
      f"t = {src.time_at(k):.3f} sec  "
      f"frame {k}/{src.n_frames() - 1}"
    )

  def sync_scrub_position() -> None:
    src = current_source()
    if src.n_frames() == 0:
      return
    k = min(state["frame"], src.n_frames() - 1)
    scrub_time_val.text = f"t = {src.time_at(k):.3f} sec"
    cur = int(round(frame_scrub.value or 0))
    if cur != k:
      frame_scrub.set_value(float(k))

  def sync_scrub_bounds() -> None:
    src = current_source()
    last = max(0, src.n_frames() - 1)
    with frame_scrub._props.suspend_updates():
      frame_scrub._props["min"] = 0
      frame_scrub._props["max"] = last
    frame_scrub.update()
    if state["frame"] > last:
      state["frame"] = last
    sync_scrub_position()

  def on_scrub_change() -> None:
    src = current_source()
    if src.n_frames() == 0:
      return
    new_k = int(round(frame_scrub.value or 0))
    new_k = min(max(0, new_k), src.n_frames() - 1)
    # Ignore echo when playback updates the slider to match state["frame"].
    if new_k == state["frame"]:
      return
    state["frame"] = new_k
    state["last_rendered_frame"] = -1
    scrub_time_val.text = f"t = {src.time_at(new_k):.3f} sec"
    render_frame()

  def render_frame() -> None:
    src = current_source()
    k = min(state["frame"], src.n_frames() - 1)
    if k != state["last_rendered_frame"]:
      render_pendulum(k)
      state["last_rendered_frame"] = k
      if state["show_timeseries"]:
        ts_chart.set_frame(k)
    sync_scrub_position()

  def on_pool_change() -> None:
    state["stage"] = int(stage_sel.value)
    state["split"] = str(split_sel.value)
    pool = open_pool(state["data_root"], state["stage"], state["split"])
    state["traj"] = min(int(traj_sel.value or 0), max(0, pool.n_traj - 1))
    traj_sel.value = state["traj"]
    state["frame"] = 0
    state["frame_accum"] = 0.0
    state["last_rendered_frame"] = -1
    invalidate_playback_cache()
    refresh_meta(meta_label)
    reload_timeseries()
    sync_scrub_bounds()
    render_frame()

  def on_mode_change() -> None:
    state["mode"] = str(mode_sel.value)
    state["frame"] = 0
    state["frame_accum"] = 0.0
    state["last_rendered_frame"] = -1
    invalidate_playback_cache()
    refresh_meta(meta_label)
    reload_timeseries()
    sync_scrub_bounds()
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
    src = current_source()
    if src.n_frames() < 2:
      return 1.0 / UI_HZ
    return max(float(src.time_at(1) - src.time_at(0)), 1e-9)

  def on_timeseries_toggle() -> None:
    state["show_timeseries"] = bool(timeseries_toggle.value)
    ts_plot.set_visibility(state["show_timeseries"])
    if state["show_timeseries"]:
      reload_timeseries()
      state["last_rendered_frame"] = -1
      render_frame()

  def on_export_gif() -> None:
    from double_pendulum.plots import export_gif_batch

    idx = int(traj_sel.value or 0)
    paths = export_gif_batch(
      state["data_root"],
      Path("figures/gifs"),
      stage=state["stage"],
      split=state["split"],
      indices=[idx],
      stride=max(1, current_source().n_frames() // 60),
    )
    ui.notify(f"Wrote {paths[0]}" if paths else "Nothing exported")

  stage_sel.on_value_change(lambda _: on_pool_change())
  split_sel.on_value_change(lambda _: on_pool_change())
  traj_sel.on_value_change(lambda _: on_pool_change())
  mode_sel.on_value_change(lambda _: on_mode_change())
  play_btn.on_click(toggle_play)
  reset_btn.on_click(on_reset)
  trail_toggle.on_value_change(lambda _: on_trail())
  timeseries_toggle.on_value_change(lambda _: on_timeseries_toggle())
  speed_slider.on_value_change(lambda _: on_speed_change())
  frame_scrub.on_value_change(lambda _: on_scrub_change())
  on_speed_change()
  gif_btn.on_click(on_export_gif)

  refresh_meta(meta_label)
  reload_timeseries()
  sync_scrub_bounds()
  render_frame()

  def tick() -> None:
    if state["running"]:
      src = current_source()
      n = src.n_frames()
      if n > 1:
        dt = sim_dt()
        state["frame_accum"] += (1.0 / UI_HZ) / dt * state["playback_speed"]
        step = int(state["frame_accum"])
        if step > 0:
          state["frame_accum"] -= step
          state["frame"] = (state["frame"] + step) % n
    render_frame()

  ui.timer(1 / UI_HZ, tick)


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
