"""Live double-pendulum sandbox in the browser (RK4, slider-driven ICs).

Run ``python app.py`` and open http://localhost:8765.

This app integrates the ODE on every animation tick. For trajectories loaded
from ``data/*.npz`` pools, use ``app_data.py`` on port 8766 instead.

Rendering and page styling live in ``srcs/viz.py`` so both apps stay
visually consistent. Overview: ``docs/visualization.md``.
"""

from __future__ import annotations

import math

from nicegui import ui

from srcs.physics import (
  DoublePendulum,
  PendulumParams,
  PendulumState,
)
from srcs.visualization.viz import SHARED_HEAD_HTML, build_svg

PORT = 8765


def _rad(deg: float) -> float:
  return math.radians(deg)


def create_app() -> None:
  """Register the NiceGUI page, widgets, and ~60 Hz animation timer."""
  sim = DoublePendulum(
    params=PendulumParams(),
    state=PendulumState(theta1=_rad(135), theta2=_rad(90)),
    dt=1 / 240,
  )
  sim.max_trail = 500
  running = {"value": True}
  show_trail = {"value": True}

  ui.add_head_html(SHARED_HEAD_HTML)

  with ui.column().classes("w-full items-center q-pa-md").style(
    "max-width: 1100px; margin: 0 auto; gap: 1.25rem;"
  ):
    with ui.row().classes("w-full items-start justify-center").style(
      "gap: 1.25rem; flex-wrap: wrap;"
    ):
      stage = ui.html(
        build_svg(sim, show_trail=True), sanitize=False
      ).classes("stage")

      with ui.column().classes("panel").style(
        "width: min(320px, 92vw); gap: 0.55rem;"
      ):
        status = ui.label("Running").classes("meta")
        time_label = ui.label("t = 0.00 s").classes("meta")

        with ui.row().style("gap: 0.5rem; flex-wrap: wrap;"):
          play_btn = ui.button("Pause")
          reset_btn = ui.button("Reset", color="secondary")

        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("theta1 (degrees)").classes("slider-caption")
            theta1_val = ui.label("135").classes("meta slider-value")
          theta1 = ui.slider(min=0, max=180, value=135, step=1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("theta2 (degrees)").classes("slider-caption")
            theta2_val = ui.label("90").classes("meta slider-value")
          theta2 = ui.slider(min=0, max=180, value=90, step=1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("omega1 (deg/s)").classes("slider-caption")
            omega1_val = ui.label("0").classes("meta slider-value")
          omega1 = ui.slider(min=-360, max=360, value=0, step=5)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("omega2 (deg/s)").classes("slider-caption")
            omega2_val = ui.label("0").classes("meta slider-value")
          omega2 = ui.slider(min=-360, max=360, value=0, step=5)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Mass m1").classes("slider-caption")
            m1_val = ui.label("1.0").classes("meta slider-value")
          m1 = ui.slider(min=0, max=3.0, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Mass m2").classes("slider-caption")
            m2_val = ui.label("1.0").classes("meta slider-value")
          m2 = ui.slider(min=0, max=3.0, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Length L1").classes("slider-caption")
            l1_val = ui.label("1.0").classes("meta slider-value")
          l1 = ui.slider(min=0.4, max=1.6, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Length L2").classes("slider-caption")
            l2_val = ui.label("1.0").classes("meta slider-value")
          l2 = ui.slider(min=0.4, max=1.6, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Gravity g").classes("slider-caption")
            g_val = ui.label("9.8").classes("meta slider-value")
          g = ui.slider(min=0.0, max=20.0, value=9.81, step=0.1)
        trail_toggle = ui.switch("Show trail", value=True)

  def apply_params_from_ui() -> None:
    sim.params = PendulumParams(
      m1=float(m1.value),
      m2=float(m2.value),
      l1=float(l1.value),
      l2=float(l2.value),
      g=float(g.value),
    )

  def sync_state_from_ui() -> None:
    apply_params_from_ui()
    sim.reset(
      PendulumState(
        theta1=_rad(float(theta1.value)),
        theta2=_rad(float(theta2.value)),
        omega1=_rad(float(omega1.value)),
        omega2=_rad(float(omega2.value)),
      )
    )

  def toggle_play() -> None:
    running["value"] = not running["value"]
    play_btn.text = "Pause" if running["value"] else "Play"
    status.text = "Running" if running["value"] else "Paused"

  def on_reset() -> None:
    sync_state_from_ui()
    stage.set_content(build_svg(sim, show_trail=show_trail["value"]))
    time_label.text = f"t = {sim.time:.2f} s"

  def on_trail_toggle() -> None:
    show_trail["value"] = bool(trail_toggle.value)
    if not show_trail["value"]:
      sim.trail.clear()

  play_btn.on_click(toggle_play)
  reset_btn.on_click(on_reset)
  trail_toggle.on_value_change(lambda _: on_trail_toggle())

  def bind_slider(
    slider,
    value_label,
    fmt,
  ) -> None:
    def on_change(_: object) -> None:
      value_label.text = fmt(float(slider.value))
      on_reset()

    slider.on_value_change(on_change)
    value_label.text = fmt(float(slider.value))

  bind_slider(theta1, theta1_val, lambda v: f"{v:.0f}")
  bind_slider(theta2, theta2_val, lambda v: f"{v:.0f}")
  bind_slider(omega1, omega1_val, lambda v: f"{v:.0f}")
  bind_slider(omega2, omega2_val, lambda v: f"{v:.0f}")
  bind_slider(m1, m1_val, lambda v: f"{v:.1f}")
  bind_slider(m2, m2_val, lambda v: f"{v:.1f}")
  bind_slider(l1, l1_val, lambda v: f"{v:.1f}")
  bind_slider(l2, l2_val, lambda v: f"{v:.1f}")
  bind_slider(g, g_val, lambda v: f"{v:.1f}")

  def tick() -> None:
    if running["value"]:
      sim.step(4, record_trail=show_trail["value"])
    stage.set_content(build_svg(sim, show_trail=show_trail["value"]))
    time_label.text = f"t = {sim.time:.2f} s"

  ui.timer(1 / 60, tick)


def main() -> None:
  create_app()
  ui.run(
    title="Double Pendulum",
    host="0.0.0.0",
    port=PORT,
    reload=False,
    show=False,
  )


if __name__ in {"__main__", "__mp_main__"}:
  main()
