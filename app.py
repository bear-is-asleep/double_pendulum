"""Interactive double-pendulum browser UI."""

from __future__ import annotations

import math

from nicegui import ui

from double_pendulum.physics import (
  DoublePendulum,
  PendulumParams,
  PendulumState,
)

PORT = 8765
CANVAS = 640
MARGIN = 28


def _rad(deg: float) -> float:
  return math.radians(deg)


def build_svg(
  sim: DoublePendulum,
  *,
  show_trail: bool,
  width: int = CANVAS,
  height: int = CANVAS,
) -> str:
  """Render pendulum, trail, and pivot as SVG markup."""
  span = sim.params.l1 + sim.params.l2
  scale = (min(width, height) / 2 - MARGIN) / max(span, 0.1)
  cx, cy = width / 2, height / 2

  def tx(x: float, y: float) -> tuple[float, float]:
    return cx + x * scale, cy - y * scale

  x1, y1, x2, y2 = sim.positions()
  px, py = tx(0, 0)
  a1x, a1y = tx(x1, y1)
  a2x, a2y = tx(x2, y2)

  trail = sim.trail if show_trail else []
  trail_pts = " ".join(f"{tx(x, y)[0]:.1f},{tx(x, y)[1]:.1f}" for x, y in trail)
  trail_poly = (
    f'<polyline points="{trail_pts}" fill="none" '
    'stroke="#c45c26" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" opacity="0.85"/>'
    if len(trail) > 1
    else ""
  )

  return f"""
  <svg viewBox="0 0 {width} {height}" width="100%" height="100%"
       xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Double pendulum">
    <defs>
      <radialGradient id="floor" cx="50%" cy="42%" r="62%">
        <stop offset="0%" stop-color="#e8eef5"/>
        <stop offset="100%" stop-color="#c5d0dc"/>
      </radialGradient>
    </defs>
    <rect width="{width}" height="{height}" fill="url(#floor)"/>
    {trail_poly}
    <line x1="{px}" y1="{py}" x2="{a1x}" y2="{a1y}"
          stroke="#1a2a3a" stroke-width="4" stroke-linecap="round"/>
    <line x1="{a1x}" y1="{a1y}" x2="{a2x}" y2="{a2y}"
          stroke="#1a2a3a" stroke-width="4" stroke-linecap="round"/>
    <circle cx="{px}" cy="{py}" r="7" fill="#1a2a3a"/>
    <circle cx="{a1x}" cy="{a1y}" r="{8 + 4 * sim.params.m1}" fill="#2f6f8f"/>
    <circle cx="{a2x}" cy="{a2y}" r="{8 + 4 * sim.params.m2}" fill="#c45c26"/>
  </svg>
  """


def create_app() -> None:
  """Register NiceGUI page and shared simulation state."""
  sim = DoublePendulum(
    params=PendulumParams(),
    state=PendulumState(theta1=_rad(135), theta2=_rad(90)),
    dt=1 / 240,
  )
  sim.max_trail = 500
  running = {"value": True}
  show_trail = {"value": True}

  ui.add_head_html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
    <style>
      :root {
        --ink: #1a2a3a;
        --mist: #d7e0ea;
        --panel: #f4f7fa;
        --accent: #c45c26;
        --steel: #2f6f8f;
      }
      body {
        margin: 0;
        font-family: "Source Sans 3", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(1200px 700px at 15% -10%, #eef3f8 0%, transparent 55%),
          radial-gradient(900px 600px at 100% 0%, #e3ebe4 0%, transparent 50%),
          linear-gradient(165deg, #f7f9fb 0%, #d9e2ec 100%);
        min-height: 100vh;
      }
      .brand {
        font-family: Fraunces, Georgia, serif;
        font-weight: 700;
        font-size: clamp(2rem, 5vw, 3.4rem);
        letter-spacing: -0.02em;
        line-height: 1.05;
        margin: 0;
      }
      .lede {
        max-width: 34rem;
        font-size: 1.05rem;
        opacity: 0.82;
        margin: 0.6rem 0 0;
      }
      .stage {
        background: transparent;
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 18px 40px rgba(26, 42, 58, 0.12);
        border: 1px solid rgba(26, 42, 58, 0.08);
        aspect-ratio: 1 / 1;
        width: min(640px, 92vw);
      }
      .panel {
        background: color-mix(in srgb, var(--panel) 88%, white);
        border: 1px solid rgba(26, 42, 58, 0.08);
        border-radius: 16px;
        padding: 1rem 1.1rem 1.15rem;
      }
      .meta {
        font-variant-numeric: tabular-nums;
        font-size: 0.95rem;
        opacity: 0.9;
      }
      .slider-block {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        width: 100%;
      }
      .slider-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        width: 100%;
        gap: 0.5rem;
      }
      .slider-caption {
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0;
        line-height: 1.2;
      }
      .slider-value {
        margin: 0;
        opacity: 1;
        font-weight: 600;
      }
    </style>
    """
  )

  with ui.column().classes("w-full items-center q-pa-md").style(
    "max-width: 1100px; margin: 0 auto; gap: 1.25rem;"
  ):
    with ui.column().classes("w-full").style("gap: 0.25rem; padding-top: 0.5rem;"):
      ui.html('<p class="brand">Double Pendulum</p>', sanitize=False)
      ui.html(
        '<p class="lede">Two linked arms, one fixed pivot, and enough chaos '
        "to make identical starts drift apart. Nudge angles, spin rates, and "
        "masses, then watch the lower tip paint its trail.</p>",
        sanitize=False,
      )

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
            ui.label("θ₁ (degrees)").classes("slider-caption")
            theta1_val = ui.label("135").classes("meta slider-value")
          theta1 = ui.slider(min=0, max=180, value=135, step=1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("θ₂ (degrees)").classes("slider-caption")
            theta2_val = ui.label("90").classes("meta slider-value")
          theta2 = ui.slider(min=0, max=180, value=90, step=1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("ω₁ (deg/s)").classes("slider-caption")
            omega1_val = ui.label("0").classes("meta slider-value")
          omega1 = ui.slider(min=-360, max=360, value=0, step=5)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("ω₂ (deg/s)").classes("slider-caption")
            omega2_val = ui.label("0").classes("meta slider-value")
          omega2 = ui.slider(min=-360, max=360, value=0, step=5)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Mass m₁").classes("slider-caption")
            m1_val = ui.label("1.0").classes("meta slider-value")
          m1 = ui.slider(min=0, max=3.0, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Mass m₂").classes("slider-caption")
            m2_val = ui.label("1.0").classes("meta slider-value")
          m2 = ui.slider(min=0, max=3.0, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Length L₁").classes("slider-caption")
            l1_val = ui.label("1.0").classes("meta slider-value")
          l1 = ui.slider(min=0.4, max=1.6, value=1.0, step=0.1)
        with ui.column().classes("slider-block"):
          with ui.row().classes("slider-head"):
            ui.label("Length L₂").classes("slider-caption")
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
      # Several physics steps per frame keeps motion smooth at ~60 fps.
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
