"""SVG rendering and shared browser styling for pendulum UIs.

Coordinate system matches ``physics.cartesian``: pivot at the origin, +y upward,
theta = 0 hanging down. SVG y is flipped when mapping physics coordinates to pixels.

Typical usage::

  from double_pendulum.viz import build_svg, PendulumFrame, SHARED_HEAD_HTML

  frame = PendulumFrame(params=p, state=s, trail=[(x2, y2), ...])
  html = build_svg(frame, show_trail=True)
  # inject html into NiceGUI via ui.html(..., sanitize=False)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from double_pendulum.physics import (
  DoublePendulum,
  PendulumParams,
  PendulumState,
  cartesian,
)

# Default square viewport for the animation stage (matches CSS ``.stage`` width).
CANVAS = 640
# Padding inside the viewBox so bobs are not clipped at max extension.
MARGIN = 28


@dataclass(frozen=True)
class PendulumFrame:
  """
  Everything needed to draw one animation frame.

  ``trail`` holds lower-bob positions ``(x2, y2)`` in physics coordinates,
  oldest first. The live simulator appends to ``DoublePendulum.trail``; dataset
  playback builds the prefix ``0..k`` via ``TrajectorySource.tip_trail(k)``.
  """

  params: PendulumParams
  state: PendulumState
  trail: Sequence[tuple[float, float]] = ()

  @classmethod
  def from_sim(
    cls,
    sim: DoublePendulum,
    *,
    trail: Sequence[tuple[float, float]] | None = None,
  ) -> PendulumFrame:
    """Snapshot the current interactive simulator state."""
    t = trail if trail is not None else sim.trail
    return cls(params=sim.params, state=sim.state, trail=t)


def build_svg(
  drawable: DoublePendulum | PendulumFrame,
  *,
  show_trail: bool,
  width: int = CANVAS,
  height: int = CANVAS,
  overlay: PendulumFrame | None = None,
) -> str:
  """
  Return a complete SVG document string for one pendulum configuration.

  Parameters
  ----------
  drawable
      Either a live ``DoublePendulum`` or a pre-built ``PendulumFrame``.
  show_trail
      When True and the trail has at least two points, draw the tip polyline.
  overlay
      Optional second pendulum (e.g. neural surrogate). Drawn with dashed arms
      and a distinct palette so it sits on top of the primary trace.

  Bob radius scales slightly with mass so unequal masses are visible at a glance.
  """
  if isinstance(drawable, DoublePendulum):
    frame = PendulumFrame.from_sim(drawable)
  else:
    frame = drawable

  # Fit both arm lengths into the square viewBox with uniform scale.
  span = frame.params.l1 + frame.params.l2
  scale = (min(width, height) / 2 - MARGIN) / max(span, 0.1)
  cx, cy = width / 2, height / 2

  def tx(x: float, y: float) -> tuple[float, float]:
    # Physics +y up; SVG +y down.
    return cx + x * scale, cy - y * scale

  def arm_svg(f: PendulumFrame, *, dashed: bool) -> str:
    x1, y1, x2, y2 = cartesian(f.state, f.params)
    px, py = tx(0, 0)
    a1x, a1y = tx(x1, y1)
    a2x, a2y = tx(x2, y2)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    stroke = "#6b4c9a" if dashed else "#1a2a3a"
    bob1 = "#9b7bb8" if dashed else "#2f6f8f"
    bob2 = "#e07b4a" if dashed else "#c45c26"
    trail_poly = ""
    if show_trail and len(f.trail) > 1:
      trail_pts = " ".join(f"{tx(x, y)[0]:.1f},{tx(x, y)[1]:.1f}" for x, y in f.trail)
      trail_poly = (
        f'<polyline points="{trail_pts}" fill="none" '
        f'stroke="{bob2}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" opacity="0.85"/>'
      )
    return f"""
    {trail_poly}
    <line x1="{px}" y1="{py}" x2="{a1x}" y2="{a1y}"
          stroke="{stroke}" stroke-width="4" stroke-linecap="round"{dash}/>
    <line x1="{a1x}" y1="{a1y}" x2="{a2x}" y2="{a2y}"
          stroke="{stroke}" stroke-width="4" stroke-linecap="round"{dash}/>
    <circle cx="{px}" cy="{py}" r="7" fill="{stroke}"/>
    <circle cx="{a1x}" cy="{a1y}" r="{8 + 4 * f.params.m1}" fill="{bob1}"/>
    <circle cx="{a2x}" cy="{a2y}" r="{8 + 4 * f.params.m2}" fill="{bob2}"/>
    """

  body = arm_svg(frame, dashed=False)
  if overlay is not None:
    body += arm_svg(overlay, dashed=True)

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
    {body}
  </svg>
  """


# Injected once per page by NiceGUI apps. Class names ``brand``, ``stage``,
# ``panel``, etc. are referenced from Python when building the layout.
SHARED_HEAD_HTML = """
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
    width: 100%;
    max-width: 640px;
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
