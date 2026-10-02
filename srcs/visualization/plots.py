"""Matplotlib/Pillow exports for dataset trajectories (not run metrics).

Ground-truth plots read ``TrajectoryView`` / ``TrajectorySource`` objects.
Batch helpers discover pools under a ``data_root`` via ``data.list_pools``.

Command-line entry points (run from repo root)::

  python -m srcs.visualization.plots timeseries --data-root data --stage 1
  python -m srcs.visualization.plots gif --stage 1 --split test --indices 0

Run metric figures (``srcs.visualization.eval_figures``)::

  python -m srcs.visualization.plots eval --metrics runs/.../<run_id>/metrics.jsonl

Gifs raster frames with matplotlib rather than parsing SVG, keeping dependencies
limited to what is already in ``requirements.txt``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from srcs.simulation.data import TrajectoryView, list_pools, open_pool
from srcs.physics import cartesian
from srcs.visualization.sources import GroundTruthSource, TrajectorySource
from srcs.visualization.mpl_io import prepare_out, write_fig
from srcs.visualization.viz import PendulumFrame, build_svg


def plot_trajectory_timeseries(
  view: TrajectoryView,
  out_path: Path | str,
  *,
  title: str | None = None,
) -> Path:
  """
  Write a three-row PNG for one trajectory.

  Row 1 plots sin/cos of both angles (training targets). Row 2 plots omega1, omega2.
  Row 3 plots potential, kinetic, and total energy stored in the pool.
  """
  out = prepare_out(out_path)
  t = view.t
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
  ax_sc, ax_w, ax_e = axes

  ax_sc.plot(t, view.sin_theta1, label="sin(theta1)", color="#2f6f8f")
  ax_sc.plot(t, view.cos_theta1, label="cos(theta1)", color="#1a5a72", linestyle="--")
  ax_sc.plot(t, view.sin_theta2, label="sin(theta2)", color="#c45c26")
  ax_sc.plot(t, view.cos_theta2, label="cos(theta2)", color="#8a3d18", linestyle="--")
  ax_sc.set_ylabel("sin / cos")
  ax_sc.legend(loc="upper right", fontsize=8, ncol=2)
  ax_sc.grid(True, alpha=0.3)

  ax_w.plot(t, view.omega1, label="omega1", color="#2f6f8f")
  ax_w.plot(t, view.omega2, label="omega2", color="#c45c26")
  ax_w.set_ylabel("omega (rad/s)")
  ax_w.legend(loc="upper right", fontsize=8)
  ax_w.grid(True, alpha=0.3)

  ax_e.plot(t, view.potential, label="PE", color="#6b4c9a")
  ax_e.plot(t, view.kinetic, label="KE", color="#2f6f8f")
  ax_e.plot(t, view.energy, label="E", color="#1a2a3a", linewidth=1.2)
  ax_e.set_xlabel("t (s)")
  ax_e.set_ylabel("energy")
  ax_e.legend(loc="upper right", fontsize=8)
  ax_e.grid(True, alpha=0.3)

  if title:
    fig.suptitle(title)
  return write_fig(fig, out)


def plot_pools_timeseries(
  data_root: Path | str,
  out_dir: Path | str,
  *,
  stage: int | None = None,
  split: str = "test",
  traj_index: int = 0,
) -> list[Path]:
  """
  Scan ``data_root`` for ``stage*_{split}.npz`` files and plot ``traj_index``.

  If ``stage`` is None, every stage with that split is plotted. Skips empty pools.
  """
  root = Path(data_root)
  out = Path(out_dir)
  written: list[Path] = []
  for st, sp, _ in list_pools(root):
    if sp != split:
      continue
    if stage is not None and st != stage:
      continue
    pool = open_pool(root, st, sp)
    if pool.n_traj == 0:
      continue
    idx = min(traj_index, pool.n_traj - 1)
    view = pool.get_traj(idx)
    path = out / f"timeseries_stage{st}_{split}_traj{idx}.png"
    plot_trajectory_timeseries(
      view,
      path,
      title=f"stage {st} {split} traj {idx}",
    )
    written.append(path)
  return written


def render_frame_rgba(
  source: TrajectorySource,
  k: int,
  *,
  show_trail: bool,
  size: int = 320,
) -> NDArray[np.uint8]:
  """
  Raster one frame to RGB uint8 (height, width, 3).

  Used by ``write_trajectory_gif``; geometry matches ``viz.build_svg`` but
  uses matplotlib artists instead of SVG markup.
  """
  p = source.params()
  state = source.frame_state(k)
  trail = source.tip_trail(k) if show_trail else []
  span = max(p.l1 + p.l2, 0.1)
  margin = 0.15 * span
  fig, ax = plt.subplots(figsize=(4, 4), dpi=size // 4)
  ax.set_aspect("equal")
  ax.set_xlim(-span - margin, span + margin)
  ax.set_ylim(-span - margin, span + margin)
  ax.axis("off")
  ax.set_facecolor("#e8eef5")

  if len(trail) > 1:
    xs, ys = zip(*trail)
    ax.plot(xs, ys, color="#c45c26", linewidth=1.5, alpha=0.85)

  x1, y1, x2, y2 = cartesian(state, p)
  ax.plot([0, x1, x2], [0, y1, y2], color="#1a2a3a", linewidth=3, solid_capstyle="round")
  ax.scatter([0], [0], s=40, color="#1a2a3a", zorder=5)
  ax.scatter([x1], [y1], s=30 + 80 * p.m1, color="#2f6f8f", zorder=5)
  ax.scatter([x2], [y2], s=30 + 80 * p.m2, color="#c45c26", zorder=5)

  fig.canvas.draw()
  rgba = np.asarray(fig.canvas.buffer_rgba())
  plt.close(fig)
  return rgba[:, :, :3].copy()


def write_trajectory_gif(
  source: TrajectorySource,
  out_path: Path | str,
  *,
  stride: int = 1,
  show_trail: bool = True,
  frame_size: int = 320,
  duration_ms: int = 50,
) -> Path:
  """
  Encode an animated GIF by sampling every ``stride``-th frame.

  ``duration_ms`` is the display time per frame in milliseconds (Pillow convention).
  """
  out = prepare_out(out_path)
  frames: list[Image.Image] = []
  n = source.n_frames()
  for k in range(0, n, max(1, stride)):
    rgb = render_frame_rgba(source, k, show_trail=show_trail, size=frame_size)
    frames.append(Image.fromarray(rgb))
  if not frames:
    raise ValueError("no frames to write")
  frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=duration_ms,
    loop=0,
  )
  return out


def export_gif_batch(
  data_root: Path | str,
  out_dir: Path | str,
  *,
  stage: int,
  split: str,
  indices: list[int],
  stride: int = 2,
) -> list[Path]:
  """Convenience wrapper: open one pool and export gifs for listed trajectory indices."""
  pool = open_pool(data_root, stage, split)
  out = Path(out_dir)
  paths: list[Path] = []
  for i in indices:
    if i < 0 or i >= pool.n_traj:
      continue
    src = GroundTruthSource(pool.get_traj(i), mode="stored")
    path = out / f"pendulum_stage{stage}_{split}_traj{i}.gif"
    write_trajectory_gif(src, path, stride=stride)
    paths.append(path)
  return paths


def svg_smoke_for_source(source: TrajectorySource, k: int = 0) -> str:
  """Thin wrapper used by tests to exercise ``build_svg`` without NiceGUI."""
  frame = PendulumFrame(
    params=source.params(),
    state=source.frame_state(k),
    trail=source.tip_trail(k),
  )
  return build_svg(frame, show_trail=True)


def _cli_gif() -> None:
  parser = argparse.ArgumentParser(description="Export pendulum gifs from data pools")
  parser.add_argument("--data-root", type=Path, default=Path("data"))
  parser.add_argument("--out-dir", type=Path, default=Path("figures/gifs"))
  parser.add_argument("--stage", type=int, required=True)
  parser.add_argument("--split", default="test")
  parser.add_argument("--indices", type=int, nargs="+", default=[0])
  parser.add_argument("--stride", type=int, default=2)
  args = parser.parse_args()
  paths = export_gif_batch(
    args.data_root,
    args.out_dir,
    stage=args.stage,
    split=args.split,
    indices=args.indices,
    stride=args.stride,
  )
  for p in paths:
    print(p)


def _cli_timeseries() -> None:
  parser = argparse.ArgumentParser(description="Plot trajectory time series from pools")
  parser.add_argument("--data-root", type=Path, default=Path("data"))
  parser.add_argument("--out-dir", type=Path, default=Path("figures/timeseries"))
  parser.add_argument("--stage", type=int, default=None)
  parser.add_argument("--split", default="test")
  parser.add_argument("--traj", type=int, default=0)
  args = parser.parse_args()
  paths = plot_pools_timeseries(
    args.data_root,
    args.out_dir,
    stage=args.stage,
    split=args.split,
    traj_index=args.traj,
  )
  for p in paths:
    print(p)


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 2:
    print("usage: python -m srcs.visualization.plots gif|timeseries|eval ...")
    raise SystemExit(2)
  cmd = sys.argv.pop(1)
  if cmd == "gif":
    _cli_gif()
  elif cmd == "timeseries":
    _cli_timeseries()
  elif cmd == "eval":
    from srcs.visualization.eval_figures import main as eval_figures_main
    eval_figures_main()
  else:
    print(f"unknown command: {cmd}")
    raise SystemExit(2)
