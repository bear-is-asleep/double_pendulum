"""Matplotlib/Pillow plotting and run-metric figures.

Ground-truth plots read ``TrajectoryView`` / ``TrajectorySource`` objects.
Batch helpers discover pools under a ``data_root`` via ``data.list_pools``.

Command-line entry points (run from repo root)::

  python -m double_pendulum.plots timeseries --data-root data --stage 1
  python -m double_pendulum.plots gif --stage 1 --split test --indices 0
  python -m double_pendulum.plots eval --metrics path/to/metrics.jsonl --out-dir figures/eval

The ``eval`` subcommand renders training curves, error-vs-time, and per-stage
summary bars from JSON/JSONL/NPZ files produced during training and evaluation.
It tolerates missing or empty inputs (placeholder axes) so tests can run without
a full training run.

Gifs raster frames with matplotlib rather than parsing SVG, keeping dependencies
limited to what is already in ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from double_pendulum.data import TrajectoryView, list_pools, open_pool
from double_pendulum.physics import cartesian
from double_pendulum.sources import GroundTruthSource, TrajectorySource
from double_pendulum.viz import PendulumFrame, build_svg


def energy_drift_scalar(energy: NDArray[np.float64]) -> float:
  """Same metric as ``GroundTruthSource.energy_drift_max`` for raw energy arrays."""
  if energy.size == 0:
    return 0.0
  return float(np.max(np.abs(energy - energy[0])))


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
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
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
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  return out


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
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
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


def load_metrics_jsonl(path: Path | str) -> list[dict]:
  """Parse newline-delimited JSON training logs; returns [] if file missing or empty."""
  rows: list[dict] = []
  p = Path(path)
  if not p.is_file() or p.stat().st_size == 0:
    return rows
  with p.open(encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


def plot_training_curves(
  metrics_path: Path | str,
  out_path: Path | str,
  *,
  val_key: str = "val_loss",
  train_key: str = "train_loss",
) -> Path:
  """
  Plot ``train_key`` and ``val_key`` vs epoch (or ``global_step`` fallback).

  Missing keys are skipped. An empty log file still writes a PNG with a message.
  """
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  rows = load_metrics_jsonl(metrics_path)
  fig, ax = plt.subplots(figsize=(8, 4))
  if not rows:
    ax.text(0.5, 0.5, "no metrics", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("training curves (empty fixture)")
  else:
    x = [r.get("epoch", r.get("global_step", i)) for i, r in enumerate(rows)]
    if any(train_key in r for r in rows):
      y_train = [r.get(train_key) for r in rows]
      ax.plot(x, y_train, label=train_key, color="#2f6f8f")
    if any(val_key in r for r in rows):
      y_val = [r.get(val_key) for r in rows]
      ax.plot(x, y_val, label=val_key, color="#c45c26")
    ax.set_xlabel("epoch / step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  return out


def plot_error_vs_t(
  error_npz: Path | str,
  out_path: Path | str,
  *,
  stage: int | None = None,
) -> Path:
  """
  Plot mean prediction loss vs time.

  NPZ layout:
    * ``t`` - shape ``(n_t,)``
    * ``error`` - shape ``(n_t,)`` for a single curve, or ``(n_stage, n_t)`` for one
      curve per curriculum stage (row ``s`` corresponds to stage ``s + 1``).
  """
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  p = Path(error_npz)
  fig, ax = plt.subplots(figsize=(8, 4))
  if not p.is_file():
    ax.text(0.5, 0.5, "missing error file", ha="center", va="center", transform=ax.transAxes)
  else:
    data = np.load(p)
    t = data["t"]
    err = data["error"]
    if err.ndim == 2:
      if stage is not None:
        # stage id 1..5 in row index stage-1
        row = int(stage) - 1
        if 0 <= row < err.shape[0]:
          ax.plot(t, err[row], label=f"stage {stage}")
        else:
          ax.plot(t, err[0], label="stage 1 (fallback)")
      else:
        for i in range(err.shape[0]):
          ax.plot(t, err[i], label=f"stage {i + 1}", alpha=0.85)
    else:
      ax.plot(t, err, label="mean error")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  return out


def plot_summary_stages(summary_path: Path | str, out_path: Path | str) -> Path:
  """
  Bar chart of per-stage MSE from a run ``summary.json``.

  Looks for ``per_stage_mse`` or ``val_per_stage`` mapping stage id to scalar loss.
  """
  out = Path(out_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(8, 4))
  p = Path(summary_path)
  if not p.is_file():
    ax.text(0.5, 0.5, "no summary", ha="center", va="center", transform=ax.transAxes)
  else:
    with p.open(encoding="utf-8") as f:
      summary = json.load(f)
    stages = summary.get("per_stage_mse") or summary.get("val_per_stage") or {}
    if isinstance(stages, dict) and stages:
      keys = sorted(stages.keys(), key=lambda k: int(str(k).replace("stage", "")) if str(k).replace("stage", "").isdigit() else k)
      vals = [stages[k] for k in keys]
      ax.bar([str(k) for k in keys], vals, color="#2f6f8f")
      ax.set_ylabel("MSE")
      ax.set_xlabel("stage")
    else:
      ax.text(0.5, 0.5, "no per_stage_mse", ha="center", va="center", transform=ax.transAxes)
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  return out


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


def _cli_run_plots() -> None:
  parser = argparse.ArgumentParser(description="Phase C training/eval figures")
  parser.add_argument("--metrics", type=Path, default=Path("tests/fixtures/metrics.jsonl"))
  parser.add_argument("--error-npz", type=Path, default=Path("tests/fixtures/error_vs_t.npz"))
  parser.add_argument("--summary", type=Path, default=Path("tests/fixtures/summary.json"))
  parser.add_argument("--out-dir", type=Path, default=Path("figures/eval"))
  args = parser.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)
  print(plot_training_curves(args.metrics, args.out_dir / "training.png"))
  print(plot_error_vs_t(args.error_npz, args.out_dir / "error_vs_t.png"))
  print(plot_summary_stages(args.summary, args.out_dir / "summary_stages.png"))


if __name__ == "__main__":
  import sys

  if len(sys.argv) < 2:
    print("usage: python -m double_pendulum.plots gif|timeseries|eval ...")
    raise SystemExit(2)
  cmd = sys.argv.pop(1)
  if cmd == "gif":
    _cli_gif()
  elif cmd == "timeseries":
    _cli_timeseries()
  elif cmd == "eval":
    _cli_run_plots()
  else:
    print(f"unknown command: {cmd}")
    raise SystemExit(2)
