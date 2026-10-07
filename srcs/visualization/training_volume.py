"""Training throughput helpers for eval figures (steps, batch, trajectories)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from srcs.model.train_data import subsampled_points_per_trajectory
from srcs.simulation.data import open_pool
from srcs.utils.json_io import read_json


def epoch_x(rows: list[dict]) -> list:
  return [r.get("epoch", r.get("global_step", i)) for i, r in enumerate(rows)]


def cumulative_train_seconds_series(rows: list[dict]) -> tuple[list, NDArray[np.float64]]:
  """Epoch x-axis and cumulative wall time (s) from ``epoch_seconds`` rows."""
  if not rows:
    return [], np.zeros(0, dtype=np.float64)
  x = epoch_x(rows)
  per_epoch = np.zeros(len(rows), dtype=np.float64)
  for i, row in enumerate(rows):
    raw = row.get("epoch_seconds")
    if raw is not None and np.isfinite(raw):
      per_epoch[i] = float(raw)
  return x, np.cumsum(per_epoch)


def points_per_trajectory_from_pool(data_root: Path | str, stage: int, stride_k: int) -> int:
  pool = open_pool(data_root, stage, "train")
  return subsampled_points_per_trajectory(pool, stride_k)


def epoch_train_step_deltas(rows: list[dict]) -> NDArray[np.float64]:
  """Optimizer steps per logged epoch (``train_steps`` or ``global_step`` diffs)."""
  if not rows:
    return np.zeros(0, dtype=np.float64)
  if any(row.get("train_steps") is not None for row in rows):
    out = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
      raw = row.get("train_steps")
      if raw is not None and np.isfinite(raw):
        out[i] = float(raw)
    return out
  out = np.zeros(len(rows), dtype=np.float64)
  prev = 0
  for i, row in enumerate(rows):
    gs = row.get("global_step")
    if gs is None:
      continue
    cur = int(gs)
    out[i] = float(max(0, cur - prev))
    prev = cur
  return out


def resolve_batch_size(rows: list[dict], summary_path: Path | str | None) -> int | None:
  for row in reversed(rows):
    raw = row.get("batch_size")
    if raw is not None:
      return int(raw)
  if summary_path is not None and Path(summary_path).is_file():
    summary = read_json(summary_path)
    if summary.get("batch_size") is not None:
      return int(summary["batch_size"])
  return None


def resolve_points_per_trajectory(
  rows: list[dict],
  summary_path: Path | str | None,
) -> int | None:
  if summary_path is not None and Path(summary_path).is_file():
    summary = read_json(summary_path)
    raw = summary.get("points_per_trajectory")
    if raw is not None:
      return int(raw)
    data_root = summary.get("data_root")
    stride_k = summary.get("subsample_stride_k")
    stages = summary.get("stages")
    if data_root and stride_k is not None and isinstance(stages, list) and stages:
      return points_per_trajectory_from_pool(data_root, int(stages[0]), int(stride_k))
  return None


def cumulative_trajectories_series(
  rows: list[dict],
  *,
  batch_size: int,
  points_per_trajectory: int,
) -> NDArray[np.float64]:
  """
  Cumulative train exposure in trajectory units.

  Each optimizer step uses up to ``batch_size`` pointwise rows; divide by
  subsampled time rows per trajectory to express volume as trajectories.
  """
  if points_per_trajectory <= 0:
    return np.zeros(len(rows), dtype=np.float64)
  steps = epoch_train_step_deltas(rows)
  samples = steps * float(batch_size)
  return np.cumsum(samples / float(points_per_trajectory))


def training_volume_scale(
  rows: list[dict],
  summary_path: Path | str | None,
) -> tuple[int, int] | None:
  """Return ``(batch_size, points_per_trajectory)`` when both resolve."""
  batch_size = resolve_batch_size(rows, summary_path)
  pptr = resolve_points_per_trajectory(rows, summary_path)
  if batch_size is None or pptr is None or pptr <= 0:
    return None
  return batch_size, pptr
