"""Matplotlib figures from training logs and frozen test-pool eval artifacts.

Command-line (repo root)::

  python -m srcs.visualization.eval_figures --metrics runs/.../<run_id>/metrics.jsonl
  python -m srcs.visualization.plots eval --metrics runs/.../metrics.jsonl

Writes ``figures/<run_id>/`` by default. Sibling ``summary.json`` and
``eval_test/*.npz`` are auto-discovered when present. Missing inputs get
placeholder PNGs so CI can smoke without a full run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from numpy.typing import NDArray

from srcs.eval.channels import CHANNEL_KEYS, CHANNEL_MAE_PANELS
from srcs.eval.run_layout import RunEvalLayout, infer_val_key
from srcs.utils.json_io import load_jsonl, read_json
from srcs.utils.paths import ensure_dir
from srcs.visualization.mpl_io import prepare_out, write_fig, write_placeholder
from srcs.visualization.mpl_plots import (
  MODEL_COMPARE_COLORS,
  make_rect_grid,
  step_series,
)


def epoch_x(rows: list[dict]) -> list:
  return [r.get("epoch", r.get("global_step", i)) for i, r in enumerate(rows)]


_epoch_x = epoch_x


def as_stage_rows(values: NDArray) -> NDArray[np.float64]:
  arr = np.asarray(values, dtype=np.float64)
  if arr.ndim == 1:
    return arr.reshape(1, -1)
  return arr


_as_stage_rows = as_stage_rows
_step_series = step_series


def parse_stage_id(key: object) -> int:
  text = str(key)
  if text.startswith("stage"):
    suffix = text.removeprefix("stage")
    if suffix.isdigit():
      return int(suffix)
  return int(text)


_parse_stage_id = parse_stage_id


def stage_ids_from_npz(data: np.lib.npyio.NpzFile, n_rows: int) -> NDArray[np.int64]:
  if "stage_ids" in data.files:
    return np.asarray(data["stage_ids"], dtype=np.int64).reshape(-1)
  return np.arange(n_rows, dtype=np.int64)


_stage_ids_from_npz = stage_ids_from_npz


def _filter_stage_dict(
  stages_map: dict,
  allowed: list[int] | None,
) -> tuple[list[str], list[float]]:
  if not stages_map:
    return [], []
  items: list[tuple[int, str, float]] = []
  for k, v in stages_map.items():
    sid = _parse_stage_id(k)
    if allowed is not None and sid not in allowed:
      continue
    items.append((sid, str(sid), float(v)))
  items.sort(key=lambda x: x[0])
  return [lab for _, lab, _ in items], [val for _, _, val in items]


_STAGE_MIX_COLORS = MODEL_COMPARE_COLORS
_STAGE_MIX_OVERLAY_ALPHA = 0.38


def _draw_stage_mix_stack(
  ax,
  x: list,
  stages: list[int],
  mat: NDArray[np.float64],
  *,
  alpha: float = _STAGE_MIX_OVERLAY_ALPHA,
) -> None:
  colors = [_STAGE_MIX_COLORS[i % len(_STAGE_MIX_COLORS)] for i in range(len(stages))]
  labels = [f"stage {s}" for s in stages]
  ax.stackplot(x, *mat, labels=labels, colors=colors, alpha=alpha, zorder=1)


def plot_training_curves(
  metrics_path: Path | str,
  out_path: Path | str,
  *,
  val_key: str = "val_loss",
  train_key: str = "train_loss",
  rows: list[dict] | None = None,
) -> Path:
  """Train loss as a line; validation as scatter. Curriculum mix shaded behind."""
  out = prepare_out(out_path)
  if rows is None:
    rows = load_jsonl(metrics_path)
  if not rows:
    return write_placeholder(
      out,
      "no metrics",
      figsize=(8, 4),
      title="training curves (empty fixture)",
    )
  has_mix = metrics_has_curriculum_mix(rows)
  fig, ax = plt.subplots(figsize=(10, 5) if has_mix else (8, 4))
  x = _epoch_x(rows)
  stages: list[int] = []
  if has_mix:
    _, stages, mat = stage_fraction_series(rows)
    if stages:
      _draw_stage_mix_stack(ax, x, stages, mat)
  has_train = any(train_key in r and r.get(train_key) is not None for r in rows)
  has_val = any(val_key in r and r.get(val_key) is not None for r in rows)
  loss_vals: list[float] = []
  if has_train:
    y_train = [r.get(train_key) for r in rows]
    for v in y_train:
      if v is not None and np.isfinite(v):
        loss_vals.append(float(v))
    ax.plot(
      x,
      y_train,
      label=f"{train_key} (train)",
      color="#1a2a3a",
      linewidth=1.8,
      zorder=4,
    )
  if has_val:
    y_val = [r.get(val_key) for r in rows]
    for v in y_val:
      if v is not None and np.isfinite(v):
        loss_vals.append(float(v))
    ax.scatter(
      x,
      y_val,
      label=f"{val_key} (val)",
      color="#c45c26",
      s=28,
      zorder=5,
      edgecolors="white",
      linewidths=0.4,
    )
  if not has_train and not has_val:
    ax.text(0.5, 0.5, "no train/val keys", ha="center", va="center", transform=ax.transAxes)
  #ymax = max(loss_vals) * 1.08 if loss_vals else 1.0
  ymax = 0.4 # hard code to keep all plots on same scale
  ax.set_ylim(0.0, ymax)
  ax.set_xlabel("epoch")
  ax.set_ylabel("loss")
  ncol = min(len(stages) + 2, 4) if has_mix and stages else 1
  ax.legend(loc="upper right", fontsize=8, ncol=ncol)
  ax.grid(True, alpha=0.3)
  return write_fig(fig, out)


def metrics_has_curriculum_mix(rows: list[dict]) -> bool:
  """True when any metrics row logs non-empty ``train_stage_fraction``."""
  for row in rows:
    fr = row.get("train_stage_fraction")
    if isinstance(fr, dict) and fr:
      return True
  return False


def stage_fraction_series(
  rows: list[dict],
) -> tuple[list, list[int], NDArray[np.float64]]:
  """Epoch x-axis, sorted stage ids, and (n_stage, n_epoch) fraction matrix."""
  if not rows:
    return [], [], np.zeros((0, 0), dtype=np.float64)
  stage_set: set[int] = set()
  for row in rows:
    fr = row.get("train_stage_fraction")
    if not isinstance(fr, dict):
      continue
    for key in fr:
      stage_set.add(_parse_stage_id(key))
  stages = sorted(stage_set)
  if not stages:
    return _epoch_x(rows), [], np.zeros((0, len(rows)), dtype=np.float64)
  mat = np.zeros((len(stages), len(rows)), dtype=np.float64)
  for i, row in enumerate(rows):
    fr = row.get("train_stage_fraction")
    if not isinstance(fr, dict):
      continue
    for j, sid in enumerate(stages):
      raw = fr.get(str(sid), fr.get(sid))
      if raw is not None:
        mat[j, i] = float(raw)
  return _epoch_x(rows), stages, mat


def cumulative_train_seconds_series(rows: list[dict]) -> tuple[list, NDArray[np.float64]]:
  """Epoch x-axis and cumulative wall time (s) from ``epoch_seconds`` rows."""
  if not rows:
    return [], np.zeros(0, dtype=np.float64)
  x = _epoch_x(rows)
  per_epoch = np.zeros(len(rows), dtype=np.float64)
  for i, row in enumerate(rows):
    raw = row.get("epoch_seconds")
    if raw is not None and np.isfinite(raw):
      per_epoch[i] = float(raw)
  return x, np.cumsum(per_epoch)


def plot_cumulative_train_time(
  metrics_path: Path | str,
  out_path: Path | str,
  *,
  rows: list[dict] | None = None,
) -> Path:
  """Cumulative training wall time vs epoch from ``epoch_seconds`` in metrics."""
  out = prepare_out(out_path)
  if rows is None:
    rows = load_jsonl(metrics_path)
  if not rows:
    return write_placeholder(
      out,
      "no metrics",
      figsize=(8, 3),
      title="cumulative train time (empty fixture)",
    )
  has_timing = any(
    row.get("epoch_seconds") is not None and np.isfinite(row.get("epoch_seconds"))
    for row in rows
  )
  if not has_timing:
    return write_placeholder(
      out,
      "no epoch_seconds",
      figsize=(8, 3),
      title="cumulative train time (missing timing)",
    )
  x, cum_s = cumulative_train_seconds_series(rows)
  fig, ax = plt.subplots(figsize=(8, 3))
  ax.plot(x, cum_s, color="#2f6f8f", linewidth=1.8)
  ax.fill_between(x, 0, cum_s, alpha=0.15, color="#2f6f8f")
  ax.set_xlabel("epoch")
  ax.set_ylabel("cumulative time (s)")
  ax.set_title("cumulative training time")
  ax.grid(True, alpha=0.3)
  total = float(cum_s[-1]) if cum_s.size else 0.0
  ax.text(
    0.98,
    0.05,
    f"total {total:.1f} s",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
  )
  return write_fig(fig, out)


def plot_curriculum_stage_mix(
  metrics_path: Path | str,
  out_path: Path | str,
  *,
  rows: list[dict] | None = None,
) -> Path:
  """Stacked area of per-epoch ``train_stage_fraction`` (curriculum runs)."""
  out = prepare_out(out_path)
  if rows is None:
    rows = load_jsonl(metrics_path)
  if not rows or not metrics_has_curriculum_mix(rows):
    return write_placeholder(
      out,
      "no curriculum mix",
      figsize=(10, 4),
      title="curriculum train stage mix (empty)",
    )
  x, stages, mat = stage_fraction_series(rows)
  fig, ax = plt.subplots(figsize=(10, 4))
  _draw_stage_mix_stack(ax, x, stages, mat, alpha=0.85)
  ax.set_ylim(0.0, 1.0)
  ax.set_xlabel("epoch")
  ax.set_ylabel("train fraction")
  ax.set_title("curriculum train stage mix")
  ncol = min(len(stages), 4)
  ax.legend(loc="upper left", fontsize=8, ncol=ncol)
  ax.grid(True, alpha=0.3, axis="y")
  return write_fig(fig, out)


def plot_time_step_per_stage(
  t: NDArray[np.float64],
  values: NDArray[np.float64],
  stage_ids: NDArray[np.int64],
  *,
  ylabel: str,
  title: str,
  color: str = "#2f6f8f",
) -> plt.Figure:
  n_stage = int(values.shape[0])
  grid = make_rect_grid(n_stage, ncols_max=3)
  ymax = float(np.nanmax(values)) if values.size else 1.0
  y_hi = ymax * 1.05 if ymax > 0 else 1.0
  for idx in range(n_stage):
    ax = grid.axis_at(idx)
    sid = int(stage_ids[idx])
    _step_series(ax, t, values[idx], color=color)
    ax.set_title(f"stage {sid}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_hi)
    ax.grid(True, axis="y", alpha=0.3)
  grid.hide_unused(n_stage)
  grid.fig.suptitle(title)
  return grid.fig


def plot_error_vs_t(
  error_npz: Path | str,
  out_path: Path | str,
  *,
  stage: int | None = None,
) -> Path:
  """Step histogram per time bin; one subplot per curriculum stage."""
  out = prepare_out(out_path)
  p = Path(error_npz)
  if not p.is_file():
    return write_placeholder(out, "missing error file")

  data = np.load(p)
  t = np.asarray(data["t"], dtype=np.float64)
  err = _as_stage_rows(data["error"])
  stage_ids = _stage_ids_from_npz(data, err.shape[0])
  if stage is not None:
    mask = stage_ids == int(stage)
    if np.any(mask):
      err = err[mask]
      stage_ids = stage_ids[mask]
  fig = plot_time_step_per_stage(
    t,
    err,
    stage_ids,
    ylabel="weighted MSE",
    title="test error vs time (mean over trajectories)",
  )
  return write_fig(fig, out)


def plot_channel_mae_vs_t(
  channel_npz: Path | str,
  out_path: Path | str,
) -> Path:
  """Three-row step panels: sin theta, omega, PE/KE; subplots per stage."""
  out = prepare_out(out_path)
  p = Path(channel_npz)
  if not p.is_file():
    return write_placeholder(out, "missing channel MAE file")

  data = np.load(p)
  t = np.asarray(data["t"], dtype=np.float64)
  sample = _as_stage_rows(data["sin_theta1"])
  n_stage = sample.shape[0]
  stage_ids = _stage_ids_from_npz(data, n_stage)

  colors = ["#2f6f8f", "#c45c26", "#6b4c9a", "#059669"]

  fig, axes = plt.subplots(
    len(CHANNEL_MAE_PANELS),
    n_stage,
    figsize=(3.2 * n_stage, 3 * len(CHANNEL_MAE_PANELS)),
    squeeze=False,
  )
  for prow, (prow_title, keys) in enumerate(CHANNEL_MAE_PANELS):
    for sidx in range(n_stage):
      ax = axes[prow][sidx]
      sid = int(stage_ids[sidx])
      ymax = 0.0
      for ki, key in enumerate(keys):
        y = _as_stage_rows(data[key])[sidx]
        ymax = max(ymax, float(np.nanmax(y)))
        color = colors[ki % len(colors)]
        _step_series(ax, t, y, color=color, label=key, fill_alpha=0.12)
      if prow == 0:
        ax.set_title(f"stage {sid}")
      if sidx == 0:
        ax.set_ylabel(f"{prow_title}\nmean |error|")
        ax.legend(fontsize=7, loc="upper left")
      ax.set_xlabel("t (s)")
      ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
      ax.grid(True, axis="y", alpha=0.3)
  fig.suptitle("channel MAE vs time (step hist)")
  return write_fig(fig, out)


def plot_stage_channel_heatmap(channel_npz: Path | str, out_path: Path | str) -> Path:
  """Time-mean |error| per channel (rows) and curriculum stage (columns)."""
  out = prepare_out(out_path)
  p = Path(channel_npz)
  if not p.is_file():
    return write_placeholder(out, "missing channel file", figsize=(8, 4))

  data = np.load(p)
  rows: list[list[float]] = []
  labels: list[str] = []
  stage_ids = _stage_ids_from_npz(data, 1)
  for key in CHANNEL_KEYS:
    arr = _as_stage_rows(data[key])
    if not rows:
      stage_ids = _stage_ids_from_npz(data, arr.shape[0])
    rows.append([float(np.mean(arr[i])) for i in range(arr.shape[0])])
    labels.append(key)
  mat = np.array(rows, dtype=np.float64)
  fig, ax = plt.subplots(figsize=(8, 4))
  im = ax.imshow(mat, aspect="auto", cmap="viridis")
  ax.set_yticks(range(len(labels)), labels)
  ax.set_xticks(range(len(stage_ids)), [str(int(s)) for s in stage_ids])
  ax.set_xlabel("stage")
  ax.set_title("time-averaged mean |error|")
  thresh = float(np.median(mat)) if mat.size else 0.0
  for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
      val = mat[i, j]
      txt_color = "white" if val >= thresh else "black"
      ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=8, color=txt_color)
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  return write_fig(fig, out)


def _summary_bars(
  summary_path: Path | str | None,
  eval_summary_path: Path | str | None,
  allowed_stages: list[int] | None,
) -> tuple[dict, str, list[int] | None] | None:
  if eval_summary_path and Path(eval_summary_path).is_file():
    summary = read_json(eval_summary_path)
    stages_map = summary.get("test_stage_mse") or {}
    allowed = allowed_stages or summary.get("stage_ids")
    ylabel = "test MSE (time mean)"
  elif summary_path and Path(summary_path).is_file():
    summary = read_json(summary_path)
    stages_map = (
      summary.get("test_stage_mse")
      or summary.get("stage_val_mse")
      or summary.get("per_stage_mse")
      or summary.get("val_per_stage")
      or {}
    )
    allowed = allowed_stages or summary.get("stages")
    ylabel = "val MSE" if summary.get("stage_val_mse") else "MSE"
  else:
    return None
  allowed_list = [int(s) for s in allowed] if isinstance(allowed, list) else None
  return stages_map, ylabel, allowed_list


def stage_mse_map_from_layout(layout: RunEvalLayout) -> dict[int, float]:
  """Per-stage scalar MSE from eval_test or training summary on a run folder."""
  payload = _summary_bars(layout.summary, layout.eval_summary, None)
  if payload is None:
    raise FileNotFoundError(
      f"run {layout.run_id!r}: no eval_summary.json or summary.json",
    )
  stages_map, _, allowed_list = payload
  keys, vals = _filter_stage_dict(stages_map, allowed_list)
  return {parse_stage_id(k): v for k, v in zip(keys, vals)}


def plot_summary_stages(
  summary_path: Path | str | None,
  out_path: Path | str,
  *,
  eval_summary_path: Path | str | None = None,
  allowed_stages: list[int] | None = None,
) -> Path:
  """Bar chart of per-stage scalar MSE from eval or training summary JSON."""
  out = prepare_out(out_path)
  payload = _summary_bars(summary_path, eval_summary_path, allowed_stages)
  if payload is None:
    return write_placeholder(out, "no summary", figsize=(8, 4))
  stages_map, ylabel, allowed_list = payload
  fig, ax = plt.subplots(figsize=(8, 4))
  keys, vals = _filter_stage_dict(stages_map, allowed_list)
  if keys:
    ax.bar(keys, vals, color="#2f6f8f")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("stage")
  else:
    ax.text(0.5, 0.5, "no stage metrics", ha="center", va="center", transform=ax.transAxes)
  return write_fig(fig, out)


def render_eval_figures(
  *,
  out_dir: Path | str,
  metrics_path: Path | str | None = None,
  val_key: str = "mean_val_mse",
  train_key: str = "train_loss",
  error_npz: Path | str | None = None,
  channel_mae_npz: Path | str | None = None,
  summary_path: Path | str | None = None,
  eval_summary_path: Path | str | None = None,
) -> list[Path]:
  """Write the standard eval PNG set under ``out_dir``."""
  out = ensure_dir(out_dir)
  written: list[Path] = []
  metrics_rows: list[dict] | None = None
  if metrics_path and Path(metrics_path).is_file():
    metrics_rows = load_jsonl(metrics_path)
    written.append(
      plot_training_curves(
        metrics_path,
        out / "training.png",
        val_key=val_key,
        train_key=train_key,
        rows=metrics_rows,
      )
    )
    written.append(
      plot_cumulative_train_time(
        metrics_path,
        out / "cumulative_train_time.png",
        rows=metrics_rows,
      )
    )
  if error_npz and Path(error_npz).is_file():
    written.append(plot_error_vs_t(error_npz, out / "error_vs_t.png"))
  if channel_mae_npz and Path(channel_mae_npz).is_file():
    written.append(plot_channel_mae_vs_t(channel_mae_npz, out / "channel_mae_vs_t.png"))
    written.append(plot_stage_channel_heatmap(channel_mae_npz, out / "stage_channel_heatmap.png"))
  written.append(
    plot_summary_stages(
      summary_path,
      out / "summary_stages.png",
      eval_summary_path=eval_summary_path,
    )
  )
  return written


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Training/eval figures for one run (paths inferred from metrics.jsonl parent)",
  )
  parser.add_argument(
    "--metrics",
    type=Path,
    required=True,
    help="runs/.../<run_id>/metrics.jsonl",
  )
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=None,
    help="default: figures/<run_id>/",
  )
  parser.add_argument("--val-key", default=None, help="default: mean_val_mse or val_loss from log")
  parser.add_argument("--train-key", default="train_loss")
  parser.add_argument("--error-npz", type=Path, default=None)
  parser.add_argument("--channel-mae-npz", type=Path, default=None)
  parser.add_argument("--summary", type=Path, default=None)
  parser.add_argument("--eval-summary", type=Path, default=None)
  args = parser.parse_args(argv)

  layout = RunEvalLayout.from_metrics(args.metrics)
  rows = load_jsonl(layout.metrics)
  val_key = args.val_key or infer_val_key(rows)
  out_dir = args.out_dir or layout.default_figures_dir

  paths = render_eval_figures(
    out_dir=out_dir,
    metrics_path=layout.metrics,
    val_key=val_key,
    train_key=args.train_key,
    error_npz=args.error_npz or layout.error_npz,
    channel_mae_npz=args.channel_mae_npz or layout.channel_mae_npz,
    summary_path=args.summary or layout.summary,
    eval_summary_path=args.eval_summary or layout.eval_summary,
  )
  for p in paths:
    print(p)


if __name__ == "__main__":
  main()
