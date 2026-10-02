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


def _epoch_x(rows: list[dict]) -> list:
  return [r.get("epoch", r.get("global_step", i)) for i, r in enumerate(rows)]


def _as_stage_rows(values: NDArray) -> NDArray[np.float64]:
  arr = np.asarray(values, dtype=np.float64)
  if arr.ndim == 1:
    return arr.reshape(1, -1)
  return arr


def _step_series(
  ax,
  t: NDArray[np.float64],
  y: NDArray[np.float64],
  *,
  color: str,
  label: str | None = None,
  fill_alpha: float = 0.22,
) -> None:
  ax.fill_between(t, 0, y, step="post", alpha=fill_alpha, color=color, linewidth=0)
  ax.step(t, y, where="post", color=color, linewidth=1.0, label=label)


def _parse_stage_id(key: object) -> int:
  text = str(key)
  if text.startswith("stage"):
    suffix = text.removeprefix("stage")
    if suffix.isdigit():
      return int(suffix)
  return int(text)


def _stage_ids_from_npz(data: np.lib.npyio.NpzFile, n_rows: int) -> NDArray[np.int64]:
  if "stage_ids" in data.files:
    return np.asarray(data["stage_ids"], dtype=np.int64).reshape(-1)
  return np.arange(n_rows, dtype=np.int64)


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


def plot_training_curves(
  metrics_path: Path | str,
  out_path: Path | str,
  *,
  val_key: str = "val_loss",
  train_key: str = "train_loss",
) -> Path:
  """Train loss as a line; validation as scatter on the same axes."""
  out = prepare_out(out_path)
  rows = load_jsonl(metrics_path)
  if not rows:
    return write_placeholder(
      out,
      "no metrics",
      figsize=(8, 4),
      title="training curves (empty fixture)",
    )
  fig, ax = plt.subplots(figsize=(8, 4))
  x = _epoch_x(rows)
  has_train = any(train_key in r and r.get(train_key) is not None for r in rows)
  has_val = any(val_key in r and r.get(val_key) is not None for r in rows)
  if has_train:
    y_train = [r.get(train_key) for r in rows]
    ax.plot(x, y_train, label=f"{train_key} (train)", color="#2f6f8f", linewidth=1.5)
  if has_val:
    y_val = [r.get(val_key) for r in rows]
    ax.scatter(x, y_val, label=f"{val_key} (val)", color="#c45c26", s=28, zorder=3)
  if not has_train and not has_val:
    ax.text(0.5, 0.5, "no train/val keys", ha="center", va="center", transform=ax.transAxes)
  ax.set_xlabel("epoch")
  ax.set_ylabel("loss")
  ax.legend()
  ax.grid(True, alpha=0.3)
  return write_fig(fig, out)


def _plot_time_step_per_stage(
  t: NDArray[np.float64],
  values: NDArray[np.float64],
  stage_ids: NDArray[np.int64],
  *,
  ylabel: str,
  title: str,
  color: str = "#2f6f8f",
) -> plt.Figure:
  n_stage = int(values.shape[0])
  ncols = min(3, n_stage)
  nrows = int(np.ceil(n_stage / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
  ymax = float(np.nanmax(values)) if values.size else 1.0
  for idx in range(nrows * ncols):
    r, c = divmod(idx, ncols)
    ax = axes[r][c]
    if idx >= n_stage:
      ax.axis("off")
      continue
    sid = int(stage_ids[idx])
    y = values[idx]
    _step_series(ax, t, y, color=color)
    ax.set_title(f"stage {sid}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1.0)
    ax.grid(True, axis="y", alpha=0.3)
  fig.suptitle(title)
  return fig


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
  fig = _plot_time_step_per_stage(
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
  if metrics_path and Path(metrics_path).is_file():
    written.append(
      plot_training_curves(metrics_path, out / "training.png", val_key=val_key, train_key=train_key)
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
