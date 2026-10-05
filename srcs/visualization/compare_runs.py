"""Overlay training and test-pool figures for multiple runs (YAML-driven).

Command-line (repo root)::

  python -m srcs.visualization.compare_runs --config configs/vis/baseline_vs_curriculum.yaml
  python -m srcs.visualization.compare_runs --name baseline_vs_curriculum
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from srcs.eval.channels import CHANNEL_MAE_PANELS
from srcs.eval.run_layout import RunEvalLayout, infer_val_key
from srcs.loader import load_vis_compare_config
from srcs.utils.json_io import load_jsonl
from srcs.utils.paths import ensure_dir, resolve_run_dir
from srcs.utils.yaml_io import read_mapping
from srcs.visualization.eval_figures import (
  as_stage_rows,
  cumulative_train_seconds_series,
  epoch_x,
  parse_stage_id,
  stage_ids_from_npz,
  stage_mse_map_from_layout,
)
from srcs.visualization.mpl_io import prepare_out, write_fig
from srcs.visualization.mpl_plots import (
  CHANNEL_LINESTYLES,
  StagedTimeSeries,
  align_staged_time_series,
  assert_compatible_time_grids,
  decorate_time_axis,
  draw_epoch_lines,
  draw_grouped_stage_bars,
  make_rect_grid,
  palette_color,
  row_for_stage,
  stage_ids_union,
  step_series,
  ymax_for_stages,
)

_DEFAULT_TRAIN_YLIM = 0.4


@dataclass(frozen=True)
class CompareModel:
  label: str
  layout: RunEvalLayout


@dataclass(frozen=True)
class CompareSpec:
  name: str
  out_dir: Path
  models: tuple[CompareModel, ...]
  val_key: str | None
  train_key: str
  train_ylim: float


def compare_spec_from_mapping(
  raw: dict,
  *,
  config_stem: str | None = None,
) -> CompareSpec:
  name = str(raw.get("name") or config_stem or "compare")
  entries = [
    CompareModel(
      str(entry["label"]),
      RunEvalLayout.from_run_dir(resolve_run_dir(entry["run_dir"])),
    )
    for entry in raw["models"]
  ]
  if len(entries) < 2:
    raise ValueError("compare config needs at least 2 models")
  out_raw = raw.get("out_dir")
  out_dir = Path(out_raw) if out_raw else Path("figures") / f"compare_{name}"
  return CompareSpec(
    name=name,
    out_dir=out_dir,
    models=tuple(entries),
    val_key=raw.get("val_key"),
    train_key=str(raw.get("train_key", "train_loss")),
    train_ylim=float(raw.get("train_ylim", _DEFAULT_TRAIN_YLIM)),
  )


def load_compare_config(path: Path | str) -> CompareSpec:
  p = Path(path)
  return compare_spec_from_mapping(read_mapping(p), config_stem=p.stem)


def load_compare_config_by_name(name: str) -> CompareSpec:
  return compare_spec_from_mapping(load_vis_compare_config(name), config_stem=name)


def _metrics_by_model(spec: CompareSpec) -> dict[str, list[dict]]:
  return {m.label: load_jsonl(m.layout.metrics) for m in spec.models}


def _resolve_val_key(spec: CompareSpec, metrics: dict[str, list[dict]]) -> str:
  if spec.val_key:
    return spec.val_key
  labels = [m.label for m in spec.models]
  key = infer_val_key(metrics[labels[0]])
  for label in labels[1:]:
    other = infer_val_key(metrics[label])
    if other != key:
      warnings.warn(
        f"val key {other!r} for {label!r} differs from {key!r}; using {key!r}",
        stacklevel=2,
      )
  return key


def _stage_ids_from_metrics(metrics: dict[str, list[dict]]) -> list[int]:
  found: set[int] = set()
  for rows in metrics.values():
    for row in rows:
      svm = row.get("stage_val_mse")
      if isinstance(svm, dict):
        for k in svm:
          found.add(parse_stage_id(k))
  return sorted(found)


def _stage_val_series(rows: list[dict], stage: int) -> list[float]:
  out: list[float] = []
  for row in rows:
    svm = row.get("stage_val_mse")
    if not isinstance(svm, dict):
      out.append(np.nan)
      continue
    raw = svm.get(str(stage), svm.get(stage))
    out.append(float(raw) if raw is not None else np.nan)
  return out


def _load_staged_error(model: CompareModel) -> StagedTimeSeries:
  path = model.layout.error_npz
  if not path.is_file():
    raise FileNotFoundError(f"model {model.label!r}: missing eval_test/error_vs_t.npz at {path}")
  data = np.load(path)
  t = np.asarray(data["t"], dtype=np.float64)
  values = as_stage_rows(data["error"])
  stage_ids = stage_ids_from_npz(data, values.shape[0])
  return StagedTimeSeries(model.label, t, values, stage_ids)


def _load_staged_channel_mae(model: CompareModel) -> tuple[StagedTimeSeries, np.lib.npyio.NpzFile]:
  path = model.layout.channel_mae_npz
  if not path.is_file():
    raise FileNotFoundError(
      f"model {model.label!r}: missing eval_test/channel_mae_vs_t.npz at {path}",
    )
  data = np.load(path)
  t = np.asarray(data["t"], dtype=np.float64)
  sample = as_stage_rows(data["sin_theta1"])
  stage_ids = stage_ids_from_npz(data, sample.shape[0])
  row = StagedTimeSeries(model.label, t, sample, stage_ids)
  return row, data


def plot_compare_training(
  spec: CompareSpec,
  out_path: Path | str,
  *,
  val_key: str,
  metrics: dict[str, list[dict]],
) -> Path:
  out = prepare_out(out_path)
  fig, ax = plt.subplots(figsize=(10, 5))
  for i, model in enumerate(spec.models):
    rows = metrics[model.label]
    if not rows:
      continue
    color = palette_color(i)
    x = epoch_x(rows)
    ax.plot(
      x,
      [r.get(spec.train_key) for r in rows],
      label=f"{model.label} ({spec.train_key})",
      color=color,
      linewidth=1.8,
    )
    y_val = [r.get(val_key) for r in rows]
    if any(v is not None for v in y_val):
      ax.scatter(
        x,
        y_val,
        label=f"{model.label} ({val_key})",
        color=color,
        s=24,
        edgecolors="white",
        linewidths=0.4,
        zorder=5,
      )
  ax.set_ylim(0.0, spec.train_ylim)
  ax.set_xlabel("epoch")
  ax.set_ylabel("loss")
  ax.set_title("training curves (compare)")
  ax.legend(loc="upper right", fontsize=8)
  ax.grid(True, alpha=0.3)
  return write_fig(fig, out)


def plot_compare_cumulative_train_time(
  spec: CompareSpec,
  out_path: Path | str,
  *,
  metrics: dict[str, list[dict]],
) -> Path:
  out = prepare_out(out_path)
  fig, ax = plt.subplots(figsize=(8, 3))
  lines: list[tuple[list, list, str, str]] = []
  for i, model in enumerate(spec.models):
    x, cum_s = cumulative_train_seconds_series(metrics[model.label])
    if x:
      lines.append((x, cum_s, model.label, palette_color(i)))
  draw_epoch_lines(ax, lines)
  ax.set_xlabel("epoch")
  ax.set_ylabel("cumulative time (s)")
  ax.set_title("cumulative training time (compare)")
  ax.legend(loc="upper left", fontsize=8)
  ax.grid(True, alpha=0.3)
  return write_fig(fig, out)


def plot_compare_stage_val_mse(
  spec: CompareSpec,
  out_path: Path | str,
  *,
  metrics: dict[str, list[dict]],
) -> Path:
  out = prepare_out(out_path)
  stages = _stage_ids_from_metrics(metrics)
  if not stages:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "no stage_val_mse in metrics", ha="center", va="center", transform=ax.transAxes)
    return write_fig(fig, out)

  grid = make_rect_grid(len(stages), ncols_max=3)
  for idx, sid in enumerate(stages):
    ax = grid.axis_at(idx)
    ax.set_title(f"stage {sid}")
    series: list[tuple[list, list, str, str]] = []
    for i, model in enumerate(spec.models):
      rows = metrics[model.label]
      series.append((epoch_x(rows), _stage_val_series(rows, sid), model.label, palette_color(i)))
    draw_epoch_lines(ax, series, linewidth=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val MSE")
    ax.grid(True, alpha=0.3)
    if idx == 0:
      ax.legend(fontsize=7, loc="upper right")
  grid.hide_unused(len(stages))
  grid.fig.suptitle("per-stage validation MSE (compare)")
  return write_fig(grid.fig, out)


def plot_compare_error_vs_t(spec: CompareSpec, out_path: Path | str) -> Path:
  out = prepare_out(out_path)
  rows = align_staged_time_series(
    [_load_staged_error(m) for m in spec.models],
    context="error_vs_t",
  )
  stages = stage_ids_union([r.stage_ids for r in rows])
  grid = make_rect_grid(len(stages), ncols_max=3)
  ymax = ymax_for_stages(rows, stages)
  for idx, sid in enumerate(stages):
    ax = grid.axis_at(idx)
    ax.set_title(f"stage {sid}")
    for mi, row in enumerate(rows):
      ridx = row_for_stage(row.stage_ids, sid)
      if ridx is None:
        continue
      step_series(
        ax,
        row.t,
        row.values[ridx],
        color=palette_color(mi),
        label=row.label,
        fill_alpha=0.1,
      )
    decorate_time_axis(ax, ylabel="weighted MSE", ymax=ymax, legend=idx == 0)
  grid.hide_unused(len(stages))
  grid.fig.suptitle("test error vs time (compare)")
  return write_fig(grid.fig, out)


def plot_compare_channel_mae_vs_t(spec: CompareSpec, out_path: Path | str) -> Path:
  out = prepare_out(out_path)
  loaded: list[tuple[StagedTimeSeries, np.lib.npyio.NpzFile]] = [
    _load_staged_channel_mae(m) for m in spec.models
  ]
  align_staged_time_series([row for row, _ in loaded], context="channel_mae_vs_t")
  stages = stage_ids_union([row.stage_ids for row, _ in loaded])
  n_stage = len(stages)
  fig, axes = plt.subplots(
    len(CHANNEL_MAE_PANELS),
    n_stage,
    figsize=(3.2 * n_stage, 3 * len(CHANNEL_MAE_PANELS)),
    squeeze=False,
  )
  for prow, (prow_title, keys) in enumerate(CHANNEL_MAE_PANELS):
    for sidx, sid in enumerate(stages):
      ax = axes[prow][sidx]
      for mi, (row, data) in enumerate(loaded):
        ridx = row_for_stage(row.stage_ids, sid)
        if ridx is None:
          continue
        color = palette_color(mi)
        for ki, key in enumerate(keys):
          y = as_stage_rows(data[key])[ridx]
          ls = CHANNEL_LINESTYLES[ki % len(CHANNEL_LINESTYLES)]
          step_series(
            ax,
            row.t,
            y,
            color=color,
            label=f"{row.label} {key}" if prow == 0 and sidx == 0 else None,
            fill_alpha=0.06,
            linestyle=ls,
          )
      if prow == 0:
        ax.set_title(f"stage {sid}")
      if sidx == 0:
        ax.set_ylabel(f"{prow_title}\nmean |error|")
      ax.set_xlabel("t (s)")
      ax.grid(True, axis="y", alpha=0.3)
      if prow == 0 and sidx == 0:
        ax.legend(fontsize=6, loc="upper left")
  fig.suptitle("channel MAE vs time (compare)")
  return write_fig(fig, out)


def plot_compare_summary_stages(spec: CompareSpec, out_path: Path | str) -> Path:
  out = prepare_out(out_path)
  per_model: list[tuple[str, dict[int, float]]] = []
  stage_set: set[int] = set()
  for model in spec.models:
    mse = stage_mse_map_from_layout(model.layout)
    per_model.append((model.label, mse))
    stage_set.update(mse.keys())
  stages = sorted(stage_set)
  if not stages:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "no stage metrics", ha="center", va="center", transform=ax.transAxes)
    return write_fig(fig, out)

  fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.2), 4))
  draw_grouped_stage_bars(ax, stages, per_model)
  ax.set_title("per-stage summary (compare)")
  ax.legend(fontsize=8)
  return write_fig(fig, out)


def render_compare_figures(spec: CompareSpec) -> list[Path]:
  out = ensure_dir(spec.out_dir)
  metrics = _metrics_by_model(spec)
  val_key = _resolve_val_key(spec, metrics)
  writers = [
    lambda: plot_compare_training(spec, out / "training.png", val_key=val_key, metrics=metrics),
    lambda: plot_compare_cumulative_train_time(spec, out / "cumulative_train_time.png", metrics=metrics),
    lambda: plot_compare_stage_val_mse(spec, out / "stage_val_mse.png", metrics=metrics),
    lambda: plot_compare_error_vs_t(spec, out / "error_vs_t.png"),
    lambda: plot_compare_channel_mae_vs_t(spec, out / "channel_mae_vs_t.png"),
    lambda: plot_compare_summary_stages(spec, out / "summary_stages.png"),
  ]
  return [fn() for fn in writers]


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Overlay eval figures for multiple runs")
  parser.add_argument("--config", type=Path, default=None, help="configs/vis/<name>.yaml or any path")
  parser.add_argument("--name", default=None, help="stem under configs/vis/ (alternative to --config)")
  args = parser.parse_args(argv)
  if args.config is not None:
    spec = load_compare_config(args.config)
  elif args.name is not None:
    spec = load_compare_config_by_name(args.name)
  else:
    parser.error("pass --config or --name")
  for p in render_compare_figures(spec):
    print(p)


if __name__ == "__main__":
  main()
