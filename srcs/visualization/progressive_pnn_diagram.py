"""Schematic Progressive PNN layout at each unlocked curriculum stage."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle

from srcs.eval.run_layout import RunEvalLayout
from srcs.loader import load_model_config
from srcs.model.mlp import resolve_hidden_layer_widths
from srcs.utils.json_io import read_json
from srcs.utils.yaml_io import read_mapping
from srcs.visualization.mpl_io import prepare_out, write_fig, write_placeholder

_PROGRESSIVE_STRATEGY = "progressive"
_FROZEN_FACE = "#d8d8d8"
_FROZEN_EDGE = "#555555"
_TRAIN_FACE = "#cfe8ff"
_TRAIN_EDGE = "#1a5fb4"
_IO_FACE = "#f5f5f5"
_IO_EDGE = "#333333"
_LATERAL_COLOR = "#e66100"


def _strategy_label(data: dict[str, Any]) -> str | None:
  strategy = data.get("strategy")
  if isinstance(strategy, str) and strategy:
    return strategy
  experiment = data.get("experiment")
  if isinstance(experiment, str) and experiment:
    return experiment
  return None


def is_progressive_run(layout: RunEvalLayout) -> bool:
  """True when run artifacts mark strategy ``progressive``."""
  for path in (layout.summary, layout.run_dir / "config.yaml"):
    if not path.is_file():
      continue
    data = read_json(path) if path.suffix == ".json" else read_mapping(path)
    if _strategy_label(data) == _PROGRESSIVE_STRATEGY:
      return True
    if data.get("model_type") == "progressive_pnn":
      return True
  return layout.run_id.startswith("progressive_")


def _model_stem_from_run(layout: RunEvalLayout) -> str:
  for path in (layout.run_dir / "config.yaml", layout.summary):
    if not path.is_file():
      continue
    data = read_json(path) if path.suffix == ".json" else read_mapping(path)
    train = data.get("train")
    if isinstance(train, dict):
      model_key = train.get("model")
      if isinstance(model_key, str) and model_key:
        return model_key
    if "hidden_width" in data and "hidden_depth" in data:
      return "progressive"
  return "progressive"


_RUN_ARCH_KEYS = (
  "hidden_width",
  "hidden_width_baseline",
  "hidden_depth",
  "effective_depth",
  "input_dim",
  "output_dim",
  "use_lateral",
)


def _run_arch_overlay(layout: RunEvalLayout) -> dict[str, Any]:
  """Per-column widths and dims from the trained run (not static model YAML)."""
  for path in (layout.run_dir / "config.yaml", layout.summary):
    if not path.is_file():
      continue
    data = read_json(path) if path.suffix == ".json" else read_mapping(path)
    overlay = {k: data[k] for k in _RUN_ARCH_KEYS if k in data}
    if overlay:
      return overlay
  return {}


def _arch_cfg_from_run(layout: RunEvalLayout) -> dict[str, Any]:
  cfg = load_model_config(_model_stem_from_run(layout))
  cfg.update(_run_arch_overlay(layout))
  return cfg


def _layer_y_positions(
  n_layers: int,
  *,
  layer_h: float,
  gap: float,
) -> list[float]:
  # Bottom (input) to top (output); index 0 = input slot, 1..n = hidden, last = output.
  ys: list[float] = [0.0]
  y = layer_h + gap
  for _ in range(n_layers):
    ys.append(y)
    y += layer_h + gap
  ys.append(y)
  return ys


def _draw_box(
  ax: plt.Axes,
  x: float,
  y: float,
  w: float,
  h: float,
  label: str,
  *,
  face: str,
  edge: str,
  hatch: str | None = None,
  lw: float = 1.2,
) -> Rectangle:
  rect = Rectangle(
    (x, y),
    w,
    h,
    facecolor=face,
    edgecolor=edge,
    linewidth=lw,
    hatch=hatch,
  )
  ax.add_patch(rect)
  ax.text(
    x + w / 2,
    y + h / 2,
    label,
    ha="center",
    va="center",
    fontsize=7,
    wrap=True,
  )
  return rect


def _draw_column_stack(
  ax: plt.Axes,
  x_center: float,
  *,
  col_index: int,
  widths: list[int],
  input_dim: int,
  output_dim: int,
  frozen: bool,
  box_w: float,
  layer_h: float,
  gap: float,
) -> list[tuple[float, float]]:
  """Draw one column; return (x, y) center per hidden layer for lateral arrows."""
  ys = _layer_y_positions(len(widths), layer_h=layer_h, gap=gap)
  x0 = x_center - box_w / 2
  face = _FROZEN_FACE if frozen else _TRAIN_FACE
  edge = _FROZEN_EDGE if frozen else _TRAIN_EDGE
  hatch = "///" if frozen else None

  _draw_box(
    ax,
    x0,
    ys[0],
    box_w,
    layer_h,
    f"in\n{input_dim}",
    face=_IO_FACE,
    edge=_IO_EDGE,
  )
  hidden_centers: list[tuple[float, float]] = []
  for ell, w in enumerate(widths):
    y = ys[ell + 1]
    _draw_box(
      ax,
      x0,
      y,
      box_w,
      layer_h,
      f"h{ell}\n{w}",
      face=face,
      edge=edge,
      hatch=hatch,
    )
    hidden_centers.append((x_center, y + layer_h / 2))
    if ell < len(widths) - 1:
      ax.plot([x_center, x_center], [y + layer_h, ys[ell + 2]], color=edge, lw=1.0)
  y_out = ys[-1]
  ax.plot(
    [x_center, x_center],
    [ys[len(widths)] + layer_h, y_out],
    color=edge,
    lw=1.0,
  )
  _draw_box(
    ax,
    x0,
    y_out,
    box_w,
    layer_h,
    f"out\n{output_dim}",
    face=_IO_FACE,
    edge=_IO_EDGE,
  )
  ax.text(
    x_center,
    ys[-1] + layer_h + gap * 0.6,
    f"col {col_index}",
    ha="center",
    va="bottom",
    fontsize=8,
    fontweight="bold",
  )
  return hidden_centers


def _draw_lateral(
  ax: plt.Axes,
  xy_from: tuple[float, float],
  xy_to: tuple[float, float],
) -> None:
  arrow = FancyArrowPatch(
    xy_from,
    xy_to,
    arrowstyle="-|>",
    mutation_scale=8,
    linewidth=0.9,
    color=_LATERAL_COLOR,
    linestyle="dashed",
    connectionstyle="arc3,rad=0.08",
    zorder=0,
  )
  ax.add_patch(arrow)


def draw_pnn_snapshot_panel(
  ax: plt.Axes,
  active_max_stage: int,
  *,
  widths: list[int],
  input_dim: int,
  output_dim: int,
  use_lateral: bool,
) -> None:
  """One curriculum snapshot: columns ``0..active_max_stage``, last column trains."""
  n_columns = active_max_stage + 1
  trainable_col = active_max_stage
  col_gap = 1.4
  box_w = 0.85
  layer_h = 0.55
  gap = 0.12
  ys = _layer_y_positions(len(widths), layer_h=layer_h, gap=gap)
  y_top = ys[-1] + layer_h + 0.35

  centers_per_col: list[list[tuple[float, float]]] = []
  for j in range(n_columns):
    x_center = j * col_gap
    frozen = j < trainable_col
    centers_per_col.append(
      _draw_column_stack(
        ax,
        x_center,
        col_index=j,
        widths=widths,
        input_dim=input_dim,
        output_dim=output_dim,
        frozen=frozen,
        box_w=box_w,
        layer_h=layer_h,
        gap=gap,
      )
    )

  if use_lateral and trainable_col > 0:
    tgt = centers_per_col[trainable_col]
    for j in range(trainable_col):
      src = centers_per_col[j]
      for ell in range(len(widths)):
        x_from = src[ell][0] + box_w * 0.55
        x_to = tgt[ell][0] - box_w * 0.55
        _draw_lateral(ax, (x_from, src[ell][1]), (x_to, tgt[ell][1]))

  frozen_note = (
    f"columns 0..{trainable_col - 1} frozen"
    if trainable_col > 0
    else "column 0 training"
  )
  ax.set_title(
    f"stage {active_max_stage} unlocked\n"
    f"{n_columns} column(s); {frozen_note}",
    fontsize=9,
  )
  ax.set_xlim(-0.6, (n_columns - 1) * col_gap + 0.6)
  ax.set_ylim(-0.15, y_top)
  ax.set_aspect("equal")
  ax.axis("off")


def plot_pnn_stage_snapshots(
  cfg: dict[str, Any],
  stages: list[int],
  out_path: Path | str,
  *,
  use_lateral: bool | None = None,
) -> Path:
  """Multi-panel schematic for each ``active_max_stage`` in ``stages``."""
  out = prepare_out(out_path)
  if not stages:
    return write_placeholder(
      out,
      "no stages",
      figsize=(6, 3),
      title="PNN architecture (empty stages)",
    )
  widths = resolve_hidden_layer_widths(cfg)
  input_dim = int(cfg.get("input_dim", 8))
  output_dim = int(cfg.get("output_dim", 6))
  lateral = bool(cfg.get("use_lateral", True)) if use_lateral is None else use_lateral

  n_panels = len(stages)
  fig, axes = plt.subplots(1, n_panels, figsize=(3.8 * n_panels, 5.2))
  if n_panels == 1:
    axes_list = [axes]
  else:
    axes_list = list(axes)

  for ax, stage in zip(axes_list, stages, strict=True):
    draw_pnn_snapshot_panel(
      ax,
      int(stage),
      widths=widths,
      input_dim=input_dim,
      output_dim=output_dim,
      use_lateral=lateral,
    )

  legend_handles = [
    Rectangle((0, 0), 1, 1, facecolor=_TRAIN_FACE, edgecolor=_TRAIN_EDGE),
    Rectangle(
      (0, 0),
      1,
      1,
      facecolor=_FROZEN_FACE,
      edgecolor=_FROZEN_EDGE,
      hatch="///",
    ),
    FancyArrowPatch(
      (0, 0),
      (1, 0),
      arrowstyle="-|>",
      color=_LATERAL_COLOR,
      linestyle="dashed",
    ),
  ]
  fig.legend(
    legend_handles,
    ["trainable column", "frozen column", "lateral (if enabled)"],
    loc="lower center",
    ncol=3,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
  )
  fig.suptitle(
    "Progressive PNN (one full column per unlocked stage)",
    fontsize=11,
    y=0.98,
  )
  # Leave room for suptitle and bottom legend (plain tight_layout clips both).
  return write_fig(fig, out, tight_rect=(0.0, 0.11, 1.0, 0.90))


def render_pnn_diagram_for_run(
  layout: RunEvalLayout,
  out_dir: Path | str,
  allowed_stages: list[int] | None,
) -> list[Path]:
  """Write ``pnn_architecture_stages.png`` for progressive runs; else []."""
  if not is_progressive_run(layout):
    return []
  cfg = _arch_cfg_from_run(layout)
  stages = _stages_for_diagram(layout, allowed_stages)
  out = Path(out_dir) / "pnn_architecture_stages.png"
  return [plot_pnn_stage_snapshots(cfg, stages, out)]


def _stages_for_diagram(
  layout: RunEvalLayout,
  allowed_stages: list[int] | None,
) -> list[int]:
  if allowed_stages:
    return sorted({int(s) for s in allowed_stages})
  for path in (layout.summary, layout.run_dir / "config.yaml"):
    if not path.is_file():
      continue
    data = read_json(path) if path.suffix == ".json" else read_mapping(path)
    stages = data.get("stages")
    if isinstance(stages, list) and stages:
      ids = sorted({int(s) for s in stages})
      return ids[:3] if len(ids) > 3 else ids
  return [0, 1, 2]


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Draw Progressive PNN schematics at selected curriculum stages",
  )
  parser.add_argument(
    "--config",
    default="progressive",
    help="model config stem (default progressive)",
  )
  parser.add_argument(
    "--stages",
    default="0,1,2",
    help="comma-separated active_max_stage values (default 0,1,2)",
  )
  parser.add_argument(
    "-o",
    "--out",
    type=Path,
    default=Path("figures/pnn_architecture_stages.png"),
  )
  parser.add_argument(
    "--no-lateral",
    action="store_true",
    help="omit lateral arrows",
  )
  args = parser.parse_args(argv)
  stages = [int(s.strip()) for s in args.stages.split(",") if s.strip()]
  cfg = load_model_config(args.config)
  path = plot_pnn_stage_snapshots(
    cfg,
    stages,
    args.out,
    use_lateral=not args.no_lateral,
  )
  print(path)


if __name__ == "__main__":
  main()
