"""Theory plots for adaptive curriculum inlet (no training run required).

Command-line (repo root)::

  python -m srcs.visualization.curriculum_inlet_plots --model curriculum
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from numpy.typing import NDArray

from srcs.loader import load_model_config
from srcs.train.curriculum_mix import (
  AdaptiveInletKnobs,
  exp_template_fractions,
  inlet_knobs_from_cfg,
  step_adaptive_inlet_fractions,
  validate_curriculum_cfg,
)
from srcs.visualization.mpl_io import prepare_out, write_fig
from srcs.visualization.mpl_plots import MODEL_COMPARE_COLORS


def _stage_colors(stages: list[int]) -> list[str]:
  return [MODEL_COMPARE_COLORS[i % len(MODEL_COMPARE_COLORS)] for i in range(len(stages))]


def _chunk_for_delta(delta: float, knobs: AdaptiveInletKnobs) -> float:
  if delta <= knobs.min_delta:
    return 0.0
  return min(knobs.max_chunk, knobs.gain * delta)


def fractions_after_one_step(
  knobs: AdaptiveInletKnobs,
  *,
  last_stage: int,
  active_max: int,
  prev_fr: dict[int, float],
  wval_delta: float,
) -> dict[int, float]:
  out = step_adaptive_inlet_fractions(
    prev_fr,
    active_max,
    last_stage,
    wval_delta,
    knobs,
  )
  return out.fractions


def sweep_delta_response(
  knobs: AdaptiveInletKnobs,
  *,
  last_stage: int,
  active_max: int,
  prev_fr: dict[int, float],
  delta_max: float,
  n_points: int,
) -> tuple[NDArray[np.float64], dict[int, NDArray[np.float64]], NDArray[np.float64]]:
  """Per-stage fractions and inlet chunk vs one-step wval drop."""
  deltas = np.linspace(0.0, float(delta_max), int(n_points))
  stage_ids = list(range(0, last_stage + 1))
  fr_mat = {s: np.zeros(len(deltas), dtype=np.float64) for s in stage_ids}
  chunks = np.zeros(len(deltas), dtype=np.float64)
  for i, delta in enumerate(deltas):
    fr = fractions_after_one_step(
      knobs,
      last_stage=last_stage,
      active_max=active_max,
      prev_fr=prev_fr,
      wval_delta=float(delta),
    )
    chunks[i] = _chunk_for_delta(float(delta), knobs)
    for s in stage_ids:
      fr_mat[s][i] = float(fr.get(s, 0.0))
  return deltas, fr_mat, chunks


def simulate_cumulative_inlet(
  knobs: AdaptiveInletKnobs,
  *,
  last_stage: int,
  per_epoch_delta: float,
  max_steps: int,
  extend_until_saturated: bool = True,
) -> tuple[NDArray[np.float64], list[int], NDArray[np.float64]]:
  """
  Repeat identical qualifying wval drops; x is cumulative applied val decrease.

  Optionally keep stepping until ``active_max == last_stage`` and frontier stage
  fraction nears the exp template (up to ``max_steps`` extra).
  """
  fr: dict[int, float] = {0: 1.0}
  active_max = 0
  cum = 0.0
  cum_points: list[float] = [0.0]
  fr_rows: list[dict[int, float]] = [dict(fr)]
  target_final = exp_template_fractions(last_stage, knobs.decay_lambda)
  target_frontier = float(target_final.get(last_stage, 1.0))

  def apply_synthetic_step() -> None:
    nonlocal fr, active_max, cum
    delta = float(per_epoch_delta)
    out = step_adaptive_inlet_fractions(
      fr,
      active_max,
      last_stage,
      delta,
      knobs,
    )
    fr = dict(out.fractions)
    active_max = out.active_max_stage
    if out.inlet_chunk > 0.0 and delta > knobs.min_delta:
      cum += delta
    cum_points.append(cum)
    fr_rows.append(dict(fr))

  for _ in range(int(max_steps)):
    apply_synthetic_step()

  extra_cap = int(max_steps) if extend_until_saturated else 0
  for _ in range(extra_cap):
    if active_max >= last_stage and float(fr.get(last_stage, 0.0)) >= 0.95 * target_frontier:
      break
    apply_synthetic_step()

  stage_set: set[int] = set()
  for row in fr_rows:
    stage_set.update(row.keys())
  stages = sorted(stage_set)
  mat = np.zeros((len(stages), len(cum_points)), dtype=np.float64)
  for j, row in enumerate(fr_rows):
    for i, sid in enumerate(stages):
      mat[i, j] = float(row.get(sid, 0.0))
  return (
    np.asarray(cum_points, dtype=np.float64),
    stages,
    mat,
  )


def _knobs_summary(knobs: AdaptiveInletKnobs) -> str:
  return (
    f"gain={knobs.gain} max_chunk={knobs.max_chunk} min_delta={knobs.min_delta} "
    f"lambda={knobs.decay_lambda} unlock={knobs.unlock_fraction} "
    f"prior_floor={knobs.prior_floor}"
  )


def plot_inlet_vs_val_delta(
  knobs: AdaptiveInletKnobs,
  *,
  last_stage: int,
  out_path: Path | str,
  delta_max: float | None = None,
) -> Path:
  """Val drop (x) vs stage fractions and inlet chunk after one blend step."""
  out = prepare_out(out_path)
  if delta_max is None:
    delta_max = max(
      knobs.min_delta * 4.0,
      knobs.max_chunk / max(knobs.gain, 1.0e-12) * 1.2,
      0.05,
    )

  fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
  fig.suptitle("Single-step inlet vs wval drop\n" + _knobs_summary(knobs), fontsize=10)

  scenarios = [
    ("start: stage 0 only", 0, {0: 1.0}),
  ]
  if last_stage >= 1:
    mid = exp_template_fractions(1, knobs.decay_lambda)
    scenarios.append(("active_max=1, at exp template", 1, mid))

  # When stage 1+ exists, top panel uses dashed "at exp template" case only (less clutter).
  emphasis_only_top = last_stage >= 1

  for label, active_max, prev_fr in scenarios:
    deltas, fr_mat, chunks = sweep_delta_response(
      knobs,
      last_stage=last_stage,
      active_max=active_max,
      prev_fr=prev_fr,
      delta_max=delta_max,
      n_points=200,
    )
    stages = sorted(fr_mat.keys())
    colors = _stage_colors(stages)
    at_template = active_max >= 1
    linestyle = "--" if at_template else "-"
    linewidth = 1.2 if at_template else 1.0
    for sid, color in zip(stages, colors):
      introduced = fr_mat[sid] - float(prev_fr.get(sid, 0.0))
      if at_template or not emphasis_only_top:
        axes[0].plot(
          deltas,
          introduced,
          color=color,
          label=f"stage {sid}",
          linewidth=linewidth,
          linestyle=linestyle,
        )
      # Faint solid "stage 0 only" reference when emphasis case is shown.
      mid_alpha = 0.35 if (emphasis_only_top and not at_template) else 1.0
      axes[1].plot(
        deltas,
        fr_mat[sid],
        color=color,
        label=f"stage {sid}" if (at_template or not emphasis_only_top) else None,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=mid_alpha,
      )
    axes[2].plot(
      deltas,
      chunks,
      label=f"chunk ({label})",
      linewidth=linewidth,
      linestyle=linestyle,
      alpha=0.35 if (emphasis_only_top and not at_template) else 1.0,
    )

  axes[0].axvline(knobs.min_delta, color="0.4", linestyle=":", linewidth=1.0)
  if emphasis_only_top:
    axes[0].set_ylabel("introduced fraction (delta f)\nactive_max=1, exp template")
  else:
    axes[0].set_ylabel("introduced fraction (delta f)")
  axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
  axes[0].grid(True, alpha=0.3)

  axes[1].axvline(knobs.min_delta, color="0.4", linestyle=":", linewidth=1.0)
  axes[1].text(
    knobs.min_delta,
    0.02,
    "mix_min_val_delta",
    rotation=90,
    va="bottom",
    ha="right",
    fontsize=8,
  )
  cap_delta = knobs.max_chunk / max(knobs.gain, 1.0e-12)
  if cap_delta <= delta_max:
    axes[2].axvline(cap_delta, color="0.35", linestyle=":", linewidth=1.0)

  axes[1].set_ylabel("train fraction (after min projection)")
  axes[1].set_ylim(0.0, 1.0)
  axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
  axes[1].grid(True, alpha=0.3)

  axes[2].set_xlabel("wval drop (prev - curr)")
  axes[2].set_ylabel("mix_inlet_chunk")
  axes[2].set_ylim(0.0, knobs.max_chunk * 1.05)
  axes[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
  axes[2].grid(True, alpha=0.3)

  return write_fig(fig, out)


def plot_exp_templates(
  knobs: AdaptiveInletKnobs,
  *,
  last_stage: int,
  out_path: Path | str,
) -> Path:
  """Reference exp templates before slack min projection (one row per active_max)."""
  out = prepare_out(out_path)
  fig, ax = plt.subplots(figsize=(7, 4))
  fig.suptitle(
    "Exp mix templates (pre passed_stage_min)\n" + _knobs_summary(knobs),
    fontsize=10,
  )
  x = np.arange(0, last_stage + 1)
  width = 0.8 / max(1, last_stage + 1)
  for active_max in range(0, last_stage + 1):
    tmpl = exp_template_fractions(active_max, knobs.decay_lambda)
    heights = [float(tmpl.get(int(s), 0.0)) for s in x]
    offset = (active_max - last_stage / 2.0) * width
    ax.bar(x + offset, heights, width=width * 0.9, label=f"active_max={active_max}")
  ax.set_xticks(x)
  ax.set_xticklabels([str(int(s)) for s in x])
  ax.set_xlabel("stage id")
  ax.set_ylabel("template fraction")
  ax.set_ylim(0.0, 1.0)
  ax.legend(fontsize=8)
  ax.grid(True, axis="y", alpha=0.3)
  return write_fig(fig, out)


def plot_fractions_vs_cumulative_drop(
  knobs: AdaptiveInletKnobs,
  *,
  last_stage: int,
  out_path: Path | str,
  per_epoch_delta: float = 0.03,
  max_steps: int = 200,
) -> Path:
  cum, stages, mat = simulate_cumulative_inlet(
    knobs,
    last_stage=last_stage,
    per_epoch_delta=per_epoch_delta,
    max_steps=max_steps,
  )
  out = prepare_out(out_path)
  fig, ax = plt.subplots(figsize=(8, 4.5))
  fig.suptitle(
    f"Synthetic path: {per_epoch_delta} wval drop per step\n" + _knobs_summary(knobs),
    fontsize=10,
  )

  colors = _stage_colors(stages)
  labels = [f"stage {s}" for s in stages]
  ax.stackplot(cum, *mat, labels=labels, colors=colors, alpha=0.85)
  ax.set_ylabel("train fraction")
  ax.set_xlabel("cumulative wval drop (sum over qualifying steps)")
  ax.set_ylim(0.0, 1.0)
  ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
  ax.grid(True, alpha=0.3)
  ax.axhline(
    knobs.unlock_fraction,
    color="0.2",
    linestyle=":",
    linewidth=1.0,
  )
  ax.text(
    cum[-1] * 0.02 if len(cum) else 0.0,
    knobs.unlock_fraction + 0.01,
    "mix_unlock_fraction",
    fontsize=8,
    color="0.2",
  )

  return write_fig(fig, out)


def render_curriculum_inlet_figures(
  cfg: dict[str, Any],
  *,
  last_stage: int,
  out_dir: Path | str,
  per_epoch_delta: float = 0.03,
  max_steps: int = 200,
) -> list[Path]:
  validate_curriculum_cfg(cfg)
  knobs = inlet_knobs_from_cfg(cfg)
  root = Path(out_dir)
  root.mkdir(parents=True, exist_ok=True)
  paths = [
    plot_exp_templates(knobs, last_stage=last_stage, out_path=root / "inlet_exp_templates.png"),
    plot_inlet_vs_val_delta(
      knobs,
      last_stage=last_stage,
      out_path=root / "inlet_fraction_vs_val_delta.png",
    ),
    plot_fractions_vs_cumulative_drop(
      knobs,
      last_stage=last_stage,
      out_path=root / "inlet_fractions_vs_cumulative_val_drop.png",
      per_epoch_delta=per_epoch_delta,
      max_steps=max_steps,
    ),
  ]
  return paths


def _parse_stage_list(text: str) -> list[int]:
  return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plot adaptive curriculum inlet from configs/models YAML",
  )
  parser.add_argument(
    "--model",
    default="curriculum",
    help="configs/models/<name>.yaml stem (default: curriculum)",
  )
  parser.add_argument(
    "--stages",
    default="0,1,2,3",
    help="Stage ids for last_stage ceiling (default: 0,1,2,3)",
  )
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("figures/curriculum_inlet_theory"),
    help="Output directory for PNGs",
  )
  parser.add_argument(
    "--per-epoch-delta",
    type=float,
    default=0.03,
    help="Synthetic wval drop per step for cumulative plot",
  )
  parser.add_argument(
    "--max-steps",
    type=int,
    default=200,
    help="Synthetic steps before optional saturation extension",
  )
  return parser


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  stages = _parse_stage_list(args.stages)
  if not stages:
    raise SystemExit("--stages must list at least one stage id")
  cfg = load_model_config(str(args.model))
  paths = render_curriculum_inlet_figures(
    cfg,
    last_stage=int(max(stages)),
    out_dir=args.out_dir,
    per_epoch_delta=float(args.per_epoch_delta),
    max_steps=int(args.max_steps),
  )
  for path in paths:
    print(path)


if __name__ == "__main__":
  main()
