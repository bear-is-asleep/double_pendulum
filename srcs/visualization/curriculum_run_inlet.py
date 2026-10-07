"""Curriculum inlet theory figures for a finished training run folder."""

from __future__ import annotations

from pathlib import Path

from srcs.eval.run_layout import RunEvalLayout
from srcs.loader import load_model_config
from srcs.train.curriculum_mix import inlet_cfg_slice_from_mapping
from srcs.utils.json_io import read_json
from srcs.utils.yaml_io import read_mapping
from srcs.visualization.curriculum_inlet_plots import render_curriculum_inlet_figures

_CURRICULUM_STRATEGY = "curriculum"
_DEFAULT_MODEL_STEM = "curriculum"
_INLET_THEORY_SUBDIR = "inlet_theory"


def _strategy_label(data: dict) -> str | None:
  strategy = data.get("strategy")
  if isinstance(strategy, str) and strategy:
    return strategy
  experiment = data.get("experiment")
  if isinstance(experiment, str) and experiment:
    return experiment
  return None


def _read_run_record(path: Path) -> dict:
  if path.suffix == ".json":
    return read_json(path)
  return read_mapping(path)


def is_curriculum_run(layout: RunEvalLayout) -> bool:
  """True when run artifacts mark strategy ``curriculum`` (not baseline / other)."""
  if layout.summary.is_file():
    if _strategy_label(_read_run_record(layout.summary)) == _CURRICULUM_STRATEGY:
      return True
  config_path = layout.run_dir / "config.yaml"
  if config_path.is_file():
    if _strategy_label(_read_run_record(config_path)) == _CURRICULUM_STRATEGY:
      return True
  return layout.run_id.startswith("curriculum_")


def curriculum_inlet_cfg_from_run(layout: RunEvalLayout) -> dict | None:
  """Inlet knobs from ``summary.json`` / ``config.yaml``; None if not curriculum."""
  if not is_curriculum_run(layout):
    return None
  for path in (layout.summary, layout.run_dir / "config.yaml"):
    if not path.is_file():
      continue
    overlay = inlet_cfg_slice_from_mapping(_read_run_record(path))
    if overlay is not None:
      return overlay
  if layout.run_id.startswith("curriculum_"):
    return inlet_cfg_slice_from_mapping(load_model_config(_DEFAULT_MODEL_STEM))
  return None


def last_stage_for_inlet_plots(
  layout: RunEvalLayout,
  allowed_stages: list[int] | None,
) -> int:
  """Ceiling stage for theory plots (vis YAML ``stages`` overrides summary)."""
  if allowed_stages:
    return max(allowed_stages)
  if layout.summary.is_file():
    summary = read_json(layout.summary)
    stages = summary.get("stages")
    if isinstance(stages, list) and stages:
      return max(int(s) for s in stages)
    if "curriculum_last_stage" in summary:
      return int(summary["curriculum_last_stage"])
  return 0


def render_inlet_theory_for_run(
  layout: RunEvalLayout,
  out_dir: Path | str,
  allowed_stages: list[int] | None,
) -> list[Path]:
  """Write ``inlet_theory/*.png`` when ``layout`` is a curriculum run; else []."""
  inlet_cfg = curriculum_inlet_cfg_from_run(layout)
  if inlet_cfg is None:
    return []
  return render_curriculum_inlet_figures(
    inlet_cfg,
    last_stage=last_stage_for_inlet_plots(layout, allowed_stages),
    out_dir=Path(out_dir) / _INLET_THEORY_SUBDIR,
  )
