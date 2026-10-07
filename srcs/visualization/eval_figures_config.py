"""YAML spec for single-run eval figure export (`configs/vis/<name>.yaml`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from srcs.eval.run_layout import RunEvalLayout
from srcs.loader import load_vis_eval_config
from srcs.utils.paths import resolve_run_dir
from srcs.utils.yaml_io import read_mapping


@dataclass(frozen=True)
class EvalFiguresSpec:
  layout: RunEvalLayout
  out_dir: Path
  val_key: str | None
  train_key: str
  allowed_stages: list[int] | None
  error_npz: Path | None
  channel_mae_npz: Path | None
  summary_path: Path | None
  eval_summary_path: Path | None


def parse_stages_yaml(raw: object) -> list[int] | None:
  if raw is None:
    return None
  if not isinstance(raw, list):
    raise TypeError("'stages' must be a list of integers")
  return [int(s) for s in raw]


def layout_from_plot_mapping(raw: dict) -> RunEvalLayout:
  metrics_raw = raw.get("metrics")
  run_dir_raw = raw.get("run_dir")
  if metrics_raw is not None:
    return RunEvalLayout.from_metrics(Path(metrics_raw))
  if run_dir_raw is not None:
    root = Path(run_dir_raw)
    metrics_in_root = root / "metrics.jsonl"
    if metrics_in_root.is_file():
      return RunEvalLayout.from_metrics(metrics_in_root)
    return RunEvalLayout.from_run_dir(resolve_run_dir(run_dir_raw))
  raise KeyError("plot config must set 'metrics' or 'run_dir'")


def eval_figures_spec_from_mapping(raw: dict) -> EvalFiguresSpec:
  """
  Build spec from YAML.

  Keys: ``metrics`` or ``run_dir``, optional ``out_dir``, ``stages``,
  ``val_key``, ``train_key``, and path overrides for NPZ/JSON artifacts.
  """
  layout = layout_from_plot_mapping(raw)
  out_raw = raw.get("out_dir")
  out_dir = Path(out_raw) if out_raw else layout.default_figures_dir
  return EvalFiguresSpec(
    layout=layout,
    out_dir=out_dir,
    val_key=raw.get("val_key"),
    train_key=str(raw.get("train_key", "train_loss")),
    allowed_stages=parse_stages_yaml(raw.get("stages")),
    error_npz=Path(raw["error_npz"]) if raw.get("error_npz") else None,
    channel_mae_npz=Path(raw["channel_mae_npz"]) if raw.get("channel_mae_npz") else None,
    summary_path=Path(raw["summary"]) if raw.get("summary") else None,
    eval_summary_path=Path(raw["eval_summary"]) if raw.get("eval_summary") else None,
  )


def load_eval_figures_config(path: Path | str) -> EvalFiguresSpec:
  return eval_figures_spec_from_mapping(read_mapping(Path(path)))


def load_eval_figures_config_by_name(name: str) -> EvalFiguresSpec:
  return eval_figures_spec_from_mapping(load_vis_eval_config(name))
