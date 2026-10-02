"""Resolve run-folder paths for eval metrics and figure exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from srcs.utils.json_io import dir_from_json_field
from srcs.utils.paths import resolve_checkpoint_file, resolve_run_dir


@dataclass(frozen=True)
class RunEvalLayout:
  """Standard layout under ``runs/.../<run_id>/``."""

  run_dir: Path
  run_id: str
  metrics: Path
  summary: Path
  error_npz: Path
  channel_mae_npz: Path
  eval_summary: Path

  @property
  def default_figures_dir(self) -> Path:
    return Path("figures") / self.run_id

  @property
  def checkpoint(self) -> Path:
    return resolve_checkpoint_file(self.run_dir)

  @classmethod
  def from_metrics(cls, metrics_path: Path | str) -> RunEvalLayout:
    metrics = Path(metrics_path).resolve()
    if metrics.name != "metrics.jsonl":
      raise ValueError(f"expected metrics.jsonl, got {metrics.name!r}")
    if not metrics.is_file():
      raise FileNotFoundError(metrics)
    run_dir = metrics.parent
    eval_test = run_dir / "eval_test"
    return cls(
      run_dir=run_dir,
      run_id=run_dir.name,
      metrics=metrics,
      summary=run_dir / "summary.json",
      error_npz=eval_test / "error_vs_t.npz",
      channel_mae_npz=eval_test / "channel_mae_vs_t.npz",
      eval_summary=eval_test / "eval_summary.json",
    )

  @classmethod
  def from_run_dir(cls, run_dir: Path | str) -> RunEvalLayout:
    root = resolve_run_dir(run_dir)
    return cls.from_metrics(root / "metrics.jsonl")


def infer_data_root(run_dir: Path) -> Path:
  """Best-effort data root for test-pool eval when not passed explicitly."""
  for relative in ("eval_test/eval_summary.json", "summary.json"):
    found = dir_from_json_field(run_dir / relative, "data_root")
    if found is not None:
      return found

  # runs/<experiment>/<run_id> -> data/<experiment>
  parent = run_dir.parent
  if parent.name != "runs":
    candidate = Path("data") / parent.name
    if candidate.is_dir():
      return candidate

  return Path("data")


def infer_val_key(rows: list[dict]) -> str:
  """Validation column on the training curve. Falls back to mean_val_mse."""
  if any("mean_val_mse" in r for r in rows):
    return "mean_val_mse"
  if any("val_loss" in r for r in rows):
    return "val_loss"
  return "mean_val_mse"
