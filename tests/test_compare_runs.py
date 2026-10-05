"""Smoke tests for multi-run compare figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from srcs.eval.channels import CHANNEL_KEYS
from srcs.loader import load_vis_compare_config
from srcs.visualization.compare_runs import (
  CompareModel,
  CompareSpec,
  load_compare_config_by_name,
  render_compare_figures,
)
from srcs.visualization.mpl_plots import assert_compatible_time_grids


def _write_run(
  root: Path,
  name: str,
  *,
  t: np.ndarray,
  error_row: np.ndarray,
) -> Path:
  run = root / name
  run.mkdir(parents=True)
  (run / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
  metrics = run / "metrics.jsonl"
  lines = [
    {
      "epoch": 1,
      "train_loss": 0.5,
      "mean_val_mse": 0.4,
      "epoch_seconds": 2.0,
      "stage_val_mse": {"0": 0.3, "1": 0.35},
    },
    {
      "epoch": 2,
      "train_loss": 0.3,
      "mean_val_mse": 0.25,
      "epoch_seconds": 1.5,
      "stage_val_mse": {"0": 0.2, "1": 0.22},
    },
  ]
  metrics.write_text(
    "\n".join(json.dumps(row) for row in lines) + "\n",
    encoding="utf-8",
  )
  eval_test = run / "eval_test"
  eval_test.mkdir()
  err = np.stack([error_row, error_row * 1.1])
  np.savez(
    eval_test / "error_vs_t.npz",
    t=t,
    error=err,
    stage_ids=np.array([0, 1], dtype=np.int32),
  )
  mae = {k: err for k in CHANNEL_KEYS}
  np.savez(eval_test / "channel_mae_vs_t.npz", t=t, **mae)
  (eval_test / "eval_summary.json").write_text(
    json.dumps({"test_stage_mse": {"0": 0.1, "1": 0.12}}),
    encoding="utf-8",
  )
  return run


def test_vis_compare_config_loads() -> None:
  cfg = load_vis_compare_config("baseline_vs_curriculum")
  assert cfg["name"] == "baseline_vs_curriculum"
  assert len(cfg["models"]) == 2


def test_render_compare_figures_smoke(tmp_path: Path) -> None:
  t = np.linspace(0.0, 1.0, 5)
  row = np.linspace(0.05, 0.15, 5)
  run_a = _write_run(tmp_path, "run_a", t=t, error_row=row)
  run_b = _write_run(tmp_path, "run_b", t=t, error_row=row * 0.9)
  from srcs.eval.run_layout import RunEvalLayout

  spec = CompareSpec(
    name="smoke",
    out_dir=tmp_path / "fig",
    models=(
      CompareModel("a", RunEvalLayout.from_run_dir(run_a)),
      CompareModel("b", RunEvalLayout.from_run_dir(run_b)),
    ),
    val_key="mean_val_mse",
    train_key="train_loss",
    train_ylim=0.4,
  )
  paths = render_compare_figures(spec)
  names = {p.name for p in paths}
  expected = {
    "training.png",
    "cumulative_train_time.png",
    "stage_val_mse.png",
    "error_vs_t.png",
    "channel_mae_vs_t.png",
    "summary_stages.png",
  }
  assert expected <= names
  for p in paths:
    assert p.is_file() and p.stat().st_size > 300


def test_time_grid_mismatch_raises() -> None:
  t_a = np.linspace(0, 1, 4)
  t_b = np.linspace(0, 1, 5)
  with pytest.raises(ValueError, match="time grid mismatch"):
    assert_compatible_time_grids(
      t_a,
      t_b,
      label_a="a",
      label_b="b",
      context="error_vs_t",
    )


def test_load_compare_config_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Name-based load resolves run_dir from a temp vis yaml."""
  vis_dir = tmp_path / "configs" / "vis"
  vis_dir.mkdir(parents=True)
  t = np.linspace(0, 1, 3)
  row = np.array([0.1, 0.2, 0.15])
  run_a = _write_run(tmp_path, "ra", t=t, error_row=row)
  run_b = _write_run(tmp_path, "rb", t=t, error_row=row)
  (vis_dir / "pair.yaml").write_text(
    f"name: pair\n"
    f"out_dir: {tmp_path / 'out'}\n"
    f"models:\n"
    f"  - label: one\n    run_dir: {run_a}\n"
    f"  - label: two\n    run_dir: {run_b}\n",
    encoding="utf-8",
  )
  monkeypatch.setattr(
    "srcs.loader._CONFIG_ROOT",
    tmp_path / "configs",
  )
  monkeypatch.setitem(
    __import__("srcs.loader", fromlist=["_KIND_DIRS"])._KIND_DIRS,
    "vis",
    vis_dir,
  )
  spec = load_compare_config_by_name("pair")
  assert spec.out_dir == tmp_path / "out"
  paths = render_compare_figures(spec)
  assert len(paths) == 6
