"""Pytest coverage for test-pool metrics and eval figure helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from srcs.eval.channels import CHANNEL_KEYS, per_step_channel_abs_errors
from srcs.visualization.eval_figures import (
  plot_error_vs_t,
  plot_summary_stages,
  plot_training_curves,
)
from srcs.eval.pool_eval import EvalTestResult, weighted_loss_per_step, write_eval_artifacts
from srcs.eval.run_layout import RunEvalLayout


def test_weighted_loss_per_step_zero_on_match() -> None:
  pred = np.random.randn(12, 6)
  loss = weighted_loss_per_step(pred, pred, loss_weight_theta=2.0, loss_weight_omega=1.0)
  assert loss.shape == (12,)
  assert np.allclose(loss, 0.0)


def test_per_step_channel_abs_errors_keys() -> None:
  """Channel dict matches CHANNEL_KEYS and uses pred column layout."""
  n = 5
  pred = np.zeros((n, 6))
  pred[:, 0] = 0.1
  pred[:, 4] = 0.2
  pe = np.zeros(n)
  ke = np.zeros(n)

  class _View:
    sin_theta1 = np.zeros(n)
    sin_theta2 = np.zeros(n)
    omega1 = np.zeros(n)
    omega2 = np.zeros(n)
    potential = np.zeros(n)
    kinetic = np.zeros(n)

  errs = per_step_channel_abs_errors(_View(), pred, pe, ke)
  assert set(errs) == set(CHANNEL_KEYS)
  assert np.allclose(errs["sin_theta1"], 0.1)
  assert np.allclose(errs["omega1"], 0.2)


def test_weighted_loss_per_step_shape_mismatch() -> None:
  pred = np.zeros((3, 6))
  target = np.zeros((4, 6))
  with pytest.raises(ValueError, match="shape mismatch"):
    weighted_loss_per_step(pred, target, loss_weight_theta=1.0, loss_weight_omega=1.0)


def test_write_eval_artifacts_roundtrip(tmp_path: Path) -> None:
  t = np.linspace(0.0, 0.3, 4)
  err = np.array([[0.1, 0.2, 0.15, 0.05]], dtype=np.float64)
  mae = {k: err.copy() for k in CHANNEL_KEYS}
  result = EvalTestResult(
    stage_ids=(1,),
    t=t,
    error_vs_t=err,
    channel_mae=mae,
    test_stage_mse={"1": 0.125},
    n_traj_by_stage={"1": 2},
  )
  paths = write_eval_artifacts(
    result,
    tmp_path,
    checkpoint="ckpt.pt",
    data_root=str(tmp_path / "data"),
  )
  summary = json.loads(paths["eval_summary"].read_text(encoding="utf-8"))
  assert summary["test_stage_mse"]["1"] == 0.125
  loaded = np.load(paths["error_vs_t"])
  assert loaded["stage_ids"].tolist() == [1]
  assert loaded["error"].shape == (1, 4)


def test_plot_error_vs_t_respects_stage_ids(tmp_path: Path) -> None:
  t = np.linspace(0, 0.2, 4)
  err = np.stack([np.linspace(0.1, 0.2, 4), np.linspace(0.3, 0.4, 4)])
  np.savez(tmp_path / "e.npz", t=t, error=err, stage_ids=np.array([0, 2], dtype=np.int32))
  out = plot_error_vs_t(tmp_path / "e.npz", tmp_path / "err.png")
  assert out.is_file() and out.stat().st_size > 400


def test_run_eval_layout_from_metrics(tmp_path: Path) -> None:
  run = tmp_path / "my_run_id"
  run.mkdir()
  metrics = run / "metrics.jsonl"
  metrics.write_text('{"epoch": 1, "mean_val_mse": 0.1}\n', encoding="utf-8")
  layout = RunEvalLayout.from_metrics(metrics)
  assert layout.run_id == "my_run_id"
  assert layout.default_figures_dir == Path("figures") / "my_run_id"
  assert layout.error_npz == run / "eval_test" / "error_vs_t.npz"


def test_eval_figure_fixtures(tmp_path: Path) -> None:
  root = Path(__file__).parent / "fixtures"
  np.savez(
    tmp_path / "error_vs_t.npz",
    t=np.linspace(0, 1, 5),
    error=np.linspace(0.1, 0.5, 5),
    stage_ids=np.array([1], dtype=np.int32),
  )
  plot_training_curves(root / "metrics.jsonl", tmp_path / "train.png")
  plot_error_vs_t(tmp_path / "error_vs_t.npz", tmp_path / "err.png")
  plot_summary_stages(root / "summary.json", tmp_path / "sum.png")
  for name in ("train.png", "err.png", "sum.png"):
    p = tmp_path / name
    assert p.is_file() and p.stat().st_size > 200


def test_plot_summary_from_eval_summary(tmp_path: Path) -> None:
  eval_path = tmp_path / "eval_summary.json"
  eval_path.write_text(
    '{"test_stage_mse": {"0": 0.01, "1": 0.02, "99": 0.5}, "stage_ids": [0, 1]}',
    encoding="utf-8",
  )
  out = plot_summary_stages(None, tmp_path / "bars.png", eval_summary_path=eval_path)
  assert out.is_file() and out.stat().st_size > 200
