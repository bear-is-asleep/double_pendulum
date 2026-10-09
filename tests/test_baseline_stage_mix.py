"""Baseline stage mix fractions and weighted val logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import copy

from srcs.loader import load_model_config, load_sampler_config
from srcs.model.train_data import (
  BASELINE_STAGE_MIX_EQUAL,
  baseline_stage_mix_fractions,
  make_weighted_stage_loader,
  resolve_baseline_stage_mix,
)
from srcs.simulation.data import build_pool, carve_validation, pool_path, save_pool
from srcs.train.baseline import BaselineTrainer
from srcs.utils.json_io import load_jsonl


def _write_tiny_stage_pools(
  data_root: Path,
  sampler_cfg: dict,
  *,
  val_fraction: float,
  stages: tuple[int, ...] = (0, 1),
  n_traj: int = 6,
) -> None:
  tiny = copy.deepcopy(sampler_cfg)
  tiny["T"] = 0.4
  tiny["dt"] = 0.05
  tiny["probe_time"] = 0.2
  tiny["max_draw_attempts"] = 300
  gen = np.random.default_rng(0)
  data_root.mkdir(parents=True, exist_ok=True)
  for stage in stages:
    raw = build_pool(stage, n_traj, tiny, split="train", rng=gen)
    train, val = carve_validation(raw, val_fraction, gen)
    save_pool(pool_path(data_root, stage, "train"), train, overwrite_frozen=True)
    save_pool(pool_path(data_root, stage, "val"), val, overwrite_frozen=True)


def test_baseline_stage_mix_fractions_equal() -> None:
  fr = baseline_stage_mix_fractions([0, 1, 2], {0: 10, 1: 20, 2: 30}, BASELINE_STAGE_MIX_EQUAL)
  assert fr == {0: 1.0 / 3.0, 1: 1.0 / 3.0, 2: 1.0 / 3.0}


def test_stage_mix_sampler_equal_stages() -> None:
  x0 = np.zeros((40, 8), dtype=np.float64)
  x1 = np.ones((60, 8), dtype=np.float64)
  y0 = np.zeros((40, 6), dtype=np.float64)
  y1 = np.zeros((60, 6), dtype=np.float64)
  loader = make_weighted_stage_loader(
    {0: (x0, y0), 1: (x1, y1)},
    {0: 0.5, 1: 0.5},
    batch_size=10,
    seed=0,
  )
  stage_hits = {0: 0, 1: 0}
  for xb, _ in loader:
    for row in xb:
      stage_hits[0 if float(row[1].item()) < 0.5 else 1] += 1
  assert stage_hits[0] + stage_hits[1] == 100
  assert 30 <= stage_hits[0] <= 70


@pytest.fixture(autouse=True)
def _no_terminal_log_tee(monkeypatch: pytest.MonkeyPatch) -> None:
  """Pytest capture closes stderr; skip tee that wraps it during train smoke."""
  monkeypatch.setattr(
    "srcs.train.base.attach_training_terminal_log",
    lambda _run_dir: None,
  )


def test_baseline_equal_mix_trial_logs_weighted_val(tmp_path: Path) -> None:
  cfg = load_model_config("baseline")
  sampler = load_sampler_config()
  data_root = tmp_path / "data"
  _write_tiny_stage_pools(data_root, sampler, val_fraction=0.25, stages=(0, 1), n_traj=6)
  trial = dict(cfg)
  trial["hidden_width"] = 16
  trial["hidden_depth"] = 1
  trial["subsample_stride_k"] = 2
  trial["max_epochs"] = 1
  trial["early_stop_patience"] = 5
  trial["batch_size"] = 16
  trial["stage_pass_mse"] = 1e9
  trial["baseline_stage_mix"] = BASELINE_STAGE_MIX_EQUAL

  BaselineTrainer().train_trial(
    trial,
    data_root=data_root,
    stages=[0, 1],
    runs_root=tmp_path / "runs",
    run_id="baseline_equal_mix_smoke",
    device_name="cpu",
    max_epochs=1,
  )
  row = load_jsonl(tmp_path / "runs" / "baseline_equal_mix_smoke" / "metrics.jsonl")[0]
  assert row["baseline_stage_mix"] == BASELINE_STAGE_MIX_EQUAL
  mix = row["train_stage_fraction"]
  assert abs(float(mix["0"]) - 0.5) < 1e-5
  assert abs(float(mix["1"]) - 0.5) < 1e-5
  assert "weighted_val_mse" in row
  assert np.isfinite(row["weighted_val_mse"])


def test_resolve_baseline_stage_mix_default() -> None:
  cfg = load_model_config("baseline")
  assert resolve_baseline_stage_mix(cfg) == BASELINE_STAGE_MIX_EQUAL
