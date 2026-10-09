"""Progressive PNN trainer integration smoke."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from srcs.loader import load_model_config, load_sampler_config, sampler_val_fraction
from srcs.model.checkpoint import load_model_from_checkpoint, predict_at_times
from srcs.model.progressive_net import ProgressiveNet
from srcs.simulation.data import build_pool, carve_validation, pool_path, save_pool
from srcs.train.base import clears_pass_bar
from srcs.train.progressive_pnn import ProgressivePnnTrainer
from srcs.utils.run_logging import close_run_log


@pytest.fixture(autouse=True)
def _no_terminal_tee(monkeypatch: pytest.MonkeyPatch) -> None:
  def _noop_attach(run_dir: Path | str) -> Path:
    log_path = Path(run_dir).resolve() / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    return log_path

  monkeypatch.setattr(
    "srcs.train.base.attach_training_terminal_log",
    _noop_attach,
  )
  yield
  close_run_log()


def _tiny_sampler_cfg() -> dict:
  cfg = copy.deepcopy(load_sampler_config())
  cfg["T"] = 0.35
  cfg["dt"] = 0.05
  cfg["probe_time"] = 0.15
  cfg["max_draw_attempts"] = 400
  cfg["seed"] = 11
  return cfg


def _write_tiny_stage_pools(
  data_root: Path,
  sampler_cfg: dict,
  *,
  val_fraction: float,
  stages: tuple[int, ...] = (0, 1),
  n_traj: int = 10,
) -> None:
  tiny = copy.deepcopy(sampler_cfg)
  tiny["T"] = 0.4
  tiny["dt"] = 0.05
  tiny["probe_time"] = 0.2
  tiny["max_draw_attempts"] = 300
  gen = np.random.default_rng(4)
  data_root.mkdir(parents=True, exist_ok=True)
  for stage in stages:
    raw = build_pool(stage, n_traj, tiny, split="train", rng=gen)
    train, val = carve_validation(raw, val_fraction, gen)
    save_pool(pool_path(data_root, stage, "train"), train, overwrite_frozen=True)
    save_pool(pool_path(data_root, stage, "val"), val, overwrite_frozen=True)


def test_progressive_trial_writes_metrics(tmp_path: Path) -> None:
  cfg = load_model_config("progressive")
  sampler = _tiny_sampler_cfg()
  data_root = tmp_path / "data"
  _write_tiny_stage_pools(
    data_root,
    sampler,
    val_fraction=sampler_val_fraction(sampler),
    stages=(0, 1),
  )

  trial = dict(cfg)
  trial["hidden_width"] = 24
  trial["hidden_depth"] = 1
  trial["subsample_stride_k"] = 2
  trial["max_epochs"] = 6
  trial["early_stop_patience"] = 20
  trial["batch_size"] = 32
  trial["stage_pass_mse"] = 1e9
  trial["mix_unlock_fraction"] = 0.5
  trial["mix_inlet_gain"] = 3.0

  runs_root = tmp_path / "runs"
  res = ProgressivePnnTrainer().train_trial(
    trial,
    data_root=data_root,
    stages=[0, 1],
    runs_root=runs_root,
    run_id="progressive_w24_d1_k2_seed0_smoke",
    device_name="cpu",
  )
  assert res.run_dir.is_dir()
  summary = json.loads((res.run_dir / "summary.json").read_text(encoding="utf-8"))
  assert summary["experiment"] == "progressive"
  assert summary["strategy"] == "progressive"
  assert summary["n_columns"] >= 1
  assert summary["n_params_total"] >= summary["n_params_trainable"]
  assert clears_pass_bar(list(res.stage_val), {0: 1e9, 1: 1e9})

  lines = (res.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
  records = [json.loads(line) for line in lines if line]
  assert any("n_columns" in row for row in records)
  max_cols = max(int(row.get("n_columns", 1)) for row in records)
  if max_cols >= 2:
    assert any(row.get("optimizer_reset") for row in records)

  ckpt = res.run_dir / "checkpoints" / "best.pt"
  model, meta, _ = load_model_from_checkpoint(ckpt, device="cpu")
  assert isinstance(model, ProgressiveNet)
  row = np.zeros(8, dtype=np.float64)
  row[7] = 9.8
  out = predict_at_times(model, np.array([0.0]), row, column_index=0)
  assert out.shape == (1, 6)
