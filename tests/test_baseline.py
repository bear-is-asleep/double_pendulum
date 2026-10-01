"""Step 5: baseline train smoke (tiny pools, no full data/)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from srcs.loader import load_baseline_lock, load_model_config, load_sampler_config
from srcs.train.baseline import (
  clears_pass_bar,
  iter_search_grid,
  train_baseline_trial,
)
from srcs.simulation.data import build_pool, carve_validation, pool_path, save_pool
from srcs.model import build_mlp, weighted_surrogate_loss
from srcs.model.mlp import count_parameters
from srcs.model.train_data import pointwise_subsample


def _tiny_sampler_cfg() -> dict:
  cfg = copy.deepcopy(load_sampler_config())
  cfg["T"] = 0.35
  cfg["dt"] = 0.05
  cfg["probe_time"] = 0.15
  cfg["max_draw_attempts"] = 400
  cfg["seed"] = 11
  return cfg


def test_mlp_forward_and_param_count() -> None:
  cfg = load_model_config("baseline")
  model = build_mlp(cfg)
  x = torch.randn(5, int(cfg["input_dim"]))
  y = model(x)
  assert y.shape == (5, int(cfg["output_dim"]))
  assert count_parameters(model) > 0


def test_weighted_loss_shape() -> None:
  pred = torch.randn(8, 6)
  target = torch.randn(8, 6)
  loss = weighted_surrogate_loss(
    pred,
    target,
    loss_weight_theta=1.0,
    loss_weight_omega=0.25,
  )
  assert loss.ndim == 0


def test_pointwise_subsample_stride() -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 2, cfg, rng=np.random.default_rng(0))
  x_full, _ = pool.pointwise()
  x_sub, _ = pointwise_subsample(pool, 2)
  assert x_sub.shape[0] < x_full.shape[0]
  assert x_sub.shape[1] == 8


def test_baseline_trial_writes_run_dir(tmp_path: Path) -> None:
  cfg = load_model_config("baseline")
  sampler = _tiny_sampler_cfg()
  data_root = tmp_path / "data"
  data_root.mkdir()
  rng = np.random.default_rng(2)
  raw = build_pool(1, 8, sampler, rng=rng)
  train, val = carve_validation(raw, 0.25, rng)
  save_pool(pool_path(data_root, 1, "train"), train)
  save_pool(pool_path(data_root, 1, "val"), val)

  trial = dict(cfg)
  trial["hidden_width"] = 24
  trial["hidden_depth"] = 1
  trial["subsample_stride_k"] = 2
  trial["max_epochs"] = 2
  trial["early_stop_patience"] = 10
  trial["batch_size"] = 32
  trial["stage_pass_mse"] = 1e9  # smoke: do not require real convergence

  runs_root = tmp_path / "runs"
  res = train_baseline_trial(
    trial,
    data_root=data_root,
    stages=[1],
    runs_root=runs_root,
    run_id="baseline_w24_d1_k2_seed0_smoke",
    device_name="cpu",
    max_epochs=2,
  )
  assert res.run_dir.is_dir()
  assert (res.run_dir / "metrics.jsonl").exists()
  assert (res.run_dir / "summary.json").exists()
  assert (res.run_dir / "checkpoints" / "best.pt").exists()
  summary = json.loads((res.run_dir / "summary.json").read_text(encoding="utf-8"))
  assert summary["run_id"] == res.run_id
  assert clears_pass_bar(list(res.stage_val), trial["stage_pass_mse"])


def test_baseline_search_grid_yields_combos() -> None:
  cfg = load_model_config("baseline")
  combos = list(iter_search_grid(cfg))
  assert len(combos) == 3 * 2 * 3  # widths x depths x k from baseline.yaml


def _write_tiny_stage_pools(
  data_root: Path,
  sampler_cfg: dict,
  *,
  val_fraction: float,
  stages: tuple[int, ...] = (0, 1),
  n_traj: int = 4,
  rng: np.random.Generator | None = None,
) -> None:
  """Minimal on-disk train/val pools for baseline smoke tests."""
  tiny = copy.deepcopy(sampler_cfg)
  tiny["T"] = 0.4
  tiny["dt"] = 0.05
  tiny["probe_time"] = 0.2
  tiny["max_draw_attempts"] = 300
  gen = rng if rng is not None else np.random.default_rng(0)
  data_root.mkdir(parents=True, exist_ok=True)
  for stage in stages:
    raw = build_pool(stage, n_traj, tiny, split="train", rng=gen)
    train, val = carve_validation(raw, val_fraction, gen)
    save_pool(pool_path(data_root, stage, "train"), train, overwrite_frozen=True)
    save_pool(pool_path(data_root, stage, "val"), val, overwrite_frozen=True)


def test_baseline_search_smoke_trial(tmp_path: Path) -> None:
  """End-to-end: tiny pools, one short trial, run artifacts on disk."""
  model_cfg = load_model_config("baseline")
  sampler_cfg = _tiny_sampler_cfg()
  data_root = tmp_path / "data"
  runs_root = tmp_path / "runs"
  _write_tiny_stage_pools(
    data_root,
    sampler_cfg,
    val_fraction=float(model_cfg["val_fraction"]),
  )

  trial_cfg = dict(model_cfg)
  trial_cfg["hidden_width"] = 32
  trial_cfg["hidden_depth"] = 1
  trial_cfg["subsample_stride_k"] = 2
  trial_cfg["max_epochs"] = 2
  trial_cfg["early_stop_patience"] = 5
  trial_cfg["stage_pass_mse"] = 1e9

  res = train_baseline_trial(
    trial_cfg,
    data_root=data_root,
    stages=[0, 1],
    runs_root=runs_root,
    run_id="baseline_w32_d1_k2_seed0_smoke",
    device_name="cpu",
    max_epochs=2,
  )
  assert res.run_dir.name == "baseline_w32_d1_k2_seed0_smoke"
  assert (res.run_dir / "checkpoints" / "last.pt").exists()


def test_baseline_lock_loads() -> None:
  lock = load_baseline_lock()
  assert lock["hidden_width"] == 128
  assert lock["subsample_stride_k"] == 4
  assert "stage_pass_mse" in lock
