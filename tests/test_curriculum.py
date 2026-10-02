"""Step 7: curriculum strategy (sequential stages, single-stage train pools)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from srcs.loader import load_model_config, load_sampler_config
from srcs.simulation.data import build_pool, carve_validation, pool_path, save_pool
from srcs.train.base import clears_pass_bar
from srcs.train.curriculum import CurriculumTrainer, should_advance_curriculum_stage


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


def test_should_advance_curriculum_stage() -> None:
  assert should_advance_curriculum_stage(0.5, 1.0, 0, 5, 1, 10)
  assert should_advance_curriculum_stage(2.0, 1.0, 5, 5, 3, 10)
  assert should_advance_curriculum_stage(2.0, 1.0, 0, 5, 10, 10)
  assert not should_advance_curriculum_stage(2.0, 1.0, 2, 5, 3, 10)


def test_curriculum_trial_writes_run_dir(tmp_path: Path) -> None:
  cfg = load_model_config("curriculum")
  sampler = _tiny_sampler_cfg()
  data_root = tmp_path / "data"
  _write_tiny_stage_pools(
    data_root,
    sampler,
    val_fraction=float(cfg["val_fraction"]),
    stages=(0, 1),
  )

  trial = dict(cfg)
  trial["hidden_width"] = 24
  trial["hidden_depth"] = 1
  trial["subsample_stride_k"] = 2
  trial["max_epochs_per_stage"] = 2
  trial["early_stop_patience"] = 10
  trial["batch_size"] = 32
  trial["stage_pass_mse"] = 1e9

  runs_root = tmp_path / "runs"
  res = CurriculumTrainer().train_trial(
    trial,
    data_root=data_root,
    stages=[0, 1],
    runs_root=runs_root,
    run_id="curriculum_w24_d1_k2_seed0_smoke",
    device_name="cpu",
    max_epochs=2,
  )
  assert res.run_dir.is_dir()
  assert (res.run_dir / "metrics.jsonl").exists()
  summary = json.loads((res.run_dir / "summary.json").read_text(encoding="utf-8"))
  assert summary["experiment"] == "curriculum"
  assert summary["curriculum_last_stage"] == 1
  assert summary["strategy"] == "curriculum"
  assert all(np.isfinite(row.mse) for row in res.stage_val)
  assert clears_pass_bar(list(res.stage_val), trial["stage_pass_mse"])

  lines = (res.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
  records = [json.loads(line) for line in lines if line]
  stages_seen = {r["curriculum_stage"] for r in records}
  assert stages_seen == {0, 1}
