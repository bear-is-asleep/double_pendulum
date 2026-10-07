"""Curriculum strategy: adaptive inlet smoke."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from srcs.loader import load_model_config, load_sampler_config, sampler_val_fraction
from srcs.simulation.data import build_pool, carve_validation, pool_path, save_pool
from srcs.train.base import clears_pass_bar
from srcs.train.curriculum import CurriculumTrainer
from srcs.utils.run_logging import close_run_log


@pytest.fixture(autouse=True)
def _no_terminal_tee(monkeypatch: pytest.MonkeyPatch) -> None:
  def _noop_attach(run_dir: Path | str) -> Path:
    log_path = Path(run_dir).resolve() / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    return log_path

  monkeypatch.setattr(
    "srcs.train.curriculum.attach_training_terminal_log",
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


def test_curriculum_trial_writes_run_dir(tmp_path: Path) -> None:
  cfg = load_model_config("curriculum")
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
  trial["max_epochs"] = 4
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
  )
  assert res.run_dir.is_dir()
  assert (res.run_dir / "metrics.jsonl").exists()
  summary = json.loads((res.run_dir / "summary.json").read_text(encoding="utf-8"))
  assert summary["experiment"] == "curriculum"
  assert summary["strategy"] == "curriculum"
  assert "mix_inlet_gain" in summary
  assert all(np.isfinite(row.mse) for row in res.stage_val)
  assert clears_pass_bar(list(res.stage_val), {0: 1e9, 1: 1e9})

  lines = (res.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
  records = [json.loads(line) for line in lines if line]
  assert len(records) >= 1
  for row in records:
    fr = row["train_stage_fraction"]
    assert math.isclose(sum(float(v) for v in fr.values()), 1.0, rel_tol=1e-5)
    assert "mix_weighted_val_mse" in row
    assert "active_max_stage" in row
    assert "mix_val_delta" in row
    assert "mix_inlet_reason" in row
    assert "mix_stagnation_epochs" in row
    assert "mix_inlet_terminal" in row
    assert "mix_stagnation_flat_at_fire" in row
