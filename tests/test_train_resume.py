"""Resume training from checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from srcs.loader import load_model_config, load_sampler_config
from srcs.simulation.data import build_pool, carve_validation, pool_path, save_pool
from srcs.train import __main__ as train_main
from srcs.train.baseline import BaselineTrainer
from srcs.train.curriculum_inlet_loop import InletRunState
from srcs.train.base import assert_checkpoint_cfg_compatible
from srcs.train.curriculum_inlet_loop import pack_inlet_state, unpack_inlet_state
from srcs.utils.json_io import load_jsonl
from srcs.utils.run_logging import close_run_log


@pytest.fixture(autouse=True)
def _disable_training_terminal_log(monkeypatch: pytest.MonkeyPatch):
  def _noop_attach(run_dir: Path) -> Path:
    return Path(run_dir) / "train.log"

  monkeypatch.setattr(
    "srcs.train.base.attach_training_terminal_log",
    _noop_attach,
  )
  close_run_log()
  yield
  close_run_log()


def _tiny_data_root(tmp_path: Path) -> Path:
  sampler = copy.deepcopy(load_sampler_config())
  sampler["T"] = 0.35
  sampler["dt"] = 0.05
  sampler["probe_time"] = 0.15
  sampler["max_draw_attempts"] = 400
  sampler["seed"] = 11
  data_root = tmp_path / "data"
  data_root.mkdir()
  rng = np.random.default_rng(2)
  raw = build_pool(1, 8, sampler, rng=rng)
  train, val = carve_validation(raw, 0.25, rng)
  save_pool(pool_path(data_root, 1, "train"), train)
  save_pool(pool_path(data_root, 1, "val"), val)
  return data_root


def _baseline_trial_cfg() -> dict:
  cfg = load_model_config("baseline")
  trial = dict(cfg)
  trial["hidden_width"] = 24
  trial["hidden_depth"] = 1
  trial["subsample_stride_k"] = 2
  trial["early_stop_patience"] = 10
  trial["batch_size"] = 32
  trial["stage_pass_mse"] = 1e9
  return trial


def test_inlet_state_pack_unpack_round_trip() -> None:
  state = InletRunState(
    train_fractions={0: 0.7, 1: 0.3},
    active_max=1,
    prev_wval=0.42,
    stale_epochs=2,
    stagnation_epochs=3,
    epochs_at_active_max=4,
    best_wval=0.4,
    was_inlet_terminal=True,
  )
  packed = pack_inlet_state(state)
  restored = unpack_inlet_state(packed)
  assert restored.train_fractions == state.train_fractions
  assert restored.active_max == state.active_max
  assert restored.prev_wval == state.prev_wval
  assert restored.stale_epochs == state.stale_epochs
  assert restored.stagnation_epochs == state.stagnation_epochs
  assert restored.epochs_at_active_max == state.epochs_at_active_max
  assert restored.best_wval == state.best_wval
  assert restored.was_inlet_terminal == state.was_inlet_terminal


def test_assert_cfg_compatible_rejects_arch_mismatch() -> None:
  saved = {"model_type": "mlp", "hidden_width": 16, "hidden_depth": 2}
  job = {"model_type": "mlp", "hidden_width": 32, "hidden_depth": 2}
  with pytest.raises(ValueError, match="hidden_width"):
    assert_checkpoint_cfg_compatible(saved, job)


def test_baseline_resume_continues_epochs(tmp_path: Path) -> None:
  data_root = _tiny_data_root(tmp_path)
  trial = _baseline_trial_cfg()
  trial["max_epochs"] = 2
  runs_root = tmp_path / "runs"
  run_id = "resume_smoke"
  trainer = BaselineTrainer()
  first = trainer.train_trial(
    trial,
    data_root=data_root,
    stages=[1],
    runs_root=runs_root,
    run_id=run_id,
    device_name="cpu",
    max_epochs=2,
  )
  last_ckpt = first.run_dir / "checkpoints" / "last.pt"
  before = torch.load(last_ckpt, map_location="cpu", weights_only=True)
  weight_key = next(k for k in before["model_state_dict"] if k.endswith(".weight"))
  before_w = before["model_state_dict"][weight_key].clone()

  trial_resume = dict(trial)
  trial_resume["max_epochs"] = 4
  second = trainer.train_trial(
    trial_resume,
    data_root=data_root,
    stages=[1],
    runs_root=runs_root,
    run_id=run_id,
    device_name="cpu",
    max_epochs=4,
    resume_from=last_ckpt,
  )
  assert second.run_dir.resolve() == first.run_dir.resolve()
  rows = load_jsonl(second.run_dir / "metrics.jsonl")
  epochs = [int(r["epoch"]) for r in rows]
  assert epochs == [1, 2, 3, 4]
  after = torch.load(last_ckpt, map_location="cpu", weights_only=True)
  after_w = after["model_state_dict"][weight_key]
  assert not torch.allclose(before_w, after_w)


def test_main_rejects_force_with_resume(tmp_path: Path) -> None:
  data_root = _tiny_data_root(tmp_path)
  trial = _baseline_trial_cfg()
  trial["max_epochs"] = 1
  runs_root = tmp_path / "runs"
  run_id = "force_guard"
  res = BaselineTrainer().train_trial(
    trial,
    data_root=data_root,
    stages=[1],
    runs_root=runs_root,
    run_id=run_id,
    device_name="cpu",
    max_epochs=1,
  )
  ckpt = res.run_dir / "checkpoints" / "last.pt"
  code = train_main.main(
    [
      "--data-root",
      str(data_root),
      "--runs-root",
      str(runs_root),
      "--run-id",
      run_id,
      "--stages",
      "1",
      "--max-epochs",
      "2",
      "--force",
      "--resume-checkpoint",
      str(ckpt),
    ]
  )
  assert code == 1
  assert res.run_dir.is_dir()
