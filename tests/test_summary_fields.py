"""Unified summary.json strategy fields."""

import math

from srcs.loader import load_model_config
from srcs.train.summary_fields import strategy_summary_fields


def test_baseline_inlet_keys_are_nan() -> None:
  cfg = load_model_config("baseline")
  fields = strategy_summary_fields(cfg, stages=[0, 1], n_params_trainable=100)
  assert fields["strategy"] == "baseline"
  assert math.isnan(fields["mix_inlet_gain"])
  assert math.isnan(fields["n_columns"])
  assert fields["n_params_trainable"] == 100.0
  assert fields["n_params_total"] == 100.0


def test_curriculum_inlet_keys_populated() -> None:
  cfg = load_model_config("curriculum")
  fields = strategy_summary_fields(
    cfg,
    stages=[0, 1, 2],
    run_outcome={"final_active_max_stage": 1},
  )
  assert fields["mix_inlet_gain"] == cfg["mix_inlet_gain"]
  assert fields["curriculum_last_stage"] == 2.0
  assert math.isnan(fields["progressive_last_stage"])
  assert fields["final_active_max_stage"] == 1.0


def test_progressive_pnn_fields() -> None:
  cfg = load_model_config("progressive")
  cfg["n_columns"] = 3
  cfg["_n_params_total"] = 500
  cfg["_n_params_trainable"] = 120
  fields = strategy_summary_fields(cfg, stages=[0, 1])
  assert fields["model_type"] == "progressive_pnn"
  assert fields["n_columns"] == 3.0
  assert fields["use_lateral"] is True
  assert fields["n_params_total"] == 500.0
  assert not math.isnan(fields["mix_inlet_gain"])
