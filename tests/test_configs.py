"""Step 2: sampler + model YAML keys load and merge."""

from __future__ import annotations

from srcs.loader import (
  list_model_configs,
  list_smoke_presets,
  load_model_config,
  load_sampler_config,
  load_smoke_preset,
)
from srcs.simulation.sampler import stage_id_bounds

SAMPLER_REQUIRED = {
  "seed",
  "T",
  "dt",
  "l",
  "g_min",
  "g_max",
  "g_zero",
  "g_low",
  "g_high",
  "m_min",
  "m_equal",
  "omega0_max",
  "omega_traj_max",
  "den_min",
  "energy_abs_tol",
  "energy_rel_tol",
  "energy_eps",
  "probe_time",
  "max_draw_attempts",
  "angle_wrap",
  "stage_min",
  "num_stages",
  "stage0",
  "stage1",
  "stage2",
  "stage3",
  "stage4",
  "stage5",
  "stage6",
  "pools",
}

MODEL_REQUIRED = {
  "name",
  "strategy",
  "model_type",
  "input_dim",
  "output_dim",
  "hidden_width",
  "hidden_depth",
  "activation",
  "lr",
  "batch_size",
  "subsample_stride_k",
  "loss_weight_theta",
  "loss_weight_omega",
  "seed",
  "max_epochs",
  "early_stop_patience",
  "val_fraction",
  "stage_pass_mse",
  "n_ensemble",
}


def test_sampler_yaml_loads() -> None:
  cfg = load_sampler_config()
  missing = SAMPLER_REQUIRED - cfg.keys()
  assert not missing, f"sampler.yaml missing keys: {sorted(missing)}"
  assert "g_stage1" not in cfg
  lo, hi = stage_id_bounds(cfg)
  assert lo == 0
  assert hi == 6
  assert cfg["g_min"] >= 0
  assert cfg["stage5"]["pe_min"] > 0
  assert cfg["stage6"]["pe_min"] > 0
  assert cfg["stage6"]["mass_diff_min"] > 0
  n_pools = hi - lo + 1
  assert len(cfg["pools"]["train"]) == n_pools


def test_smoke_presets_load() -> None:
  names = list_smoke_presets()
  assert "small" in names
  assert "sanity" in names
  for name in names:
    preset = load_smoke_preset(name)
    assert preset["name"] == name
    assert preset["data"]["train_per_stage"] >= 1
    assert preset["train"]["model"] == "baseline"


def test_model_configs_merge() -> None:
  names = list_model_configs()
  assert names == ["active", "baseline", "curriculum", "progressive"]
  for name in names:
    cfg = load_model_config(name)
    missing = MODEL_REQUIRED - cfg.keys()
    assert not missing, f"{name}.yaml missing after merge: {sorted(missing)}"
  active = load_model_config("active")
  assert active["n_ensemble"] == 2
  assert active["lambda_start"] < active["lambda_end"]
  prog = load_model_config("progressive")
  assert prog["depth_start"] < prog["hidden_depth"]
  assert prog["depth_grow_fracs"]
