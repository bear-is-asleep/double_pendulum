"""Step 2: sampler + model YAML keys load and merge."""

from __future__ import annotations

from configs.loader import list_model_configs, load_model_config, load_sampler_config

SAMPLER_REQUIRED = {
  "seed",
  "T",
  "dt",
  "l",
  "g_min",
  "g_max",
  "g_low",
  "g_high",
  "g_stage1",
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
  "num_stages",
  "stage1",
  "stage2",
  "stage3",
  "stage4",
  "stage5",
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
  assert cfg["g_min"] >= 0
  assert cfg["stage4"]["pe_min"] > 0
  assert cfg["stage5"]["pe_min"] > 0
  assert cfg["stage5"]["mass_diff_min"] > 0
  assert len(cfg["pools"]["train"]) == cfg["num_stages"]


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
