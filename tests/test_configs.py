"""Step 2: sampler + model YAML keys load and merge."""

from __future__ import annotations

from pathlib import Path

from srcs.loader import (
  DEFAULT_CONFIG_STEM,
  list_configs,
  load_model_config,
  load_train_config,
  load_sampler_config,
  paired_data_config,
  sampler_val_fraction,
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
  "val_fraction",
}

def test_sampler_yaml_loads() -> None:
  cfg = load_sampler_config()
  missing = SAMPLER_REQUIRED - cfg.keys()
  assert not missing, f"data/full.yaml missing keys: {sorted(missing)}"
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


def test_data_configs_load() -> None:
  names = list_configs("data")
  assert "full" in names
  assert "small" in names
  assert "sanity" in names
  full = load_sampler_config(data_config="full")
  small = load_sampler_config(data_config="small")
  assert full["pools"]["train"] != small["pools"]["train"]
  assert small["pools"]["train"][0] < full["pools"]["train"][0]


def test_progressive_model_inherits_curriculum_inlet() -> None:
  curriculum = load_model_config("curriculum")
  progressive = load_model_config("progressive")
  assert progressive["strategy"] == "progressive"
  assert progressive["model_type"] == "progressive_pnn"
  assert progressive["mix_inlet_gain"] == curriculum["mix_inlet_gain"]
  assert progressive["passed_stage_min_fraction"] == curriculum["passed_stage_min_fraction"]


def test_train_job_configs_load() -> None:
  names = list_configs("train")
  assert "small" in names
  for name in names:
    job = load_train_config(name)
    assert job["stages"]
    assert "data" not in job
  assert load_train_config("small")["train"]["model"] == "baseline"
  assert load_train_config("curriculum_small")["train"]["model"] == "curriculum"


def test_paired_data_config() -> None:
  assert paired_data_config("small", None) == "small"
  assert paired_data_config("full", None) == "full"
  assert paired_data_config("sanity", None) == "sanity"
  assert paired_data_config("small", "full") == "full"
  assert paired_data_config(None, None) == DEFAULT_CONFIG_STEM


def test_pool_job_from_train_uses_data_yaml_sizes() -> None:
  from srcs.simulation.generate_pools import PoolGenerateJob

  job = PoolGenerateJob.from_train_job(
    load_train_config("small"),
    data_config="small",
    train_stem="small",
  )
  assert job.train_counts is None
  assert job.test_counts is None
  assert job.stages == [0, 1]
  small = load_sampler_config(data_config="small")
  assert sampler_val_fraction(small) == 0.2


def test_pool_job_sanity_pairs_tiny_data_yaml() -> None:
  from srcs.simulation.generate_pools import PoolGenerateJob

  job = PoolGenerateJob.from_train_job(
    load_train_config("sanity"),
    data_config=paired_data_config("sanity", None),
    train_stem="sanity",
  )
  assert job.data_config == "sanity"
  sanity = load_sampler_config(data_config="sanity")
  assert sanity["pools"]["train"][0] == 4


def test_generate_data_accepts_sampler_cfg_flag() -> None:
  from srcs.simulation.generate_data import build_parser

  args = build_parser().parse_args(
    ["--data-root", "data/x", "--data-config", "small", "--overwrite-frozen"]
  )
  assert args.data_config == "small"


def test_generate_data_accepts_data_yaml_path() -> None:
  from srcs.simulation.generate_data import build_parser, pool_job_from_args

  args = build_parser().parse_args(
    ["configs/data/full.yaml", "--data-root", "data/v1", "--seed", "0"]
  )
  job = pool_job_from_args(args)
  assert job.data_config == "full"
  assert job.data_root == Path("data/v1")
  assert job.train_stem is None
