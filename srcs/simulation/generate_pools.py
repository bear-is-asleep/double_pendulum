"""Shared pool generation (used by ``generate_data`` and smoke presets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from srcs.loader import load_model_config, load_sampler_config
from srcs.simulation.data import generate_all_stages
from srcs.simulation.sampler import pool_index, stage_id_bounds


def parse_int_list(text: str | None) -> list[int] | None:
  if text is None:
    return None
  parts = [p.strip() for p in text.split(",") if p.strip()]
  return [int(p) for p in parts]


def parse_stages(text: str | None, sampler_cfg: dict[str, Any]) -> list[int]:
  lo, hi = stage_id_bounds(sampler_cfg)
  if text is None:
    return list(range(lo, hi + 1))
  stages = parse_int_list(text)
  assert stages is not None
  for s in stages:
    if s < lo or s > hi:
      raise ValueError(f"stage {s} out of {lo}..{hi}")
  return stages


def pad_counts_for_stages(
  stages: list[int],
  counts: list[int],
  sampler_cfg: dict[str, Any],
  pool_key: str,
) -> list[int]:
  lo, hi = stage_id_bounds(sampler_cfg)
  n_pools = hi - lo + 1
  if len(counts) == n_pools:
    return counts
  if len(counts) != len(stages):
    raise ValueError(
      f"counts length {len(counts)} must match stages {len(stages)} or all pool slots"
    )
  full = list(sampler_cfg["pools"][pool_key])
  for stage, n in zip(stages, counts):
    full[pool_index(stage, sampler_cfg)] = n
  return full


def generate_pools(
  data_root: Path | str,
  *,
  stages: list[int] | None = None,
  train_counts: list[int] | None = None,
  test_counts: list[int] | None = None,
  val_fraction: float | None = None,
  model: str = "baseline",
  seed: int | None = None,
  overwrite_frozen: bool = False,
) -> dict[int, dict[str, Path]]:
  sampler_cfg = load_sampler_config()
  model_cfg = load_model_config(model)
  lo, hi = stage_id_bounds(sampler_cfg)
  stage_list = list(range(lo, hi + 1)) if stages is None else list(stages)
  for s in stage_list:
    if s < lo or s > hi:
      raise ValueError(f"stage {s} out of {lo}..{hi}")

  vf = (
    float(val_fraction)
    if val_fraction is not None
    else float(model_cfg["val_fraction"])
  )

  train_ns = train_counts
  test_ns = test_counts
  if train_ns is not None:
    train_ns = pad_counts_for_stages(stage_list, train_ns, sampler_cfg, "train")
  if test_ns is not None:
    test_ns = pad_counts_for_stages(stage_list, test_ns, sampler_cfg, "test")

  data_root = Path(data_root)
  data_root.mkdir(parents=True, exist_ok=True)
  return generate_all_stages(
    data_root,
    sampler_cfg,
    val_fraction=vf,
    stages=stage_list,
    train_counts=train_ns,
    test_counts=test_ns,
    seed=seed,
    overwrite_frozen=overwrite_frozen,
  )
