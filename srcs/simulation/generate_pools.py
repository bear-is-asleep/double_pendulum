"""Shared pool generation for ``generate_data`` CLI and library callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from srcs.loader import (
  DEFAULT_CONFIG_STEM,
  load_model_config,
  load_sampler_config,
  paired_data_config,
  sampler_val_fraction,
)
from srcs.utils.paths import ensure_dir
from srcs.utils.yaml_io import print_mapping_yaml
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


@dataclass(frozen=True)
class PoolGenerateJob:
  """Resolved inputs for one ``generate_pools`` run."""

  data_root: Path
  stages: list[int] | None = None
  train_counts: list[int] | None = None
  test_counts: list[int] | None = None
  val_fraction: float | None = None
  model: str = "baseline"
  data_config: str = DEFAULT_CONFIG_STEM
  train_stem: str | None = None
  seed: int | None = None
  overwrite_frozen: bool = False

  @classmethod
  def from_train_job(
    cls,
    train_job: dict[str, Any],
    *,
    data_root: Path | None = None,
    data_config: str | None = None,
    train_stem: str | None = None,
    seed: int | None = None,
    overwrite_frozen: bool = False,
  ) -> PoolGenerateJob:
    """Stages from train YAML; pool sizes and val_fraction from ``configs/data``."""
    stages = [int(s) for s in train_job["stages"]]
    stem = train_stem
    data_cfg = data_config or paired_data_config(stem, None)
    root = data_root if data_root is not None else Path(train_job["data_root"])
    return cls(
      data_root=root,
      stages=stages,
      train_counts=None,
      test_counts=None,
      val_fraction=None,
      data_config=data_cfg,
      train_stem=stem,
      seed=seed,
      overwrite_frozen=overwrite_frozen,
    )


def generate_pools_from_job(job: PoolGenerateJob) -> dict[int, dict[str, Path]]:
  return generate_pools(
    job.data_root,
    stages=job.stages,
    train_counts=job.train_counts,
    test_counts=job.test_counts,
    val_fraction=job.val_fraction,
    model=job.model,
    data_config=job.data_config,
    train_stem=job.train_stem,
    seed=job.seed,
    overwrite_frozen=job.overwrite_frozen,
  )


def generate_pools(
  data_root: Path | str,
  *,
  stages: list[int] | None = None,
  train_counts: list[int] | None = None,
  test_counts: list[int] | None = None,
  val_fraction: float | None = None,
  model: str = "baseline",
  data_config: str = DEFAULT_CONFIG_STEM,
  train_stem: str | None = None,
  seed: int | None = None,
  overwrite_frozen: bool = False,
) -> dict[int, dict[str, Path]]:
  sampler_cfg = load_sampler_config(data_config=data_config)
  model_cfg = load_model_config(model)
  lo, hi = stage_id_bounds(sampler_cfg)
  stage_list = list(range(lo, hi + 1)) if stages is None else list(stages)
  for s in stage_list:
    if s < lo or s > hi:
      raise ValueError(f"stage {s} out of {lo}..{hi}")

  vf = (
    float(val_fraction)
    if val_fraction is not None
    else sampler_val_fraction(sampler_cfg)
  )

  train_ns = train_counts
  test_ns = test_counts
  if train_ns is not None:
    train_ns = pad_counts_for_stages(stage_list, train_ns, sampler_cfg, "train")
  if test_ns is not None:
    test_ns = pad_counts_for_stages(stage_list, test_ns, sampler_cfg, "test")

  data_root = ensure_dir(data_root)
  print_mapping_yaml(
    "data generation parameters",
    {
      "train_stem": train_stem,
      "data_config": data_config,
      "model": model,
      "data_root": data_root,
      "stages": stage_list,
      "train_counts": train_ns,
      "test_counts": test_ns,
      "val_fraction": vf,
      "seed": seed,
      "overwrite_frozen": overwrite_frozen,
      "sampler": sampler_cfg,
      "model_cfg": model_cfg,
    },
  )
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
