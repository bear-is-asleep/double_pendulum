"""Pointwise train/val tensors from ``.npz`` pools (stride k is train-time only).

X columns match ``PoolData.pointwise``: ``(t, IC angles/ω, m1, m2, g)`` — length ``l`` omitted.
Y columns: six sin/cos + ω heads used by the surrogate loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from srcs.simulation.data import PoolData, open_pool
from srcs.simulation.sampler import stage_id_bounds


def _stack_pointwise(
  pool: PoolData,
  t_idx: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """One row per (trajectory, time index in ``t_idx``)."""
  n = pool.n_traj
  nt_sub = int(t_idx.shape[0])
  t_col = np.tile(pool.t[t_idx], n)
  p = np.repeat(pool.params, nt_sub, axis=0)
  x = np.column_stack(
    [
      t_col,
      p[:, 0],
      p[:, 1],
      p[:, 2],
      p[:, 3],
      p[:, 4],
      p[:, 5],
      p[:, 7],
    ]
  )
  y = np.column_stack(
    [
      pool.sin_theta1[:, t_idx].reshape(-1),
      pool.cos_theta1[:, t_idx].reshape(-1),
      pool.sin_theta2[:, t_idx].reshape(-1),
      pool.cos_theta2[:, t_idx].reshape(-1),
      pool.omega1[:, t_idx].reshape(-1),
      pool.omega2[:, t_idx].reshape(-1),
    ]
  )
  return x.astype(np.float64, copy=False), y.astype(np.float64, copy=False)


def pointwise_subsample(
  pool: PoolData,
  stride_k: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  """Every ``stride_k``-th time index per trajectory; IC columns unchanged."""
  if stride_k < 1:
    raise ValueError(f"stride_k must be >= 1, got {stride_k}")
  if pool.n_traj == 0:
    return pool.pointwise()
  t_idx = np.arange(0, pool.n_t, stride_k, dtype=np.int64)
  return _stack_pointwise(pool, t_idx)


def concat_xy(
  pairs: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  if not pairs:
    return (
      np.zeros((0, 8), dtype=np.float64),
      np.zeros((0, 6), dtype=np.float64),
    )
  xs, ys = zip(*pairs)
  return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_stage_split_xy(
  data_root: Path | str,
  stage: int,
  split: str,
  stride_k: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  pool = open_pool(data_root, stage, split)
  return pointwise_subsample(pool, stride_k)


def load_mixed_xy(
  data_root: Path | str,
  stages: list[int],
  split: str,
  stride_k: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  pairs = [load_stage_split_xy(data_root, s, split, stride_k) for s in stages]
  return concat_xy(pairs)


def curriculum_stage_ids(sampler_cfg: dict) -> list[int]:
  lo, hi = stage_id_bounds(sampler_cfg)
  return list(range(lo, hi + 1))


def make_loader(
  x: NDArray[np.float64],
  y: NDArray[np.float64],
  *,
  batch_size: int,
  shuffle: bool,
  seed: int,
) -> DataLoader:
  if x.shape[0] == 0:
    raise ValueError("empty training set")
  gen = torch.Generator()
  gen.manual_seed(int(seed))
  ds = TensorDataset(
    torch.from_numpy(x.astype(np.float32, copy=False)),
    torch.from_numpy(y.astype(np.float32, copy=False)),
  )
  return DataLoader(
    ds,
    batch_size=int(batch_size),
    shuffle=shuffle,
    generator=gen if shuffle else None,
  )
