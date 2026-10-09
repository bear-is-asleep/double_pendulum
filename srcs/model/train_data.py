"""Pointwise train/val tensors from ``.npz`` pools (stride k is train-time only).

X columns match ``PoolData.pointwise``: ``(t, IC angles/omega, m1, m2, g)`` - length ``l`` omitted.
Y columns: six sin/cos + omega heads used by the surrogate loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Sampler, TensorDataset

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


def subsampled_points_per_trajectory(pool: PoolData, stride_k: int) -> int:
  """Time rows per trajectory after stride ``k`` (train-time subsample)."""
  if stride_k < 1:
    raise ValueError(f"stride_k must be >= 1, got {stride_k}")
  if pool.n_traj == 0:
    return 0
  return int(np.arange(0, pool.n_t, stride_k, dtype=np.int64).shape[0])


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


def fractions_from_stage_row_counts(counts: dict[int, int]) -> dict[int, float]:
  """Normalize per-stage train row counts to sampling fractions."""
  total = sum(counts.values())
  if total <= 0:
    raise ValueError("empty mixed train set across stages")
  return {stage: n / total for stage, n in counts.items() if n > 0}


BASELINE_STAGE_MIX_PROPORTIONAL = "proportional"
BASELINE_STAGE_MIX_EQUAL = "equal"


def resolve_baseline_stage_mix(cfg: dict) -> str:
  """Baseline mixed-pool train sampling: row counts (default) or equal per stage."""
  mode = str(cfg.get("baseline_stage_mix", BASELINE_STAGE_MIX_PROPORTIONAL)).strip().lower()
  if mode not in (BASELINE_STAGE_MIX_PROPORTIONAL, BASELINE_STAGE_MIX_EQUAL):
    raise ValueError(
      "baseline_stage_mix must be "
      f"'{BASELINE_STAGE_MIX_PROPORTIONAL}' or '{BASELINE_STAGE_MIX_EQUAL}', got {mode!r}"
    )
  return mode


def baseline_stage_mix_fractions(
  stages: list[int],
  row_counts: dict[int, int],
  mode: str,
) -> dict[int, float]:
  """Stage weights for training sampler and weighted val MSE (must match train_loss)."""
  if mode == BASELINE_STAGE_MIX_EQUAL:
    if not stages:
      raise ValueError("baseline equal stage mix requires at least one stage")
    share = 1.0 / float(len(stages))
    return {int(s): share for s in stages}
  return fractions_from_stage_row_counts(row_counts)


def mixed_train_stage_fractions(
  data_root: Path | str,
  stages: list[int],
  stride_k: int,
) -> dict[int, float]:
  """Row-count weights for uniform shuffle over concatenated stage train pools."""
  counts: dict[int, int] = {}
  for stage in stages:
    x, _ = load_stage_split_xy(data_root, stage, "train", stride_k)
    counts[stage] = int(x.shape[0])
  return fractions_from_stage_row_counts(counts)


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


class _StageMixSampler(Sampler[int]):
  """Pick stage by ``fractions``, then a uniform row inside that stage's concat block."""

  def __init__(
    self,
    spans: list[tuple[int, int]],
    stage_probs: torch.Tensor,
    num_samples: int,
    generator: torch.Generator,
  ) -> None:
    self._spans = spans
    self._stage_probs = stage_probs
    self._num_samples = int(num_samples)
    self._generator = generator

  def __iter__(self):
    for _ in range(self._num_samples):
      si = int(torch.multinomial(self._stage_probs, 1, generator=self._generator).item())
      lo, hi = self._spans[si]
      yield int(torch.randint(lo, hi, (1,), generator=self._generator).item())

  def __len__(self) -> int:
    return self._num_samples


def _make_stage_mix_loader(
  stage_xy: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
  fractions: dict[int, float],
  *,
  batch_size: int,
  seed: int,
  eps: float,
  tag_stages: bool,
) -> DataLoader:
  active = sorted(
    s for s, f in fractions.items() if f > eps and s in stage_xy
  )
  if not active:
    raise ValueError("no active stages with training data")

  xs: list[NDArray[np.float64]] = []
  ys: list[NDArray[np.float64]] = []
  stage_cols: list[NDArray[np.int64]] = []
  spans: list[tuple[int, int]] = []
  probs: list[float] = []
  offset = 0
  for stage in active:
    x, y = stage_xy[stage]
    n = int(x.shape[0])
    if n == 0:
      continue
    xs.append(x)
    ys.append(y)
    if tag_stages:
      stage_cols.append(np.full(n, int(stage), dtype=np.int64))
    spans.append((offset, offset + n))
    probs.append(float(fractions[stage]))
    offset += n
  if not xs:
    raise ValueError("empty training set across active stages")

  x_all = np.concatenate(xs, axis=0)
  y_all = np.concatenate(ys, axis=0)
  stage_probs = torch.as_tensor(probs, dtype=torch.double)
  stage_probs = stage_probs / stage_probs.sum()

  gen = torch.Generator()
  gen.manual_seed(int(seed))
  sampler = _StageMixSampler(spans, stage_probs, int(x_all.shape[0]), gen)

  fields = [
    torch.from_numpy(x_all.astype(np.float32, copy=False)),
    torch.from_numpy(y_all.astype(np.float32, copy=False)),
  ]
  if tag_stages:
    fields.append(torch.from_numpy(np.concatenate(stage_cols, axis=0)))
  return DataLoader(
    TensorDataset(*fields),
    batch_size=int(batch_size),
    sampler=sampler,
  )


def make_weighted_stage_loader(
  stage_xy: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
  fractions: dict[int, float],
  *,
  batch_size: int,
  seed: int,
  eps: float = 1.0e-12,
) -> DataLoader:
  """Train loader: stage mix ``fractions``, uniform rows within each stage."""
  return _make_stage_mix_loader(
    stage_xy,
    fractions,
    batch_size=batch_size,
    seed=seed,
    eps=eps,
    tag_stages=False,
  )


def make_weighted_stage_loader_tagged(
  stage_xy: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]],
  fractions: dict[int, float],
  *,
  batch_size: int,
  seed: int,
  eps: float = 1.0e-12,
) -> DataLoader:
  """Same as ``make_weighted_stage_loader`` with a third tensor ``stage_id`` per row."""
  return _make_stage_mix_loader(
    stage_xy,
    fractions,
    batch_size=batch_size,
    seed=seed,
    eps=eps,
    tag_stages=True,
  )


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
