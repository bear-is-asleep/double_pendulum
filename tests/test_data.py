"""Step 4: ``.npz`` pools, freeze, val carve, pointwise expand."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from srcs.loader import load_model_config, load_sampler_config
from srcs.simulation.data import (
  FrozenPoolError,
  build_pool,
  carve_validation,
  freeze_pool,
  generate_stage,
  list_pools,
  load_pool,
  open_pool,
  pool_path,
  save_pool,
)
from srcs.physics import wrap_angle


def _tiny_sampler_cfg() -> dict:
  cfg = copy.deepcopy(load_sampler_config())
  cfg["T"] = 0.5
  cfg["dt"] = 0.05
  cfg["probe_time"] = 0.25
  cfg["max_draw_attempts"] = 200
  cfg["seed"] = 7
  return cfg


def test_build_pool_shapes_and_wrap() -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(2, 3, cfg, split="train", rng=np.random.default_rng(0))
  nt = int(np.round(cfg["T"] / cfg["dt"])) + 1
  assert pool.n_traj == 3
  assert pool.n_t == nt
  assert pool.params.shape == (3, 8)
  assert pool.sin_theta1.shape == (3, nt)
  assert pool.t.shape == (nt,)
  # wrapped debug angles in (-pi, pi]
  assert np.all(pool.theta1 > -np.pi - 1e-12)
  assert np.all(pool.theta1 <= np.pi + 1e-12)
  # sin/cos match wrapped theta
  assert np.allclose(pool.sin_theta1, np.sin(pool.theta1))
  assert np.allclose(pool.cos_theta1, np.cos(pool.theta1))


def test_pointwise_dims() -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 2, cfg, rng=np.random.default_rng(1))
  x, y = pool.pointwise()
  assert x.shape == (2 * pool.n_t, 8)
  assert y.shape == (2 * pool.n_t, 6)
  # first time sample of traj 0: t=0, IC from params
  assert x[0, 0] == pool.t[0]
  np.testing.assert_allclose(x[0, 1:5], pool.params[0, 0:4])
  np.testing.assert_allclose(x[0, 5:7], pool.params[0, 4:6])  # m1, m2
  assert x[0, 7] == pool.params[0, 7]  # g (l omitted)
  np.testing.assert_allclose(
    y[0],
    [
      pool.sin_theta1[0, 0],
      pool.cos_theta1[0, 0],
      pool.sin_theta2[0, 0],
      pool.cos_theta2[0, 0],
      pool.omega1[0, 0],
      pool.omega2[0, 0],
    ],
  )


def test_carve_validation_disjoint() -> None:
  cfg = _tiny_sampler_cfg()
  raw = build_pool(2, 10, cfg, rng=np.random.default_rng(2))
  train, val = carve_validation(raw, 0.3, np.random.default_rng(3))
  assert train.n_traj + val.n_traj == 10
  assert val.n_traj == 3
  assert train.split == "train"
  assert val.split == "val"
  # no shared param rows
  train_set = {tuple(r) for r in train.params}
  val_set = {tuple(r) for r in val.params}
  assert train_set.isdisjoint(val_set)


def test_save_load_roundtrip(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(3, 2, cfg, rng=np.random.default_rng(4))
  path = pool_path(tmp_path, 3, "train")
  save_pool(path, pool)
  loaded = load_pool(path)
  assert loaded.stage == 3
  assert loaded.split == "train"
  assert loaded.frozen is False
  np.testing.assert_allclose(loaded.params, pool.params)
  np.testing.assert_allclose(loaded.energy, pool.energy)


def test_frozen_refuse_overwrite(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  pool = freeze_pool(build_pool(1, 1, cfg, split="test", rng=np.random.default_rng(5)))
  path = pool_path(tmp_path, 1, "test")
  save_pool(path, pool)
  assert load_pool(path).frozen is True
  other = build_pool(1, 1, cfg, split="test", rng=np.random.default_rng(6))
  with pytest.raises(FrozenPoolError):
    save_pool(path, other)
  # force allowed
  save_pool(path, freeze_pool(other), overwrite_frozen=True)
  assert open_pool(tmp_path, 1, "test").n_traj == 1


def test_generate_stage_writes_three_splits(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  model = load_model_config("baseline")
  paths = generate_stage(
    2,
    tmp_path,
    cfg,
    train_n=6,
    test_n=2,
    val_fraction=float(model["val_fraction"]),
    rng=np.random.default_rng(8),
  )
  assert set(paths) == {"train", "val", "test"}
  train = open_pool(tmp_path, 2, "train")
  val = open_pool(tmp_path, 2, "val")
  test = open_pool(tmp_path, 2, "test")
  assert train.n_traj + val.n_traj == 6
  assert test.n_traj == 2
  assert test.frozen is True
  assert train.frozen is False
  listed = list_pools(tmp_path)
  assert [(s, sp) for s, sp, _ in listed] == [
    (2, "train"),
    (2, "val"),
    (2, "test"),
  ]


def test_get_traj_frame_at() -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(2, 1, cfg, rng=np.random.default_rng(9))
  view = pool.get_traj(0)
  ic = view.initial_state()
  assert ic.theta1 == pool.params[0, 0]
  frame0 = view.frame_at(0)
  assert abs(wrap_angle(frame0.theta1) - pool.theta1[0, 0]) < 1e-9
  assert frame0.omega1 == pool.omega1[0, 0]


def test_list_pools_empty(tmp_path) -> None:
  assert list_pools(tmp_path) == []
