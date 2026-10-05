"""MLP width specs: scalar vs per-layer list."""

from __future__ import annotations

from srcs.model.mlp import build_mlp, resolve_hidden_layer_widths
from srcs.train.run_dir import strategy_run_id


def _tiny_cfg(**overrides):
  base = {
    "input_dim": 4,
    "output_dim": 2,
    "hidden_width": 8,
    "hidden_depth": 2,
    "activation": "tanh",
  }
  base.update(overrides)
  return base


def test_scalar_width_repeats_per_layer() -> None:
  cfg = _tiny_cfg(hidden_width=16, hidden_depth=3)
  assert resolve_hidden_layer_widths(cfg) == [16, 16, 16]
  model = build_mlp(cfg)
  assert list(model.children())[0].out_features == 16
  assert list(model.children())[2].out_features == 16


def test_list_width_per_layer() -> None:
  cfg = _tiny_cfg(hidden_width=[32, 16], hidden_depth=2)
  assert resolve_hidden_layer_widths(cfg) == [32, 16]
  model = build_mlp(cfg)
  assert list(model.children())[0].out_features == 32
  assert list(model.children())[2].in_features == 32
  assert list(model.children())[2].out_features == 16
  assert list(model.children())[4].in_features == 16


def test_effective_depth_truncates_list() -> None:
  cfg = _tiny_cfg(hidden_width=[32, 16, 8], hidden_depth=3, effective_depth=2)
  assert resolve_hidden_layer_widths(cfg) == [32, 16]


def test_strategy_run_id_list_width() -> None:
  cfg = _tiny_cfg(hidden_width=[128, 64], hidden_depth=2, subsample_stride_k=4, seed=0)
  rid = strategy_run_id(cfg, "curriculum")
  assert "w128x64" in rid
