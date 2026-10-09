"""ProgressiveNet column + lateral smoke tests."""

from __future__ import annotations

import copy

import torch

from srcs.model.progressive_net import (
  ProgressiveNet,
  build_progressive_net,
  count_parameters,
)


def _tiny_cfg(**overrides):
  base = {
    "input_dim": 8,
    "output_dim": 6,
    "hidden_width": 16,
    "hidden_depth": 2,
    "activation": "tanh",
    "dropout": 0.0,
    "use_lateral": True,
    "n_columns": 1,
  }
  base.update(overrides)
  return base


def test_forward_shape_single_column() -> None:
  cfg = _tiny_cfg()
  net = build_progressive_net(cfg)
  x = torch.randn(5, 8)
  y = net.forward(x, column_index=0)
  assert y.shape == (5, 6)


def test_add_column_increases_params() -> None:
  cfg = _tiny_cfg(n_columns=1)
  net = build_progressive_net(cfg)
  n0 = count_parameters(net, trainable_only=False)
  net.add_column(1)
  n1 = count_parameters(net, trainable_only=False)
  assert n1 > n0
  x = torch.randn(3, 8)
  out = net.forward(x, column_index=1)
  assert out.shape == (3, 6)


def test_lateral_off_matches_independent_columns() -> None:
  cfg_lat = _tiny_cfg(use_lateral=True, n_columns=2)
  cfg_off = _tiny_cfg(use_lateral=False, n_columns=2)
  torch.manual_seed(0)
  net_lat = build_progressive_net(cfg_lat)
  torch.manual_seed(0)
  net_off = build_progressive_net(cfg_off)
  x = torch.randn(4, 8)
  y_lat = net_lat.forward(x, column_index=1)
  y_off = net_off.forward(x, column_index=1)
  assert y_lat.shape == y_off.shape
  assert not torch.allclose(y_lat, y_off)


def test_freeze_columns_before_blocks_grad() -> None:
  cfg = _tiny_cfg(n_columns=1)
  net = build_progressive_net(cfg)
  net.add_column(1)
  net.freeze_columns_before(1)
  w0 = net.columns[0].blocks[0].fc.weight
  assert not w0.requires_grad
  w1 = net.columns[1].blocks[0].fc.weight
  assert w1.requires_grad
  x = torch.randn(2, 8, requires_grad=False)
  out = net.forward(x, column_index=1)
  loss = out.sum()
  loss.backward()
  assert w0.grad is None
  assert w1.grad is not None


def test_frozen_column_weights_unchanged_after_train_column1() -> None:
  cfg = _tiny_cfg(n_columns=1, hidden_width=8, hidden_depth=1)
  net = build_progressive_net(cfg)
  snap0 = copy.deepcopy(net.columns[0].state_dict())
  net.add_column(1)
  net.freeze_columns_before(1)
  opt = torch.optim.SGD(
    [p for p in net.parameters() if p.requires_grad],
    lr=0.05,
  )
  for _ in range(5):
    x = torch.randn(8, 8)
    pred = net.forward(x, column_index=1)
    loss = pred.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
  for key, val in snap0.items():
    assert torch.allclose(net.columns[0].state_dict()[key], val)


def test_add_column_matches_existing_device() -> None:
  cfg = _tiny_cfg(n_columns=1, hidden_width=8, hidden_depth=1)
  net = build_progressive_net(cfg)
  net.to(torch.device("cpu"))
  anchor = next(net.parameters()).device
  net.add_column(1)
  new_dev = next(net.columns[1].parameters()).device
  assert new_dev == anchor
  x = torch.randn(3, 8, device=anchor)
  out = net.forward(x, column_index=1)
  assert out.shape == (3, 6)


def test_add_column_forward_on_mps_if_available() -> None:
  if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
    return
  cfg = _tiny_cfg(n_columns=1, hidden_width=8, hidden_depth=1, dropout=0.0)
  net = build_progressive_net(cfg).to("mps")
  net.add_column(1)
  net.freeze_columns_before(1)
  x = torch.randn(4, 8, device="mps")
  out = net.forward_batch_by_stage(x, torch.zeros(4, dtype=torch.long, device="mps"))
  assert out.device.type == "mps"
  assert out.shape == (4, 6)


def test_forward_batch_by_stage() -> None:
  cfg = _tiny_cfg(n_columns=2)
  net = build_progressive_net(cfg)
  x = torch.randn(6, 8)
  stages = torch.tensor([0, 0, 1, 1, 0, 1])
  batched = net.forward_batch_by_stage(x, stages)
  manual = torch.empty_like(batched)
  for i in range(6):
    manual[i] = net.forward(x[i : i + 1], column_index=int(stages[i]))
  assert torch.allclose(batched, manual, atol=1e-5)
