"""Checkpoint load and inference smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from srcs.model.checkpoint import (
  build_pointwise_inputs,
  load_model_from_checkpoint,
  predict_at_times,
  resolve_checkpoint_file,
)
from srcs.model.mlp import build_mlp
from srcs.model.progressive_net import build_progressive_net
from srcs.loader import load_model_config
from srcs.train.run_dir import init_run_dir, save_checkpoint
def test_build_pointwise_inputs_shape() -> None:
  t = np.linspace(0, 1, 5)
  row = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 1.0, 9.8])
  x = build_pointwise_inputs(t, row)
  assert x.shape == (5, 8)
  assert np.allclose(x[:, 0], t)
  assert np.allclose(x[:, 1], 0.1)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
  cfg = load_model_config("baseline")
  cfg["hidden_width"] = 16
  cfg["hidden_depth"] = 2
  run_id = "test_ckpt_roundtrip"
  run_dir = init_run_dir(tmp_path, run_id, cfg)
  model = build_mlp(cfg)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  ckpt_path = run_dir / "checkpoints" / "best.pt"
  ckpt_path.parent.mkdir(parents=True, exist_ok=True)
  save_checkpoint(
    ckpt_path,
    model=model,
    optimizer=opt,
    epoch=1,
    global_step=10,
    val_metric=0.5,
    cfg=cfg,
  )
  resolved = resolve_checkpoint_file(ckpt_path)
  assert resolved == ckpt_path.resolve()
  loaded, meta, _ = load_model_from_checkpoint(ckpt_path)
  assert isinstance(loaded, nn.Module)
  assert meta["val_metric"] == pytest.approx(0.5)
  t = np.array([0.0, 0.1, 0.2])
  row = np.zeros(8, dtype=np.float64)
  row[7] = 9.8
  out = predict_at_times(loaded, t, row)
  assert out.shape == (3, 6)
  assert np.all(np.isfinite(out))


def test_progressive_checkpoint_predict_needs_column(tmp_path: Path) -> None:
  cfg = load_model_config("progressive")
  cfg["hidden_width"] = 12
  cfg["hidden_depth"] = 1
  cfg["n_columns"] = 2
  run_dir = init_run_dir(tmp_path, "pnn_ckpt", cfg)
  model = build_progressive_net(cfg)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  ckpt_path = run_dir / "checkpoints" / "best.pt"
  ckpt_path.parent.mkdir(parents=True, exist_ok=True)
  save_checkpoint(
    ckpt_path,
    model=model,
    optimizer=opt,
    epoch=1,
    global_step=1,
    val_metric=0.1,
    cfg=cfg,
  )
  loaded, meta, _ = load_model_from_checkpoint(ckpt_path)
  assert meta["config"]["model_type"] == "progressive_pnn"
  t = np.array([0.0, 0.05])
  row = np.zeros(8, dtype=np.float64)
  row[7] = 9.8
  with pytest.raises(TypeError, match="column_index"):
    predict_at_times(loaded, t, row)
  out = predict_at_times(loaded, t, row, column_index=1)
  assert out.shape == (2, 6)
