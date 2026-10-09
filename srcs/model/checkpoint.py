"""Rebuild MLP from run checkpoints and run pointwise inference (Step 8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from srcs.model.mlp import build_mlp
from srcs.model.progressive_net import ProgressiveNet, build_progressive_net, pnn_stage_ready
from srcs.utils.paths import (
  load_run_config,
  resolve_checkpoint_file,
  resolve_run_dir,
)

__all__ = [
  "build_pointwise_inputs",
  "load_model_from_checkpoint",
  "predict_at_times",
  "predict_one",
  "resolve_checkpoint_file",
  "resolve_run_dir",
]


def build_pointwise_inputs(
  t: NDArray[np.float64],
  params_row: NDArray[np.float64],
) -> NDArray[np.float64]:
  """
  Model input rows for one trajectory IC row.

  ``params_row`` columns match ``data.PARAM_COLS``; length ``l`` (index 6) omitted.
  """
  row = np.asarray(params_row, dtype=np.float64).reshape(-1)
  if row.shape[0] < 8:
    raise ValueError(f"params_row must have 8 columns, got {row.shape}")
  tt = np.asarray(t, dtype=np.float64).reshape(-1)
  return np.column_stack(
    [
      tt,
      np.full(tt.shape[0], row[0]),
      np.full(tt.shape[0], row[1]),
      np.full(tt.shape[0], row[2]),
      np.full(tt.shape[0], row[3]),
      np.full(tt.shape[0], row[4]),
      np.full(tt.shape[0], row[5]),
      np.full(tt.shape[0], row[7]),
    ]
  )


def load_model_from_checkpoint(
  checkpoint_path: Path | str,
  *,
  device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any], Path]:
  """Rebuild ``nn.Module`` and return ``(model, config, ckpt_file)``."""
  ckpt_file = resolve_checkpoint_file(checkpoint_path)
  run_dir = resolve_run_dir(ckpt_file)
  ckpt = torch.load(ckpt_file, map_location=device, weights_only=True)
  if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
    raise KeyError(f"{ckpt_file}: expected dict with model_state_dict")
  cfg = load_run_config(run_dir, ckpt.get("config"))
  model_type = str(cfg.get("model_type", "mlp"))
  if model_type == "progressive_pnn":
    model = build_progressive_net(cfg)
  else:
    model = build_mlp(cfg)
  model.load_state_dict(ckpt["model_state_dict"])
  model.to(device)
  model.eval()
  meta = {
    "config": cfg,
    "val_metric": ckpt.get("val_metric"),
    "epoch": ckpt.get("epoch"),
    "run_dir": str(run_dir),
    "checkpoint": str(ckpt_file),
  }
  return model, meta, ckpt_file


def inference_column_index(cfg: dict[str, Any], stage: int) -> int | None:
  """PNN eval uses column ``stage``; vanilla MLP uses default ``model(x)``."""
  if str(cfg.get("model_type", "mlp")) != "progressive_pnn":
    return None
  return int(stage)


@torch.no_grad()
def predict_at_times(
  model: nn.Module,
  t: NDArray[np.float64],
  params_row: NDArray[np.float64],
  *,
  device: str | torch.device = "cpu",
  batch_size: int = 4096,
  column_index: int | None = None,
) -> NDArray[np.float64]:
  """Return shape ``(len(t), 6)`` surrogate targets."""
  x = build_pointwise_inputs(t, params_row)
  if x.shape[0] == 0:
    return np.zeros((0, 6), dtype=np.float64)
  out_parts: list[NDArray[np.float64]] = []
  for start in range(0, x.shape[0], batch_size):
    chunk = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False)).to(device)
    if isinstance(model, ProgressiveNet):
      if column_index is None:
        raise TypeError(
          "ProgressiveNet inference requires column_index (use stage id for test eval)"
        )
      if not pnn_stage_ready(model, int(column_index)):
        raise IndexError(
          f"checkpoint has n_columns={model.n_columns}, cannot infer column {column_index}"
        )
      pred = model.forward(chunk, column_index=int(column_index))
    else:
      if column_index is not None:
        raise TypeError("column_index is only valid for ProgressiveNet checkpoints")
      pred = model(chunk)
    pred = pred.cpu().numpy()
    out_parts.append(pred.astype(np.float64, copy=False))
  return np.concatenate(out_parts, axis=0)


@torch.no_grad()
def predict_one(
  model: nn.Module,
  t: float,
  params_row: NDArray[np.float64],
  *,
  device: str | torch.device = "cpu",
) -> NDArray[np.float64]:
  """Single time step; shape ``(6,)``. Used by live ``app.py`` in v2."""
  arr = predict_at_times(
    model,
    np.array([t], dtype=np.float64),
    params_row,
    device=device,
    batch_size=1,
  )
  return arr[0]
