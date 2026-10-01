"""Unit tests for shared train/epoch helpers."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from srcs.model import build_mlp, weighted_surrogate_loss
from srcs.train.epoch import (
  BestCheckpointTracker,
  LossWeights,
  eval_loader_mse,
  run_train_epoch,
)
from srcs.loader import load_model_config


def test_run_train_epoch_increments_steps() -> None:
  cfg = load_model_config("baseline")
  model = build_mlp({**cfg, "hidden_width": 16, "hidden_depth": 1})
  opt = torch.optim.SGD(model.parameters(), lr=0.01)
  x = torch.randn(20, int(cfg["input_dim"]))
  y = torch.randn(20, int(cfg["output_dim"]))
  loader = DataLoader(TensorDataset(x, y), batch_size=5)
  weights = LossWeights(theta=1.0, omega=0.25)
  steps = run_train_epoch(model, loader, torch.device("cpu"), opt, weights)
  assert steps == 4


def test_best_tracker_improves_on_lower_val() -> None:
  cfg = load_model_config("baseline")
  model = build_mlp({**cfg, "hidden_width": 8, "hidden_depth": 1})
  tracker = BestCheckpointTracker()
  assert tracker.observe(2.0, model)
  assert tracker.observe(1.0, model)
  assert tracker.best_mean == 1.0
  assert not tracker.observe(1.5, model)
  assert tracker.stale_epochs == 1


def test_eval_loader_matches_manual_mean() -> None:
  cfg = load_model_config("baseline")
  model = build_mlp({**cfg, "hidden_width": 8, "hidden_depth": 1})
  in_dim = int(cfg["input_dim"])
  x = torch.randn(6, in_dim)
  y = torch.randn(6, 6)
  loader = DataLoader(TensorDataset(x, y), batch_size=3)
  weights = LossWeights(theta=1.0, omega=0.25)
  got = eval_loader_mse(model, loader, torch.device("cpu"), weights)
  model.eval()
  with torch.no_grad():
    out = model(x)
    expected = float(
      weighted_surrogate_loss(
        out,
        y,
        loss_weight_theta=1.0,
        loss_weight_omega=0.25,
      ).item()
    )
  assert abs(got - expected) < 1e-5
