"""Shared one-epoch train/eval for baseline (Step 5) and main train loop (Step 9)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from srcs.model.loss import weighted_surrogate_loss

StageTaggedPredictFn = Callable[[nn.Module, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class LossWeights:
  theta: float
  omega: float


def loss_weights_from_cfg(cfg: dict[str, Any]) -> LossWeights:
  return LossWeights(
    theta=float(cfg["loss_weight_theta"]),
    omega=float(cfg["loss_weight_omega"]),
  )


def adam_on_trainable(model: nn.Module, lr: float) -> optim.Adam:
  """Fresh Adam over parameters that still require grad (PNN column unlock)."""
  params = [p for p in model.parameters() if p.requires_grad]
  return optim.Adam(params, lr=float(lr))


def pick_device(preferred: str | None = None) -> torch.device:
  if preferred:
    return torch.device(preferred)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def run_train_epoch(
  model: nn.Module,
  loader: DataLoader,
  device: torch.device,
  optimizer: optim.Optimizer,
  weights: LossWeights,
) -> int:
  """Run one full pass over ``loader``. Returns optimizer steps taken."""
  model.train()
  steps = 0
  for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = weighted_surrogate_loss(
      pred,
      yb,
      loss_weight_theta=weights.theta,
      loss_weight_omega=weights.omega,
    )
    loss.backward()
    optimizer.step()
    steps += 1
  return steps


def _accumulate_surrogate_mse(
  pred: Tensor,
  yb: Tensor,
  weights: LossWeights,
  total: float,
  n: int,
) -> tuple[float, int]:
  loss = weighted_surrogate_loss(
    pred,
    yb,
    loss_weight_theta=weights.theta,
    loss_weight_omega=weights.omega,
  )
  batch_n = int(yb.shape[0])
  return total + float(loss.item()) * batch_n, n + batch_n


def eval_loader_mse(
  model: nn.Module,
  loader: DataLoader,
  device: torch.device,
  weights: LossWeights,
) -> float:
  """Weighted surrogate MSE averaged over all samples in ``loader``."""
  model.eval()
  total = 0.0
  n = 0
  with torch.no_grad():
    for xb, yb in loader:
      xb = xb.to(device)
      yb = yb.to(device)
      pred = model(xb)
      total, n = _accumulate_surrogate_mse(pred, yb, weights, total, n)
  if n == 0:
    return float("nan")
  return total / n


def eval_numpy_xy_mse(
  model: nn.Module,
  x: NDArray[np.float64],
  y: NDArray[np.float64],
  device: torch.device,
  weights: LossWeights,
  batch_size: int,
  predict_fn: Callable[[Tensor], Tensor],
) -> float:
  """Chunked val MSE when forward is not plain ``model(x)``."""
  if x.shape[0] == 0:
    return float("nan")
  model.eval()
  total = 0.0
  n = 0
  with torch.no_grad():
    for start in range(0, x.shape[0], batch_size):
      xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False)).to(
        device
      )
      yb = torch.from_numpy(y[start : start + batch_size].astype(np.float32, copy=False)).to(
        device
      )
      pred = predict_fn(xb)
      total, n = _accumulate_surrogate_mse(pred, yb, weights, total, n)
  return total / n if n else float("nan")


def run_tagged_train_epoch(
  model: nn.Module,
  loader: DataLoader,
  device: torch.device,
  optimizer: optim.Optimizer,
  weights: LossWeights,
  predict_tagged: StageTaggedPredictFn,
) -> int:
  """Train epoch for loaders yielding ``(x, y, stage_id)``."""
  model.train()
  steps = 0
  for xb, yb, stage_ids in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    stage_ids = stage_ids.to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = predict_tagged(model, xb, stage_ids)
    loss = weighted_surrogate_loss(
      pred,
      yb,
      loss_weight_theta=weights.theta,
      loss_weight_omega=weights.omega,
    )
    loss.backward()
    optimizer.step()
    steps += 1
  return steps


def eval_tagged_loader_mse(
  model: nn.Module,
  loader: DataLoader,
  device: torch.device,
  weights: LossWeights,
  predict_tagged: StageTaggedPredictFn,
) -> float:
  """Mean surrogate MSE for stage-tagged loaders."""
  model.eval()
  total = 0.0
  n = 0
  with torch.no_grad():
    for xb, yb, stage_ids in loader:
      xb = xb.to(device)
      yb = yb.to(device)
      stage_ids = stage_ids.to(device)
      pred = predict_tagged(model, xb, stage_ids)
      total, n = _accumulate_surrogate_mse(pred, yb, weights, total, n)
  if n == 0:
    return float("nan")
  return total / n


@dataclass
class BestCheckpointTracker:
  """Track best val metric; optional ``best.pt`` write via callback."""

  best_mean: float = field(default_factory=lambda: float("inf"))
  best_state: dict[str, Any] | None = None
  stale_epochs: int = 0

  def observe(
    self,
    mean_val: float,
    model: nn.Module,
  ) -> bool:
    """
    Update best weights on strict improvement (finite val only).

    Returns whether this epoch improved ``best_mean``.
    """
    improved = np.isfinite(mean_val) and mean_val < self.best_mean
    if not improved:
      self.stale_epochs += 1
      return False
    self.best_mean = float(mean_val)
    self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    self.stale_epochs = 0
    return True

  def should_stop(self, patience: int) -> bool:
    return self.stale_epochs >= patience

  def restore_best(self, model: nn.Module) -> None:
    if self.best_state is not None:
      model.load_state_dict(self.best_state)


def save_best_if_improved(
  tracker: BestCheckpointTracker,
  mean_val: float,
  *,
  model: nn.Module,
  optimizer: optim.Optimizer,
  ckpt_dir: Path,
  epoch: int,
  global_step: int,
  cfg: dict[str, Any],
  save_fn,
  resume_state: dict[str, Any] | None = None,
) -> bool:
  """
  Observe val metric and write ``best.pt`` when improved.

  ``save_fn`` is ``run_dir.save_checkpoint`` (injected to avoid import cycles).
  """
  improved = tracker.observe(mean_val, model)
  if improved:
    save_fn(
      ckpt_dir / "best.pt",
      model=model,
      optimizer=optimizer,
      epoch=epoch,
      global_step=global_step,
      val_metric=mean_val,
      cfg=cfg,
      resume_state=resume_state,
    )
  return improved
