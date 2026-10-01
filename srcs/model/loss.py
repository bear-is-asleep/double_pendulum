"""Sin/cos + omega MSE (locked training target; no plain theta MSE)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def weighted_surrogate_loss(
  pred: Tensor,
  target: Tensor,
  *,
  loss_weight_theta: float,
  loss_weight_omega: float,
) -> Tensor:
  """
  pred/target shape (batch, 6): four trig heads then omega1, omega2.
  Theta weight applies to all four sin/cos channels together.
  """
  if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != 6:
    raise ValueError(f"expected (B, 6) pred/target, got {pred.shape} vs {target.shape}")
  theta_mse = nn.functional.mse_loss(pred[:, :4], target[:, :4])
  omega_mse = nn.functional.mse_loss(pred[:, 4:], target[:, 4:])
  return loss_weight_theta * theta_mse + loss_weight_omega * omega_mse
