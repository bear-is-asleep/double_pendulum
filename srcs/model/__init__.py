"""MLP surrogate and locked training loss."""

from srcs.model.loss import weighted_surrogate_loss
from srcs.model.mlp import build_mlp, count_parameters

__all__ = [
  "build_mlp",
  "count_parameters",
  "weighted_surrogate_loss",
]
