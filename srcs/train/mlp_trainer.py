"""Shared MLP build hooks for strategy trainers (Step 6/7)."""

from __future__ import annotations

from typing import Any

from torch import nn

from srcs.model.mlp import build_mlp, count_parameters


class MlpTrainerMixin:
  """``build_model`` / param count / summary fields for vanilla ANN strategies."""

  def build_model(self, cfg: dict[str, Any]) -> nn.Module:
    return build_mlp(cfg)

  def count_parameters(self, model: nn.Module) -> int:
    return count_parameters(model)

  def extra_summary_fields(self, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
      "hidden_width": int(cfg["hidden_width"]),
      "hidden_depth": int(cfg["hidden_depth"]),
      "strategy": str(cfg.get("strategy", "")),
    }
