"""Baseline establishment: train mixed pools, per-stage val, pass bar (Step 5)."""

from __future__ import annotations

from typing import Any, Iterator

from srcs.train.base import MixedPoolTrialResult, MixedPoolTrainer
from srcs.train.mlp_trainer import MlpTrainerMixin

BaselineTrialResult = MixedPoolTrialResult


class BaselineTrainer(MlpTrainerMixin, MixedPoolTrainer):
  def experiment_name(self) -> str:
    return "baseline_establishment"


def iter_search_grid(model_cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
  """Cartesian product of ``baseline_search`` lists in the baseline model YAML."""
  grid = model_cfg.get("baseline_search") or {}
  widths = grid.get("hidden_widths") or [int(model_cfg["hidden_width"])]
  depths = grid.get("hidden_depths") or [int(model_cfg["hidden_depth"])]
  ks = grid.get("subsample_stride_k") or [int(model_cfg["subsample_stride_k"])]
  for w in widths:
    for d in depths:
      for k in ks:
        trial = dict(model_cfg)
        trial["hidden_width"] = int(w)
        trial["hidden_depth"] = int(d)
        trial["subsample_stride_k"] = int(k)
        yield trial


def pick_smallest_passing(trials: list[BaselineTrialResult]) -> BaselineTrialResult | None:
  passing = [t for t in trials if t.passes_bar]
  if not passing:
    return None
  passing.sort(key=lambda t: (t.n_params, t.mean_val_mse))
  return passing[0]
