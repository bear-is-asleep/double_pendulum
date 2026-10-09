"""Unified ``summary.json`` schema: same keys every strategy, NaN when N/A."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from srcs.train.curriculum_mix import inlet_cfg_slice_from_mapping, inlet_knobs_from_cfg

# JSON-safe sentinel for unused knobs / architecture fields.
SUMMARY_NA = float(np.nan)


def hidden_width_field(cfg: dict[str, Any]) -> int | list[int]:
  raw = cfg["hidden_width"]
  if isinstance(raw, (list, tuple)):
    return [int(x) for x in raw]
  return int(raw)


def _inlet_mix_values(cfg: Mapping[str, Any]) -> dict[str, float | int]:
  """Inlet knobs from cfg, or all ``SUMMARY_NA`` when strategy has no inlet YAML."""
  keys = (
    "passed_stage_min_fraction",
    "mix_inlet_gain",
    "mix_inlet_max_chunk",
    "mix_min_val_delta",
    "mix_decay_lambda",
    "mix_unlock_fraction",
    "mix_stagnation_wval_band",
    "mix_stagnation_patience",
    "mix_stagnation_chunk",
    "mix_inlet_terminal_frac_tol",
    "mix_min_epochs_per_stage",
  )
  if inlet_cfg_slice_from_mapping(cfg) is None:
    return {k: SUMMARY_NA for k in keys}
  knobs = inlet_knobs_from_cfg(dict(cfg))
  return {
    "passed_stage_min_fraction": knobs.prior_floor,
    "mix_inlet_gain": knobs.gain,
    "mix_inlet_max_chunk": knobs.max_chunk,
    "mix_min_val_delta": knobs.min_delta,
    "mix_decay_lambda": knobs.decay_lambda,
    "mix_unlock_fraction": knobs.unlock_fraction,
    "mix_stagnation_wval_band": knobs.stagnation_band,
    "mix_stagnation_patience": knobs.stagnation_patience,
    "mix_stagnation_chunk": knobs.stagnation_chunk,
    "mix_inlet_terminal_frac_tol": knobs.terminal_frac_tol,
    "mix_min_epochs_per_stage": knobs.min_epochs_per_stage,
  }


def _cfg_int(cfg: Mapping[str, Any], key: str) -> float:
  if key not in cfg:
    return SUMMARY_NA
  return float(int(cfg[key]))


def strategy_summary_fields(
  cfg: Mapping[str, Any],
  *,
  stages: list[int] | None = None,
  n_params_trainable: int | None = None,
  run_outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """
  Full strategy block for ``summary.json``.

  Trainers call this once from ``trial_summary``; missing inlet / PNN fields are NaN.
  """
  strategy = str(cfg.get("strategy", ""))
  model_type = str(cfg.get("model_type", "mlp"))
  is_pnn = model_type == "progressive_pnn"

  trainable = n_params_trainable
  if trainable is None:
    trainable = cfg.get("_n_params_trainable")
  total = cfg.get("_n_params_total")
  if total is None and trainable is not None and not isinstance(trainable, float):
    total = trainable

  last_stage = SUMMARY_NA
  if stages:
    last_stage = float(max(int(s) for s in stages))

  outcome = dict(run_outcome or {})
  final_active = outcome.get("final_active_max_stage", SUMMARY_NA)
  curriculum_last = outcome.get(
    "curriculum_last_stage",
    last_stage if strategy == "curriculum" else SUMMARY_NA,
  )
  progressive_last = outcome.get(
    "progressive_last_stage",
    last_stage if strategy == "progressive" else SUMMARY_NA,
  )

  fields: dict[str, Any] = {
    "strategy": strategy,
    "hidden_width": hidden_width_field(dict(cfg)),
    "hidden_depth": int(cfg["hidden_depth"]),
    "model_type": model_type,
    "use_lateral": bool(cfg.get("use_lateral", True)) if is_pnn else SUMMARY_NA,
    "n_columns": _cfg_int(cfg, "n_columns") if is_pnn else SUMMARY_NA,
    "n_params_trainable": float(trainable) if trainable is not None else SUMMARY_NA,
    "n_params_total": float(total) if total is not None else SUMMARY_NA,
    "final_active_max_stage": float(final_active)
    if final_active is not SUMMARY_NA and final_active is not None
    else SUMMARY_NA,
    "curriculum_last_stage": float(curriculum_last)
    if curriculum_last is not SUMMARY_NA and curriculum_last is not None
    else SUMMARY_NA,
    "progressive_last_stage": float(progressive_last)
    if progressive_last is not SUMMARY_NA and progressive_last is not None
    else SUMMARY_NA,
  }
  fields.update(_inlet_mix_values(cfg))
  return fields
