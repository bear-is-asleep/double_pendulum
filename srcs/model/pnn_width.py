"""Per-column hidden widths for progressive PNN vs locked baseline MLP."""

from __future__ import annotations

from typing import Any

from srcs.model.mlp import build_mlp, resolve_hidden_layer_widths
from srcs.model.progressive_net import build_progressive_net, count_parameters

_MATCH_BASELINE_PARAMS = "match_baseline_params"


def _cfg_at_baseline_width(cfg: dict[str, Any]) -> dict[str, Any]:
  out = dict(cfg)
  out["hidden_width"] = cfg.get("hidden_width_baseline", cfg["hidden_width"])
  return out


def _baseline_layer_widths(cfg: dict[str, Any]) -> list[int]:
  return resolve_hidden_layer_widths(_cfg_at_baseline_width(cfg))


def _write_column_hidden_width(cfg: dict[str, Any], scaled: list[int]) -> None:
  raw_base = cfg.get("hidden_width_baseline", cfg["hidden_width"])
  if isinstance(raw_base, (list, tuple)):
    cfg["hidden_width"] = scaled
    return
  uniq = set(scaled)
  cfg["hidden_width"] = scaled[0] if len(uniq) == 1 else scaled


def _scale_layers_by_divisor(widths: list[int], divisor: int) -> list[int]:
  d = max(1, int(divisor))
  if d <= 1:
    return [int(w) for w in widths]
  return [max(1, int(w) // d) for w in widths]


def _scale_layers_by_multiplier(widths: list[int], multiplier: float) -> list[int]:
  m = float(multiplier)
  return [max(1, int(round(int(w) * m))) for w in widths]


def _resolve_planned_divisor(raw: object, n_planned_stages: int) -> int:
  if raw is None:
    return max(1, int(n_planned_stages))
  if isinstance(raw, str):
    key = raw.strip().lower()
    if key == _MATCH_BASELINE_PARAMS:
      raise ValueError("match mode must not call _resolve_planned_divisor")
    if key in ("stages", "auto"):
      return max(1, int(n_planned_stages))
    if key in ("1", "off", "none", "false"):
      return 1
    raise ValueError(
      f"pnn_column_width_divisor must be int, stages, off, or {_MATCH_BASELINE_PARAMS}, got {raw!r}"
    )
  divisor = int(raw)
  if divisor < 1:
    raise ValueError(f"pnn_column_width_divisor must be >= 1, got {divisor}")
  return divisor


def _mlp_param_count(cfg: dict[str, Any]) -> int:
  model = build_mlp(_cfg_at_baseline_width(cfg))
  return sum(p.numel() for p in model.parameters())


def _pnn_param_count(
  cfg: dict[str, Any],
  column_widths: list[int],
  n_columns: int,
) -> int:
  trial = dict(cfg)
  trial["hidden_width"] = column_widths
  trial["n_columns"] = int(n_columns)
  net = build_progressive_net(trial)
  return count_parameters(net, trainable_only=False)


def _match_column_widths_to_baseline(
  cfg: dict[str, Any],
  base_widths: list[int],
  *,
  n_planned_stages: int,
  target_params: int,
) -> tuple[list[int], float, int]:
  """Binary search width scale so full PNN param count is near baseline MLP."""
  n_cols = max(1, int(n_planned_stages))
  target = int(target_params)

  def count_at(scale: float) -> tuple[list[int], int]:
    widths = _scale_layers_by_multiplier(base_widths, scale)
    return widths, _pnn_param_count(cfg, widths, n_cols)

  w_min, n_min = count_at(0.0)
  w_full, n_full = count_at(1.0)
  if n_min > target:
    return w_min, 0.0, n_min
  if n_full < target:
    return w_full, 1.0, n_full

  best_w, best_scale, best_n = w_full, 1.0, n_full
  best_err = abs(n_full - target)
  lo, hi = 0.0, 1.0
  for _ in range(48):
    mid = (lo + hi) / 2.0
    w_mid, n_mid = count_at(mid)
    err = abs(n_mid - target)
    if err < best_err:
      best_w, best_scale, best_n = w_mid, mid, n_mid
      best_err = err
    if n_mid > target:
      hi = mid
    else:
      lo = mid

  return best_w, best_scale, best_n


def apply_pnn_column_width_scale(
  cfg: dict[str, Any],
  *,
  n_planned_stages: int,
) -> None:
  """
  Set per-column ``hidden_width`` from locked baseline spec.

  ``pnn_column_width_divisor``: ``stages``, int >= 2, ``off`` / ``1``, or
  ``match_baseline_params``.
  """
  if "hidden_width_baseline" not in cfg:
    cfg["hidden_width_baseline"] = cfg["hidden_width"]

  raw = cfg.get("pnn_column_width_divisor", "stages")
  if isinstance(raw, str) and raw.strip().lower() == _MATCH_BASELINE_PARAMS:
    base = _baseline_layer_widths(cfg)
    target = _mlp_param_count(cfg)
    widths, scale, achieved = _match_column_widths_to_baseline(
      cfg,
      base,
      n_planned_stages=n_planned_stages,
      target_params=target,
    )
    _write_column_hidden_width(cfg, widths)
    cfg["pnn_column_width_divisor"] = _MATCH_BASELINE_PARAMS
    cfg["pnn_width_scale"] = float(scale)
    cfg["pnn_baseline_mlp_n_params"] = int(target)
    cfg["pnn_matched_n_params"] = int(achieved)
    return

  divisor = _resolve_planned_divisor(raw, n_planned_stages)
  if divisor <= 1:
    cfg["pnn_column_width_divisor"] = 1
    return

  scaled = _scale_layers_by_divisor(_baseline_layer_widths(cfg), divisor)
  _write_column_hidden_width(cfg, scaled)
  cfg["pnn_column_width_divisor"] = divisor
