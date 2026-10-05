"""Vanilla MLP surrogate (Step 5/6). Rebuild from model YAML before load_state_dict."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
  key = name.lower().strip()
  if key == "tanh":
    return nn.Tanh()
  if key == "relu":
    return nn.ReLU()
  if key == "gelu":
    return nn.GELU()
  raise ValueError(f"unsupported activation {name!r}")


def resolve_hidden_layer_widths(cfg: dict[str, Any]) -> list[int]:
  """
  One width per hidden block (length == effective depth).

  ``hidden_width`` may be an int (same width every layer) or a sequence with one
  entry per layer at full ``hidden_depth``. Progressive training may pass a
  smaller ``effective_depth``; list specs are truncated to that depth.
  """
  depth = int(cfg.get("effective_depth", cfg["hidden_depth"]))
  if depth < 1:
    raise ValueError(f"hidden depth must be >= 1, got {depth}")

  raw = cfg["hidden_width"]
  if isinstance(raw, (list, tuple)):
    widths = [int(x) for x in raw]
    full_depth = int(cfg["hidden_depth"])
    if len(widths) != full_depth:
      raise ValueError(
        f"hidden_width list length {len(widths)} must match hidden_depth {full_depth}"
      )
    return widths[:depth]

  width = int(raw)
  return [width] * depth


def build_mlp(cfg: dict[str, Any]) -> nn.Module:
  """
  Stack: Linear -> act per hidden layer, then Linear to ``output_dim``.

  ``cfg`` must include input_dim, output_dim, hidden_width, hidden_depth, activation.
  Progressive training may pass a smaller ``effective_depth`` without mutating YAML.
  """
  in_dim = int(cfg["input_dim"])
  out_dim = int(cfg["output_dim"])
  widths = resolve_hidden_layer_widths(cfg)
  activation_name = str(cfg["activation"])

  layers: list[nn.Module] = []
  prev = in_dim
  for w in widths:
    layers.extend([nn.Linear(prev, w), _activation(activation_name)])
    prev = w
  layers.append(nn.Linear(prev, out_dim))
  return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
