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


def build_mlp(cfg: dict[str, Any]) -> nn.Module:
  """
  Stack: Linear -> act repeated ``hidden_depth`` times, then Linear to ``output_dim``.

  ``cfg`` must include input_dim, output_dim, hidden_width, hidden_depth, activation.
  Progressive training may pass a smaller ``effective_depth`` without mutating YAML.
  """
  in_dim = int(cfg["input_dim"])
  out_dim = int(cfg["output_dim"])
  width = int(cfg["hidden_width"])
  depth = int(cfg.get("effective_depth", cfg["hidden_depth"]))
  if depth < 1:
    raise ValueError(f"hidden depth must be >= 1, got {depth}")

  layers: list[nn.Module] = [nn.Linear(in_dim, width), _activation(str(cfg["activation"]))]
  for _ in range(depth - 1):
    layers.extend([nn.Linear(width, width), _activation(str(cfg["activation"]))])
  layers.append(nn.Linear(width, out_dim))
  return nn.Sequential(*layers)


def count_parameters(model: nn.Module) -> int:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
