"""Progressive PNN per-column width scaling."""

from __future__ import annotations

from srcs.model.pnn_width import _scale_layers_by_divisor, apply_pnn_column_width_scale
from srcs.model.progressive_net import build_progressive_net, count_parameters


def test_scale_layers_by_divisor() -> None:
  assert _scale_layers_by_divisor([128, 64], 4) == [32, 16]


def test_apply_stages_divisor() -> None:
  cfg = {
    "hidden_width": [128, 64],
    "hidden_depth": 2,
    "pnn_column_width_divisor": "stages",
  }
  apply_pnn_column_width_scale(cfg, n_planned_stages=4)
  assert cfg["hidden_width_baseline"] == [128, 64]
  assert cfg["hidden_width"] == [32, 16]
  assert cfg["pnn_column_width_divisor"] == 4
  net = build_progressive_net(
    {
      **cfg,
      "input_dim": 8,
      "output_dim": 6,
      "activation": "tanh",
      "dropout": 0.0,
    }
  )
  assert net.widths == [32, 16]


def test_apply_off_leaves_baseline_width() -> None:
  cfg = {
    "hidden_width": [128, 64],
    "hidden_depth": 2,
    "pnn_column_width_divisor": "off",
  }
  apply_pnn_column_width_scale(cfg, n_planned_stages=4)
  assert cfg["hidden_width"] == [128, 64]
  assert cfg["pnn_column_width_divisor"] == 1


def test_match_baseline_params_near_mlp_count() -> None:
  cfg = {
    "input_dim": 8,
    "output_dim": 6,
    "hidden_width": [128, 64],
    "hidden_depth": 2,
    "activation": "tanh",
    "dropout": 0.0,
    "use_lateral": True,
    "pnn_column_width_divisor": "match_baseline_params",
  }
  apply_pnn_column_width_scale(cfg, n_planned_stages=4)
  target = int(cfg["pnn_baseline_mlp_n_params"])
  achieved = int(cfg["pnn_matched_n_params"])
  assert cfg["pnn_column_width_divisor"] == "match_baseline_params"
  assert abs(achieved - target) / target < 0.05
  net = build_progressive_net({**cfg, "n_columns": 4})
  assert count_parameters(net, trainable_only=False) == achieved
