"""Progressive PNN architecture schematic figures."""

from __future__ import annotations

from pathlib import Path

import yaml

from srcs.eval.run_layout import RunEvalLayout
from srcs.loader import load_model_config
from srcs.visualization.progressive_pnn_diagram import (
  _arch_cfg_from_run,
  draw_pnn_snapshot_panel,
  plot_pnn_stage_snapshots,
)


def test_plot_pnn_stage_snapshots_smoke(tmp_path: Path) -> None:
  cfg = {
    "hidden_width": [32, 16],
    "hidden_depth": 2,
    "input_dim": 8,
    "output_dim": 6,
    "use_lateral": True,
  }
  out = plot_pnn_stage_snapshots(cfg, [0, 1, 2], tmp_path / "pnn.png")
  assert out.is_file()
  assert out.stat().st_size > 800


def test_draw_panel_three_columns_smoke() -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(4, 5))
  draw_pnn_snapshot_panel(
    ax,
    2,
    widths=[8, 4],
    input_dim=8,
    output_dim=6,
    use_lateral=True,
  )
  plt.close(fig)


def test_cli_config_loads_progressive_defaults() -> None:
  cfg = load_model_config("progressive")
  assert cfg.get("strategy") == "progressive"
  assert int(cfg["hidden_depth"]) >= 1


def test_arch_cfg_from_run_uses_saved_column_widths(tmp_path: Path) -> None:
  run_dir = tmp_path / "progressive_w512_d2_k4"
  run_dir.mkdir()
  cfg_blob = {
    "strategy": "progressive",
    "model_type": "progressive_pnn",
    "train": {"model": "progressive"},
    "hidden_width": 512,
    "hidden_depth": 2,
    "input_dim": 8,
    "output_dim": 6,
  }
  (run_dir / "config.yaml").write_text(
    yaml.safe_dump(cfg_blob, sort_keys=False),
    encoding="utf-8",
  )
  (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
  layout = RunEvalLayout.from_run_dir(run_dir)
  cfg = _arch_cfg_from_run(layout)
  assert cfg["hidden_width"] == 512
