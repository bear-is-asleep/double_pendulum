"""Smoke tests for curriculum inlet theory plots."""

from __future__ import annotations

from pathlib import Path

from srcs.loader import load_model_config
from srcs.visualization.curriculum_inlet_plots import render_curriculum_inlet_figures


def test_render_curriculum_inlet_figures(tmp_path: Path) -> None:
  cfg = load_model_config("curriculum")
  paths = render_curriculum_inlet_figures(
    cfg,
    last_stage=3,
    out_dir=tmp_path / "inlet",
    max_steps=20,
  )
  assert len(paths) == 3
  for path in paths:
    assert path.is_file()
    assert path.stat().st_size > 500
