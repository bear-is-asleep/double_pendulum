"""Eval hook: curriculum inlet theory figures from run folders."""

from __future__ import annotations

import json
from pathlib import Path

from srcs.eval.run_layout import RunEvalLayout
from srcs.visualization.curriculum_run_inlet import (
  is_curriculum_run,
  render_inlet_theory_for_run,
)


def _minimal_inlet_summary(**extra: object) -> dict:
  base = {
    "passed_stage_min_fraction": 0.05,
    "mix_inlet_gain": 1.4,
    "mix_inlet_max_chunk": 0.1,
    "mix_min_val_delta": 1e-5,
    "mix_decay_lambda": 0.0,
    "mix_unlock_fraction": 0.15,
  }
  base.update(extra)
  return base


def test_baseline_run_skips_inlet_theory(tmp_path: Path) -> None:
  run = tmp_path / "baseline_w512_d2_k4_s0123"
  run.mkdir()
  (run / "metrics.jsonl").write_text('{"epoch": 1, "train_loss": 1.0}\n', encoding="utf-8")
  (run / "summary.json").write_text(
    json.dumps({"strategy": "baseline", **_minimal_inlet_summary()}),
    encoding="utf-8",
  )
  layout = RunEvalLayout.from_metrics(run / "metrics.jsonl")
  assert not is_curriculum_run(layout)
  assert render_inlet_theory_for_run(layout, tmp_path / "fig", [0, 1, 2, 3]) == []


def test_curriculum_run_writes_inlet_theory(tmp_path: Path) -> None:
  run = tmp_path / "curriculum_w512_d2_k4_s0123"
  run.mkdir()
  (run / "metrics.jsonl").write_text('{"epoch": 1, "train_loss": 1.0}\n', encoding="utf-8")
  (run / "summary.json").write_text(
    json.dumps(
      _minimal_inlet_summary(strategy="curriculum", stages=[0, 1, 2, 3]),
    ),
    encoding="utf-8",
  )
  layout = RunEvalLayout.from_metrics(run / "metrics.jsonl")
  paths = render_inlet_theory_for_run(layout, tmp_path / "fig", [0, 1, 2, 3])
  assert len(paths) == 3
  assert all(p.parent.name == "inlet_theory" for p in paths)
