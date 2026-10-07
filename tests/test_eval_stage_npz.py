"""Stage row helpers for eval_test NPZ figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from srcs.eval.channels import CHANNEL_KEYS
from srcs.visualization.eval_figures import plot_channel_mae_vs_t, plot_error_vs_t
from srcs.visualization.eval_stage_npz import (
  filter_stage_id_list,
  merge_stage_allowlist,
  subset_stage_rows,
)


def test_subset_stage_rows_preserves_order() -> None:
  stage_ids = np.array([2, 0, 1], dtype=np.int64)
  filtered, rows = subset_stage_rows(stage_ids, [1, 0])
  assert filtered.tolist() == [0, 1]
  assert rows.tolist() == [1, 2]


def test_filter_stage_id_list_yaml_order() -> None:
  assert filter_stage_id_list([2, 0, 1], [1, 0]) == [1, 0]
  assert filter_stage_id_list([2, 0, 1], None) == [0, 1, 2]


def test_merge_stage_allowlist_intersects() -> None:
  assert merge_stage_allowlist([0, 1, 3], [0, 1, 2]) == [0, 1]
  assert merge_stage_allowlist(None, [0, 1]) == [0, 1]
  assert merge_stage_allowlist([2], None) == [2]


def test_plot_error_vs_t_allowed_stages_subset(tmp_path: Path) -> None:
  t = np.linspace(0, 0.2, 4)
  err = np.stack([np.full(4, 0.1), np.full(4, 0.9)])
  npz = tmp_path / "e.npz"
  np.savez(npz, t=t, error=err, stage_ids=np.array([0, 1], dtype=np.int32))
  out_all = plot_error_vs_t(npz, tmp_path / "all.png")
  out_one = plot_error_vs_t(npz, tmp_path / "one.png", allowed_stages=[0])
  assert out_all.stat().st_size > out_one.stat().st_size


def test_plot_channel_mae_empty_stage_filter_placeholder(tmp_path: Path) -> None:
  t = np.linspace(0, 0.1, 3)
  mae = {k: np.array([[0.1, 0.2, 0.1]], dtype=np.float64) for k in CHANNEL_KEYS}
  npz = tmp_path / "c.npz"
  np.savez(npz, t=t, stage_ids=np.array([1], dtype=np.int32), **mae)
  out = plot_channel_mae_vs_t(npz, tmp_path / "empty.png", allowed_stages=[99])
  assert out.is_file()
  assert out.stat().st_size < 25_000
