"""Step 4b: SVG smoke, reader wiring, plot shape checks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from srcs.loader import load_sampler_config
from srcs.simulation.data import build_pool, save_pool
from srcs.physics import DoublePendulum, PendulumParams, PendulumState
from srcs.visualization.plots import (
  plot_trajectory_timeseries,
  svg_smoke_for_source,
  write_trajectory_gif,
)
from srcs.visualization.live_timeseries import series_from_ground_truth
from srcs.visualization.sources import GroundTruthSource
from srcs.visualization.surrogate import SurrogateSource
from srcs.visualization.viz import build_svg


def _tiny_sampler_cfg() -> dict:
  cfg = copy.deepcopy(load_sampler_config())
  cfg["T"] = 0.4
  cfg["dt"] = 0.1
  cfg["probe_time"] = 0.2
  cfg["max_draw_attempts"] = 200
  cfg["seed"] = 11
  return cfg


def test_build_svg_smoke_sim() -> None:
  sim = DoublePendulum(
    params=PendulumParams(),
    state=PendulumState(theta1=1.0, theta2=0.5),
  )
  svg = build_svg(sim, show_trail=False)
  assert "<svg" in svg and 'viewBox="0 0 640 640"' in svg
  assert "polyline" not in svg


def test_ground_truth_source_reader_wiring(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 2, cfg, split="test", rng=np.random.default_rng(0))
  save_pool(tmp_path / "stage1_test.npz", pool)
  view = pool.get_traj(0)
  src = GroundTruthSource(view, mode="stored")
  assert src.n_frames() == pool.n_t
  s0 = src.frame_state(0)
  assert np.isfinite(s0.theta1)
  trail = src.tip_trail(1)
  assert len(trail) == 2
  svg = svg_smoke_for_source(src, k=0)
  assert "circle" in svg


def test_playback_series_shapes() -> None:
  from srcs.visualization.comparison_data import reference_from_ground_truth
  from srcs.visualization.surrogate import state_on_pred_grid

  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 1, cfg, rng=np.random.default_rng(5))
  view = pool.get_traj(0)
  src = GroundTruthSource(view, mode="stored")
  s = series_from_ground_truth(src)
  assert s.t.shape == s.sin_theta1.shape == s.omega1.shape
  assert s.t.shape[0] == pool.n_t
  # Playback and the comparison chart must share the same stored columns.
  ref = reference_from_ground_truth(src)
  for name in ("t", "sin_theta1", "sin_theta2", "omega1", "omega2", "potential", "kinetic"):
    assert np.array_equal(getattr(s, name), getattr(ref, name))
  energy = src.view.energy
  assert src.energy_drift_max() == float(np.max(np.abs(energy - energy[0])))
  full = src.tip_trail(src.n_frames() - 1)
  assert src.tip_trail(1) == full[:2]
  pred = np.zeros((src.n_frames(), 6), dtype=np.float64)
  assert state_on_pred_grid(pred, view, 0) == view.frame_at(0)


def test_chart_traces_values_and_residuals() -> None:
  from srcs.visualization.comparison_data import CHART_LAYER_COLOR_SETS, LayerSeries, ReferenceSeries
  from srcs.visualization.live_timeseries import ComparisonChartState, chart_traces

  t = np.array([0.0, 1.0])
  zeros = np.zeros(2)
  ref = ReferenceSeries(
    t=t,
    sin_theta1=np.array([0.0, 1.0]),
    sin_theta2=zeros,
    omega1=zeros,
    omega2=zeros,
    potential=zeros,
    kinetic=zeros,
  )
  layer = LayerSeries(
    layer_id="nn:x",
    label="x",
    sin_theta1=np.array([0.0, 0.25]),
    sin_theta2=zeros,
    omega1=zeros,
    omega2=zeros,
    potential=zeros,
    kinetic=zeros,
    trace_colors=CHART_LAYER_COLOR_SETS[0],
  )
  values = ComparisonChartState(
    ref=ref,
    layers=[layer],
    visible={"stored": True, "nn:x": True},
    display_mode="values",
  )
  names = [spec.name for spec in chart_traces(values) if spec.name.startswith("sin theta1")]
  assert names == ["sin theta1 (stored)", "sin theta1 (x)"]
  errors = ComparisonChartState(
    ref=ref,
    layers=[layer],
    visible={"stored": True, "nn:x": True},
    display_mode="errors",
  )
  sin1 = [spec for spec in chart_traces(errors) if spec.name.startswith("sin theta1")]
  assert len(sin1) == 1
  assert np.allclose(sin1[0].y, np.array([0.0, 0.75]))


def test_reintegrate_mode_same_length(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 1, cfg, rng=np.random.default_rng(2))
  view = pool.get_traj(0)
  stored = GroundTruthSource(view, mode="stored")
  reint = GroundTruthSource(view, mode="reintegrate")
  assert stored.n_frames() == reint.n_frames()


def test_plot_timeseries_png(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 1, cfg, rng=np.random.default_rng(3))
  out = plot_trajectory_timeseries(pool.get_traj(0), tmp_path / "ts.png")
  assert out.is_file() and out.stat().st_size > 500


def test_gif_export(tmp_path) -> None:
  cfg = _tiny_sampler_cfg()
  pool = build_pool(1, 1, cfg, rng=np.random.default_rng(4))
  src = GroundTruthSource(pool.get_traj(0), mode="stored")
  out = write_trajectory_gif(src, tmp_path / "x.gif", stride=2, frame_size=160)
  assert out.is_file() and out.stat().st_size > 100


def test_build_svg_multi_overlay() -> None:
  from srcs.physics import PendulumParams, PendulumState
  from srcs.visualization.viz import LayerStyle, PendulumFrame

  p = PendulumParams()
  primary = PendulumFrame(params=p, state=PendulumState(0.5, -0.3, 0.1, -0.2))
  overlay = PendulumFrame(params=p, state=PendulumState(0.6, -0.2, 0.0, 0.1))
  style = LayerStyle(dashed=True, stroke="#6b4c9a", bob1="#9b7bb8", bob2="#e07b4a")
  svg = build_svg(primary, show_trail=False, overlays=[(overlay, style)])
  assert svg.count('stroke-dasharray="6 4"') >= 1


def test_surrogate_load_requires_file(tmp_path: Path) -> None:
  missing = tmp_path / "nope.pt"
  s = SurrogateSource(checkpoint_path=missing)
  with pytest.raises(FileNotFoundError):
    s.load()
