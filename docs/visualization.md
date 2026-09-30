# Visualization

This repo has two browser apps and a small plotting library. All of them share the same SVG look (`double_pendulum/viz.py`) and the same way to feed trajectories into the UI (`double_pendulum/sources.py`).

## Two apps

| App | Command | Port | Data source |
| --- | --- | --- | --- |
| Live sandbox | `python app.py` | 8765 | On-the-fly RK4 (`DoublePendulum` in `physics.py`) |
| Dataset browser | `python app_data.py` | 8766 | `.npz` pools under `data/` (see `double_pendulum/data.py`) |

**Live sandbox** is for playing with sliders: you change ICs and parameters and the sim integrates forward every frame.

**Dataset browser** steps through trajectories that were already simulated and saved. You pick curriculum stage (1-5), split (`train` / `val` / `test`), and trajectory index. Under the pendulum animation, a matplotlib panel shows sin/cos, omega, and PE/KE with a red cursor synced to the current frame (toggle **Show time series**). Controls sit in a column on the right. Playback reads either:

- **stored** — sin/cos, ω, and energy arrays from disk (what the network will train on), or  
- **reintegrate** — RK4 from the stored IC and physical parameters on the same time grid (sanity check against the simulator).

The dataset app also shows read-only metadata: `g`, masses, and max energy drift `|E(t) − E(0)|` on the stored series.

## On-disk pools (what `app_data` expects)

Pools live as `data/stage{S}_{split}.npz` (for example `data/stage1_test.npz`). Each file holds many trajectories on a shared time vector `t`. Per-trajectory arrays include wrapped angles, sin/cos targets, ω, and PE/KE/E. Full schema and array names are documented in the module docstring of `double_pendulum/data.py`.

Generate pools with the data CLI (see `double_pendulum/generate_data.py`). Until pools exist, `app_data.py` shows a short message instead of the animator.

## Trajectory sources (plug-in shape)

Anything that drives the animator or gif export should implement the `TrajectorySource` protocol in `sources.py`:

- `n_frames()`, `time_at(k)`, `params()`, `frame_state(k)`, `tip_trail(k)`

**GroundTruthSource** wraps a `TrajectoryView` from `open_pool(...).get_traj(i)`.

**SurrogateSource** is reserved for neural-network predictions over time. It is a stub today: the UI has a disabled “ANN overlay” switch and `predict_series` raises until a checkpoint loader rebuilds the MLP and runs inference. When wired, the app will draw the ground-truth arm solid and the surrogate dashed (see `build_svg(..., overlay=...)` in `viz.py`).

## Static plots and gifs

`double_pendulum/plots.py` handles matplotlib figures and Pillow gifs. It does not import NiceGUI.

**Time series (PNG)** — three panels per trajectory: sin/cos (not raw θ, so branch cuts do not lie to you), ω₁/ω₂, and PE/KE/total E.

```bash
python -m double_pendulum.plots timeseries --data-root data --stage 1 --split test --traj 0
```

**Gifs** — raster frames via matplotlib (same geometry as the SVG), written with Pillow:

```bash
python -m double_pendulum.plots gif --stage 1 --split test --indices 0 1 --stride 2
```

**Training / eval figures** — read run artifacts and write PNGs. These work on empty or toy files so CI can smoke them without a trained model:

```bash
python -m double_pendulum.plots eval --metrics runs/foo/metrics.jsonl --out-dir figures/eval
```

Expected inputs:

- `metrics.jsonl` — one JSON object per line; optional keys `epoch` or `global_step`, `train_loss`, `val_loss`
- `error_vs_t.npz` — keys `t` (1D) and `error` (1D or 2D with one row per stage)
- `summary.json` — optional `per_stage_mse` or `val_per_stage` dict for a bar chart

## Layout of Python modules

```
double_pendulum/viz.py      SVG + shared CSS (no NiceGUI)
double_pendulum/sources.py  TrajectorySource, GroundTruthSource, SurrogateSource
double_pendulum/plots.py    PNG/gif + eval plot helpers + CLI
app.py                      live RK4 UI
app_data.py                 pool browser UI
```

Tests for SVG smoke, reader wiring, and plot output size live in `tests/test_viz.py`.

## Extending

1. **New trajectory backend** — implement `TrajectorySource`, build `PendulumFrame` per frame, call `build_svg`.
2. **Surrogate overlay** — implement `SurrogateSource.predict_series` to return `(n_t, 6)` with columns `(sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁, ω₂)`; decode angles with `atan2` for cartesian drawing; pass a second `PendulumFrame` as `overlay`.
3. **New export** — reuse `render_frame_rgba` or `plot_trajectory_timeseries` from `plots.py` rather than duplicating geometry.
