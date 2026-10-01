# Visualization

This repo has two browser apps and a small plotting library. All of them share the same SVG look (`srcs/viz.py`) and the same way to feed trajectories into the UI (`srcs/sources.py`).

## Two apps

| App | Command | Port | Data source |
| --- | --- | --- | --- |
| Live sandbox | `python app.py` | 8765 | On-the-fly RK4 (`DoublePendulum` in `physics.py`) |
| Dataset browser | `python app_data.py` | 8766 | `.npz` pools under `data/` (see `srcs/data.py`) |

**Live sandbox** is for playing with sliders: you change ICs and parameters and the sim integrates forward every frame.

**Dataset browser** steps through saved trajectories (stage, split, traj index). Change **data_root** in the sidebar (Apply) to point at another pool directory. Under the animation, a Plotly panel shows sin θ₁/θ₂, ω, and PE/KE synced to the current frame (**Show time series**). Reference curves come from **stored** disk labels.

**Checkpoint comparison (v1, `app_data` only):** use **Add model** with a path to `runs/.../checkpoints/best.pt`. Each loaded net draws a dashed pendulum (pointwise prediction over the pool time grid, same IC row as training). Toggle **Stored** and each **NN** layer independently. **Chart: values** overlays stored vs models; **Chart: errors** plots stored − NN per channel. Shared UI lives in `srcs/visualization/compare_panel.py` for a future `app.py` port.

The dataset app shows read-only metadata: `g`, masses, and max energy drift `|E(t) − E(0)|` on the stored series.

## On-disk pools (what `app_data` expects)

Pools live as `data/stage{S}_{split}.npz` (for example `data/stage1_test.npz`). Each file holds many trajectories on a shared time vector `t`. Per-trajectory arrays include wrapped angles, sin/cos targets, ω, and PE/KE/E. Full schema and array names are documented in the module docstring of `srcs/data.py`.

Generate pools with the data CLI (see `srcs/generate_data.py`). Until pools exist, `app_data.py` shows a short message instead of the animator.

## Trajectory sources (plug-in shape)

Anything that drives the animator or gif export should implement the `TrajectorySource` protocol in `sources.py`:

- `n_frames()`, `time_at(k)`, `params()`, `frame_state(k)`, `tip_trail(k)`

**GroundTruthSource** wraps a `TrajectoryView` from `open_pool(...).get_traj(i)`.

**SurrogateSource** (`srcs/visualization/surrogate.py`) loads a checkpoint and predicts the 6-D training target over time. **GroundTruthSource** still reads `.npz` pools via `srcs/simulation/data.py`. Multi-arm SVG uses `build_svg(..., overlays=[...])` in `viz.py`.

## Static plots and gifs

`srcs/plots.py` handles matplotlib figures and Pillow gifs. It does not import NiceGUI.

**Time series (PNG)** — three panels per trajectory: sin/cos (not raw θ, so branch cuts do not lie to you), ω₁/ω₂, and PE/KE/total E.

```bash
python -m srcs.visualization.plots timeseries --data-root data --stage 1 --split test --traj 0
```

**Gifs** — raster frames via matplotlib (same geometry as the SVG), written with Pillow:

```bash
python -m srcs.visualization.plots gif --stage 1 --split test --indices 0 1 --stride 2
```

**Training / eval figures** — read run artifacts and write PNGs. These work on empty or toy files so CI can smoke them without a trained model:

```bash
python -m srcs.visualization.plots eval --metrics runs/foo/metrics.jsonl --out-dir figures/eval
```

Expected inputs:

- `metrics.jsonl` — one JSON object per line; optional keys `epoch` or `global_step`, `train_loss`, `val_loss`
- `error_vs_t.npz` — keys `t` (1D) and `error` (1D or 2D with one row per stage)
- `summary.json` — optional `per_stage_mse` or `val_per_stage` dict for a bar chart

## Layout of Python modules

```
srcs/viz.py      SVG + shared CSS (no NiceGUI)
srcs/sources.py  TrajectorySource, GroundTruthSource, SurrogateSource
srcs/plots.py    PNG/gif + eval plot helpers + CLI
app.py                      live RK4 UI
app_data.py                 pool browser UI
```

Tests for SVG smoke, reader wiring, and plot output size live in `tests/test_viz.py`.

## Extending

1. **New trajectory backend** — implement `TrajectorySource`, build `PendulumFrame` per frame, call `build_svg`.
2. **Surrogate overlay** — implement `SurrogateSource.predict_series` to return `(n_t, 6)` with columns `(sin θ₁, cos θ₁, sin θ₂, cos θ₂, ω₁, ω₂)`; decode angles with `atan2` for cartesian drawing; pass a second `PendulumFrame` as `overlay`.
3. **New export** — reuse `render_frame_rgba` or `plot_trajectory_timeseries` from `plots.py` rather than duplicating geometry.
