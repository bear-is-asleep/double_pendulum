# Double Pendulum

Interactive Python simulation of a planar double pendulum. Dynamics use the Lagrangian equations of motion with fixed-step RK4. Two NiceGUI apps share the same SVG styling:

| Script | URL | Purpose |
| --- | --- | --- |
| `app.py` | http://localhost:8765 | Live sandbox: sliders change ICs and parameters, sim integrates forward |
| `app_data.py` | http://localhost:8766 | Browse `.npz` trajectory pools (stage / split / traj index) |

Full visualization guide (sources, CLIs, file formats): **[docs/visualization.md](docs/visualization.md)**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Generate dataset pools before using `app_data.py` (see `double_pendulum/generate_data.py`).

## Plotting (CLI)

```bash
python -m double_pendulum.plots timeseries --data-root data --stage 1
python -m double_pendulum.plots gif --stage 1 --split test --indices 0
python -m double_pendulum.plots eval --out-dir figures/eval
```

## Tests

```bash
python -m pytest tests/
```

## Package layout

- `double_pendulum/physics.py` — parameters, derivatives, RK4, energy, cartesian mapping
- `double_pendulum/data.py` — `.npz` pool schema, load/save, `TrajectoryView`
- `double_pendulum/viz.py` — shared SVG + CSS (no UI framework)
- `double_pendulum/sources.py` — `TrajectorySource`, ground truth and surrogate stubs
- `double_pendulum/plots.py` — PNG/gif and training-metric figures
- `app.py` / `app_data.py` — NiceGUI entrypoints
