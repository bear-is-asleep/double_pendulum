# Double Pendulum

Interactive Python simulation of a planar double pendulum. The arms follow the classic Lagrangian equations of motion, integrated with fixed-step RK4. A NiceGUI browser UI lets you tweak angles, masses, and lengths while the lower tip paints a trail.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:8765](http://localhost:8765).

## Checks

```bash
python tests/test_physics.py
```

## Layout

- `double_pendulum/physics.py` — parameters, derivatives, RK4, cartesian mapping
- `app.py` — browser UI and animation loop
- `tests/test_physics.py` — equilibrium, coordinates, energy drift
