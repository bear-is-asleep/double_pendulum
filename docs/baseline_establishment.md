# Baseline establishment (Step 5)

Lock **FC width, depth, train subsample k, LR, batch** before curriculum / active / progressive runs. Pass bar: every curriculum stage val loss ≤ `stage_pass_mse` in model YAML (default `0.01`).

## Files


| Path                                | Role                                                   |
| ----------------------------------- | ------------------------------------------------------ |
| `configs/models/base.yaml`          | Default ANN + train knobs; optional `baseline_search` grid |
| `configs/baseline_lock.yaml`        | Frozen knobs + winning `run_id` after search           |
| `srcs/simulation/generate_pools.py` | Shared pool generation API                             |
| `srcs/simulation/generate_data.py`  | Data YAML pools or train job stages + paired data stem |
| `configs/data/*.yaml`               | Sampler bounds, per-stage pool sizes, `val_fraction`   |
| `configs/train/*.yaml`              | Training job: paths, stages, train overrides           |
| `srcs/train/epoch.py`               | Shared train/eval epoch helpers (Step 9 reuses)        |
| `srcs/train/baseline.py`            | One trial train + per-stage val + `runs/` artifacts    |
| `srcs/train/baseline_search.py`     | CLI grid search → updates lock file                    |
| `srcs/train/__main__.py`            | `python -m srcs.train` (``--config small               |
| `srcs/train/registry.py`            | Model YAML stem → trainer class                        |




## Human workflow

1. Generate full `data/` pools (Step 4).
2. Run search (long job, not for agents):
  ```bash
   python -m srcs.train.baseline_search --data-root data --runs-root runs
  ```
3. Confirm `configs/baseline_lock.yaml` points at the smallest passing net (`pick_smallest_passing` sorts by param count).
4. Keep locked values in experiments 1–4; do not float width / depth / k / LR / batch after lock.



## Run folder

Each trial writes `runs/<run_id>/` with `config.yaml`, `metrics.jsonl`, `summary.json`, `checkpoints/best.pt` and `last.pt` (PyTorch dict with `model_state_dict`).

## Small test (local)


| Step           | Command                                         | Typical time |
| -------------- | ----------------------------------------------- | ------------ |
| Unit smoke     | `python -m pytest tests/test_baseline.py -q`    | Under ~10 s  |
| Data           | `python -m srcs.simulation.generate_data small` | Under ~30 s  |
| Train          | `python -m srcs.train --config small --force`   | Under ~30 s  |
| Full job data  | `python -m srcs.simulation.generate_data full`  | ~1–2 min     |
| Full job train | `python -m srcs.train --config full --force`    | ~3–8 min CPU |


```bash
python -m pytest tests/test_baseline.py -q

python -m srcs.simulation.generate_data small
python -m srcs.train --config small --force

python -m srcs.simulation.generate_data full
python -m srcs.train --config full --force
python -m srcs.train --config full --model curriculum --force
```

Curriculum (`configs/models/curriculum.yaml`) runs one loop to `max_epochs` with **adaptive inlet**: train starts on stage 0 only; when `mix_weighted_val_mse` drops epoch-to-epoch by more than `mix_min_val_delta`, a chunk `min(mix_inlet_max_chunk, mix_inlet_gain * delta)` blends the mix toward an exponential template over stages `0..active_max_stage`. If `|wval_delta| <= mix_stagnation_wval_band` for `mix_stagnation_patience` consecutive epochs (counted even when a val-drop inlet fired that epoch), a **stagnation inlet** applies `mix_stagnation_chunk` (not capped by `mix_inlet_max_chunk`) from the post-adaptive mix when both apply. Set `mix_stagnation_patience` to `0` to disable. No qualifying drop and no stagnation trigger freezes `train_stage_fraction`. `active_max_stage` unlocks higher stages when the frontier fraction reaches `mix_unlock_fraction`. `passed_stage_min_fraction` is enforced via slack-weighted projection. Early stop uses global `early_stop_patience` on wval only after the train mix reaches the terminal inlet target (`active_max_stage == last_stage` and fractions within `mix_inlet_terminal_frac_tol` of the projected exp template). Metrics log `train_stage_fraction`, `mix_val_delta`, `mix_inlet_reason`, `mix_stagnation_epochs`, `active_max_stage`, and `mix_inlet_chunk` each epoch.

Plots (separate from train):

```bash
# Export test metrics into runs/<run_id>/eval_test/ (optional --plot -> figures/<run_id>/)
python -m srcs.eval --run runs/baseline_small/baseline_w512_d2_k4 --plot

# Figures from metrics.jsonl only (paths + figures/<run_id>/ inferred)
python -m srcs.visualization.plots eval \
  --metrics runs/baseline_small/baseline_w512_d2_k4_seed0_5m/metrics.jsonl
```

`data/sanity` uses `val_fraction=0.25` so tiny val pools are non-empty. Full grid search:

```bash
python -m srcs.train.baseline_search --data-root data --runs-root runs
```

