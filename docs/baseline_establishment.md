# Baseline establishment (Step 5)

Lock **FC width, depth, train subsample k, LR, batch** before curriculum / active / progressive runs. Pass bar: every curriculum stage val loss ≤ `stage_pass_mse` in model YAML (default `0.01`).

## Files


| Path                                 | Role                                                |
| ------------------------------------ | --------------------------------------------------- |
| `configs/models/base.yaml`           | Default train + architecture knobs                  |
| `configs/models/baseline.yaml`       | `baseline_search` grid (width, depth, k)            |
| `configs/baseline_lock.yaml`         | Frozen knobs + winning `run_id` after search        |
| `srcs/simulation/generate_pools.py`  | Shared pool generation API                          |
| `srcs/simulation/generate_data.py`   | Data YAML pools or train job stages + paired data stem |
| `configs/data/*.yaml`                | Sampler bounds, per-stage pool sizes, `val_fraction` |
| `configs/train/*.yaml`               | Training job: paths, stages, train overrides          |
| `srcs/train/epoch.py`                | Shared train/eval epoch helpers (Step 9 reuses)     |
| `srcs/train/baseline.py`             | One trial train + per-stage val + `runs/` artifacts |
| `srcs/train/baseline_search.py`      | CLI grid search → updates lock file                 |
| `srcs/train/__main__.py`             | ``python -m srcs.train`` (``--config small|full``)  |
| `srcs/train/registry.py`             | Model YAML stem → trainer class                       |


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

| Step | Command | Typical time |
| --- | --- | --- |
| Unit smoke | `python -m pytest tests/test_baseline.py -q` | Under ~10 s |
| Data | `python -m srcs.simulation.generate_data small` | Under ~30 s |
| Train | `python -m srcs.train --config small --force` | Under ~30 s |
| Full job data | `python -m srcs.simulation.generate_data full` | ~1–2 min |
| Full job train | `python -m srcs.train --config full --force` | ~3–8 min CPU |

```bash
python -m pytest tests/test_baseline.py -q

python -m srcs.simulation.generate_data small
python -m srcs.train --config small --force

python -m srcs.simulation.generate_data full
python -m srcs.train --config full --force
python -m srcs.train --config full --model curriculum --force
```

Curriculum (`configs/models/curriculum.yaml`) uses cumulative train mix on stages `0..s`, val only on stages in that mix, and `mix_weighted_val_mse` to ramp the `(s-1)` / `s` pair between `mix_val_mse_min` (favor stage `s`) and `mix_val_mse_max` (favor stage `s-1`). Each segment runs until `max_epochs_per_stage`, segment `early_stop_patience`, or (non-final segments only) target-stage val MSE clears `stage_pass_mse`. Metrics rows log `train_stage_fraction` each epoch.

Plots (separate from train):

```bash
# Export test metrics into runs/<run_id>/eval_test/ (optional --plot -> figures/<run_id>/)
python -m srcs.eval --run runs/baseline_small/baseline_w512_d2_k4_seed0_5m --plot

# Figures from metrics.jsonl only (paths + figures/<run_id>/ inferred)
python -m srcs.visualization.plots eval \
  --metrics runs/baseline_small/baseline_w512_d2_k4_seed0_5m/metrics.jsonl
```

`data/sanity` uses `val_fraction=0.25` so tiny val pools are non-empty. Full grid search:

```bash
python -m srcs.train.baseline_search --data-root data --runs-root runs
```
