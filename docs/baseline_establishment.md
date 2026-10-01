# Baseline establishment (Step 5)

Lock **FC width, depth, train subsample k, LR, batch** before curriculum / active / progressive runs. Pass bar: every curriculum stage val loss ≤ `stage_pass_mse` in model YAML (default `0.01`).

## Files


| Path                                 | Role                                                |
| ------------------------------------ | --------------------------------------------------- |
| `configs/models/base.yaml`           | Default train + architecture knobs                  |
| `configs/models/baseline.yaml`       | `baseline_search` grid (width, depth, k)            |
| `configs/baseline_lock.yaml`         | Frozen knobs + winning `run_id` after search        |
| `srcs/simulation/generate_pools.py`  | Shared pool generation API                          |
| `srcs/simulation/generate_data.py`   | Full/custom pool CLI                                |
| `configs/smoke/*.yaml`               | Paired data + train smoke presets                   |
| `srcs/simulation/generate_smoke_data.py` | Write pools from smoke YAML                         |
| `srcs/train/epoch.py`                | Shared train/eval epoch helpers (Step 9 reuses)     |
| `srcs/train/baseline.py`             | One trial train + per-stage val + `runs/` artifacts |
| `srcs/train/baseline_search.py`      | CLI grid search → updates lock file                 |
| `srcs/train/baseline_smoke.py`       | Train from ``configs/smoke/*.yaml`` (pools on disk)   |


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
| Data | `python -m srcs.simulation.generate_smoke_data small` | Under ~30 s |
| Train | `python -m srcs.train.baseline_smoke small --force` | Under ~30 s |
| Sanity data | `python -m srcs.simulation.generate_smoke_data sanity` | ~1–2 min |
| Sanity train | `python -m srcs.train.baseline_smoke sanity --force` | ~3–8 min CPU |

```bash
python -m pytest tests/test_baseline.py -q

python -m srcs.simulation.generate_smoke_data small
python -m srcs.train.baseline_smoke small --force

python -m srcs.simulation.generate_smoke_data sanity
python -m srcs.train.baseline_smoke sanity --force
```

Plots (separate from train):

```bash
python -m srcs.visualization.plots eval \
  --metrics runs/baseline_small/baseline_w64_d2_k4_seed0_5m/metrics.jsonl \
  --val-key mean_val_mse \
  --out-dir figures/baseline_small
```

`small` preset uses `val_fraction=0.25` so tiny val pools are non-empty. Full grid search:

```bash
python -m srcs.train.baseline_search --data-root data --runs-root runs
```
