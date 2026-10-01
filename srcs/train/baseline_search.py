"""
Baseline hyperparameter search (Step 5).

Human runs full grid on real ``data/`` pools. Smoke checks live in ``tests/test_baseline.py``.

Example::

  python -m srcs.train.baseline_search --data-root data --runs-root runs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from srcs.loader import load_model_config, load_sampler_config
from srcs.model.train_data import curriculum_stage_ids
from srcs.train.baseline import (
  iter_search_grid,
  pick_smallest_passing,
  train_baseline_trial,
)
from srcs.train.run_dir import baseline_run_id

logger = logging.getLogger(__name__)


def _write_lock_file(path: Path, winner_cfg: dict, run_id: str) -> None:
  lock = {
    "locked_from_run_id": run_id,
    "hidden_width": int(winner_cfg["hidden_width"]),
    "hidden_depth": int(winner_cfg["hidden_depth"]),
    "subsample_stride_k": int(winner_cfg["subsample_stride_k"]),
    "lr": float(winner_cfg["lr"]),
    "batch_size": int(winner_cfg["batch_size"]),
    "stage_pass_mse": float(winner_cfg["stage_pass_mse"]),
    "note": "Do not change for experiments 1-4. Re-run search if data or pass bar moves.",
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(lock, f, sort_keys=False)


def _winner_cfg_from_run(winner_run_dir: Path, base_cfg: dict) -> dict:
  summary = json.loads((winner_run_dir / "summary.json").read_text(encoding="utf-8"))
  cfg = dict(base_cfg)
  cfg["hidden_width"] = summary["hidden_width"]
  cfg["hidden_depth"] = summary["hidden_depth"]
  cfg["subsample_stride_k"] = summary["subsample_stride_k"]
  return cfg


def run_baseline_search(
  *,
  model_cfg: dict,
  data_root: Path,
  runs_root: Path,
  lock_out: Path,
  stages: list[int],
  device: str | None = None,
) -> None:
  """Grid over ``baseline_search`` in model YAML; write lock from smallest passing trial."""
  if not data_root.is_dir():
    raise FileNotFoundError(f"data-root missing: {data_root} (generate pools first)")

  results = []
  for trial_cfg in iter_search_grid(model_cfg):
    rid = baseline_run_id(trial_cfg)
    logger.info("trial %s", rid)
    try:
      res = train_baseline_trial(
        trial_cfg,
        data_root=data_root,
        stages=stages,
        runs_root=runs_root,
        run_id=rid,
        device_name=device,
      )
    except FileExistsError:
      logger.warning("skip existing run %s", rid)
      continue
    results.append(res)
    logger.info(
      "finished %s mean_val=%.6f pass=%s n_params=%s",
      rid,
      res.mean_val_mse,
      res.passes_bar,
      res.n_params,
    )

  winner = pick_smallest_passing(results)
  if winner is None:
    logger.error("no trial cleared stage_pass_mse=%s", model_cfg["stage_pass_mse"])
    raise SystemExit(1)

  winner_cfg = _winner_cfg_from_run(winner.run_dir, load_model_config("baseline"))
  _write_lock_file(lock_out, winner_cfg, winner.run_id)
  logger.info("wrote lock %s from %s", lock_out, winner.run_id)


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  parser = argparse.ArgumentParser(description="Baseline width/depth/k search")
  parser.add_argument("--data-root", type=Path, default=Path("data"))
  parser.add_argument("--runs-root", type=Path, default=Path("runs"))
  parser.add_argument("--lock-out", type=Path, default=Path("configs/baseline_lock.yaml"))
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  model_cfg = load_model_config("baseline")
  sampler_cfg = load_sampler_config()
  run_baseline_search(
    model_cfg=model_cfg,
    data_root=args.data_root,
    runs_root=args.runs_root,
    lock_out=args.lock_out,
    stages=curriculum_stage_ids(sampler_cfg),
    device=args.device,
  )


if __name__ == "__main__":
  main()
