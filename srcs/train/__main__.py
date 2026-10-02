"""
Train CLI. Smoke presets: ``configs/smoke/<name>.yaml``.

Examples::

  python -m srcs.simulation.generate_smoke_data small
  python -m srcs.train --smoke small --force
  python -m srcs.train --smoke sanity --model curriculum --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from srcs.loader import list_smoke_presets
from srcs.train.base import prepare_run_dir, require_stage_pools
from srcs.train.job import resolve_train_job
from srcs.train.registry import list_train_model_names, trainer_for_model

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Train surrogate from pools on disk")
  p.add_argument(
    "--smoke",
    choices=list_smoke_presets(),
    default=None,
    help="Use paths and train overrides from configs/smoke/<name>.yaml",
  )
  p.add_argument(
    "--model",
    choices=list_train_model_names(),
    default=None,
    help="Model YAML stem (default: baseline, or preset train.model when --smoke)",
  )
  p.add_argument("--data-root", type=Path, default=None, help="NPZ pool directory")
  p.add_argument("--runs-root", type=Path, default=None, help="Output runs/ root")
  p.add_argument("--run-id", default=None, help="Run folder name under runs-root")
  p.add_argument(
    "--stages",
    default=None,
    help="Comma-separated curriculum stage ids",
  )
  p.add_argument("--force", action="store_true", help="Replace existing run dir")
  p.add_argument("--device", default=None, help="Torch device override")
  p.add_argument(
    "--max-epochs",
    type=int,
    default=None,
    help="Epoch cap (per stage for curriculum)",
  )
  p.add_argument("--seed", type=int, default=None, help="Train / dataloader seed")
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
  )

  try:
    job = resolve_train_job(
      smoke=args.smoke,
      model=args.model,
      data_root=args.data_root,
      runs_root=args.runs_root,
      run_id=args.run_id,
      stages_text=args.stages,
      seed=args.seed,
      max_epochs=args.max_epochs,
    )
  except ValueError as exc:
    logger.error("%s", exc)
    return 1

  device = args.device if args.device is not None else job.device

  try:
    require_stage_pools(job.data_root, job.stages)
    prepare_run_dir(job.runs_root, job.run_id, args.force)
    res = trainer_for_model(job.model_name).train_trial(
      job.cfg,
      data_root=job.data_root,
      stages=job.stages,
      runs_root=job.runs_root,
      run_id=job.run_id,
      device_name=device,
      max_epochs=job.max_epochs,
    )
  except (FileExistsError, FileNotFoundError, KeyError) as exc:
    logger.error("%s", exc)
    return 1

  print(f"run_dir: {res.run_dir}")
  print(f"best_checkpoint: {res.run_dir / 'checkpoints' / 'best.pt'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
