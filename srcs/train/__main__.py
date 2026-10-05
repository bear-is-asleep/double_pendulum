"""
Train CLI. Job configs: ``configs/train/<name>.yaml`` (``small``, ``full``, ...).

Examples::

  python -m srcs.simulation.generate_data small
  python -m srcs.train --config small --force
  python -m srcs.train --config full --model curriculum --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from srcs.loader import list_configs
from srcs.train.base import prepare_run_dir, require_stage_pools
from srcs.train.job import resolve_train_job
from srcs.train.registry import list_train_model_names, trainer_for_model
from srcs.utils.yaml_io import print_mapping_yaml

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Train surrogate from pools on disk")
  p.add_argument(
    "--config",
    choices=list_configs("train"),
    default=None,
    help="Training job from configs/train/<name>.yaml (same stems as data: small, full)",
  )
  p.add_argument(
    "--model",
    choices=list_train_model_names(),
    default=None,
    help="Model YAML stem (default: baseline, or train.model in job YAML)",
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
      config=args.config,
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

  print_mapping_yaml(
    "training parameters",
    {
      "train_config": args.config,
      "model": job.model_name,
      "data_root": job.data_root,
      "runs_root": job.runs_root,
      "run_id": job.run_id,
      "stages": job.stages,
      "device": device,
      "max_epochs": job.max_epochs,
      "train": job.cfg,
    },
  )

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
      max_epochs=args.max_epochs,
    )
  except (FileExistsError, FileNotFoundError, KeyError) as exc:
    logger.error("%s", exc)
    return 1

  print(f"run_dir: {res.run_dir}")
  print(f"best_checkpoint: {res.run_dir / 'checkpoints' / 'best.pt'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
