"""CLI to generate Stage 0-6 ``.npz`` pools under ``data/``.

Full pools (``configs/data/full.yaml``)::

  python -m srcs.simulation.generate_data --data-root data

Matched small/full stems (``configs/data`` + ``configs/train``)::

  python -m srcs.simulation.generate_data small
  python -m srcs.train --config small --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from argparse import Namespace
from pathlib import Path

from srcs.loader import (
  DEFAULT_CONFIG_STEM,
  config_yaml_path,
  load_sampler_config,
  load_train_config,
  paired_data_config,
)
from srcs.simulation.generate_pools import (
  PoolGenerateJob,
  generate_pools_from_job,
  parse_int_list,
  parse_stages,
)

logger = logging.getLogger(__name__)


def _print_written_paths(paths: dict[int, dict[str, Path]]) -> None:
  for stage, split_paths in sorted(paths.items()):
    for split, path in split_paths.items():
      print(f"stage {stage} {split}: {path}")


def _pool_config_ref(ref: str) -> tuple[str, Path]:
  """Train or data YAML (path or stem). Train stem wins when both exist."""
  raw = Path(ref)
  if raw.is_file():
    kind = "train" if raw.parent.name == "train" else "data"
    return kind, raw.resolve()
  try:
    return "train", config_yaml_path("train", ref)
  except FileNotFoundError:
    return "data", config_yaml_path("data", ref)


def pool_job_from_args(args: Namespace) -> PoolGenerateJob:
  """Map CLI flags to a single pool-generation job."""
  sampler_path: Path | None = None
  if args.config is not None:
    kind, path = _pool_config_ref(args.config)
    if kind == "train":
      train_stem = path.stem
      train_job = load_train_config(train_stem)
      data_cfg = paired_data_config(train_stem, args.data_config)
      return PoolGenerateJob.from_train_job(
        train_job,
        data_root=args.data_root,
        data_config=data_cfg,
        train_stem=train_stem,
        seed=args.seed,
        overwrite_frozen=args.overwrite_frozen,
      )
    sampler_path = path
    data_cfg = path.stem
  elif args.data_config is not None:
    sampler_path = config_yaml_path("data", args.data_config)
    data_cfg = sampler_path.stem
  else:
    data_cfg = DEFAULT_CONFIG_STEM

  if sampler_path is not None:
    sampler_cfg = load_sampler_config(path=sampler_path)
  else:
    sampler_cfg = load_sampler_config(data_config=data_cfg)
  root = args.data_root if args.data_root is not None else Path("data")
  return PoolGenerateJob(
    data_root=root,
    stages=parse_stages(args.stages, sampler_cfg),
    train_counts=parse_int_list(args.train_counts),
    test_counts=parse_int_list(args.test_counts),
    val_fraction=args.val_fraction,
    model=args.model,
    data_config=data_cfg,
    seed=args.seed,
    overwrite_frozen=args.overwrite_frozen,
  )


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Generate double-pendulum data pools")
  p.add_argument(
    "config",
    nargs="?",
    default=None,
    metavar="CONFIG",
    help="Train or data YAML path or stem",
  )
  p.add_argument(
    "--data-config",
    default=None,
    help=f"Sampler YAML path or stem (default: {DEFAULT_CONFIG_STEM})",
  )
  p.add_argument(
    "--data-root",
    type=Path,
    default=None,
    help="Output dir for stage{S}_{split}.npz",
  )
  p.add_argument(
    "--model",
    default="baseline",
    help="Model YAML name (logged only; default: baseline)",
  )
  p.add_argument(
    "--stages",
    default=None,
    help="Comma stages (default: all; ignored with train CONFIG)",
  )
  p.add_argument(
    "--train-counts",
    default=None,
    help="Comma train sizes (ignored with train CONFIG)",
  )
  p.add_argument(
    "--test-counts",
    default=None,
    help="Comma test sizes (ignored with train CONFIG)",
  )
  p.add_argument(
    "--val-fraction",
    type=float,
    default=None,
    help="Override data YAML val_fraction (ignored with train CONFIG)",
  )
  p.add_argument("--seed", type=int, default=None, help="RNG seed override")
  p.add_argument(
    "--overwrite-frozen",
    action="store_true",
    help="Allow replacing frozen test .npz files",
  )
  p.add_argument("-v", "--verbose", action="store_true")
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
  )

  try:
    paths = generate_pools_from_job(pool_job_from_args(args))
  except (FileNotFoundError, KeyError, ValueError) as exc:
    logger.error("%s", exc)
    return 1

  _print_written_paths(paths)
  return 0


if __name__ == "__main__":
  sys.exit(main())
