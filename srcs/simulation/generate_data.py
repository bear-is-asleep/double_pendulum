"""CLI to generate Stage 0–6 ``.npz`` pools under ``data/``.

Human-owned job: full pool sizes from ``configs/sampler.yaml``. Agents smoke-test
with tiny ``--train-counts`` / ``--test-counts`` only.

Example (full pools — do not run casually)::

  python -m srcs.simulation.generate_data --data-root data

Tiny smoke::

  python -m srcs.simulation.generate_data --data-root /tmp/dp_smoke \\
    --stages 1 --train-counts 2 --test-counts 1 --val-fraction 0.5

Preset sizes for baseline train smoke::

  python -m srcs.simulation.generate_smoke_data small
  python -m srcs.simulation.generate_smoke_data sanity
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from srcs.loader import load_sampler_config
from srcs.simulation.generate_pools import (
  generate_pools,
  parse_int_list,
  parse_stages,
)


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Generate double-pendulum data pools")
  p.add_argument(
    "--data-root",
    type=Path,
    default=Path("data"),
    help="Directory for stage{S}_{split}.npz (default: data)",
  )
  p.add_argument(
    "--model",
    default="baseline",
    help="Model YAML name for val_fraction (default: baseline)",
  )
  p.add_argument(
    "--stages",
    default=None,
    help="Comma stages, e.g. 1,2,3 (default: all)",
  )
  p.add_argument(
    "--train-counts",
    default=None,
    help="Comma train sizes per stage id 0..6 (overrides YAML)",
  )
  p.add_argument(
    "--test-counts",
    default=None,
    help="Comma test sizes per stage id 0..6 (overrides YAML)",
  )
  p.add_argument(
    "--val-fraction",
    type=float,
    default=None,
    help="Override model YAML val_fraction",
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

  sampler_cfg = load_sampler_config()
  stages = parse_stages(args.stages, sampler_cfg)
  train_counts = parse_int_list(args.train_counts)
  test_counts = parse_int_list(args.test_counts)

  paths = generate_pools(
    args.data_root,
    stages=stages,
    train_counts=train_counts,
    test_counts=test_counts,
    val_fraction=args.val_fraction,
    model=args.model,
    seed=args.seed,
    overwrite_frozen=args.overwrite_frozen,
  )
  for stage, split_paths in sorted(paths.items()):
    for split, path in split_paths.items():
      print(f"stage {stage} {split}: {path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
