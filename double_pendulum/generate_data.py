"""CLI to generate Stage 1–5 ``.npz`` pools under ``data/``.

Human-owned job: full pool sizes from ``configs/sampler.yaml``. Agents smoke-test
with tiny ``--train-counts`` / ``--test-counts`` only.

Example (full pools — do not run casually)::

  python -m double_pendulum.generate_data --data-root data

Tiny smoke::

  python -m double_pendulum.generate_data --data-root /tmp/dp_smoke \\
    --stages 1 --train-counts 2 --test-counts 1 --val-fraction 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from configs.loader import load_model_config, load_sampler_config
from double_pendulum.data import generate_all_stages


def _parse_int_list(text: str | None) -> list[int] | None:
  if text is None:
    return None
  parts = [p.strip() for p in text.split(",") if p.strip()]
  return [int(p) for p in parts]


def _parse_stages(text: str | None, num_stages: int) -> list[int]:
  if text is None:
    return list(range(1, num_stages + 1))
  stages = _parse_int_list(text)
  assert stages is not None
  for s in stages:
    if s < 1 or s > num_stages:
      raise argparse.ArgumentTypeError(f"stage {s} out of 1..{num_stages}")
  return stages


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
    help="Comma train sizes per stage index 1..5 (overrides YAML)",
  )
  p.add_argument(
    "--test-counts",
    default=None,
    help="Comma test sizes per stage index 1..5 (overrides YAML)",
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
  model_cfg = load_model_config(args.model)
  num_stages = int(sampler_cfg.get("num_stages", 5))
  stages = _parse_stages(args.stages, num_stages)
  train_counts = _parse_int_list(args.train_counts)
  test_counts = _parse_int_list(args.test_counts)
  val_fraction = (
    float(args.val_fraction)
    if args.val_fraction is not None
    else float(model_cfg["val_fraction"])
  )

  # Pad overrides to full stage list length when only generating a subset
  if train_counts is not None and len(train_counts) != num_stages:
    if len(train_counts) == len(stages):
      full = list(sampler_cfg["pools"]["train"])
      for s, n in zip(stages, train_counts):
        full[s - 1] = n
      train_counts = full
    else:
      raise SystemExit(
        "--train-counts must list all stages or match --stages length"
      )
  if test_counts is not None and len(test_counts) != num_stages:
    if len(test_counts) == len(stages):
      full = list(sampler_cfg["pools"]["test"])
      for s, n in zip(stages, test_counts):
        full[s - 1] = n
      test_counts = full
    else:
      raise SystemExit(
        "--test-counts must list all stages or match --stages length"
      )

  paths = generate_all_stages(
    args.data_root,
    sampler_cfg,
    val_fraction=val_fraction,
    stages=stages,
    train_counts=train_counts,
    test_counts=test_counts,
    seed=args.seed,
    overwrite_frozen=args.overwrite_frozen,
  )
  for stage, split_paths in sorted(paths.items()):
    for split, path in split_paths.items():
      print(f"stage {stage} {split}: {path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
