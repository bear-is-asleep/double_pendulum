"""Preset small pools for baseline train smoke (see ``configs/smoke/*.yaml``).

Examples::

  python -m srcs.simulation.generate_smoke_data small
  python -m srcs.simulation.generate_smoke_data sanity --data-root data/baseline_small
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from srcs.loader import list_smoke_presets, load_smoke_preset
from srcs.simulation.generate_pools import generate_pools


def _counts_for_stages(stages: list[int], n: int) -> list[int]:
  return [n] * len(stages)


def run_preset(name: str, *, data_root: Path | None, overwrite_frozen: bool) -> int:
  preset = load_smoke_preset(name)
  root = data_root if data_root is not None else Path(preset["data_root"])
  stages = [int(s) for s in preset["stages"]]
  data = preset["data"]
  paths = generate_pools(
    root,
    stages=stages,
    train_counts=_counts_for_stages(stages, int(data["train_per_stage"])),
    test_counts=_counts_for_stages(stages, int(data["test_per_stage"])),
    val_fraction=float(data["val_fraction"]),
    overwrite_frozen=overwrite_frozen,
  )
  for stage, split_paths in sorted(paths.items()):
    for split, path in split_paths.items():
      print(f"stage {stage} {split}: {path}")
  return 0


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="Generate preset smoke/sanity data pools")
  sub = p.add_subparsers(dest="command", required=True)
  for name in list_smoke_presets():
    preset = load_smoke_preset(name)
    sp = sub.add_parser(name, help=f"Write pools from configs/smoke/{name}.yaml")
    sp.add_argument(
      "--data-root",
      type=Path,
      default=None,
      help=f"Override default data root ({preset['data_root']})",
    )
    sp.add_argument(
      "--overwrite-frozen",
      action="store_true",
      help="Allow replacing frozen test .npz files",
    )
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
  )
  try:
    return run_preset(
      args.command,
      data_root=args.data_root,
      overwrite_frozen=args.overwrite_frozen,
    )
  except (FileNotFoundError, KeyError) as exc:
    logging.error("%s", exc)
    return 1


if __name__ == "__main__":
  sys.exit(main())
