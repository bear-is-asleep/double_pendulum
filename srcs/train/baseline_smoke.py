"""
Train-only baseline smoke (Step 5). Generate pools first with ``generate_smoke_data``.

Presets live in ``configs/smoke/<name>.yaml``.

Examples::

  python -m srcs.simulation.generate_smoke_data small
  python -m srcs.train.baseline_smoke small --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from srcs.loader import list_smoke_presets, load_model_config, load_smoke_preset
from srcs.simulation.data import pool_path
from srcs.train.baseline import train_baseline_trial

logger = logging.getLogger(__name__)

def _parse_int_list(text: str) -> list[int]:
  return [int(p.strip()) for p in text.split(",") if p.strip()]


def _train_cfg_from_preset(preset: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
  train = dict(preset["train"])
  model_name = str(train.pop("model", "baseline"))
  device = train.pop("device", None)
  if device is not None and not isinstance(device, str):
    device = None
  cfg = load_model_config(model_name)
  cfg.update(train)
  return cfg, device


def _require_pools(data_root: Path, stages: Sequence[int]) -> None:
  missing: list[str] = []
  for stage in stages:
    for split in ("train", "val"):
      path = pool_path(data_root, stage, split)
      if not path.is_file():
        missing.append(str(path))
  if missing:
    raise FileNotFoundError(
      "missing pool files (run generate_smoke_data first):\n  "
      + "\n  ".join(missing)
    )


def _prepare_run_dir(runs_root: Path, run_id: str, force: bool) -> None:
  run_dir = runs_root / run_id
  if run_dir.exists():
    if not force:
      raise FileExistsError(
        f"run dir already exists: {run_dir} (pass --force to replace)"
      )
    shutil.rmtree(run_dir)


def run_preset(
  name: str,
  *,
  data_root: Path | None,
  runs_root: Path | None,
  run_id: str | None,
  stages: Sequence[int] | None,
  force: bool,
  device_name: str | None,
  max_epochs: int | None,
) -> Path:
  preset = load_smoke_preset(name)
  root = data_root if data_root is not None else Path(preset["data_root"])
  runs = runs_root if runs_root is not None else Path(preset["runs_root"])
  rid = run_id if run_id is not None else str(preset["run_id"])
  stage_list = list(stages if stages is not None else preset["stages"])

  cfg, preset_device = _train_cfg_from_preset(preset)
  epochs = int(max_epochs if max_epochs is not None else cfg["max_epochs"])
  cfg["max_epochs"] = epochs
  device = device_name if device_name is not None else preset_device

  _require_pools(root, stage_list)
  _prepare_run_dir(runs, rid, force)
  res = train_baseline_trial(
    cfg,
    data_root=root,
    stages=stage_list,
    runs_root=runs,
    run_id=rid,
    device_name=device,
    max_epochs=epochs,
  )
  return res.run_dir


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    description="Baseline train smoke (configs/smoke/*.yaml; pools must exist)",
  )
  sub = p.add_subparsers(dest="command", required=True)

  for name in list_smoke_presets():
    preset = load_smoke_preset(name)
    sp = sub.add_parser(name, help=f"Train using configs/smoke/{name}.yaml")
    sp.add_argument(
      "--data-root",
      type=Path,
      default=None,
      help=f"Pools directory (default: {preset['data_root']})",
    )
    sp.add_argument(
      "--runs-root",
      type=Path,
      default=None,
      help=f"Run output root (default: {preset['runs_root']})",
    )
    sp.add_argument("--run-id", default=None, help=f"default: {preset['run_id']}")
    sp.add_argument(
      "--stages",
      default=None,
      help="Comma stage ids (default: preset stages)",
    )
    sp.add_argument("--force", action="store_true", help="Replace existing run dir")
    sp.add_argument("--device", default=None, help="Torch device override")
    sp.add_argument(
      "--max-epochs",
      type=int,
      default=None,
      help="Override preset max_epochs",
    )
  return p


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
  )
  stages = _parse_int_list(args.stages) if args.stages else None

  try:
    run_dir = run_preset(
      args.command,
      data_root=args.data_root,
      runs_root=args.runs_root,
      run_id=args.run_id,
      stages=stages,
      force=args.force,
      device_name=args.device,
      max_epochs=args.max_epochs,
    )
  except (FileExistsError, FileNotFoundError, KeyError) as exc:
    logger.error("%s", exc)
    return 1

  print(f"run_dir: {run_dir}")
  print(f"best_checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
