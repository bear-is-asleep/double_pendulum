"""Pointwise test-pool eval: MAE vs time, weighted MSE vs time, JSON summary."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from srcs.eval.channels import CHANNEL_KEYS, per_step_channel_abs_errors
from srcs.eval.run_layout import RunEvalLayout, infer_data_root
from srcs.model.checkpoint import (
  load_model_from_checkpoint,
  predict_at_times,
)
from srcs.model.targets import decode_pred_row
from srcs.physics.core import PendulumParams, kinetic_energy, potential_energy
from srcs.simulation.data import open_pool, pool_path
from srcs.train.epoch import loss_weights_from_cfg, pick_device
from srcs.utils.json_io import read_json, write_json
from srcs.utils.paths import ensure_dir, resolve_checkpoint_file, resolve_run_dir
from srcs.utils.yaml_io import read_mapping

logger = logging.getLogger(__name__)


def weighted_loss_per_step(
  pred: NDArray[np.float64],
  target: NDArray[np.float64],
  *,
  loss_weight_theta: float,
  loss_weight_omega: float,
) -> NDArray[np.float64]:
  """Shape ``(n_t,)`` weighted sin/cos + omega MSE per time index."""
  if pred.shape != target.shape:
    raise ValueError(f"pred/target shape mismatch {pred.shape} vs {target.shape}")
  theta_mse = np.mean((pred[:, :4] - target[:, :4]) ** 2, axis=1)
  omega_mse = np.mean((pred[:, 4:] - target[:, 4:]) ** 2, axis=1)
  return loss_weight_theta * theta_mse + loss_weight_omega * omega_mse


def _gt_target_matrix(view) -> NDArray[np.float64]:
  return np.column_stack(
    [
      view.sin_theta1,
      view.cos_theta1,
      view.sin_theta2,
      view.cos_theta2,
      view.omega1,
      view.omega2,
    ]
  )


def _pred_pe_ke(pred: NDArray[np.float64], params: PendulumParams) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
  n = pred.shape[0]
  pe = np.empty(n, dtype=np.float64)
  ke = np.empty(n, dtype=np.float64)
  for i in range(n):
    state = decode_pred_row(pred[i])
    pe[i] = potential_energy(state, params)
    ke[i] = kinetic_energy(state, params)
  return pe, ke


def _resolve_stages(
  stages: list[int] | None,
  run_dir: Path | None,
) -> list[int]:
  if stages:
    return [int(s) for s in stages]
  if run_dir is not None:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
      summary = read_json(summary_path)
      raw = summary.get("stages")
      if isinstance(raw, list) and raw:
        return [int(s) for s in raw]
    cfg_path = run_dir / "config.yaml"
    if cfg_path.is_file():
      cfg = read_mapping(cfg_path)
      raw = cfg.get("stages")
      if isinstance(raw, list) and raw:
        return [int(s) for s in raw]
  return list(range(7))


@dataclass(frozen=True)
class EvalTestResult:
  stage_ids: tuple[int, ...]
  t: NDArray[np.float64]
  error_vs_t: NDArray[np.float64]
  channel_mae: dict[str, NDArray[np.float64]]
  test_stage_mse: dict[str, float]
  n_traj_by_stage: dict[str, int]


def evaluate_test_pools(
  checkpoint_path: Path | str,
  *,
  data_root: Path | str,
  stages: list[int] | None = None,
  device_name: str | None = None,
) -> EvalTestResult:
  """Mean over test trajectories at each time bin; one row per curriculum stage."""
  device = pick_device(device_name)
  ckpt_path = Path(checkpoint_path)
  run_dir = resolve_run_dir(ckpt_path)
  model, meta, _ckpt_file = load_model_from_checkpoint(ckpt_path, device=device)
  cfg = meta["config"]
  weights = loss_weights_from_cfg(cfg)
  stage_list = _resolve_stages(stages, run_dir)
  root = Path(data_root)

  t_ref: NDArray[np.float64] | None = None
  stage_ids: list[int] = []
  error_rows: list[NDArray[np.float64]] = []
  mae_rows: dict[str, list[NDArray[np.float64]]] = {k: [] for k in CHANNEL_KEYS}
  test_stage_mse: dict[str, float] = {}
  n_traj_by_stage: dict[str, int] = {}

  for stage in stage_list:
    path = pool_path(root, stage, "test")
    if not path.is_file():
      logger.warning("skip stage %s: missing %s", stage, path)
      continue
    pool = open_pool(root, stage, "test")
    if pool.n_traj == 0:
      logger.warning("skip stage %s: empty test pool", stage)
      continue
    if t_ref is None:
      t_ref = np.asarray(pool.t, dtype=np.float64)
    elif pool.t.shape != t_ref.shape or not np.allclose(pool.t, t_ref):
      logger.warning("stage %s time grid differs from first stage; using stage t", stage)
      t_ref = np.asarray(pool.t, dtype=np.float64)

    sum_loss = np.zeros(pool.n_t, dtype=np.float64)
    sum_mae = {k: np.zeros(pool.n_t, dtype=np.float64) for k in CHANNEL_KEYS}
    for i in range(pool.n_traj):
      view = pool.get_traj(i)
      pred = predict_at_times(model, view.t, view.params, device=device)
      target = _gt_target_matrix(view)
      sum_loss += weighted_loss_per_step(
        pred,
        target,
        loss_weight_theta=weights.theta,
        loss_weight_omega=weights.omega,
      )
      pend = view.pendulum_params()
      pred_pe, pred_ke = _pred_pe_ke(pred, pend)
      step_mae = per_step_channel_abs_errors(view, pred, pred_pe, pred_ke)
      for k in CHANNEL_KEYS:
        sum_mae[k] += step_mae[k]

    n_traj = pool.n_traj
    mean_loss = sum_loss / n_traj
    stage_ids.append(stage)
    error_rows.append(mean_loss)
    test_stage_mse[str(stage)] = float(np.mean(mean_loss))
    n_traj_by_stage[str(stage)] = n_traj
    for k in CHANNEL_KEYS:
      mae_rows[k].append(sum_mae[k] / n_traj)

  if t_ref is None or not stage_ids:
    raise ValueError("no test pools evaluated; check data_root and stages")

  return EvalTestResult(
    stage_ids=tuple(stage_ids),
    t=t_ref,
    error_vs_t=np.stack(error_rows, axis=0),
    channel_mae={k: np.stack(mae_rows[k], axis=0) for k in CHANNEL_KEYS},
    test_stage_mse=test_stage_mse,
    n_traj_by_stage=n_traj_by_stage,
  )


def write_eval_artifacts(
  result: EvalTestResult,
  out_dir: Path | str,
  *,
  checkpoint: str,
  data_root: str,
  extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
  out = ensure_dir(out_dir)
  stage_ids = np.asarray(result.stage_ids, dtype=np.int32)
  np.savez(out / "error_vs_t.npz", t=result.t, error=result.error_vs_t, stage_ids=stage_ids)
  np.savez(out / "channel_mae_vs_t.npz", t=result.t, stage_ids=stage_ids, **result.channel_mae)
  summary = {
    "test_stage_mse": result.test_stage_mse,
    "n_traj_by_stage": result.n_traj_by_stage,
    "stage_ids": list(result.stage_ids),
    "checkpoint": checkpoint,
    "data_root": data_root,
  }
  if extra:
    summary.update(extra)
  summary_path = out / "eval_summary.json"
  write_json(summary_path, summary)
  return {
    "error_vs_t": out / "error_vs_t.npz",
    "channel_mae": out / "channel_mae_vs_t.npz",
    "eval_summary": summary_path,
  }


def run_eval_and_write(
  checkpoint_path: Path | str,
  *,
  data_root: Path | str,
  stages: list[int] | None = None,
  device_name: str | None = None,
  out_dir: Path | str | None = None,
) -> dict[str, Path]:
  run_dir = resolve_run_dir(checkpoint_path)
  dest = Path(out_dir) if out_dir is not None else run_dir / "eval_test"
  ckpt = resolve_checkpoint_file(checkpoint_path)
  result = evaluate_test_pools(
    checkpoint_path,
    data_root=data_root,
    stages=stages,
    device_name=device_name,
  )
  return write_eval_artifacts(
    result,
    dest,
    checkpoint=str(ckpt),
    data_root=str(data_root),
  )


def cli(argv: list[str] | None = None) -> None:
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser(description="Evaluate checkpoint on frozen test pools")
  parser.add_argument(
    "--run",
    type=Path,
    default=None,
    help="Run dir with metrics.jsonl (sets checkpoint, data_root, stages, figures/)",
  )
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--data-root", type=Path, default=None)
  parser.add_argument("--stages", type=int, nargs="*", default=None)
  parser.add_argument("--device", default=None)
  parser.add_argument("--out-dir", type=Path, default=None)
  parser.add_argument(
    "--plot",
    action="store_true",
    help="Also write figures/<run_id>/ via srcs.visualization.eval_figures",
  )
  parser.add_argument("--plot-out", type=Path, default=None, help="override figures dir")
  args = parser.parse_args(argv)

  layout: RunEvalLayout | None = None
  if args.run is not None:
    layout = RunEvalLayout.from_run_dir(args.run)
    checkpoint = layout.checkpoint
    data_root = args.data_root or infer_data_root(layout.run_dir)
    stages = args.stages
  elif args.checkpoint is not None:
    checkpoint = args.checkpoint
    run_dir = resolve_run_dir(checkpoint)
    data_root = args.data_root or infer_data_root(run_dir)
    stages = args.stages
    if layout is None and (run_dir / "metrics.jsonl").is_file():
      layout = RunEvalLayout.from_run_dir(run_dir)
  else:
    parser.error("provide --run or --checkpoint")

  if data_root is None or not Path(data_root).is_dir():
    parser.error(f"data-root not found: {data_root!r} (pass --data-root)")

  paths = run_eval_and_write(
    checkpoint,
    data_root=data_root,
    stages=stages,
    device_name=args.device,
    out_dir=args.out_dir,
  )
  for p in paths.values():
    print(p)

  if args.plot or args.plot_out is not None:
    from srcs.eval.run_layout import infer_val_key
    from srcs.utils.json_io import load_jsonl
    from srcs.visualization.eval_figures import render_eval_figures

    if layout is None:
      run_dir = resolve_run_dir(checkpoint)
      layout = RunEvalLayout.from_run_dir(run_dir)
    out_fig = args.plot_out or layout.default_figures_dir
    rows = load_jsonl(layout.metrics)
    render_eval_figures(
      out_dir=out_fig,
      metrics_path=layout.metrics,
      val_key=infer_val_key(rows),
      error_npz=paths["error_vs_t"],
      channel_mae_npz=paths["channel_mae"],
      summary_path=layout.summary,
      eval_summary_path=paths["eval_summary"],
    )
    print(out_fig)
