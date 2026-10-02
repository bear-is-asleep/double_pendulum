"""Frozen test-set evaluation and metric artifact exports (no plotting)."""

__all__ = [
  "CHANNEL_KEYS",
  "RunEvalLayout",
  "evaluate_test_pools",
  "infer_data_root",
  "infer_val_key",
  "load_jsonl",
  "write_eval_artifacts",
]


def __getattr__(name: str):
  if name in ("evaluate_test_pools", "write_eval_artifacts"):
    from srcs.eval import pool_eval as pe
    return getattr(pe, name)
  if name == "CHANNEL_KEYS":
    from srcs.eval.channels import CHANNEL_KEYS
    return CHANNEL_KEYS
  if name == "load_jsonl":
    from srcs.utils.json_io import load_jsonl
    return load_jsonl
  if name in ("RunEvalLayout", "infer_data_root", "infer_val_key"):
    from srcs.eval import run_layout as rl
    return getattr(rl, name)
  raise AttributeError(name)
