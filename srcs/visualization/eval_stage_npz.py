"""Stage ids and row indexing for per-stage eval_test NPZ arrays."""

from __future__ import annotations

from collections.abc import Iterable

from numpy.typing import NDArray
import numpy as np


def parse_stage_id(key: object) -> int:
  text = str(key)
  if text.startswith("stage"):
    suffix = text.removeprefix("stage")
    if suffix.isdigit():
      return int(suffix)
  return int(text)


def as_stage_rows(values: NDArray) -> NDArray[np.float64]:
  arr = np.asarray(values, dtype=np.float64)
  if arr.ndim == 1:
    return arr.reshape(1, -1)
  return arr


def stage_ids_from_npz(data: np.lib.npyio.NpzFile, n_rows: int) -> NDArray[np.int64]:
  if "stage_ids" in data.files:
    return np.asarray(data["stage_ids"], dtype=np.int64).reshape(-1)
  return np.arange(n_rows, dtype=np.int64)


def stage_row_mask(
  stage_ids: NDArray[np.int64],
  allowed_stages: list[int] | None,
) -> NDArray[np.bool_]:
  """True per row when ``allowed_stages`` is None or stage id is listed."""
  if allowed_stages is None:
    return np.ones(stage_ids.shape[0], dtype=np.bool_)
  return np.isin(stage_ids, np.asarray(allowed_stages, dtype=np.int64))


def subset_stage_rows(
  stage_ids: NDArray[np.int64],
  allowed_stages: list[int] | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
  """Stage ids to plot and matching row indices into per-stage NPZ arrays."""
  mask = stage_row_mask(stage_ids, allowed_stages)
  row_idx = np.flatnonzero(mask)
  return stage_ids[mask], row_idx


def filter_stage_dict(
  stages_map: dict,
  allowed: list[int] | None,
) -> tuple[list[str], list[float]]:
  if not stages_map:
    return [], []
  items: list[tuple[int, str, float]] = []
  for k, v in stages_map.items():
    sid = parse_stage_id(k)
    if allowed is not None and sid not in allowed:
      continue
    items.append((sid, str(sid), float(v)))
  items.sort(key=lambda x: x[0])
  return [lab for _, lab, _ in items], [val for _, _, val in items]


def filter_stage_id_list(
  discovered: Iterable[int],
  allowed_stages: list[int] | None,
) -> list[int]:
  """Keep ``allowed_stages`` order when set; else sorted unique ``discovered``."""
  found = {int(s) for s in discovered}
  if allowed_stages is None:
    return sorted(found)
  return [int(s) for s in allowed_stages if int(s) in found]


def merge_stage_allowlist(
  explicit: list[int] | None,
  from_summary: object,
) -> list[int] | None:
  """
  Combine YAML/CLI ``stages`` with summary ``stage_ids`` / ``stages``.

  When both are set, keep only stages present in both (explicit order).
  """
  from_list = [int(s) for s in from_summary] if isinstance(from_summary, list) else None
  if explicit is None:
    return from_list
  if from_list is None:
    return explicit
  from_set = set(from_list)
  return [s for s in explicit if s in from_set]
