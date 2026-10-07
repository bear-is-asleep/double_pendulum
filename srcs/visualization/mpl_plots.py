"""Reusable matplotlib axes helpers for eval and compare figures."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

MODEL_COMPARE_COLORS = ["#2f6f8f", "#c45c26", "#6b4c9a", "#059669", "#1a2a3a", "#8a3d18"]
CHANNEL_LINESTYLES = ("-", "--", ":", "-.")


def palette_color(index: int) -> str:
  return MODEL_COMPARE_COLORS[index % len(MODEL_COMPARE_COLORS)]


def step_series(
  ax,
  t: NDArray[np.float64],
  y: NDArray[np.float64],
  *,
  color: str,
  label: str | None = None,
  fill_alpha: float = 0.22,
  linestyle: str = "-",
) -> None:
  ax.fill_between(t, 0, y, step="post", alpha=fill_alpha, color=color, linewidth=0)
  ax.step(
    t,
    y,
    where="post",
    color=color,
    linewidth=1.0,
    label=label,
    linestyle=linestyle,
  )


def assert_compatible_time_grids(
  t_a: NDArray[np.float64],
  t_b: NDArray[np.float64],
  *,
  label_a: str,
  label_b: str,
  context: str,
) -> None:
  if t_a.shape != t_b.shape or not np.allclose(t_a, t_b):
    raise ValueError(
      f"{context}: time grid mismatch between {label_a!r} and {label_b!r} "
      f"(shapes {t_a.shape} vs {t_b.shape})"
    )


@dataclass(frozen=True)
class RectGrid:
  fig: plt.Figure
  axes: NDArray
  nrows: int
  ncols: int

  def axis_at(self, index: int):
    r, c = divmod(index, self.ncols)
    return self.axes[r][c]

  def hide_unused(self, n_used: int) -> None:
    for idx in range(n_used, self.nrows * self.ncols):
      self.axis_at(idx).axis("off")


def make_rect_grid(
  n_panels: int,
  *,
  ncols_max: int = 3,
  cell_wh: tuple[float, float] = (4.0, 3.0),
) -> RectGrid:
  ncols = min(ncols_max, max(n_panels, 1))
  nrows = int(np.ceil(max(n_panels, 1) / ncols))
  fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(cell_wh[0] * ncols, cell_wh[1] * nrows),
    squeeze=False,
  )
  return RectGrid(fig=fig, axes=axes, nrows=nrows, ncols=ncols)


def draw_epoch_lines(
  ax,
  series: Sequence[tuple[Sequence, Sequence, str, str]],
  *,
  linewidth: float = 1.6,
) -> None:
  for x, y, label, color in series:
    ax.plot(x, y, label=label, color=color, linewidth=linewidth)


def draw_train_stage_fraction_compare(
  ax,
  x: Sequence,
  stage_ids: list[int],
  mat: NDArray[np.float64],
  *,
  model_label: str,
  color: str,
  allowed_stages: frozenset[int] | None = None,
  linewidth: float = 1.4,
) -> None:
  """Overlay one model's per-stage mix; linestyle = stage, color = model."""
  for si, sid in enumerate(stage_ids):
    if allowed_stages is not None and sid not in allowed_stages:
      continue
    ls = CHANNEL_LINESTYLES[si % len(CHANNEL_LINESTYLES)]
    ax.plot(
      x,
      mat[si],
      color=color,
      linestyle=ls,
      linewidth=linewidth,
      label=f"{model_label} s{sid}",
    )


def draw_grouped_stage_bars(
  ax,
  stages: list[int],
  per_model: Sequence[tuple[str, dict[int, float]]],
  *,
  ylabel: str = "MSE (time mean)",
) -> None:
  n_models = len(per_model)
  x = np.arange(len(stages), dtype=np.float64)
  width = 0.8 / max(n_models, 1)
  for mi, (label, mse) in enumerate(per_model):
    offsets = x - 0.4 + width / 2 + mi * width
    vals = [mse.get(s, np.nan) for s in stages]
    ax.bar(offsets, vals, width=width, label=label, color=palette_color(mi))
  ax.set_xticks(x, [str(s) for s in stages])
  ax.set_xlabel("stage")
  ax.set_ylabel(ylabel)
  ax.grid(True, axis="y", alpha=0.3)


@dataclass(frozen=True)
class StagedTimeSeries:
  label: str
  t: NDArray[np.float64]
  values: NDArray[np.float64]
  stage_ids: NDArray[np.int64]


def align_staged_time_series(
  rows: Sequence[StagedTimeSeries],
  *,
  context: str,
) -> list[StagedTimeSeries]:
  if not rows:
    return []
  ref = rows[0]
  for row in rows[1:]:
    assert_compatible_time_grids(
      ref.t,
      row.t,
      label_a=ref.label,
      label_b=row.label,
      context=context,
    )
  return list(rows)


def stage_ids_union(stage_id_lists: Sequence[NDArray[np.int64]]) -> list[int]:
  found: set[int] = set()
  for arr in stage_id_lists:
    for sid in arr:
      found.add(int(sid))
  return sorted(found)


def row_for_stage(stage_ids: NDArray[np.int64], stage: int) -> int | None:
  matches = np.where(stage_ids == stage)[0]
  if matches.size == 0:
    return None
  return int(matches[0])


def ymax_for_stages(
  rows: Sequence[StagedTimeSeries],
  stages: list[int],
) -> float:
  peak = 0.0
  for sid in stages:
    for row in rows:
      idx = row_for_stage(row.stage_ids, sid)
      if idx is None:
        continue
      peak = max(peak, float(np.nanmax(row.values[idx])))
  return peak * 1.05 if peak > 0 else 1.0


def decorate_time_axis(
  ax,
  *,
  ylabel: str,
  ymax: float,
  legend: bool = False,
) -> None:
  ax.set_xlabel("t (s)")
  ax.set_ylabel(ylabel)
  ax.set_ylim(0, ymax)
  ax.grid(True, axis="y", alpha=0.3)
  if legend:
    ax.legend(fontsize=7, loc="upper left")
