"""Shared matplotlib Agg save helpers for visualization PNG exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from srcs.utils.paths import ensure_parent_dir


def prepare_out(out_path: Path | str) -> Path:
  return ensure_parent_dir(out_path)


def write_fig(
  fig: plt.Figure,
  out: Path,
  *,
  dpi: int = 120,
  tight_rect: tuple[float, float, float, float] | None = None,
) -> Path:
  if tight_rect is not None:
    fig.tight_layout(rect=tight_rect)
  else:
    fig.tight_layout()
  fig.savefig(out, dpi=dpi)
  plt.close(fig)
  return out


def write_placeholder(
  out: Path,
  message: str,
  *,
  figsize: tuple[float, float] = (6, 3),
  title: str | None = None,
  dpi: int = 120,
) -> Path:
  fig, ax = plt.subplots(figsize=figsize)
  ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
  if title:
    ax.set_title(title)
  return write_fig(fig, out, dpi=dpi)
