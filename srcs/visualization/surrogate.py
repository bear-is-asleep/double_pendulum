"""Surrogate trajectory layers for pool playback and future live app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from srcs.model.checkpoint import load_model_from_checkpoint, predict_at_times
from srcs.physics.core import PendulumParams, PendulumState, cartesian, kinetic_energy, potential_energy
from srcs.simulation.data import TrajectoryView


def decode_pred_row(row: NDArray[np.float64]) -> PendulumState:
  """Six-head row -> angles via atan2 and omega heads."""
  return PendulumState(
    theta1=float(np.arctan2(row[0], row[1])),
    theta2=float(np.arctan2(row[2], row[3])),
    omega1=float(row[4]),
    omega2=float(row[5]),
  )


def series_from_pred(
  t: NDArray[np.float64],
  pred: NDArray[np.float64],
  params: PendulumParams,
) -> dict[str, NDArray[np.float64]]:
  """Sin theta, omega, PE/KE arrays from ``(n_t, 6)`` predictions."""
  n = int(t.shape[0])
  sin_theta1 = pred[:, 0].astype(np.float64, copy=False)
  sin_theta2 = pred[:, 2].astype(np.float64, copy=False)
  omega1 = pred[:, 4].astype(np.float64, copy=False)
  omega2 = pred[:, 5].astype(np.float64, copy=False)
  potential = np.empty(n, dtype=np.float64)
  kinetic = np.empty(n, dtype=np.float64)
  for i in range(n):
    state = decode_pred_row(pred[i])
    pe = potential_energy(state, params)
    ke = kinetic_energy(state, params)
    potential[i] = pe
    kinetic[i] = ke
  return {
    "sin_theta1": sin_theta1,
    "sin_theta2": sin_theta2,
    "omega1": omega1,
    "omega2": omega2,
    "potential": potential,
    "kinetic": kinetic,
  }


@dataclass
class SurrogateSource:
  """
  MLP predictions over a pool time grid for one checkpoint.

  IC row comes from ``TrajectoryView.params`` (training contract).
  """

  checkpoint_path: Path
  label: str = ""
  device: str = "cpu"
  _model: object = field(default=None, repr=False)
  _meta: dict = field(default_factory=dict, repr=False)
  _pred_cache_key: tuple | None = field(default=None, repr=False)
  _pred: NDArray[np.float64] | None = field(default=None, repr=False)
  _tips: list[tuple[float, float]] | None = field(default=None, repr=False)

  def __post_init__(self) -> None:
    self.checkpoint_path = Path(self.checkpoint_path)
    if not self.label:
      self.label = self.checkpoint_path.parent.parent.name

  def load(self) -> None:
    if self._model is not None:
      return
    model, meta, _ = load_model_from_checkpoint(self.checkpoint_path, device=self.device)
    self._model = model
    self._meta = meta

  def meta(self) -> dict:
    self.load()
    return dict(self._meta)

  def is_ready(self) -> bool:
    return self._model is not None

  def invalidate_cache(self) -> None:
    self._pred_cache_key = None
    self._pred = None
    self._tips = None

  def predict_series(
    self,
    view: TrajectoryView,
    cache_key: tuple,
  ) -> NDArray[np.float64]:
    """Return ``(n_t, 6)``; cache on ``cache_key`` (stage, split, traj index)."""
    if self._pred is not None and self._pred_cache_key == cache_key:
      return self._pred
    self.load()
    assert self._model is not None
    pred = predict_at_times(
      self._model,
      np.asarray(view.t, dtype=np.float64),
      view.params,
      device=self.device,
    )
    self._pred = pred
    self._pred_cache_key = cache_key
    self._tips = None
    return pred

  def frame_state(self, k: int, view: TrajectoryView, cache_key: tuple) -> PendulumState:
    # Pool IC matches stored frame 0; anchor SVG so all arms start aligned.
    if k == 0:
      return view.frame_at(0)
    pred = self.predict_series(view, cache_key)
    return decode_pred_row(pred[k])

  def tip_trail(
    self,
    k: int,
    view: TrajectoryView,
    params: PendulumParams,
    cache_key: tuple,
  ) -> list[tuple[float, float]]:
    if self._tips is None or self._pred_cache_key != cache_key:
      pred = self.predict_series(view, cache_key)
      tips: list[tuple[float, float]] = []
      for i in range(pred.shape[0]):
        state = view.frame_at(0) if i == 0 else decode_pred_row(pred[i])
        _, _, x2, y2 = cartesian(state, params)
        tips.append((x2, y2))
      self._tips = tips
    return self._tips[: k + 1]

  def layer_id(self) -> str:
    return f"nn:{self.label}"


STORED_LAYER_ID = "stored"

# Dashed overlay palettes (index 0 unused; stored uses primary solid in build_svg).
OVERLAY_PALETTES: list[dict[str, str]] = [
  {"stroke": "#c2410c", "bob1": "#fb923c", "bob2": "#0891b2"},
  {"stroke": "#15803d", "bob1": "#4ade80", "bob2": "#be185d"},
  {"stroke": "#4338ca", "bob1": "#818cf8", "bob2": "#b45309"},
  {"stroke": "#0f766e", "bob1": "#2dd4bf", "bob2": "#7e22ce"},
]
