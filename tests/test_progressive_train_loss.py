"""Progressive train_loss logging matches inlet-weighted mix (not active-only loader)."""

from __future__ import annotations

from srcs.train.base import StageValLoss
from srcs.train.curriculum_inlet_loop import mix_weighted_val_mse


def test_mix_weighted_train_jumps_less_than_active_only_on_new_stage() -> None:
  """When stage 2 enters the mix, frozen cols 0/1 keep weighted train_loss stable."""
  fractions = {0: 0.2, 1: 0.3, 2: 0.5}
  stage_train = [
    StageValLoss(stage=0, mse=0.10),
    StageValLoss(stage=1, mse=0.12),
    StageValLoss(stage=2, mse=0.90),
  ]
  mix = mix_weighted_val_mse(2, fractions, stage_train)
  active_only = 0.90
  assert mix < active_only
  assert mix < 0.55
