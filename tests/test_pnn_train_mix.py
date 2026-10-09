"""PNN train mix filters frozen-column stages."""

from srcs.train.pnn_train_mix import pnn_train_stage_fractions


def test_pnn_train_fractions_drop_frozen_stages() -> None:
  out = pnn_train_stage_fractions({0: 0.4, 1: 0.6}, active_max_stage=1)
  assert out == {1: 1.0}


def test_pnn_train_fractions_renormalize_active() -> None:
  out = pnn_train_stage_fractions({1: 0.25, 2: 0.75}, active_max_stage=1)
  assert out[1] == 0.25
  assert out[2] == 0.75
