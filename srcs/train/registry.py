"""Map model YAML stem -> trainer (Step 7+ strategies)."""

from __future__ import annotations

from typing import Callable

from srcs.train.base import StrategyTrainer

_TRAINER_FACTORIES: dict[str, Callable[[], StrategyTrainer]] = {}


def _register() -> None:
  if _TRAINER_FACTORIES:
    return
  from srcs.train.baseline import BaselineTrainer
  from srcs.train.curriculum import CurriculumTrainer

  _TRAINER_FACTORIES["baseline"] = BaselineTrainer
  _TRAINER_FACTORIES["curriculum"] = CurriculumTrainer


def list_train_model_names() -> list[str]:
  _register()
  return sorted(_TRAINER_FACTORIES)


def trainer_for_model(model_name: str) -> StrategyTrainer:
  _register()
  factory = _TRAINER_FACTORIES.get(model_name)
  if factory is None:
    known = ", ".join(list_train_model_names()) or "(none)"
    raise KeyError(f"no trainer for model {model_name!r}; known: {known}")
  return factory()
