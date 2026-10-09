"""Train mix rules specific to progressive PNN (frozen columns do not backprop)."""


def pnn_train_stage_fractions(
  inlet_fractions: dict[int, float],
  active_max_stage: int,
) -> dict[int, float]:
  """
  Keep only stages whose column may still receive gradients.

  Prior stages stay in the inlet mix for ``wval`` and logged ``train_loss``;
  optimizer steps use only the active column (``stage >= active_max_stage``).
  """
  kept = {
    int(s): float(f)
    for s, f in inlet_fractions.items()
    if int(s) >= int(active_max_stage) and float(f) > 0.0
  }
  if not kept:
    return {int(active_max_stage): 1.0}
  total = sum(kept.values())
  return {s: f / total for s, f in kept.items()}
