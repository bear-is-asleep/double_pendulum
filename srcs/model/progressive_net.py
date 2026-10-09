"""Progressive Neural Network columns with optional lateral adapters (experiment #4)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from srcs.model.mlp import _activation, resolve_dropout_rate, resolve_hidden_layer_widths


class _ColumnHiddenBlock(nn.Module):
  """One hidden layer in a column: main linear + laterals from prior columns."""

  def __init__(
    self,
    in_dim: int,
    out_dim: int,
    *,
    n_prior: int,
    use_lateral: bool,
    activation: nn.Module,
    dropout_p: float,
  ) -> None:
    super().__init__()
    self.fc = nn.Linear(in_dim, out_dim)
    self.use_lateral = use_lateral and n_prior > 0
    self.laterals = (
      nn.ModuleList([nn.Linear(out_dim, out_dim, bias=False) for _ in range(n_prior)])
      if self.use_lateral
      else nn.ModuleList()
    )
    self.act = activation
    self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

  def forward(
    self,
    x_in: torch.Tensor,
    prior_h: list[torch.Tensor] | None,
  ) -> torch.Tensor:
    pre = self.fc(x_in)
    if self.use_lateral and prior_h is not None:
      for i, lat in enumerate(self.laterals):
        pre = pre + lat(prior_h[i])
    h = self.act(pre)
    h = self.dropout(h)
    return h


class _PnnColumn(nn.Module):
  def __init__(
    self,
    column_index: int,
    widths: list[int],
    *,
    input_dim: int,
    output_dim: int,
    activation_name: str,
    dropout_p: float,
    use_lateral: bool,
  ) -> None:
    super().__init__()
    self.column_index = int(column_index)
    act_mod = _activation(activation_name)
    blocks: list[_ColumnHiddenBlock] = []
    prev = input_dim
    for ell, w in enumerate(widths):
      blocks.append(
        _ColumnHiddenBlock(
          prev,
          w,
          n_prior=column_index,
          use_lateral=use_lateral,
          activation=act_mod,
          dropout_p=dropout_p,
        )
      )
      prev = w
    self.blocks = nn.ModuleList(blocks)
    self.head = nn.Linear(prev, output_dim)

  def forward_hidden(
    self,
    x: torch.Tensor,
    prior_hidden_per_col: list[list[torch.Tensor]] | None,
  ) -> list[torch.Tensor]:
    """Post-activation hidden states per layer (for laterals into later columns)."""
    hs: list[torch.Tensor] = []
    cur = x
    for ell, block in enumerate(self.blocks):
      prior_h = None
      if prior_hidden_per_col is not None and block.use_lateral:
        prior_h = [prior_hidden_per_col[i][ell] for i in range(self.column_index)]
      cur = block(cur, prior_h)
      hs.append(cur)
    return hs

  def forward_from_hidden(self, last_h: torch.Tensor) -> torch.Tensor:
    return self.head(last_h)


class ProgressiveNet(nn.Module):
  """One column per curriculum stage; Rusu-style laterals from frozen columns."""

  def __init__(self, cfg: dict[str, Any]) -> None:
    super().__init__()
    self.cfg = dict(cfg)
    self.input_dim = int(cfg["input_dim"])
    self.output_dim = int(cfg["output_dim"])
    self.widths = resolve_hidden_layer_widths(cfg)
    self.use_lateral = bool(cfg.get("use_lateral", True))
    self._activation_name = str(cfg["activation"])
    self._dropout_p = resolve_dropout_rate(cfg)
    self.columns = nn.ModuleList()

  @property
  def n_columns(self) -> int:
    return len(self.columns)

  def add_column(self, index: int) -> None:
    if index != self.n_columns:
      raise ValueError(f"add_column expects index {self.n_columns}, got {index}")
    device: torch.device | None = None
    if self.n_columns > 0:
      device = next(self.parameters()).device
    col = _PnnColumn(
      index,
      self.widths,
      input_dim=self.input_dim,
      output_dim=self.output_dim,
      activation_name=self._activation_name,
      dropout_p=self._dropout_p,
      use_lateral=self.use_lateral,
    )
    if device is not None:
      col = col.to(device)
    self.columns.append(col)

  def freeze_columns_before(self, j: int) -> None:
    for c in range(min(j, self.n_columns)):
      for p in self.columns[c].parameters():
        p.requires_grad = False

  def forward(self, x: torch.Tensor, column_index: int) -> torch.Tensor:
    if column_index < 0 or column_index >= self.n_columns:
      raise IndexError(
        f"column_index {column_index} out of range for n_columns={self.n_columns}"
      )
    hidden_per_col: list[list[torch.Tensor]] = []
    for j in range(column_index + 1):
      prior = hidden_per_col if self.use_lateral else None
      hs = self.columns[j].forward_hidden(x, prior)
      hidden_per_col.append(hs)
      if j == column_index:
        return self.columns[j].forward_from_hidden(hs[-1])
    raise RuntimeError("unreachable")

  def forward_batch_by_stage(
    self,
    x: torch.Tensor,
    stage_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Pointwise forward when each row maps to column ``stage_ids[i]``."""
    pred_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    for stage in stage_ids.unique(sorted=True).tolist():
      stage_i = int(stage)
      mask = stage_ids == stage_i
      idx = mask.nonzero(as_tuple=False).squeeze(1)
      pred_chunks.append(self.forward(x[mask], column_index=stage_i))
      index_chunks.append(idx)
    pred_cat = torch.cat(pred_chunks, dim=0)
    idx_cat = torch.cat(index_chunks, dim=0)
    _, order = torch.sort(idx_cat)
    return pred_cat[order]


def build_progressive_net(cfg: dict[str, Any]) -> ProgressiveNet:
  net = ProgressiveNet(cfg)
  n_init = int(cfg.get("n_columns", 1))
  if n_init < 1:
    raise ValueError(f"n_columns must be >= 1, got {n_init}")
  for j in range(n_init):
    net.add_column(j)
  return net


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
  if trainable_only:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  return sum(p.numel() for p in model.parameters())


def sync_progressive_cfg(cfg: dict[str, Any], model: ProgressiveNet) -> None:
  """Persist column count and model type on the run config blob."""
  cfg["model_type"] = "progressive_pnn"
  cfg["n_columns"] = int(model.n_columns)


def pnn_stage_ready(model: nn.Module, stage: int) -> bool:
  if not isinstance(model, ProgressiveNet):
    return True
  return int(stage) < model.n_columns


def pnn_stage_predict(model: nn.Module, xb: torch.Tensor, stage: int) -> torch.Tensor:
  if not isinstance(model, ProgressiveNet):
    raise TypeError("pnn_stage_predict requires ProgressiveNet")
  return model.forward(xb, column_index=int(stage))


def pnn_tagged_predict(
  model: nn.Module,
  xb: torch.Tensor,
  stage_ids: torch.Tensor,
) -> torch.Tensor:
  if not isinstance(model, ProgressiveNet):
    raise TypeError("pnn_tagged_predict requires ProgressiveNet")
  return model.forward_batch_by_stage(xb, stage_ids)


def refresh_pnn_param_counts(cfg: dict[str, Any], model: ProgressiveNet) -> tuple[int, int]:
  trainable = count_parameters(model, trainable_only=True)
  total = count_parameters(model, trainable_only=False)
  cfg["_n_params_trainable"] = trainable
  cfg["_n_params_total"] = total
  return trainable, total
