"""Projector: Linear(512,512,no bias) -> BN -> ReLU -> Linear(512,128,bias).  No output BN (spec §5.2)."""
from __future__ import annotations

from typing import Any

from torch import nn


def build_projector(p: dict[str, Any], *, in_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, p["hidden_dim"], bias=p["hidden_linear_bias"])]
    if p["hidden_batchnorm"]:
        layers.append(nn.BatchNorm1d(p["hidden_dim"]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(p["hidden_dim"], p["output_dim"], bias=p["output_linear_bias"]))
    if p["output_batchnorm"]:
        raise ValueError("output BN is not part of the frozen projector")
    return nn.Sequential(*layers)
