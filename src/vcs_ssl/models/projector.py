"""Projector: Linear(512,512,no bias) -> BN -> ReLU -> Linear(512,128,bias).  No output BN (spec §5.2)."""
from __future__ import annotations

from typing import Any

from torch import nn


def build_projector(p: dict[str, Any], *, in_dim: int) -> nn.Sequential:
    depth = int(p.get("depth", 2))  # number of Linear layers (2 = frozen default); each hidden block = Linear(no bias)-BN-ReLU
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(d, p["hidden_dim"], bias=p["hidden_linear_bias"]))
        if p["hidden_batchnorm"]:
            layers.append(nn.BatchNorm1d(p["hidden_dim"]))
        layers.append(nn.ReLU(inplace=True))
        d = p["hidden_dim"]
    layers.append(nn.Linear(d, p["output_dim"], bias=p["output_linear_bias"]))
    if p["output_batchnorm"]:
        # named variant: affine-free BN on the projector output (before the L2 normalization applied by the objective)
        layers.append(nn.BatchNorm1d(p["output_dim"], affine=False))
    return nn.Sequential(*layers)


def build_predictor(p: dict[str, Any]) -> nn.Sequential:
    """Named variant (BYOL/SimSiam-style asymmetry): Linear(out,hidden,no bias)-BN-ReLU-Linear(hidden,out) on the student branch."""
    return nn.Sequential(nn.Linear(p["output_dim"], p["hidden_dim"], bias=False), nn.BatchNorm1d(p["hidden_dim"]), nn.ReLU(inplace=True),
                         nn.Linear(p["hidden_dim"], p["output_dim"], bias=True))
