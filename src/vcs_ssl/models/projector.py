"""Projector: Linear(512,512,no bias) -> BN -> ReLU -> Linear(512,128,bias).  No output BN (spec §5.2)."""
from __future__ import annotations

from typing import Any

from torch import nn


def build_projector(p: dict[str, Any], *, in_dim: int) -> nn.Sequential:
    if p.get("kind", "mlp") == "bn_only":
        # named variant: no projector; the critic reads L2(BN(h)) with an affine-free BN (scale control for similarity critics)
        if p["output_dim"] != in_dim:
            raise ValueError("bn_only projector requires output_dim == in_dim")
        return nn.Sequential(nn.BatchNorm1d(in_dim, affine=False))
    depth = int(p.get("depth", 2))  # number of Linear layers (2 = frozen default; 1 = single linear map); each hidden block = Linear(no bias)-BN-ReLU
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
