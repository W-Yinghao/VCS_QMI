"""Critic construction.

The collaborator's reference ``PairCritic`` (two equal hidden layers, last layer Xavier-uniform gain 0.1) is used whenever the config
describes it.  Other depths / last-layer gains use :class:`PairCriticMLP`, an explicitly named hyper-parameter variant with the same
contract (ordered concat, ReLU MLP, tanh output, no BN/dropout, pointwise per pair); for ``[w, w]`` and gain 0.1 it is structurally
identical to the reference (tested).
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from reference.ssl_core import PairCritic


class PairCriticMLP(nn.Module):
    """Ordered-pair MLP critic with arbitrary hidden widths; no batch normalization or dropout."""

    def __init__(self, feature_dim: int, hidden_dims: list[int], last_layer_gain: float = 0.1) -> None:
        super().__init__()
        if feature_dim < 1 or not hidden_dims or any(w < 1 for w in hidden_dims):
            raise ValueError("dimensions must be positive")
        if last_layer_gain <= 0:
            raise ValueError("last layer gain must be > 0")
        self.feature_dim = feature_dim
        layers: list[nn.Module] = []
        d = 2 * feature_dim
        for w in hidden_dims:
            layers += [nn.Linear(d, w), nn.ReLU()]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=last_layer_gain)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape:
            raise ValueError("critic inputs must have matching [N,D] shapes")
        if left.shape[1] != self.feature_dim:
            raise ValueError("critic input dimension does not match configuration")
        return torch.tanh(self.net(torch.cat((left, right), dim=-1)).squeeze(-1))


def critic_impl_name(c: dict[str, Any]) -> str:
    hd = list(c["hidden_dims"])
    if len(hd) == 2 and hd[0] == hd[1] and float(c["last_layer_xavier_gain"]) == 0.1:
        return "reference.ssl_core.PairCritic"
    return "vcs_ssl.models.critic.PairCriticMLP"


def build_critic(c: dict[str, Any], *, feature_dim: int) -> nn.Module:
    if not c["enabled"]:
        raise ValueError("build_critic called for a method without a critic")
    if c["input"] != "ordered_concat" or c["activation"] != "relu" or c["output"] != "tanh" or c["batchnorm"] or c["dropout"] != 0.0:
        raise ValueError("critic config is not an ordered-concat ReLU/tanh MLP without BN/dropout")
    if c["last_layer_bias"] != 0.0:
        raise ValueError("last layer bias must be initialized to 0")
    hd = [int(w) for w in c["hidden_dims"]]
    if critic_impl_name(c) == "reference.ssl_core.PairCritic":
        return PairCritic(feature_dim=feature_dim, hidden_dim=hd[0])
    return PairCriticMLP(feature_dim, hd, last_layer_gain=float(c["last_layer_xavier_gain"]))
