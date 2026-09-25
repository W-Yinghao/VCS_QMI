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


class InteractCritic(nn.Module):
    """Named variant: MLP on [z1, z2, z1*z2, |z1-z2|] (explicit interaction features), ReLU, tanh output, pointwise per pair."""

    def __init__(self, feature_dim: int, hidden_dims: list[int], last_layer_gain: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        layers: list[nn.Module] = []
        d = 4 * feature_dim
        for w in hidden_dims:
            layers += [nn.Linear(d, w), nn.ReLU()]
            d = w
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=last_layer_gain)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        x = torch.cat((left, right, left * right, (left - right).abs()), dim=-1)
        return torch.tanh(self.net(x).squeeze(-1))


class BilinearConcatCritic(nn.Module):
    """Named variant: tanh( z1^T W z2 + MLP([z1; z2]) ); W and the MLP last layer Xavier-uniform with the same small gain."""

    def __init__(self, feature_dim: int, hidden_dims: list[int], last_layer_gain: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.W = nn.Parameter(torch.empty(feature_dim, feature_dim))
        nn.init.xavier_uniform_(self.W, gain=last_layer_gain)
        self.mlp = PairCriticMLP(feature_dim, hidden_dims, last_layer_gain)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        bil = ((left @ self.W) * right).sum(-1)
        logit = self.mlp.net(torch.cat((left, right), dim=-1)).squeeze(-1)
        return torch.tanh(bil + logit)


class CosineCritic(nn.Module):
    """Named variant: tanh(a * <z1, z2> + b) with two learnable scalars (similarity-form critic)."""

    def __init__(self, feature_dim: int, scale_init: float = 1.0) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.scale * (left * right).sum(-1) + self.bias)


CRITIC_INPUTS = ("ordered_concat", "concat_interact", "bilinear_concat", "cosine")


def critic_impl_name(c: dict[str, Any]) -> str:
    inp = c["input"]
    if inp == "concat_interact":
        return "vcs_ssl.models.critic.InteractCritic"
    if inp == "bilinear_concat":
        return "vcs_ssl.models.critic.BilinearConcatCritic"
    if inp == "cosine":
        return "vcs_ssl.models.critic.CosineCritic"
    hd = list(c["hidden_dims"])
    if len(hd) == 2 and hd[0] == hd[1] and float(c["last_layer_xavier_gain"]) == 0.1:
        return "reference.ssl_core.PairCritic"
    return "vcs_ssl.models.critic.PairCriticMLP"


def build_critic(c: dict[str, Any], *, feature_dim: int) -> nn.Module:
    if not c["enabled"]:
        raise ValueError("build_critic called for a method without a critic")
    if c["input"] not in CRITIC_INPUTS or c["activation"] != "relu" or c["output"] != "tanh" or c["batchnorm"] or c["dropout"] != 0.0:
        raise ValueError(f"critic config must use input in {CRITIC_INPUTS}, ReLU, tanh output, no BN/dropout")
    if c["last_layer_bias"] != 0.0:
        raise ValueError("last layer bias must be initialized to 0")
    hd = [int(w) for w in c["hidden_dims"]]
    gain = float(c["last_layer_xavier_gain"])
    name = critic_impl_name(c)
    if name == "reference.ssl_core.PairCritic":
        return PairCritic(feature_dim=feature_dim, hidden_dim=hd[0])
    if name.endswith("InteractCritic"):
        return InteractCritic(feature_dim, hd, last_layer_gain=gain)
    if name.endswith("BilinearConcatCritic"):
        return BilinearConcatCritic(feature_dim, hd, last_layer_gain=gain)
    if name.endswith("CosineCritic"):
        return CosineCritic(feature_dim, scale_init=1.0)
    return PairCriticMLP(feature_dim, hd, last_layer_gain=gain)
