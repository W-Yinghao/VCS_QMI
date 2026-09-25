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
    """Named variant: tanh(a * <z1, z2> + b); a, b scalars (a optionally fixed; b optionally calibrated at init)."""

    def __init__(self, feature_dim: int, scale_init: float = 1.0, scale_fixed: bool = False) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(float(scale_init)), requires_grad=not scale_fixed)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.scale * (left * right).sum(-1) + self.bias)


class InteractOnlyCritic(nn.Module):
    """Named variant: MLP on [z1*z2, |z1-z2|] only (no raw z1/z2 channel); symmetric in the two views; ReLU; tanh output."""

    def __init__(self, feature_dim: int, hidden_dims: list[int], last_layer_gain: float = 0.1) -> None:
        super().__init__()
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
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.net(torch.cat((left * right, (left - right).abs()), dim=-1)).squeeze(-1))


class SharedMetricCritic(nn.Module):
    """Named variant: tanh(a * <normalize(W z1), normalize(W z2)> + b) with one shared square W (init identity); starts equal to CosineCritic."""

    def __init__(self, feature_dim: int, scale_init: float = 1.0, scale_fixed: bool = False, eps: float = 1e-8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.W = nn.Parameter(torch.eye(feature_dim))
        self.scale = nn.Parameter(torch.tensor(float(scale_init)), requires_grad=not scale_fixed)
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        w1 = torch.nn.functional.normalize(left @ self.W.T, dim=1, eps=self.eps)
        w2 = torch.nn.functional.normalize(right @ self.W.T, dim=1, eps=self.eps)
        return torch.tanh(self.scale * (w1 * w2).sum(-1) + self.bias)


class MonoSplineCritic(nn.Module):
    """Named variant: T = tanh(f(<z1,z2>)) with f a monotone increasing piecewise-linear spline on [-1, 1]:
    f(s) = c0 + sum_k softplus(w_k) * relu(s - t_k), knots t_k = -1 + 2k/M (k = 0..M-1). Init: f(s) = s (equal to cosine a=1, b=0)."""

    def __init__(self, feature_dim: int, knots: int = 8) -> None:
        super().__init__()
        if knots < 1:
            raise ValueError("knots must be >= 1")
        self.feature_dim = feature_dim
        self.register_buffer("knots", torch.linspace(-1.0, 1.0, knots + 1)[:-1])
        w = torch.full((knots,), -8.0)  # softplus(-8) = 3.4e-4: other knots start (numerically) flat; Adam still moves them
        w[0] = float(torch.log(torch.expm1(torch.tensor(1.0))))  # softplus(w0) = 1
        self.w = nn.Parameter(w)
        self.c0 = nn.Parameter(torch.tensor(-1.0))

    def f(self, s: Tensor) -> Tensor:
        slopes = torch.nn.functional.softplus(self.w)
        return self.c0 + (slopes * torch.relu(s.unsqueeze(-1) - self.knots)).sum(-1)

    def embed(self, z: Tensor) -> Tensor:
        return z

    def score_matrix(self, C: Tensor) -> Tensor:
        return torch.tanh(self.f(C))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.f((left * right).sum(-1)))


class DiagMetricCritic(nn.Module):
    """Named variant: tanh(a * <normalize(w*z1), normalize(w*z2)> + b) with a learnable per-dimension weight w (init ones)."""

    def __init__(self, feature_dim: int, scale_init: float = 1.0, scale_fixed: bool = False, eps: float = 1e-8) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.w = nn.Parameter(torch.ones(feature_dim))
        self.scale = nn.Parameter(torch.tensor(float(scale_init)), requires_grad=not scale_fixed)
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.eps = eps

    def embed(self, z: Tensor) -> Tensor:
        return torch.nn.functional.normalize(z * self.w, dim=1, eps=self.eps)

    def score_matrix(self, C: Tensor) -> Tensor:
        return torch.tanh(self.scale * C + self.bias)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.scale * (self.embed(left) * self.embed(right)).sum(-1) + self.bias)


# matrix-form hooks for the all-pairs sampler (similarity-type critics only)
def _cos_embed(self, z):  # noqa: ANN001
    return z


def _cos_score_matrix(self, C):  # noqa: ANN001
    return torch.tanh(self.scale * C + self.bias)


CosineCritic.embed = _cos_embed
CosineCritic.score_matrix = _cos_score_matrix


def _sm_embed(self, z):  # noqa: ANN001
    return torch.nn.functional.normalize(z @ self.W.T, dim=1, eps=self.eps)


SharedMetricCritic.embed = _sm_embed
SharedMetricCritic.score_matrix = _cos_score_matrix

CRITIC_INPUTS = ("ordered_concat", "concat_interact", "bilinear_concat", "cosine", "interact_only", "shared_metric", "mono_spline", "diag_metric")


def critic_impl_name(c: dict[str, Any]) -> str:
    inp = c["input"]
    if inp == "concat_interact":
        return "vcs_ssl.models.critic.InteractCritic"
    if inp == "bilinear_concat":
        return "vcs_ssl.models.critic.BilinearConcatCritic"
    if inp == "cosine":
        return "vcs_ssl.models.critic.CosineCritic"
    if inp == "interact_only":
        return "vcs_ssl.models.critic.InteractOnlyCritic"
    if inp == "shared_metric":
        return "vcs_ssl.models.critic.SharedMetricCritic"
    if inp == "mono_spline":
        return "vcs_ssl.models.critic.MonoSplineCritic"
    if inp == "diag_metric":
        return "vcs_ssl.models.critic.DiagMetricCritic"
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
        return CosineCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), scale_fixed=bool(c.get("cosine_scale_fixed", False)))
    if name.endswith("InteractOnlyCritic"):
        return InteractOnlyCritic(feature_dim, hd, last_layer_gain=gain)
    if name.endswith("SharedMetricCritic"):
        return SharedMetricCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), scale_fixed=bool(c.get("cosine_scale_fixed", False)))
    if name.endswith("MonoSplineCritic"):
        return MonoSplineCritic(feature_dim, knots=int(hd[0]))  # hidden_dims[0] = number of knots (documented reuse of the field)
    if name.endswith("DiagMetricCritic"):
        return DiagMetricCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), scale_fixed=bool(c.get("cosine_scale_fixed", False)))
    return PairCriticMLP(feature_dim, hd, last_layer_gain=gain)
