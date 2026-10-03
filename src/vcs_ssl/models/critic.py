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

    def __init__(self, feature_dim: int, scale_init: float = 1.0, scale_fixed: bool = False, bias_init: float = 0.0) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(float(scale_init)), requires_grad=not scale_fixed)
        self.bias = nn.Parameter(torch.tensor(float(bias_init)))  # bias_init 0.0 (every config before P104) = the frozen init

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        """f = a<z1, z2> + b (P104: shared interface of learned / fixed / noisy angular critics)."""
        return self.scale * (left * right).sum(-1) + self.bias

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.scale * (left * right).sum(-1) + self.bias)


class FixedCosineCritic(nn.Module):
    """P104 G line (package v3 §5.1): T = tanh(a<z1, z2> + b) with a, b FIXED — registered as buffers, not parameters, so the critic has
    zero trainable parameters and nothing re-introduces a free output affine.  Same interface as CosineCritic (scale, bias, logits,
    forward, embed, score_matrix); ``trainable_affine`` is recorded explicitly (do not infer it from model.parameters())."""

    trainable_affine = False

    def __init__(self, feature_dim: int, scale: float, bias: float) -> None:
        super().__init__()
        if not (torch.isfinite(torch.tensor(float(scale))) and torch.isfinite(torch.tensor(float(bias)))):
            raise ValueError("fixed affine parameters must be finite")
        self.feature_dim = feature_dim
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("bias", torch.tensor(float(bias)))

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        return self.scale * (left * right).sum(-1) + self.bias

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.logits(left, right))

    def embed(self, z: Tensor) -> Tensor:
        return z

    def score_matrix(self, C: Tensor) -> Tensor:
        return torch.tanh(self.scale * C + self.bias)


class FixedCurvedCosineCritic(nn.Module):
    """P126 (v6 V6-CURVE): fixed anchored-quadratic angular scorer, T = tanh f with
        f(s) = a·s + b + a·λ·s(1 − s) = a[(s − κ) + λ s(1 − s)],   κ = −b/a  (the AFFINE anchor; not the zero of f when λ ≠ 0).
    a, b, λ are buffers fixed for the whole run (no trainable affine; the optimizer gets no critic group).  f(0) = b = −aκ, f(1) = a + b
    = a(1 − κ) for every λ; f′(s) = a[1 + λ(1 − 2s)] ∈ [a/4, 7a/4] on s ∈ [−1, 1] when a > 0 and |λ| ≤ 1/4, so f is strictly monotone.
    logits(left, right), forward and score_matrix(C) all go through ``logits_from_similarity`` (the all-view matrix path and the matched-JS
    loss read the same non-linearity).  λ = 0 evaluates exactly FixedCosineCritic's expression a·s + b (the curvature term is skipped), so
    logits, losses and gradients equal A-P3's bit for bit.  Inputs must already be row-L2-normalised (no silent renormalisation)."""

    trainable_affine = False
    is_curved = True

    def __init__(self, feature_dim: int, scale: float, bias: float, curvature: float) -> None:
        super().__init__()
        vals = torch.tensor([float(scale), float(bias), float(curvature)], dtype=torch.float64)
        if not bool(torch.isfinite(vals).all()):
            raise ValueError("fixed curved scorer parameters must be finite")
        if not float(scale) > 0.0:
            raise ValueError("fixed curved scorer needs a > 0 (monotonicity a/4 <= f' <= 7a/4)")
        if abs(float(curvature)) > 0.25:
            raise ValueError("fixed curved scorer needs |lambda| <= 1/4 (keeps f strictly monotone on [-1, 1])")
        self.feature_dim = feature_dim
        self._lam = float(curvature)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("bias", torch.tensor(float(bias)))
        self.register_buffer("curvature", torch.tensor(float(curvature)))

    def logits_from_similarity(self, s: Tensor) -> Tensor:
        f = self.scale * s + self.bias
        if self._lam != 0.0:
            f = f + self.scale * self.curvature * s * (1.0 - s)
        return f

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        return self.logits_from_similarity((left * right).sum(-1))

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.logits(left, right))

    def embed(self, z: Tensor) -> Tensor:
        return z

    def score_matrix(self, C: Tensor) -> Tensor:
        return torch.tanh(self.logits_from_similarity(C))

    def slope_from_similarity(self, s: Tensor) -> Tensor:
        """f′(s) = a[1 + λ(1 − 2s)] (closed form; tests compare it with autograd)."""
        return self.scale * (1.0 + self.curvature * (1.0 - 2.0 * s))

    def curve_record(self) -> dict[str, Any]:
        """Run-manifest record: the anchor κ, the endpoints, the slope bounds on [−1, 1] and the ACTUAL zero s0 of f (bisection, float64)."""
        a, b, lam = float(self.scale), float(self.bias), self._lam
        ends = (a * (1.0 + 3.0 * lam), a * (1.0 - lam))  # f'(-1), f'(1); f' is linear in s, so these are the extremes on [-1, 1]
        return {"formula": "f(s) = a*s + b + a*lambda*s*(1-s); T = tanh f", "a": a, "b": b, "kappa_anchor": -b / a, "lambda": lam,
                "actual_zero_s0": curve_actual_zero(a, b, lam), "f_at_0": b, "f_at_1": a + b,
                "slope_min_on_cos": min(ends), "slope_max_on_cos": max(ends),
                "slope_bounds_guaranteed": [a / 4.0, 7.0 * a / 4.0]}


def curve_logit_float(s: float, a: float, b: float, lam: float) -> float:
    return a * s + b + a * lam * s * (1.0 - s)


def curve_actual_zero(a: float, b: float, lam: float, iters: int = 200) -> float | None:
    """Unique zero of f(s) = a·s + b + a·λ·s(1 − s) on [−1, 1] (f strictly increasing there for a > 0, |λ| ≤ 1/4); None if f has no sign change
    on [−1, 1] (then every cosine maps to one sign of T)."""
    lo, hi = -1.0, 1.0
    flo, fhi = curve_logit_float(lo, a, b, lam), curve_logit_float(hi, a, b, lam)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if curve_logit_float(mid, a, b, lam) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


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


class RFFTanhCritic(nn.Module):
    """S-Kernel (Server Spec v2 §3.4): fixed random-Fourier features of the pair vector w = [z1; z2], a trainable linear read-out
    with intercept, tanh output — the same J, pairing and gradient routing as the neural critics, only the function class changes.

        φ(w) = sqrt(2/m) cos(Ω w + b),   Ω = Ω0 / σ,  Ω0 ~ N(0, I) fixed,  b ~ U(0, 2π) fixed,   T = tanh(θᵀ [φ(w); 1]).

    σ (Gaussian-kernel bandwidth) = ``bandwidth_multiple`` × the FIT median pairwise distance of w, measured once at the start of the run
    (``set_bandwidth``; stored as a buffer so checkpoints / resume / evaluation carry it).  Ω0 and b are drawn from the critic's own
    seed stream at construction (recorded through ``critic_init_sha256``).  θ is Xavier-uniform with the configured small gain (an exactly
    zero θ would block the encoder gradient at step 1, as for the MLP critics).  Trainable parameters: m + 1.
    """

    def __init__(self, feature_dim: int, n_features: int, bandwidth_multiple: float, last_layer_gain: float = 0.1) -> None:
        super().__init__()
        if feature_dim < 1 or n_features < 1 or bandwidth_multiple <= 0 or last_layer_gain <= 0:
            raise ValueError("feature_dim, n_features >= 1; bandwidth_multiple, gain > 0")
        self.feature_dim = int(feature_dim)
        self.n_features = int(n_features)
        self.bandwidth_multiple = float(bandwidth_multiple)
        self.register_buffer("omega0", torch.randn(self.n_features, 2 * self.feature_dim))
        self.register_buffer("phase", torch.rand(self.n_features) * (2.0 * torch.pi))
        self.register_buffer("sigma", torch.tensor(1.0))  # replaced by set_bandwidth() at run start
        self.register_buffer("calibrated", torch.tensor(0, dtype=torch.int64))
        self.readout = nn.Linear(self.n_features, 1, bias=True)  # θ (m weights + intercept)
        nn.init.xavier_uniform_(self.readout.weight, gain=last_layer_gain)
        nn.init.zeros_(self.readout.bias)

    @torch.no_grad()
    def set_bandwidth(self, median_pair_distance: float) -> float:
        if not (median_pair_distance > 0):
            raise ValueError("median pair distance must be > 0")
        self.sigma.fill_(self.bandwidth_multiple * float(median_pair_distance))
        self.calibrated.fill_(1)
        return float(self.sigma)

    def features(self, w: Tensor) -> Tensor:
        return torch.cos(w @ (self.omega0 / self.sigma).T + self.phase) * (2.0 / self.n_features) ** 0.5

    def logit(self, left: Tensor, right: Tensor) -> Tensor:
        return self.readout(self.features(torch.cat((left, right), dim=-1))).squeeze(-1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
            raise ValueError("critic inputs must have matching [N,D] shapes with D = feature_dim")
        return torch.tanh(self.logit(left, right))


# ----------------------------------------------------------------------------------------------------------------------
# P95 (package v1 estimator improvements inside full SSL; owner 2026-09-29).  Each is a named critic variant; the frozen recipe critics
# above are untouched.  All outputs stay in [-1, 1] (the collaborator's bounded-critic contract); J, pairing and detach are unchanged.
class _BatchStats:
    """Accumulates detached per-call statistics while the critic is in training mode; ``pop_stats`` returns call-averaged floats
    (one call = the positive or the negative block of one view pair, so the average weights P and Q equally = E_M)."""

    def _stats_reset(self) -> None:
        self._acc: dict[str, Tensor] = {}
        self._n_calls = 0

    def _stats_add(self, **vals: Tensor) -> None:
        if not self.training:
            return
        for k, v in vals.items():
            v = v.detach().float()
            self._acc[k] = v if k not in self._acc else self._acc[k] + v
        self._n_calls += 1

    def pop_stats(self) -> dict[str, float]:
        n = max(1, self._n_calls)
        out = {k: float(v) / n for k, v in self._acc.items()}
        out.update(self._extra_stats())
        self._stats_reset()
        return out

    def _extra_stats(self) -> dict[str, float]:
        return {}


class ResidualCosineMLPCritic(_BatchStats, nn.Module):
    """P95 variant 1 (v1 plan §5.1, bounded residual): T = (1 − λ)·tanh(a⟨z1, z2⟩ + b) + λ·tanh(g([z1; z2])), λ fixed.
    The cosine part is the recipe critic (a0 from the config); g is the recipe's ordered-concat MLP class (hidden_dims, ReLU,
    last layer Xavier gain, bias 0).  |T| ≤ 1 because it is a convex combination of two tanh outputs."""

    def __init__(self, feature_dim: int, hidden_dims: list[int], *, lam: float, scale_init: float, last_layer_gain: float = 0.1) -> None:
        super().__init__()
        if not (0.0 < lam < 1.0):
            raise ValueError("residual lambda must be in (0, 1)")
        self.feature_dim = feature_dim
        self.lam = float(lam)
        self.cos = CosineCritic(feature_dim, scale_init=scale_init)
        self.mlp = PairCriticMLP(feature_dim, hidden_dims, last_layer_gain=last_layer_gain)
        self._stats_reset()

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        tc = self.cos(left, right)
        tm = self.mlp(left, right)
        t = (1.0 - self.lam) * tc + self.lam * tm
        self._stats_add(res_cos_part_mean=((1.0 - self.lam) * tc).mean(), res_mlp_part_mean=(self.lam * tm).mean(),
                        res_cos_T2=tc.square().mean(), res_mlp_T2=tm.square().mean())
        return t

    def _extra_stats(self) -> dict[str, float]:
        return {"res_lambda": self.lam, "cos_scale": float(self.cos.scale.detach()), "cos_bias": float(self.cos.bias.detach())}


class DictionarySimplexCritic(_BatchStats, nn.Module):
    """P95 variant 2 (v1 plan §5.2, simplex dictionary): T = Σ_j w_j T_j with w = softmax(θ), θ learnable (init 0 = equal weights),
    members {cosine (a0 from config), ordered-concat MLP, bilinear_concat}; all trained jointly by the same J.  Logs w and the dictionary
    disagreement D(w) = Σ_j w_j E_M[T_j²] − E_M[T_w²] (≥ 0; the exact gain of the mix over the weighted member average)."""

    MEMBERS = ("cosine", "mlp", "bilinear")

    def __init__(self, feature_dim: int, hidden_dims: list[int], *, scale_init: float, last_layer_gain: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.cos = CosineCritic(feature_dim, scale_init=scale_init)
        self.mlp = PairCriticMLP(feature_dim, hidden_dims, last_layer_gain=last_layer_gain)
        self.bil = BilinearConcatCritic(feature_dim, hidden_dims, last_layer_gain=last_layer_gain)
        self.theta = nn.Parameter(torch.zeros(3))
        self._stats_reset()

    def weights(self) -> Tensor:
        return torch.softmax(self.theta, dim=0)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        ts = torch.stack((self.cos(left, right), self.mlp(left, right), self.bil(left, right)), dim=-1)  # [N, 3]
        w = self.weights()
        t = ts @ w
        if self.training:
            sq = ts.square().mean(0)
            self._stats_add(dict_member_T2_weighted=(sq * w).sum(), dict_mix_T2=t.square().mean(),
                            dict_T2_cos=sq[0], dict_T2_mlp=sq[1], dict_T2_bil=sq[2])
        return t

    def _extra_stats(self) -> dict[str, float]:
        w = self.weights().detach()
        out = {f"dict_w_{m}": float(w[i]) for i, m in enumerate(self.MEMBERS)}
        out.update({"cos_scale": float(self.cos.scale.detach()), "cos_bias": float(self.cos.bias.detach())})
        return out

    def pop_stats(self) -> dict[str, float]:
        out = super().pop_stats()
        if "dict_member_T2_weighted" in out:
            out["dict_disagreement_D"] = out["dict_member_T2_weighted"] - out["dict_mix_T2"]
        return out


class NoisyCosineCritic(_BatchStats, CosineCritic):
    """P95 variant 3 (v1 plan §6, observation scale): the recipe cosine critic reading u = z + τ·ε/√d on both sides of every pair
    (positive and negative blocks each draw fresh ε; z is the L2-normalised projector output and is *not* re-normalised).  Training
    mode only; evaluation / hold-out read clean z.  Noise stream: counter-based, ε of call c drawn from a generator seeded with
    (noise_seed, c); noise_seed comes from the critic's own seed stream at construction and both are buffers, so checkpoints / resume
    reproduce the sequence.  Parameters and their init are identical to the recipe critic (same state_dict keys for scale / bias)."""

    is_noisy = True  # P104: objectives use logits() so that the matched-JS control sees the same noisy logit

    def __init__(self, feature_dim: int, scale_init: float = 1.0, *, tau: float) -> None:
        CosineCritic.__init__(self, feature_dim, scale_init=scale_init)
        if not (tau > 0):
            raise ValueError("observation noise tau must be > 0")
        self.tau = float(tau)
        self.register_buffer("noise_seed", torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64))
        self.register_buffer("noise_calls", torch.zeros(1, dtype=torch.int64))
        self._stats_reset()

    def _noise(self, x: Tensor) -> Tensor:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(self.noise_seed) * 1_000_003 + int(self.noise_calls))
        self.noise_calls += 1
        return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * (self.tau / x.shape[1] ** 0.5)

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        """Noisy logit f = a<u1, u2> + b, u = z + tau*eps/sqrt(d) with fresh eps per call and side (training mode only)."""
        if self.training:
            el, er = self._noise(left), self._noise(right)
            self._stats_add(noise_coord_sd_emp=torch.cat((el, er)).std(), noise_total_rms_emp=torch.cat((el, er)).square().sum(-1).mean().sqrt())
            left, right = left + el, right + er
        return self.scale * (left * right).sum(-1) + self.bias

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.tanh(self.logits(left, right))

    def score_matrix(self, C: Tensor) -> Tensor:  # noqa: ARG002
        raise TypeError("P104 (package v3 §6.4): the noisy critic has no matrix hook — a score_matrix path would skip the noise; "
                        "use the per-pair path (forward / logits)")

    def embed(self, z: Tensor) -> Tensor:  # noqa: ARG002
        raise TypeError("the noisy critic has no matrix hook (package v3 §6.4)")

    def _extra_stats(self) -> dict[str, float]:
        return {"noise_total_rms": self.tau, "noise_coordinate_sd": self.tau / self.feature_dim ** 0.5}


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

CRITIC_INPUTS = ("ordered_concat", "concat_interact", "bilinear_concat", "cosine", "interact_only", "shared_metric", "mono_spline", "diag_metric", "rff_tanh", "residual_cosine_mlp", "dictionary_simplex")


def critic_impl_name(c: dict[str, Any]) -> str:
    inp = c["input"]
    if inp == "residual_cosine_mlp":
        return "vcs_ssl.models.critic.ResidualCosineMLPCritic"
    if inp == "dictionary_simplex":
        return "vcs_ssl.models.critic.DictionarySimplexCritic"
    if inp == "cosine" and float(c.get("observation_noise_tau", 0.0)) > 0:
        return "vcs_ssl.models.critic.NoisyCosineCritic"
    if inp == "cosine" and c.get("affine_mode", "learned") == "fixed_curved":  # P126 (v6 V6-CURVE)
        return "vcs_ssl.models.critic.FixedCurvedCosineCritic"
    if inp == "cosine" and c.get("affine_mode", "learned") == "fixed":
        return "vcs_ssl.models.critic.FixedCosineCritic"
    if inp == "rff_tanh":
        return "vcs_ssl.models.critic.RFFTanhCritic"
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
    if name.endswith("ResidualCosineMLPCritic"):
        return ResidualCosineMLPCritic(feature_dim, hd, lam=float(c["residual_lambda"]), scale_init=float(c.get("cosine_scale_init", 1.0)), last_layer_gain=gain)
    if name.endswith("DictionarySimplexCritic"):
        return DictionarySimplexCritic(feature_dim, hd, scale_init=float(c.get("cosine_scale_init", 1.0)), last_layer_gain=gain)
    if name.endswith("NoisyCosineCritic"):
        if c.get("cosine_scale_fixed", False):
            raise ValueError("observation noise is defined for the recipe cosine critic (learnable scale)")
        return NoisyCosineCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), tau=float(c["observation_noise_tau"]))
    if name.endswith("RFFTanhCritic"):
        return RFFTanhCritic(feature_dim, int(c.get("rff_features", 1024)), float(c.get("rff_bandwidth_multiple", 1.0)), last_layer_gain=gain)
    if name.endswith("InteractCritic"):
        return InteractCritic(feature_dim, hd, last_layer_gain=gain)
    if name.endswith("BilinearConcatCritic"):
        return BilinearConcatCritic(feature_dim, hd, last_layer_gain=gain)
    if name.endswith("FixedCurvedCosineCritic"):  # before the generic *CosineCritic match
        return FixedCurvedCosineCritic(feature_dim, float(c.get("cosine_scale_init", 1.0)), float(c.get("cosine_bias_init", 0.0)),
                                       float(c["curvature_lambda"]))
    if name.endswith("FixedCosineCritic"):
        return FixedCosineCritic(feature_dim, float(c.get("cosine_scale_init", 1.0)), float(c.get("cosine_bias_init", 0.0)))
    if name.endswith("CosineCritic"):
        return CosineCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), scale_fixed=bool(c.get("cosine_scale_fixed", False)),
                            bias_init=float(c.get("cosine_bias_init", 0.0)))
    if name.endswith("InteractOnlyCritic"):
        return InteractOnlyCritic(feature_dim, hd, last_layer_gain=gain)
    if name.endswith("SharedMetricCritic"):
        return SharedMetricCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), scale_fixed=bool(c.get("cosine_scale_fixed", False)))
    if name.endswith("MonoSplineCritic"):
        return MonoSplineCritic(feature_dim, knots=int(hd[0]))  # hidden_dims[0] = number of knots (documented reuse of the field)
    if name.endswith("DiagMetricCritic"):
        return DiagMetricCritic(feature_dim, scale_init=float(c.get("cosine_scale_init", 1.0)), scale_fixed=bool(c.get("cosine_scale_fixed", False)))
    return PairCriticMLP(feature_dim, hd, last_layer_gain=gain)
