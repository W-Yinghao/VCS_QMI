"""Synthetic settings with analytic / quadrature truth, PMI functions for oracle critics, and the R1 contaminations.

Every setting returns joint samples (x, y) with x, y of shape [n, d] and exposes ``pmi(x, y)`` = log p(x, y) − log p(x) − log p(y), from
which the oracle critics of every estimator follow (pre-activation f*: VCS pmi/2 so that T = tanh f* = eta; JS, InfoNCE, DV, SMILE: pmi; NWJ: 1 + pmi).
Every setting is also written as a reparameterised sampler: ``sample_base`` draws the parameter-free base variables, ``from_base`` maps them
to y for a given dependence parameter, so the channel derivative d/dθ of a fixed critic's objective can flow through the generator with common
random numbers (package v2 spec §5.3); the marginals of x and y do not depend on the parameter in any setting, so Q is parameter-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

LEVELS_NATS = (2.0, 4.0, 6.0, 8.0, 10.0)
XOR_PAIRS = 10


def rho_from_mi(mi: float, d: int) -> float:
    """d independent bivariate-normal coordinates with correlation rho: MI = −d/2·log(1 − rho²)."""
    return math.sqrt(1.0 - math.exp(-2.0 * mi / d))


def mi_gaussian(rho: float, d: int) -> float:
    return -0.5 * d * math.log(1.0 - rho * rho)


def pmi_gaussian(x: torch.Tensor, y: torch.Tensor, rho) -> torch.Tensor:
    """Per-sample PMI of d independent bivariate normals (unit variances, correlation rho); rho may be a tensor (for autograd)."""
    r2 = rho * rho
    per = -0.5 * torch.log1p(-r2 * torch.ones((), dtype=x.dtype, device=x.device)) - (r2 * x * x - 2.0 * rho * x * y + r2 * y * y) / (2.0 * (1.0 - r2))
    return per.sum(-1)


@dataclass
class Setting:
    name: str
    d: int
    level: int            # index into LEVELS_NATS (or −1 when built from an arbitrary MI)
    param: float          # rho (gaussian / cubic) or c (xor_mixture)
    mi: float             # truth in nats

    # -- reparameterised sampler --------------------------------------------------------------------------------------
    def sample_base(self, n: int, gen: torch.Generator, device=None) -> tuple:
        raise NotImplementedError

    def from_base(self, base: tuple, param, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sample(self, n: int, gen: torch.Generator, device=None):
        return self.from_base(self.sample_base(n, gen), self.param, device)

    def pmi(self, x: torch.Tensor, y: torch.Tensor, param=None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def param_name(self) -> str:
        return "c" if self.name == "xor_mixture" else "rho"


class Gaussian(Setting):
    def sample_base(self, n, gen, device=None):
        return torch.randn(n, self.d, generator=gen), torch.randn(n, self.d, generator=gen)

    def from_base(self, base, param, device=None):
        x, e = base
        y = param * x + torch.sqrt(1.0 - param * param) * e if torch.is_tensor(param) else param * x + math.sqrt(1.0 - param ** 2) * e
        return x.to(device), y.to(device)

    def pmi(self, x, y, param=None):
        return pmi_gaussian(x, y, self.param if param is None else param)


class Cubic(Gaussian):
    """y ↦ y³ coordinate-wise (invertible): MI unchanged; the PMI is evaluated at the pre-image.  The classical (Lebesgue) CS reference
    density is not square-integrable here (p(y) ~ |y|^{-2/3} at 0), so no classical-CS truth is attached to this setting."""
    def from_base(self, base, param, device=None):
        x, y = Gaussian.from_base(self, base, param, device)
        return x, y ** 3

    def pmi(self, x, y, param=None):
        return pmi_gaussian(x, torch.sign(y) * y.abs().pow(1.0 / 3.0), self.param if param is None else param)


class XorMixture(Setting):
    """d independent 2-d pairs; each pair ~ ½ N(0, Σ₊) + ½ N(0, Σ₋), Σ± = [[1, ±c], [±c, 1]].  Marginals are N(0, 1); the covariance is 0;
    the dependence is in the sign pattern.  x = (x_1..x_d), y = (y_1..y_d)."""
    def sample_base(self, n, gen, device=None):
        x = torch.randn(n, self.d, generator=gen); e = torch.randn(n, self.d, generator=gen)
        s = torch.where(torch.rand(n, self.d, generator=gen) < 0.5, 1.0, -1.0)
        return x, e, s

    def from_base(self, base, param, device=None):
        x, e, s = base
        y = s * param * x + torch.sqrt(1.0 - param * param) * e if torch.is_tensor(param) else s * param * x + math.sqrt(1.0 - param * param) * e
        return x.to(device), y.to(device)

    def pmi(self, x, y, param=None):
        c = self.param if param is None else param; a = 1.0 - c * c
        # log[½ N(Σ₊) + ½ N(Σ₋)] − log N(x) − log N(y), per pair, summed
        q_plus = (x * x - 2 * c * x * y + y * y) / (2 * a); q_minus = (x * x + 2 * c * x * y + y * y) / (2 * a)
        log_joint = -math.log(2 * math.pi) - 0.5 * torch.log(a * torch.ones((), dtype=x.dtype, device=x.device)) + torch.logaddexp(-q_plus, -q_minus) - math.log(2.0)
        log_marg = -math.log(2 * math.pi) - 0.5 * (x * x + y * y)
        return (log_joint - log_marg).sum(-1)


def mi_xor_pair(c: float, grid: int = 801, lim: float = 7.0) -> float:
    """MI of one 2-d mixture pair by quadrature on a grid (nats)."""
    t = np.linspace(-lim, lim, grid); X, Y = np.meshgrid(t, t, indexing="ij"); a = 1.0 - c * c
    def n2(rho):
        return np.exp(-(X * X - 2 * rho * X * Y + Y * Y) / (2 * (1 - rho * rho))) / (2 * np.pi * np.sqrt(1 - rho * rho))
    p = 0.5 * n2(c) + 0.5 * n2(-c); px = np.exp(-X * X / 2) / np.sqrt(2 * np.pi); py = np.exp(-Y * Y / 2) / np.sqrt(2 * np.pi)
    w = (t[1] - t[0]) ** 2; m = p > 1e-300
    return float((p[m] * (np.log(p[m]) - np.log(px[m]) - np.log(py[m]))).sum() * w)


def c_from_mi_xor(mi_per_pair: float) -> float:
    lo, hi = 0.0, 0.999999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mi_xor_pair(mid) < mi_per_pair: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def setting_from_mi(name: str, mi: float, d: int = 20, level: int = -1) -> Setting:
    """A setting at an arbitrary total MI (nats).  xor_mixture ignores d (XOR_PAIRS pairs, x and y are XOR_PAIRS-dimensional)."""
    if name == "gaussian":
        return Gaussian(name, d, level, rho_from_mi(mi, d), mi)
    if name == "cubic":
        return Cubic(name, d, level, rho_from_mi(mi, d), mi)
    if name == "xor_mixture":
        c = c_from_mi_xor(mi / XOR_PAIRS)
        return XorMixture(name, XOR_PAIRS, level, c, XOR_PAIRS * mi_xor_pair(c))
    raise ValueError(name)


def make_setting(name: str, level: int, d: int = 20) -> Setting:
    return setting_from_mi(name, LEVELS_NATS[level], d, level)


def truths_by_mc(setting: Setting, n: int = 400_000, seed: int = 123, device=None) -> dict:
    """Own-target truths from the analytic PMI: S = E_M[tanh(pmi/2)²]; JS = E_P[log σ(pmi)] + E_Q[log(1 − σ(pmi))] + log 4 (= 2·JSD);
    MI analytic (Gaussian) or quadrature (mixture); E_Q by a shuffled partner (independent product sample)."""
    gen = torch.Generator().manual_seed(seed); x, y = setting.sample(n, gen, device)
    perm = torch.randperm(n, generator=gen).to(x.device)
    pp, pq = setting.pmi(x, y).double(), setting.pmi(x, y[perm]).double()
    tp, tq = torch.tanh(pp / 2), torch.tanh(pq / 2)
    S = 0.5 * (tp * tp).mean() + 0.5 * (tq * tq).mean()
    js = torch.nn.functional.logsigmoid(pp).mean() + torch.nn.functional.logsigmoid(-pq).mean() + math.log(4.0)
    return {"S": float(S), "JS2": float(js), "MI": float(setting.mi)}


def pad_sides(x: torch.Tensor, y: torch.Tensor, k: int, gen: torch.Generator):
    """Irrelevant-dimension design (spec §5.1): append k independent N(0, 1) coordinates to each side (own stream per call); the padded
    coordinates cancel in p/q, so S(X; Y) = S(X_s; Y_s) and every PMI is evaluated on the leading signal coordinates only."""
    if k == 0:
        return x, y
    return torch.cat([x, torch.randn(len(x), k, generator=gen, dtype=x.dtype)], 1), torch.cat([y, torch.randn(len(y), k, generator=gen, dtype=y.dtype)], 1)


# ------------------------------------------------------------------------------------------------------------ R1 contaminations
def contaminate(x: torch.Tensor, y: torch.Tensor, eps: float, kind: str, gen: torch.Generator, high: Setting | None = None):
    """Replace a fraction eps of the joint pairs.  'independent': y from another sample; 'outlier': x shifted by +5 (5σ) and y from the
    highest-MI step (`high`, its own x discarded); 'heavy': y += Student-t(2) noise.  Returns (x, y, mask of contaminated rows)."""
    n = len(x); k = int(round(eps * n)); mask = torch.zeros(n, dtype=torch.bool, device=x.device)
    if k == 0:
        return x, y, mask
    idx = torch.randperm(n, generator=gen)[:k].to(x.device); mask[idx] = True
    x, y = x.clone(), y.clone()
    if kind == "independent":
        src = torch.randperm(n, generator=gen)[:k].to(x.device); y[idx] = y[src]
    elif kind == "outlier":
        x[idx] = x[idx] + 5.0
        hx, hy = high.sample(k, gen, x.device); y[idx] = hy
    elif kind == "heavy":
        z = torch.randn(k, x.shape[1], generator=gen); u = torch.rand(k, 1, generator=gen).clamp_min(1e-6)
        # t_2 noise: z / sqrt(chi2_2 / 2) with chi2_2 = -2 log(u)
        y[idx] = y[idx] + (z / torch.sqrt(-torch.log(u))).to(x.device)
    else:
        raise ValueError(kind)
    return x, y, mask
