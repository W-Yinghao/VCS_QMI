"""Estimator package v1 (§4.2, §6.1): Gaussian generator with its posterior oracle, and role-separated datasets.

X ~ N(0, I_d),  Y = rho X + sqrt(1 - rho^2) E,  rho = sqrt(1 - exp(-2 I / d))  (I = generator scale in nats, not the VCS truth).
Q = independent standard Gaussians on both sides.  log ratio l_rho(x, y) and eta = tanh(l / 2).
Every role (FIT / TUNE / SELECT / EVAL / TRUTH) and every side (P joint, Q product) draws from its own seeded stream.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

ROLES = ("FIT", "TUNE", "SELECT", "EVAL", "TRUTH")
DEFAULT_SIZES = {"FIT": 4096, "TUNE": 1024, "SELECT": 1024, "EVAL": 32768, "TRUTH": 300000}   # per distribution (P and Q each)


def rho_from_I(I: float, d: int) -> float:
    return math.sqrt(1.0 - math.exp(-2.0 * I / d))


def log_ratio(x: torch.Tensor, y: torch.Tensor, rho) -> torch.Tensor:
    d = x.shape[-1]; r2 = rho * rho
    return -0.5 * d * torch.log1p(-r2 * torch.ones((), dtype=x.dtype, device=x.device)) + \
        (2 * rho * (x * y).sum(-1) - r2 * ((x * x).sum(-1) + (y * y).sum(-1))) / (2 * (1 - r2))


def eta(x, y, rho):
    return torch.tanh(0.5 * log_ratio(x, y, rho))


def _gen(seed_parts) -> torch.Generator:
    h = int(hashlib.sha256(repr(tuple(seed_parts)).encode()).hexdigest()[:15], 16)
    return torch.Generator().manual_seed(h)


def sample_p(n, d, rho, gen, dtype=torch.float64):
    x = torch.randn(n, d, generator=gen, dtype=dtype); e = torch.randn(n, d, generator=gen, dtype=dtype)
    return x, rho * x + math.sqrt(1 - rho * rho) * e, e


def sample_q(n, d, gen, dtype=torch.float64):
    return torch.randn(n, d, generator=gen, dtype=dtype), torch.randn(n, d, generator=gen, dtype=dtype)


@dataclass
class RoleData:
    xp: torch.Tensor; yp: torch.Tensor; xq: torch.Tensor; yq: torch.Tensor; ep: torch.Tensor   # ep = base noise of the P side (for d/d rho)


def make_gaussian(I: float, d: int = 20, seed: int = 0, sizes: dict | None = None) -> tuple[float, dict]:
    sizes = {**DEFAULT_SIZES, **(sizes or {})}; rho = rho_from_I(I, d); out = {}
    for role in ROLES:
        xp, yp, ep = sample_p(sizes[role], d, rho, _gen(("P", role, I, d, seed)))
        xq, yq = sample_q(sizes[role], d, _gen(("Q", role, I, d, seed)))
        out[role] = RoleData(xp, yp, xq, yq, ep)
    return rho, out


def truth(I: float, d: int, rho: float, T: RoleData) -> dict:
    """S = E_M eta^2 = 1/2 E_P eta^2 + 1/2 E_Q eta^2 and the oracle J(eta) on the TRUTH pairs (MC consistency: J(eta) -> S)."""
    ep, eq = eta(T.xp, T.yp, rho), eta(T.xq, T.yq, rho)
    s_terms_p, s_terms_q = ep ** 2, eq ** 2
    S = 0.5 * s_terms_p.mean() + 0.5 * s_terms_q.mean()
    se = math.sqrt(0.25 * float(s_terms_p.var()) / len(ep) + 0.25 * float(s_terms_q.var()) / len(eq))
    J = (ep - 0.5 * ep ** 2).mean() + (-eq - 0.5 * eq ** 2).mean()
    jp, jq = ep - 0.5 * ep ** 2, -eq - 0.5 * eq ** 2
    Jse = math.sqrt(float(jp.var()) / len(ep) + float(jq.var()) / len(eq))
    return {"I_generator": I, "d": d, "rho": rho, "S_truth": float(S), "S_truth_se": se, "J_oracle_truth": float(J), "J_oracle_truth_se": Jse,
            "n_truth_per_distribution": len(ep)}


def roles_hash(data: dict) -> str:
    h = hashlib.sha256()
    for role in ROLES:
        r = data[role]
        for t in (r.xp, r.yp, r.xq, r.yq):
            h.update(t[:64].numpy().tobytes()); h.update(str(tuple(t.shape)).encode())
    return h.hexdigest()
