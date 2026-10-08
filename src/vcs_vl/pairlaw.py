"""VL1 pair laws (port of the VCS-VL-SERVER-v1-20261008 package reference `reference/vl_pair_reference.py`, unchanged semantics), plus
the batched per-image loss used by the VL1 fitters.  Finite-law utilities only: they do not provide an oracle for natural images."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F


def _law(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError('Expected a non-empty 2-D probability table.')
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('Probability entries must be finite and nonnegative.')
    if not np.isclose(x.sum(), 1., atol=1e-10, rtol=1e-10):
        raise ValueError('Probability table must sum to one.')
    return x


def pair_laws(compatibility: np.ndarray, region_prior: np.ndarray | None = None):
    """Uniform eligible regions by default; aliases normalized within region.

    Do not remove positive-compatible cells from Q.
    """
    a = np.asarray(compatibility, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Invalid compatibility table.')
    if np.any(a.sum(1) == 0) or np.any(a.sum(0) == 0):
        raise ValueError('Remove/record empty regions or phrases explicitly first.')
    m = len(a)
    prior = np.ones(m) / m if region_prior is None else np.asarray(region_prior, dtype=np.float64)
    if prior.shape != (m,) or not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError('Region prior must be finite, positive and match rows.')
    if not np.isclose(prior.sum(), 1., atol=1e-10, rtol=1e-10):
        raise ValueError('Region prior must sum to one.')
    p = prior[:, None] * a / a.sum(1, keepdims=True)
    q = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    return _law(p), _law(q)


def eta_and_s(p: np.ndarray, q: np.ndarray):
    p, q = _law(p), _law(q)
    if p.shape != q.shape:
        raise ValueError('P and Q must have the same support table.')
    den = p + q
    eta = np.divide(p - q, den, out=np.zeros_like(p), where=den > 0)
    s = float(np.sum(0.5 * den * eta ** 2))
    return eta, s


def j_readout(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> float:
    p, q = _law(p), _law(q)
    t = np.asarray(t, dtype=np.float64)
    if p.shape != q.shape or p.shape != t.shape or not np.isfinite(t).all():
        raise ValueError('Mismatched or nonfinite score table.')
    if np.max(np.abs(t)) > 1 + 1e-12:
        raise ValueError('This reference requires a bounded critic.')
    return float(np.sum(p * (t - 0.5*t*t) + q * (-t - 0.5*t*t)))


def diagonal_channel(m: int, rho: float):
    if not isinstance(m, int) or m < 2 or not 0 <= rho <= 1:
        raise ValueError('m>=2 and rho in [0,1] required.')
    q = np.ones((m, m), dtype=np.float64) / (m*m)
    p = rho * np.eye(m) / m + (1-rho)*q
    analytic_s = rho*rho*(m-1) / ((2+rho*(m-1))*(2-rho))
    return p, q, analytic_s


def pushforward(p: np.ndarray, q: np.ndarray, observation_ids: np.ndarray):
    """Merge states the actual observed input cannot distinguish."""
    p, q = _law(p), _law(q)
    ids = np.asarray(observation_ids)
    if p.shape != q.shape or ids.shape != p.shape:
        raise ValueError('Observation map must match the law shape.')
    _, inv = np.unique(ids.ravel(), return_inverse=True)
    pp = np.bincount(inv, weights=p.ravel()).reshape(-1, 1)
    qq = np.bincount(inv, weights=q.ravel()).reshape(-1, 1)
    return _law(pp), _law(qq)


def posterior_rank_score(half_log_ratio: np.ndarray, region_prior: np.ndarray):
    f = np.asarray(half_log_ratio, dtype=np.float64)
    prior = np.asarray(region_prior, dtype=np.float64)
    if f.ndim != 2 or prior.shape != (f.shape[0],) or np.any(prior <= 0):
        raise ValueError('Invalid logits or prior.')
    if not np.isfinite(f).all() or not np.isfinite(prior).all() or not np.isclose(prior.sum(),1):
        raise ValueError('Use finite logits and a normalized positive prior.')
    return 2*f + np.log(prior[:, None])


def probability_to_common_t(raw_logit: torch.Tensor, positive_prior: float = .5):
    if not 0 < positive_prior < 1:
        raise ValueError('positive_prior must be strictly between zero and one.')
    prior_logit = float(np.log(positive_prior/(1-positive_prior)))
    return torch.tanh((raw_logit-prior_logit)/2)


def image_losses(f: torch.Tensor, p: np.ndarray, q: np.ndarray, objective: str):
    """Each scene is normalized before averaging scenes of different sizes."""
    p, q = _law(p), _law(q)
    if tuple(f.shape) != p.shape or p.shape != q.shape:
        raise ValueError('Score and law shapes differ.')
    pt = torch.as_tensor(p, dtype=f.dtype, device=f.device)
    qt = torch.as_tensor(q, dtype=f.dtype, device=f.device)
    if objective == 'vcs':
        t = torch.tanh(f)
        return .5 * ((pt*(1-t).square()).sum() + (qt*(1+t).square()).sum())
    if objective == 'balanced_logistic':
        return (pt*F.softplus(-2*f)).sum() + (qt*F.softplus(2*f)).sum()
    if objective == 'conditional_softmax':
        # Region distribution conditional on each phrase; weighted by phrase marginal.
        # Softmax scores are not automatically calibrated density-ratio logits.
        return -(pt * F.log_softmax(2*f, dim=0)).sum()
    raise ValueError(f'Unsupported objective: {objective}')


def batch_loss(scores: Sequence[torch.Tensor], laws, objective: str):
    if not scores or len(scores) != len(laws):
        raise ValueError('Need one law per non-empty scene.')
    return torch.stack([image_losses(f, p, q, objective)
                        for f, (p, q) in zip(scores, laws)]).mean()


def assert_image_disjoint(role_to_keys: dict[str, Sequence[str]]):
    """Keys must be canonical source-image identities, not dataset-local aliases."""
    owner: dict[str, str] = {}
    for role, keys in role_to_keys.items():
        for key in set(keys):
            if key in owner and owner[key] != role:
                raise ValueError(f'Image leakage: {key}: {owner[key]} vs {role}')
            owner[key] = role


def padded_batch_loss(f: torch.Tensor, p: torch.Tensor, q: torch.Tensor, objective: str) -> torch.Tensor:
    """Vectorised `batch_loss` for a batch of scenes padded to [n, R, W]: p / q are each scene's laws with zeros in the padding (so padding
    contributes nothing), every scene's p and q sum to one, and scenes are averaged with equal weight (one total weight per image)."""
    if f.shape != p.shape or p.shape != q.shape or f.ndim != 3:
        raise ValueError("expected matching [n, R, W] tensors")
    if objective == "vcs":
        t = torch.tanh(f)
        per = 0.5 * ((p * (1 - t).square()).sum((1, 2)) + (q * (1 + t).square()).sum((1, 2)))
    elif objective == "balanced_logistic":
        per = (p * F.softplus(-2 * f)).sum((1, 2)) + (q * F.softplus(2 * f)).sum((1, 2))
    elif objective == "conditional_softmax":
        valid = p.sum(2, keepdim=True).expand_as(p) > 0  # padded regions excluded from the softmax over regions
        lsm = torch.log_softmax((2 * f).masked_fill(~valid, -torch.inf), dim=1)
        per = -(p * lsm.masked_fill(~valid, 0.0)).sum((1, 2))
    else:
        raise ValueError(f"Unsupported objective: {objective}")
    return per.mean()


def padded_j(f: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-scene common J (T = tanh f) for padded laws: returns [n]."""
    t = torch.tanh(f)
    return (p * (t - 0.5 * t * t)).sum((1, 2)) + (q * (-t - 0.5 * t * t)).sum((1, 2))
