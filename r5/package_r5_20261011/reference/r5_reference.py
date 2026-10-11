"""VCS r5: loss / diagnostic reference, not a server training runner.

All new operations use explicit real-region masks. P/Q are per-image laws;
zero-mass query columns are padding or excluded from the matching objective.
Task metrics must still count uncovered deployment queries as errors.
"""
from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

OBJECTIVES = ("vcs", "balanced_logistic", "distribution_softmax", "set_softmax")


def make_laws(
    compatibility: Tensor, real_regions: Tensor, reference_weights: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Build [B,R,W] P/Q. Each image must have at least one covered query.

    P is uniform within the valid set for each covered query. Queries are
    uniform within image. Q has full support on the explicit real regions.
    """
    if compatibility.ndim != 3 or real_regions.shape != compatibility.shape[:2]:
        raise ValueError("Expected compatibility [B,R,W], real_regions [B,R].")
    if real_regions.dtype != torch.bool:
        raise ValueError("real_regions must be boolean.")
    a = compatibility.to(torch.float64) if not compatibility.is_floating_point() else compatibility
    if not torch.isfinite(a).all() or not ((a == 0) | (a == 1)).all():
        raise ValueError("Compatibility must be a finite binary matrix.")
    if (a.masked_select(~real_regions[:, :, None].expand_as(a)) != 0).any():
        raise ValueError("Padded regions cannot be valid.")
    if not real_regions.any(1).all():
        raise ValueError("Images without proposals must be handled outside training.")
    k = a.sum(1)
    covered = k > 0
    if not covered.any(1).all():
        raise ValueError("Exclude and count images with no covered training query.")
    pi = covered.to(a.dtype) / covered.sum(1, keepdim=True)
    p = a / k.clamp_min(1)[:, None, :] * pi[:, None, :]
    if reference_weights is None:
        w = real_regions.to(a.dtype)
    else:
        if reference_weights.shape != real_regions.shape:
            raise ValueError("Reference weights must have shape [B,R].")
        if not torch.isfinite(reference_weights[real_regions]).all() or not (reference_weights[real_regions] > 0).all():
            raise ValueError("Every real candidate needs a positive finite reference weight.")
        w = torch.where(real_regions, reference_weights.to(a.dtype), torch.zeros_like(a[:, :, 0]))
    nu = w / w.sum(1, keepdim=True)
    q = nu[:, :, None] * pi[:, None, :]
    return p, q


def _validate(f: Tensor, p: Tensor, q: Tensor, real_regions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if f.ndim != 3 or f.shape != p.shape or p.shape != q.shape:
        raise ValueError("f/P/Q must have the same [B,R,W] shape.")
    if f.shape[0] == 0 or min(f.shape[1:]) == 0:
        raise ValueError("Empty batches or padded dimensions are not allowed.")
    if not f.is_floating_point() or f.dtype not in (torch.float32, torch.float64):
        raise ValueError("Reference expects float32/float64; cast AMP logits before loss.")
    if p.dtype != f.dtype or q.dtype != f.dtype:
        raise ValueError("Use common float dtype for f/P/Q.")
    if real_regions.dtype != torch.bool or real_regions.shape != f.shape[:2]:
        raise ValueError("Explicit boolean real-region mask required.")
    if not real_regions.any(1).all():
        raise ValueError("No real proposal in an image.")
    for law in (p, q):
        if not torch.isfinite(law).all() or (law < 0).any():
            raise ValueError("P/Q must be finite and nonnegative.")
        if not torch.allclose(law.sum((1, 2)), torch.ones_like(law[:, 0, 0]), atol=2e-6, rtol=2e-6):
            raise ValueError("P/Q must have unit mass per image.")
        if (law.masked_select(~real_regions[:, :, None].expand_as(law)) != 0).any():
            raise ValueError("P/Q assign mass to a padded region.")
    pi = p.sum(1)
    if not torch.allclose(pi, q.sum(1), atol=2e-6, rtol=2e-6):
        raise ValueError("Proposal-reference P/Q must share the query marginal.")
    active = pi > 0
    real_active = real_regions[:, :, None] & active[:, None, :]
    if not (q[real_active] > 0).all():
        raise ValueError("Reference must have full real-region support on active queries.")
    if not torch.isfinite(f[real_active]).all():
        raise ValueError("Active logits must be finite.")
    safe_f = torch.where(real_active, f, torch.zeros_like(f))
    return safe_f, pi, active


def _probabilities(f: Tensor, p: Tensor, real_regions: Tensor, active: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stable all-candidate and legal-set log-probabilities; empty queries safe."""
    real = real_regions[:, :, None].expand_as(f)
    u = 2 * f
    u_all = u.masked_fill(~real, -torch.inf)
    logz = torch.logsumexp(u_all, dim=1)
    legal_safe = (p > 0) | (real & ~active[:, None, :])
    u_pos = u.masked_fill(~legal_safe, -torch.inf)
    logz_pos = torch.logsumexp(u_pos, dim=1)
    log_prob = torch.where(real, u - logz[:, None, :], torch.zeros_like(u))
    prob = torch.where(real, torch.exp(log_prob), torch.zeros_like(u))
    legal_prob = torch.exp(u_pos - logz_pos[:, None, :])
    return log_prob, prob, legal_prob, logz - logz_pos


def per_image_loss(f: Tensor, p: Tensor, q: Tensor, real_regions: Tensor, objective: str) -> Tensor:
    """One normalized loss per image. Mean this vector once per full image batch."""
    sf, pi, active = _validate(f, p, q, real_regions)
    if objective == "vcs":
        t = torch.tanh(sf)
        return .5 * (p * (1 - t).square() + q * (1 + t).square()).sum((1, 2))
    if objective == "balanced_logistic":
        return (p * F.softplus(-2 * sf) + q * F.softplus(2 * sf)).sum((1, 2))
    log_prob, _, _, set_loss = _probabilities(sf, p, real_regions, active)
    if objective == "distribution_softmax":
        return -(p * log_prob).sum((1, 2))
    if objective == "set_softmax":
        return (pi * set_loss).sum(1)
    raise ValueError(f"Unknown objective {objective!r}; allowed: {OBJECTIVES}")


def batch_loss(f: Tensor, p: Tensor, q: Tensor, real_regions: Tensor, objective: str) -> Tensor:
    return per_image_loss(f, p, q, real_regions, objective).mean()


def j_per_image(f: Tensor, p: Tensor, q: Tensor, real_regions: Tensor) -> Tensor:
    sf, _, _ = _validate(f, p, q, real_regions)
    t = torch.tanh(sf)
    return (p * (t - .5 * t.square()) + q * (-t - .5 * t.square())).sum((1, 2))


def analytic_f_gradient(f: Tensor, p: Tensor, q: Tensor, real_regions: Tensor, objective: str) -> Tensor:
    """Derivative of SUM of per-image losses, before the final batch mean.

    Multiply by a for cosine derivatives. This is not an encoder gradient.
    """
    sf, pi, active = _validate(f, p, q, real_regions)
    t = torch.tanh(sf)
    if objective == "vcs":
        g = (p * (t - 1) + q * (t + 1)) * (1 - t.square())
    elif objective == "balanced_logistic":
        g = p * (t - 1) + q * (t + 1)
    else:
        _, prob, legal_prob, _ = _probabilities(sf, p, real_regions, active)
        if objective == "distribution_softmax":
            g = 2 * (pi[:, None, :] * prob - p)
        elif objective == "set_softmax":
            g = 2 * pi[:, None, :] * (prob - legal_prob)
        else:
            raise ValueError("Unknown objective.")
    return torch.where(real_regions[:, :, None] & active[:, None, :], g, torch.zeros_like(g))


def uniform_set_decomposition(f: Tensor, p: Tensor, q: Tensor, real_regions: Tensor) -> dict[str, Tensor]:
    """D2 per query and per image. Requires uniform P within each legal set."""
    sf, pi, active = _validate(f, p, q, real_regions)
    legal = p > 0
    k = legal.sum(1)
    expected = legal.to(p.dtype) / k.clamp_min(1)[:, None, :] * pi[:, None, :]
    if not torch.allclose(p, expected, atol=2e-6, rtol=2e-6):
        raise ValueError("D2 uniform version is not valid for nonuniform positive targets.")
    u = 2 * sf
    safe_legal = legal | (~active[:, None, :] & real_regions[:, :, None])
    logza = torch.logsumexp(u.masked_fill(~safe_legal, -torch.inf), dim=1)
    mean_legal_u = (u * legal).sum(1) / k.clamp_min(1)
    logk = k.clamp_min(1).to(f.dtype).log()
    kl = torch.where(active, logza - mean_legal_u - logk, torch.zeros_like(pi))
    return {"k": k, "log_k": logk, "kl_uniform_to_legal": kl,
            "per_image_extra": (pi * (logk + kl)).sum(1)}


def corrected_scores(f: Tensor, nu: Tensor, real_regions: Tensor) -> Tensor:
    """2f+log nu on real candidates; padded candidates get -inf."""
    if f.ndim != 3 or nu.shape != f.shape[:2] or real_regions.shape != nu.shape:
        raise ValueError("Expected f[B,R,W], nu/mask[B,R].")
    if not torch.isfinite(nu[real_regions]).all() or not (nu[real_regions] > 0).all():
        raise ValueError("Positive finite reference weights required.")
    w = torch.where(real_regions, nu, torch.zeros_like(nu))
    w = w / w.sum(1, keepdim=True)
    logw = torch.log(torch.where(real_regions, w, torch.ones_like(w)))
    return (2 * f + logw[:, :, None]).masked_fill(~real_regions[:, :, None], -torch.inf)


def query_attention(f: Tensor, p: Tensor, q: Tensor, rank_scores: Tensor,
                    objective: str, top_k: int, alpha: float = 1.0) -> dict[str, float | int | None]:
    """One covered query, conditional p/q sums=1. Only invalid candidates selected.

    Pass self deployment scores or scores from a fixed shared reference order.
    None marks an undefined gradient fraction; never silently replace it by zero.
    """
    if f.ndim != 1 or not (p.shape == q.shape == rank_scores.shape == f.shape):
        raise ValueError("One-dimensional aligned arrays required.")
    if top_k < 1 or alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("Invalid top_k or affine gain.")
    mask = torch.ones((1, len(f)), dtype=torch.bool, device=f.device)
    ff, pp, qq = [x[None, :, None] for x in (f, p, q)]
    g = analytic_f_gradient(ff, pp, qq, mask, objective)[0, :, 0]
    invalid = p == 0
    ids = torch.nonzero(invalid, as_tuple=False).flatten()
    if len(ids) == 0:
        return {"invalid_count": 0, "selected_count": 0, "q_mass": 0.0,
                "q_fraction_invalid": None, "gradient_fraction_invalid": None,
                "abs_f_gradient": 0.0, "abs_cosine_gradient": 0.0}
    if not torch.isfinite(rank_scores[ids]).all():
        raise ValueError("Finite ranking scores required.")
    order = torch.argsort(rank_scores[ids], descending=True, stable=True)
    chosen = ids[order[:top_k]]
    qm, qtot = float(q[chosen].sum()), float(q[ids].sum())
    gm, gtot = float(g[chosen].abs().sum()), float(g[ids].abs().sum())
    return {"invalid_count": len(ids), "selected_count": len(chosen), "q_mass": qm,
            "q_fraction_invalid": qm/qtot if qtot > 0 else None,
            "gradient_fraction_invalid": gm/gtot if gtot > 0 else None,
            "abs_f_gradient": gm, "abs_cosine_gradient": alpha * gm}


def nested_invalid_ids(query_id: str, invalid_ids: Sequence[str], repeat: int,
                       counts: Sequence[int] = (1, 3, 7, 15, 31)) -> dict[str, list[str]]:
    """Stable shared, nested candidate subsets. Strict common support required."""
    ids = sorted(str(x) for x in invalid_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Proposal IDs must be unique within a query.")
    if not counts or any(c < 1 for c in counts) or repeat < 0:
        raise ValueError("Invalid counts/repeat.")
    if len(ids) < max(counts):
        raise ValueError("Do not silently replace a strict count by all available candidates.")
    key = f"R5-nested-v1|{query_id}|{repeat}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    order = np.asarray(ids, dtype=object)[rng.permutation(len(ids))].tolist()
    return {**{str(c): order[:c] for c in counts}, "all": order}


def choose_cal_lambda(cal_values: Mapping[float, float]) -> float:
    """The only inputs are CAL image-macro values; smallest lambda breaks ties."""
    if not cal_values or any(not np.isfinite(k) or not np.isfinite(v) for k, v in cal_values.items()):
        raise ValueError("Nonempty finite CAL grid required.")
    best = max(cal_values.values())
    return min(k for k, v in cal_values.items() if v == best)


def image_macro_hit(records: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    """records=(scores[R,W], compatibility[R,W]); uncovered queries count as errors.

    Uses every query, including those with no valid proposal. An image with zero
    proposals but nonzero queries has accuracy zero. Candidate ties use input order.
    """
    if not records:
        raise ValueError("No images.")
    means = []
    for score, a in records:
        score, a = np.asarray(score), np.asarray(a)
        if score.ndim != 2 or score.shape != a.shape or score.shape[1] == 0:
            raise ValueError("Each image needs aligned candidates and at least one query.")
        if not np.isfinite(score).all() or not np.isin(a, [0, 1]).all():
            raise ValueError("Finite scores and binary compatibility required.")
        if score.shape[0] == 0:
            means.append(0.0)
        else:
            pred = score.argmax(0)
            means.append(float(a[pred, np.arange(score.shape[1])].mean()))
    return float(np.mean(means))


def single_positive_counterexample(m: int = 50, epsilon: float = .001) -> dict[str, float | int | bool]:
    """Finite illustration only, not a claim about natural images."""
    if m < 2 or not 0 < epsilon < 2 / (m + 1):
        raise ValueError("Choose m>=2 and 0<epsilon<2/(m+1).")
    p = np.zeros(m); p[0] = 1
    q = np.ones(m)/m
    eta = (p-q)/(p+q)
    t = eta.copy(); t[1] = eta[0]+epsilon
    S = float(np.sum(.5*(p+q)*eta**2))
    J = float(np.sum(p*(t-.5*t*t)+q*(-t-.5*t*t)))
    return {"m":m, "S":S, "J":J, "gap":S-J, "wrong_top1":bool(t.argmax()!=0),
            "gap_closed_form":float((t[1]+1)**2/(2*m))}
