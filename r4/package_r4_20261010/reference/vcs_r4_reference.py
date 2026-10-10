"""VCS r4: independent mathematical/record-reading references, not a trainer.

All concentration functions require the critic and representation to be fixed
outside evaluation.  Simulation checks do not prove novelty or model quality.
Dependencies: Python 3.10+, NumPy.  Optional Torch is used only by a test.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np


def _arr(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError('Expected finite numbers.')
    return a


def probability(x: Any) -> np.ndarray:
    a = _arr(x)
    if a.size == 0 or np.any(a < 0) or not np.isclose(a.sum(), 1., atol=1e-10):
        raise ValueError('Probability weights must be nonnegative and sum to one.')
    return a


def pq(p: Any, q: Any) -> tuple[np.ndarray, np.ndarray]:
    p, q = probability(p), probability(q)
    if p.shape != q.shape:
        raise ValueError('P and Q must have the same support shape.')
    return p, q


def oracle(p: Any, q: Any) -> dict[str, Any]:
    p, q = pq(p, q)
    m = (p + q) / 2
    eta = np.divide(p-q, p+q, out=np.zeros_like(p), where=(p+q) > 0)
    return {'M': m, 'eta': eta, 'S': float(np.sum(m * eta**2))}


def common_j(t: Any, p: Any, q: Any, *, check_bound: bool = True) -> float:
    t = _arr(t); p, q = pq(p, q)
    if t.shape != p.shape:
        raise ValueError('Score shape must match the distributions.')
    if check_bound and np.any(np.abs(t) > 1 + 1e-12):
        raise ValueError('Bounded-J implementation requires |T| <= 1.')
    return float(np.sum(p*(t-.5*t*t) + q*(-t-.5*t*t)))


def rpc(t: Any, p: Any, q: Any, alpha: float = 1., beta: float = 1., gamma: float = 1.) -> float:
    t = _arr(t); p, q = pq(p, q)
    if t.shape != p.shape or min(alpha, beta, gamma) <= 0:
        raise ValueError('Matching shape and positive RPC parameters required.')
    return float(np.sum(p*t-alpha*q*t-.5*(beta*p+gamma*q)*t*t))


def rpc_oracle(p: Any, q: Any, alpha: float, beta: float, gamma: float) -> np.ndarray:
    p, q = pq(p, q)
    if min(alpha, beta, gamma) <= 0:
        raise ValueError('Use positive RPC parameters.')
    den = beta*p+gamma*q
    return np.divide(p-alpha*q, den, out=np.zeros_like(p), where=den > 0)


def mixture_cs(s: float) -> float:
    if not 0 <= s < 1:
        raise ValueError('CS transform requires 0 <= S < 1; no silent clipping.')
    return 2*(math.log1p(s)-math.log1p(-s))


def contaminate_eta(eta: Any, rho: float) -> np.ndarray:
    e = _arr(eta)
    if not 0 <= rho < 1 or np.any(np.abs(e) > 1):
        raise ValueError('Require |eta| <= 1 and 0 <= rho < 1.')
    return (1-rho)*e/(1-rho*e)


def recover_eta(observed: Any, rho: float) -> np.ndarray:
    u = _arr(observed)
    if not 0 <= rho < 1:
        raise ValueError('Require 0 <= rho < 1.')
    lo = -(1-rho)/(1+rho)
    if np.any(u < lo-1e-12) or np.any(u > 1+1e-12):
        raise ValueError('Outside the oracle contamination range; do not silently clip.')
    return u/(1-rho+rho*u)


def cross_pair_kernel(t: Any) -> np.ndarray:
    """t[i,j] = T(X_i,Y_j). Returns symmetric k for fixed-critic U-statistic.

    The diagonal is retained for mathematical enumeration; U-averages exclude it.
    """
    t = _arr(t)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or len(t) < 2 or np.any(np.abs(t) > 1+1e-12):
        raise ValueError('Expected n x n bounded scores with n >= 2.')
    a = np.diag(t)-.5*np.diag(t)**2
    b = -t-.5*t*t
    return .5*(a[:, None]+a[None, :])+.5*(b+b.T)


def u_mean(k: Any) -> float:
    k = _arr(k)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or len(k) < 2 or not np.allclose(k, k.T):
        raise ValueError('Expected symmetric square kernel and n >= 2.')
    n = len(k)
    return float((k.sum()-np.trace(k))/(n*(n-1)))


def direct_cross_j(t: Any) -> float:
    t = _arr(t)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or len(t) < 2:
        raise ValueError('Expected square scores and n >= 2.')
    a = np.diag(t)-.5*np.diag(t)**2
    b = -t-.5*t*t
    return float(a.mean()+(b.sum()-np.trace(b))/(len(t)*(len(t)-1)))


def finite_u_variance(k_states: Any, state_probs: Any, n: int) -> dict[str, float]:
    """Exact fixed-kernel variance for iid finite-state W, not fitted data."""
    k = _arr(k_states); p = probability(state_probs)
    if p.ndim != 1 or k.shape != (len(p), len(p)) or not np.allclose(k,k.T) or n < 2:
        raise ValueError('Kernel/state dimensions or n invalid.')
    mu = float(p@k@p)
    h1 = k@p-mu
    h2 = k-mu-h1[:,None]-h1[None,:]
    v1 = float(p@(h1*h1)); v2 = float(np.sum(p[:,None]*p[None,:]*h2*h2))
    return {'mean':mu,'first_projection_variance':v1,'degenerate_variance':v2,
            'variance':4*v1/n+2*v2/(n*(n-1))}


def delete_one_variance(k: Any) -> dict[str, Any]:
    k = _arr(k); u = u_mean(k); n = len(k)
    if n < 3:
        raise ValueError('Delete-one variance requires n >= 3.')
    total_edges = (k.sum()-np.trace(k))/2
    incident = k.sum(axis=1)-np.diag(k)
    loo = (total_edges-incident)/((n-1)*(n-2)/2)
    v = float((n-1)/n*np.sum((loo-loo.mean())**2))
    return {'mean':u,'leave_one_out':loo,'variance':v,'se':math.sqrt(max(v,0.)),
            'interpretation':'approximate_non_degenerate_fixed_critic_only'}


def bounded_radius(n: int, delta: float = .05, regime: str = 'image_local') -> float:
    """Conservative two-sided radius, not a two-sided CI for oracle S."""
    if n < 1 or not 0 < delta < 1:
        raise ValueError('Invalid n or delta.')
    if regime not in {'image_local','cross_pair_mcd'}:
        raise ValueError('Unknown independence regime.')
    constant = 8. if regime == 'image_local' else 32.
    return math.sqrt(constant*math.log(2/delta)/n)


def oracle_lcb(j_hat: float, radius: float) -> float:
    if not math.isfinite(j_hat) or radius < 0 or not math.isfinite(radius):
        raise ValueError('Invalid readout/radius.')
    return max(0., j_hat-radius)


def shift_sampling_variance(values: Any, k: int) -> float:
    v = _arr(values)
    if v.ndim != 1 or len(v) < 2 or not 1 <= k <= len(v):
        raise ValueError('Require a finite population >= 2 and 1 <= K <= M.')
    return float((1-k/len(v))*v.var(ddof=1)/k)


def group_macro(values: Any, group_ids: Any) -> float:
    v = _arr(values); g = np.asarray(group_ids)
    if v.ndim != 1 or g.shape != v.shape or len(v) == 0:
        raise ValueError('Need equal nonempty 1-D values/groups.')
    return float(np.mean([v[g == x].mean() for x in np.unique(g)]))


def group_bootstrap(values: Any, group_ids: Any, reps: int=1000, seed: int=0) -> np.ndarray:
    """Bootstrap a mean of within-image means; NOT cross-image product pairs."""
    v = _arr(values); g = np.asarray(group_ids)
    if v.ndim != 1 or g.shape != v.shape or len(v) == 0 or reps < 1:
        raise ValueError('Bad image-local bootstrap input.')
    means = np.array([v[g==x].mean() for x in np.unique(g)])
    rng = np.random.default_rng(seed)
    return means[rng.integers(0,len(means),(reps,len(means)))].mean(axis=1)


def proposal_laws(compatible: Any, proposal_prior: Any|None=None, query_prior: Any|None=None) -> tuple[np.ndarray,np.ndarray]:
    """Rows=proposals, cols=queries. Q_nu includes positives and unused regions.

    Queries without any valid proposal must be handled outside this closed-set
    training law. They may not be silently dropped from final evaluation.
    """
    a = _arr(compatible)
    if a.ndim != 2 or min(a.shape) < 1 or np.any(a < 0) or np.any(a.sum(axis=0) <= 0):
        raise ValueError('Each training query needs at least one compatible proposal.')
    m,q = a.shape
    nu = probability(np.ones(m)/m if proposal_prior is None else proposal_prior)
    pi = probability(np.ones(q)/q if query_prior is None else query_prior)
    if nu.shape != (m,) or pi.shape != (q,) or np.any(nu <= 0) or np.any(pi <= 0):
        raise ValueError('Full-support proposal/query priors required.')
    p = a/a.sum(axis=0,keepdims=True)*pi[None,:]
    reference = nu[:,None]*pi[None,:]
    return p, reference


def corrected_rank(logit_half_ratio: Any, proposal_prior: Any) -> np.ndarray:
    f=_arr(logit_half_ratio); nu=probability(proposal_prior)
    if f.shape != nu.shape or np.any(nu <= 0):
        raise ValueError('Positive prior matching score shape required.')
    return 2*f+np.log(nu)


def read_common_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Do not substitute native JS/KL readings when common posterior data is absent."""
    p = row.get('posterior') or {}
    j = p.get('J_eval')
    if j is None:
        return {'common_J': None, 'common_J_minus_S': None, 'posterior_mse': p.get('posterior_mse'),
                'status':'missing_common_readout','reason':'posterior.J_eval absent; native fallback forbidden'}
    j=float(j)
    if not math.isfinite(j):
        raise ValueError('Nonfinite common J.')
    r=p.get('R_eval')
    if r is not None and not math.isclose(float(r),1-j,rel_tol=1e-6,abs_tol=1e-8):
        raise ValueError('Same-evaluation R != 1-J; inspect schema/weights.')
    return {'common_J':j,'common_J_minus_S':p.get('J_eval_minus_S'),
            'posterior_mse':p.get('posterior_mse'),'status':'available'}


def matched_error_changes(clean_v: Any, noisy_v: Any, clean_l: Any, noisy_l: Any) -> dict[str, Any]:
    cv,nv,cl,nl=map(_arr,(clean_v,noisy_v,clean_l,noisy_l))
    if any(a.shape != cv.shape for a in (nv,cl,nl)) or cv.ndim != 1 or len(cv) == 0:
        raise ValueError('Paired one-dimensional seed arrays required.')
    dv,dl=nv-cv,nl-cl
    return {'absolute_difference':nv-nl,'vcs_degradation':dv,'js_degradation':dl,
            'degradation_difference':dv-dl,
            'target_note':'must be explicitly labelled clean or contaminated truth'}
