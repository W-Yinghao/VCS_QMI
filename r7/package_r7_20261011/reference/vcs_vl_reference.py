"""CPU reference for the r7 research definitions, not a production VLM runner.

Only finite, explicitly specified probability laws have an oracle S here.
The metric helpers are unit-test fixtures; official benchmark implementations
remain authoritative. Requires Python >=3.10 and NumPy.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import math
import numpy as np

Array = np.ndarray


def law(values: Sequence[float] | Array) -> Array:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('A law must be nonempty, finite and nonnegative.')
    if not np.isclose(a.sum(), 1., atol=1e-12, rtol=1e-10):
        raise ValueError('A law must sum to one; no silent renormalization.')
    return a


def eta_s(p: Array, q: Array) -> tuple[Array, float]:
    p, q = law(p), law(q)
    if p.shape != q.shape:
        raise ValueError('P and Q must have the same support shape.')
    den = p + q
    t = np.divide(p-q, den, out=np.zeros_like(p), where=den > 0)
    return t, float(np.sum(.5*den*t*t))


def j(p: Array, q: Array, t: Array) -> float:
    p, q = law(p), law(q)
    t = np.asarray(t, dtype=np.float64)
    if p.shape != q.shape or t.shape != p.shape or not np.isfinite(t).all():
        raise ValueError('Law/critic shapes must agree and critic must be finite.')
    if np.any(np.abs(t) > 1+1e-12):
        raise ValueError('This reference uses a bounded critic.')
    return float(np.sum(p*(t-.5*t*t)+q*(-t-.5*t*t)))


def common_risk(p: Array, q: Array, t: Array) -> float:
    # The identity is evaluated using a separate expression.
    p,q=law(p),law(q); t=np.asarray(t,dtype=float)
    _=j(p,q,t)
    return float(.5*np.sum(p*(1-t)**2+q*(-1-t)**2))


def cs_transform(s: float) -> float:
    if not math.isfinite(s) or not 0 <= s <= 1:
        raise ValueError('CS transform accepts population S in [0,1], not arbitrary empirical J.')
    return math.inf if s == 1 else 2*(math.log1p(s)-math.log1p(-s))


def pushforward(p: Array, q: Array, observation_id: Array) -> tuple[Array, Array]:
    p,q=law(p),law(q); ids=np.asarray(observation_id)
    if p.shape != q.shape or ids.shape != p.shape:
        raise ValueError('Observation map shape must equal law shape.')
    _, inv = np.unique(ids.ravel(), return_inverse=True)
    return np.bincount(inv,weights=p.ravel()),np.bincount(inv,weights=q.ravel())


def task_posterior(c: Array, weights: Array, observation_id: Array) -> Array:
    c=np.asarray(c,dtype=float); w=law(weights); ids=np.asarray(observation_id)
    if c.shape != w.shape or ids.shape != c.shape or not np.isin(c,[-1.,1.]).all():
        raise ValueError('Signed binary labels and matching arrays are required.')
    _,inv=np.unique(ids.ravel(),return_inverse=True)
    mass=np.bincount(inv,weights=w.ravel()); total=np.bincount(inv,weights=(w*c).ravel())
    post=np.divide(total,mass,out=np.zeros_like(total),where=mass>0)
    return post[inv].reshape(c.shape)


def finite_risk_delta(c: Array, a: Array, b: Array) -> tuple[Array,Array,Array]:
    c,a,b=(np.asarray(x,dtype=float) for x in (c,a,b))
    if c.shape != a.shape or c.shape != b.shape or not np.isfinite([c,a,b]).all():
        raise ValueError('All arrays must be finite and have the same shape.')
    h=(c-a)**2-(c-b)**2; d=(b-a)**2; residual=2*(c-b)*(b-a)
    return h,d,residual


def conditional_s(p_cond: Array, q_cond: Array, condition_weights: Array) -> tuple[float,Array]:
    p_cond=np.asarray(p_cond,dtype=float); q_cond=np.asarray(q_cond,dtype=float)
    weights=law(condition_weights)
    if p_cond.ndim != 2 or p_cond.shape != q_cond.shape or weights.shape != (len(p_cond),):
        raise ValueError('Expected [condition,state] laws and [condition] weights.')
    vals=np.array([eta_s(p,q)[1] for p,q in zip(p_cond,q_cond)])
    global_s=eta_s(weights[:,None]*p_cond,weights[:,None]*q_cond)[1]
    return global_s,vals


def sigmoid(logit: Array) -> Array:
    z=np.asarray(logit,dtype=float)
    # Stable for finite real logits, no accidental overflow warnings.
    return np.exp(-np.logaddexp(0.,-z))


def probability_at_prior(t_balanced: Array, positive_prior: float) -> Array:
    """Prior-shift conversion; NOT valid for arbitrary changes in P or Q."""
    t=np.asarray(t_balanced,dtype=float)
    if not 0 < positive_prior < 1 or not np.isfinite(t).all() or np.any(abs(t)>1):
        raise ValueError('Valid prior and bounded balanced-posterior critic required.')
    numerator=positive_prior*(1+t)
    denominator=numerator+(1-positive_prior)*(1-t)
    return numerator/denominator


def binary_brier(y: Array, probabilities: Array) -> float:
    y=np.asarray(y,dtype=float); d=np.asarray(probabilities,dtype=float)
    if y.shape!=d.shape or not np.isin(y,[0.,1.]).all() or not np.isfinite(d).all() or np.any((d<0)|(d>1)):
        raise ValueError('Binary labels and probabilities in [0,1] required.')
    return float(np.mean((y-d)**2))


def positive_loss_decomposition(logits: Array, valid: Array) -> tuple[float,float,float]:
    """Returns distribution CE, set-event CE, KL(U_valid||p(.|valid))."""
    logits=np.asarray(logits,dtype=float); valid=np.asarray(valid,dtype=bool)
    if logits.ndim!=1 or valid.shape!=logits.shape or not valid.any() or not np.isfinite(logits).all():
        raise ValueError('Finite candidate logits with at least one valid item required.')
    log_all=float(np.logaddexp.reduce(logits)); log_valid=float(np.logaddexp.reduce(logits[valid]))
    ce_dist=float(np.mean(log_all-logits[valid])); ce_set=log_all-log_valid
    kl=float(-math.log(valid.sum())-np.mean(logits[valid]-log_valid))
    return ce_dist,ce_set,kl


def risk_coverage(losses: Array, confidence: Array) -> tuple[Array,Array]:
    losses=np.asarray(losses,dtype=float); confidence=np.asarray(confidence,dtype=float)
    if losses.ndim!=1 or losses.shape!=confidence.shape or len(losses)==0 or not np.isfinite([losses,confidence]).all():
        raise ValueError('Nonempty matching 1D arrays required.')
    order=np.argsort(-confidence,kind='stable')
    k=np.arange(1,len(losses)+1)
    return k/len(losses), np.cumsum(losses[order])/k


def time_iou(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a)!=2 or len(b)!=2 or not np.isfinite([a,b]).all() or a[1]<=a[0] or b[1]<=b[0]:
        raise ValueError('Each time window must have two finite, increasing endpoints.')
    inter=max(0.,min(a[1],b[1])-max(a[0],b[0]))
    return inter/((a[1]-a[0])+(b[1]-b[0])-inter)


def fixture_ap_at_r(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Toy AP@R only. Official ECCV query lists and evaluation are required."""
    if not relevant or len(set(retrieved))!=len(retrieved):
        raise ValueError('Nonempty relevance and a duplicate-free ranking required.')
    r=len(relevant); hits=0; total=0.
    for k,item in enumerate(retrieved[:r],1):
        if item in relevant:
            hits+=1; total+=hits/k
    return total/r


def assert_group_disjoint(roles: Mapping[str,Sequence[str]]) -> None:
    owner={}
    for role,groups in roles.items():
        for g in set(groups):
            if g in owner and owner[g]!=role:
                raise ValueError(f'Group leakage: {g} is in {owner[g]} and {role}.')
            owner[g]=role


def effective_sample_size(weights: Array) -> float:
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Finite nonnegative nonzero weights required.')
    return float(w.sum()**2/np.sum(w*w))


def earliest_budget(predicted_increase: Sequence[float], costs: Sequence[float], tolerance: float) -> int:
    """Smallest-cost predicted feasible action; no correctness guarantee."""
    r,c=np.asarray(predicted_increase,dtype=float),np.asarray(costs,dtype=float)
    if r.ndim!=1 or r.shape!=c.shape or not np.isfinite([r,c]).all() or np.any(c<0) or tolerance<0:
        raise ValueError('Matching finite arrays and nonnegative costs/tolerance required.')
    good=np.flatnonzero(r<=tolerance)
    if len(good)==0:
        raise ValueError('No predicted-feasible action; use an explicit fallback policy.')
    return int(good[np.argmin(c[good])])
