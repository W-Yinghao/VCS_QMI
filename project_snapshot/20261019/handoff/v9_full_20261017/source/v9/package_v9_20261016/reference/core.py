"""V9 mathematical reference, not a benchmark implementation.

All finite-sample interval interpretations are conditional on a score fixed outside
three independent bounded evaluation samples and on the specified P/Q mixture.
No function infers semantic truth, licence, independence or clean data from a file name.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Sequence
import math
import numpy as np


def vector(x, name="array"):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D array")
    return a


def probability_mass(x, name="mass"):
    a = vector(x, name)
    if np.any(a < 0) or not np.isclose(a.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError(f"{name} must be nonnegative and sum to one")
    return a


def finite_target(p, q):
    p, q = probability_mass(p, "p"), probability_mass(q, "q")
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    den = p + q
    eta = np.divide(p-q, den, out=np.zeros_like(p), where=den > 0)
    m = den / 2
    return {"eta": eta, "m": m, "S": float(np.dot(m, eta**2))}


def finite_j(p, q, t):
    p, q, t = probability_mass(p, "p"), probability_mass(q, "q"), vector(t, "T")
    if p.shape != q.shape or p.shape != t.shape:
        raise ValueError("shapes differ")
    return float(np.dot(p-q, t) - .5*np.dot(p+q, t*t))


def mixture_mass(p, q, rho: float):
    p, q = probability_mass(p, "p"), probability_mass(q, "q")
    if p.shape != q.shape or not 0 <= rho <= 1:
        raise ValueError("invalid mixture")
    return (1-rho)*p+rho*q


@dataclass(frozen=True)
class RatioEstimate:
    raw: float | None
    projected: float | None
    denominator: float
    status: str


def rho_from_means(a: float, b: float, c: float, atol: float = 1e-12):
    if not all(math.isfinite(v) for v in (a,b,c)) or atol <= 0:
        raise ValueError("invalid means or numerical tolerance")
    gap = a-b
    if abs(gap) <= atol:
        return RatioEstimate(None, None, gap, "unidentified_at_numerical_tolerance")
    raw = (a-c)/gap
    return RatioEstimate(raw, min(1.,max(0.,raw)), gap,
                         "outside_parameter_space" if raw < 0 or raw > 1 else "point_estimate")


@dataclass(frozen=True)
class IntervalSet:
    lower: float | None
    upper: float | None
    status: str
    alpha: float | None = None


def interval_from_radii(a: float,b: float,c: float,e_p:float,e_q:float,e_r:float):
    """Exact linear feasibility set; no asymptotic ratio normal approximation."""
    vals = (a,b,c,e_p,e_q,e_r)
    if not all(math.isfinite(x) for x in vals) or min(e_p,e_q,e_r)<0:
        raise ValueError("invalid moments/radii")
    d0, gap = c-a, a-b
    h0, h1 = e_r+e_p, e_q-e_p
    lo, hi = 0., 1.
    for A,B in ((gap-h1,h0-d0),(-gap-h1,h0+d0)):
        if abs(A) < 1e-15:
            if B < -1e-14:
                return IntervalSet(None,None,"empty")
        elif A > 0:
            hi = min(hi,B/A)
        else:
            lo = max(lo,B/A)
    if lo > hi+1e-13:
        return IntervalSet(None,None,"empty")
    lo,hi = max(0.,lo),min(1.,hi)
    if lo > hi: lo = hi = (lo+hi)/2
    return IntervalSet(lo,hi,"full_parameter_space" if lo<=1e-13 and hi>=1-1e-13 else "bounded_interval")


def rho_confidence_set(p_scores,q_scores,r_scores,alpha=.05,*,assumptions_verified=False):
    """Hoeffding+union inversion for independent cluster-level means in [-1,1].

    A True flag is a contract, NOT an automatic test of independence/mixing.
    Shared-image cross-pairs require a separate cluster construction/proof.
    """
    if not assumptions_verified:
        raise ValueError("finite-sample interpretation requires verified assumptions")
    if not 0 < alpha < 1: raise ValueError("alpha must be in (0,1)")
    arr = [vector(x,n) for x,n in zip((p_scores,q_scores,r_scores),('P','Q','R'))]
    if any(np.any(np.abs(x)>1+1e-12) for x in arr):
        raise ValueError("cluster scores must lie in [-1,1]")
    es = [math.sqrt(2*math.log(6/alpha)/len(x)) for x in arr]
    z = interval_from_radii(*(float(x.mean()) for x in arr),*es)
    return IntervalSet(z.lower,z.upper,z.status,alpha)


def clean_posterior(t, rho:float):
    """Clean-component posterior from CLEAN P-vs-Q T; not R-vs-Q T."""
    a=vector(t,"T")
    if not 0 <= rho <= 1 or np.any(abs(a)>1+1e-12):
        raise ValueError("invalid T or rho")
    num=(1-rho)*(1+a); den=num+rho*(1-a)
    if np.any(den <= 0):
        raise ValueError("posterior undefined at a zero-mixture-support point")
    return num/den


def brier(y,p,weights=None):
    y,p=vector(y,'labels'),vector(p,'probabilities')
    if y.shape!=p.shape or np.any((y<0)|(y>1)|(p<0)|(p>1)):
        raise ValueError("invalid labels or probabilities")
    a=(y-p)**2
    if weights is None:return float(a.mean())
    ww=vector(weights,'weights')
    if ww.shape!=a.shape or np.any(ww<0) or ww.sum()<=0:raise ValueError("invalid weights")
    return float(np.dot(ww,a)/ww.sum())


def logloss(y,p,eps=1e-12):
    y,p=vector(y,'labels'),vector(p,'probabilities')
    if y.shape!=p.shape or np.any((y<0)|(y>1)|(p<0)|(p>1)) or not 0<eps<.5:
        raise ValueError("invalid labels/probabilities/epsilon")
    z=np.clip(p,eps,1-eps)
    return float(np.mean(-y*np.log(z)-(1-y)*np.log1p(-z)))


def effective_sample_size(weights):
    a=vector(weights,'weights')
    if np.any(a<0):raise ValueError("negative weight")
    return float(a.sum()**2/np.dot(a,a)) if np.dot(a,a)>0 else 0.


def rank_mass_weights(scores,total_mass:float):
    """Tie-stable baseline: same number-equivalent mass, not same probabilities."""
    a=vector(scores,'scores')
    if not 0<=total_mass<=len(a):raise ValueError("invalid total mass")
    order=np.argsort(-a,kind='stable');out=np.zeros_like(a)
    k=int(math.floor(total_mass));out[order[:k]]=1.
    if k<len(a):out[order[k]]=total_mass-k
    return out


def role_disjointness(rows):
    """Check each underlying unit occurs in only one independent score cluster.

    Each row has role, cluster_id, and units (image/donor IDs).
    Repeated rows from a cluster must be pre-aggregated before this check.
    """
    seen={};clusters=set()
    for r in rows:
        key=(str(r['role']),str(r['cluster_id']))
        if key in clusters:raise ValueError("duplicate cluster; aggregate its scores first")
        clusters.add(key)
        units=r.get('units')
        if not isinstance(units,list) or not units or len(set(units))!=len(units):
            raise ValueError("missing or duplicated provenance units")
        for u in units:
            if u in seen:raise ValueError(f"underlying unit {u!r} reused across clusters")
            seen[u]=key
    return True


def retrieval_recall(scores,image_ids,caption_image_ids,ks=(1,5,10)):
    """Fixed gallery; any GT caption counts for I2T; stable ID-order ties.

    This CPU reference checks metric definitions; use each public benchmark's
    official evaluator and exact split/gallery for reported paper results.
    """
    s=np.asarray(scores,dtype=float); im=list(image_ids);cap=list(caption_image_ids)
    if s.shape!=(len(im),len(cap)) or not np.isfinite(s).all() or not im or not cap:
        raise ValueError("invalid score matrix")
    if len(set(im))!=len(im) or not set(cap)<=set(im) or not set(im)<=set(cap):
        raise ValueError("GT mappings must cover every query/gallery image")
    if not all(isinstance(k,int) and k>0 for k in ks):raise ValueError("invalid k")
    oi=np.argsort(-s,axis=1,kind='stable');ot=np.argsort(-s,axis=0,kind='stable')
    out={}
    for k in ks:
        out[f'i2t_R{k}']=100*np.mean([any(cap[j]==v for j in oi[i,:k]) for i,v in enumerate(im)])
        out[f't2i_R{k}']=100*np.mean([any(im[i]==v for i in ot[:k,j]) for j,v in enumerate(cap)])
    out['rsum']=float(sum(out.values()));out['mR']=out['rsum']/(2*len(ks))
    return {k:float(v) for k,v in out.items()}


def mean_ci_vector_consistency(a,b,c,rho):
    """Descriptive multi-score mixture residual; zero does NOT certify no shift."""
    a,b,c=vector(a),vector(b),vector(c)
    if a.shape!=b.shape or a.shape!=c.shape or not 0<=rho<=1:raise ValueError("invalid arguments")
    return c-((1-rho)*a+rho*b)
