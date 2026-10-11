"""CPU reference identities for r6. Not a natural-data trainer or registration system."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import math
from typing import Hashable, Mapping, Sequence
import numpy as np


def law(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('A finite nonnegative probability table is required.')
    if not np.isclose(x.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError('The probability table must sum to one.')
    return x


def eta_s(p, q):
    p, q = law(p), law(q)
    if p.shape != q.shape:
        raise ValueError('P and Q must have equal shape.')
    d = p + q
    eta = np.divide(p-q, d, out=np.zeros_like(d), where=d > 0)
    return eta, float(np.sum(0.5*d*eta*eta))


def j_value(p, q, t):
    p, q = law(p), law(q)
    t = np.asarray(t, dtype=np.float64)
    if p.shape != q.shape or p.shape != t.shape or not np.isfinite(t).all():
        raise ValueError('P, Q, T must be finite and have equal shape.')
    if np.max(np.abs(t)) > 1+1e-12:
        raise ValueError('This reference checks the bounded-critic convention.')
    return float(np.sum(p*(t-0.5*t*t)+q*(-t-0.5*t*t)))


def rpc_value(p, q, t, alpha=1., beta=1., gamma=1.):
    p, q = law(p), law(q)
    t = np.asarray(t, dtype=np.float64)
    if p.shape != q.shape or t.shape != p.shape:
        raise ValueError('Shape mismatch.')
    return float(np.sum(p*t-alpha*q*t-0.5*beta*p*t*t-0.5*gamma*q*t*t))


def cs_transform(s: float) -> float:
    if not np.isfinite(s) or s < 0 or s > 1:
        raise ValueError('The transform is for a population S in [0,1].')
    return math.inf if s == 1 else float(2*(np.log1p(s)-np.log1p(-s)))


def binary_sensor_law(rho: float):
    if not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError('rho must lie in [-1,1].')
    signs = np.array([-1., 1.])
    p = (1 + rho * np.outer(signs, signs))/4
    q = np.ones((2, 2))/4
    return p, q, float(rho*rho/(4-rho*rho))


@dataclass(frozen=True)
class State:
    probability: float
    u: Hashable
    v: Hashable
    c: int


def checked_states(states: Sequence[State]):
    if not states or any(s.c not in (-1,1) for s in states):
        raise ValueError('Binary C values must be -1 or +1.')
    law([s.probability for s in states])
    return tuple(states)


def conditional_means(states: Sequence[State], full: bool=False) -> dict:
    states = checked_states(states)
    total, weighted = defaultdict(float), defaultdict(float)
    for s in states:
        key = (s.u,s.v) if full else s.u
        total[key] += s.probability
        weighted[key] += s.probability*s.c
    return {k: weighted[k]/total[k] for k in total if total[k] > 0}


def evidence_summary(states: Sequence[State]) -> dict:
    states = checked_states(states)
    cmean = sum(s.probability*s.c for s in states)
    if abs(cmean) > 1e-10:
        raise ValueError('The S-risk connection here uses a balanced binary task.')
    a, b = conditional_means(states), conditional_means(states,True)
    su = sum(s.probability*a[s.u]**2 for s in states)
    sb = sum(s.probability*b[(s.u,s.v)]**2 for s in states)
    ru = sum(s.probability*(s.c-a[s.u])**2 for s in states)
    rb = sum(s.probability*(s.c-b[(s.u,s.v)])**2 for s in states)
    gap = sum(s.probability*(b[(s.u,s.v)]-a[s.u])**2 for s in states)
    mass, gains = defaultdict(float), defaultdict(float)
    for s in states:
        mass[s.u] += s.probability
        gains[s.u] += s.probability*(b[(s.u,s.v)]-a[s.u])**2
    return {'S_cheap':su, 'S_full':sb, 'risk_cheap':ru, 'risk_full':rb,
            'delta':gap, 'conditional_gain':{u:gains[u]/mass[u] for u in mass}}


def risk_components(c, a, b):
    c, a, b = (np.asarray(v,dtype=np.float64) for v in (c,a,b))
    if c.shape != a.shape or a.shape != b.shape or not all(np.isfinite(x).all() for x in (c,a,b)):
        raise ValueError('Finite equal-shape arrays are required.')
    square = (b-a)**2
    residual = 2*(b-a)*(c-b)
    utility = (c-a)**2-(c-b)**2
    return square, residual, utility


def finite_model_utility(states, a: Mapping, b: Mapping):
    states = checked_states(states)
    w = np.array([s.probability for s in states])
    z,r,h = risk_components([s.c for s in states], [a[s.u] for s in states], [b[(s.u,s.v)] for s in states])
    return {'square':float(w@z), 'residual':float(w@r), 'actual_gain':float(w@h)}


def make_task(name: str) -> list[State]:
    signs=(-1,1)
    if name == 'redundant':
        return [State(.5,x,x,x) for x in signs]
    if name == 'irrelevant':
        return [State(.25,x,v,x) for x in signs for v in signs]
    if name == 'xor':
        return [State(.25,x,v,x*v) for x in signs for v in signs]
    if name == 'context':
        return [State(.125,(k,x),v,x if k==0 else x*v) for k in (0,1) for x in signs for v in signs]
    raise ValueError('Unknown controlled task.')


def oracle_policy_risk(states, cost: float):
    if not np.isfinite(cost) or cost < 0:
        raise ValueError('Nonnegative finite cost required.')
    e=evidence_summary(states); a=conditional_means(states); b=conditional_means(states,True)
    r=c=0.
    for s in states:
        take=e['conditional_gain'][s.u] > cost
        pred=b[(s.u,s.v)] if take else a[s.u]
        r += s.probability*(s.c-pred)**2
        c += s.probability*cost*take
    return {'task_risk':r,'acquisition_cost':c,'total':r+c}


def total_cost(cheap, policy, extra, taken):
    if min(cheap,policy,extra) < 0 or not np.isfinite([cheap,policy,extra]).all():
        raise ValueError('Costs must be finite and nonnegative.')
    a=np.asarray(taken)
    if a.size == 0 or np.any((a != 0)&(a != 1)):
        raise ValueError('Acquisition flags must be 0/1 and nonempty.')
    return float(cheap+policy+extra*a.mean())


def validate_policy_inputs(inputs):
    allowed={'cheap_input','cheap_features','cheap_confidence','available_quality'}
    illegal=set(inputs)-allowed
    if illegal:
        raise ValueError('Policy reads unavailable or undefined fields: '+str(sorted(illegal)))
    return True


def assert_group_disjoint(roles: Mapping[str,Sequence[Hashable]]):
    seen={}
    for role, groups in roles.items():
        for g in set(groups):
            if g in seen and seen[g] != role:
                raise ValueError(f'Group leakage: {g} in {seen[g]} and {role}')
            seen[g]=role
    return True


def stable_group_roles(groups: Sequence[str], salt: str):
    """Illustrative hash split; datasets with official splits must preserve them."""
    out={k:[] for k in ('FIT_TASK','FIT_POLICY','CAL','DEV')}
    for g in sorted(set(groups)):
        u=int.from_bytes(hashlib.sha256((salt+'|'+g).encode()).digest()[:8],'big')/2**64
        role='FIT_TASK' if u<.6 else 'FIT_POLICY' if u<.8 else 'CAL' if u<.9 else 'DEV'
        out[role].append(g)
    return out


def shift_demo(seed=711, n_scenes=40, width=256, true_shift=3, flip_a=.1, flip_b=.15):
    """Discrete circular sensor example. Uses the GENERATOR's known uniform Q.
    No continuous registration gradients and no natural-data performance claim.
    """
    if n_scenes<1 or width<32 or not 0 <= flip_a <= .5 or not 0 <= flip_b <= .5:
        raise ValueError('Invalid controlled sensor specification.')
    rng=np.random.default_rng(seed)
    latent=rng.choice([-1.,1.],size=(n_scenes,width))
    u=latent*np.where(rng.random(latent.shape)<flip_a,-1.,1.)
    v=np.roll(latent,true_shift,axis=1)*np.where(rng.random(latent.shape)<flip_b,-1.,1.)
    rho=(1-2*flip_a)*(1-2*flip_b)
    p,q,s=binary_sensor_law(rho); t,_=eta_s(p,q)
    curve={}
    for shift in range(-6,7):
        aligned=np.roll(v,-shift,axis=1)
        index=((u>0).astype(int)*2+(aligned>0).astype(int)).ravel()
        phat=np.bincount(index,minlength=4).reshape(2,2)/index.size
        curve[str(shift)]=j_value(phat,q,t)
    recovered=max(curve,key=curve.get)
    return {'target_id':'controlled_population_uniform_product','seed':seed,'true_shift':true_shift,
            'recovered_shift':int(recovered),'analytic_rho':rho,'analytic_aligned_S':s,'J_curve':curve,
            'n_independent_scenes':n_scenes,'note':'discrete CPU implementation check, not image registration performance'}
