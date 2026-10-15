"""CPU definitions for V8.1. No model fitting or test-label access.

These are independent diagnostic utilities, not replacements for the frozen
V8-04 / V8-05 evaluators. NumPy float64 is used throughout.
"""
from __future__ import annotations
import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Sequence
import numpy as np
from scipy.stats import kendalltau, rankdata

LABEL_ROLES = {'FIT_AUDIT', 'CAL_SELECT', 'CAL_PROB', 'DEV'}
LABEL_FIELDS = {'human_hist', 'human_mean', 'votes', 'overall_votes', 'hist', 'rating', 'human_score'}
REQUIRED = {'dataset', 'role', 'method', 'seed', 'checkpoint_id', 'prompt_id', 'pool_id', 'image_id', 'candidate_index', 'score'}
MODEL_KEY = ('dataset', 'role', 'method', 'seed', 'checkpoint_id')
LEVELS = np.arange(1., 6.)


def array1(values: Sequence[float], *, nonempty: bool = True) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or (nonempty and not a.size) or not np.isfinite(a).all():
        raise ValueError('Expected a nonempty, finite, one-dimensional vector')
    return a


def simplex(values: Sequence[float]) -> np.ndarray:
    a = array1(values)
    if a.shape != (5,) or np.any(a < 0.) or not np.isclose(a.sum(), 1., atol=1e-8, rtol=0.):
        raise ValueError('Expected 5 nonnegative probabilities summing to one')
    return a


def histogram(votes: Sequence[float]) -> np.ndarray:
    v = array1(votes)
    if np.any(v != np.round(v)) or np.any((v < 1) | (v > 5)):
        raise ValueError('Votes must be integers in 1..5')
    return np.bincount(v.astype(int), minlength=6)[1:].astype(float) / len(v)


def tails(q: Sequence[float]) -> np.ndarray:
    h = simplex(q)
    return np.cumsum(h[::-1])[::-1][1:]


def moments(q: Sequence[float]) -> dict[str, Any]:
    h = simplex(q)
    mu = float(h @ LEVELS)
    return {'u': tails(h).tolist(), 'mu': mu, 'rating_variance': float(h @ (LEVELS-mu)**2)}


def distribution_metrics(q: Sequence[float], human_hist: Sequence[float], eps: float = 1e-12) -> dict[str, float]:
    if not 0 < eps < 1:
        raise ValueError('eps must be in (0,1)')
    p, h = simplex(q), simplex(human_hist)
    u, y = tails(p), tails(h)
    risk = y*(1-u)**2 + (1-y)*u**2
    fit = (u-y)**2
    const = y*(1-y)
    d = {
        'rps_random_rater': float(risk.mean()),
        'cdf_fit_mse': float(fit.mean()),
        'vote_constant': float(const.mean()),
        'rps_identity_residual': float(np.max(np.abs(risk-fit-const))),
        'hist_log_loss': float(-(h*np.log(np.maximum(p, eps))).sum()),
        'mu_mse': float(((p-h) @ LEVELS)**2),
        'rating_variance': float(p @ (LEVELS-p@LEVELS)**2),
        'log_probability_floor': eps,
    }
    for k in range(4):
        d[f'brier_Hge{k+2}'] = float(risk[k])
    return d


def mse_decomposition(pred: Sequence[float], target: Sequence[float], weights: Sequence[float] | None = None) -> dict[str, float]:
    p, y = array1(pred), array1(target)
    if len(p) != len(y):
        raise ValueError('Length mismatch')
    w = np.ones(len(p), dtype=float) if weights is None else array1(weights)
    if len(w) != len(p) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Weights must match and have positive mass')
    w = w/w.sum(); e = p-y; b = float(w@e)
    mse, within = float(w@(e*e)), float(w@((e-b)**2))
    return {'mse': mse, 'mean_error': b, 'bias_squared': b*b,
            'centered_mse': within, 'identity_residual': mse-b*b-within}


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    a, b = array1(x), array1(y)
    if len(a) != len(b):
        raise ValueError('Length mismatch')
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(np.corrcoef(rankdata(a, method='average'), rankdata(b, method='average'))[0, 1])


def kendall_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    a, b = array1(x), array1(y)
    if len(a) != len(b):
        raise ValueError('Length mismatch')
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    v = float(kendalltau(a, b, variant='b').statistic)
    return v if math.isfinite(v) else None


def pairwise_accuracy(pred: Sequence[float], target: Sequence[float]) -> dict[str, Any]:
    p, y = array1(pred), array1(target)
    if len(p) != len(y):
        raise ValueError('Length mismatch')
    i, j = np.triu_indices(len(p), 1)
    dy, ds = y[i]-y[j], p[i]-p[j]
    ok = dy != 0
    n = int(ok.sum())
    acc = None if not n else float(np.where(ds[ok] == 0, .5, (np.sign(ds[ok]) == np.sign(dy[ok])).astype(float)).mean())
    return {'pair_accuracy': acc, 'informative_pairs': n, 'all_pairs': int(len(i)),
            'label_tied_pairs': int((~ok).sum()), 'model_tied_informative_pairs': int((ds[ok] == 0).sum())}


def top_index(scores: Sequence[float]) -> int:
    """The caller supplies the frozen candidate order. Exact ties keep the first."""
    return int(np.argmax(array1(scores)))


def selection_metrics(scores: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    s, y = array1(scores), array1(target)
    if len(s) != len(y):
        raise ValueError('Length mismatch')
    selected = float(y[top_index(s)])
    return {'selected_human_mean': selected, 'regret': float(y.max()-selected),
            'random_expected_human_mean': float(y.mean()),
            'observed_pool_max': float(y.max())}


def decision_agreement(a: Sequence[float], b: Sequence[float], target: Sequence[float] | None = None) -> dict[str, Any]:
    x, y = array1(a), array1(b)
    if len(x) != len(y):
        raise ValueError('Candidate sets differ')
    sx, sy = set(np.flatnonzero(x == x.max())), set(np.flatnonzero(y == y.max()))
    d = {'top1_agreement': float(top_index(x) == top_index(y)),
         'topset_intersects': float(bool(sx & sy)), 'topset_jaccard': len(sx & sy)/len(sx | sy),
         'top_tie_a': float(len(sx)>1), 'top_tie_b': float(len(sy)>1),
         'spearman': spearman(x, y), 'kendall_tau_b': kendall_b(x, y)}
    if target is not None:
        t = array1(target)
        if len(t) != len(x): raise ValueError('Target size mismatch')
        d['selected_quality_delta'] = float(t[top_index(x)] - t[top_index(y)])
    return d


def swap_decomposition(agreements: Sequence[float], deltas: Sequence[float]) -> dict[str, Any]:
    a, d = array1(agreements), array1(deltas)
    if len(a) != len(d) or not np.isin(a, [0., 1.]).all():
        raise ValueError('Agreement must be binary with matching deltas')
    if np.any(np.abs(d[a == 1]) > 1e-10):
        raise ValueError('Same selected candidate cannot have different shared target value')
    changed = a == 0; rate = float(changed.mean())
    conditional = float(d[changed].mean()) if changed.any() else None
    lhs, rhs = float(d.mean()), 0. if conditional is None else rate*conditional
    return {'disagreement_rate': rate, 'delta_when_disagree': conditional,
            'overall_delta': lhs, 'identity_residual': lhs-rhs}


def cluster_bootstrap(values: Sequence[float], cluster_ids: Sequence[str], *, reps: int = 2000, seed: int = 2718) -> dict[str, Any]:
    """Mean of pool scalars with cluster resampling. Not pooled-correlation bootstrap."""
    v = array1(values)
    if len(v) != len(cluster_ids) or reps < 2:
        raise ValueError('Bad bootstrap inputs')
    groups: dict[str, list[float]] = defaultdict(list)
    for x, g in zip(v, cluster_ids): groups[str(g)].append(float(x))
    keys = sorted(groups)
    out: dict[str, Any] = {'mean': float(v.mean()), 'n_pools': len(v), 'n_clusters': len(keys), 'reps': reps, 'seed': seed}
    if len(keys) < 2:
        return {**out, 'ci95': None, 'status': 'insufficient_clusters'}
    sums = np.array([sum(groups[k]) for k in keys]); ns = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed); vals = np.empty(reps)
    for r in range(reps):
        ix = rng.integers(0, len(keys), len(keys))
        vals[r] = sums[ix].sum()/ns[ix].sum()
    return {**out, 'ci95': np.quantile(vals, [.025, .975]).tolist(), 'status': 'development_interval_no_p_value'}


def validate_rows(rows: list[dict], *, labelled: bool) -> list[dict]:
    if not rows: raise ValueError('No rows')
    seen = set(); indices: dict[tuple, set] = defaultdict(set)
    out=[]
    for r0 in rows:
        r=dict(r0)
        if missing := REQUIRED-set(r): raise ValueError(f'Missing fields: {sorted(missing)}')
        for k in ('dataset','role','method','checkpoint_id','prompt_id','pool_id','image_id'):
            if not isinstance(r[k], str) or not r[k]: raise ValueError(f'Nonempty string required: {k}')
        if labelled and r['role'] not in LABEL_ROLES:
            raise ValueError(f'Role not allowed for V8.1 labelled diagnostics: {r["role"]}')
        if not labelled and LABEL_FIELDS.intersection(r):
            raise ValueError('Labels (including dummy histograms) are forbidden in label-free input')
        if not isinstance(r['candidate_index'], int) or isinstance(r['candidate_index'], bool) or r['candidate_index'] < 0:
            raise ValueError('candidate_index must be a nonnegative integer')
        s=float(r['score'])
        if not math.isfinite(s): raise ValueError('Nonfinite score')
        r['score']=s; r.setdefault('score_kind','rating_mean'); r.setdefault('cluster_id',r['prompt_id'])
        if r['score_kind'] not in {'rating_mean','native_score'}: raise ValueError('Unknown score_kind')
        if r['score_kind']=='rating_mean' and not 1.-1e-5 <= s <= 5.+1e-5: raise ValueError('rating_mean outside 1..5')
        if 'q' in r:
            q=simplex(r['q']); m=moments(q)
            if r['score_kind']=='rating_mean' and not math.isclose(s,m['mu'],abs_tol=2e-5,rel_tol=0):
                raise ValueError('q and score disagree')
            if 'u' in r and not np.allclose(array1(r['u']),m['u'],atol=2e-5,rtol=0): raise ValueError('q/u inconsistent')
        elif 'u' in r:
            u=array1(r['u'])
            if len(u)!=4 or np.any((u<0)|(u>1)) or np.any(np.diff(u)>1e-10): raise ValueError('Invalid tails')
            if r['score_kind']=='rating_mean' and not math.isclose(s,1+float(u.sum()),abs_tol=2e-5,rel_tol=0): raise ValueError('u/score mismatch')
        if labelled:
            if 'votes' in r:
                h=histogram(r['votes'])
                if 'human_hist' in r and not np.allclose(h,simplex(r['human_hist']),atol=1e-10,rtol=0): raise ValueError('votes/hist mismatch')
                r.setdefault('human_hist',h.tolist())
            if 'human_hist' in r:
                h=simplex(r['human_hist']); hm=float(h@LEVELS)
                if 'human_mean' in r and not math.isclose(float(r['human_mean']),hm,abs_tol=1e-8,rel_tol=0): raise ValueError('human mean/hist mismatch')
                r['human_mean']=hm
            if 'human_mean' not in r: raise ValueError('Labelled input needs human_mean, hist or votes')
            if not math.isfinite(float(r['human_mean'])) or not 1 <= float(r['human_mean']) <= 5: raise ValueError('Invalid human mean')
        key=tuple(r[k] for k in MODEL_KEY)+(r['pool_id'],r['image_id'])
        poolkey=tuple(r[k] for k in MODEL_KEY)+(r['pool_id'],)
        if key in seen: raise ValueError('Duplicate prediction row')
        if r['candidate_index'] in indices[poolkey]: raise ValueError('Duplicate candidate index')
        seen.add(key);indices[poolkey].add(r['candidate_index']);out.append(r)
    return out


def mean_or_none(values: Sequence[Any]) -> float | None:
    good=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(good)) if good else None


def summarize(rows: list[dict], *, labelled: bool) -> dict:
    rr=validate_rows(rows,labelled=labelled)
    models: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rr: models[tuple(r[k] for k in MODEL_KEY)][r['pool_id']].append(r)
    for pools in models.values():
        for pool in pools:
            pools[pool].sort(key=lambda r:r['candidate_index'])
            if len({r['prompt_id'] for r in pools[pool]}) != 1 or len({r['cluster_id'] for r in pools[pool]}) != 1:
                raise ValueError('Inconsistent prompt/cluster within a pool')
    per_pool=[]; levels=[]; agreements=[]
    for key,pools in sorted(models.items(),key=lambda kv:str(kv[0])):
        identity=dict(zip(MODEL_KEY,key)); pp=[]
        for pool,rs in sorted(pools.items()):
            s=[r['score'] for r in rs]; row={**identity,'pool_id':pool,'prompt_id':rs[0]['prompt_id'],'cluster_id':rs[0]['cluster_id'],'n_candidates':len(rs),
                 'selected_image_id':rs[top_index(s)]['image_id'],'score_variance':float(np.var(s)),'top_tie_count':int(np.sum(np.array(s)==max(s)))}
            if labelled:
                y=[r['human_mean'] for r in rs]
                row.update(pairwise_accuracy(s,y)); row.update(selection_metrics(s,y));row['spearman_to_human']=spearman(s,y);row['kendall_to_human']=kendall_b(s,y)
                if all(r['score_kind']=='rating_mean' for r in rs): row.update(mse_decomposition(s,y))
                if all('q' in r and 'human_hist' in r for r in rs):
                    dm=[distribution_metrics(r['q'],r['human_hist']) for r in rs]
                    for k in dm[0]: row[k]=float(np.mean([d[k] for d in dm]))
            per_pool.append(row);pp.append(row)
        metric_keys=sorted({k for p in pp for k,v in p.items() if isinstance(v,(float,int)) and k not in {'seed','n_candidates','top_tie_count'}})
        levels.append({**identity,'n_pools':len(pp),'n_images':sum(p['n_candidates'] for p in pp),
                       'pool_macro':{k:mean_or_none([p.get(k) for p in pp]) for k in metric_keys},
                       'n_defined':{k:sum(p.get(k) is not None for p in pp) for k in metric_keys},
                       'interpretation':'pool_macro; do not treat mean within-pool correlations as pooled Spearman'})
    by_role=defaultdict(list)
    for key in models: by_role[key[:2]].append(key)
    pair_summaries=[]
    for role,keys in by_role.items():
        for ka,kb in combinations(sorted(keys,key=str),2):
            if ka[2] != kb[2] and ka[3] != kb[3] and ka[3] != -1 and kb[3] != -1:
                continue  # no arbitrary cross-seed, cross-method pairing
            A,B=models[ka],models[kb]
            if set(A)!=set(B): raise ValueError('Methods have different pools; no silent intersection')
            pair_rows=[]
            for pool in sorted(A):
                aa,bb=A[pool],B[pool]
                if [(r['image_id'],r['candidate_index']) for r in aa] != [(r['image_id'],r['candidate_index']) for r in bb]:
                    raise ValueError('Methods have different candidate sets/order')
                if [r['cluster_id'] for r in aa] != [r['cluster_id'] for r in bb]: raise ValueError('Cluster mismatch')
                y=None
                if labelled:
                    y=[r['human_mean'] for r in aa]
                    if not np.allclose(y,[r['human_mean'] for r in bb],atol=1e-8,rtol=0): raise ValueError('Methods have different labels')
                    for ra,rb in zip(aa,bb):
                        if 'human_hist' in ra and 'human_hist' in rb and not np.allclose(ra['human_hist'],rb['human_hist'],atol=1e-8,rtol=0):
                            raise ValueError('Methods have different vote histograms')
                d=decision_agreement([r['score'] for r in aa],[r['score'] for r in bb],y)
                row={'dataset':role[0],'role':role[1], 'a':dict(zip(MODEL_KEY,ka)),'b':dict(zip(MODEL_KEY,kb)),
                     'pool_id':pool,'prompt_id':aa[0]['prompt_id'],'cluster_id':aa[0]['cluster_id'],**d}
                agreements.append(row);pair_rows.append(row)
            metrics=['top1_agreement','topset_intersects','topset_jaccard','top_tie_a','top_tie_b','spearman','kendall_tau_b']
            ps={'a':dict(zip(MODEL_KEY,ka)),'b':dict(zip(MODEL_KEY,kb)),'n_pools':len(pair_rows),
                'pool_macro':{k:mean_or_none([r[k] for r in pair_rows]) for k in metrics},
                'n_defined':{k:sum(r[k] is not None for r in pair_rows) for k in metrics}}
            if labelled:
                ps['swap']=swap_decomposition([r['top1_agreement'] for r in pair_rows],[r['selected_quality_delta'] for r in pair_rows])
                ps['quality_delta_ci']=cluster_bootstrap([r['selected_quality_delta'] for r in pair_rows],[r['cluster_id'] for r in pair_rows])
            pair_summaries.append(ps)
    return {'schema_version':'V81.1','labelled':labelled,'status':'development_diagnostics_not_confirmation',
            'n_rows':len(rr),'model_levels':levels,'per_pool':per_pool,'pair_summaries':pair_summaries,'agreement_rows':agreements}
