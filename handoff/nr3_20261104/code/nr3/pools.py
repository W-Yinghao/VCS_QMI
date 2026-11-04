"""NR3 fixed FIT diagnostic pools (CAL-A seed 20261104031, READ-B seed 20261104032), truth-free.
Per direction: n P pairs = observed records (query, its observed partner) and n Q_h pairs = query from an independent
observed-record draw (p_query) and candidate ~ q(.|query) = alpha p_cand + (1-alpha) Uniform(base FIT Top100(query)).
log w = exact full-mixture log(q/p_cand) for P and Q_h pairs (nr2.reference.log_w). Draw order is fixed:
for d in (i2t, t2i): P records, Q records, component u, marginal record m, hard slot h."""
import numpy as np
import torch
from nr2.reference import log_w


def build(marg, top, seed, n=32768, alpha=.5):
    g = np.random.default_rng(int(seed)); N = len(marg.im); out = {}
    for d in ('i2t', 't2i'):
        rp, rq = g.integers(0, N, n), g.integers(0, N, n)
        u, m, h = g.random(n), g.integers(0, N, n), g.integers(0, top[d].shape[1], n)
        if d == 'i2t':
            qP, cP, qQ, marginal, pc = marg.im[rp], marg.tx[rp], marg.im[rq], marg.tx[m], marg.p_txt
        else:
            qP, cP, qQ, marginal, pc = marg.tx[rp], marg.im[rp], marg.tx[rq], marg.im[m], marg.p_img
        cQ = np.where(u < alpha, marginal, top[d][qQ, h])
        out[d] = dict(P_records=rp, P_query=qP, P_cand=cP, P_logw=log_w(qP, cP, top[d], pc, alpha), P_hard=(top[d][qP] == cP[:, None]).any(1),
                      Q_records=rq, Q_query=qQ, Q_cand=cQ, Q_logw=log_w(qQ, cQ, top[d], pc, alpha), Q_hard=(top[d][qQ] == cQ[:, None]).any(1))
    return out


def save(pool, path):
    np.savez(path, **{f'{d}__{k}': v for d, x in pool.items() for k, v in x.items()})


def load(path):
    z = np.load(path); out = {'i2t': {}, 't2i': {}}
    for k in z.files:
        d, kk = k.split('__', 1); out[d][kk] = z[k]
    return out


@torch.no_grad()
def model_scores(model, X, Y, d, q, c, chunk=16384):
    """s for pairs (query q, candidate c) of direction d (i2t: image query, text candidate)."""
    res = []
    for s in range(0, len(q), chunk):
        qi = torch.as_tensor(q[s:s+chunk], device=X.device); ci = torch.as_tensor(c[s:s+chunk], device=X.device)
        z = model.pairs(X[qi], Y[ci]) if d == 'i2t' else model.pairs(X[ci], Y[qi])
        res.append(z.double().cpu().numpy())
    return np.concatenate(res)


@torch.no_grad()
def base_cos(Xn, Yn, d, q, c, chunk=65536):
    res = []
    for s in range(0, len(q), chunk):
        qi = torch.as_tensor(q[s:s+chunk], device=Xn.device); ci = torch.as_tensor(c[s:s+chunk], device=Xn.device)
        res.append(((Xn[qi] * Yn[ci]).sum(-1) if d == 'i2t' else (Xn[ci] * Yn[qi]).sum(-1)).cpu().numpy())
    return np.concatenate(res)


def stack(pool, fP, fQ):
    """[2, n] arrays in direction order (i2t, t2i) for nr3.ref_init."""
    return (np.stack([fP['i2t'], fP['t2i']]), np.stack([fQ['i2t'], fQ['t2i']]),
            np.stack([pool['i2t']['P_logw'], pool['t2i']['P_logw']]), np.stack([pool['i2t']['Q_logw'], pool['t2i']['Q_logw']]))
