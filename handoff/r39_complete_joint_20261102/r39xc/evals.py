"""R37C port of the R35/R37 evaluators to CC152K (per_captions = PER_CAP = 1, explicit identity instead of //5).
R35 evaluators (no training data enters here except FIT score vectors for calibration).
Rankings use native_eval.py (upstream IN2R i2t/t2i, verbatim copy of R34's copy) + tie bounds."""
from __future__ import annotations
import numpy as np
import torch
from native_eval import i2t, t2i, upstream_metrics, tie_bounded
from common import N_DEV_IMG, N_DEV_CAP, PER_CAP

DEV_GT = np.arange(N_DEV_CAP) // PER_CAP   # CC152K: identity (caption k <-> image k)
K_TOP = 100


# ----------------------------------------------------------------------------- J estimators

@torch.no_grad()
def j_from_matrix(g, m):
    """R34 record U-statistic on a full [n_img, n_cap] score matrix; m = caption -> image map."""
    g = torch.as_tensor(g)
    n = g.shape[1]
    mt = torch.as_tensor(m, device=g.device)
    t = torch.tanh(g / 2).double()
    tp = t[mt, torch.arange(n, device=g.device)]
    ep_lin, ep_quad = tp.mean(), (tp * tp).mean()
    counts = torch.bincount(mt.cpu(), minlength=g.shape[0]).double().to(g.device)   # CPU bincount (CUDA one is non-deterministic)
    lin_all = (counts[:, None] * t).sum(); quad_all = (counts[:, None] * t * t).sum()
    eq_lin = (lin_all - tp.sum()) / (n * (n - 1)); eq_quad = (quad_all - (tp * tp).sum()) / (n * (n - 1))
    j = ep_lin - .5 * ep_quad - eq_lin - .5 * eq_quad
    return dict(J=float(j), E_P_T=float(ep_lin), E_P_T2=float(ep_quad), E_Q_T=float(eq_lin), E_Q_T2=float(eq_quad))


def j_weighted(g, wp, wq):
    """J with explicit P/Q weights over the same (flattened) score array."""
    g = np.asarray(g, np.float64).ravel(); wp = np.asarray(wp, np.float64).ravel(); wq = np.asarray(wq, np.float64).ravel()
    t = np.tanh(g / 2)
    return float((wp * (t - .5 * t * t)).sum() - (wq * (t + .5 * t * t)).sum())


def bce_weighted(g, wp, wq):
    g = np.asarray(g, np.float64).ravel()
    return float((np.asarray(wp).ravel() * np.logaddexp(0, -g)).sum() + (np.asarray(wq).ravel() * np.logaddexp(0, g)).sum())


def sat(g):
    t = np.tanh(np.asarray(g, np.float64) / 2)
    return dict(frac_absT_ge_0p999=float((np.abs(t) >= .999).mean()), g_mean=float(np.mean(g)),
                g_p01=float(np.quantile(g, .01)), g_p50=float(np.quantile(g, .5)), g_p99=float(np.quantile(g, .99)))


# ----------------------------------------------------------------------------- common positive affine calibration

def fit_affine(g, wp, wq, criterion='balanced_logistic'):
    """Fit g' = a*g + b (a = exp(la) > 0) on FIT-only score arrays with P/Q weights. criterion:
    'balanced_logistic' (E_P softplus(-g') + E_Q softplus(g'), convex) or 'max_J' (maximise J(g'), started at
    the logistic solution). float64 LBFGS. Returns dict(a, b, value, converged)."""
    g = torch.as_tensor(np.asarray(g, np.float64).ravel())
    wp = torch.as_tensor(np.asarray(wp, np.float64).ravel()); wq = torch.as_tensor(np.asarray(wq, np.float64).ravel())
    sd = float(g.std()) or 1.0
    la0, b0 = -np.log(sd), -float(g.mean()) / sd

    def run(obj, la_init, b_init):
        la = torch.tensor(float(la_init), dtype=torch.float64, requires_grad=True)
        b = torch.tensor(float(b_init), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([la, b], lr=1, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            z = torch.exp(la) * g + b
            if obj == 'balanced_logistic':
                v = (wp * torch.nn.functional.softplus(-z)).sum() + (wq * torch.nn.functional.softplus(z)).sum()
            else:
                t = torch.tanh(z / 2)
                v = -((wp * (t - .5 * t * t)).sum() - (wq * (t + .5 * t * t)).sum())
            v.backward()
            return v
        for _ in range(5):
            opt.step(closure)
        with torch.no_grad():
            z = torch.exp(la) * g + b
            if obj == 'balanced_logistic':
                v = float((wp * torch.nn.functional.softplus(-z)).sum() + (wq * torch.nn.functional.softplus(z)).sum())
            else:
                t = torch.tanh(z / 2); v = float((wp * (t - .5 * t * t)).sum() - (wq * (t + .5 * t * t)).sum())
        return float(torch.exp(la)), float(b), v, bool(torch.isfinite(la) and torch.isfinite(b))

    a, b, v, ok = run('balanced_logistic', la0, b0)
    if criterion == 'balanced_logistic':
        return dict(a=a, b=b, value=v, finite=ok, criterion=criterion)
    a2, b2, v2, ok2 = run('max_J', np.log(a), b)
    return dict(a=a2, b=b2, value=v2, finite=ok2, criterion=criterion, start_from_logistic=dict(a=a, b=b))


def rank_invariance(s, a, b):
    """positive affine on the peer-mean score: both-direction argsort identical (checked on DEV/top lists)."""
    z = a * s + b
    return bool(np.array_equal(np.argsort(-s, axis=0, kind='stable'), np.argsort(-z, axis=0, kind='stable')) and
                np.array_equal(np.argsort(-s, axis=1, kind='stable'), np.argsort(-z, axis=1, kind='stable')))


# ----------------------------------------------------------------------------- DEV candidate structure (frozen H)

def dev_tables(nat):
    """frozen-H native DEV top-100 per query (stable argsort on the native peer-mean score), the H top-31
    non-positive list members for DEV list-J (clean annotation, evaluator only), coverage."""
    gt_mask = np.zeros_like(nat, dtype=bool); gt_mask[DEV_GT, np.arange(N_DEV_CAP)] = True
    i_top = np.argsort(-nat, axis=1, kind='stable')[:, :K_TOP]            # [n_img,100] captions
    t_top = np.argsort(-nat, axis=0, kind='stable')[:K_TOP].T.copy()      # [n_cap,100] images
    cov_i = ((i_top // PER_CAP) == np.arange(N_DEV_IMG)[:, None]).any(1)
    cov_t = (t_top == DEV_GT[:, None]).any(1)
    hard_i = np.stack([r[(r // PER_CAP) != q][:10] for q, r in enumerate(i_top)])          # [n_img,10] H hardest non-GT captions
    hard_t = np.stack([r[r != DEV_GT[c]][:10] for c, r in enumerate(t_top)])          # [n_cap,10] H hardest non-GT images
    rng = np.random.default_rng(20261121)                                              # fixed random negatives (evaluation only)
    rand_i = np.stack([rng.choice(np.setdiff1d(np.arange(N_DEV_CAP), np.arange(PER_CAP * q, PER_CAP * q + PER_CAP)), 10, replace=False) for q in range(N_DEV_IMG)])
    rand_t = np.stack([rng.choice(np.setdiff1d(np.arange(N_DEV_IMG), [DEV_GT[c]]), 10, replace=False) for c in range(N_DEV_CAP)])
    return dict(i_top=i_top, t_top=t_top, cov_i=cov_i, cov_t=cov_t, hard_i=hard_i, hard_t=hard_t, rand_i=rand_i, rand_t=rand_t)


def dev_list_positions(tab):
    """DEV lists for list-J (evaluator, clean annotation), one per DEV record k: i2t = image k//PER_CAP with caption k
    in slot 0 + the first 31 H-top captions of that image other than k; t2i = caption k with image k//PER_CAP in slot 0
    + the first 31 H-top images other than k//PER_CAP. Returned as positions into the top-100 tables plus clean A."""
    i_top, t_top = tab['i_top'], tab['t_top']
    pos_i = np.zeros((N_DEV_CAP, 31), np.int64); pos_t = np.zeros((N_DEV_CAP, 31), np.int64)
    for k in range(N_DEV_CAP):
        pos_i[k] = np.nonzero(i_top[k // PER_CAP] != k)[0][:31]
        pos_t[k] = np.nonzero(t_top[k] != k // PER_CAP)[0][:31]
    caps = np.concatenate([np.arange(N_DEV_CAP)[:, None], i_top[np.arange(N_DEV_CAP)[:, None] // PER_CAP, pos_i]], 1)
    ai = (caps // PER_CAP == (np.arange(N_DEV_CAP) // PER_CAP)[:, None]).astype(np.float64); ai /= ai.sum(1, keepdims=True)
    at = np.zeros((N_DEV_CAP, 32)); at[:, 0] = 1.0
    return pos_i, pos_t, ai, at


def dev_list_scores(gi_top, gt_top, g_gt, pos_i, pos_t):
    k = np.arange(N_DEV_CAP)
    li = np.concatenate([g_gt[:, None], gi_top[k[:, None] // PER_CAP, pos_i]], 1)
    lt = np.concatenate([g_gt[:, None], gt_top[k[:, None], pos_t]], 1)
    return li, lt


def list_j(gi, ai, gt, at):
    """bidirectional mean of the per-row list J (A/Q_C weights), float64."""
    def rows(g, a):
        t = np.tanh(np.asarray(g, np.float64) / 2)
        return (a * (t - .5 * t * t)).sum(1) - (t + .5 * t * t).mean(1)
    ji, jt = float(rows(gi, ai).mean()), float(rows(gt, at).mean())
    return dict(J=.5 * (ji + jt), J_i2t=ji, J_t2i=jt)


def list_weights(ai, at):
    """flattened P/Q weights for affine calibration of list scores (both directions, 0.5 each)."""
    ni, nt = len(ai), len(at)
    wp = np.concatenate([.5 * ai.ravel() / ni, .5 * at.ravel() / nt])
    wq = np.concatenate([np.full(ai.size, .5 / ai.size), np.full(at.size, .5 / at.size)])
    return wp, wq


def composite_metrics(nat, tab, gi_top, gt_top):
    """Two-stage retrieval: inside the frozen-H top-100 of each query, order by critic g (peer mean); outside,
    keep the H order. Implemented as a composite score matrix (top-100 entries lifted above every H score) so the
    upstream IN2R i2t/t2i program gives the ranks. gi_top [n_img,100], gt_top [n_cap,100]."""
    i_top, t_top = tab['i_top'], tab['t_top']
    off_i = float(nat.max() - gi_top.min() + 1.0); off_t = float(nat.max() - gt_top.min() + 1.0)
    Si = nat.copy(); Si[np.arange(N_DEV_IMG)[:, None], i_top] = off_i + gi_top
    St = nat.copy(); St[t_top, np.arange(N_DEV_CAP)[:, None]] = off_t + gt_top
    ri, (ranks_i, top1_i, _, _) = i2t(N_DEV_IMG, Si, PER_CAP, return_ranks=True)
    rt, (ranks_t, top1_t, _, _) = t2i(N_DEV_IMG, St, PER_CAP, return_ranks=True)
    m = dict(i2t_r1=ri[0], i2t_r5=ri[1], i2t_r10=ri[2], i2t_medr=float(ri[3]), i2t_meanr=float(ri[4]),
             t2i_r1=rt[0], t2i_r5=rt[1], t2i_r10=rt[2], t2i_medr=float(rt[3]), t2i_meanr=float(rt[4]))
    m['rsum'] = float(ri[0] + ri[1] + ri[2] + rt[0] + rt[1] + rt[2])
    tb_i = tie_bounded(Si, PER_CAP); tb_t = tie_bounded(St, PER_CAP)
    tb = dict(i2t_optimistic=tb_i['i2t_optimistic'], i2t_pessimistic=tb_i['i2t_pessimistic'],
              t2i_optimistic=tb_t['t2i_optimistic'], t2i_pessimistic=tb_t['t2i_pessimistic'],
              n_i2t_queries_gt_tied=tb_i['n_i2t_queries_gt_tied'], n_t2i_queries_gt_tied=tb_t['n_t2i_queries_gt_tied'])
    tb['rsum_optimistic'] = sum(tb[f'{d}_optimistic'][f'r{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10))
    tb['rsum_pessimistic'] = sum(tb[f'{d}_pessimistic'][f'r{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10))
    cov = dict(coverage_i2t=float(tab['cov_i'].mean()), coverage_t2i=float(tab['cov_t'].mean()),
               cond_r1_i2t_covered=100.0 * float((ranks_i[tab['cov_i']] < 1).mean()),
               cond_r1_t2i_covered=100.0 * float((ranks_t[tab['cov_t']] < 1).mean()),
               n_i2t_covered=int(tab['cov_i'].sum()), n_t2i_covered=int(tab['cov_t'].sum()))
    return m, tb, cov, dict(ranks_i=ranks_i.astype(np.int64), ranks_t=ranks_t.astype(np.int64),
                            top1_i=top1_i.astype(np.int64), top1_t=top1_t.astype(np.int64))


def full_metrics(s):
    """direct full-DEV ranking with the peer-mean critic score (A/B)."""
    ri, (ranks_i, top1_i, _, _) = i2t(N_DEV_IMG, s, PER_CAP, return_ranks=True)
    rt, (ranks_t, top1_t, _, _) = t2i(N_DEV_IMG, s, PER_CAP, return_ranks=True)
    m = dict(i2t_r1=ri[0], i2t_r5=ri[1], i2t_r10=ri[2], i2t_medr=float(ri[3]), i2t_meanr=float(ri[4]),
             t2i_r1=rt[0], t2i_r5=rt[1], t2i_r10=rt[2], t2i_medr=float(rt[3]), t2i_meanr=float(rt[4]))
    m['rsum'] = float(ri[0] + ri[1] + ri[2] + rt[0] + rt[1] + rt[2])
    return m, tie_bounded(s, PER_CAP), dict(ranks_i=ranks_i.astype(np.int64), ranks_t=ranks_t.astype(np.int64),
                                   top1_i=top1_i.astype(np.int64), top1_t=top1_t.astype(np.int64))


# ----------------------------------------------------------------------------- rank transitions / error location

def transitions(rk, rk_nat, top1, top1_nat, tab):
    """rank migration vs frozen H and error location of wrong top-1s (DEV clean annotation)."""
    out = {}
    for d in ('i', 't'):
        name = 'i2t' if d == 'i' else 't2i'
        rc, rn = rk[f'ranks_{d}'], rk_nat[f'ranks_{d}']
        diff = rc - rn
        out[f'{name}_rescued_r1'] = int(((rn >= 1) & (rc < 1)).sum()); out[f'{name}_lost_r1'] = int(((rn < 1) & (rc >= 1)).sum())
        out[f'{name}_rescued_r5'] = int(((rn >= 5) & (rc < 5)).sum()); out[f'{name}_lost_r5'] = int(((rn < 5) & (rc >= 5)).sum())
        out[f'{name}_rescued_r10'] = int(((rn >= 10) & (rc < 10)).sum()); out[f'{name}_lost_r10'] = int(((rn < 10) & (rc >= 10)).sum())
        out[f'{name}_n_improved'] = int((diff < 0).sum()); out[f'{name}_n_worsened'] = int((diff > 0).sum()); out[f'{name}_n_same'] = int((diff == 0).sum())
        out[f'{name}_mean_rank_delta'] = float(diff.mean())
        # error location: wrong top-1 inside H top-10 / top-100 of that query
        t1 = top1[f'top1_{d}']; wrong = rc >= 1
        if d == 'i':
            htop = tab['i_top']; inside10 = np.array([t1[q] in htop[q, :10] for q in range(len(t1))]); inside100 = np.array([t1[q] in htop[q] for q in range(len(t1))])
        else:
            htop = tab['t_top']; inside10 = np.array([t1[q] in htop[q, :10] for q in range(len(t1))]); inside100 = np.array([t1[q] in htop[q] for q in range(len(t1))])
        out[f'{name}_n_wrong_top1'] = int(wrong.sum())
        out[f'{name}_wrong_top1_in_Htop10'] = float(inside10[wrong].mean()) if wrong.any() else float('nan')
        out[f'{name}_wrong_top1_in_Htop100'] = float(inside100[wrong].mean()) if wrong.any() else float('nan')
        # hubs: candidates that are top-1 for >=5 queries
        cc = np.bincount(t1.astype(np.int64)); cn = np.bincount(top1_nat[f'top1_{d}'].astype(np.int64), minlength=len(cc))
        cc = np.pad(cc, (0, max(0, len(cn) - len(cc))))
        out[f'{name}_max_top1_count'] = int(cc.max()); out[f'{name}_n_hubs_ge5'] = int((cc >= 5).sum())
        out[f'{name}_n_new_hubs_ge5_vs_H'] = int(((cc >= 5) & (cn < 5)).sum())
        out[f'{name}_H_max_top1_count'] = int(cn.max()); out[f'{name}_H_n_hubs_ge5'] = int((cn >= 5).sum())
    return out


def pair_win_rates(pos_t, hard_t, rand_t, pos_i, hard_i, rand_i):
    """positive vs frozen-H hard negatives (10 hardest) and vs 10 fixed random negatives, per direction."""
    return dict(t2i_win_vs_Hhardest=float((pos_t > hard_t[:, 0]).mean()), t2i_pairacc_Hhard10=float((pos_t[:, None] > hard_t).mean()),
                t2i_pairacc_random10=float((pos_t[:, None] > rand_t).mean()),
                i2t_win_vs_Hhardest=float((pos_i > hard_i[:, 0]).mean()), i2t_pairacc_Hhard10=float((pos_i[:, None] > hard_i).mean()),
                i2t_pairacc_random10=float((pos_i[:, None] > rand_i).mean()))
