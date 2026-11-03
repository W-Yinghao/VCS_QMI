"""Upstream IN2R evaluation.py i2t/t2i, copied verbatim (code_r30v11/evaluation.py, sha 33032d54...),
plus deterministic tie-bounded recalls (all ties ahead / no tie ahead). Ranking object = score matrix
sims[images, captions] (higher = better); here the peer-averaged critic g or native cosine."""
from __future__ import annotations
import numpy as np


def i2t(npts, sims, per_captions=1, return_ranks=False):
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, per_captions=1, return_ranks=False):
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


# --------------------------------------------------------------------------- wrappers (not upstream)

def upstream_metrics(sims, per_captions=5):
    """Native readout: the six recalls of upstream i2t/t2i and their sum (as in R25 validate)."""
    npts = sims.shape[0]
    r, (ri_ranks, *_ ) = i2t(npts, sims, per_captions, return_ranks=True)
    rt, (rt_ranks, *_ ) = t2i(npts, sims, per_captions, return_ranks=True)
    out = dict(i2t_r1=r[0], i2t_r5=r[1], i2t_r10=r[2], i2t_medr=float(r[3]), i2t_meanr=float(r[4]),
               t2i_r1=rt[0], t2i_r5=rt[1], t2i_r10=rt[2], t2i_medr=float(rt[3]), t2i_meanr=float(rt[4]))
    out['rsum'] = float(r[0] + r[1] + r[2] + rt[0] + rt[1] + rt[2])
    return out, ri_ranks.astype(np.int64), rt_ranks.astype(np.int64)


def tie_bounded(sims, per_captions=5):
    """0-based ranks with ties all ahead (pessimistic) and no tie ahead (optimistic); exact-equality ties.
    i2t: best over the per_captions GT captions. Also counts queries whose GT score is tied with any candidate."""
    s = np.asarray(sims)
    n_img, n_cap = s.shape
    gt_img = np.arange(n_cap) // per_captions
    # t2i (column queries)
    pos = s[gt_img, np.arange(n_cap)]
    greater = (s > pos[None, :]).sum(0)
    equal = (s == pos[None, :]).sum(0) - 1
    t_opt, t_pes = greater, greater + equal
    # i2t (row queries)
    i_opt = np.full(n_img, np.iinfo(np.int64).max); i_pes = i_opt.copy(); i_tied = np.zeros(n_img, bool)
    for k in range(per_captions):
        cols = np.arange(n_img) * per_captions + k
        p = s[np.arange(n_img), cols]
        g = (s > p[:, None]).sum(1)
        e = (s == p[:, None]).sum(1) - 1
        i_opt = np.minimum(i_opt, g); i_pes = np.minimum(i_pes, g + e); i_tied |= e > 0
    def rec(r):
        return {f'r{k}': 100.0 * float((r < k).mean()) for k in (1, 5, 10)}
    res = {}
    for name, ro, rp in (('i2t', i_opt, i_pes), ('t2i', t_opt, t_pes)):
        res[f'{name}_optimistic'] = rec(ro); res[f'{name}_pessimistic'] = rec(rp)
    res['rsum_optimistic'] = sum(res[f'{d}_optimistic'][f'r{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10))
    res['rsum_pessimistic'] = sum(res[f'{d}_pessimistic'][f'r{k}'] for d in ('i2t', 't2i') for k in (1, 5, 10))
    res['n_t2i_queries_gt_tied'] = int((equal > 0).sum()); res['n_i2t_queries_gt_tied'] = int(i_tied.sum())
    res['any_exact_tie_at_gt'] = bool(res['n_t2i_queries_gt_tied'] or res['n_i2t_queries_gt_tied'])
    return res
