"""NR2 production interface for the package contract `proposal_reference.py` (byte copy in this package).

Pieces (all truth-free; the training process only sees FIT cache + observed.tsv + calibration + base tables):
  * build_base_topk     : fixed hard proposal lists from the frozen base FIT cosine (no labels / observed table).
  * Marginals           : p_I, p_T as observed-record occurrence frequencies (never inferred from clean IDs).
  * log_w               : exact log(q_d(y|x)/p_Y(y)), q_d = alpha p_Y + (1-alpha) Uniform(TopK_base(x)) -- the full
                          mixture probability, not the probability of the component that produced the draw.
  * ReferenceSampler    : per step and direction, Bq iid observed records (query + its observed positive) and R iid
                          proposal draws per query. Collisions (observed positive, same image, repeated IDs) are kept.
  * step_loss           : 0.5 * (i2t + t2i) mean of the package `sampled_rows` per-query losses.
Deployment ranking uses s only; nothing here is imported by the evaluator.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from .proposal_reference import sampled_rows

MODES = ('U', 'HIW', 'HREF')


def id_keys(ids):
    """Same tiebreak key as nr1.evaluate.id_tiebreak (resolves exactly equal scores only)."""
    return np.array([hashlib.sha256(x.encode()).hexdigest() for x in ids])


@torch.no_grad()
def build_base_topk(q_feat, g_feat, g_ids, k=100, chunk=2048, device='cuda', margin=28):
    """Top-k gallery rows per query under float64 base cosine, ties broken by gallery ID hash.
    Inputs: query features [Nq,d], gallery features [Ng,d], gallery IDs. No labels are accepted.
    Returns (int32 [Nq,k] gallery rows, dict of boundary statistics)."""
    keys = id_keys(g_ids)
    G = F.normalize(torch.as_tensor(np.asarray(g_feat), dtype=torch.float64, device=device), dim=1)
    out = np.empty((len(q_feat), k), dtype=np.int32)
    min_gap = np.inf; boundary_ties = 0
    kk = min(k + margin, len(g_ids))
    for s in range(0, len(q_feat), chunk):
        Q = F.normalize(torch.as_tensor(np.asarray(q_feat[s:s+chunk]), dtype=torch.float64, device=device), dim=1)
        sc = Q @ G.T
        v, ix = torch.topk(sc, kk, dim=1)
        v, ix = v.cpu().numpy(), ix.cpu().numpy()
        for r in range(len(ix)):
            o = np.lexsort((keys[ix[r]], -v[r]))
            vr, ir = v[r][o], ix[r][o]
            # the k-th kept score must be strictly above the first score outside the topk window,
            # otherwise an equal-score item outside the window could have a smaller ID key
            if not vr[k-1] > vr[-1]:
                raise RuntimeError('tie extends beyond top-k window; increase margin')
            boundary_ties += int(vr[k-1] == vr[k])
            min_gap = min(min_gap, float(vr[k-1] - vr[k]))
            out[s + r] = ir[:k]
    return out, dict(min_boundary_gap=min_gap, boundary_exact_ties=boundary_ties, k=k)


class Marginals:
    def __init__(self, im, tx, n_img, n_txt):
        self.im, self.tx = np.asarray(im, np.int64), np.asarray(tx, np.int64)
        n = len(self.im)
        self.p_img = np.bincount(self.im, minlength=n_img).astype(np.float64) / n
        self.p_txt = np.bincount(self.tx, minlength=n_txt).astype(np.float64) / n
        # observed relation (complete observed map, no truth): pair codes for collision / NCE multipositive masks
        self.n_txt = n_txt
        self.codes = np.unique(self.im * n_txt + self.tx)

    def observed(self, img_rows, txt_rows):
        c = np.asarray(img_rows, np.int64) * self.n_txt + np.asarray(txt_rows, np.int64)
        pos = np.searchsorted(self.codes, c)
        pos = np.minimum(pos, len(self.codes) - 1)
        return self.codes[pos] == c


def log_w(query_rows, cand_rows, top, p_cand, alpha):
    """Exact log(q(y|x)/p(y)) for arrays query_rows [...], cand_rows [...] (same shape). float64."""
    q = np.asarray(query_rows); y = np.asarray(cand_rows)
    if alpha == 1.0:
        return np.zeros(y.shape, dtype=np.float64)
    k = top.shape[1]
    member = (top[q.reshape(-1)] == y.reshape(-1, 1)).any(1).reshape(y.shape)
    pc = p_cand[y]
    if (pc <= 0).any():
        raise ValueError('candidate outside registered marginal support')
    w = alpha + (1 - alpha) * member / (k * pc)
    return np.log(w)


class ReferenceSampler:
    """Independent locked streams: positives per direction and reference draws per direction.
    The reference stream draws (component u, marginal record m, hard slot h) for every slot in every mode
    (coupled common random numbers); mode U uses only m (alpha=1)."""

    def __init__(self, marg: Marginals, top_i2t, top_t2i, alpha, seed, n_query=32, n_ref=63):
        if not (0 < alpha <= 1):
            raise ValueError('alpha in (0,1]')
        self.m, self.alpha, self.nq, self.nr = marg, float(alpha), int(n_query), int(n_ref)
        self.top = dict(i2t=np.asarray(top_i2t), t2i=np.asarray(top_t2i))
        if self.top['i2t'].shape[0] != len(marg.p_img) or self.top['t2i'].shape[0] != len(marg.p_txt):
            raise ValueError('proposal tables must cover every query row')
        self.rng = {k: np.random.default_rng([int(seed), i]) for i, k in
                    enumerate(['pos_i2t', 'pos_t2i', 'ref_i2t', 'ref_t2i'], start=1)}

    def state(self):
        return {k: g.bit_generator.state for k, g in self.rng.items()}

    def set_state(self, st):
        for k, g in self.rng.items():
            g.bit_generator.state = st[k]

    def _draw_ref(self, direction, qrows):
        g = self.rng['ref_' + direction]; n = len(self.m.im)
        u = g.random((len(qrows), self.nr))
        mrec = g.integers(0, n, size=(len(qrows), self.nr))
        h = g.integers(0, self.top[direction].shape[1], size=(len(qrows), self.nr))
        marginal = (self.m.tx if direction == 'i2t' else self.m.im)[mrec]
        hard = self.top[direction][qrows[:, None], h]
        return np.where(u < self.alpha, marginal, hard)

    def draw(self):
        """Returns dict per direction: query rows [Bq], candidate rows [Bq,1+R] (slot 0 = observed positive),
        log_w [Bq,1+R] (float64, exact mixture probability; zeros when alpha=1), observed mask [Bq,1+R],
        hard-member flag [Bq,1+R]."""
        out = {}
        n = len(self.m.im)
        for d in ('i2t', 't2i'):
            r = self.rng['pos_' + d].integers(0, n, size=self.nq)
            if d == 'i2t':
                qrows, pos = self.m.im[r], self.m.tx[r]
            else:
                qrows, pos = self.m.tx[r], self.m.im[r]
            ref = self._draw_ref(d, qrows)
            cand = np.concatenate([pos[:, None], ref], 1)
            pc = self.m.p_txt if d == 'i2t' else self.m.p_img
            lw = log_w(np.repeat(qrows[:, None], cand.shape[1], 1), cand, self.top[d], pc, self.alpha)
            member = (self.top[d][qrows][:, None, :] == cand[:, :, None]).any(-1)
            if d == 'i2t':
                obs = self.m.observed(np.repeat(qrows[:, None], cand.shape[1], 1), cand)
            else:
                obs = self.m.observed(cand, np.repeat(qrows[:, None], cand.shape[1], 1))
            if not obs[:, 0].all():
                raise RuntimeError('designated P slot is not an observed pair')
            out[d] = dict(records=r, query=qrows, cand=cand, log_w=lw, observed=obs, hard=member)
        return out


def pair_scores(model, X, Y, batch):
    """s for every (query, slot) pair, both directions; X,Y = full FIT global features (device tensors)."""
    out = {}
    for d, b in batch.items():
        q = torch.as_tensor(b['query'], device=X.device)
        c = torch.as_tensor(b['cand'], device=X.device)
        qq = q[:, None].expand_as(c).reshape(-1)
        cc = c.reshape(-1)
        if d == 'i2t':
            s = model.pairs(X[qq], Y[cc])
        else:
            s = model.pairs(X[cc], Y[qq])
        out[d] = s.reshape(c.shape)
    return out


def step_loss(scores, batch, objective, mode):
    tot = 0.
    for d in ('i2t', 't2i'):
        s = scores[d]
        lw = torch.as_tensor(batch[d]['log_w'], device=s.device, dtype=s.dtype)
        if mode == 'U':
            lw = torch.zeros_like(s)
        mask = torch.as_tensor(batch[d]['observed'], device=s.device) if objective == 'NCE' else None
        tot = tot + .5 * sampled_rows(s, lw, objective, mode, mask).mean()
    return tot


def own_q_j(scores, batch, mode):
    """Training-batch J of each direction under the unit's own reference (U/HIW: estimator of the original
    product-Q J with s; HREF: J on g=s-log w under Q_d)."""
    r = {}
    for d in ('i2t', 't2i'):
        s = scores[d].detach().double()
        lw = torch.as_tensor(batch[d]['log_w'], device=s.device, dtype=s.dtype)
        z = s - lw if mode == 'HREF' else s
        t = torch.tanh(z / 2)
        wq = torch.exp(-lw[:, 1:]) if mode == 'HIW' else torch.ones_like(t[:, 1:])
        r[d] = float((t[:, 0] - t[:, 0] ** 2 / 2).mean() - (wq * (t[:, 1:] + t[:, 1:] ** 2 / 2)).mean())
    return r
