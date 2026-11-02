"""R37C unit trainer = R37 train_unit.py ported to CC152K (only dataset constants: PER_CAP = 1 / identity instead of //5 in the
DEV readout and the step-0 tie rule, r37c code hashing, G units only). R37 unit trainer = R36 train_unit.py (copied; R37 changes: per-parent conditions, TRUST_NATIVE_NULL (B1) and
N_PRESENT_ONLY (B2) supervision, r37 code hashing; no oracle units in R37). R36 unit trainer. Non-oracle process (train_unit.py): M-RANDOM, T (TRUST_HARD), N (OBS_MISSING_NULL), V (C-L
clean replication, 8000 + 4000 HEAD_CONTROL continuation). Oracle process (train_oracle.py -> main(oracle=module)):
M-SHIELD_ORACLE / M-RELABEL_AVAILABLE_ORACLE only, outputs under OUT/oracle_runs. Every unit starts from a FRESH
head (R35 build_heads: common FIT-only calibration a0 s + b0, zero-output residual) on the frozen H features; no
C/D/A/B head is ever loaded (V phase 2 loads only its own phase-1 step-8000 head). Local encoders stay frozen.
Readouts, candidate tables, calibration, scoring and two-stage DEV evaluation are the R35 code (copied verbatim).
Usage: python -B train_unit.py --units id1,id2,...   (units in order; DONE skipped; resumable)
DEV only enters no-grad readouts. TEST files are never opened (audit guard)."""
from __future__ import annotations
import argparse, glob, hashlib, os, re, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, RUNS, ORACLE_RUNS, PEERS, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, PER_CAP, install_guards, guard_record, assert_no_oracle_loaded,
                    sha256, write_json, read_json, utc, state_hash, EVAL_ONLY_MODULES)
from heads import build_heads, n_params, NullSlots
from objectives import loss_list, list_rows, loss_list_pq, j_parts, loss_list_masked
import evals as EV
from data35 import load_cond, shard_mm, build_lists, load_local_encoders, load_raw, N_SLOTS
import data36 as D36

# ============================================================================ configuration / context


def cfg_of(uid):
    return read_json(ROOT / 'configs' / f'{uid}.json')


def code_hashes():
    return {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r37c' / '*.py')))}


class Ctx:
    """per-condition data (+ local encoders/raw features for L units), loaded once per process."""
    def __init__(self, cond, device, need_local, fit_maps=None):
        self.cond, self.dev = cond, device
        self.d = load_cond(cond, device)
        self.fit_maps = fit_maps or {'observed': self.d['obs_np']}
        tab = self.d['dev_tab']
        self.pos_i, self.pos_t, self.ai_dev, self.at_dev = EV.dev_list_positions(tab)
        self.need_local = need_local
        assert not need_local, 'R37C: G units only'
        if need_local:
            self.raw = load_raw(device)
            self.enc0, self.enc_info = load_local_encoders(cond, device)
        nat, _, rk = EV.full_metrics(self.d['native_dev'])
        self.nat_metrics, self.nat_rk = nat, rk
        self.present = D36.presence_probs(self.d)   # R37: fixed p_d per direction (N_PRESENT_ONLY normalisation)


def set_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================ scoring (A/B: records, full DEV)
# ============================================================================ scoring (C/D: candidate lists, top-100 rerank)

def project_items(h, enc, raw, split, img_ids, cap_ids, grad=False, enc_grad=False, chunk=512):
    """local projections of unique images/captions: V [nu,36,128], W [nc,Lb,128], Wmask, and inverse maps."""
    ui, inv_i = np.unique(img_ids, return_inverse=True); uc, inv_c = np.unique(cap_ids, return_inverse=True)
    rs = 'train' if split == 'fit' else 'dev'
    ims = raw[f'{rs}_ims']; tok = raw[f'{rs}_tok']; ln = raw[f'{rs}_len']
    dev = ims.device
    ui_t = torch.as_tensor(ui, device=dev); uc_t = torch.as_tensor(uc, device=dev)
    lb = int(ln[uc_t].max())
    ctx_g = torch.enable_grad() if grad else torch.no_grad()
    V, W, M = [], [], []
    with ctx_g:
        for i in range(0, len(ui), chunk):
            with (torch.enable_grad() if enc_grad else torch.no_grad()):
                r = enc.regions(ims[ui_t[i:i + chunk]])
            V.append(h.local.proj_regions(r))
        for i in range(0, len(uc), chunk):
            cc = uc_t[i:i + chunk]
            with (torch.enable_grad() if enc_grad else torch.no_grad()):
                w, m = enc.tokens(tok[cc], ln[cc])
            w = h.local.proj_words(w)
            if w.shape[1] < lb:
                w = torch.nn.functional.pad(w, (0, 0, 0, lb - w.shape[1])); m = torch.nn.functional.pad(m, (0, lb - m.shape[1]))
            W.append(w); M.append(m)
    return torch.cat(V), torch.cat(W), torch.cat(M), torch.as_tensor(inv_i, device=dev), torch.as_tensor(inv_c, device=dev)


def g_pairs_lists(h, enc, ctx, p, split, img_idx, cap_idx, grad=False, enc_grad=False, s_exact=None, chunk=8192, proj=None):
    """per-peer scores of explicit (image, caption) pairs. s_exact (fp64 array) replaces the fp32 skip term in
    the affine part (evaluation: exact native shard score)."""
    d = ctx.d
    X, Y = d[(split, 'img', p)], d[(split, 'cap', p)]
    dev = X.device
    img_idx = np.asarray(img_idx); cap_idx = np.asarray(cap_idx)
    if h.local is not None:
        V, W, M, inv_i, inv_c = proj if proj is not None else project_items(h, enc, ctx.raw, split, img_idx, cap_idx, grad, enc_grad)
        rmask_full = torch.ones(V.shape[0], V.shape[1], dtype=torch.bool, device=dev)
    outs = []
    for i in range(0, len(img_idx), chunk):
        ii = torch.as_tensor(img_idx[i:i + chunk], device=dev); cc = torch.as_tensor(cap_idx[i:i + chunk], device=dev)
        x, y = X[ii], Y[cc]; s = (x * y).sum(-1)
        lf = None
        if h.local is not None:
            vi, ci = inv_i[i:i + chunk], inv_c[i:i + chunk]
            lmax = int(M[ci].sum(1).max())
            lf = h.local.pair_features(V[vi], W[ci][:, :lmax], rmask_full[vi], M[ci][:, :lmax])
        hid = h.glob.hidden(x, y, s)
        if lf is not None:
            hid = torch.cat([hid, lf], -1)
        r = h.out(hid).squeeze(-1)
        if s_exact is None:
            outs.append(h.affine(s) + r)
        else:
            a, b = (float(t) for t in h.affine.ab())
            outs.append(a * torch.as_tensor(s_exact[i:i + chunk], device=dev) + b + r.double())
    return torch.cat(outs)


def dev_pairs(tab):
    i_top, t_top = tab['i_top'], tab['t_top']
    parts = dict(i_top=(np.repeat(np.arange(N_DEV_IMG), i_top.shape[1]), i_top.ravel()),
                 t_top=(t_top.ravel(), np.repeat(np.arange(N_DEV_CAP), t_top.shape[1])),
                 gt=(np.arange(N_DEV_CAP) // PER_CAP, np.arange(N_DEV_CAP)),
                 rand_i=(np.repeat(np.arange(N_DEV_IMG), 10), tab['rand_i'].ravel()),
                 rand_t=(tab['rand_t'].ravel(), np.repeat(np.arange(N_DEV_CAP), 10)))
    img = np.concatenate([v[0] for v in parts.values()]); cap = np.concatenate([v[1] for v in parts.values()])
    sizes = [len(v[0]) for v in parts.values()]
    return img, cap, list(parts), sizes


def lists_from_locks(ctx, key_i, key_t):
    L, d = ctx.d['locks'], ctx.d
    return build_lists(L[key_i], L[key_t], d['obs_np'], d['cand_i2t'], d['cand_t2i'], L[key_i + '_perm'], L[key_t + '_perm'])


def list_pairs(lists):
    (qi, caps, ai), (qc, imgs, at) = lists
    img = np.concatenate([np.repeat(qi, N_SLOTS), imgs.ravel()]); cap = np.concatenate([caps.ravel(), np.repeat(qc, N_SLOTS)])
    return img, cap, ai, at


def readout_lists(heads, encs, ctx, final=False):
    """C/D readout: two-stage DEV retrieval (H top-100 reranked by the peer-mean critic g, rest in H order) on the
    full query set, coverage + conditional R@1, pair win rates, DEV list-J (clean annotation) and FIT list-J on a
    locked FIT list set (observed), raw and with a common FIT-only positive affine calibration."""
    d = ctx.d; tab = d['dev_tab']
    t0 = time.time()
    heads.eval()
    img, cap, names, sizes = dev_pairs(tab)
    with torch.no_grad():
        gsum = 0
        for p in PEERS:
            h = heads[p]
            proj = project_items(h, encs[p], ctx.raw, 'dev', img, cap) if h.local is not None else None
            s_ex = d['dev_s'][p][torch.as_tensor(img, device=ctx.dev), torch.as_tensor(cap, device=ctx.dev)].cpu().numpy()
            gsum = gsum + g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'dev', img, cap, s_exact=s_ex, proj=proj)
        g = (gsum / 2).cpu().numpy()
    parts = dict(zip(names, np.split(g, np.cumsum(sizes)[:-1])))
    gi_top = parts['i_top'].reshape(N_DEV_IMG, -1); gt_top = parts['t_top'].reshape(N_DEV_CAP, -1)
    m, tb, cov, rk = EV.composite_metrics(d['native_dev'], tab, gi_top, gt_top)
    rec = dict(dev_retrieval=m, dev_retrieval_tie_bounded=tb, candidate_coverage=cov)
    g_gt = parts['gt']
    pos_i = g_gt.reshape(N_DEV_IMG, PER_CAP).max(1)
    hard_i = np.take_along_axis(gi_top, np.stack([np.nonzero((r // PER_CAP) != q)[0][:10] for q, r in enumerate(tab['i_top'])]), 1)
    hard_t = np.take_along_axis(gt_top, np.stack([np.nonzero(r != c // PER_CAP)[0][:10] for c, r in enumerate(tab['t_top'])]), 1)
    rec['pair_win_rates'] = EV.pair_win_rates(g_gt, hard_t, parts['rand_t'].reshape(N_DEV_CAP, 10), pos_i, hard_i,
                                              parts['rand_i'].reshape(N_DEV_IMG, 10))
    li, lt = EV.dev_list_scores(gi_top, gt_top, g_gt, ctx.pos_i, ctx.pos_t)
    rec['dev_list_clean_J_raw'] = EV.list_j(li, ctx.ai_dev, lt, ctx.at_dev)
    # FIT locked lists: eval (raw J) and calibration (affine fit)
    def fit_lists(ki, kt):
        lists = lists_from_locks(ctx, ki, kt)
        im, cp, ai, at = list_pairs(lists)
        with torch.no_grad():
            gs = 0
            for p in PEERS:
                gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', im, cp).double()
        gg = (gs / 2).cpu().numpy(); n = len(ai)
        return gg[:n * N_SLOTS].reshape(n, N_SLOTS), gg[n * N_SLOTS:].reshape(n, N_SLOTS), ai, at
    ei, et, eai, eat = fit_lists('lev_i', 'lev_t')
    rec['fit_list_observed_J_raw'] = EV.list_j(ei, eai, et, eat)
    ci, ct, cai, cat = fit_lists('lcal_i', 'lcal_t')
    wp, wq = EV.list_weights(cai, cat)
    cal = {}
    for crit in ('balanced_logistic', 'max_J'):
        f = EV.fit_affine(np.concatenate([ci.ravel(), ct.ravel()]), wp, wq, crit)
        cal[crit] = dict(a=f['a'], b=f['b'], fit_value=f['value'],
                         fit_list_observed_J_eval=EV.list_j(f['a'] * ei + f['b'], eai, f['a'] * et + f['b'], eat)['J'],
                         dev_list_clean_J=EV.list_j(f['a'] * li + f['b'], ctx.ai_dev, f['a'] * lt + f['b'], ctx.at_dev)['J'],
                         dev_rank_invariant=bool(EV.rank_invariance(gi_top, f['a'], f['b']) and EV.rank_invariance(gt_top, f['a'], f['b'])) if final else None)
    rec['calibration_affine_fit_only'] = cal
    rec['T_saturation_dev_top100'] = EV.sat(np.concatenate([gi_top.ravel(), gt_top.ravel()[::PER_CAP]]))
    rec['heads_ab'] = {p: [float(t) for t in heads[p].affine.ab()] for p in PEERS}
    rec['readout_seconds'] = time.time() - t0
    return rec, dict(gi_top=gi_top, gt_top=gt_top, g_gt=g_gt), rk


# ============================================================================ training

def _copy_enc(e):
    import copy
    c = copy.deepcopy(e); c.set_modes(); return c


class _shallow:
    def __init__(self, ctx, **kw):
        self.__dict__.update(ctx.__dict__); self.__dict__.update(kw)

# ============================================================================ R36 readout additions

def _spearman_rows(a, b):
    ra = a.argsort(1).argsort(1).astype(np.float64); rb = b.argsort(1).argsort(1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return (ra * rb).sum(1) / np.where(den > 0, den, np.nan)


@torch.no_grad()
def fit_pair_scores(heads, encs, ctx, img, cap):
    gs, ss = 0, 0
    for p in PEERS:
        gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', img, cap).double()
        x = ctx.d[('fit', 'img', p)][torch.as_tensor(img, device=ctx.dev)]; y = ctx.d[('fit', 'cap', p)][torch.as_tensor(cap, device=ctx.dev)]
        ss = ss + (x * y).sum(-1).double()
    return (gs / 2).cpu().numpy(), (ss / 2).cpu().numpy()


def readout_r36(heads, nulls, encs, ctx, final=False):
    """R35 readout_lists (unchanged) + R36 additions on the locked FIT records: Spearman(g, native) and within-row
    g spread on the locked forced lists; for N units the null-list J (33 slots, Q = Uniform(33)) split into its
    real-slot and null-slot parts, null-label fraction and the null parameters. All FIT, observed labels only."""
    rec, arrs, rk = readout_lists(heads, encs, ctx, final=final)
    t0 = time.time()
    (qi, caps, ai), (qc, imgs, at) = lists_from_locks(ctx, 'lev_i', 'lev_t')
    im, cp, _, _ = list_pairs(((qi, caps, ai), (qc, imgs, at)))
    g, s = fit_pair_scores(heads, encs, ctx, im, cp)
    n = len(ai) * N_SLOTS
    G = np.concatenate([g[:n].reshape(-1, N_SLOTS), g[n:].reshape(-1, N_SLOTS)]); S = np.concatenate([s[:n].reshape(-1, N_SLOTS), s[n:].reshape(-1, N_SLOTS)])
    sp = _spearman_rows(G, S)
    rec['fit_forced_lists_score_structure'] = dict(spearman_g_vs_native_row_mean=float(np.nanmean(sp)), frac_rows_spearman_negative=float(np.nanmean(sp < 0)),
                                                   g_std_within_row_mean=float(G.std(1).mean()), native_std_within_row_mean=float(S.std(1).mean()))
    if nulls is not None:
        (qi, caps, ai), (qc, imgs, at) = D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t')
        out = {}
        for dn, di, ids, q, a in (('i2t', 0, caps, qi, ai), ('t2i', 1, imgs, qc, at)):
            real = ids != D36.NULL_ID
            if dn == 'i2t':
                img_, cap_ = np.repeat(q, D36.K_NULL).reshape(ids.shape)[real], ids[real]
            else:
                img_, cap_ = ids[real], np.repeat(q, D36.K_NULL).reshape(ids.shape)[real]
            gr, _ = fit_pair_scores(heads, encs, ctx, img_, cap_)
            gg = np.zeros(ids.shape); gg[real] = gr
            gg[~real] = float(np.mean([float(nulls.v[p][di]) for p in PEERS]))
            t = np.tanh(gg / 2); K = gg.shape[1]
            part = a * (t - .5 * t * t) - (t + .5 * t * t) / K
            out[dn] = dict(J_total=float(part.sum(1).mean()), J_real_part=float((part * real).sum(1).mean()), J_null_part=float((part * ~real).sum(1).mean()),
                           null_label_fraction=float(a[~real].mean()), observed_pos_real_slots_mean=float(((a > 0) & real).sum(1).mean()),
                           null_rank_mean=float((gg > gg[~real][:, None]).sum(1).mean()))
        out['J_total_bidirectional'] = .5 * (out['i2t']['J_total'] + out['t2i']['J_total'])
        out['null_params'] = {p: [float(x) for x in nulls.v[p].detach().cpu()] for p in PEERS}
        rec['fit_null_lists_observed_J_raw'] = out
    rec['readout_seconds'] += time.time() - t0
    return rec, arrs, rk


# ============================================================================ training

def make_opt(cfg, heads, nulls):
    o = cfg['optimizer']
    ps = list(heads.parameters()) + (list(nulls.parameters()) if nulls is not None else [])
    return torch.optim.AdamW([dict(params=ps, lr=o['lr_head'])], betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'])


def peer_params(heads, nulls, p):
    return list(heads[p].parameters()) + ([nulls.v[p]] if nulls is not None else [])


def _pairs(lists):
    """flatten real (image, caption) pairs: i2t rows then t2i rows; returns masks of real slots per direction."""
    (qi, caps, ai), (qc, imgs, at) = lists
    ri, rt = caps != D36.NULL_ID, imgs != D36.NULL_ID
    img = np.concatenate([np.repeat(qi, caps.shape[1]).reshape(caps.shape)[ri], imgs[rt]])
    cap = np.concatenate([caps[ri], np.repeat(qc, imgs.shape[1]).reshape(imgs.shape)[rt]])
    return img, cap, ri, rt


def step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle):
    d = ctx.d; sup = cfg['supervision']
    lists, idx = D36.sample(sup, rng, rng2, d, cfg['n_queries'], pool)
    (qi, caps, ai), (qc, imgs, at) = lists
    stats = {}
    q_i = q_t = v_i = v_t = None
    if cfg['oracle']:
        (oi, ot), cnt = oracle.apply(sup, lists)
        ai, q_i, v_i = oi; at, q_t, v_t = ot
        stats.update({f'oracle_{k}': v for k, v in cnt.items()})
    img, cap, ri, rt = _pairs(lists)
    A_i = torch.as_tensor(ai, device=ctx.dev); A_t = torch.as_tensor(at, device=ctx.dev)
    shield = sup == 'SHIELD_ORACLE'
    if shield:
        Qi, Qt = torch.as_tensor(q_i, device=ctx.dev), torch.as_tensor(q_t, device=ctx.dev)
        Vi, Vt = torch.as_tensor(v_i, device=ctx.dev), torch.as_tensor(v_t, device=ctx.dev)
    n_i = int(ri.sum())
    ri_t, rt_t = torch.as_tensor(ri, device=ctx.dev), torch.as_tensor(rt, device=ctx.dev)
    present_only = sup == 'N_PRESENT_ONLY'
    if present_only:
        mi, mt = D36.valid_rows(lists)
        Mi = torch.as_tensor(mi, device=ctx.dev, dtype=torch.float32); Mt = torch.as_tensor(mt, device=ctx.dev, dtype=torch.float32)
        stats.update(valid_rows_i2t=float(mi.sum()), valid_rows_t2i=float(mt.sum()), empty_valid_batch_i2t=float(mi.sum() == 0),
                     empty_valid_batch_t2i=float(mt.sum() == 0))
    loss, jb, gl = 0.0, 0.0, []
    for p in PEERS:
        h = heads[p]
        proj = project_items(h, encs[p], ctx.raw, 'fit', img, cap, grad=True, enc_grad=False) if h.local is not None else None
        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=False, proj=proj, chunk=1 << 30)
        if nulls is None:
            gi, gt = g[:n_i].reshape(caps.shape), g[n_i:].reshape(imgs.shape)
        else:
            gi = torch.zeros(caps.shape, device=ctx.dev, dtype=g.dtype); gt = torch.zeros(imgs.shape, device=ctx.dev, dtype=g.dtype)
            gi = gi.masked_scatter(ri_t, g[:n_i]); gt = gt.masked_scatter(rt_t, g[n_i:])
            gi = torch.where(ri_t, gi, nulls.score(p, 0, 1)[0]); gt = torch.where(rt_t, gt, nulls.score(p, 1, 1)[0])
        gi.retain_grad(); gt.retain_grad(); gl.append((gi, gt))
        if shield:
            li, lt = loss_list_pq(cfg['objective'], gi, A_i, Qi, Vi), loss_list_pq(cfg['objective'], gt, A_t, Qt, Vt)
        elif present_only:
            li = loss_list_masked(cfg['objective'], gi, A_i, Mi, ctx.present['p_i2t'])
            lt = loss_list_masked(cfg['objective'], gt, A_t, Mt, ctx.present['p_t2i'])
        else:
            li, lt = loss_list(cfg['objective'], gi, A_i), loss_list(cfg['objective'], gt, A_t)
        loss = loss + .5 * (.5 * li + .5 * lt)
        with torch.no_grad():
            jb += .5 * float(.5 * list_rows(gi, A_i, 'J').mean() + .5 * list_rows(gt, A_t, 'J').mean())
    stats.update(n_pos_slots_i2t_rows_sum=float((ai > 0).sum()), n_unique_images=int(len(np.unique(img))), n_unique_caps=int(len(np.unique(cap))))
    if nulls is not None:
        stats.update(null_label_rows_i2t=float(ai[~ri].sum()), null_label_rows_t2i=float(at[~rt].sum()),
                     observed_pos_real_slots_i2t=float(((ai > 0) & ri).sum()), observed_pos_real_slots_t2i=float(((at > 0) & rt).sum()),
                     effective_pos_rows_i2t=float(((ai > 0) & ri).any(1).sum()), effective_pos_rows_t2i=float(((at > 0) & rt).any(1).sum()))
    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt)
    return loss, jb, (img, cap, idx[2], idx[3]), stats, gl, ctxs


def grad_class_stats(gl, c):
    """mean dL/dg per slot class (real A>0, real A=0, null), averaged over peers and directions (window stat)."""
    out = {}
    for dn, k, a, r in (('i2t', 0, c['ai'], c['ri']), ('t2i', 1, c['at'], c['rt'])):
        G = np.mean([gl[j][k].grad.detach().double().cpu().numpy() for j in range(len(gl))], 0)
        for cl, m in (('real_Apos', (a > 0) & r), ('real_A0', (a == 0) & r), ('null', ~r)):
            if m.any():
                out[f'dLdg_{dn}_{cl}'] = float(G[m].mean())
    return out


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state, torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def save_ckpt(path, cfg, step, heads, nulls, opt, rng, rng2, lineage):
    obj = dict(step=step, unit_id=cfg['unit_id'], phase=cfg.get('phase', 1), heads=heads.state_dict(), nulls=nulls.state_dict() if nulls is not None else None,
               optimizer=opt.state_dict(), rng=rng_state(rng), rng2=rng2.bit_generator.state if rng2 is not None else None, lineage=lineage, config=cfg)
    tmp = str(path) + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)


def unit_out(cfg, gate_out=None):
    base = Path(gate_out) / cfg['unit_id'] if gate_out else (ORACLE_RUNS if cfg['oracle'] else RUNS) / cfg['unit_id']
    return base / f"phase{cfg['phase']}" if cfg['block'] == 'V' else base


def run_unit(cfg, ctx, pool=None, out_dir=None, steps=None, ckpt_steps=None, log=print, dev_nan=False, stop_at=None, no_readout=False, oracle=None):
    """Train one unit (one V phase) to `steps` with readouts + checkpoints at ckpt_steps; resumes from the latest
    checkpoint in out_dir. V phase 2: loads its own phase-1 step-8000 head (+ optimizer RESET, new sampler stream)."""
    steps = cfg['steps'] if steps is None else steps
    ckpt_steps = sorted(set(cfg['ckpt_steps'] if ckpt_steps is None else ckpt_steps) | {steps})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if cfg['oracle']:
        assert oracle is not None, 'oracle unit needs the oracle runner'
    else:
        assert oracle is None and 'oracle_runs' not in str(out), 'non-oracle unit must not use oracle outputs'
    dev = ctx.dev; d = ctx.d
    if dev_nan:  # isolation probe: replace DEV caches by NaN; training must be bitwise unchanged
        d = dict(d)
        for p in PEERS:
            for k in ('img', 'cap'):
                d[('dev', k, p)] = torch.full_like(d[('dev', k, p)], float('nan'))
        raw = dict(ctx.raw) if ctx.need_local else None
        if raw is not None:
            raw['dev_ims'] = torch.full_like(raw['dev_ims'], float('nan'))
        ctx = _shallow(ctx, d=d, raw=raw)
    a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
    heads = build_heads(cfg['structure'], a0, b0, cfg['init_seed']).to(dev)
    nulls = NullSlots().to(dev) if cfg['supervision'] in D36.NULL_SUPS else None
    encs = None
    lineage = dict(cache_manifest_sha256=d['manifest_sha256'], prep_manifest_sha256=d['prep_manifest_sha256'], candidate_table_sha256=d.get('cand_sha256'),
                   trust_sha256=d.get('trust_sha256') if cfg['supervision'] in D36.TRUST_SUPS else None, present_probs=ctx.present if cfg['supervision'] == 'N_PRESENT_ONLY' else None,
                   r35_prep_manifest_sha256=d.get('r35_prep_manifest_sha256'), cond=cfg['cond'], a0b0=d['a0b0'], parent_H=d['manifest']['parent'],
                   code_sha256=code_hashes(), config_sha256=cfg.get('_config_sha256'), head_init='fresh build_heads(%s, a0, b0, %d)' % (cfg['structure'], cfg['init_seed']))
    if ctx.need_local:
        encs = torch.nn.ModuleDict({p: _copy_enc(ctx.enc0[p]) for p in PEERS})
        for p in PEERS:
            for q in encs[p].parameters():
                q.requires_grad_(False)          # R36: local encoders frozen in every unit
        lineage['local_encoder_source'] = ctx.enc_info
    boundary = None
    if cfg.get('parent_ckpt'):                  # V phase 2 only: own phase-1 head at step 8000
        assert cfg['block'] == 'V' and cfg['phase'] == 2 and 'oracle_runs' not in cfg['parent_ckpt']
        par = torch.load(cfg['parent_ckpt'], map_location=dev, weights_only=False)
        assert par['step'] == cfg['parent_step'] and par['unit_id'] == cfg['unit_id'] and par['phase'] == 1
        heads.load_state_dict(par['heads'])
        lineage['parent_ckpt'] = dict(path=cfg['parent_ckpt'], sha256=sha256(cfg['parent_ckpt']), step=par['step'])
        boundary = dict(parent_ckpt=cfg['parent_ckpt'], parent_sha256=lineage['parent_ckpt']['sha256'], parent_step=par['step'],
                        parent_head_state_sha256=state_hash(heads), optimizer='fresh AdamW (parent optimizer state NOT loaded)',
                        sampler_seed=cfg['sampler_seed'], parent_sampler_seed=par['config']['sampler_seed'])
    init_hash = state_hash(heads) + (state_hash(nulls) if nulls is not None else '')
    opt = make_opt(cfg, heads, nulls)
    if boundary is not None:
        boundary['optimizer_state_entries_at_reset'] = len(opt.state_dict()['state'])
    rng = np.random.default_rng(cfg['sampler_seed'])
    rng2 = np.random.default_rng(cfg['sampler_seed2']) if cfg.get('sampler_seed2') else None
    start = 0
    done_ck = sorted(int(re.search(r'ckpt_step(\d+)\.pt$', f).group(1)) for f in glob.glob(str(out / 'ckpt_step*.pt')))
    done_ck = [k for k in done_ck if (out / f'eval_step{k}.json').exists() and k <= steps]
    if done_ck:
        k = done_ck[-1]
        ck = torch.load(out / f'ckpt_step{k}.pt', map_location=dev, weights_only=False)
        heads.load_state_dict(ck['heads']); opt.load_state_dict(ck['optimizer'])
        if nulls is not None:
            nulls.load_state_dict(ck['nulls'])
        # RNG states must be CPU ByteTensors; torch.load(map_location=dev) moves them to the GPU (R35 gate 1081274 resume TypeError)
        rng.bit_generator.state = ck['rng']['numpy']; torch.set_rng_state(ck['rng']['torch_cpu'].cpu())
        if ck['rng']['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng']['torch_cuda']])
        if rng2 is not None:
            rng2.bit_generator.state = ck['rng2']
        start = k
        log(f"resume {cfg['unit_id']} phase {cfg.get('phase', 1)} from step {k}")
    if boundary is not None:
        write_json(out / 'PHASE_BOUNDARY_RECEIPT.json', dict(boundary, utc=utc(), resumed_from_step=start))
    hist = read_json(out / 'TRAIN_HISTORY.json') if (start and (out / 'TRAIN_HISTORY.json').exists()) else dict(windows=[], train_seconds=0.0, eval_seconds=0.0, batch_hash=[], stats_sum={})
    hist['windows'] = [w for w in hist['windows'] if w['end_step'] <= start]
    win = []; gwin = []; t_train = 0.0; t_eval = 0.0
    chain = bytes.fromhex(hist['batch_hash'][-1]) if hist['batch_hash'] else b''

    def upd(chain, *arrs):
        h = hashlib.sha256(chain)
        for x in arrs:
            h.update(np.ascontiguousarray(x).tobytes())
        return h.digest()
    for step in range(start, steps + 1):
        if step in ckpt_steps and not (step == start and done_ck):
            te = time.time()
            if no_readout:
                rec, arrs, rk = dict(dev_retrieval=dict(rsum=float('nan')), readout_seconds=0.0, no_readout=True), None, None
            else:
                rec, arrs, rk = readout_r36(heads, nulls, encs, ctx, final=(step == steps))
            rec['step'] = step; rec['phase'] = cfg.get('phase', 1)
            if rk is not None:
                rec['transitions_vs_H'] = EV.transitions(rk, ctx.nat_rk, rk, ctx.nat_rk, d['dev_tab'])
            if win:
                w = np.array([(x[0], x[1], x[2]) for x in win])
                rec['train_window'] = dict(n=len(win), loss_mean=float(w[:, 0].mean()), batch_J_mean=float(w[:, 1].mean()), grad_norm_mean=float(w[:, 2].mean()), loss_last=float(w[-1, 0]))
                keys = sorted({k for g in gwin for k in g})
                rec['train_window']['slot_grad_means'] = {k: float(np.mean([g[k] for g in gwin if k in g])) for k in keys}
                hist['windows'].append(dict(end_step=step, **rec['train_window']))
                win = []; gwin = []
            if rk is not None and step == 0 and cfg.get('phase', 1) == 1:
                # fail-fast step-0 check (replaces the standalone GPU gate; user 2026-11-02 「取消gate，直接提交正式实验」):
                # step-0 two-stage ranks must equal the frozen-H native ranks, ties (|Δs| <= 1e-6) excepted — same rule as gate.step0_native
                nat = d['native_dev']; TOL = 1e-6
                di = np.nonzero(rk['ranks_i'] != ctx.nat_rk['ranks_i'])[0]; dt = np.nonzero(rk['ranks_t'] != ctx.nat_rk['ranks_t'])[0]
                tied_i = all(any((np.abs(nat[i] - nat[i, PER_CAP * i + k]) <= TOL).sum() > 1 for k in range(PER_CAP)) for i in di)
                tied_t = all((np.abs(nat[:, c] - nat[c // PER_CAP, c]) <= TOL).sum() > 1 for c in dt)
                rec['step0_native_check'] = dict(n_i2t_rank_diff=int(len(di)), n_t2i_rank_diff=int(len(dt)), rsum=rec['dev_retrieval']['rsum'], native_rsum=ctx.nat_metrics['rsum'],
                                                 passed=bool((len(di) == 0 and len(dt) == 0) or (tied_i and tied_t)))
                write_json(out / 'eval_step0.json', dict(rec, step=0))
                assert rec['step0_native_check']['passed'], f"step-0 does not reproduce native H: {rec['step0_native_check']}"
            if rk is not None and (step == steps or step == 0):
                np.savez(out / f'dev_scores_step{step}.npz', **arrs)
                if step == steps:
                    np.savez(out / f'dev_ranks_step{step}.npz', **rk)
            t_eval += time.time() - te
            write_json(out / f'eval_step{step}.json', rec)
            log(f"{cfg['unit_id']} p{cfg.get('phase', 1)} step {step} rsum {rec['dev_retrieval']['rsum']:.2f} ({rec['readout_seconds']:.1f}s)")
            if not dev_nan:
                save_ckpt(out / f'ckpt_step{step}.pt', cfg, step, heads, nulls, opt, rng, rng2, lineage)
            hist['train_seconds'] += t_train; hist['eval_seconds'] += t_eval; t_train = t_eval = 0.0
            hist['batch_hash'].append(chain.hex()); hist['last_step'] = step
            write_json(out / 'TRAIN_HISTORY.json', hist)
        if step == steps or (stop_at is not None and step == stop_at):
            break
        t0 = time.time()
        heads.train()
        loss, jb, idx, st, gl, cx = step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle)
        chain = upd(chain, *idx)
        for k2, v2 in st.items():
            hist['stats_sum'][k2] = hist['stats_sum'].get(k2, 0) + v2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gns = []
        for p in PEERS:
            ps = [q for q in peer_params(heads, nulls, p) if q.grad is not None]
            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))
        if not (np.isfinite(gns).all() and torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite at step {step}')
        gwin.append(grad_class_stats(gl, cx))
        opt.step()
        win.append((float(loss), jb, float(np.mean(gns))))
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t_train += time.time() - t0
    return dict(heads=heads, nulls=nulls, encs=encs, init_hash=init_hash, final_hash=state_hash(heads) + (state_hash(nulls) if nulls is not None else ''),
                batch_hash=chain.hex(), hist=hist, out=out, boundary=boundary)


def finalize(cfg, ctx, res_list, t_wall, base_out):
    """RESULT.json/DONE.json at the unit dir. V: two phases (primary = phase-2 step 4000 = total 12000; secondary = phase-1 8000)."""
    phases = []
    for c, res in res_list:
        out = res['out']
        curves = [read_json(out / f'eval_step{k}.json') for k in sorted(set(c['ckpt_steps']) | {c['steps']})]
        phases.append(dict(phase=c.get('phase', 1), steps=c['steps'], curves=curves, final=curves[-1], train_history=res['hist'],
                           identity=dict(init_state_sha256=res['init_hash'], final_state_sha256=res['final_hash'], batch_index_sha256=res['batch_hash']),
                           boundary=res.get('boundary')))
    fin = phases[-1]['final']
    dev = ctx.dev
    c0, r0 = res_list[-1]
    result = dict(unit_id=cfg['unit_id'], status='COMPLETE', config=cfg, total_steps=sum(p['steps'] for p in phases), final=fin, phases=phases,
                  primary_endpoint=cfg['primary_endpoint'], native_H_dev=ctx.nat_metrics, parent=dict(cond=cfg['cond'], **{k: ctx.d['manifest']['parent'][k] for k in ('unit', 'model_best', 'model_best_sha256', 'epoch', 'best_rsum')}),
                  present_probs=ctx.present,
                  diff_vs_native={k: fin['dev_retrieval'][k] - ctx.nat_metrics[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')},
                  code_sha256=code_hashes(),
                  batch_stats={f"phase{p['phase']}": p['train_history']['stats_sum'] for p in phases},
                  cost=dict(params_trainable=n_params(r0['heads']) + (n_params(r0['nulls']) if r0['nulls'] is not None else 0),
                            train_seconds=sum(p['train_history']['train_seconds'] for p in phases), eval_seconds=sum(p['train_history']['eval_seconds'] for p in phases),
                            unit_wall_seconds_this_process=t_wall, peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(dev)) if dev.type == 'cuda' else 0,
                            pairs_per_update_per_peer=2 * cfg['n_queries'] * N_SLOTS),
                  env=dict(gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu', host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                           torch=torch.__version__, deterministic=torch.are_deterministic_algorithms_enabled()),
                  test_guard=guard_record(), finished_utc=utc())
    if cfg['block'] == 'V' and len(res_list) == 2:
        p1, p2 = res_list[0][1]['out'], res_list[1][1]['out']
        z1 = np.load(p1 / f"dev_scores_step{res_list[0][0]['steps']}.npz"); z2 = np.load(p2 / 'dev_scores_step0.npz')
        result['phase2_step0_equals_phase1_final_scores'] = dict(bitwise=all(np.array_equal(z1[k], z2[k]) for k in ('gi_top', 'gt_top', 'g_gt')),
                                                                 max_abs=float(max(np.abs(z1[k] - z2[k]).max() for k in ('gi_top', 'gt_top', 'g_gt'))))
    result['artifacts'] = {str(f.relative_to(base_out)): sha256(f) for f in sorted(base_out.rglob('*_step*.npz'))}
    write_json(base_out / 'RESULT.json', result)
    write_json(base_out / 'DONE.json', dict(status='COMPLETE', finished_utc=utc(), rsum=fin['dev_retrieval']['rsum'], job=os.environ.get('SLURM_JOB_ID'),
                                            gpu=result['env']['gpu']))
    return result


def phase_cfgs(cfg):
    if cfg['block'] != 'V':
        return [dict(cfg, phase=1)]
    p1 = dict(cfg, phase=1, steps=cfg['steps'], ckpt_steps=cfg['ckpt_steps'], sampler_seed=cfg['sampler_seed'])
    p2 = dict(cfg, phase=2, steps=cfg['phase2_steps'], ckpt_steps=cfg['phase2_ckpt_steps'], sampler_seed=cfg['phase2_sampler_seed'], parent_step=cfg['steps'])
    return [p1, p2]


def check_frozen(uids):
    frozen = read_json(ROOT / 'PROTOCOL_FROZEN.json')
    for f, h in frozen['code_sha256'].items():
        if f in frozen.get('code_not_bound', []):
            continue
        assert sha256(ROOT / 'r37c' / f) == h, f'code differs from frozen protocol: {f}'
    for u in uids:
        assert sha256(ROOT / 'configs' / f'{u}.json') == frozen['config_sha256'][u], f'config differs from frozen: {u}'
    assert frozen.get('gpu_gate', {}).get('waived_by_user') or read_json(ROOT / 'GPU_GATE.json')['ALL_PASS'], 'GPU gate not passed'
    for f, h in frozen['parent_cache_manifest_sha256'].items():
        assert sha256(Path(f)) == h, f'parent cache/prep manifest changed since freeze: {f}'


def pool_of(cfg, ctx):
    return ctx.d['trust_subset'] if cfg['supervision'] in D36.TRUST_SUPS else None


def main(oracle=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--units', required=True)
    ap.add_argument('--gate-out', default=''); ap.add_argument('--gate-steps', type=int, default=0)
    a = ap.parse_args()
    gate = bool(a.gate_out)
    if oracle is None:
        install_guards(role='trainer'); assert_no_oracle_loaded()
        assert not [m for m in sys.modules if m.split('.')[-1] in EVAL_ONLY_MODULES], 'evaluator-only module loaded in a trainer'
    set_determinism()
    uids = a.units.split(',')
    cfgs = [cfg_of(u) for u in uids]
    for c in cfgs:
        if oracle is None:
            assert not c['oracle'], f"oracle unit {c['unit_id']} refused in the non-oracle runner"
        else:
            assert c['oracle'], f"non-oracle unit {c['unit_id']} refused in the oracle runner"
        c['_config_sha256'] = sha256(ROOT / 'configs' / f"{c['unit_id']}.json")
    if not gate:
        check_frozen(uids)
    device = torch.device('cuda')
    ctxs = {}
    failed = []
    for cfg in cfgs:
        base = Path(a.gate_out) / cfg['unit_id'] if gate else (ORACLE_RUNS if cfg['oracle'] else RUNS) / cfg['unit_id']
        if (base / 'DONE.json').exists():
            print('skip DONE', cfg['unit_id']); continue
        try:
            key = (cfg['cond'], cfg['structure'] == 'L')
            if key not in ctxs:
                ctxs[key] = Ctx(cfg['cond'], device, need_local=key[1])
            ctx = ctxs[key]
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time(); res_list = []
            for pc in phase_cfgs(cfg):
                if gate:
                    gs = a.gate_steps
                    pc = dict(pc, steps=gs, ckpt_steps=[0, gs])
                    if pc['phase'] == 2:
                        pc['parent_step'] = res_list[0][0]['steps']
                if pc['phase'] == 2:
                    pc['parent_ckpt'] = str(unit_out(dict(pc, phase=1), a.gate_out or None) / f"ckpt_step{pc['parent_step']}.pt")
                    assert Path(pc['parent_ckpt']).exists(), f"phase-1 checkpoint missing: {pc['parent_ckpt']}"
                res = run_unit(pc, ctx, pool=pool_of(pc, ctx), out_dir=unit_out(pc, a.gate_out or None), oracle=oracle)
                res_list.append((pc, res))
            r = finalize(cfg, ctx, res_list, time.time() - t0, base)
            print('R37C_UNIT_DONE', cfg['unit_id'], 'rsum', r['final']['dev_retrieval']['rsum'], 'native', ctx.nat_metrics['rsum'], flush=True)
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True)
            base.mkdir(parents=True, exist_ok=True)
            write_json(base / f"FAILED_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(unit=cfg['unit_id'], traceback=tb, utc=utc()))
            failed.append(cfg['unit_id'])
    print('R37C_JOB_END failed', failed)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
