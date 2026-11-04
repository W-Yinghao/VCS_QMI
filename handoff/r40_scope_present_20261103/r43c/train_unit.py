"""R41-F: R40 r40c train_unit.py (R41 freeze-file lookup; encoder-lr-variant reference mode in r40ext). R40: R39 train_unit.py + R40 edits (update scope image_only/text_only/both_joint via cfg adaptation -> Student(modalities); N_PRESENT_ONLY
in-unit checks (p_d recomputed from the full FIT observed relation, masked-row zero gradient every update, valid-row/unique-query/empty-batch
logs); per-modality/head/null pre-clip gradient norms + shared clip multiplier; R40 one-step active-path rule (all units); read-only
reused references R39 N-JOINT / R39 ORACLE-FROZEN asserted in-unit (init hashes, student init, batch chain); R39X A1 direction-sum
tolerance 1e-5 adopted before any unit; R40 freeze file). Training path otherwise byte-identical to R39.
R39: R38 train_unit.py + R39 edits (tools/patch_train_unit.py: COMPLETE label hook, JOINT student global encoders, two AdamW groups,
per-peer joint clip, in-unit cache-equivalence / one-step / frozen-tensor checks, R39 readouts). R38: train_unit.py of R37 (r37c/) + R38 edits (directional G readout per row direction, common cosine schedule, read-only reference
(R36/R37C) init/batch-chain identity, first-update graph check, direction-gradient diagnostic); shared/constant path unchanged.
R37C unit trainer = R37 train_unit.py ported to CC152K (only dataset constants: PER_CAP = 1 / identity instead of //5 in the
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
import r38ext as X
import r39ext as R9
import r40ext as R40
from common import FROZEN_FILE, ADAPTATIONS

# ============================================================================ configuration / context


def cfg_of(uid):
    return read_json(ROOT / 'configs' / f'{uid}.json')


def code_hashes():
    return {Path(f).name: sha256(f) for f in sorted(glob.glob(str(Path(__file__).resolve().parent / '*.py')))}


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
        self.completion = None; self.joint = None   # R39: loaded on demand (COMPLETE units / JOINT units)

    def ensure_completion(self):
        if self.completion is None:
            self.completion = R9.Completion(self.cond, self.d)
        return self.completion

    def ensure_joint(self):
        if self.joint is None:
            self.joint = R9.JointInputs(self.cond, self.d, self.dev)
        return self.joint


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


def g_pairs_lists(h, enc, ctx, p, split, img_idx, cap_idx, grad=False, enc_grad=False, s_exact=None, chunk=8192, proj=None, direction=None, stu=None):
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
        if stu is not None:   # R39 JOINT: phi takes the current student's global vectors; the skip term s stays the frozen-H native score
            x, y = stu[0][stu[1][i:i + chunk]], stu[2][stu[3][i:i + chunk]]
        lf = None
        if h.local is not None:
            vi, ci = inv_i[i:i + chunk], inv_c[i:i + chunk]
            lmax = int(M[ci].sum(1).max())
            lf = h.local.pair_features(V[vi], W[ci][:, :lmax], rmask_full[vi], M[ci][:, :lmax])
        hid = h.glob.hidden(x, y, s)
        if lf is not None:
            hid = torch.cat([hid, lf], -1)
        if getattr(h, 'directional', False):   # R38: each row scored with its own direction's output + affine (shared trunk)
            assert direction is not None, 'directional head needs the per-row direction (0 = i2t, 1 = t2i)'
            dd = torch.as_tensor(np.broadcast_to(np.asarray(direction), img_idx.shape)[i:i + chunk].copy(), device=dev)
            outs.append(h.rows(s, hid, dd, None if s_exact is None else torch.as_tensor(s_exact[i:i + chunk], device=dev)))
            continue
        r = h.out(hid).squeeze(-1)
        if s_exact is None:
            outs.append(h.affine(s) + r)
        else:
            a, b = (float(t) for t in h.affine.ab())
            outs.append(a * torch.as_tensor(s_exact[i:i + chunk], device=dev) + b + r.double())
    return torch.cat(outs)


def stu_pairs(stud, ctx, p, split, img, cap, grad=False):
    """R39 JOINT: student vectors of explicit pairs (unique IDs encoded once, re-expanded by index); None for frozen units."""
    if stud is None:
        return None
    return ctx.joint.pairs(stud[p], p, split, img, cap, grad)


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


def readout_lists(heads, encs, ctx, final=False, stud=None):
    """C/D readout: two-stage DEV retrieval (H top-100 reranked by the peer-mean critic g, rest in H order) on the
    full query set, coverage + conditional R@1, pair win rates, DEV list-J (clean annotation) and FIT list-J on a
    locked FIT list set (observed), raw and with a common FIT-only positive affine calibration."""
    d = ctx.d; tab = d['dev_tab']
    t0 = time.time()
    heads.eval()
    img, cap, names, sizes = dev_pairs(tab)
    directional = getattr(heads[PEERS[0]], 'directional', False)
    if directional:   # R38: GT pairs scored once per direction (g_gt: i2t, g_gt_t2i: t2i); every other DEV part in its own direction
        a_, b_ = sum(sizes[:2]), sum(sizes[:3]); assert names[2] == 'gt'
        img, cap, names, sizes = np.concatenate([img, img[a_:b_]]), np.concatenate([cap, cap[a_:b_]]), names + ['gt_t2i'], sizes + [b_ - a_]
    DIRN = dict(i_top=0, t_top=1, gt=0, rand_i=0, rand_t=1, gt_t2i=1)
    dirs = np.concatenate([np.full(n, DIRN[k], np.int64) for k, n in zip(names, sizes)])
    with torch.no_grad():
        gsum = 0
        for p in PEERS:
            h = heads[p]
            proj = project_items(h, encs[p], ctx.raw, 'dev', img, cap) if h.local is not None else None
            s_ex = d['dev_s'][p][torch.as_tensor(img, device=ctx.dev), torch.as_tensor(cap, device=ctx.dev)].cpu().numpy()
            gsum = gsum + g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'dev', img, cap, s_exact=s_ex, proj=proj, direction=dirs, stu=stu_pairs(stud, ctx, p, 'dev', img, cap))
        g = (gsum / 2).cpu().numpy()
    parts = dict(zip(names, np.split(g, np.cumsum(sizes)[:-1])))
    gi_top = parts['i_top'].reshape(N_DEV_IMG, -1); gt_top = parts['t_top'].reshape(N_DEV_CAP, -1)
    m, tb, cov, rk = EV.composite_metrics(d['native_dev'], tab, gi_top, gt_top)
    rec = dict(dev_retrieval=m, dev_retrieval_tie_bounded=tb, candidate_coverage=cov)
    g_gt = parts['gt']
    g_gt_t2i = parts['gt_t2i'] if directional else g_gt
    pos_i = g_gt.reshape(N_DEV_IMG, PER_CAP).max(1)
    hard_i = np.take_along_axis(gi_top, np.stack([np.nonzero((r // PER_CAP) != q)[0][:10] for q, r in enumerate(tab['i_top'])]), 1)
    hard_t = np.take_along_axis(gt_top, np.stack([np.nonzero(r != c // PER_CAP)[0][:10] for c, r in enumerate(tab['t_top'])]), 1)
    rec['pair_win_rates'] = EV.pair_win_rates(g_gt_t2i, hard_t, parts['rand_t'].reshape(N_DEV_CAP, 10), pos_i, hard_i,
                                              parts['rand_i'].reshape(N_DEV_IMG, 10))
    li, lt = EV.dev_list_scores(gi_top, gt_top, g_gt, ctx.pos_i, ctx.pos_t)
    if directional:
        lt = EV.dev_list_scores(gi_top, gt_top, g_gt_t2i, ctx.pos_i, ctx.pos_t)[1]
    rec['dev_list_clean_J_raw'] = EV.list_j(li, ctx.ai_dev, lt, ctx.at_dev)
    # FIT locked lists: eval (raw J) and calibration (affine fit)
    def fit_lists(ki, kt):
        lists = lists_from_locks(ctx, ki, kt)
        im, cp, ai, at = list_pairs(lists)
        fd = np.concatenate([np.zeros(len(ai) * N_SLOTS, np.int64), np.ones(len(at) * N_SLOTS, np.int64)])
        with torch.no_grad():
            gs = 0
            for p in PEERS:
                gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', im, cp, direction=fd, stu=stu_pairs(stud, ctx, p, 'fit', im, cp)).double()
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
    if directional:
        rec['heads_ab_t2i'] = {p: [float(t) for t in heads[p].affine_t2i.ab()] for p in PEERS}
    rec['readout_seconds'] = time.time() - t0
    arrs = dict(gi_top=gi_top, gt_top=gt_top, g_gt=g_gt)
    if directional:
        arrs['g_gt_t2i'] = g_gt_t2i
    return rec, arrs, rk


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
def fit_pair_scores(heads, encs, ctx, img, cap, direction=None, stud=None):
    gs, ss = 0, 0
    for p in PEERS:
        gs = gs + g_pairs_lists(heads[p], encs[p] if encs else None, ctx, p, 'fit', img, cap, direction=direction, stu=stu_pairs(stud, ctx, p, 'fit', img, cap)).double()
        x = ctx.d[('fit', 'img', p)][torch.as_tensor(img, device=ctx.dev)]; y = ctx.d[('fit', 'cap', p)][torch.as_tensor(cap, device=ctx.dev)]
        ss = ss + (x * y).sum(-1).double()
    return (gs / 2).cpu().numpy(), (ss / 2).cpu().numpy()


def readout_r36(heads, nulls, encs, ctx, final=False, stud=None):
    """R35 readout_lists (unchanged) + R36 additions on the locked FIT records: Spearman(g, native) and within-row
    g spread on the locked forced lists; for N units the null-list J (33 slots, Q = Uniform(33)) split into its
    real-slot and null-slot parts, null-label fraction and the null parameters. All FIT, observed labels only."""
    if stud is not None:
        ctx.joint.clear_eval()
    rec, arrs, rk = readout_lists(heads, encs, ctx, final=final, stud=stud)
    t0 = time.time()
    (qi, caps, ai), (qc, imgs, at) = lists_from_locks(ctx, 'lev_i', 'lev_t')
    im, cp, _, _ = list_pairs(((qi, caps, ai), (qc, imgs, at)))
    g, s = fit_pair_scores(heads, encs, ctx, im, cp, direction=np.concatenate([np.zeros(len(ai) * N_SLOTS, np.int64), np.ones(len(at) * N_SLOTS, np.int64)]), stud=stud)
    n = len(ai) * N_SLOTS
    G = np.concatenate([g[:n].reshape(-1, N_SLOTS), g[n:].reshape(-1, N_SLOTS)]); S = np.concatenate([s[:n].reshape(-1, N_SLOTS), s[n:].reshape(-1, N_SLOTS)])
    sp = _spearman_rows(G, S)
    rec['fit_forced_lists_score_structure'] = dict(spearman_g_vs_native_row_mean=float(np.nanmean(sp)), frac_rows_spearman_negative=float(np.nanmean(sp < 0)),
                                                   g_std_within_row_mean=float(G.std(1).mean()), native_std_within_row_mean=float(S.std(1).mean()))
    if nulls is not None:
        rec['fit_null_lists_observed_J_raw'] = null_list_j(heads, nulls, encs, ctx, D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t'), stud)
        if ctx.completion is not None:   # R39 COMPLETE: same locked lists, completed labels (observed-label J above stays the N label object)
            cl, cst = ctx.completion.apply(D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t'))
            rec['fit_null_lists_completed_J_raw'] = dict(null_list_j(heads, nulls, encs, ctx, cl, stud), completed_rows=cst)
    if stud is not None:   # R39 JOINT: student vs frozen-H global vectors over all FIT/DEV items
        rec['r39_student_feature_change'] = ctx.joint.feature_change(stud)
        ctx.joint.clear_eval()
    rec['readout_seconds'] += time.time() - t0
    return rec, arrs, rk


def null_list_j(heads, nulls, encs, ctx, lists, stud):
    """R36 null-list J (33 slots, Q = Uniform(33)) split into real-slot and null-slot parts for given (locked) lists/labels."""
    (qi, caps, ai), (qc, imgs, at) = lists
    if True:
        out = {}
        for dn, di, ids, q, a in (('i2t', 0, caps, qi, ai), ('t2i', 1, imgs, qc, at)):
            real = ids != D36.NULL_ID
            if dn == 'i2t':
                img_, cap_ = np.repeat(q, D36.K_NULL).reshape(ids.shape)[real], ids[real]
            else:
                img_, cap_ = ids[real], np.repeat(q, D36.K_NULL).reshape(ids.shape)[real]
            gr, _ = fit_pair_scores(heads, encs, ctx, img_, cap_, direction=di, stud=stud)
            gg = np.zeros(ids.shape); gg[real] = gr
            gg[~real] = float(np.mean([float(nulls.v[p][di]) for p in PEERS]))
            t = np.tanh(gg / 2); K = gg.shape[1]
            part = a * (t - .5 * t * t) - (t + .5 * t * t) / K
            out[dn] = dict(J_total=float(part.sum(1).mean()), J_real_part=float((part * real).sum(1).mean()), J_null_part=float((part * ~real).sum(1).mean()),
                           null_label_fraction=float(a[~real].mean()), observed_pos_real_slots_mean=float(((a > 0) & real).sum(1).mean()),
                           null_rank_mean=float((gg > gg[~real][:, None]).sum(1).mean()))
        out['J_total_bidirectional'] = .5 * (out['i2t']['J_total'] + out['t2i']['J_total'])
        out['null_params'] = {p: [float(x) for x in nulls.v[p].detach().cpu()] for p in PEERS}
        return out


# ============================================================================ training

def make_opt(cfg, heads, nulls, stud=None):
    """R39: group 0 = heads (+ nulls), lr_head (unchanged R38 group); JOINT adds group 1 = student global encoders, lr_encoder."""
    o = cfg['optimizer']
    ps = list(heads.parameters()) + (list(nulls.parameters()) if nulls is not None else [])
    groups = [dict(params=ps, lr=o['lr_head'], base_lr=o['lr_head'], name='head')]
    if stud is not None:
        groups.append(dict(params=[q for p in PEERS for q in stud[p].trainable()], lr=o['lr_encoder'], base_lr=o['lr_encoder'], name='student_encoder'))
    return torch.optim.AdamW(groups, betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'])


def peer_params(heads, nulls, p, stud=None):
    return list(heads[p].parameters()) + ([nulls.v[p]] if nulls is not None else []) + (stud[p].trainable() if stud is not None else [])


def _pairs(lists):
    """flatten real (image, caption) pairs: i2t rows then t2i rows; returns masks of real slots per direction."""
    (qi, caps, ai), (qc, imgs, at) = lists
    ri, rt = caps != D36.NULL_ID, imgs != D36.NULL_ID
    img = np.concatenate([np.repeat(qi, caps.shape[1]).reshape(caps.shape)[ri], imgs[rt]])
    cap = np.concatenate([caps[ri], np.repeat(qc, imgs.shape[1]).reshape(imgs.shape)[rt]])
    return img, cap, ri, rt


def step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud=None):
    d = ctx.d; sup = cfg['supervision']
    lists, idx = D36.sample(sup, rng, rng2, d, cfg['n_queries'], pool)
    stats = {}
    if sup == 'COMPLETE_XCHECK':   # R39: fixed cross-checked completion of observed-missing rows (only all-null rows change; asserted)
        lists, cst = ctx.completion.apply(lists)
        stats.update(cst)
    (qi, caps, ai), (qc, imgs, at) = lists
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
    dirs = np.concatenate([np.zeros(n_i, np.int64), np.ones(len(img) - n_i, np.int64)])
    ri_t, rt_t = torch.as_tensor(ri, device=ctx.dev), torch.as_tensor(rt, device=ctx.dev)
    present_only = sup == 'N_PRESENT_ONLY'
    if present_only:
        mi, mt = D36.valid_rows(lists)
        Mi = torch.as_tensor(mi, device=ctx.dev, dtype=torch.float32); Mt = torch.as_tensor(mt, device=ctx.dev, dtype=torch.float32)
        stats.update(valid_rows_i2t=float(mi.sum()), valid_rows_t2i=float(mt.sum()), empty_valid_batch_i2t=float(mi.sum() == 0),
                     empty_valid_batch_t2i=float(mt.sum() == 0))
        stats.update(valid_unique_queries_i2t=float(len(np.unique(qi[mi]))), valid_unique_queries_t2i=float(len(np.unique(qc[mt]))),   # R40 logs
                     unique_queries_i2t=float(len(np.unique(qi))), unique_queries_t2i=float(len(np.unique(qc))),
                     present_weight_i2t=float(mi.sum()) / (len(mi) * ctx.present['p_i2t']), present_weight_t2i=float(mt.sum()) / (len(mt) * ctx.present['p_t2i']))
    loss, jb, gl = 0.0, 0.0, []
    dl_i = dl_t = 0.0
    for p in PEERS:
        h = heads[p]
        proj = project_items(h, encs[p], ctx.raw, 'fit', img, cap, grad=True, enc_grad=False) if h.local is not None else None
        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=False, proj=proj, chunk=1 << 30, direction=dirs,
                          stu=stu_pairs(stud, ctx, p, 'fit', img, cap, grad=True))
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
        dl_i = dl_i + .25 * li; dl_t = dl_t + .25 * lt
        with torch.no_grad():
            jb += .5 * float(.5 * list_rows(gi, A_i, 'J').mean() + .5 * list_rows(gt, A_t, 'J').mean())
    stats.update(n_pos_slots_i2t_rows_sum=float((ai > 0).sum()), n_unique_images=int(len(np.unique(img))), n_unique_caps=int(len(np.unique(cap))))
    if nulls is not None:
        stats.update(null_label_rows_i2t=float(ai[~ri].sum()), null_label_rows_t2i=float(at[~rt].sum()),
                     observed_pos_real_slots_i2t=float(((ai > 0) & ri).sum()), observed_pos_real_slots_t2i=float(((at > 0) & rt).sum()),
                     effective_pos_rows_i2t=float(((ai > 0) & ri).any(1).sum()), effective_pos_rows_t2i=float(((at > 0) & rt).any(1).sum()))
    ctxs = dict(ai=ai, at=at, ri=ri, rt=rt, dir_losses=(dl_i, dl_t), dirs=dirs, present_masks=(mi, mt) if present_only else None)
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


def save_ckpt(path, cfg, step, heads, nulls, opt, rng, rng2, lineage, stud=None):
    obj = dict(step=step, unit_id=cfg['unit_id'], phase=cfg.get('phase', 1), heads=heads.state_dict(), nulls=nulls.state_dict() if nulls is not None else None,
               student=stud.state_dict() if stud is not None else None,
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
    joint = cfg.get('encoder') == 'joint_global'
    stud = None
    if cfg['supervision'] == 'COMPLETE_XCHECK':
        lineage['completion_table_sha256'] = ctx.completion.sha256
        lineage['completion_prep_manifest'] = dict(teachers=ctx.completion.manifest['teachers'], rule=ctx.completion.manifest['rule'])
    if joint:   # R39 JOINT: fresh student copies of the parent's global encoders (both peers), built from the parent checkpoint
        stud = R9.build_students(ctx.joint.ck, dev, ADAPTATIONS[cfg['adaptation']])   # R40: enabled modalities only (frozen modality: parent, requires_grad False)
        stud_ref = {p: {n: q.detach().clone() for n, q in stud[p].named_parameters()} for p in PEERS}
        frozen0 = {p: R9.state_digest(stud[p].frozen_state()) for p in PEERS}
        cache0 = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS})
        lineage['student'] = dict(parent_ckpt=str(R9.C.PARENT_CKPT), parent_sha256=ctx.joint.manifest['parent']['model_best_sha256'],
                                  init_state_sha256=state_hash(stud), joint_prep_manifest=dict(tokens=ctx.joint.manifest['tokens'], equivalence=ctx.joint.manifest.get('equivalence_summary')),
                                  trainable_params=int(sum(q.numel() for p in PEERS for q in stud[p].trainable())),
                                  adaptation=cfg['adaptation'], modalities=list(ADAPTATIONS[cfg['adaptation']]),
                                  trainable_params_by_modality={m: int(sum(q.numel() for p in PEERS for n, q in stud[p].named_parameters() if q.requires_grad and n.startswith(m + '.'))) for m in ('img', 'txt')},
                                  frozen_modality_params=int(sum(q.numel() for p in PEERS for n, q in stud[p].named_parameters() if not q.requires_grad and n.split('.')[0] not in stud[p].modalities)))
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
    cache40 = R9.state_digest({f'{s_}_{k_}_{p_}': d[(s_, k_, p_)] for s_ in ('fit', 'dev') for k_ in ('img', 'cap') for p_ in PEERS})   # R40: frozen-H cache digest (all units)
    sched = cfg['optimizer'].get('schedule', 'constant'); assert sched in ('constant', 'cosine'), sched
    directional = getattr(heads[PEERS[0]], 'directional', False)
    assert directional == (cfg['structure'] == 'GD')
    ref = X.reference(cfg)
    r38 = dict(schedule=sched, directional=directional, init_state_sha256_shared_keys=X.state_hash_shared(heads, nulls),
               reference_unit=ref['unit'] if ref else None, reference_init_equal=None, reference_chain_equal_at=[], lr_at_readout={})
    if ref is not None:   # read-only reuse check: same fresh head init (shared keys) as the R36/R37C reference unit of this parent/route
        r38['reference_init_equal'] = r38['init_state_sha256_shared_keys'] == ref['init']
        assert r38['reference_init_equal'], f"init differs from reference {ref['unit']}"
    assert (cfg['adaptation'] != 'frozen') == joint, (cfg['adaptation'], cfg.get('encoder'))
    xrefs = R40.extra_references(cfg)   # R40: read-only reused references (R39 N-JOINT / R39 ORACLE-FROZEN): init hashes + batch chain
    r40 = dict(adaptation=cfg['adaptation'], references={k: dict(unit=v['unit'], init_equal=init_hash == v['init'], chain_equal_at=[],
                                                               student_init_equal=(state_hash(stud) == v['student_init']) if v['student_init'] else None) for k, v in xrefs.items()},
               grad_split_windows=[])
    for k, v in r40['references'].items():
        assert v['init_equal'], f'head/null init differs from {k} {v["unit"]}'
        assert v['student_init_equal'] in (None, True), f'student init differs from {k} {v["unit"]}'
    if cfg['supervision'] == 'N_PRESENT_ONLY':   # R40: p_d fixed from the full FIT observed relation (recomputed independently, asserted)
        r40['present_check'] = R40.present_recheck(d, ctx.present, D36.null_lists, D36.valid_rows)
        write_json(out / 'PRESENT_CHECK.json', r40['present_check'])
        assert r40['present_check']['passed'], r40['present_check']
        r40['masked_grad'] = dict(updates_checked=0, violations=0, masked_rows=0, valid_rows_nonzero_grad=0)
    opt = make_opt(cfg, heads, nulls, stud)
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
        if stud is not None:
            stud.load_state_dict(ck['student'])
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
    hist.setdefault('r38', {}).update({k: v for k, v in r38.items() if k not in hist['r38']})
    hist['r38'].setdefault('dir_grad_diag', {})
    hist.setdefault('r39', dict(joint=joint, supervision=cfg['supervision'], lr_encoder_at_readout={}, param_distance={}, grad_split_windows=[]))
    hist.setdefault('r40', {})
    for k_, v_ in r40.items():
        if k_ == 'references':
            for rk_, rv_ in v_.items():
                hist['r40'].setdefault('references', {}).setdefault(rk_, rv_)
        elif k_ != 'grad_split_windows' or k_ not in hist['r40']:
            hist['r40'].setdefault(k_, v_)
    gs40 = []
    gs_win = []
    last_lr = None; gc = None; last_lr_enc = None
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
                rec, arrs, rk = readout_r36(heads, nulls, encs, ctx, final=(step == steps), stud=stud)
                if stud is not None:
                    rec['r39_param_distance'] = ctx.joint.param_distance(stud, stud_ref); hist['r39']['param_distance'][str(step)] = rec['r39_param_distance']
            rec['step'] = step; rec['phase'] = cfg.get('phase', 1)
            if rk is not None:
                rec['transitions_vs_H'] = EV.transitions(rk, ctx.nat_rk, rk, ctx.nat_rk, d['dev_tab'])
            if win:
                w = np.array([(x[0], x[1], x[2]) for x in win])
                rec['train_window'] = dict(n=len(win), loss_mean=float(w[:, 0].mean()), batch_J_mean=float(w[:, 1].mean()), grad_norm_mean=float(w[:, 2].mean()), loss_last=float(w[-1, 0]))
                keys = sorted({k for g in gwin for k in g})
                rec['train_window']['slot_grad_means'] = {k: float(np.mean([g[k] for g in gwin if k in g])) for k in keys}
                if gs_win:
                    a_ = np.array(gs_win)
                    rec['train_window']['r39_grad_split'] = dict(head_norm_mean=float(a_[:, 0].mean()), encoder_norm_mean=float(a_[:, 1].mean()),
                                                                 total_norm_mean=float(a_[:, 2].mean()), clip_coef_mean=float(a_[:, 3].mean()), frac_clipped=float((a_[:, 3] < 1).mean()))
                    hist['r39']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r39_grad_split'])); gs_win = []
                if gs40:
                    a_ = np.array(gs40)
                    rec['train_window']['r40_grad_split'] = {nm_: float(a_[:, j_].mean()) for j_, nm_ in enumerate(('head_norm_mean', 'null_norm_mean', 'img_norm_mean', 'txt_norm_mean', 'total_norm_mean', 'clip_coef_mean'))}
                    rec['train_window']['r40_grad_split']['frac_clipped'] = float((a_[:, 5] < 1).mean())
                    hist['r40']['grad_split_windows'].append(dict(end_step=step, **rec['train_window']['r40_grad_split'])); gs40 = []
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
                save_ckpt(out / f'ckpt_step{step}.pt', cfg, step, heads, nulls, opt, rng, rng2, lineage, stud)
            hist['train_seconds'] += t_train; hist['eval_seconds'] += t_eval; t_train = t_eval = 0.0
            hist['batch_hash'].append(chain.hex()); hist['last_step'] = step
            if ref is not None:   # read-only reuse check: identical sampled lists (batch-index hash chain) as the reference unit
                kk = len(hist['batch_hash']) - 1
                assert hist['batch_hash'][-1] == ref['chain'][kk], f"batch chain differs from reference {ref['unit']} at step {step}"
                if step not in hist['r38']['reference_chain_equal_at']:
                    hist['r38']['reference_chain_equal_at'].append(step)
            for rk_, rv_ in xrefs.items():   # R40: same batch-index hash chain as the reused R39 N-JOINT / ORACLE-FROZEN unit
                kk = len(hist['batch_hash']) - 1
                assert hist['batch_hash'][-1] == rv_['chain'][kk], f"batch chain differs from {rk_} {rv_['unit']} at step {step}"
                if step not in hist['r40']['references'][rk_]['chain_equal_at']:
                    hist['r40']['references'][rk_]['chain_equal_at'].append(step)
            hist['r38']['lr_at_readout'][str(step)] = last_lr
            hist['r39']['lr_encoder_at_readout'][str(step)] = last_lr_enc
            write_json(out / 'TRAIN_HISTORY.json', hist)
        if step == steps or (stop_at is not None and step == stop_at):
            break
        t0 = time.time()
        heads.train()
        if stud is not None:
            stud.train()
        loss, jb, idx, st, gl, cx = step_lists(cfg, heads, nulls, encs, ctx, rng, rng2, pool, oracle, stud)
        if step == 0 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 fail-fast: student forward (grad on, training batch) == native cache
            eq = dict(train_batch_grad=ctx.joint.cache_equivalence(stud, idx[0], idx[1], 'fit', grad=True))
            ctx.joint.clear_eval()
            eq['dev_all_nograd'] = {k: v for k, v in ctx.joint.cache_equivalence(stud, np.arange(N_DEV_IMG), np.arange(N_DEV_CAP), 'dev', grad=False).items()}
            tol = cfg['joint_rule']['equivalence_tol_max_abs']
            eq['tol_max_abs'] = tol; eq['passed'] = bool(all(v['max_abs'] <= tol for part in ('train_batch_grad', 'dev_all_nograd') for v in eq[part].values()))
            hist['r39']['step0_cache_equivalence'] = eq; write_json(out / 'STEP0_CACHE_EQUIVALENCE.json', eq)
            assert eq['passed'], f'R39 student forward differs from the native cache: {eq}'
        chain = upd(chain, *idx)
        for k2, v2 in st.items():
            hist['stats_sum'][k2] = hist['stats_sum'].get(k2, 0) + v2
        diag = None
        if step in ckpt_steps or step <= 1:   # R38 diagnostic: per-direction gradients on this training batch (no effect on .grad)
            diag, flat = X.dir_grads(heads, nulls, PEERS, *cx['dir_losses'])
            hist['r38']['dir_grad_diag'][str(step)] = diag
            for a_, b_ in gl:   # autograd.grad fills .grad of the retain_grad score tensors: reset so grad_class_stats sees only loss.backward()
                a_.grad = None; b_.grad = None
        if step == 0 and cfg.get('phase', 1) == 1:
            gc = first_step_graph_check(cfg, heads, nulls, ctx, idx, cx, diag, a0, b0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cx['present_masks'] is not None:   # R40: masked (absent) rows must get exactly zero dL/dg on every slot, every update
            v_, nm_, nz_ = R40.masked_rows_grad(gl, cx['present_masks'])
            mg = hist['r40']['masked_grad']; mg['updates_checked'] += 1; mg['violations'] += v_; mg['masked_rows'] += nm_; mg['valid_rows_nonzero_grad'] += nz_
            assert v_ == 0, f'N_PRESENT_ONLY: {v_} masked rows with nonzero gradient at step {step}'
        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step rule: update-1 gradients (pre-clip) of the declared-active tensors
            act40 = R40.active_named(heads, nulls, stud); grads40 = R40.grad_record(act40)
        if step == 0 and cfg.get('phase', 1) == 1:   # total gradient = sum of the two direction gradients (pre-clip)
            named, gi_, gt_ = flat
            err = max(float((((q.grad if q.grad is not None else torch.zeros_like(q)) - (a + b)).abs().max())) for (_, _, q), a, b in zip(named, gi_, gt_))
            scale = max(float(q.grad.abs().max()) for (_, _, q) in named if q.grad is not None)
            gc['sum_of_direction_grads_equals_total'] = dict(max_abs_err=err, max_abs_grad=scale, passed=bool(err <= 1e-5 * scale + 1e-12), tol_rel=1e-5,
                                                             note='R40: R39X amendment A1 tolerance adopted before any R40 unit (fp32 reduction order)')
            gc['passed'] = bool(gc['passed'] and gc['sum_of_direction_grads_equals_total']['passed'])
            hist['r38']['step1_graph_check'] = gc
            write_json(out / 'STEP1_GRAPH_CHECK.json', gc)
            assert gc['passed'], f'R38 first-update graph check failed: {gc}'
        if step == 1 and cfg.get('phase', 1) == 1 and gc is not None:   # update 1: nonzero outputs -> trunk gets both directions; routing still exact
            sec = {}
            for p in PEERS:
                d_ = diag[p]; zz = lambda g, k: d_.get(g, {}).get(k, 0.0)
                sec[p] = dict(trunk_from_i2t=zz('trunk', 'norm_i2t'), trunk_from_t2i=zz('trunk', 'norm_t2i'), trunk_cos=zz('trunk', 'cos'),
                              i2t_loss_to_t2i_extras=zz('out_t2i', 'norm_i2t') + zz('affine_t2i', 'norm_i2t'), t2i_loss_to_i2t_output=zz('out_i2t', 'norm_t2i') + zz('affine_i2t', 'norm_t2i'))
            ok2 = all(sec[p]['trunk_from_i2t'] > 0 and sec[p]['trunk_from_t2i'] > 0 for p in PEERS)
            if directional:
                ok2 &= all(sec[p]['i2t_loss_to_t2i_extras'] == 0.0 and sec[p]['t2i_loss_to_i2t_output'] == 0.0 for p in PEERS)
            gc['second_update'] = dict(sec, passed=bool(ok2)); gc['passed'] = bool(gc['passed'] and ok2)
            hist['r38']['step1_graph_check'] = gc
            write_json(out / 'STEP1_GRAPH_CHECK.json', gc)
            assert gc['passed'], f'R38 second-update graph check failed: {gc}'
        if step == 1 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 update-1 check: every student peer gets a gradient (head out is nonzero now)
            hist['r39']['update1_student_grad_norm'] = {p: float(torch.sqrt(sum((q.grad.double() ** 2).sum() for q in stud[p].trainable() if q.grad is not None))) for p in PEERS}
            assert all(v > 0 for v in hist['r39']['update1_student_grad_norm'].values()), hist['r39']['update1_student_grad_norm']
        gns = []
        for p in PEERS:
            ps = [q for q in peer_params(heads, nulls, p, stud) if q.grad is not None]
            if stud is not None:   # R39: one joint clip per peer (head + null + student); record the split before clipping
                hn = float(torch.sqrt(sum((q.grad.double() ** 2).sum() for q in peer_params(heads, nulls, p) if q.grad is not None)))
                en = float(torch.sqrt(sum((q.grad.double() ** 2).sum() for q in stud[p].trainable() if q.grad is not None)))
            nrm = lambda qs: float(torch.sqrt(sum(((q.grad.double() ** 2).sum() for q in qs if q.grad is not None), torch.zeros((), dtype=torch.float64, device=dev))))
            hn40 = nrm(list(heads[p].parameters())); nn40 = nrm([nulls.v[p]]) if nulls is not None else 0.0   # R40 split (pre-clip)
            in40 = nrm([q for n_, q in stud[p].named_parameters() if q.requires_grad and n_.startswith('img.')]) if stud is not None else 0.0
            tn40 = nrm([q for n_, q in stud[p].named_parameters() if q.requires_grad and n_.startswith('txt.')]) if stud is not None else 0.0
            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))
            gs40.append((hn40, nn40, in40, tn40, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))
            if stud is not None:
                gs_win.append((hn, en, gns[-1], min(1.0, cfg['clip_per_peer'] / (gns[-1] + 1e-6))))
        if not (np.isfinite(gns).all() and torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite at step {step}')
        gwin.append(grad_class_stats(gl, cx))
        if sched == 'cosine':   # R38 common cosine (all objectives): lr(t) = lr_head * lr_multiplier(t), t = update index; optimizer never reset
            for pg in opt.param_groups:   # R39: each group's own base lr x the common multiplier (group 0 = R38 head group, same value)
                pg['lr'] = pg['base_lr'] * X.lr_multiplier(step, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor'])
        last_lr = opt.param_groups[0]['lr']
        last_lr_enc = opt.param_groups[1]['lr'] if len(opt.param_groups) > 1 else None
        if step == 1 and cfg.get('phase', 1) == 1:
            before40 = R40.snapshot(act40)
        opt.step()
        if step == 1 and cfg.get('phase', 1) == 1:   # R40 one-step active-path rule (every unit; frozen modality bit-unchanged)
            fz40 = R40.frozen_modality_named(stud)
            o40 = R40.one_step_rule(opt, act40, before40, grads40, fz40, {n_: stud_ref[n_.split('.')[1]][n_.split('.', 2)[2]] for n_, _ in fz40})
            if stud is not None:
                o40['frozen_state_unchanged'] = {p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS}
                o40['passed'] = bool(o40['passed'] and all(o40['frozen_state_unchanged'].values()))
            o40['frozen_H_cache_unchanged'] = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache40
            o40['passed'] = bool(o40['passed'] and o40['frozen_H_cache_unchanged'])
            hist['r40']['one_step_check'] = {k_: v_ for k_, v_ in o40.items() if k_ not in ('per_tensor', 'frozen_modality')}
            write_json(out / 'ONE_STEP_CHECK_R40.json', o40)
            assert o40['passed'], f"R40 one-step active-path check failed: { {k_: v_ for k_, v_ in o40.items() if k_ not in ('per_tensor', 'frozen_modality')} }"
        if step == 1 and cfg.get('phase', 1) == 1 and stud is not None:   # R39 update-1 check: student changed, frozen tensors / parent cache unchanged
            ch = {p: sum(int(not torch.equal(q.detach(), stud_ref[p][n])) for n, q in stud[p].named_parameters() if q.requires_grad) for p in PEERS}
            ntr = {p: sum(1 for q in stud[p].trainable()) for p in PEERS}
            fr = {p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS}
            ca = R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache0
            oc = dict(student_tensors_changed=ch, student_trainable_tensors=ntr, frozen_embedding_and_bn_stats_unchanged=fr, frozen_H_cache_unchanged=bool(ca),
                      parent_ckpt_sha256_unchanged=sha256(R9.C.PARENT_CKPT) == lineage['student']['parent_sha256'])
            oc['all_trainable_changed_vs_parent_R39_rule'] = bool(all(ch[p] == ntr[p] for p in PEERS))   # R40: recorded only; the R40 rule (ONE_STEP_CHECK_R40.json) decides
            oc['passed'] = bool(o40['passed'] and all(fr.values()) and ca and oc['parent_ckpt_sha256_unchanged'])
            hist['r39']['one_step_check'] = oc; write_json(out / 'ONE_STEP_CHECK.json', oc)
            assert oc['passed'], f'R39 one-step check failed: {oc}'
        if step == 0 and cfg.get('phase', 1) == 1 and cfg['supervision'] == 'COMPLETE_XCHECK':   # R39 COMPLETE one-step record (invariants asserted per batch)
            write_json(out / 'ONE_STEP_CHECK.json', dict(supervision='COMPLETE_XCHECK', first_batch=st, rule='only all-null rows change; changed rows = 0.5 agreed + 0.5 null (asserted every batch)', passed=True))
        win.append((float(loss), jb, float(np.mean(gns))))
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t_train += time.time() - t0
    if stud is not None:   # R39: end-of-unit frozen-tensor invariance (parent cache, student embedding/BN stats)
        hist['r39']['end_frozen_check'] = dict(frozen_H_cache_unchanged=R9.state_digest({f'{s}_{k}_{p}': d[(s, k, p)] for s in ('fit', 'dev') for k in ('img', 'cap') for p in PEERS}) == cache0,
                                               frozen_embedding_and_bn_stats_unchanged={p: R9.state_digest(stud[p].frozen_state()) == frozen0[p] for p in PEERS})
    return dict(heads=heads, nulls=nulls, encs=encs, stud=stud, init_hash=init_hash, final_hash=state_hash(heads) + (state_hash(nulls) if nulls is not None else '') + (state_hash(stud) if stud is not None else ''),
                batch_hash=chain.hex(), hist=hist, out=out, boundary=boundary)


@torch.no_grad()
def first_step_graph_check(cfg, heads, nulls, ctx, idx, cx, diag, a0, b0):
    """R38 in-unit check on the first training batch, before the first update (replaces a standalone GPU gate):
    directional: (1) routed scores == all-i2t scores == all-t2i scores bitwise (both outputs zero, both affines a0/b0);
    (2) == the scores of a freshly built shared G head of the same seed (trunk/out/affine bitwise those of 'G');
    (3) routing: dL_i2t/d(out_t2i, affine_t2i) == 0 and dL_t2i/d(out_i2t, affine_i2t) == 0 exactly, the own-direction output
    and the shared trunk get nonzero gradients from each direction. shared: both directions reach the single output.
    The sum check (total grad == sum of direction grads) is added after loss.backward()."""
    img, cap = idx[0], idx[1]; dirs = cx['dirs']
    out = dict(directional=bool(getattr(heads[PEERS[0]], 'directional', False)), n_pairs=int(len(img)), n_i2t_pairs=int((dirs == 0).sum()))
    ok = True
    if out['directional']:
        with torch.random.fork_rng(devices=[]):   # fresh same-seed shared head without disturbing the run's CPU RNG
            ref_heads = build_heads('G', a0, b0, cfg['init_seed']).to(ctx.dev)
        eq = {}
        for p in PEERS:
            kw = dict(chunk=1 << 30)
            g_r = g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, direction=dirs, **kw)
            g_0 = g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, direction=0, **kw)
            g_1 = g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, direction=1, **kw)
            g_s = g_pairs_lists(ref_heads[p], None, ctx, p, 'fit', img, cap, **kw)
            eq[p] = dict(routed_eq_i2t=bool(torch.equal(g_r, g_0)), routed_eq_t2i=bool(torch.equal(g_r, g_1)), routed_eq_fresh_shared_G=bool(torch.equal(g_r, g_s)))
            ok &= all(eq[p].values())
        out['step0_scores'] = eq
        rt = {}
        for p in PEERS:
            d = diag[p]
            z = lambda g, k: d.get(g, {}).get(k, 0.0)
            rt[p] = dict(i2t_loss_to_t2i_extras=z('out_t2i', 'norm_i2t') + z('affine_t2i', 'norm_i2t'), t2i_loss_to_i2t_output=z('out_i2t', 'norm_t2i') + z('affine_i2t', 'norm_t2i'),
                         i2t_loss_to_own_output=z('out_i2t', 'norm_i2t'), t2i_loss_to_own_output=z('out_t2i', 'norm_t2i'),
                         trunk_from_i2t=z('trunk', 'norm_i2t'), trunk_from_t2i=z('trunk', 'norm_t2i'))
            ok &= rt[p]['i2t_loss_to_t2i_extras'] == 0.0 and rt[p]['t2i_loss_to_i2t_output'] == 0.0 and rt[p]['i2t_loss_to_own_output'] > 0 and rt[p]['t2i_loss_to_own_output'] > 0
        out['routing'] = rt
    else:
        rt = {p: dict(out_from_i2t=diag[p]['out_i2t']['norm_i2t'], out_from_t2i=diag[p]['out_i2t']['norm_t2i'], trunk_from_i2t=diag[p]['trunk']['norm_i2t'], trunk_from_t2i=diag[p]['trunk']['norm_t2i']) for p in PEERS}
        ok &= all(rt[p]['out_from_i2t'] > 0 and rt[p]['out_from_t2i'] > 0 for p in PEERS)   # trunk grads are 0 at update 0 (zero output): checked at update 1
        out['routing'] = rt
    sched = cfg['optimizer'].get('schedule', 'constant')
    out['lr_update0'] = cfg['optimizer']['lr_head'] * (X.lr_multiplier(0, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor']) if sched == 'cosine' else 1.0)
    out['lr_last_update'] = cfg['optimizer']['lr_head'] * (X.lr_multiplier(cfg['steps'] - 1, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor']) if sched == 'cosine' else 1.0)
    ok &= out['lr_update0'] == cfg['optimizer']['lr_head'] and (sched == 'constant' or abs(out['lr_last_update'] - .1 * cfg['optimizer']['lr_head']) < 1e-18)
    out['passed'] = bool(ok); out['utc'] = utc()
    return out


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
                  code_sha256=code_hashes(), r38=phases[-1]['train_history'].get('r38'),
                  batch_stats={f"phase{p['phase']}": p['train_history']['stats_sum'] for p in phases},
                  r39=phases[-1]['train_history'].get('r39'), r40=phases[-1]['train_history'].get('r40'), adaptation=cfg['adaptation'],
                  cost=dict(params_trainable=n_params(r0['heads']) + (n_params(r0['nulls']) if r0['nulls'] is not None else 0) + (sum(q.numel() for p in PEERS for q in r0['stud'][p].trainable()) if r0.get('stud') is not None else 0),
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
    from common import FROZEN_FILES_R41
    parts = [read_json(ROOT / f) for f in FROZEN_FILES_R41 if (ROOT / f).exists()]   # R41-F freeze file
    frozen = next(fr for fr in parts if all(u in fr['config_sha256'] for u in uids))
    here = Path(__file__).resolve().parent
    code = frozen['code_sha256_by_tree'][here.name]   # R38: one frozen code set per tree (r38f / r38c)
    assert set(code) == {Path(f).name for f in glob.glob(str(here / '*.py'))}, 'code file set differs from frozen protocol'
    for u in uids:
        assert read_json(ROOT / 'configs' / f'{u}.json')['code_tree'] == here.name, f'unit {u} belongs to another code tree'
    for f, h in code.items():
        if f in frozen.get('code_not_bound', []):
            continue
        assert sha256(Path(__file__).resolve().parent / f) == h, f'code differs from frozen protocol: {f}'
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
            if cfg['supervision'] == 'COMPLETE_XCHECK':
                ctx.ensure_completion()
            if cfg.get('encoder') == 'joint_global':
                ctx.ensure_joint()
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
            print('R41_UNIT_DONE', cfg['unit_id'], 'rsum', r['final']['dev_retrieval']['rsum'], 'native', ctx.nat_metrics['rsum'], flush=True)
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True)
            base.mkdir(parents=True, exist_ok=True)
            write_json(base / f"FAILED_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(unit=cfg['unit_id'], traceback=tb, utc=utc()))
            failed.append(cfg['unit_id'])
    print('R41_JOB_END failed', failed)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
