"""R35 unit trainer (non-oracle process): A (ID/G heads on all observed records), B-TRUST (G head on the frozen
native-GMM subset), C (G/L heads on 32-slot frozen-H candidate lists), D (C-L step-8000 parent: local encoder
unfreeze vs HEAD_CONTROL). B-ORACLE units are refused here (train_oracle.py, separate process).
Usage: python -B train_unit.py --units id1,id2,...   (units run sequentially; DONE units skipped; resumable)
DEV only enters no-grad readouts. TEST files are never opened (audit guard)."""
from __future__ import annotations
import argparse, glob, hashlib, os, re, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ROOT, RUNS, PEERS, N_TRAIN_CAP, N_DEV_IMG, N_DEV_CAP, install_guards, guard_record, assert_no_oracle_loaded,
                    sha256, write_json, read_json, utc, state_hash)
from heads import build_heads, n_params
from objectives import pq_weights, loss_records, j_value, loss_list, list_rows
import evals as EV
from data35 import load_cond, shard_mm, sample_lists, build_lists, load_local_encoders, load_raw, N_SLOTS

# ============================================================================ configuration / context


def cfg_of(uid):
    return read_json(ROOT / 'configs' / f'{uid}.json')


def code_hashes():
    return {Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r35' / '*.py')))}


class Ctx:
    """per-condition data (+ local encoders/raw features for L units), loaded once per process."""
    def __init__(self, cond, device, need_local, fit_maps=None):
        self.cond, self.dev = cond, device
        self.d = load_cond(cond, device)
        self.fit_maps = fit_maps or {'observed': self.d['obs_np']}
        tab = self.d['dev_tab']
        self.pos_i, self.pos_t, self.ai_dev, self.at_dev = EV.dev_list_positions(tab)
        self.need_local = need_local
        if need_local:
            self.raw = load_raw(device)
            self.enc0, self.enc_info = load_local_encoders(cond, device)
        nat, _, rk = EV.full_metrics(self.d['native_dev'])
        self.nat_metrics, self.nat_rk = nat, rk


def set_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================ scoring (A/B: records, full DEV)

@torch.no_grad()
def g_full_dev_peer(h, d, p, chunk=8):
    """per-peer DEV [1014,5070] score in float64: affine on the exact native shard score (ID: of the transformed
    vectors) + residual (fp32)."""
    x, y = d[('dev', 'img', p)], d[('dev', 'cap', p)]
    a, b = (float(t) for t in h.affine.ab())
    if h.kind == 'ID':
        return a * shard_mm(h.tx(x), h.ty(y)) + b
    g = a * d['dev_s'][p] + b
    m = len(y)
    for i in range(0, len(x), chunk):
        xi = x[i:i + chunk]; n = len(xi)
        xx = xi[:, None, :].expand(n, m, -1).reshape(n * m, -1); yy = y[None].expand(n, m, -1).reshape(n * m, -1)
        s = (xx * yy).sum(-1)
        g[i:i + n] += h.out(h.glob.hidden(xx, yy, s)).reshape(n, m).double()
    return g


@torch.no_grad()
def g_pairs_records(h, d, p, img_idx, cap_idx, chunk=16384):
    x_all, y_all = d[('fit', 'img', p)], d[('fit', 'cap', p)]
    out = []
    for i in range(0, len(img_idx), chunk):
        ii = torch.as_tensor(img_idx[i:i + chunk], device=x_all.device); cc = torch.as_tensor(cap_idx[i:i + chunk], device=x_all.device)
        x, y = x_all[ii], y_all[cc]
        if h.kind == 'ID':
            a, b = h.affine.ab(); out.append((a * (h.tx(x) * h.ty(y)).sum(-1) + b).double())
        else:
            out.append(h.paired(x, y, (x * y).sum(-1)).double())
    return torch.cat(out).cpu().numpy()


def readout_records(heads, ctx, final=False):
    """A/B readout at a checkpoint step: full-DEV ranking (+ tie bounds), J (record U-statistic) raw and with the
    common FIT-only positive affine calibration, FIT-eval J, saturation."""
    d, L = ctx.d, ctx.d['locks']
    t0 = time.time()
    heads.eval()
    gd = {p: g_full_dev_peer(heads[p], d, p) for p in PEERS}
    s = ((gd['A'] + gd['B']) / 2)
    s_np = s.cpu().numpy()
    rec = {}
    m, tb, rk = EV.full_metrics(s_np)
    rec['dev_retrieval'] = m; rec['dev_retrieval_tie_bounded'] = tb
    for name, mp in (('dev_observed', d['dev_obs']), ('dev_clean', d['dev_clean'])):
        per = {p: EV.j_from_matrix(gd[p], mp) for p in PEERS}
        rec[name] = dict(peer=per, J_mean_of_peers=float(np.mean([v['J'] for v in per.values()])),
                         J_of_peer_mean_score=EV.j_from_matrix(s, mp)['J'])
    # FIT: calibration and J-eval sets (observed map; oracle process adds its truth map)
    def fit_set(pk, qk, imap):
        ip, cp = imap[L[pk]], L[pk]; iq, cq = imap[L[qk][:, 0]], L[qk][:, 1]
        gp = np.mean([g_pairs_records(heads[p], d, p, ip, cp) for p in PEERS], 0)
        gq = np.mean([g_pairs_records(heads[p], d, p, iq, cq) for p in PEERS], 0)
        g = np.concatenate([gp, gq]); wp = np.r_[np.full(len(gp), 1 / len(gp)), np.zeros(len(gq))]; wq = np.r_[np.zeros(len(gp)), np.full(len(gq), 1 / len(gq))]
        return g, wp, wq
    cal = {}
    for mname, imap in ctx.fit_maps.items():
        g_ev, wp_ev, wq_ev = fit_set('ev_p', 'ev_q', imap)
        rec[f'fit_{mname}_J_raw'] = EV.j_weighted(g_ev, wp_ev, wq_ev)
        if mname == 'observed':
            g_c, wp_c, wq_c = fit_set('cal_p', 'cal_q', imap)
            for crit in ('balanced_logistic', 'max_J'):
                f = EV.fit_affine(g_c, wp_c, wq_c, crit)
                z = f['a'] * s + f['b']
                cal[crit] = dict(a=f['a'], b=f['b'], fit_value=f['value'],
                                 fit_observed_J_eval=EV.j_weighted(f['a'] * g_ev + f['b'], wp_ev, wq_ev),
                                 dev_observed_J=EV.j_from_matrix(z, d['dev_obs'])['J'], dev_clean_J=EV.j_from_matrix(z, d['dev_clean'])['J'],
                                 dev_rank_invariant=EV.rank_invariance(s_np, f['a'], f['b']) if final else None)
    rec['calibration_affine_fit_only'] = cal
    rec['T_saturation_dev_peer_mean'] = EV.sat(s_np.ravel()[::97])
    rec['heads_ab'] = {p: [float(t) for t in heads[p].affine.ab()] for p in PEERS}
    rec['readout_seconds'] = time.time() - t0
    return rec, s_np, rk


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
                 gt=(np.arange(N_DEV_CAP) // 5, np.arange(N_DEV_CAP)),
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
    pos_i = g_gt.reshape(N_DEV_IMG, 5).max(1)
    hard_i = np.take_along_axis(gi_top, np.stack([np.nonzero((r // 5) != q)[0][:10] for q, r in enumerate(tab['i_top'])]), 1)
    hard_t = np.take_along_axis(gt_top, np.stack([np.nonzero(r != c // 5)[0][:10] for c, r in enumerate(tab['t_top'])]), 1)
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
    rec['T_saturation_dev_top100'] = EV.sat(np.concatenate([gi_top.ravel(), gt_top.ravel()[::5]]))
    rec['heads_ab'] = {p: [float(t) for t in heads[p].affine.ab()] for p in PEERS}
    rec['readout_seconds'] = time.time() - t0
    return rec, dict(gi_top=gi_top, gt_top=gt_top, g_gt=g_gt), rk


# ============================================================================ training


def make_opt(cfg, heads, encs):
    o = cfg['optimizer']
    groups = [dict(params=list(heads.parameters()), lr=o['lr_head'])]
    if cfg.get('d_mode') == 'joint':
        groups.append(dict(params=[q for p in PEERS for q in encs[p].trainable()], lr=o['lr_local']))
    return torch.optim.AdamW(groups, betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'])


def peer_params(cfg, heads, encs, p):
    ps = list(heads[p].parameters())
    if cfg.get('d_mode') == 'joint':
        ps += encs[p].trainable()
    return ps


def step_records(cfg, heads, d, pool, imap, rng, pw, qw):
    B = cfg['batch_size']
    idx = pool[rng.integers(0, len(pool), B)]
    ii = imap[idx]
    it, iit = torch.as_tensor(idx, device=pw.device), torch.as_tensor(ii, device=pw.device)
    loss, jb = 0.0, 0.0
    for p in PEERS:
        g = heads[p].matrix(d[('fit', 'img', p)][iit], d[('fit', 'cap', p)][it])
        loss = loss + .5 * loss_records(cfg['objective'], g, pw, qw)
        jb += .5 * float(j_value(g.detach(), pw, qw))
    stats = dict(dup_record_pairs=int(B - len(np.unique(idx))), same_image_offdiag_pairs=int(((ii[:, None] == ii[None, :]).sum() - B) // 2))
    return loss, jb, idx, stats


def step_lists(cfg, heads, encs, ctx, rng):
    d = ctx.d
    (qi, caps, ai), (qc, imgs, at) = sample_lists(rng, d['obs_np'], d['cand_i2t'], d['cand_t2i'], cfg['n_queries'])[0]
    img = np.concatenate([np.repeat(qi, N_SLOTS), imgs.ravel()]); cap = np.concatenate([caps.ravel(), np.repeat(qc, N_SLOTS)])
    A_i = torch.as_tensor(ai, device=ctx.dev); A_t = torch.as_tensor(at, device=ctx.dev)
    n = len(qi) * N_SLOTS
    joint = cfg.get('d_mode') == 'joint'
    loss, jb = 0.0, 0.0
    for p in PEERS:
        h = heads[p]
        proj = project_items(h, encs[p], ctx.raw, 'fit', img, cap, grad=True, enc_grad=joint) if h.local is not None else None
        g = g_pairs_lists(h, encs[p] if encs else None, ctx, p, 'fit', img, cap, grad=True, enc_grad=joint, proj=proj, chunk=1 << 30)
        gi, gt = g[:n].reshape(-1, N_SLOTS), g[n:].reshape(-1, N_SLOTS)
        loss = loss + .5 * (.5 * loss_list(cfg['objective'], gi, A_i) + .5 * loss_list(cfg['objective'], gt, A_t))
        with torch.no_grad():
            jb += .5 * float(.5 * list_rows(gi, A_i, 'J').mean() + .5 * list_rows(gt, A_t, 'J').mean())
    stats = dict(n_pos_slots_i2t=float(ai.astype(bool).sum(1).mean()), n_unique_images=int(len(np.unique(img))), n_unique_caps=int(len(np.unique(cap))))
    return loss, jb, (img, cap), stats


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state, torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def save_ckpt(path, cfg, step, heads, encs, opt, rng, lineage):
    obj = dict(step=step, unit_id=cfg['unit_id'], heads=heads.state_dict(), optimizer=opt.state_dict(), rng=rng_state(rng), lineage=lineage,
               config=cfg)
    if cfg.get('d_mode') == 'joint':
        obj['local_encoders_trainable'] = {p: {n: v for n, v in encs[p].state_dict().items() if not n.startswith('embed.')} for p in PEERS}
    tmp = str(path) + '.tmp'; torch.save(obj, tmp); os.replace(tmp, path)


def run_unit(cfg, ctx, imap=None, pool=None, out_dir=None, steps=None, ckpt_steps=None, log=print, dev_nan=False, stop_at=None, no_readout=False):
    """Train one unit to `steps` (default cfg['steps']) with readouts+checkpoints at ckpt_steps; resumes from the
    latest checkpoint in out_dir. imap/pool: record->image map and record pool (A: observed/all, TRUST: observed/
    subset; ORACLE: injected by the oracle runner). stop_at: stop after that many updates (resume test)."""
    steps = cfg['steps'] if steps is None else steps
    ckpt_steps = sorted(set(cfg['ckpt_steps'] if ckpt_steps is None else ckpt_steps) | {steps})
    out = Path(out_dir) if out_dir else RUNS / cfg['unit_id']
    out.mkdir(parents=True, exist_ok=True)
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
    block = cfg['block']
    lists = block in ('C', 'D')
    a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
    heads = build_heads(cfg['structure'], a0, b0, cfg['init_seed']).to(dev)
    encs = None
    lineage = dict(cache_manifest_sha256=d['manifest_sha256'], prep_manifest_sha256=d['prep_manifest_sha256'],
                   candidate_table_sha256=d.get('cand_sha256') if lists else None, a0b0=d['a0b0'],
                   parent_H=d['manifest']['parent'], code_sha256=code_hashes(), config_sha256=cfg.get('_config_sha256'))
    if ctx.need_local:
        encs = torch.nn.ModuleDict({p: _copy_enc(ctx.enc0[p]) for p in PEERS})
        lineage['local_encoder_source'] = ctx.enc_info
    if block == 'D':
        par = torch.load(cfg['parent_ckpt'], map_location=dev, weights_only=False)
        assert par['step'] == cfg.get('parent_step', 8000) and par['unit_id'] == cfg['parent_unit']
        heads.load_state_dict(par['heads'])
        lineage['parent_ckpt'] = dict(path=cfg['parent_ckpt'], sha256=sha256(cfg['parent_ckpt']), step=par['step'])
        if cfg['d_mode'] == 'head_control':
            for p in PEERS:
                for q in encs[p].parameters():
                    q.requires_grad_(False)
    init_hash = state_hash(heads)
    opt = make_opt(cfg, heads, encs)
    rng = np.random.default_rng(cfg['sampler_seed'])
    if not lists:
        pw, qw = pq_weights(cfg['batch_size'], dev)
    start = 0
    done_ck = sorted(int(re.search(r'ckpt_step(\d+)\.pt$', f).group(1)) for f in glob.glob(str(out / 'ckpt_step*.pt')))
    done_ck = [k for k in done_ck if (out / f'eval_step{k}.json').exists() and k <= steps]
    if done_ck:
        k = done_ck[-1]
        ck = torch.load(out / f'ckpt_step{k}.pt', map_location=dev, weights_only=False)
        heads.load_state_dict(ck['heads']); opt.load_state_dict(ck['optimizer'])
        if cfg.get('d_mode') == 'joint':
            for p in PEERS:
                encs[p].load_state_dict({**encs[p].state_dict(), **ck['local_encoders_trainable'][p]})
        # RNG states must be CPU ByteTensors; torch.load(map_location=dev) moved them to the GPU (gate 1081274 resume TypeError)
        rng.bit_generator.state = ck['rng']['numpy']; torch.set_rng_state(ck['rng']['torch_cpu'].cpu())
        if ck['rng']['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng']['torch_cuda']])
        start = k
        log(f"resume {cfg['unit_id']} from step {k}")
    hist = read_json(out / 'TRAIN_HISTORY.json') if (start and (out / 'TRAIN_HISTORY.json').exists()) else dict(windows=[], train_seconds=0.0, eval_seconds=0.0, batch_hash=[], stats_sum={})
    hist['windows'] = [w for w in hist['windows'] if w['end_step'] <= start]
    win = []; t_train = 0.0; t_eval = 0.0
    chain = bytes.fromhex(hist['batch_hash'][-1]) if hist['batch_hash'] else b''   # sha256 chain over batch indices

    def upd(chain, *arrs):
        h = hashlib.sha256(chain)
        for x in arrs:
            h.update(np.ascontiguousarray(x).tobytes())
        return h.digest()
    for step in range(start, steps + 1):
        if step in ckpt_steps and not (step == start and done_ck):
            te = time.time()
            if no_readout:   # technical gate runs only (identity/determinism/isolation/resume): no DEV readout
                rec, arrs, rk = dict(dev_retrieval=dict(rsum=float('nan')), readout_seconds=0.0, no_readout=True), None, None
            elif lists:
                rec, arrs, rk = readout_lists(heads, encs, ctx, final=(step == steps))
            else:
                rec, s_np, rk = readout_records(heads, ctx, final=(step == steps))
            rec['step'] = step
            if rk is not None:
                rec['transitions_vs_H'] = EV.transitions(rk, ctx.nat_rk, rk, ctx.nat_rk, d['dev_tab'])
            if win:
                w = np.array([(x[0], x[1], x[2]) for x in win])
                rec['train_window'] = dict(n=len(win), loss_mean=float(w[:, 0].mean()), batch_J_mean=float(w[:, 1].mean()), grad_norm_mean=float(w[:, 2].mean()), loss_last=float(w[-1, 0]))
                hist['windows'].append(dict(end_step=step, **rec['train_window']))
                win = []
            if step == steps and rk is not None:
                if lists:
                    np.savez(out / f'dev_scores_step{step}.npz', **arrs)
                else:
                    np.save(out / f'dev_scores_step{step}.npy', s_np)
                np.savez(out / f'dev_ranks_step{step}.npz', **rk)
            if step == 0 and block == 'D' and rk is not None:
                np.savez(out / 'dev_scores_step0.npz', **arrs)
            t_eval += time.time() - te
            write_json(out / f'eval_step{step}.json', rec)
            log(f"{cfg['unit_id']} step {step} rsum {rec['dev_retrieval']['rsum']:.2f} ({rec['readout_seconds']:.1f}s)")
            if not dev_nan:
                save_ckpt(out / f'ckpt_step{step}.pt', cfg, step, heads, encs, opt, rng, lineage)
            hist['train_seconds'] += t_train; hist['eval_seconds'] += t_eval; t_train = t_eval = 0.0
            hist['batch_hash'].append(chain.hex()); hist['last_step'] = step
            write_json(out / 'TRAIN_HISTORY.json', hist)
        if step == steps or (stop_at is not None and step == stop_at):
            break
        t0 = time.time()
        heads.train()
        if lists:
            loss, jb, idx, st = step_lists(cfg, heads, encs, ctx, rng)
            chain = upd(chain, idx[0], idx[1])
        else:
            loss, jb, idx, st = step_records(cfg, heads, d, pool, imap, rng, pw, qw)
            chain = upd(chain, idx)
        for k2, v2 in st.items():
            hist['stats_sum'][k2] = hist['stats_sum'].get(k2, 0) + v2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gns = []
        for p in PEERS:
            ps = [q for q in peer_params(cfg, heads, encs, p) if q.grad is not None]
            gns.append(float(torch.nn.utils.clip_grad_norm_(ps, cfg['clip_per_peer'])))
        if not (np.isfinite(gns).all() and torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite at step {step}')
        opt.step()
        if cfg.get('d_mode') == 'joint':
            for p in PEERS:
                encs[p].set_modes()
        win.append((float(loss), jb, float(np.mean(gns))))
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t_train += time.time() - t0
    return dict(heads=heads, encs=encs, init_hash=init_hash, final_hash=state_hash(heads) + (state_hash(*[encs[p] for p in PEERS]) if cfg.get('d_mode') == 'joint' else ''),
                batch_hash=chain.hex(), hist=hist, out=out)


def _copy_enc(e):
    import copy
    c = copy.deepcopy(e); c.set_modes(); return c


class _shallow:
    def __init__(self, ctx, **kw):
        self.__dict__.update(ctx.__dict__); self.__dict__.update(kw)


def finalize(cfg, ctx, res, t_wall):
    out = res['out']
    steps = cfg['steps']
    curves = [read_json(out / f'eval_step{k}.json') for k in sorted(set(cfg['ckpt_steps']) | {steps})]
    fin = curves[-1]
    dev = ctx.dev
    result = dict(unit_id=cfg['unit_id'], status='COMPLETE', config=cfg, step=steps, final=fin, curves=curves,
                  native_H_dev=ctx.nat_metrics,
                  diff_vs_native={k: fin['dev_retrieval'][k] - ctx.nat_metrics[k] for k in ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')},
                  identity=dict(init_state_sha256=res['init_hash'], final_state_sha256=res['final_hash'], batch_index_sha256=res['batch_hash'],
                                code_sha256=code_hashes()),
                  batch_stats=res['hist']['stats_sum'],
                  cost=dict(params_trainable=n_params(res['heads']) + (sum(q.numel() for p in PEERS for q in res['encs'][p].trainable()) if cfg.get('d_mode') == 'joint' else 0),
                            train_seconds=res['hist']['train_seconds'], eval_seconds=res['hist']['eval_seconds'], unit_wall_seconds_this_process=t_wall,
                            peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(dev)) if dev.type == 'cuda' else 0,
                            pairs_per_update_per_peer=(cfg['batch_size'] ** 2 if cfg['block'] in ('A', 'B') else 2 * cfg['n_queries'] * N_SLOTS)),
                  env=dict(gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu', host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'),
                           torch=torch.__version__, deterministic=torch.are_deterministic_algorithms_enabled()),
                  test_guard=guard_record(), finished_utc=utc())
    if cfg['block'] == 'D':
        pz = Path(cfg['parent_ckpt']).parent / f"dev_scores_step{cfg.get('parent_step', 8000)}.npz"
        z0 = np.load(out / 'dev_scores_step0.npz'); zp = np.load(pz)
        result['step0_equals_parent_scores'] = dict(bitwise=all(np.array_equal(z0[k], zp[k]) for k in ('gi_top', 'gt_top', 'g_gt')),
                                                    max_abs=float(max(np.abs(z0[k] - zp[k]).max() for k in ('gi_top', 'gt_top', 'g_gt'))), parent_scores=str(pz))
    result['artifacts'] = {f.name: sha256(f) for f in sorted(out.glob(f'*step{steps}*'))}
    write_json(out / 'RESULT.json', result)
    write_json(out / 'DONE.json', dict(status='COMPLETE', finished_utc=utc(), rsum=fin['dev_retrieval']['rsum'], job=os.environ.get('SLURM_JOB_ID'),
                                       gpu=result['env']['gpu']))
    return result


def ensure_parent(cfg):
    if cfg['block'] == 'D':
        assert Path(cfg['parent_ckpt']).exists(), f"parent checkpoint missing: {cfg['parent_ckpt']}"


def check_frozen(uids):
    frozen = read_json(ROOT / 'PROTOCOL_FROZEN.json')
    for f, h in frozen['code_sha256'].items():
        if f in frozen.get('code_not_bound', []):
            continue
        assert sha256(ROOT / 'r35' / f) == h, f'code differs from frozen protocol: {f}'
    for u in uids:
        assert sha256(ROOT / 'configs' / f'{u}.json') == frozen['config_sha256'][u], f'config differs from frozen: {u}'
    assert read_json(ROOT / 'GPU_GATE.json')['ALL_PASS'], 'GPU gate not passed'


def main(oracle_imap=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--units', required=True)
    ap.add_argument('--gate-out', default=''); ap.add_argument('--gate-steps', type=int, default=0)
    a = ap.parse_args()
    gate = bool(a.gate_out)
    install_guards()
    if oracle_imap is None:
        assert_no_oracle_loaded()
    set_determinism()
    uids = a.units.split(',')
    cfgs = [cfg_of(u) for u in uids]
    for c in cfgs:
        if oracle_imap is None:
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
        out = (Path(a.gate_out) / cfg['unit_id']) if gate else RUNS / cfg['unit_id']
        if (out / 'DONE.json').exists():
            print('skip DONE', cfg['unit_id']); continue
        try:
            ensure_parent(cfg)
            key = (cfg['cond'], cfg['structure'] == 'L')
            if key not in ctxs:
                ctxs[key] = Ctx(cfg['cond'], device, need_local=key[1])
                if oracle_imap is not None:
                    ctxs[key].fit_maps = {'observed': ctxs[key].d['obs_np'], 'truth': oracle_imap}
            ctx = ctxs[key]
            if cfg['supervision'] == 'TRUST':
                pool, imap = ctx.d['trust_subset'], ctx.d['obs_np']
            elif cfg['supervision'] == 'ORACLE':
                pool, imap = np.arange(N_TRAIN_CAP), oracle_imap
            else:
                pool, imap = np.arange(N_TRAIN_CAP), ctx.d['obs_np']
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            if gate:
                res = run_unit(cfg, ctx, imap=imap, pool=pool, out_dir=out, steps=a.gate_steps, ckpt_steps=[0, a.gate_steps])
                cfg = dict(cfg, steps=a.gate_steps, ckpt_steps=[0, a.gate_steps])
            else:
                res = run_unit(cfg, ctx, imap=imap, pool=pool)
            r = finalize(cfg, ctx, res, time.time() - t0)
            print('R35_UNIT_DONE', cfg['unit_id'], 'rsum', r['final']['dev_retrieval']['rsum'], 'native', ctx.nat_metrics['rsum'], flush=True)
        except Exception:
            tb = traceback.format_exc(); print(tb, flush=True)
            write_json(out / f"FAILED_{os.environ.get('SLURM_JOB_ID', 'local')}.json", dict(unit=cfg['unit_id'], traceback=tb, utc=utc()))
            failed.append(cfg['unit_id'])
    print('R35_JOB_END failed', failed)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
