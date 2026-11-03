"""diag6 items 3 + 4 (GPU; zero new training results; no TEST file opened, trainer-role guards of the unit's own code tree).
Imports the frozen code tree of the unit (r38f / r38c / r37s_critic train_unit.py, read-only) and works on the saved checkpoints only.

 replay  (item 4b): for every segment [k, k') between saved checkpoints, restore ckpt k exactly as the frozen resume path does (heads, nulls,
         AdamW state, numpy/torch RNG) and re-execute the frozen training step (train_unit.step_lists -> loss.backward -> per-peer
         clip_grad_norm_ -> schedule -> AdamW) for the RECORDED batches; record per update and peer the pre-clip norm of the clipped
         parameter set (head + null), of the head part alone and of the null part. Fidelity: batch-index hash chain at k' must equal the
         recorded TRAIN_HISTORY chain; head/null state at k' is compared with the saved ckpt k' (bitwise flag + max |diff|). The replay
         state is discarded; nothing is written to the unit directory.
 grad    (item 3): at every saved checkpoint, on 64 fixed FIT batches (the locked list-eval records lev_i/lev_t, N 33-slot lists, rows
         16b..16b+15 of each direction) per peer: gradients g_i2t = d(0.25 L_i2t)/dtheta, g_t2i = d(0.25 L_t2i)/dtheta (the training
         loss split of train_unit.step_lists), cos per parameter group, Euclidean share <g_d, G>/|G|^2 (G = g_i2t + g_t2i) and the
         first-order share of each direction in the actual AdamW update Delta that the saved optimizer state would take on that batch
         (per-peer clip at 2.0 as in training, decoupled weight decay included; <g_d, Delta>/<G, Delta>). Full-set rows = mean over the
         64 batches (gradients averaged before the angle).
 null    (item 4a): at steps 0/1000/8000, 4 fixed batches: dL/dg of the real slots and the head-parameter gradients with the null
         scalars shifted by delta in {-4,-1,+1,+4} vs unshifted (pre-clip): max |diff|, bitwise flags; and the per-peer clip coefficient
         under each shift (the only coupling path through the shared clip).
Usage: python -B grad_replay.py --parent <parent_key> [--modes replay,grad,null] [--units u1,u2]"""
from __future__ import annotations
import argparse, hashlib, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import units6 as U6
ap = argparse.ArgumentParser(); ap.add_argument('--parent', required=True); ap.add_argument('--modes', default='replay,grad,null'); ap.add_argument('--units', default='')
A = ap.parse_args()
P = U6.PARENTS[A.parent]
sys.path.insert(0, str(U6.TREES[P['tree']]))
import numpy as np
import torch
import train_unit as TU
from common import install_guards, guard_record, assert_no_oracle_loaded, PEERS
import data36 as D36
import r38ext as X
from heads import build_heads, NullSlots
from objectives import loss_list
install_guards(role='trainer'); assert_no_oracle_loaded()
TU.set_determinism()
dev = torch.device('cuda')
MODES = A.modes.split(',')
OUT = U6.OUT6 / 'grad_replay' / A.parent; OUT.mkdir(parents=True, exist_ok=True)
ctx = TU.Ctx(P['cond'], dev, need_local=False)
a0, b0 = ctx.d['a0b0']['a0'], ctx.d['a0b0']['b0']

units = [u for u in U6.unit_index() if u['parent'] == A.parent and u['route'] == 'N']
if A.units:
    keep = set(A.units.split(',')); units = [u for u in units if u['unit'] in keep]


def build(cfg):
    heads = build_heads(cfg['structure'], a0, b0, cfg['init_seed']).to(dev)
    nulls = NullSlots().to(dev) if cfg['supervision'] in D36.NULL_SUPS else None
    opt = TU.make_opt(cfg, heads, nulls)
    return heads, nulls, opt


def load_ck(path, heads, nulls, opt=None, rng=None):
    ck = torch.load(path, map_location=dev, weights_only=False)
    heads.load_state_dict(ck['heads'])
    if nulls is not None:
        nulls.load_state_dict(ck['nulls'])
    if opt is not None:
        opt.load_state_dict(ck['optimizer'])
    if rng is not None:   # frozen resume path (train_unit.run_unit)
        rng.bit_generator.state = ck['rng']['numpy']; torch.set_rng_state(ck['rng']['torch_cpu'].cpu())
        if ck['rng']['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in ck['rng']['torch_cuda']])
    return ck


def upd(chain, *arrs):
    h = hashlib.sha256(chain)
    for x in arrs:
        h.update(np.ascontiguousarray(x).tobytes())
    return h.digest()


def state_diff(mod, sd):
    mx, bit = 0.0, True
    for k, v in mod.state_dict().items():
        w = sd[k].to(v.device)
        bit &= bool(torch.equal(v, w)); mx = max(mx, float((v.double() - w.double()).abs().max()))
    return mx, bit


def replay(u, cfg, hist):
    ck_steps = sorted(set(cfg['ckpt_steps']) | {cfg['steps']})
    assert len(hist['batch_hash']) == len(ck_steps), (len(hist['batch_hash']), ck_steps)
    heads, nulls, opt = build(cfg)
    rng = np.random.default_rng(cfg['sampler_seed']); rng2 = np.random.default_rng(cfg['sampler_seed2']) if cfg.get('sampler_seed2') else None
    sched = cfg['optimizer'].get('schedule', 'constant'); clip = cfg['clip_per_peer']
    pool = TU.pool_of(cfg, ctx)
    T = cfg['steps']
    tot = np.full((T, 2), np.nan); hd = np.full((T, 2), np.nan); nl = np.full((T, 2), np.nan); lrs = np.full(T, np.nan)
    segs = []
    for si, (k, k2) in enumerate(zip(ck_steps[:-1], ck_steps[1:])):
        load_ck(Path(u['dir']) / f'ckpt_step{k}.pt', heads, nulls, opt, rng)
        if rng2 is not None:
            raise NotImplementedError('rng2 units are not N units')
        chain = bytes.fromhex(hist['batch_hash'][si])
        t0 = time.time()
        for step in range(k, k2):
            heads.train()
            loss, jb, idx, st, gl, cx = TU.step_lists(cfg, heads, nulls, None, ctx, rng, rng2, pool, None)
            chain = upd(chain, *idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for j, p in enumerate(PEERS):
                ps = [q for q in TU.peer_params(heads, nulls, p) if q.grad is not None]
                with torch.no_grad():
                    hsq = sum(float(q.grad.double().pow(2).sum()) for q in heads[p].parameters() if q.grad is not None)
                    nsq = float(nulls.v[p].grad.double().pow(2).sum()) if (nulls is not None and nulls.v[p].grad is not None) else 0.0
                tot[step, j] = float(torch.nn.utils.clip_grad_norm_(ps, clip)); hd[step, j] = math.sqrt(hsq); nl[step, j] = math.sqrt(nsq)
            if not (np.isfinite(tot[step]).all() and torch.isfinite(loss)):
                raise FloatingPointError(f'nonfinite at step {step}')
            if sched == 'cosine':
                for pg in opt.param_groups:
                    pg['lr'] = cfg['optimizer']['lr_head'] * X.lr_multiplier(step, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor'])
            lrs[step] = opt.param_groups[0]['lr']
            opt.step()
        ck2 = torch.load(Path(u['dir']) / f'ckpt_step{k2}.pt', map_location=dev, weights_only=False)
        hm, hb = state_diff(heads, ck2['heads'])
        nm, nb = state_diff(nulls, ck2['nulls']) if nulls is not None else (0.0, True)
        segs.append(dict(start=k, end=k2, chain_equal=chain.hex() == hist['batch_hash'][si + 1], head_state_bitwise=hb, head_state_max_abs_diff=hm,
                         null_state_bitwise=nb, null_state_max_abs_diff=nm, seconds=time.time() - t0))
    np.savez(OUT / f'clip_{u["unit"]}.npz', total_preclip=tot, head_preclip=hd, null_preclip=nl, lr=lrs)
    trig = tot > clip   # clip_grad_norm_: coef = clip/(norm+1e-6) < 1  <=>  norm > clip - 1e-6
    trig_eps = tot + 1e-6 > clip
    only_null = trig_eps & (hd + 1e-6 <= clip)
    coef = np.minimum(1.0, clip / (tot + 1e-6))
    summ = dict(n_updates=int(T), clip=clip, trigger_rate=float(trig_eps.mean()), trigger_rate_strict_gt=float(trig.mean()),
                trigger_rate_by_peer=[float(trig_eps[:, j].mean()) for j in range(2)],
                trigger_rate_steps_lt250=float(trig_eps[:250].mean()), trigger_rate_steps_250_2000=float(trig_eps[250:2000].mean()), trigger_rate_steps_2000_8000=float(trig_eps[2000:].mean()),
                n_triggered_only_because_of_null=int(only_null.sum()), frac_triggered_only_because_of_null=float(only_null.mean()),
                mean_clip_coef_when_triggered=float(coef[trig_eps].mean()) if trig_eps.any() else None, min_clip_coef=float(coef.min()),
                preclip_norm_mean=float(tot.mean()), preclip_norm_p50=float(np.median(tot)), preclip_norm_p99=float(np.quantile(tot, .99)), preclip_norm_max=float(tot.max()),
                null_share_of_norm_sq_mean=float(np.mean(nl ** 2 / np.maximum(tot ** 2, 1e-300))),
                fidelity=dict(all_chains_equal=all(s['chain_equal'] for s in segs), all_states_bitwise=all(s['head_state_bitwise'] and s['null_state_bitwise'] for s in segs),
                              max_head_state_abs_diff=max(s['head_state_max_abs_diff'] for s in segs), max_null_state_abs_diff=max(s['null_state_max_abs_diff'] for s in segs)),
                segments=segs)
    # cross-check with the logged window means of the pre-clip norm (eval_step*.json train_window.grad_norm_mean = mean over peers and window)
    xc = []
    for k, k2 in zip(ck_steps[:-1], ck_steps[1:]):
        e = U6.read_json(Path(u['dir']) / f'eval_step{k2}.json').get('train_window')
        if e:
            xc.append(dict(end=k2, logged=e['grad_norm_mean'], replay=float(tot[k:k2].mean()), abs_diff=abs(e['grad_norm_mean'] - float(tot[k:k2].mean()))))
    summ['window_grad_norm_crosscheck'] = xc
    return summ


def fixed_lists():
    (qi, caps, ai), (qc, imgs, at) = D36.locked_null_lists(ctx.d, 'lev_i', 'lev_t')
    return (qi, caps, ai), (qc, imgs, at)


FL = fixed_lists()
NB, BS = 64, 16


def batch(b):
    (qi, caps, ai), (qc, imgs, at) = FL
    s = slice(b * BS, (b + 1) * BS)
    return (qi[s], caps[s], ai[s]), (qc[s], imgs[s], at[s])


def peer_losses(cfg, heads, nulls, p, lists, keep_scores=False):
    """frozen step_lists score assembly for one peer on given lists; returns 0.25*L_i2t, 0.25*L_t2i (the training split) and scores."""
    (qi, caps, ai), (qc, imgs, at) = lists
    img, cap, ri, rt = TU._pairs(lists)
    n_i = int(ri.sum())
    dirs = np.concatenate([np.zeros(n_i, np.int64), np.ones(len(img) - n_i, np.int64)])
    ri_t, rt_t = torch.as_tensor(ri, device=dev), torch.as_tensor(rt, device=dev)
    g = TU.g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, grad=True, chunk=1 << 30, direction=dirs)
    gi = torch.zeros(caps.shape, device=dev, dtype=g.dtype).masked_scatter(ri_t, g[:n_i])
    gt = torch.zeros(imgs.shape, device=dev, dtype=g.dtype).masked_scatter(rt_t, g[n_i:])
    gi = torch.where(ri_t, gi, nulls.score(p, 0, 1)[0]); gt = torch.where(rt_t, gt, nulls.score(p, 1, 1)[0])
    A_i = torch.as_tensor(ai, device=dev); A_t = torch.as_tensor(at, device=dev)
    return .25 * loss_list(cfg['objective'], gi, A_i), .25 * loss_list(cfg['objective'], gt, A_t), gi, gt, ri_t, rt_t


def named_peer(heads, nulls, p):
    nm = [(X.group_of(n), q) for n, q in heads[p].named_parameters()]
    if nulls is not None:
        nm.append(('null', nulls.v[p]))
    return nm


def grads(L, ps):
    g = torch.autograd.grad(L, ps, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(q) if x is None else x.detach() for x, q in zip(g, ps)]


def adamw_delta(opt, ps, G, lr, clip):
    """AdamW update (torch formula, decoupled wd) the saved optimizer state would take for gradient G after the per-peer clip."""
    gn = math.sqrt(sum(float(x.double().pow(2).sum()) for x in G))
    c = min(1.0, clip / (gn + 1e-6))
    pg = opt.param_groups[0]; b1, b2 = pg['betas']; eps = pg['eps']; wd = pg['weight_decay']
    D = []
    for q, x in zip(ps, G):
        st = opt.state.get(q, {})
        m = st.get('exp_avg', torch.zeros_like(q)); v = st.get('exp_avg_sq', torch.zeros_like(q)); t = float(st.get('step', torch.tensor(0.))) + 1
        x = (x * c).double(); m2 = b1 * m.double() + (1 - b1) * x; v2 = b2 * v.double() + (1 - b2) * x * x
        bc1 = 1 - b1 ** t; bc2 = 1 - b2 ** t
        D.append(-lr * wd * q.detach().double() - (lr / bc1) * m2 / (v2.sqrt() / math.sqrt(bc2) + eps))
    return D, c, gn


def dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def grad_diag(u, cfg):
    ck_steps = sorted(set(cfg['ckpt_steps']) | {cfg['steps']})
    heads, nulls, opt = build(cfg)
    sched = cfg['optimizer'].get('schedule', 'constant'); clip = cfg['clip_per_peer']
    rows = []
    for k in ck_steps:
        load_ck(Path(u['dir']) / f'ckpt_step{k}.pt', heads, nulls, opt)
        heads.train()
        upd_idx = min(k, cfg['steps'] - 1)
        lr = cfg['optimizer']['lr_head'] * (X.lr_multiplier(upd_idx, cfg['steps'], 'cosine', cfg['optimizer']['cosine_floor']) if sched == 'cosine' else 1.0)
        for p in PEERS:
            nm = named_peer(heads, nulls, p); ps = [q for _, q in nm]; grp = [g for g, _ in nm]
            acc_i = [torch.zeros_like(q) for q in ps]; acc_t = [torch.zeros_like(q) for q in ps]
            per = []
            for b in range(NB):
                Li, Lt, *_ = peer_losses(cfg, heads, nulls, p, batch(b))
                gi = grads(Li, ps); gt = grads(Lt, ps)
                for z, x in zip(acc_i, gi): z += x
                for z, x in zip(acc_t, gt): z += x
                per.append(stats(gi, gt, grp, ps, opt, lr, clip))
            gi = [x / NB for x in acc_i]; gt = [x / NB for x in acc_t]
            full = stats(gi, gt, grp, ps, opt, lr, clip)
            for key in full:
                vals = np.array([r[key] for r in per], float)
                rows.append(dict(unit=u['unit'], parent=u['parent'], architecture=u['architecture'], schedule=u['schedule'], objective=u['objective'],
                                 source=u['source'], step=k, peer=p, quantity=key, full_set=full[key], batch_mean=float(np.nanmean(vals)) if np.isfinite(vals).any() else float('nan'),
                                 batch_frac_negative=float(np.mean(vals[np.isfinite(vals)] < 0)) if np.isfinite(vals).any() else float('nan'), n_batches=NB))
    return rows


def stats(gi, gt, grp, ps, opt, lr, clip):
    out = {}
    groups = sorted(set(grp)) + ['head', 'all']
    for gname in groups:
        sel = [j for j, g in enumerate(grp) if (g == gname) or (gname == 'all') or (gname == 'head' and g != 'null')]
        a = [gi[j] for j in sel]; b = [gt[j] for j in sel]
        na, nb = math.sqrt(dot(a, a)), math.sqrt(dot(b, b))
        out[f'cos_{gname}'] = dot(a, b) / (na * nb) if na > 0 and nb > 0 else float('nan')
        out[f'norm_i2t_{gname}'] = na; out[f'norm_t2i_{gname}'] = nb
    G = [x + y for x, y in zip(gi, gt)]
    gg = dot(G, G)
    out['euclid_share_i2t'] = dot(gi, G) / gg if gg > 0 else float('nan'); out['euclid_share_t2i'] = dot(gt, G) / gg if gg > 0 else float('nan')
    D, c, gn = adamw_delta(opt, ps, G, lr, clip)
    tot = dot(G, D)
    out['adam_share_i2t'] = dot(gi, D) / tot if tot != 0 else float('nan'); out['adam_share_t2i'] = dot(gt, D) / tot if tot != 0 else float('nan')
    nd = math.sqrt(dot(D, D)); ni, nt = math.sqrt(dot(gi, gi)), math.sqrt(dot(gt, gt))
    out['cos_i2t_vs_minus_update'] = -dot(gi, D) / (ni * nd) if ni > 0 and nd > 0 else float('nan')
    out['cos_t2i_vs_minus_update'] = -dot(gt, D) / (nt * nd) if nt > 0 and nd > 0 else float('nan')
    out['first_order_dLi2t_along_update'] = dot(gi, D); out['first_order_dLt2i_along_update'] = dot(gt, D)
    out['clip_coef'] = c; out['preclip_norm'] = gn
    return out


def null_check(u, cfg):
    heads, nulls, opt = build(cfg)
    clip = cfg['clip_per_peer']
    rows = []
    for k in (0, 1000, 8000):
        load_ck(Path(u['dir']) / f'ckpt_step{k}.pt', heads, nulls)
        heads.train()
        for b in range(4):
            for p in PEERS:
                base_v = nulls.v[p].detach().clone()
                hps = list(heads[p].parameters())
                ref = None
                for delta in (0.0, -4.0, -1.0, 1.0, 4.0):
                    with torch.no_grad():
                        nulls.v[p].copy_(base_v + delta)
                    Li, Lt, gi, gt, ri_t, rt_t = peer_losses(cfg, heads, nulls, p, batch(b))
                    L = Li + Lt
                    gs = torch.autograd.grad(L, [gi, gt] + hps + [nulls.v[p]], allow_unused=True)
                    dgi, dgt = gs[0].detach(), gs[1].detach(); ph = [x.detach() if x is not None else torch.zeros_like(q) for x, q in zip(gs[2:-1], hps)]
                    dn = gs[-1].detach()
                    real = torch.cat([dgi[ri_t], dgt[rt_t]])
                    hn = math.sqrt(sum(float(x.double().pow(2).sum()) for x in ph)); nn_ = float(dn.double().pow(2).sum()) ** .5
                    tn = math.sqrt(hn ** 2 + nn_ ** 2)
                    rec = dict(real=real, ph=ph, coef=min(1.0, clip / (tn + 1e-6)), tn=tn, hn=hn, nn=nn_)
                    if ref is None:
                        ref = rec
                    else:
                        rows.append(dict(unit=u['unit'], parent=u['parent'], architecture=u['architecture'], schedule=u['schedule'], objective=u['objective'], step=k, batch=b, peer=p, delta=delta,
                                         real_logit_grad_bitwise_equal=bool(torch.equal(rec['real'], ref['real'])),
                                         real_logit_grad_max_abs_diff=float((rec['real'] - ref['real']).abs().max()), real_logit_grad_max_abs=float(ref['real'].abs().max()),
                                         head_param_grad_bitwise_equal=all(bool(torch.equal(x, y)) for x, y in zip(rec['ph'], ref['ph'])),
                                         head_param_grad_max_abs_diff=max(float((x - y).abs().max()) for x, y in zip(rec['ph'], ref['ph'])),
                                         preclip_norm_base=ref['tn'], preclip_norm_shifted=rec['tn'], null_grad_norm_base=ref['nn'], null_grad_norm_shifted=rec['nn'],
                                         clip_coef_base=ref['coef'], clip_coef_shifted=rec['coef']))
                with torch.no_grad():
                    nulls.v[p].copy_(base_v)
    return rows


all_grad, all_null, summaries = [], [], {}
t_all = time.time()
for u in units:
    res = U6.read_json(Path(u['dir']) / 'RESULT.json'); cfg = res['config']; hist = U6.read_json(Path(u['dir']) / 'TRAIN_HISTORY.json')
    assert cfg['supervision'] == 'OBS_MISSING_NULL', cfg['supervision']
    t0 = time.time(); rec = dict(unit=u['unit'], parent=u['parent'], architecture=u['architecture'], schedule=u['schedule'], objective=u['objective'],
                                 source=u['source'], train_gpu=res['env']['gpu'], train_job=res['env']['job'])
    if 'replay' in MODES:
        rec['replay'] = replay(u, cfg, hist)
    if 'grad' in MODES:
        all_grad += grad_diag(u, cfg)
    if 'null' in MODES:
        all_null += null_check(u, cfg)
    rec['seconds'] = time.time() - t0
    summaries[u['unit']] = rec
    print('DIAG6_UNIT', u['unit'], {k: v for k, v in rec.get('replay', {}).items() if k in ('trigger_rate', 'fidelity')}, round(rec['seconds'], 1), flush=True)
    U6.write_json(OUT / 'SUMMARY.json', dict(parent=A.parent, modes=MODES, units=summaries, gpu=torch.cuda.get_device_name(0), host=os.uname().nodename,
                                            job=os.environ.get('SLURM_JOB_ID'), test_guard=guard_record(), seconds=time.time() - t_all))
if all_grad:
    U6.wcsv(OUT / 'grad_fixed_fit.csv', all_grad)
if all_null:
    U6.wcsv(OUT / 'null_invariance.csv', all_null)
print('DIAG6_JOB_END', A.parent, len(summaries), time.time() - t_all)
