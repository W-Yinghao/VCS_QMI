"""R40 section 7 zero-training mechanism readouts on existing R39 / R39X JOINT checkpoints (no training, no TEST, observed labels only;
Flickr truth is used ONLY by the separate evaluator diag7/eval_truth_f60.py on the arrays exported here).
  item 1  per unit (step 8000): DEV Recall of the frozen H (native), the final critic g (two-stage, from eval_step8000.json), the student
          cosine on the same H-top100 (two-stage), the student cosine on the full DEV gallery, and (item 5) the un-centred student cosine.
  item 2  per unit: parameter distance to the parent per modality; FIT feature change (student vs frozen-H cache cosine) on a fixed FIT subset,
          stratified by FIT present / missing (observed partner in the natural top-32 of the item as query); per-group pre-clip gradient
          norms (head, null, image, text) of the unit's own loss on 16 fixed FIT batches (separate stream [4040, 7]) split into present-row
          and missing-row parts, with the per-peer clip multiplier of the total.
  item 3  common states (the N-JOINT-J unit at steps 4000 and 8000): on the same 16 FIT batches, dL/dg of J / BCE / NCE on the same logits per
          slot class (real A>0, real A=0, null), parameter-gradient cosines between objectives per group, and the change of g on the batch
          after one unit-length normalised gradient step per objective (eta 0.01 in parameter L2) - not a training update.
  item 5  centring: re-encoding DEV in a permuted order and other chunk sizes vs the standard tables (registered fixed parent native-batch
          centring); un-centred student cosine as a geometry readout only.
  item 6  on the real batch logits: max |d(-J)/dg - (1 - T^2) dBCE/dg|, T = tanh(g/2) (implementation identity; not a robustness theorem).
Usage: python -B diag7/readouts.py <tree dir> <dataset tag> <unit_dir,...> [--common <J unit dir>]"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag'); ap.add_argument('units'); ap.add_argument('--common', default='')
ap.add_argument('--fit-subset', type=int, default=20000); ap.add_argument('--batches', type=int, default=16)
A = ap.parse_args()
sys.path.insert(0, A.tree)
import torch
import common as C
C.install_guards('trainer')
import data36 as D36
import evals as EV
import objectives as O
import r39ext as R9
import train_unit as TU
from heads import build_heads, NullSlots

W = Path(__file__).resolve().parents[1]
OUTD = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/diag7') / A.tag
OUTD.mkdir(parents=True, exist_ok=True)
dev = torch.device('cuda')
TU.set_determinism()
PEERS = C.PEERS


def lazy_joint(ctx):
    """JointInputs with the FIT region features left on the NFS memmap (zero-copy tensor; read pages on index_select) instead of a 9-44 GB copy."""
    real = R9.np

    class _NP:
        def __getattr__(self, k):
            return getattr(real, k)

        @staticmethod
        def array(x, *a, **k):
            return x
    flag = C.RAW_FIT_ON_GPU; C.RAW_FIT_ON_GPU = False; R9.np = _NP()
    try:
        return ctx.ensure_joint()
    finally:
        R9.np = real; C.RAW_FIT_ON_GPU = flag


def load_state(unit_dir, step, ctx):
    ck = torch.load(Path(unit_dir) / f'ckpt_step{step}.pt', map_location=dev, weights_only=False)
    cfg = ck['config']; d = ctx.d
    heads = build_heads(cfg['structure'], d['a0b0']['a0'], d['a0b0']['b0'], cfg['init_seed']).to(dev); heads.load_state_dict(ck['heads'])
    nulls = NullSlots().to(dev); nulls.load_state_dict(ck['nulls'])
    stud = R9.build_students(ctx.joint.ck, dev); stud.load_state_dict(ck['student'])
    heads.eval(); stud.train()
    return cfg, heads, nulls, stud


def metrics_two_stage(ctx, S):
    tab = ctx.d['dev_tab']
    gi = np.take_along_axis(S, tab['i_top'], 1)
    gt = S[tab['t_top'], np.arange(S.shape[1])[:, None]]
    m, _, _, _ = EV.composite_metrics(ctx.d['native_dev'], tab, gi, gt)
    return m


@torch.no_grad()
def dev_vectors(ctx, stud, centred=True, order=None, chunk=None):
    out = {}
    for p in PEERS:
        for kind, n in (('img', C.N_DEV_IMG), ('cap', C.N_DEV_CAP)):
            ids = np.arange(n) if order is None else order[kind]
            if centred:
                v = ctx.joint.encode(stud[p], p, 'dev', kind, ids, grad=False, chunk=chunk)
            else:
                v = torch.cat([ctx.joint.raw_encode(stud[p], 'dev', kind, torch.as_tensor(ids[i:i + 512])) for i in range(0, len(ids), 512)])
            out[(p, kind)] = v
    return out


def sim(vecs):
    return (sum(vecs[(p, 'img')] @ vecs[(p, 'cap')].T for p in PEERS) / len(PEERS)).double().cpu().numpy()


def present_flags(d):
    obs = d['obs_np']; ci = d['cand_i2t'].astype(np.int64); ct = d['cand_t2i'].astype(np.int64)
    return (obs[ci] == np.arange(C.N_TRAIN_IMG)[:, None]).any(1), (ct == obs[:, None]).any(1)


def fixed_batches(ctx, n):
    rng = np.random.default_rng([4040, 7])
    return [D36.sample('OBS_MISSING_NULL', rng, None, ctx.d, 16, None)[0] for _ in range(n)]


def batch_scores(ctx, heads, nulls, stud, lists):
    """per peer (gi, gt) [rows, 33] exactly as train_unit.step_lists (student pairs, frozen-H skip, null scalar)."""
    (qi, caps, ai), (qc, imgs, at) = lists
    img, cap, ri, rt = TU._pairs(lists); n_i = int(ri.sum())
    dirs = np.concatenate([np.zeros(n_i, np.int64), np.ones(len(img) - n_i, np.int64)])
    ri_t, rt_t = torch.as_tensor(ri, device=dev), torch.as_tensor(rt, device=dev)
    out = []
    for p in PEERS:
        g = TU.g_pairs_lists(heads[p], None, ctx, p, 'fit', img, cap, grad=True, chunk=1 << 30, direction=dirs, stu=ctx.joint.pairs(stud[p], p, 'fit', img, cap, grad=True))
        gi = torch.zeros(caps.shape, device=dev, dtype=g.dtype).masked_scatter(ri_t, g[:n_i]); gt = torch.zeros(imgs.shape, device=dev, dtype=g.dtype).masked_scatter(rt_t, g[n_i:])
        gi = torch.where(ri_t, gi, nulls.score(p, 0, 1)[0]); gt = torch.where(rt_t, gt, nulls.score(p, 1, 1)[0])
        out.append((gi, gt))
    return out, (ri, rt)


def groups(heads, nulls, stud, p):
    return dict(head=list(heads[p].parameters()), null=[nulls.v[p]], img=[q for n, q in stud[p].named_parameters() if q.requires_grad and n.startswith('img.')],
                txt=[q for n, q in stud[p].named_parameters() if q.requires_grad and n.startswith('txt.')])


def gnorm(gs):
    return float(torch.sqrt(sum(((g.double() ** 2).sum() for g in gs if g is not None), torch.zeros((), dtype=torch.float64, device=dev))))


def peer_loss(obj, gi, gt, ai, at, mi=None, mt=None):
    Ai = torch.as_tensor(ai, device=dev); At = torch.as_tensor(at, device=dev)
    if mi is None:
        return .5 * (.5 * O.loss_list(obj, gi, Ai) + .5 * O.loss_list(obj, gt, At))
    li = O.loss_list_masked(obj, gi, Ai, torch.as_tensor(mi, device=dev, dtype=gi.dtype), 1.0)
    lt = O.loss_list_masked(obj, gt, At, torch.as_tensor(mt, device=dev, dtype=gt.dtype), 1.0)
    return .5 * (.5 * li + .5 * lt)


def item2(ctx, unit_dir, cfg, heads, nulls, stud, batches, pres):
    obj = cfg['objective']
    acc = {p: {f'{part}_{g}': [] for part in ('total', 'present_rows', 'missing_rows') for g in ('head', 'null', 'img', 'txt')} for p in PEERS}
    clip = {p: [] for p in PEERS}
    for lists in batches:
        mi, mt = D36.valid_rows(lists)
        sc, _ = batch_scores(ctx, heads, nulls, stud, lists)
        for j, p in enumerate(PEERS):
            gi, gt = sc[j]; G = groups(heads, nulls, stud, p); flat = [q for v in G.values() for q in v]
            parts = dict(total=peer_loss(obj, gi, gt, lists[0][2], lists[1][2]), present_rows=peer_loss(obj, gi, gt, lists[0][2], lists[1][2], mi, mt),
                         missing_rows=peer_loss(obj, gi, gt, lists[0][2], lists[1][2], ~mi, ~mt))
            for part, L in parts.items():
                gr = torch.autograd.grad(L, flat, retain_graph=True, allow_unused=True); k = 0
                for gname, qs in G.items():
                    acc[p][f'{part}_{gname}'].append(gnorm(gr[k:k + len(qs)])); k += len(qs)
                if part == 'total':
                    clip[p].append(min(1.0, cfg['clip_per_peer'] / (gnorm(gr) + 1e-6)))
    out = {p: {k: float(np.mean(v)) for k, v in acc[p].items()} for p in PEERS}
    for p in PEERS:
        out[p]['clip_multiplier_mean'] = float(np.mean(clip[p])); out[p]['frac_clipped'] = float(np.mean(np.array(clip[p]) < 1))
    # feature change on a fixed FIT subset, by present / missing
    rng = np.random.default_rng(4041)
    si = np.sort(rng.choice(C.N_TRAIN_IMG, min(A.fit_subset, C.N_TRAIN_IMG), replace=False)); sc_ = np.sort(rng.choice(C.N_TRAIN_CAP, min(A.fit_subset, C.N_TRAIN_CAP), replace=False))
    fc = {}; export = dict(img_ids=si, cap_ids=sc_)
    with torch.no_grad():
        for p in PEERS:
            for kind, ids, pr in (('img', si, pres[0][si]), ('cap', sc_, pres[1][sc_])):
                u = ctx.joint.encode(stud[p], p, 'fit', kind, ids, grad=False); ref = ctx.d[('fit', kind, p)][torch.as_tensor(ids, device=dev)]
                cs = (u * ref).sum(1).double().cpu().numpy(); export[f'cos_{kind}_{p}'] = cs
                fc[f'{kind}_{p}'] = dict(all=float(cs.mean()), present=float(cs[pr].mean()), missing=float(cs[~pr].mean()) if (~pr).any() else None,
                                         n_present=int(pr.sum()), n_missing=int((~pr).sum()), p05=float(np.quantile(cs, .05)))
    pd = {}
    par = R9.build_students(ctx.joint.ck, dev); pref = {p: dict(par[p].named_parameters()) for p in PEERS}
    for p in PEERS:
        for m in ('img', 'txt'):
            a = b = 0.0
            for n, q in stud[p].named_parameters():
                if q.requires_grad and n.startswith(m + '.'):
                    a += float(((q.detach() - pref[p][n].detach()) ** 2).sum()); b += float((pref[p][n].detach() ** 2).sum())
            pd[f'{m}_{p}'] = dict(l2=a ** .5, rel_l2=(a / b) ** .5)
    return dict(grad_norms_fixed_fit_batches=out, fit_feature_change=fc, param_distance=pd), export


def item3(ctx, unit_dir, step, batches):
    cfg, heads, nulls, stud = load_state(unit_dir, step, ctx)
    objs = ('J', 'BCE', 'NCE'); res = {o: dict(dLdg={}, dg={}) for o in objs}; cosg = {}; ident = 0.0; ex = []
    for b, lists in enumerate(batches):
        (qi, caps, ai), (qc, imgs, at) = lists
        sc, (ri, rt) = batch_scores(ctx, heads, nulls, stud, lists)
        cls = dict(i2t=dict(pos=(ai > 0) & ri, neg=(ai == 0) & ri, null=~ri), t2i=dict(pos=(at > 0) & rt, neg=(at == 0) & rt, null=~rt))
        rec = dict(qi=qi, caps=caps, ai=ai, qc=qc, imgs=imgs, at=at, g_i=np.mean([s[0].detach().double().cpu().numpy() for s in sc], 0), g_t=np.mean([s[1].detach().double().cpu().numpy() for s in sc], 0))
        pg = {}
        for o in objs:
            dgi, dgt, pgs = [], [], []
            for j, p in enumerate(PEERS):
                gi, gt = sc[j]; G = groups(heads, nulls, stud, p); flat = [q for v in G.values() for q in v]
                L = peer_loss(o, gi, gt, ai, at)
                gr = torch.autograd.grad(L, [gi, gt] + flat, retain_graph=True, allow_unused=True)
                dgi.append(gr[0].double().cpu().numpy()); dgt.append(gr[1].double().cpu().numpy())
                pgs.append([torch.zeros_like(q) if g is None else g for g, q in zip(gr[2:], flat)])
            pg[o] = pgs
            rec[f'dLdg_i_{o}'] = np.mean(dgi, 0); rec[f'dLdg_t_{o}'] = np.mean(dgt, 0)
            for dn, M in cls.items():
                for c, m in M.items():
                    v = rec[f'dLdg_{dn[0]}_{o}'][m]
                    res[o]['dLdg'].setdefault(f'{dn}_{c}', []).append((float(v.mean()) if v.size else np.nan, float((v > 0).mean()) if v.size else np.nan))
        # item 6 identity on these logits (per peer, production objectives)
        for j in range(len(PEERS)):
            for gx, ax in ((sc[j][0], ai), (sc[j][1], at)):
                g_ = gx.detach().double().clone().requires_grad_(); Ax = torch.as_tensor(ax, device=dev).double()
                gj, = torch.autograd.grad(O.loss_list('J', g_, Ax), g_); gb, = torch.autograd.grad(O.loss_list('BCE', g_, Ax), g_)
                ident = max(ident, float((gj - (1 - torch.tanh(g_ / 2) ** 2) * gb).abs().max()))
        # parameter-gradient cosines between objectives per group (peer A and B pooled)
        for a_, b_ in (('J', 'BCE'), ('J', 'NCE'), ('BCE', 'NCE')):
            for j, p in enumerate(PEERS):
                G = groups(heads, nulls, stud, p); k = 0
                for gname, qs in G.items():
                    x = torch.cat([t.double().flatten() for t in pg[a_][j][k:k + len(qs)]]); y = torch.cat([t.double().flatten() for t in pg[b_][j][k:k + len(qs)]]); k += len(qs)
                    den = float(x.norm() * y.norm())
                    cosg.setdefault(f'{a_}_vs_{b_}_{gname}', []).append(float((x @ y) / den) if den > 0 else np.nan)
        # one normalised step per objective (eta 0.01 in parameter L2, per peer over head+null+student), Delta g on this batch
        for o in objs:
            with torch.no_grad():
                saved = []
                for j, p in enumerate(PEERS):
                    G = groups(heads, nulls, stud, p); flat = [q for v in G.values() for q in v]
                    nr = gnorm(pg[o][j]) + 1e-30
                    for q, g in zip(flat, pg[o][j]):
                        saved.append((q, q.detach().clone())); q.sub_(0.01 * g / nr)
            sc2, _ = batch_scores(ctx, heads, nulls, stud, lists)
            with torch.no_grad():
                for q, v in saved:
                    q.copy_(v)
            dgi = np.mean([s[0].detach().double().cpu().numpy() for s in sc2], 0) - rec['g_i']; dgt = np.mean([s[1].detach().double().cpu().numpy() for s in sc2], 0) - rec['g_t']
            rec[f'dg_i_{o}'] = dgi; rec[f'dg_t_{o}'] = dgt
            for dn, M in cls.items():
                for c, m in M.items():
                    v = (dgi if dn == 'i2t' else dgt)[m]
                    res[o]['dg'].setdefault(f'{dn}_{c}', []).append((float(v.mean()) if v.size else np.nan, float((v < 0).mean()) if v.size else np.nan))
        ex.append(rec)
    summ = {o: {kind: {k: dict(mean=float(np.nanmean([x[0] for x in v])), frac_pushed_down=float(np.nanmean([x[1] for x in v]))) for k, v in res[o][kind].items()}
                for kind in ('dLdg', 'dg')} for o in objs}
    np.savez_compressed(OUTD / f'item3_{Path(unit_dir).name}_step{step}.npz', **{f'b{b}_{k}': v for b, r in enumerate(ex) for k, v in r.items()})
    return dict(state=f'{Path(unit_dir).name}@{step}', per_objective=summ, param_grad_cosine={k: float(np.nanmean(v)) for k, v in cosg.items()},
                J_BCE_identity_max_abs=ident, eta_parameter_l2=0.01, note='dLdg "frac_pushed_down" = fraction of slots with dL/dg > 0 (descent lowers g); dg frac = fraction with Delta g < 0')


def main():
    t0 = time.time()
    first = Path(A.units.split(',')[0])
    cfg0 = json.load(open(first / 'RESULT.json'))['config']
    ctx = TU.Ctx(cfg0['cond'], dev, need_local=False); lazy_joint(ctx)
    pres = present_flags(ctx.d)
    batches = fixed_batches(ctx, A.batches)
    out = dict(tag=A.tag, tree=A.tree, cond=cfg0['cond'], native_H=ctx.nat_metrics, units={}, fit_present_fraction=dict(img=float(pres[0].mean()), cap=float(pres[1].mean())))
    for ud in A.units.split(','):
        ud = Path(ud); res = json.load(open(ud / 'RESULT.json')); cfg, heads, nulls, stud = load_state(ud, 8000, ctx)
        ctx.joint.clear_eval()
        V = dev_vectors(ctx, stud); S = sim(V)
        Vn = dev_vectors(ctx, stud, centred=False); Sn = sim(Vn)
        rng = np.random.default_rng(5); order = dict(img=rng.permutation(C.N_DEV_IMG), cap=rng.permutation(C.N_DEV_CAP))
        Vp = dev_vectors(ctx, stud, order=order, chunk=64)
        inv = {k: np.argsort(v) for k, v in order.items()}
        reorder = max(float((Vp[(p, k)][torch.as_tensor(inv[k], device=dev)] - V[(p, k)]).abs().max()) for p in PEERS for k in ('img', 'cap'))
        u = dict(unit=ud.name, objective=cfg['objective'], supervision=cfg['supervision'],
                 item1=dict(final_g_two_stage=res['final']['dev_retrieval'], student_cosine_two_stage_H_top100=metrics_two_stage(ctx, S),
                            student_cosine_full_gallery=EV.full_metrics(S)[0], parent_native_full=ctx.nat_metrics),
                 item5=dict(reorder_chunk64_max_abs_vs_standard=reorder, uncentred_student_cosine_full_gallery=EV.full_metrics(Sn)[0],
                            uncentred_student_cosine_two_stage=metrics_two_stage(ctx, Sn), centring='fixed parent per-native-batch vector (R39 registered patch); un-centred = encoder output (already L2) without that subtraction'))
        i2, export = item2(ctx, ud, cfg, heads, nulls, stud, batches, pres)
        u['item2'] = i2
        np.savez_compressed(OUTD / f'item2_{ud.name}.npz', **export)
        out['units'][ud.name] = u
        print('unit', ud.name, 'g', res['final']['dev_retrieval']['rsum'], 'cos2', u['item1']['student_cosine_two_stage_H_top100']['rsum'],
              'cosfull', u['item1']['student_cosine_full_gallery']['rsum'], 'reorder', reorder, round(time.time() - t0), flush=True)
        del heads, nulls, stud; ctx.joint.clear_eval(); torch.cuda.empty_cache()
    if A.common:
        out['item3'] = [item3(ctx, A.common, s, batches) for s in (4000, 8000)]
    out['fixed_fit_batches'] = dict(stream='[4040, 7]', n=A.batches, queries_per_direction=16, supervision_lists='N (OBS_MISSING_NULL)')
    out['seconds'] = time.time() - t0; out['test_guard'] = C.guard_record()
    (OUTD / 'READOUTS.json').write_text(json.dumps(out, indent=1, default=float))
    (W / 'diag7' / f'READOUTS_{A.tag}.json').write_text(json.dumps(out, indent=1, default=float))
    print('R40_DIAG7_DONE', A.tag, round(time.time() - t0))


if __name__ == '__main__':
    main()
