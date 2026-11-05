#!/usr/bin/env python3
"""NR5 D0: read-only decomposition of the 12 NR4 TAIL_LORA8 units at the fixed endpoint (stage 4000 = step 12000). No training,
no checkpoint/model selection. Per condition (3 objectives):
DEV (all queries, frozen base Top100 from the frozen DEV cache, float64, ID-hash ties; ranks beyond 100 keep base order):
  E0G0 = frozen encoder + NR3 folded prefix head (= NR4 stage 0)    E1G1 = adapted encoder + endpoint head (= NR4 endpoint)
  E1G0 = adapted encoder + prefix head                              E0G1 = frozen encoder + endpoint head
  E1COS = cosine of the adapted features reranking the same base Top100
  full gallery: E1 cosine over all pairs; E1G1 current score over all pairs (pair-chunked).
E1 DEV features = the NR4 endpoint evaluator's FEATURES.npz (sha256 checked against its PREDICTIONS_SHA256.json); E1G1/E0G0 base-Top100
orders are checked equal to the stored NR4 eval_012000 / NR3 eval_008000 orders. Orders/scores are written and hashed before the
evaluator-only DEV positives are read.
FIT readout (truth-free, same checkpoints): HREF on the NR3 READ-B pool (g = s - log w) raw and with a positive affine fitted ONLY
on the NR3 CAL-A pool; product-Q (log w = 0) on the NR2 fit_readout pool (seed 2026110451) raw and with an affine fitted only on a
new product-Q CAL pool (seed 2026110461, same construction). J values under different Q are reported side by side, not compared.
E1 FIT features: blocks 10-11 + pooler of the NR4 endpoint LoRA on the verified NR4 hidden cache (fixed B=32), for pool items only."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from nr1.io import FeatureBank, load_observed, read_json, write_json, sha256
from nr1.models import DirectScorer
from nr1.evaluate import order_rows, ranks_from_orders, report_metrics, id_tiebreak
from nr2.reference import Marginals
from nr3 import pools as PL
from nr3.ref_init import fit_reference_affine, pooled_j
from nr4.paths import paths
from nr4.tail_encoder import load_siglip, TailEncoder
from nr4.train import HiddenTail
from nr5.paths import OUT
spec = importlib.util.spec_from_file_location('fr', '/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code/site/fit_readout.py')
FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
N4 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/runs'); N3 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104')
TOPK = 100


def rerank(base_oi, base_ot, ids_t, ids_i, si, st):
    """si: [nI, K] scores of base top-K text candidates per image (same order as base_oi[:, :K]); st likewise."""
    ni, nt = base_oi.copy(), base_ot.copy(); k1, k2 = id_tiebreak(ids_t), id_tiebreak(ids_i)
    for i in range(len(ni)):
        c = base_oi[i, :TOPK]; ni[i, :TOPK] = c[np.lexsort((k1[c], -si[i]))]
    for t in range(len(nt)):
        c = base_ot[t, :TOPK]; nt[t, :TOPK] = c[np.lexsort((k2[c], -st[t]))]
    return ni, nt


@torch.no_grad()
def topk_scores(fn, U, V, oi, ot, chunk=65536):
    ia = np.repeat(np.arange(len(U)), TOPK); ca = oi[:, :TOPK].reshape(-1); ib = ot[:, :TOPK].reshape(-1); cb = np.repeat(np.arange(len(V)), TOPK)
    def sc(a, b):
        return np.concatenate([fn(U[torch.as_tensor(a[s:s+chunk], device=U.device)], V[torch.as_tensor(b[s:s+chunk], device=U.device)]).cpu().numpy() for s in range(0, len(a), chunk)])
    return sc(ia, ca).reshape(len(U), TOPK), sc(ib, cb).reshape(len(V), TOPK)


def spearman_rows(a, b):
    ra, rb = a.argsort(1).argsort(1).astype(float), b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return float(((ra*rb).sum(1) / np.sqrt((ra**2).sum(1)*(rb**2).sum(1))).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); a = ap.parse_args(); c = a.cond
    od = OUT/'d0'/c; od.mkdir(parents=True, exist_ok=True); P = paths(c); dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    db = FeatureBank(P['dev'], 'dev'); E0x = torch.from_numpy(np.array(db.x, copy=True)).double().to(dev); E0y = torch.from_numpy(np.array(db.y, copy=True)).double().to(dev)
    base = (F.normalize(E0x, dim=1) @ F.normalize(E0y, dim=1).T).cpu().numpy(); boi, bot = order_rows(base, db.text_ids), order_rows(base.T, db.image_ids)
    pre = torch.load(P['prefix'], map_location='cpu', weights_only=False); G0 = DirectScorer(**pre['binding']['model_config']); G0.load_state_dict(pre['model']); G0 = G0.double().to(dev).eval()
    fb = FeatureBank(P['fit'], 'fit'); im, tx = load_observed(P['observed'], fb); marg = Marginals(im, tx, len(fb.image_ids), len(fb.text_ids))
    tab = np.load(P['tables']); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    poolA, poolB = PL.load(N3/'prep'/c/'POOL_CAL_A.npz'), PL.load(N3/'prep'/c/'POOL_READ_B.npz')
    p_, qi_, qt_, _ = FR.pools(im, tx, top, marg)
    g = np.random.default_rng(2026110461); NP = FR.NP; pc_, qic_, qtc_ = g.integers(len(im), size=NP), g.integers(len(im), size=NP), g.integers(len(im), size=NP)
    pq = dict(READ=dict(Pi=im[p_], Pt=tx[p_], Qi=im[qi_], Qt=tx[qt_]), CAL=dict(Pi=im[pc_], Pt=tx[pc_], Qi=im[qic_], Qt=tx[qtc_]))
    hpool = {nm: pl for nm, pl in (('CAL_A', poolA), ('READ_B', poolB))}
    # all FIT items touched by the readout pools
    imgs, txts = [], []
    for pl in hpool.values():
        for d, x in pl.items():
            for kq, kc in (('P_query', 'P_cand'), ('Q_query', 'Q_cand')):
                (imgs, txts)[0 if d == 'i2t' else 1].append(x[kq]); (txts, imgs)[0 if d == 'i2t' else 1].append(x[kc])
    for x in pq.values(): imgs += [x['Pi'], x['Qi']]; txts += [x['Pt'], x['Qt']]
    ui, ut = np.unique(np.concatenate(imgs)), np.unique(np.concatenate(txts))
    X0 = torch.from_numpy(np.array(fb.x, copy=True)).to(dev); Y0 = torch.from_numpy(np.array(fb.y, copy=True)).to(dev)
    hi = np.load(P['hidden']/'img_h10_f32.npy', mmap_mode='r'); ht = np.load(P['hidden']/'txt_h10_f32.npy', mmap_mode='r')
    m, _ = load_siglip(dev, lora=True, seed=1001); enc = TailEncoder(m, None, None, dev); htl = HiddenTail(enc, hi, ht)
    res = dict(condition=c, fixed_endpoint='NR4 stage 4000 (step 12000)', no_selection=True, test_read=False, units={}, fit_pool_items=dict(images=int(len(ui)), texts=int(len(ut))))

    def fit_readout(head, X, Y):
        out = {}
        f = {nm: ({d: PL.model_scores(head, X, Y, d, pl[d]['P_query'], pl[d]['P_cand']) for d in pl}, {d: PL.model_scores(head, X, Y, d, pl[d]['Q_query'], pl[d]['Q_cand']) for d in pl}) for nm, pl in hpool.items()}
        stA, stB = PL.stack(hpool['CAL_A'], *f['CAL_A']), PL.stack(hpool['READ_B'], *f['READ_B'])
        fa = fit_reference_affine(*stA, initial=(1., 0.), a_bounds=(.1, 100.), b_bounds=(-20., 20.), maxiter=500)
        out['HREF'] = dict(READ_B_J_raw=pooled_j(*stB), CAL_A_affine=[fa.a, fa.b], READ_B_J_CALaffine=pooled_j(*stB, fa.a, fa.b), affine_at_bound=fa.at_bound)
        s = {nm: (FR.scores(head, X, Y, x['Pi'], x['Pt']), FR.scores(head, X, Y, x['Qi'], x['Qt'])) for nm, x in pq.items()}
        z = lambda v: np.zeros_like(v)
        fq = fit_reference_affine(np.stack([s['CAL'][0]]*2), np.stack([s['CAL'][1]]*2), np.stack([z(s['CAL'][0])]*2), np.stack([z(s['CAL'][1])]*2), initial=(1., 0.), a_bounds=(.1, 100.), b_bounds=(-20., 20.), maxiter=500)
        out['productQ'] = dict(READ_J_raw=FR.jv(np.tanh(s['READ'][0]/2), np.tanh(s['READ'][1]/2)), CAL_affine=[fq.a, fq.b],
                               READ_J_CALaffine=FR.jv(np.tanh((fq.a*s['READ'][0]+fq.b)/2), np.tanh((fq.a*s['READ'][1]+fq.b)/2)), affine_at_bound=fq.at_bound)
        return out

    for obj in ('J', 'BCE', 'NCE'):
        run = N4/f'NR4_TAIL_LORA8_{c}_G_{obj}_s1001'; ck = torch.load(run/'step_012000.pt', map_location='cpu', weights_only=False)
        ps = read_json(run/'eval_012000'/'PREDICTIONS_SHA256.json')
        if sha256(run/'eval_012000'/'FEATURES.npz') != ps['features_sha256']: raise ValueError('NR4 endpoint feature file hash mismatch')
        fz = np.load(run/'eval_012000'/'FEATURES.npz'); E1x = torch.from_numpy(fz['image']).double().to(dev); E1y = torch.from_numpy(fz['text']).double().to(dev)
        G1 = DirectScorer(**ck['binding']['model_config']); G1.load_state_dict(ck['model']); G1 = G1.double().to(dev).eval()
        t0 = time.perf_counter(); combos = {}
        cosf = lambda x, y: (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)
        for name, fn, U, V in (('E0G0', G0.pairs, E0x, E0y), ('E1G1', G1.pairs, E1x, E1y), ('E1G0', G0.pairs, E1x, E1y), ('E0G1', G1.pairs, E0x, E0y), ('E1COS', cosf, E1x, E1y)):
            tt = time.perf_counter(); si, st = topk_scores(fn, U, V, boi, bot); torch.cuda.synchronize(); sec = time.perf_counter() - tt
            ni, nt = rerank(boi, bot, db.text_ids, db.image_ids, si, st); combos[name] = dict(si=si, st=st, ni=ni, nt=nt, seconds=sec)
        tt = time.perf_counter()
        with torch.no_grad():
            Sc = (F.normalize(E1x, dim=1) @ F.normalize(E1y, dim=1).T).cpu().numpy()
            Sg = torch.cat([G1.matrix(E1x[i:i+64], E1y, pair_chunk=65536) for i in range(0, len(E1x), 64)]).cpu().numpy()
        torch.cuda.synchronize(); fsec = time.perf_counter() - tt
        full = dict(E1COS_full=(order_rows(Sc, db.text_ids), order_rows(Sc.T, db.image_ids)), E1G1_full=(order_rows(Sg, db.text_ids), order_rows(Sg.T, db.image_ids)))
        r4 = np.load(run/'eval_012000'/'PREDICTIONS.npz'); r3 = np.load(N3/'runs'/f'NR3_B_NCE2{obj}_{c}_G_s1001'/'eval_008000'/'PREDICTIONS.npz')
        checks = dict(E1G1_equals_NR4_endpoint_order=bool(np.array_equal(r4['i_order'], combos['E1G1']['ni']) and np.array_equal(r4['t_order'], combos['E1G1']['nt'])),
                      E0G0_equals_NR3_stage0_order=bool(np.array_equal(r3['i_order'], combos['E0G0']['ni']) and np.array_equal(r3['t_order'], combos['E0G0']['nt'])),
                      E1COS_full_top100_equals_NR4_student_order=bool(np.array_equal(r4['student_i_order'], full['E1COS_full'][0][:, :TOPK]) and np.array_equal(r4['student_t_order'], full['E1COS_full'][1][:, :TOPK])))
        sv = {f'{k}_{x}': v[x] for k, v in combos.items() for x in ('si', 'st', 'ni', 'nt')}; sv.update({f'{k}_i_order_top100': v[0][:, :TOPK] for k, v in full.items()}); sv.update({f'{k}_t_order_top100': v[1][:, :TOPK] for k, v in full.items()})
        np.savez_compressed(od/f'D0_SCORES_{obj}.npz', **sv); pred_sha = sha256(od/f'D0_SCORES_{obj}.npz')
        # ---- truth (evaluator only), after scores/orders are hashed
        truth = read_json(P['truth']); Pos = truth['positives']
        bi, bt = ranks_from_orders(boi, bot, db.image_ids, db.text_ids, Pos)
        U = dict(scores_sha256=pred_sha, checks=checks, combos={}, full_gallery={}, full_gallery_seconds=fsec, n_queries=len(bi)+len(bt))
        R = {}
        for k, v in combos.items():
            ri, rt = ranks_from_orders(v['ni'], v['nt'], db.image_ids, db.text_ids, Pos); R[k] = (ri, rt)
            U['combos'][k] = dict(metrics=report_metrics(ri, rt), seconds=v['seconds'], seconds_per_1000_queries=1000*v['seconds']/(len(ri)+len(rt)),
                                  **{d: dict(rescued=int(((b > 1) & (r == 1)).sum()), lost=int(((b == 1) & (r > 1)).sum()), base_R1_errors=int((b > 1).sum()),
                                             base_R1_errors_still_wrong=int(((b > 1) & (r > 1)).sum()), new_R1_errors=int(((b == 1) & (r > 1)).sum()))
                                     for d, b, r in (('i2t', bi, ri), ('t2i', bt, rt))},
                                  spearman_vs_E1G1=dict(i2t=spearman_rows(v['si'], combos['E1G1']['si']), t2i=spearman_rows(v['st'], combos['E1G1']['st'])))
        for k, (oi_, ot_) in full.items():
            ri, rt = ranks_from_orders(oi_, ot_, db.image_ids, db.text_ids, Pos); U['full_gallery'][k] = report_metrics(ri, rt); R[k] = (ri, rt)
        np.savez_compressed(od/f'D0_RANKS_EVAL_ONLY_{obj}.npz', base_i=bi, base_t=bt, **{f'{k}_i': v[0] for k, v in R.items()}, **{f'{k}_t': v[1] for k, v in R.items()})
        # ---- FIT readout
        for n, p in m.named_parameters():
            if p.requires_grad: p.data.copy_(ck['adapter'][n].to(dev))
        tt = time.perf_counter()
        with torch.no_grad():
            X1 = X0.clone(); Y1 = Y0.clone()
            X1[torch.as_tensor(ui, device=dev)] = htl.encode('image', ui, False); Y1[torch.as_tensor(ut, device=dev)] = htl.encode('text', ut, False)
        torch.cuda.synchronize(); U['fit_feature_seconds'] = time.perf_counter() - tt
        G0f = DirectScorer(**pre['binding']['model_config']); G0f.load_state_dict(pre['model']); G0f = G0f.to(dev).eval()
        G1f = DirectScorer(**ck['binding']['model_config']); G1f.load_state_dict(ck['model']); G1f = G1f.to(dev).eval()
        U['fit_readout'] = {k: fit_readout(h, X, Y) for k, h, X, Y in (('E0G0', G0f, X0, Y0), ('E1G1', G1f, X1, Y1), ('E1G0', G0f, X1, Y1), ('E0G1', G1f, X0, Y0))}
        res['units'][f'NR4_TAIL_LORA8_{c}_G_{obj}_s1001'] = U; write_json(od/'D0.json', res)
        print(obj, {k: round(v['metrics']['rSum'], 2) for k, v in U['combos'].items()}, {k: round(v['rSum'], 2) for k, v in U['full_gallery'].items()}, checks, flush=True)
    res['done'] = True; write_json(od/'D0.json', res)


if __name__ == '__main__':
    main()
