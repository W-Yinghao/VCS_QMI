"""CR4 real small-sample acceptance (GPU recorded; no TEST, no truth).

 P1 CE branch of packed_loss == CR3 conditional_loss on real COMPLETE batches (loss, G-gradient); RECT path untouched
 P2 single-positive: CC OBS / J_MASS_OBS full-graph single-positive in both directions; PC == CE loss and gradient
 P3 multi-positive real batch: PC positive partials <= 0, zero-mass candidate partials >= 0; CE = sep + H + KL
 P4 numerics: padding / no-negative rows / extreme logits finite (no NaN loss or grad)
 P5 PC uses the identical PackedBatch: same pairs/targets/mask/query weights as CR3 COMPLETE; added candidates not queries; 0.5/0.5
 P6 score invariance with a nonzero head (chunk, reorder, ID permutation)
 P7 exact resume PC (250 -> 300 == 300)
 P8 same draws PC vs CE (100 steps) and COMPLETE-CE replay == CR3 COMPLETE step 250 (bitwise iff same GPU/driver as CR3 group)
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr4_student as S                                               # noqa: E402
C1 = S.C1
from reference.support_complete import FixedRelation, conditional_loss   # noqa: E402
from reference.positive_exclusive import packed_loss, ce_decomposition, positive_exclusive_rows   # noqa: E402
from cr1.models import GlobalScorer                                   # noqa: E402
from cr1.io import read_json                                          # noqa: E402

CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR2 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
CR3 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104')
R = {}


def rec(n, ok, **kw):
    R[n] = dict(ok=bool(ok), **kw); print(n, 'PASS' if ok else 'FAIL', json.dumps(kw, default=str)[:900], flush=True)


def gp(c, r):
    if r in ('OBS', 'J_MASS_OBS'): return CR3/'graphs'/c/f'{r}.npz'
    if r == 'FULL_TRUE_ORACLE': return CR3/'oracle'/c/'FULL_TRUE_ORACLE.npz'
    if r == 'KEEP_ORACLE': return CR2/'oracle'/c/f'CR2_{c}_KEEP_ORACLE_s1001.npz'
    return CR2/'relations'/c/f"CR2_{c}_{r.replace('_KEEP_TOP1', '')}_KEEP_TOP1_s1001.npz"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'; det = C1.setup_determinism(dev); t0 = time.perf_counter(); gpu = torch.cuda.get_device_name()
    rec('GPU_RECORD', True, gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), determinism=det)
    banks = {}
    def load(c):
        if c not in banks:
            b, p, g = C1.get_prepared(CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'); X, Y = C1.features(b, dev); banks[c] = (b, p, X, Y)
        return banks[c]
    def model(c, nonzero=True):
        b, p, X, Y = load(c); ini = p['student_common_initialization']; m = GlobalScorer(768, ini['a'], ini['b'], 1001).to(dev)
        if nonzero:
            with torch.no_grad():
                m.global_residual[-1].weight.normal_(0, .05, generator=torch.Generator(device=dev).manual_seed(3)); m.global_residual[-1].bias.fill_(.3)
        return m
    def batches(c, r, n=3, seed=20261104201):
        mat = load_npz(gp(c, r)).tocsr(); rel = C1.ExposedRelation(mat); fr = FixedRelation(mat); g = np.random.default_rng(seed)
        return [fr.build(fr.from_edge_indices(rel.batch_edges(g, 64)['edges']), 'COMPLETE') for _ in range(n)], fr
    def grad(m, loss):
        gr = torch.autograd.grad(loss, list(m.parameters())); return torch.cat([x.flatten() for x in gr])
    # P1
    ok = True; p1 = {}
    for c, r in [('Flickr60', 'J_KEEP_TOP1'), ('Flickr00', 'OBS'), ('CC152K_RECOVERED_20261104', 'BASE_KEEP_TOP1')]:
        b, p, X, Y = load(c); m = model(c); bs, _ = batches(c, r)
        for pb in bs:
            s = S.score_pairs(m, X, Y, pb.pairs); l1 = conditional_loss(s, pb); g1 = grad(m, l1)
            s = S.score_pairs(m, X, Y, pb.pairs); l2 = packed_loss(s, pb, kind='CE'); g2 = grad(m, l2)
            ok &= bool(torch.equal(l1, l2) and torch.equal(g1, g2)); p1.setdefault(f'{c}:{r}', []).append(float((l1-l2).abs()))
    rec('P1_CE_branch_equals_CR3_conditional_loss', ok, loss_abs=p1, note='bitwise loss and gradient; RECT trains through the unchanged CR3 code')
    # P2
    p2 = {}; ok = True
    for r in ('OBS', 'J_MASS_OBS'):
        c = 'CC152K_RECOVERED_20261104'; mat = load_npz(gp(c, r)).tocsr(); rows = np.diff(mat.indptr); cols = np.diff(mat.tocsc().indptr)
        single = bool((rows == 1).all() and ((cols == 0) | (cols == 1)).all())
        b, p, X, Y = load(c); m = model(c); bs, _ = batches(c, r)
        eq = True
        for pb in bs:
            s = S.score_pairs(m, X, Y, pb.pairs); lc = conditional_loss(s, pb); gc = grad(m, lc)
            s = S.score_pairs(m, X, Y, pb.pairs); lp = packed_loss(s, pb, kind='PC'); gpc = grad(m, lp)
            eq &= bool(torch.equal(lc, lp) and torch.equal(gc, gpc))
        ok &= single and eq; p2[r] = dict(rows_all_one=bool((rows == 1).all()), cols_le_one=bool(((cols == 0) | (cols == 1)).all()), pc_equals_ce_bitwise=eq)
    rec('P2_single_positive_identity', ok, graphs=p2)
    # P3/P4
    c = 'Flickr60'; b, p, X, Y = load(c); m = model(c); bs, _ = batches(c, 'J_KEEP_TOP1', n=2); p3 = {}
    pb = bs[0]; s = S.score_pairs(m, X, Y, pb.pairs).double().detach().requires_grad_(True)
    gidx = torch.as_tensor(pb.gather, device=dev); mk = torch.as_tensor(pb.mask, device=dev); tg = torch.as_tensor(pb.targets, device=dev)
    sv = s[gidx].masked_fill(~mk, 0.0); pc = positive_exclusive_rows(sv, tg, mk).sum(); gs, = torch.autograd.grad(pc, s)
    gview = gs[gidx] * mk  # per-pair partial aggregated over views
    zv = sv.detach().clone().requires_grad_(True); gz, = torch.autograd.grad(positive_exclusive_rows(zv, tg, mk).sum(), zv)
    pos = mk & (tg > 0); neg = mk & (tg == 0); multi = pos.sum(1) > 1
    parts = ce_decomposition(sv.detach(), tg, mk); err = float((parts['ce']-parts['separation']-parts['entropy']-parts['within_positive_kl']).abs().max())
    ok3 = float(gz[pos & multi[:, None]].max()) <= 0 and float(gz[neg].min()) >= 0 and err < 1e-10
    rec('P3_partial_signs_and_decomposition', ok3, multi_positive_views=int(multi.sum()), pc_max_positive_partial=float(gz[pos].max()),
        pc_min_negative_partial=float(gz[neg].min()), ce_decomposition_max_abs_error=err)
    ext = sv.detach()*1e3
    z1 = ext.clone().requires_grad_(True); l = positive_exclusive_rows(z1, tg, mk).sum(); g1, = torch.autograd.grad(l, z1)
    tg2 = tg.clone(); tg2[0] = 0; tg2[0, 0] = .5; tg2[0, 1] = .5; mk3 = mk.clone(); mk3[0] = False; mk3[0, 0] = True; mk3[0, 1] = True  # two positives, no negative
    z2 = sv.detach().clone().requires_grad_(True); l2 = positive_exclusive_rows(z2, tg2, mk3); g2, = torch.autograd.grad(l2.sum(), z2)
    rec('P4_numerics', bool(torch.isfinite(l) and torch.isfinite(g1).all() and torch.isfinite(l2).all() and torch.isfinite(g2).all()),
        extreme_logit_loss=float(l), no_negative_row_value=float(l2[0]))
    # P5
    ok5 = True; p5 = {}
    for c, r in [('Flickr60', 'J_KEEP_TOP1'), ('Flickr60', 'KEEP_ORACLE'), ('CC152K_RECOVERED_20261104', 'J_KEEP_TOP1')]:
        bs, fr = batches(c, r, n=5, seed=11); st = dict(extra_queries=0, weight_sum_err=0.0, dir_weight_err=0.0)
        for pbb in bs:
            st['extra_queries'] += len(pbb.views) - len({(v.direction, v.query) for v in pbb.views})
            w = pbb.query_weights; d = np.array([v.direction for v in pbb.views])
            st['weight_sum_err'] = max(st['weight_sum_err'], abs(w.sum()-1)); st['dir_weight_err'] = max(st['dir_weight_err'], abs(w[d == 'i2t'].sum()-.5), abs(w[d == 't2i'].sum()-.5))
        ok5 &= st['extra_queries'] == 0 and st['weight_sum_err'] < 1e-12 and st['dir_weight_err'] < 1e-12; p5[f'{c}:{r}'] = st
    rec('P5_same_packed_batch_and_weights', ok5, graphs=p5, note='PC consumes the unchanged CR3 FixedRelation.build(.,COMPLETE) object; only the final loss differs')
    # P6
    c = 'Flickr60'; b, p, X, Y = load(c); m = model(c); bs, _ = batches(c, 'J_KEEP_TOP1', n=1); pb = bs[0]
    with torch.no_grad():
        s0 = S.score_pairs(m, X, Y, pb.pairs); s1 = S.score_pairs(m, X, Y, pb.pairs, chunk=257)
        perm = np.random.default_rng(1).permutation(len(pb.pairs)); s2 = S.score_pairs(m, X, Y, pb.pairs[perm])
        pi = np.random.default_rng(2).permutation(len(X)); inv = np.argsort(pi); pt = np.random.default_rng(3).permutation(len(Y)); invt = np.argsort(pt)
        q = pb.pairs.copy(); q[:, 0] = inv[q[:, 0]]; q[:, 1] = invt[q[:, 1]]
        s3 = S.score_pairs(m, X[torch.as_tensor(pi, device=dev)], Y[torch.as_tensor(pt, device=dev)], q)
    ip = torch.as_tensor(perm, device=dev)
    rec('P6_score_invariance', float((s0-s1).abs().max()) < 1e-5 and float((s2-s0[ip]).abs().max()) < 1e-5 and float((s3-s0).abs().max()) == 0,
        chunk_max_abs=float((s0-s1).abs().max()), reorder_max_abs=float((s2-s0[ip]).abs().max()), id_perm_max_abs=float((s3-s0).abs().max()))
    # P7/P8
    c = 'Flickr60'; g = gp(c, 'J_KEEP_TOP1'); bank, prep = CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'
    S.train(bank, prep, g, out/'pc100', 'ACCEPT', 'PC', dev, steps=100, saves=(0, 100), draw_log=out/'pc100.jsonl')
    S.train(bank, prep, g, out/'ce100', 'ACCEPT', 'CE', dev, steps=100, saves=(0, 100), draw_log=out/'ce100.jsonl')
    same = (out/'pc100.jsonl').read_text() == (out/'ce100.jsonl').read_text()
    S.train(bank, prep, g, out/'rA', 'ACCEPT', 'PC', dev, steps=300, saves=(0, 250, 300)); (out/'rB').mkdir(exist_ok=True)
    for f in ('step_000000.pt', 'step_000250.pt', 'TRAIN.jsonl'): shutil.copy(out/'rA'/f, out/'rB'/f)
    rB = S.train(bank, prep, g, out/'rB', 'ACCEPT', 'PC', dev, steps=300, saves=(0, 250, 300))
    ka, kb = torch.load(out/'rA'/'step_000300.pt', weights_only=False), torch.load(out/'rB'/'step_000300.pt', weights_only=False)
    rec('P7_exact_resume', all(torch.equal(ka['model'][k], kb['model'][k]) for k in ka['model']) and ka['rng_numpy'] == kb['rng_numpy'] and rB['resumed_from_step'] == 250)
    S.train(bank, prep, g, out/'ce250', 'ACCEPT', 'CE', dev, steps=250, saves=(0, 250))
    k1 = torch.load(out/'ce250'/'step_000250.pt', weights_only=False, map_location='cpu')['model']
    k0 = torch.load(CR3/'students_A1'/c/'CR3_Flickr60_J_KEEP_TOP1_COMPLETE_s1001'/'step_000250.pt', weights_only=False, map_location='cpu')['model']
    bit = all(torch.equal(k1[k], k0[k]) for k in k0); dmax = max(float((k1[k]-k0[k]).abs().max()) for k in k0)
    cr3env = read_json(CR3/'students_A1'/c/'GROUP_ENV.json')
    rec('P8_same_draws_and_CE_replay', same, pc_ce_draws_identical_100=same, ce_replay_bitwise_vs_CR3=bit, ce_replay_max_abs=dmax, cr3_group_gpu=cr3env['gpu_name'], cr3_driver=cr3env['driver'],
        note='CE replay bitwise only expected on the CR3 group environment; group jobs repeat this receipt and rerun references when it fails')
    allok = all(v['ok'] for v in R.values())
    (out/'ACCEPTANCE_CR4.json').write_text(json.dumps(dict(all_pass=allok, checks=R, seconds=time.perf_counter()-t0, test_read=False, truth_read=False), indent=1, default=str))
    print('ALL_PASS' if allok else 'ACCEPTANCE_FAIL')


if __name__ == '__main__': main()
