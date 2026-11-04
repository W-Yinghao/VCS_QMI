"""CR3 real small-sample acceptance + reuse receipts (GPU locked to RTX PRO 6000). No TEST, no truth.

 A1 RECT math: package FixedRelation.build(RECT)+conditional_loss == CR1/CR2 soft_nce loss and G-gradient on real batches
 A2 same draws: RECT and COMPLETE, 100 steps each, identical 64 edge indices and query weights at every step
 A3 COMPLETE targets: per-query sum 1, full positive support present, all old rectangle candidates kept
 A4 added candidates are not queries; graph bytes unchanged
 A5 score invariance on COMPLETE pair lists (chunk, ID permutation, independent image/text reorder) with a nonzero head
 A6 two-positive fixture 0.5/0.5 kept; real image-side multi-caption support fully present
 A7 exact resume (COMPLETE, 250->300 == 300) ; hashes
 R  reuse receipts: each of the 12 historical RECT units re-run for 250 steps on this GPU via the CR3 RECT path and compared
    bitwise with its stored step-250 checkpoint; REUSE only if original GPU model == lock and bitwise equal, else RERUN.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr3_student as S                                              # noqa: E402
C1 = S.C1
from reference.support_complete import FixedRelation, conditional_loss   # noqa: E402
from cr1.objectives import soft_nce                                  # noqa: E402
from cr1.models import GlobalScorer                                  # noqa: E402

CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR2 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
G3 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/graphs')
LOCK = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
R = {}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rec(n, ok, **kw):
    R[n] = dict(ok=bool(ok), **kw); print(n, 'PASS' if ok else 'FAIL', json.dumps(kw, default=str)[:900], flush=True)


def graph_path(c, r):
    if r in ('OBS', 'J_MASS_OBS'): return G3/c/f'{r}.npz'
    if r == 'KEEP_ORACLE': return CR2/'oracle'/c/f'CR2_{c}_KEEP_ORACLE_s1001.npz'
    return CR2/'relations'/c/f"CR2_{c}_{r.replace('_KEEP_TOP1', '')}_KEEP_TOP1_s1001.npz"


HIST = [('Flickr00', 'OBS', CR1/'Flickr00'/'students'/'OBS'), ('Flickr60', 'OBS', CR1/'Flickr60'/'students'/'OBS'),
        ('CC152K_RECOVERED_20261104', 'OBS', CR1/'CC152K_RECOVERED_20261104'/'students'/'OBS')] + \
       [(c, f'{s}_KEEP_TOP1', CR2/'students'/c/f'CR2_{c}_{s}_KEEP_TOP1_s1001') for c in ('Flickr60', 'CC152K_RECOVERED_20261104') for s in ('BASE', 'J', 'BCE', 'NCE')] + \
       [('Flickr60', 'KEEP_ORACLE', CR2/'students'/'Flickr60'/'CR2_Flickr60_KEEP_ORACLE_s1001')]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'; det = C1.setup_determinism(dev); t0 = time.perf_counter(); gpu = torch.cuda.get_device_name()
    rec('GPU_LOCK', gpu == LOCK, gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), determinism=det)
    c = 'Flickr60'; bank, prep, graph = C1.get_prepared(CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'); X, Y = C1.features(bank, dev)
    ini = prep['student_common_initialization']
    m = GlobalScorer(768, ini['a'], ini['b'], 1001).to(dev)
    with torch.no_grad():
        m.global_residual[-1].weight.normal_(0, .05, generator=torch.Generator(device=dev).manual_seed(3)); m.global_residual[-1].bias.fill_(.3)
    # ---------- A1
    a1 = {}; ok = True
    for r in ('OBS', 'J_KEEP_TOP1', 'KEEP_ORACLE'):
        mat = load_npz(graph_path(c, r)).tocsr(); rel = C1.ExposedRelation(mat); fr = FixedRelation(mat); g = np.random.default_rng(20261104201)
        for _ in range(3):
            b = rel.batch_edges(g, 64); ix, tx = torch.as_tensor(b['images'], device=dev), torch.as_tensor(b['texts'], device=dev)
            m.zero_grad(); s = m.matrix(X[ix], Y[tx]).double(); mass = torch.as_tensor(b['mass']/b['mass'].max(), device=dev)
            L1 = soft_nce(s, mass, torch.as_tensor(b['image_weight'], device=dev), torch.as_tensor(b['text_weight'], device=dev)); L1.backward()
            g1 = torch.cat([p.grad.flatten() for p in m.parameters()]).clone(); m.zero_grad()
            pb = fr.build(fr.from_edge_indices(b['edges']), 'RECT'); ps = S.score_pairs(m, X, Y, pb.pairs).double()
            L2 = conditional_loss(ps, pb); L2.backward(); g2 = torch.cat([p.grad.flatten() for p in m.parameters()]).clone(); m.zero_grad()
            dl, dg = float((L1-L2).abs()), float((g1-g2).abs().max()/g1.abs().max())
            ok &= dl < 1e-6 and dg < 1e-4
            a1.setdefault(r, []).append(dict(loss_abs=dl, grad_rel_max=dg, pairs=len(pb.pairs)))
    rec('A1_RECT_math_equals_CR1', ok, per_graph=a1, note='CR3 RECT trains through the unchanged CR2 path; this checks the package RECT formulation agrees')
    # ---------- A2 same draws (100 steps) + A7 resume
    p_rel = graph_path(c, 'J_KEEP_TOP1'); C = {}
    for cons in ('RECT', 'COMPLETE'):
        d = out/f'draws_{cons}'
        C[cons] = S.train(CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared', p_rel, d, 'ACCEPT', cons, dev, steps=100, check_frozen=False, saves=(0, 100), draw_log=out/f'draws_{cons}.jsonl')
    la, lb = (out/'draws_RECT.jsonl').read_text().splitlines(), (out/'draws_COMPLETE.jsonl').read_text().splitlines()
    rec('A2_same_draws', la == lb and len(la) == 100, steps=len(la), identical=la == lb, rect_cost=C['RECT']['cost'], complete_cost=C['COMPLETE']['cost'])
    # ---------- A3/A4/A6 on several real graphs
    a3 = {}; ok3 = True
    for cc, r in [('Flickr60', 'J_KEEP_TOP1'), ('Flickr60', 'OBS'), ('Flickr00', 'OBS'), ('CC152K_RECOVERED_20261104', 'BASE_KEEP_TOP1'), ('Flickr60', 'KEEP_ORACLE')]:
        p = graph_path(cc, r); h0 = sha(p); mat = load_npz(p).tocsr(); fr = FixedRelation(mat); rel = C1.ExposedRelation(mat); g = np.random.default_rng(7)
        stats = dict(batches=20, target_sum_max_err=0.0, support_missing=0, old_candidates_missing=0, extra_queries=0, max_support_image_query=0, mean_pairs_rect=0, mean_pairs_complete=0)
        for _ in range(20):
            b = rel.batch_edges(g, 64); d = fr.from_edge_indices(b['edges']); pr = fr.build(d, 'RECT'); pc = fr.build(d, 'COMPLETE')
            stats['extra_queries'] += len(pc.views) - (len(d.images)+len(d.texts)); stats['mean_pairs_rect'] += len(pr.pairs)/20; stats['mean_pairs_complete'] += len(pc.pairs)/20
            for v in pc.views:
                stats['target_sum_max_err'] = max(stats['target_sum_max_err'], abs(v.target.sum()-1))
                src = fr.transposed if v.direction == 'i2t' else fr.matrix
                sup = src.getrow(v.query).indices; stats['support_missing'] += int(len(np.setdiff1d(sup, v.candidates)))
                old = d.texts if v.direction == 'i2t' else d.images; stats['old_candidates_missing'] += int(len(np.setdiff1d(old, v.candidates)))
                if v.direction == 'i2t': stats['max_support_image_query'] = max(stats['max_support_image_query'], len(sup))
        stats['graph_unchanged'] = sha(p) == h0
        ok3 &= stats['target_sum_max_err'] < 1e-12 and stats['support_missing'] == 0 and stats['old_candidates_missing'] == 0 and stats['extra_queries'] == 0 and stats['graph_unchanged']
        a3[f'{cc}:{r}'] = stats
    fx = FixedRelation(csr_matrix(np.array([[.5, .5, 0], [0, 0, 1.], [1., 0, 0]])))
    dd = fx.from_edge_indices(np.array([0, 2, 3])); vr = [v for v in fx.build(dd, 'RECT').views if v.direction == 't2i' and v.query == 0][0]
    vc = [v for v in fx.build(dd, 'COMPLETE').views if v.direction == 't2i' and v.query == 0][0]
    fixture = dict(rect_candidates=vr.candidates.tolist(), rect_target=vr.target.tolist(), complete_candidates=vc.candidates.tolist(), complete_target=vc.target.tolist())
    ok6 = np.allclose(sorted(vc.target[vc.target > 0]), [.5, .5]) and a3['Flickr00:OBS']['max_support_image_query'] >= 5
    rec('A3_A4_complete_targets_and_queries', ok3, graphs=a3); rec('A6_fixture_and_multicaption', ok6, fixture=fixture)
    # ---------- A5 score invariance on a COMPLETE pair list (nonzero head)
    fr = FixedRelation(load_npz(p_rel).tocsr()); rel = C1.ExposedRelation(load_npz(p_rel).tocsr()); b = rel.batch_edges(np.random.default_rng(9), 64)
    pb = fr.build(fr.from_edge_indices(b['edges']), 'COMPLETE')
    with torch.no_grad():
        s0 = S.score_pairs(m, X, Y, pb.pairs); s1 = S.score_pairs(m, X, Y, pb.pairs, chunk=257)
        perm = np.random.default_rng(1).permutation(len(pb.pairs)); s2 = S.score_pairs(m, X, Y, pb.pairs[perm])
        pi = np.random.default_rng(2).permutation(len(X)); inv = np.argsort(pi); pt = np.random.default_rng(3).permutation(len(Y)); invt = np.argsort(pt)
        Xp, Yp = X[torch.as_tensor(pi, device=dev)], Y[torch.as_tensor(pt, device=dev)]
        q = pb.pairs.copy(); q[:, 0] = inv[q[:, 0]]; q[:, 1] = invt[q[:, 1]]; s3 = S.score_pairs(m, Xp, Yp, q)
    rec('A5_score_invariance', float((s0-s1).abs().max()) < 1e-5 and float((s2-s0[torch.as_tensor(perm, device=dev)]).abs().max()) < 1e-5 and float((s3-s0).abs().max()) == 0,
        chunk_max_abs=float((s0-s1).abs().max()), reorder_max_abs=float((s2-s0[torch.as_tensor(perm, device=dev)]).abs().max()), id_perm_max_abs=float((s3-s0).abs().max()), pairs=len(pb.pairs))
    # ---------- A7 resume COMPLETE
    dA, dB = out/'resume_A', out/'resume_B'
    S.train(CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared', p_rel, dA, 'ACCEPT', 'COMPLETE', dev, steps=300, check_frozen=False, saves=(0, 250, 300))
    dB.mkdir(exist_ok=True)
    for f in ('step_000000.pt', 'step_000250.pt', 'TRAIN.jsonl'): shutil.copy(dA/f, dB/f)
    rB = S.train(CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared', p_rel, dB, 'ACCEPT', 'COMPLETE', dev, steps=300, check_frozen=False, saves=(0, 250, 300))
    ka, kb = torch.load(dA/'step_000300.pt', weights_only=False), torch.load(dB/'step_000300.pt', weights_only=False)
    rec('A7_exact_resume', all(torch.equal(ka['model'][k], kb['model'][k]) for k in ka['model']) and ka['rng_numpy'] == kb['rng_numpy'] and rB['resumed_from_step'] == 250,
        note='training code reads FIT bank, CR1 prepared, fixed graph only')
    # ---------- R reuse receipts
    recs = {}
    for cc, r, hd in HIST:
        att = (hd/'ATTEMPTS.txt').read_text().split('\n')[0].split(); orig_gpu = ' '.join(att[2:-1])
        d = out/f'reuse_{cc}_{r}'
        S.train(CR1/'banks'/cc/'fit_bank.npz', CR1/cc/'prepared', graph_path(cc, r), d, 'ACCEPT', 'RECT', dev, steps=250, check_frozen=False, saves=(0, 250))
        k1 = torch.load(d/'step_000250.pt', weights_only=False, map_location='cpu'); k0 = torch.load(hd/'step_000250.pt', weights_only=False, map_location='cpu')
        bit = all(torch.equal(k1['model'][k], k0['model'][k]) for k in k0['model'])
        diff = max(float((k1['model'][k]-k0['model'][k]).abs().max()) for k in k0['model'])
        lg = [q for d_ in ('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/logs', '/home/infres/yinwang/CS_QMI/handoff/cr2_20261104/logs') for q in Path(d_).glob(f'*_{att[0]}.out')]
        jl = next((l for l in lg[0].read_text().splitlines() if l.startswith('JOB ')), '') if lg else ''
        recs[f'{cc}:{r}'] = dict(historical_dir=str(hd), historical_attempt=' '.join(att), historical_job_line=jl, historical_gpu=orig_gpu, lock_gpu=LOCK, this_gpu=gpu,
                                 step250_bitwise=bit, step250_max_abs_param_diff=diff, decision='REUSE' if (orig_gpu == LOCK and bit) else 'RERUN',
                                 historical_step8000_sha256=sha(hd/'step_008000.pt'))
    rec('R_reuse_receipts', True, receipts=recs)
    (out/'REUSE_OR_RERUN_RECEIPTS.json').write_text(json.dumps(recs, indent=1))
    allok = all(v['ok'] for v in R.values())
    (out/'ACCEPTANCE_CR3.json').write_text(json.dumps(dict(all_pass=allok, checks=R, seconds=time.perf_counter()-t0, test_read=False, truth_read=False), indent=1, default=str))
    print('ALL_PASS' if allok else 'ACCEPTANCE_FAIL')


if __name__ == '__main__': main()
