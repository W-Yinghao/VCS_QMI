"""CR2 real small-sample acceptance (GPU). No TEST; no FIT truth.

 S1 sparse matrix: built CSR graphs vs frozen CR1 CAP weights (row mass exactly m_j, anchors, nnz/row, roundtrip)
 S2 row mass / text marginal identical to CR1 P_W; control row counts
 S3 sampler expansion: CSR(OBS) Relation == CR1 Relation.observed batches; rectangle targets == fixed R entries
 S4 soft-NCE: weighted unique-item loss == multiplicity expansion on a KEEP_SOFT batch
 S5 training-path equivalence: 250 steps OBS via the CR2 CSR trainer == CR1 OBS trainer path (same job, bitwise);
    also compared with the stored CR1 Flickr00 OBS step-250 checkpoint
 S6 score invariance (nonzero head): item reorder, chunk size, ID renaming
 S7 DEV bridge: step-0 CR2 student reproduces the NR1 base metrics on Flickr and CC DEV; nonzero head equals NR1 G
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.sparse import csr_matrix, load_npz, save_npz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'reference'))
import cr2_student as S                                          # noqa: E402
C1 = S.C1
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code/src')
import eval_student as ES                                        # noqa: E402 (CR1 bridge, frozen)
from cr1.relations import Relation                               # noqa: E402
from cr1.objectives import soft_nce                              # noqa: E402
from cr1.models import GlobalScorer                              # noqa: E402
from cr2_relations import anchored_relation                      # noqa: E402

CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
CCN = '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104'
R = {}


def rec(name, ok, **kw):
    R[name] = dict(ok=bool(ok), **kw); print(name, 'PASS' if ok else 'FAIL', json.dumps(kw, default=str)[:700], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--relations', required=True)
    ap.add_argument('--device', default='cuda'); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = a.device
    det = C1.setup_determinism(dev); t0 = time.perf_counter()
    # ---------- S1/S2 for every built graph
    s1 = {}; ok1 = True
    for cond in ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104'):
        bank = C1.Bank(CR1/'banks'/cond/'fit_bank.npz'); obs = bank.observed_image_for_text
        c = np.load(CR1/cond/'prepared'/'graph.npz')['candidates']; h = (c == obs[:, None]).any(1)
        for src in ('BASE', 'J', 'BCE', 'NCE'):
            w = np.load(CR1/cond/'relations'/f'{src}_CAP.npz')['weights']; m = w.sum(1)
            for pol in ('KEEP_SOFT', 'KEEP_TOP1'):
                u = f'CR2_{cond}_{src}_{pol}_s1001'; p = Path(a.relations)/cond/f'{u}.npz'
                A = load_npz(p).tocsr(); rm = np.asarray(A.sum(1)).ravel(); om = np.asarray(A[np.arange(len(obs)), obs]).ravel()
                nnz = np.diff(A.indptr)
                again = anchored_relation(c, w, obs, len(bank.image_global), policy=pol, image_content_keys=bank.image_content_key.astype(str), repair_fraction=.5).matrix
                d = dict(row_mass_max_abs_diff=float(np.abs(rm-m).max()), text_marginal_max_abs_diff=float(np.abs(rm/rm.sum()-m/m.sum()).max()),
                         H_rows_all_mass_on_obs=float(np.abs(om[h]-m[h]).max()) if h.any() else 0.0,
                         H0_obs_floor_ok=bool((om[~h] >= .5*m[~h]-1e-12).all()), H_rows_single_edge=bool((nnz[h] == 1).all()),
                         H0_nnz_max=int(nnz[~h].max()) if (~h).any() else 0, rebuild_bitwise=bool((A != again).nnz == 0),
                         total_mass_vs_CR1=float(A.data.sum()-w.sum()))
                good = (d['row_mass_max_abs_diff'] <= 1e-10 and d['H_rows_all_mass_on_obs'] <= 1e-12 and d['H0_obs_floor_ok'] and d['H_rows_single_edge']
                        and d['H0_nnz_max'] <= (c.shape[1]+1 if pol == 'KEEP_SOFT' else 2) and d['rebuild_bitwise'])
                ok1 &= good; s1[u] = d
        for pol in ('SUPPORT_OBS', 'RANDOM_SUPPORT'):
            u = f'CR2_{cond}_{pol}_s1001'; A = load_npz(Path(a.relations)/cond/f'{u}.npz').tocsr(); rm = np.asarray(A.sum(1)).ravel()
            kept = rm > 0; d = dict(rows_kept=int(kept.sum()), H_rows=int(h.sum()), one_hot=bool(np.allclose(A.data, 1)),
                                    edges_are_observed=bool((A.indices == obs[np.repeat(np.arange(len(obs)), np.diff(A.indptr))]).all()),
                                    support_rows_equal_H=bool((kept == h).all()) if pol == 'SUPPORT_OBS' else None,
                                    overlap_with_H=int((kept & h).sum()))
            ok1 &= d['rows_kept'] == d['H_rows'] and d['one_hot'] and d['edges_are_observed'] and (pol != 'SUPPORT_OBS' or d['support_rows_equal_H'])
            s1[u] = d
    rec('S1_S2_sparse_graphs_row_mass', ok1, graphs=s1)
    # ---------- S3 sampler expansion on Flickr00
    bank, prep, graph = C1.get_prepared(CR1/'banks'/'Flickr00'/'fit_bank.npz', CR1/'Flickr00'/'prepared')
    obs = bank.observed_image_for_text; ni = len(bank.image_global)
    obs_csr = csr_matrix((np.ones(len(obs)), (np.arange(len(obs)), obs)), shape=(len(obs), ni))
    p_obs = out/'OBS_full_csr.npz'; save_npz(p_obs, obs_csr)
    ra, rb = C1.ExposedRelation.observed(obs, ni), C1.ExposedRelation(load_npz(p_obs))
    g1, g2 = np.random.default_rng(20261104201), np.random.default_rng(20261104201); same = True
    for _ in range(200):
        x, y = ra.batch_edges(g1, 64), rb.batch_edges(g2, 64)
        same &= all(np.array_equal(x[k], y[k]) for k in ('images', 'texts', 'mass', 'edges', 'image_weight', 'text_weight'))
    soft = load_npz(Path(a.relations)/'Flickr00'/'CR2_Flickr00_J_KEEP_SOFT_s1001.npz').tocsr(); rs = C1.ExposedRelation(soft)
    gg = np.random.default_rng(20261104201); b = rs.batch_edges(gg, 64)
    full = soft[b['texts']][:, b['images']].toarray().T
    exp_ok = np.array_equal(b['mass'], full) and set(zip(rs.image_edges[b['edges']], rs.text_edges[b['edges']])) <= {(i, t) for i in b['images'] for t in b['texts']}
    rec('S3_sampler_expansion', same and exp_ok, obs_csr_vs_CR1_observed_200_batches_identical=bool(same), rectangle_targets_equal_fixed_R=bool(exp_ok),
        keep_soft_batch=dict(unique_images=len(b['images']), unique_texts=len(b['texts']), target_nnz=int((b['mass'] > 0).sum())))
    # ---------- S4 soft-NCE weighted == expanded
    X, Y = C1.features(bank, dev); ini = prep['student_common_initialization']
    m = GlobalScorer(768, ini['a'], ini['b'], 1001).to(dev)
    with torch.no_grad():
        m.global_residual[-1].weight.normal_(0, .05, generator=torch.Generator(device=dev).manual_seed(3)); m.global_residual[-1].bias.fill_(.3)
    ix, tx = torch.as_tensor(b['images'], device=dev), torch.as_tensor(b['texts'], device=dev)
    s = m.matrix(X[ix], Y[tx]).double(); mass = torch.as_tensor(b['mass']/b['mass'].max(), device=dev)
    L = soft_nce(s, mass, torch.as_tensor(b['image_weight'], device=dev), torch.as_tensor(b['text_weight'], device=dev))
    ci = torch.as_tensor(np.round(b['image_weight']*64).astype(int), device=dev); ct = torch.as_tensor(np.round(b['text_weight']*64).astype(int), device=dev)
    ai = mass/mass.sum(1, keepdim=True); at = mass.T/mass.T.sum(1, keepdim=True)
    li = -(ai*F.log_softmax(s, 1)).sum(1).repeat_interleave(ci).mean(); lt = -(at*F.log_softmax(s.T, 1)).sum(1).repeat_interleave(ct).mean()
    dd = float((L-.5*(li+lt)).abs())
    rec('S4_soft_NCE', dd < 1e-12, weighted_vs_expanded_abs=dd)
    # ---------- S5 training path equivalence (250 steps)
    pa, pb = out/'train_cr1_path', out/'train_cr2_path'
    rel = C1.ExposedRelation.observed(obs, ni); rng = np.random.default_rng(20261104201)
    mA = GlobalScorer(768, ini['a'], ini['b'], 1001).to(dev)
    def fnA(mm):
        bb = rel.batch_edges(rng, 64); i_ = torch.as_tensor(bb['images'], device=dev); t_ = torch.as_tensor(bb['texts'], device=dev)
        sc = mm.matrix(X[i_], Y[t_]); ms = torch.as_tensor(bb['mass']/bb['mass'].max(), device=dev, dtype=sc.dtype)
        return soft_nce(sc, ms, torch.as_tensor(bb['image_weight'], device=dev, dtype=sc.dtype), torch.as_tensor(bb['text_weight'], device=dev, dtype=sc.dtype)), bb['pair_count'], {}
    old = C1.SAVES; C1.SAVES = (0, 250); pa.mkdir(exist_ok=True)
    C1.optim_loop(mA, fnA, pa, 250, dev, rng, {'acceptance': 'cr1_path'}); C1.SAVES = old
    Path(str(p_obs)+'.json').write_text(json.dumps({'relation_sha256': S.sha256(p_obs), 'unit_id': 'ACCEPT_OBS'}))
    S.train(CR1/'banks'/'Flickr00'/'fit_bank.npz', CR1/'Flickr00'/'prepared', p_obs, pb, 'ACCEPT_OBS', dev, steps=250, check_frozen=False, saves=(0, 250))
    ka = torch.load(pa/'step_000250.pt', weights_only=False); kb = torch.load(pb/'step_000250.pt', weights_only=False)
    bit = all(torch.equal(ka['model'][k], kb['model'][k]) for k in ka['model'])
    kc = torch.load(CR1/'Flickr00'/'students'/'OBS'/'step_000250.pt', weights_only=False, map_location='cpu')
    cross = max(float((ka['model'][k].cpu()-kc['model'][k]).abs().max()) for k in ka['model'])
    cr1_gpu = (CR1/'Flickr00'/'students'/'OBS'/'ATTEMPTS.txt').read_text().split('\n')[0]
    rec('S5_training_path_equivalence', bit, cr2_csr_vs_cr1_path_bitwise=bit, vs_stored_CR1_F00_OBS_step250_max_abs=cross,
        stored_cr1_attempt=cr1_gpu, this_gpu=torch.cuda.get_device_name())
    # ---------- S6 score invariance with nonzero head (the 250-step model)
    m2 = GlobalScorer(768).to(dev); m2.load_state_dict(kb['model']); m2.eval()
    r_ = np.random.default_rng(5); i_ = torch.as_tensor(r_.integers(0, ni, 64), device=dev); t_ = torch.as_tensor(r_.integers(0, len(obs), 64), device=dev)
    with torch.no_grad():
        g = m2.matrix(X[i_], Y[t_]); perm = torch.randperm(64, generator=torch.Generator().manual_seed(1)).to(dev)
        gp = m2.matrix(X[i_[perm]], Y[t_[perm]]); g256 = m2.matrix(X[i_], Y[t_], pair_chunk=256)
        # ID renaming: features addressed through a renamed/permuted ID table give the same pair scores
        pi = torch.randperm(ni, generator=torch.Generator().manual_seed(2)).to(dev); inv = torch.argsort(pi)
        gr = m2.matrix(X[pi][inv[i_]], Y[t_])
    rec('S6_score_invariance', float((gp-g[perm][:, perm]).abs().max()) == 0 and float((g-g256).abs().max()) < 1e-5 and float((gr-g).abs().max()) == 0,
        reorder_max_abs=float((gp-g[perm][:, perm]).abs().max()), chunk256_vs_4096_max_abs=float((g-g256).abs().max()), id_renaming_max_abs=float((gr-g).abs().max()))
    # ---------- S7 DEV bridge step 0 (Flickr + CC) and nonzero
    from nr1.evaluate import evaluate
    from nr1.io import FeatureBank
    s7 = {}; ok7 = True
    for cond, cache, truth, base in [('Flickr00', f'{NR1}/cache/dev', f'{NR1}/evaluator_only/dev_positives.json', f'{NR1}/baseline/float64/METRICS.json'),
                                     ('CC152K_RECOVERED_20261104', f'{CCN}/cache/dev', f'{CCN}/evaluator_only/dev_positives.json', f'{CCN}/baseline/METRICS.json')]:
        ini_c = json.loads((CR1/cond/'prepared'/'PREP.json').read_text())['student_common_initialization']
        sd = out/f'step0_{cond}'; sd.mkdir(exist_ok=True)
        torch.save({'model': GlobalScorer(768, ini_c['a'], ini_c['b'], 1001).state_dict(), 'step': 0, 'dimension': 768, 'architecture': 'G'}, sd/'step_000000.pt')
        dims = FeatureBank(cache, 'dev').dims
        ES.bridge(sd/'step_000000.pt', dims, ini_c, sd/'b0.pt')
        r0 = evaluate(cache, truth, sd/'eval_000000', str(sd/'b0.pt'), dev, 100, 'float64')
        bm = json.loads(Path(base).read_text())['metrics']
        s7[cond] = dict(step0_metrics=r0['metrics'], base_metrics=bm, order_vs_base=r0['order_vs_base'])
        ok7 &= r0['metrics'] == bm and r0['order_vs_base']['i2t_queries_order_differs'] == 0 and r0['order_vs_base']['t2i_queries_order_differs'] == 0
    dims = FeatureBank(f'{NR1}/cache/dev', 'dev').dims
    ck = ES.bridge(pb/'step_000250.pt', dims, ini, out/'b250.pt')
    eq = ES.equivalence(ck, dict(**dims, architecture='G', a0=float(ini['a']), b0=float(ini['b']), seed=1001), FeatureBank(f'{NR1}/cache/dev', 'dev'), dev)
    ok7 &= eq['cr1_vs_nr1_fp32_max_abs'] == 0
    rec('S7_DEV_bridge', ok7, step0=s7, nonzero_equivalence=eq)
    allok = all(v['ok'] for v in R.values())
    (out/'ACCEPTANCE_CR2.json').write_text(json.dumps(dict(all_pass=allok, checks=R, determinism=det, seconds=time.perf_counter()-t0, test_read=False,
                                                           truth_read='DEV positives only inside the NR1 evaluator (S7)'), indent=1, default=str)+'\n')
    print('ALL_PASS' if allok else 'ACCEPTANCE_FAIL')


if __name__ == '__main__': main()
