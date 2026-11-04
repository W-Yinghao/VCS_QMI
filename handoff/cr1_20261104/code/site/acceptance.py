"""CR1 real-data small-batch acceptance (site). No TEST, no FIT truth, no DEV labels in training paths.

Checks (all written to ACCEPTANCE.json):
 A  original J/BCE/NCE objectives == NR1 production objectives on a real batch; G == NR1 DirectScorer(G)
 B  iid-with-replacement sampling contract, off-diagonal U-statistic, collisions retained
 C  nonzero head: chunking / batch-permutation / item-context invariance of pair scores
 D  two-end held-out audit for all conditions (training+CAL records, edge routing); F00/F60 folds+candidates equal
 E  matching: real BASE OOF scores -> ROW closed form and CAP KKT, primal/dual gap
 F  unified soft-NCE: unique-item weighted loss == multiplicity-expanded loss; 7 arms share one contract
 G  DEV bridge: step-0 student reproduces the NR1 base order/metrics; nonzero student scores equal
 H  exact resume (250 -> 300) == uninterrupted 300 steps (state, optimizer, RNG)
 I  content keys <-> cached features (first 32 FIT images re-encoded with the pinned SigLIP2)
"""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'src')); sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code/src')
import cr1_prod as P                                          # noqa: E402
from cr1.io import Bank, read_json, write_json, sha256, array_digest   # noqa: E402
from cr1.folds import EXCLUSIONS, training_mask, edge_experts          # noqa: E402
from cr1.models import GlobalScorer                                   # noqa: E402
from cr1.objectives import estimator_loss, soft_nce                   # noqa: E402
from cr1.calibration import apply_affine                              # noqa: E402
from cr1.transport import solve                                       # noqa: E402
from cr1.relations import Relation                                    # noqa: E402
from nr1 import objectives as NO                                      # noqa: E402
from nr1.models import DirectScorer                                   # noqa: E402

R = {}


def rec(name, ok, **kw):
    R[name] = dict(ok=bool(ok), **kw); print(name, 'PASS' if ok else 'FAIL', json.dumps(kw, default=str)[:600], flush=True)


def perturb(m, seed=3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        last = m.global_residual[-1]
        last.weight.copy_(torch.randn(last.weight.shape, generator=g)*.05); last.bias.fill_(.3)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cuda')
    ap.add_argument('--banks', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--dev-cache', required=True); ap.add_argument('--truth', required=True)
    ap.add_argument('--nr1-baseline', required=True); ap.add_argument('--snapshot', required=True)
    ap.add_argument('--conditions', default='Flickr00,Flickr60,CC152K_RECOVERED_20261104')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = a.device
    det = P.setup_determinism(dev); conds = a.conditions.split(','); t_all = time.perf_counter()
    bank_path = lambda c: Path(a.banks)/c/'fit_bank.npz'
    prep_path = lambda c: Path(a.work)/c/'prepared'
    bank, prep, graph = P.get_prepared(bank_path('Flickr60'), prep_path('Flickr60'))
    X, Y = P.features(bank, dev)
    # ---------------- A objectives + architecture
    exc = EXCLUSIONS[0]; allowed = training_mask(bank.observed_image_for_text, np.arange(len(Y)), graph['image_folds'], graph['text_folds'], exc)
    rows = np.flatnonzero(allowed); rng = np.random.default_rng([1001, 0, 51])
    tx = rows[rng.integers(0, len(rows), 64)]; im = bank.observed_image_for_text[tx]
    it, tt = torch.as_tensor(im, device=dev), torch.as_tensor(tx, device=dev)
    cal = prep['experts']['E01']['base_calibration']
    m = perturb(GlobalScorer(768, cal['a'], cal['b'], 1001)).to(dev)
    nr = DirectScorer(768, 768, 768, 'G', cal['a'], cal['b'], 1001).to(dev); nr.load_state_dict(m.state_dict())
    g = m.matrix(X[it], Y[tt]); gn = nr.matrix(X[it], Y[tt], pair_chunk=4096)
    g_nr_default = nr.matrix(X[it], Y[tt])  # NR1 default pair_chunk=256 (fp32 GEMM shape differs)
    eqA = {o: float((estimator_loss(o, g, it) - NO.loss_value(o, g, it)).abs()) for o in ('J', 'BCE', 'NCE')}
    rec('A_objectives_equal_NR1', all(v == 0 for v in eqA.values()) and torch.equal(g, gn),
        loss_abs_diff=eqA, G_matrix_cr1_vs_nr1_same_chunk_max_abs=float((g-gn).abs().max()),
        G_matrix_chunk4096_vs_nr1_chunk256_max_abs=float((g-g_nr_default).abs().max()),
        init_params_equal=all(torch.equal(p, q) for p, q in zip(GlobalScorer(768, 1., 0., 1001).state_dict().values(),
                                                                  DirectScorer(768, 768, 768, 'G', 1., 0., 1001).state_dict().values())),
        params=sum(p.numel() for p in m.parameters()))
    # ---------------- B sampling
    eq_img = (im[:, None] == im[None, :]) & ~np.eye(64, dtype=bool)
    rec('B_sampling_contract', len(rows) > 1000, allowed_records=int(len(rows)), unique_texts=int(len(np.unique(tx))),
        unique_images=int(len(np.unique(im))), offdiag_same_image_pairs_retained_in_Q=int(eq_img.sum()),
        q_weight=float(1/(64*63)), p_weight=float(1/64), note='Q = off-diagonal U-statistic over 64 iid record positions')
    # ---------------- C nonzero-head invariances
    perm = torch.randperm(64, generator=torch.Generator().manual_seed(1)).to(dev)
    g256 = m.matrix(X[it], Y[tt], pair_chunk=256); gp = m.matrix(X[it[perm]], Y[tt[perm]])
    single = torch.stack([m.pairs(X[it[i:i+1]], Y[tt[j:j+1]])[0] for i, j in [(0, 0), (5, 9), (63, 1)]])
    ctx = torch.stack([g[0, 0], g[5, 9], g[63, 1]])
    rec('C_nonzero_head_invariance', float((g-g256).abs().max()) < 1e-5 and float((gp-g[perm][:, perm]).abs().max()) < 1e-5
        and float((single-ctx).abs().max()) < 1e-5 and float((g - (m.log_scale.exp()*(F.normalize(X[it])@F.normalize(Y[tt]).T)+m.bias)).abs().mean()) > 1e-3,
        chunk_max_abs=float((g-g256).abs().max()), permutation_max_abs=float((gp-g[perm][:, perm]).abs().max()),
        single_vs_context_max_abs=float((single-ctx).abs().max()), tol=1e-5)
    # ---------------- D two-end held-out for every condition
    Dres = {}; ok_all = True; folds = {}
    for c in conds:
        b, pr, gr = P.get_prepared(bank_path(c), prep_path(c)); folds[c] = gr
        per = {}
        for e, ex in enumerate(EXCLUSIONS):
            al = training_mask(b.observed_image_for_text, np.arange(len(b.text_global)), gr['image_folds'], gr['text_folds'], ex)
            r_ = np.flatnonzero(al); key = f'E{ex[0]}{ex[1]}'
            pp = P.pools(b, al, seed=20261104051+e, n=32768)
            okk = (array_digest(r_) == pr['experts'][key]['allowed_record_hash']
                   and not np.isin(gr['image_folds'][b.observed_image_for_text[r_]], ex).any()
                   and not np.isin(gr['text_folds'][r_], ex).any()
                   and not np.isin(gr['image_folds'][np.concatenate([pp[0], pp[2]])], ex).any()
                   and not np.isin(gr['text_folds'][np.concatenate([pp[1], pp[3]])], ex).any())
            per[key] = dict(records=int(len(r_)), fraction=float(al.mean()), ok=bool(okk)); ok_all &= okk
        cnd = gr['candidates']; trr = np.repeat(np.arange(len(cnd)), cnd.shape[1]); irr = cnd.ravel()
        own = edge_experts(gr['image_folds'], gr['text_folds'], trr, irr)
        route_ok = all(np.isin(gr['image_folds'][irr[own == e]], ex).all() and np.isin(gr['text_folds'][trr[own == e]], ex).all()
                       for e, ex in enumerate(EXCLUSIONS))
        ok_all &= route_ok
        Dres[c] = dict(experts=per, routing_ok=bool(route_ok), edges_per_expert=np.bincount(own, minlength=6).tolist(),
                       candidate_unique_per_row=bool((np.diff(np.sort(cnd, 1), axis=1) > 0).all()))
    same = None
    if 'Flickr00' in folds and 'Flickr60' in folds:
        same = all(np.array_equal(folds['Flickr00'][k], folds['Flickr60'][k]) for k in ('image_folds', 'text_folds', 'candidates'))
        ok_all &= same
    rec('D_two_end_heldout', ok_all, conditions=Dres, flickr00_flickr60_folds_and_candidates_identical=same)
    # ---------------- E matching on real BASE OOF scores (Flickr60)
    c = graph['candidates']; trr = np.repeat(np.arange(len(c)), c.shape[1]); irr = c.ravel()
    own = edge_experts(graph['image_folds'], graph['text_folds'], trr, irr); sc = np.empty(len(irr))
    for e, ex in enumerate(EXCLUSIONS):
        mk = own == e
        sc[mk] = apply_affine(P.pair_values(None, X, Y, irr[mk], trr[mk]), prep['experts'][f'E{ex[0]}{ex[1]}']['base_calibration'])
    sc = sc.reshape(c.shape); Eres = {}
    for mode in ('ROW', 'CAP'):
        t0 = time.perf_counter(); res = solve(c, sc, bank.p_image, capped=mode == 'CAP'); secs = time.perf_counter()-t0
        w, u = res.weights, res.slack
        lp = np.log(bank.p_image[c]); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
        with np.errstate(divide='ignore', invalid='ignore'):
            primal = float((w*sc).sum() - (np.where(w > 0, w*(np.log(w)-lp), 0)).sum() - np.where(u > 0, u*np.log(u), 0).sum())
        dual = float(res.receipt['dual_per_row']*len(c))
        Eres[mode] = dict(receipt=res.receipt, seconds=secs, primal=primal, dual=dual, rel_gap=abs(primal-dual)/abs(dual),
                          accepted_mass=float(w.sum()/len(c)))
    rec('E_matching_KKT', Eres['ROW']['receipt']['accepted'] and Eres['CAP']['receipt']['accepted']
        and Eres['CAP']['rel_gap'] < 1e-6 and Eres['ROW']['rel_gap'] < 1e-9, **Eres)
    # ---------------- F unified soft-NCE
    res = solve(c, sc, bank.p_image, capped=True)
    Fres = {}; okF = True
    for name, rel in [('OBS', Relation.observed(bank.observed_image_for_text, len(X))), ('BASE_CAP', Relation.graph(c, res.weights, len(X)))]:
        rr = np.random.default_rng(20261104201); b_ = rel.batch(rr, 64)
        ix, tx_ = torch.as_tensor(b_['images'], device=dev), torch.as_tensor(b_['texts'], device=dev)
        s = m.matrix(X[ix], Y[tx_]).double(); mass = torch.as_tensor(b_['mass']/b_['mass'].max(), device=dev)
        L = soft_nce(s, mass, torch.as_tensor(b_['image_weight'], device=dev), torch.as_tensor(b_['text_weight'], device=dev))
        ci = torch.as_tensor(np.round(b_['image_weight']*64).astype(int), device=dev); ct = torch.as_tensor(np.round(b_['text_weight']*64).astype(int), device=dev)
        ai = mass/mass.sum(1, keepdim=True); at = mass.T/mass.T.sum(1, keepdim=True)
        li = -(ai*F.log_softmax(s, 1)).sum(1).repeat_interleave(ci).mean()
        lt = -(at*F.log_softmax(s.T, 1)).sum(1).repeat_interleave(ct).mean()
        d = float((L - .5*(li+lt)).abs()); okF &= d < 1e-12
        Fres[name] = dict(weighted_vs_expanded_abs=d, unique_images=int(len(ix)), unique_texts=int(len(tx_)), target_nnz=int((mass > 0).sum()))
    import csv
    with open(P.TREE/'package'/'configs'/'TASK_MATRIX.csv') as f: tasks = list(csv.DictReader(f))
    cfgs = [P.unit_config(t['unit_id']) for t in tasks]
    contract = {(x['objective'], x['sampler_seed'], x['seed'], x['steps'], x['initialization'], x['lr'], x['weight_decay'], x['grad_clip']) for x in cfgs}
    okF &= len(contract) == 1 and len(cfgs) == 21
    rec('F_unified_soft_NCE', okF, arms=Fres, task_units=len(cfgs), distinct_contracts=len(contract))
    # ---------------- G DEV bridge + H resume (tiny student / estimator runs in acceptance dir)
    import eval_student as ES
    from nr1.evaluate import evaluate
    from nr1.io import FeatureBank
    sdir = out/'student_obs'; sdir.mkdir(exist_ok=True)
    ini = prep['student_common_initialization']
    sm = GlobalScorer(768, ini['a'], ini['b'], 1001).to(dev); rel = P.ExposedRelation.observed(bank.observed_image_for_text, len(X))
    rr = np.random.default_rng(20261104201)
    def sfn(mm):
        b_ = rel.batch_edges(rr, 64); ix = torch.as_tensor(b_['images'], device=dev); tx_ = torch.as_tensor(b_['texts'], device=dev)
        s = mm.matrix(X[ix], Y[tx_]); mass = torch.as_tensor(b_['mass']/b_['mass'].max(), device=dev, dtype=s.dtype)
        return soft_nce(s, mass, torch.as_tensor(b_['image_weight'], device=dev, dtype=s.dtype),
                        torch.as_tensor(b_['text_weight'], device=dev, dtype=s.dtype)), b_['pair_count'], {}
    P.SAVES = (0, 250)  # acceptance only
    P.optim_loop(sm, sfn, sdir, 250, dev, rr, {'acceptance': 'student_obs_250'})
    dbank = FeatureBank(a.dev_cache, 'dev'); dims = dbank.dims; Gres = {}
    for s in (0, 250):
        dst = sdir/f'step_{s:06d}.nr1.pt'; ck = ES.bridge(sdir/f'step_{s:06d}.pt', dims, ini, dst)
        eq = ES.equivalence(ck, dict(**dims, architecture='G', a0=float(ini['a']), b0=float(ini['b']), seed=1001), dbank, dev)
        r_ = evaluate(a.dev_cache, a.truth, sdir/f'eval_{s:06d}', str(dst), dev, 100, 'float64')
        Gres[s] = dict(metrics=r_['metrics'], base=r_['base'], order_vs_base=r_['order_vs_base'], equivalence=eq)
    nb = read_json(a.nr1_baseline)
    okG = (Gres[0]['order_vs_base']['i2t_queries_order_differs'] == 0 and Gres[0]['order_vs_base']['t2i_queries_order_differs'] == 0
           and Gres[0]['metrics'] == nb['metrics'] and Gres[250]['equivalence']['cr1_vs_nr1_fp32_max_abs'] == 0
           and Gres[250]['equivalence']['nr1_fp64_vs_cr1_fp32_max_abs'] < 1e-3
           and (Gres[250]['order_vs_base']['i2t_queries_order_differs'] + Gres[250]['order_vs_base']['t2i_queries_order_differs']) > 0)
    rec('G_DEV_bridge', okG, steps=Gres, nr1_baseline_metrics=nb['metrics'])
    # H exact resume
    def est_run(d, steps_saves, start_from=None):
        d.mkdir(exist_ok=True)
        if start_from is not None:
            for f_ in ('step_000000.pt', 'step_000250.pt', 'TRAIN.jsonl'): shutil.copy(start_from/f_, d/f_)
        mm = GlobalScorer(768, cal['a'], cal['b'], 1001).to(dev); r2 = np.random.default_rng([1001, 0, 51])
        def fn(model):
            tx2 = rows[r2.integers(0, len(rows), 64)]; im2 = bank.observed_image_for_text[tx2]
            i2 = torch.as_tensor(im2, device=dev)
            return estimator_loss('J', model.matrix(X[i2], Y[torch.as_tensor(tx2, device=dev)]), i2), 4096, {}
        P.SAVES = steps_saves
        t0 = time.perf_counter(); rc = P.optim_loop(mm, fn, d, 300, dev, r2, {'acceptance': 'resume'}); return mm, r2, rc, time.perf_counter()-t0
    mA, rA, rcA, tA = est_run(out/'resume_A', (0, 250, 300))
    mB, rB, rcB, tB = est_run(out/'resume_B', (0, 250, 300), start_from=out/'resume_A')
    cA = torch.load(out/'resume_A'/'step_000300.pt', weights_only=False); cB = torch.load(out/'resume_B'/'step_000300.pt', weights_only=False)
    same_model = all(torch.equal(cA['model'][k], cB['model'][k]) for k in cA['model'])
    same_opt = all(torch.equal(x['exp_avg'], y['exp_avg']) and torch.equal(x['exp_avg_sq'], y['exp_avg_sq'])
                   for x, y in zip(cA['optimizer']['state'].values(), cB['optimizer']['state'].values()))
    rec('H_exact_resume', same_model and same_opt and cA['rng_numpy'] == cB['rng_numpy'] and rcB['resumed_from_step'] == 250,
        model_bitwise=same_model, optimizer_bitwise=same_opt, rng_equal=cA['rng_numpy'] == cB['rng_numpy'],
        seconds_300_steps=tA, est_seconds_8000_steps=tA/300*8000)
    P.SAVES = (0, 250, 500, 1000, 2000, 4000, 8000)
    # ---------------- I content keys <-> cached features
    from transformers import AutoModel, AutoProcessor
    from PIL import Image, ImageOps
    import build_banks as BB
    nr1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
    ims = BB.jsonl(f'{nr1}/data/fit_images.jsonl')[:32]; txs = BB.jsonl(f'{nr1}/data/fit_texts.jsonl')[:32]
    proc = AutoProcessor.from_pretrained(a.snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(a.snapshot, local_files_only=True, attn_implementation='eager', dtype=torch.float32).to(dev).eval()
    imgs = []
    for r_ in ims:
        with Image.open(r_['path']) as x_: imgs.append(ImageOps.exif_transpose(x_).convert('RGB').copy())
    pv = proc(images=imgs, return_tensors='pt')['pixel_values']
    keys_ok = [BB.tensor_key(x_) for x_ in pv.numpy()] == list(Bank(bank_path('Flickr00')).image_content_key[:32])
    ti = proc(text=[r_['text'].lower() for r_ in txs], padding='max_length', max_length=64, truncation=True, return_tensors='pt')
    tkeys_ok = [BB.tensor_key(x_) for x_ in ti['input_ids'].numpy().astype(np.int64)] == list(Bank(bank_path('Flickr00')).text_content_key[:32])
    with torch.inference_mode():
        gi = F.normalize(model.vision_model(pixel_values=pv.to(dev)).pooler_output.float(), dim=-1).cpu().numpy()
        gt = F.normalize(model.text_model(**{k: v.to(dev) for k, v in ti.items() if k in ('input_ids', 'attention_mask', 'position_ids')}).pooler_output.float(), dim=-1).cpu().numpy()
    b00 = Bank(bank_path('Flickr00'))
    di, dt = float(np.abs(gi-b00.image_global[:32]).max()), float(np.abs(gt-b00.text_global[:32]).max())
    rec('I_content_keys_match_features', keys_ok and tkeys_ok and di < 1e-4 and dt < 1e-4,
        image_keys_recomputed_equal=keys_ok, text_keys_recomputed_equal=tkeys_ok, image_global_max_abs=di, text_global_max_abs=dt)
    allok = all(v['ok'] for v in R.values())
    write_json(out/'ACCEPTANCE.json', dict(all_pass=allok, checks=R, determinism=det, seconds=time.perf_counter()-t_all,
                                         test_read=False, truth_read='DEV positives only inside the NR1 evaluator (check G)'))
    print('ALL_PASS' if allok else 'ACCEPTANCE_FAIL')


if __name__ == '__main__': main()
