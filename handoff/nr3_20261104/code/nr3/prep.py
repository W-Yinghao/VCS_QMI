#!/usr/bin/env python3
"""NR3 per-condition preparation (FIT only; no DEV, TEST or truth):
  1 fixed diagnostic pools CAL-A / READ-B;
  2 A: HREF-aware 2-parameter balanced-BCE calibration of the base cosine on CAL-A (L-BFGS-B, NR1 bounds, <=500 iterations,
       start = condition's NR1 a0,b0) -> calibration_HCAL.json (same binding keys as the NR1 calibration file) + receipt;
  3 B: load the registered NR2 HREF-NCE seed-1001 step-8000 checkpoint, calibrate a*s_parent+b on CAL-A (start (1,0)),
       fold exactly into G (nr3.ref_init.fold_positive_affine = package reference, byte copy), save the common prefix;
       parent files are only read.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
from nr1.io import FeatureBank, load_observed, read_json, sha256, write_json
from nr1.models import DirectScorer
from nr2.reference import Marginals
from nr3 import pools as PL
from nr3.paths import OUT, cond_paths
from nr3.ref_init import fit_reference_affine, fold_positive_affine, pooled_j

SEEDS = dict(CAL_A=20261104031, READ_B=20261104032); N_POOL = 32768; BOUNDS = dict(a=(.1, 100.), b=(-20., 20.))


def jdir(pool, fP, fQ, a, b):
    return {d: pooled_j(np.stack([fP[d], fP[d]]), np.stack([fQ[d], fQ[d]]), np.stack([pool[d]['P_logw']]*2), np.stack([pool[d]['Q_logw']]*2), a, b) for d in ('i2t', 't2i')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); a_ = ap.parse_args(); c = a_.cond
    P = cond_paths(c); od = OUT/'prep'/c; od.mkdir(parents=True, exist_ok=True)
    if (od/'PREP_DONE.json').exists(): print('exists', od); return
    bank = FeatureBank(P['fit'], 'fit'); im, tx = load_observed(P['observed'], bank)
    marg = Marginals(im, tx, len(bank.image_ids), len(bank.text_ids)); tab = np.load(P['tables']); top = dict(i2t=tab['i2t'], t2i=tab['t2i'])
    cal1 = read_json(P['calibration'])
    if cal1['cache_manifest_sha256'] != sha256(bank.root/'manifest.json') or cal1['observed_sha256'] != sha256(P['observed']):
        sys.exit('NR1 calibration binding mismatch')
    tman = read_json(Path(P['tables']).parent/'TABLES_MANIFEST.json')
    if tman['tables_sha256'] != sha256(P['tables']) or tman['cache_manifest_sha256'] != sha256(bank.root/'manifest.json'):
        sys.exit('proposal table binding mismatch')
    pools, psha = {}, {}
    for name, seed in SEEDS.items():
        pools[name] = PL.build(marg, top, seed, N_POOL); PL.save(pools[name], od/f'POOL_{name}.npz'); psha[name] = sha256(od/f'POOL_{name}.npz')
    X = torch.from_numpy(np.array(bank.x, copy=True)).cuda(); Y = torch.from_numpy(np.array(bank.y, copy=True)).cuda()
    Xn = torch.nn.functional.normalize(X.double(), dim=1); Yn = torch.nn.functional.normalize(Y.double(), dim=1)
    rec = dict(condition=c, pools=dict(seeds=SEEDS, n_per_direction_P=N_POOL, n_per_direction_Qh=N_POOL, sha256=psha,
                                       definition='nr3.pools.build: P = observed records; Q_h = independent observed-record query x proposal draw; exact full-mixture log w'),
               inputs=dict(cache_manifest_sha256=sha256(bank.root/'manifest.json'), observed_sha256=sha256(P['observed']),
                           tables_sha256=sha256(P['tables']), nr1_calibration_sha256=sha256(P['calibration']), nr1_a0=cal1['a0'], nr1_b0=cal1['b0']))
    # ---- A: HREF-aware calibration of base cosine
    f = {nm: ({d: PL.base_cos(Xn, Yn, d, pools[nm][d]['P_query'], pools[nm][d]['P_cand']) for d in pools[nm]},
              {d: PL.base_cos(Xn, Yn, d, pools[nm][d]['Q_query'], pools[nm][d]['Q_cand']) for d in pools[nm]}) for nm in pools}
    st = PL.stack(pools['CAL_A'], *f['CAL_A'])
    fa = fit_reference_affine(*st, initial=(cal1['a0'], cal1['b0']), a_bounds=BOUNDS['a'], b_bounds=BOUNDS['b'], maxiter=500)
    stB = PL.stack(pools['READ_B'], *f['READ_B'])
    recA = dict(fit=fa.json(), score='base cosine (float64, sample-wise L2 FIT globals)', start=[cal1['a0'], cal1['b0']], bounds=BOUNDS, maxiter=500,
                CAL_A_J=dict(nr1=dict(pooled=pooled_j(*st, cal1['a0'], cal1['b0']), **jdir(pools['CAL_A'], *f['CAL_A'], cal1['a0'], cal1['b0'])),
                             hcal=dict(pooled=pooled_j(*st, fa.a, fa.b), **jdir(pools['CAL_A'], *f['CAL_A'], fa.a, fa.b))),
                READ_B_J=dict(nr1=pooled_j(*stB, cal1['a0'], cal1['b0']), hcal=pooled_j(*stB, fa.a, fa.b)))
    hcal = dict(a0=fa.a, b0=fa.b, cache_manifest_sha256=cal1['cache_manifest_sha256'], observed_sha256=cal1['observed_sha256'],
                method='NR3 A: HREF-aware balanced-BCE affine of base cosine on CAL-A (g = a f + b - log w), nr3.ref_init.fit_reference_affine',
                receipt=recA, pool_CAL_A_sha256=psha['CAL_A'], replaces_for_initialisation_only=str(P['calibration']))
    write_json(od/'calibration_HCAL.json', hcal); rec['A'] = recA
    # ---- B: common prefix = calibrated, folded NR2 HREF-NCE s1001 step 8000
    par = P['parent']('NCE'); ckp = par/'step_008000.pt'
    ck = torch.load(ckp, map_location='cuda', weights_only=False)
    done = read_json(par/'DONE.json')
    if ck['step'] != 8000 or ck['binding_sha256'] != done['binding_sha256'] or ck['binding']['config']['objective'] != 'NCE' \
            or ck['binding']['config']['reference'] != 'HREF' or ck['binding']['config']['seed'] != 1001 or ck['binding']['config']['condition'] not in (c, c + '_20261104'):
        sys.exit(f'BLOCKED_ARTIFACT parent mismatch {ckp}')
    if ck['binding']['cache_manifest_sha256'] != rec['inputs']['cache_manifest_sha256'] or ck['binding']['observed_sha256'] != rec['inputs']['observed_sha256'] \
            or ck['binding']['tables_sha256'] != rec['inputs']['tables_sha256']:
        sys.exit('BLOCKED_ARTIFACT parent data binding mismatch')
    m = DirectScorer(**ck['binding']['model_config']).cuda(); m.load_state_dict(ck['model']); m.eval()
    sp = {nm: ({d: PL.model_scores(m, X, Y, d, pools[nm][d]['P_query'], pools[nm][d]['P_cand']) for d in pools[nm]},
               {d: PL.model_scores(m, X, Y, d, pools[nm][d]['Q_query'], pools[nm][d]['Q_cand']) for d in pools[nm]}) for nm in pools}
    stP = PL.stack(pools['CAL_A'], *sp['CAL_A'])
    fb = fit_reference_affine(*stP, initial=(1., 0.), a_bounds=BOUNDS['a'], b_bounds=BOUNDS['b'], maxiter=500)
    mf = fold_positive_affine(m, fb.a, fb.b); mf.eval()
    sf = {nm: ({d: PL.model_scores(mf, X, Y, d, pools[nm][d]['P_query'], pools[nm][d]['P_cand']) for d in pools[nm]},
               {d: PL.model_scores(mf, X, Y, d, pools[nm][d]['Q_query'], pools[nm][d]['Q_cand']) for d in pools[nm]}) for nm in pools}
    err = max(float(np.abs(sf[nm][k][d] - (fb.a * sp[nm][k][d] + fb.b)).max()) for nm in pools for k in (0, 1) for d in ('i2t', 't2i'))
    scale = max(float(np.abs(fb.a * sp[nm][k][d] + fb.b).max()) for nm in pools for k in (0, 1) for d in ('i2t', 't2i'))
    stPB = PL.stack(pools['READ_B'], *sp['READ_B'])
    for k, v in m.state_dict().items():
        if not torch.equal(v, ck['model'][k]): sys.exit('parent model modified in memory')
    pre = dict(step=8000, model={k: v.detach().cpu() for k, v in mf.state_dict().items()},
               binding=dict(ck['binding'], nr3_prefix=dict(parent_checkpoint=str(ckp), parent_checkpoint_sha256=sha256(ckp),
                                                           parent_binding_sha256=ck['binding_sha256'], a=fb.a, b=fb.b, condition=c)),
               binding_sha256=None, nr3_prefix=True)
    pre['binding_sha256'] = hashlib.sha256(json.dumps(pre['binding'], sort_keys=True, default=str).encode()).hexdigest()
    (od/'prefix').mkdir(exist_ok=True); torch.save(pre, od/'prefix'/'PREFIX_step_008000.pt')
    recB = dict(parent_unit=par.name, parent_checkpoint=str(ckp), parent_checkpoint_sha256=sha256(ckp), parent_binding_sha256=ck['binding_sha256'],
                fit=fb.json(), start=[1., 0.], bounds=BOUNDS, maxiter=500,
                fold=dict(method='nr3.ref_init.fold_positive_affine (byte copy of package reference)', max_abs_err_vs_a_s_plus_b=err, max_abs_score=scale,
                          rel_err=err / max(scale, 1.), trainable_param_count_parent=sum(p.numel() for p in m.parameters()),
                          trainable_param_count_prefix=sum(p.numel() for p in mf.parameters())),
                CAL_A_J=dict(raw=pooled_j(*stP), calibrated=pooled_j(*stP, fb.a, fb.b)), READ_B_J=dict(raw=pooled_j(*stPB), calibrated=pooled_j(*stPB, fb.a, fb.b)),
                prefix_checkpoint=str(od/'prefix'/'PREFIX_step_008000.pt'), prefix_checkpoint_sha256=sha256(od/'prefix'/'PREFIX_step_008000.pt'),
                prefix_binding_sha256=pre['binding_sha256'])
    rec['B'] = recB
    write_json(od/'PREP_RECEIPT.json', rec); write_json(od/'PREP_DONE.json', dict(ok=True))
    print(json.dumps(dict(A=fa.json(), B=fb.json(), fold_err=err), default=float))


if __name__ == '__main__':
    main()
