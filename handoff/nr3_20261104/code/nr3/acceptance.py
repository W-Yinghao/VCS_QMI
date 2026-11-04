#!/usr/bin/env python3
"""NR3 real-data acceptance for one condition (GPU; FIT only except the separate evaluator subprocess). Fixed tolerances.
 1 calibration/fold receipts sane: A and B fits finite, a>0, BCE not increased; fold error tiny; no extra parameters
 2 prefix DEV order check (label-free) recorded; parent checkpoint bytes unchanged
 3 A smoke (NR2 trainer unchanged + HCAL calibration): 20 steps, step-0 contract, resume bitwise; evaluator step-0 order == base
 4 B smoke (nr3.train_prefix): J/BCE/NCE 20 steps each; stage-0 checkpoint == prefix state bitwise for all three; identical sampler
   state across the three branches; J resume bitwise
 5 non-zero-head chunk invariance on the folded prefix (float64): one-shot vs 97-pair chunks
 6 static: prep/pools/d0 never open DEV cache, truth or clean observed paths"""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch
from nr1.io import FeatureBank, load_observed, read_json, sha256, write_json
from nr1.models import DirectScorer
from nr2 import train as nr2_train
from nr2.reference import Marginals, ReferenceSampler
from nr3 import train_prefix
from nr3.paths import OUT, cond_paths

N3 = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104')
results, fails = {}, []


def rec(name, ok, **kw):
    results[name] = dict(ok=bool(ok), **kw); print(('PASS ' if ok else 'FAIL ') + name, json.dumps(kw, default=float)[:500], flush=True)
    if not ok: fails.append(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); c = ap.parse_args().cond
    P = cond_paths(c); od = OUT/'prep'/c; SM = OUT/'acceptance'/c
    if SM.exists(): shutil.rmtree(SM)
    SM.mkdir(parents=True)
    R = read_json(od/'PREP_RECEIPT.json'); A, B = R['A'], R['B']
    okA = A['fit']['a'] > 0 and np.isfinite(A['fit']['final_bce']) and A['fit']['final_bce'] <= A['fit']['initial_bce'] + 1e-8
    okB = B['fit']['a'] > 0 and np.isfinite(B['fit']['final_bce']) and B['fit']['final_bce'] <= B['fit']['initial_bce'] + 1e-8
    rec('calibration_receipts', okA and okB and B['fold']['rel_err'] <= 1e-5 and B['fold']['trainable_param_count_prefix'] == B['fold']['trainable_param_count_parent'],
        A=A['fit'], B=B['fit'], fold=B['fold'])
    oc = read_json(od/'PREFIX_ORDER_CHECK.json')
    rec('prefix_dev_order_label_free', oc['max_rel_score_err'] <= 1e-5, **oc)
    rec('parent_checkpoint_unchanged', sha256(B['parent_checkpoint']) == B['parent_checkpoint_sha256'] and sha256(B['prefix_checkpoint']) == B['prefix_checkpoint_sha256'])
    args = (P['fit'], P['observed'])
    # A smoke
    cA = read_json(N3/'configs_production'/f'NR3_A_HCAL_{c}_G_J_s1001.json'); cA = dict(cA, steps=20, checkpoints=[0, 10, 20])
    a1, a2 = SM/'A_straight', SM/'A_resumed'
    nr2_train.run(cA, *args, od/'calibration_HCAL.json', P['tables'], a1)
    nr2_train.run(cA, *args, od/'calibration_HCAL.json', P['tables'], a2, max_steps=10); nr2_train.run(cA, *args, od/'calibration_HCAL.json', P['tables'], a2)
    ka, kb = torch.load(a1/'step_000020.pt', weights_only=False), torch.load(a2/'step_000020.pt', weights_only=False)
    s0 = read_json(a1/'STEP0_GRAD_CHECK.json')
    p = subprocess.run([sys.executable, '-B', '-m', 'nr1.evaluate', '--cache', str(P['dev']), '--truth', str(P['truth']), '--checkpoint', str(a1/'step_000000.pt'),
                        '--out', str(SM/'A_eval_step0'), '--device', 'cuda', '--score-dtype', 'float64'], capture_output=True, text=True)
    mt = read_json(SM/'A_eval_step0/METRICS.json') if (SM/'A_eval_step0/METRICS.json').exists() else None; bm = read_json(P['baseline'])
    rec('A_smoke_step0_resume_eval', all(torch.equal(ka['model'][k], kb['model'][k]) for k in ka['model']) and p.returncode == 0 and mt is not None
        and abs(mt['metrics']['rSum'] - bm['metrics']['rSum']) < 1e-9, step0_err=s0['step0_score_vs_affine_cosine_max_abs'], rc=p.returncode,
        rSum=mt['metrics']['rSum'] if mt else None, base=bm['metrics']['rSum'], stderr=p.stderr[-300:])
    # B smoke
    pre = torch.load(B['prefix_checkpoint'], map_location='cpu', weights_only=False); st0, ok, info = [], True, {}
    for o in ('J', 'BCE', 'NCE'):
        cB = read_json(N3/'configs_production'/f'NR3_B_NCE2{o}_{c}_G_s1001.json'); cB = dict(cB, new_updates=20, score_checks=[0, 10, 20])
        b1 = SM/f'B_{o}_straight'
        train_prefix.run(cB, *args, P['calibration'], B['prefix_checkpoint'], P['tables'], b1)
        k0 = torch.load(b1/'step_008000.pt', weights_only=False); st0.append(k0['rng_sampler'])
        ok &= all(torch.equal(k0['model'][k].cpu(), pre['model'][k]) for k in pre['model'])
        info[o] = read_json(b1/'STEP0_GRAD_CHECK.json')['step0_folded_vs_a_sparent_plus_b_max_tol_ratio']
        if o == 'J':
            b2 = SM/'B_J_resumed'
            train_prefix.run(cB, *args, P['calibration'], B['prefix_checkpoint'], P['tables'], b2, max_steps=10)
            train_prefix.run(cB, *args, P['calibration'], B['prefix_checkpoint'], P['tables'], b2)
            x1, x2 = torch.load(b1/'step_008020.pt', weights_only=False), torch.load(b2/'step_008020.pt', weights_only=False)
            ok &= all(torch.equal(x1['model'][k], x2['model'][k]) for k in x1['model'])
    same_sampler = all(json.dumps(s, default=str) == json.dumps(st0[0], default=str) for s in st0)
    rec('B_smoke_identical_start_sampler_resume', ok and same_sampler, step0_tol_ratio=info, same_sampler_state=same_sampler)
    # chunk invariance on folded prefix (non-zero head)
    bank = FeatureBank(P['fit'], 'fit', verify_hashes=False); im, tx = load_observed(P['observed'], bank)
    m = DirectScorer(**pre['binding']['model_config']).cuda().double(); m.load_state_dict(pre['model'])
    X = torch.from_numpy(np.array(bank.x[:4096], copy=True)).cuda().double(); Y = torch.from_numpy(np.array(bank.y[:4096], copy=True)).cuda().double()
    with torch.no_grad():
        whole = m.pairs(X, Y); parts = torch.cat([m.pairs(X[s:s+97], Y[s:s+97]) for s in range(0, 4096, 97)])
    rec('folded_nonzero_head_chunk_invariance', float((whole - parts).abs().max()) <= 1e-12 and float(m.global_residual[-1].weight.abs().max()) > 0,
        max_diff=float((whole - parts).abs().max()))
    srcs = {f: (N3/'code/nr3'/f).read_text() for f in ('prep.py', 'pools.py', 'd0.py')}
    bad = {f: [t for t in ("P['dev']", "P['truth']", 'clean_observed', 'evaluator_only') if t in s] for f, s in srcs.items()}
    rec('static_no_dev_truth_in_fitting_code', not any(bad.values()), found=bad)
    write_json(N3/'checks'/f'ACCEPTANCE_{c}.json', dict(all_pass=not fails, fails=fails, results=results, gpu=torch.cuda.get_device_name()))
    print('ALL_PASS' if not fails else f'FAIL {fails}'); sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
