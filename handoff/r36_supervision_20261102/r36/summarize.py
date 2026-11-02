"""R36 summary (numbers only). Reads runs/<unit>/RESULT.json (oracle units: oracle_runs/<unit>), eval_step*.json, the
gate/CPU records and the evaluator-only audit tables already in ROOT. Writes RESULTS_FIXED.csv, CURVES.csv,
RANK_TRANSITIONS.csv, NULL_READOUTS.csv, COST_LEDGER.csv, FAILURES.csv, CPU_GPU_GATES.json, PARENT_BINDING.json and
RESULTS.md. R35 reference rows (frozen H, C-G/C-L, D HEAD_CONTROL) are read from R35 RESULT.json files read-only.
Missing units are listed, never filled. Usage: python -B summarize.py [--binding-only]"""
from __future__ import annotations
import csv, glob, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, ORACLE_RUNS, PREP, R35_ROOT, R35_OUT, R34_OUT, R30_RUNS, CONDS, sha256, read_json, write_json, utc

R6 = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']
R35_REF = ['r35_C_n60_s1001_G_J', 'r35_C_n60_s1001_G_BCE', 'r35_C_n60_s1001_G_NCE', 'r35_C_n60_s1001_L_J', 'r35_C_n60_s1001_L_BCE', 'r35_C_n60_s1001_L_NCE',
           'r35_B_n60_s1001_TRUST_J', 'r35_B_n60_s1001_TRUST_BCE', 'r35_B_n60_s1001_TRUST_NCE',
           'r35_C_n00_s1001_L_J', 'r35_C_n00_s1001_L_BCE', 'r35_C_n00_s1001_L_NCE',
           'r35_D_n00_s1001_L_J_HEAD_CONTROL', 'r35_D_n00_s1001_L_BCE_HEAD_CONTROL', 'r35_D_n00_s1001_L_NCE_HEAD_CONTROL']


def wcsv(name, rows):
    if not rows:
        (ROOT / name).write_text(''); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def jcols(ev):
    c = ev['calibration_affine_fit_only']
    out = dict(J_dev_list_clean_raw=ev['dev_list_clean_J_raw']['J'], J_fit_forced_list_observed_raw=ev['fit_list_observed_J_raw']['J'],
               J_dev_list_clean_affine_logistic=c['balanced_logistic']['dev_list_clean_J'], J_dev_list_clean_affine_maxJ=c['max_J']['dev_list_clean_J'])
    if 'fit_null_lists_observed_J_raw' in ev:
        out['J_fit_null_list_observed_raw(33 slots, different object)'] = ev['fit_null_lists_observed_J_raw']['J_total_bidirectional']
    if 'fit_forced_lists_score_structure' in ev:
        out['fit_spearman_g_vs_native'] = ev['fit_forced_lists_score_structure']['spearman_g_vs_native_row_mean']
    return out


def binding():
    sb = read_json(R35_ROOT / 'STATE_BINDING.json')
    out = dict(created_utc=utc(), note='R36 reuses the R35-bound parents, R34 caches and R35 prep read-only; hashes recomputed here', conditions={})
    for cond, c in CONDS.items():
        mb = R30_RUNS / c['h_unit'] / 'model_best.pth.tar'; man = R34_OUT / 'cache' / cond / 'CACHE_MANIFEST.json'
        out['conditions'][cond] = dict(parent_unit=c['h_unit'], parent_seed=1001, model_best=str(mb), model_best_sha256=sha256(mb),
                                       r35_bound_sha256=sb['conditions'][cond]['parent']['model_best_sha256'],
                                       observed_map=c['noise_file'], observed_map_sha256=sha256(c['noise_file']), expected=c['noise_sha256'],
                                       r34_cache_manifest_sha256=sha256(man), r35_bound_cache_manifest_sha256=sb['conditions'][cond]['r34_cache']['manifest_sha256'],
                                       candidates=str(PREP / f'candidates_{cond}.npz'), candidates_sha256=sha256(PREP / f'candidates_{cond}.npz'),
                                       a0b0=sb['r35_prep']['calibration_init'][cond])
        o = out['conditions'][cond]; o['all_match'] = o['model_best_sha256'] == o['r35_bound_sha256'] and o['observed_map_sha256'] == o['expected'] and \
            o['r34_cache_manifest_sha256'] == o['r35_bound_cache_manifest_sha256'] and o['candidates_sha256'] == sb['r35_prep']['files'][f'candidates_{cond}.npz']['sha256']
    out['r35_prep_manifest_sha256'] = sha256(PREP / 'PREP_MANIFEST.json'); out['r35_bound_prep_manifest_sha256'] = sb['r35_prep']['manifest_sha256']
    out['trust_n60_sha256'] = sha256(PREP / 'trust_n60.npz'); out['trust_n60_r35'] = sb['r35_prep']['files']['trust_n60.npz']['sha256']
    out['heads'] = 'all R36 heads fresh (R35 build_heads, init_seed = head seed); V phase 2 loads only its own phase-1 step-8000 head'
    write_json(ROOT / 'PARENT_BINDING.json', out)


def main():
    binding()
    if '--binding-only' in sys.argv:
        print('PARENT_BINDING written'); return
    units = read_json(ROOT / 'configs' / 'UNITS.json')
    fixed, curves, trans, nulls, cost, fails, missing = [], [], [], [], [], [], []
    for u in units:
        cfg = read_json(ROOT / 'configs' / f'{u}.json'); d = (ORACLE_RUNS if cfg['oracle'] else RUNS) / u
        for f in sorted(glob.glob(str(d / 'FAILED_*.json'))):
            fr = read_json(f); fails.append(dict(unit=u, file=Path(f).name, utc=fr['utc'], last_line=fr['traceback'].strip().splitlines()[-1]))
        base = dict(unit=u, block=cfg['block'], cond=cfg['cond'], structure=cfg['structure'], objective=cfg['objective'], supervision=cfg['supervision'],
                    oracle=cfg['oracle'], head_seed=cfg['init_seed'])
        pdirs = [(1, d / 'phase1'), (2, d / 'phase2')] if cfg['block'] == 'V' else [(1, d)]
        for ph, pd in pdirs:
            off = cfg['steps'] if ph == 2 else 0
            for k in sorted(int(Path(f).stem.split('step')[1]) for f in glob.glob(str(pd / 'eval_step*.json'))):
                ev = read_json(pd / f'eval_step{k}.json'); tw = ev.get('train_window', {})
                curves.append(dict(base, phase=ph, step=k, total_step=off + k, **{m: ev['dev_retrieval'][m] for m in R6}, rsum=ev['dev_retrieval']['rsum'], **jcols(ev),
                                   train_loss_window=tw.get('loss_mean'), train_batch_J_window=tw.get('batch_J_mean'), grad_norm_window=tw.get('grad_norm_mean'),
                                   **{f'win_{a}': b for a, b in (tw.get('slot_grad_means') or {}).items()}))
                if 'fit_null_lists_observed_J_raw' in ev:
                    nr = ev['fit_null_lists_observed_J_raw']
                    for dr in ('i2t', 't2i'):
                        nulls.append(dict(base, step=k, direction=dr, **nr[dr], null_param_A=nr['null_params']['A'][0 if dr == 'i2t' else 1],
                                          null_param_B=nr['null_params']['B'][0 if dr == 'i2t' else 1],
                                          **{a: b for a, b in (tw.get('slot_grad_means') or {}).items() if dr in a}))
        if not (d / 'RESULT.json').exists():
            missing.append(u); continue
        R = read_json(d / 'RESULT.json'); ev = R['final']; tb = ev['dev_retrieval_tie_bounded']
        row = dict(base, endpoint=R['primary_endpoint'], total_steps=R['total_steps'], **{m: ev['dev_retrieval'][m] for m in R6}, rsum=ev['dev_retrieval']['rsum'],
                   rsum_tie_optimistic=tb['rsum_optimistic'], rsum_tie_pessimistic=tb['rsum_pessimistic'], native_H_rsum=R['native_H_dev']['rsum'],
                   delta_rsum_vs_H=R['diff_vs_native']['rsum'], **jcols(ev), **ev['candidate_coverage'], gpu=R['env']['gpu'])
        if cfg['block'] == 'V':
            p1 = R['phases'][0]['final']; row['phase1_step8000_rsum'] = p1['dev_retrieval']['rsum']
            row['phase2_step0_equals_phase1_final'] = R.get('phase2_step0_equals_phase1_final_scores', {}).get('bitwise')
        fixed.append(row)
        t = ev['transitions_vs_H']
        for dr in ('i2t', 't2i'):
            trans.append(dict(base, direction=dr, **{k[len(dr) + 1:]: v for k, v in t.items() if k.startswith(dr + '_')},
                              **{k: v for k, v in ev.get('pair_win_rates', {}).items() if k.startswith(dr)}))
        cs = R['cost']
        cost.append(dict(base, gpu=R['env']['gpu'], host=R['env']['host'], job=R['env']['job'], params_trainable=cs['params_trainable'], updates=R['total_steps'],
                         train_seconds=cs['train_seconds'], eval_seconds=cs['eval_seconds'], train_seconds_per_update=cs['train_seconds'] / max(R['total_steps'], 1),
                         pairs_per_update_per_peer=cs['pairs_per_update_per_peer'], peak_gpu_gb=cs['peak_gpu_memory_bytes'] / 2 ** 30,
                         n_attempt_failures=sum(1 for x in fails if x['unit'] == u)))
    ref = []
    for u in R35_REF:
        f = R35_OUT / 'runs' / u / 'RESULT.json'
        if f.exists():
            R = read_json(f); ev = R['final']
            ref.append(dict(unit=u, step=R['step'], **{m: ev['dev_retrieval'][m] for m in R6}, rsum=ev['dev_retrieval']['rsum'], delta_rsum_vs_H=R['diff_vs_native']['rsum'],
                            rescued_r1_i2t=ev['transitions_vs_H']['i2t_rescued_r1'], lost_r1_i2t=ev['transitions_vs_H']['i2t_lost_r1'],
                            rescued_r1_t2i=ev['transitions_vs_H']['t2i_rescued_r1'], lost_r1_t2i=ev['transitions_vs_H']['t2i_lost_r1']))
    for name, path in (('cpu_test', ROOT / 'CPU_TEST.json'), ('gpu_gate', ROOT / 'GPU_GATE.json'), ('audit', ROOT / 'AUDIT_RESULTS.json')):
        if path.exists():
            j = read_json(path); cost.append(dict(unit=name, gpu=j.get('gpu'), job=j.get('job', j.get('export_job')), train_seconds=j.get('elapsed_seconds')))
    for name, rows in (('RESULTS_FIXED.csv', fixed), ('CURVES.csv', curves), ('RANK_TRANSITIONS.csv', trans), ('NULL_READOUTS.csv', nulls),
                       ('COST_LEDGER.csv', cost), ('FAILURES.csv', fails), ('R35_REFERENCE_ROWS.csv', ref)):
        wcsv(name, rows)
    g = read_json(ROOT / 'GPU_GATE.json') if (ROOT / 'GPU_GATE.json').exists() else {}
    c = read_json(ROOT / 'CPU_TEST.json') if (ROOT / 'CPU_TEST.json').exists() else {}
    write_json(ROOT / 'CPU_GPU_GATES.json', dict(cpu=dict(ALL_PASS=c.get('ALL_PASS'), job=c.get('job'), checks={k: v['pass_'] for k, v in c.get('checks', {}).items()}),
                                                gpu=dict(ALL_PASS=g.get('ALL_PASS'), job=g.get('job'), gpu=g.get('gpu'), checks={k: v['pass_'] for k, v in g.get('checks', {}).items()})))
    # RESULTS.md
    L = ['# R36 supervision / candidate-selection results (numbers only)',
         f'generated {utc()}; {len(fixed)}/27 units complete; fixed endpoints (M/T/N step 8000; V phase-2 step 4000 = total 12000, phase-1 8000 listed); full DEV 1014 img / 5070 cap; '
         'two-stage readout (frozen-H top-100 reranked by g, rest in H order); TEST not read.', '']
    if missing:
        L += [f'Missing (not complete): {", ".join(missing)}', '']
    L += ['## Fixed-endpoint DEV retrieval', '', '| unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δ vs H | tie-opt | tie-pess | rescued/lost R@1 i2t | t2i | GPU |', '|' + '---|' * 14]
    tmap = {(t['unit'], t['direction']): t for t in trans}
    for r in fixed:
        ti, tt = tmap[(r['unit'], 'i2t')], tmap[(r['unit'], 't2i')]
        L.append('| ' + ' | '.join([r['unit']] + [f"{r[m]:.2f}" for m in R6] + [f"{r['rsum']:.2f}", f"{r['delta_rsum_vs_H']:.2f}", f"{r['rsum_tie_optimistic']:.2f}",
                                    f"{r['rsum_tie_pessimistic']:.2f}", f"{ti['rescued_r1']}/{ti['lost_r1']}", f"{tt['rescued_r1']}/{tt['lost_r1']}", r['gpu']]) + ' |')
    v = [r for r in fixed if r['block'] == 'V']
    if v:
        L += ['', '## V: phase-1 step 8000 (secondary) vs phase-2 step 4000 (primary, total 12000)', '', '| unit | rSum @8000 | rSum @12000 | phase-2 step0 == phase-1 final |', '|---|---|---|---|']
        L += [f"| {r['unit']} | {r['phase1_step8000_rsum']:.2f} | {r['rsum']:.2f} | {r['phase2_step0_equals_phase1_final']} |" for r in v]
    nat = {r['cond']: r['native_H_rsum'] for r in fixed}
    L += ['', 'Frozen H native DEV rSum: ' + ', '.join(f'{k} {x:.2f}' for k, x in sorted(nat.items())), '',
          '## R35 reference rows (read-only, seed1001, fixed endpoints)', '', '| unit | step | rSum | Δ vs H | rescued/lost R@1 i2t | t2i |', '|---|---|---|---|---|---|']
    L += [f"| {r['unit']} | {r['step']} | {r['rsum']:.2f} | {r['delta_rsum_vs_H']:.2f} | {r['rescued_r1_i2t']}/{r['lost_r1_i2t']} | {r['rescued_r1_t2i']}/{r['lost_r1_t2i']} |" for r in ref]
    L += ['', 'J columns (RESULTS_FIXED.csv): DEV list J (clean annotation, 32 slots), FIT forced-list J (observed), FIT null-list J (N only, 33 slots, Q=Uniform(33)) are different statistical objects and are not ranked against each other.',
          '', '## Cost (formal logs)', '', '| unit | GPU | updates | train s | eval s | s/update | peak GB |', '|---|---|---|---|---|---|---|']
    for r in cost:
        if 'updates' in r:
            L.append(f"| {r['unit']} | {r['gpu']} | {r['updates']} | {r['train_seconds']:.0f} | {r['eval_seconds']:.0f} | {r['train_seconds_per_update']:.4f} | {r['peak_gpu_gb']:.1f} |")
    L += ['', f'Failures / retries: {len(fails)} (FAILURES.csv). Other tables: CURVES.csv, RANK_TRANSITIONS.csv, NULL_READOUTS.csv, COST_LEDGER.csv, R35_REFERENCE_ROWS.csv; '
          'evaluator-only audit: AUDIT_RESULTS.json, PROVENANCE_AUDIT.csv, CANDIDATE_CONFUSION.csv, SCORE_GRAD_AUDIT.csv.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('R36_SUMMARY', len(fixed), 'complete;', len(missing), 'missing;', len(fails), 'failure records')


if __name__ == '__main__':
    main()
