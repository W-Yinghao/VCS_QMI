"""R35 summary (numbers only). Reads runs/<unit>/RESULT.json + eval_step*.json, the prep/gate records and
READOUTS6.json; writes RESULTS_STEP_FIXED.csv, CURVES.csv, CALIBRATION_RAW_VS_AFFINE.csv, RANK_TRANSITIONS.csv,
CANDIDATE_COVERAGE.csv, COST_LEDGER.csv, FAILURES.csv and RESULTS.md. Missing units are listed, never filled."""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, PREP, read_json, utc

R6 = ['i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10']


def wcsv(name, rows):
    if not rows:
        (ROOT / name).write_text(''); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(ROOT / name, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def jrow(ev, block):
    if block in ('A', 'B'):
        c = ev['calibration_affine_fit_only']
        return dict(J_dev_clean_raw=ev['dev_clean']['J_of_peer_mean_score'], J_dev_observed_raw=ev['dev_observed']['J_of_peer_mean_score'],
                    J_dev_clean_mean_of_peers=ev['dev_clean']['J_mean_of_peers'], J_fit_observed_raw=ev['fit_observed_J_raw'],
                    J_dev_clean_affine_logistic=c['balanced_logistic']['dev_clean_J'], J_dev_clean_affine_maxJ=c['max_J']['dev_clean_J'])
    c = ev['calibration_affine_fit_only']
    return dict(J_dev_clean_raw=ev['dev_list_clean_J_raw']['J'], J_fit_observed_raw=ev['fit_list_observed_J_raw']['J'],
                J_dev_clean_affine_logistic=c['balanced_logistic']['dev_list_clean_J'], J_dev_clean_affine_maxJ=c['max_J']['dev_list_clean_J'])


def main():
    units = read_json(ROOT / 'configs' / 'UNITS.json')
    fixed, curves, calib, trans, cover, cost, fails, missing = [], [], [], [], [], [], [], []
    for u in units:
        cfg = read_json(ROOT / 'configs' / f'{u}.json'); d = RUNS / u
        for f in sorted(glob.glob(str(d / 'FAILED_*.json'))):
            fr = read_json(f); fails.append(dict(unit=u, file=Path(f).name, utc=fr['utc'], last_line=fr['traceback'].strip().splitlines()[-1]))
        base = dict(unit=u, block=cfg['block'], cond=cfg['cond'], structure=cfg['structure_package'], objective=cfg['objective'], supervision=cfg['supervision'])
        for k in sorted(int(Path(f).stem.split('step')[1]) for f in glob.glob(str(d / 'eval_step*.json'))):
            ev = read_json(d / f'eval_step{k}.json')
            tw = ev.get('train_window', {})
            curves.append(dict(base, step=k, **{m: ev['dev_retrieval'][m] for m in R6}, rsum=ev['dev_retrieval']['rsum'], **jrow(ev, cfg['block']),
                               train_loss_window=tw.get('loss_mean'), train_batch_J_window=tw.get('batch_J_mean'), grad_norm_window=tw.get('grad_norm_mean')))
        if not (d / 'RESULT.json').exists():
            missing.append(u); continue
        R = read_json(d / 'RESULT.json'); ev = R['final']; tb = ev['dev_retrieval_tie_bounded']
        row = dict(base, step=R['step'], **{m: ev['dev_retrieval'][m] for m in R6}, rsum=ev['dev_retrieval']['rsum'],
                   rsum_tie_optimistic=tb['rsum_optimistic'], rsum_tie_pessimistic=tb['rsum_pessimistic'],
                   native_H_rsum=R['native_H_dev']['rsum'], delta_rsum_vs_H=R['diff_vs_native']['rsum'], **jrow(ev, cfg['block']),
                   gpu=R['env']['gpu'])
        if 'candidate_coverage' in ev:
            row.update(ev['candidate_coverage'])
        fixed.append(row)
        for crit, c in ev['calibration_affine_fit_only'].items():
            calib.append(dict(base, step=R['step'], criterion=crit, a=c['a'], b=c['b'], rank_invariant=c['dev_rank_invariant'],
                              J_dev_clean_raw=jrow(ev, cfg['block'])['J_dev_clean_raw'],
                              J_dev_clean_affine=c.get('dev_clean_J', c.get('dev_list_clean_J')),
                              J_fit_eval_raw=jrow(ev, cfg['block'])['J_fit_observed_raw'],
                              J_fit_eval_affine=c.get('fit_observed_J_eval', c.get('fit_list_observed_J_eval')),
                              J_dev_observed_raw=ev.get('dev_observed', {}).get('J_of_peer_mean_score'), J_dev_observed_affine=c.get('dev_observed_J')))
        t = ev['transitions_vs_H']
        for dr in ('i2t', 't2i'):
            trans.append(dict(base, direction=dr, **{k[len(dr) + 1:]: v for k, v in t.items() if k.startswith(dr + '_')},
                              **{k: v for k, v in ev.get('pair_win_rates', {}).items() if k.startswith(dr)}))
        if 'candidate_coverage' in ev:
            cover.append(dict(base, **ev['candidate_coverage'], rsum=ev['dev_retrieval']['rsum']))
        cs = R['cost']
        cost.append(dict(base, gpu=R['env']['gpu'], host=R['env']['host'], job=R['env']['job'], params_trainable=cs['params_trainable'],
                         updates=R['step'], train_seconds=cs['train_seconds'], eval_seconds=cs['eval_seconds'],
                         train_seconds_per_update=cs['train_seconds'] / max(R['step'], 1), pairs_per_update_per_peer=cs['pairs_per_update_per_peer'],
                         peak_gpu_gb=cs['peak_gpu_memory_bytes'] / 2 ** 30, n_attempt_failures=sum(1 for x in fails if x['unit'] == u)))
    # §6 readouts and prep-level coverage
    if (ROOT / 'READOUTS6.json').exists():
        r6 = read_json(ROOT / 'READOUTS6.json')
        for u, rec in r6['units'].items():
            for crit in ('balanced_logistic', 'max_J'):
                calib.append(dict(unit=u, block='R34', cond=u.split('_')[1], structure=u.split('_s1001_')[1], criterion=crit, step=2000,
                                  a=rec[crit]['a'], b=rec[crit]['b'], rank_invariant=rec[crit]['dev_rank_invariant'],
                                  J_dev_clean_raw=rec['raw']['dev_clean_J'], J_dev_clean_affine=rec[crit]['dev_clean_J'],
                                  J_fit_eval_raw=rec['raw']['fit_eval_J'], J_fit_eval_affine=rec[crit]['fit_eval_J'],
                                  J_dev_observed_raw=rec['raw']['dev_observed_J'], J_dev_observed_affine=rec[crit]['dev_observed_J']))
        for u, rec in r6['error_location'].items():
            t = rec['transitions']
            for dr in ('i2t', 't2i'):
                trans.append(dict(unit=u, block='R34', cond=u.split('_')[1], direction=dr, **{k[len(dr) + 1:]: v for k, v in t.items() if k.startswith(dr + '_')},
                                  **{k: v for k, v in rec['pair_win_rates'].items() if k.startswith(dr)}))
    pm = read_json(PREP / 'PREP_MANIFEST.json') if (PREP / 'PREP_MANIFEST.json').exists() else {}
    for cond, c in pm.get('candidates', {}).items():
        cover.append(dict(unit=f'FIT_candidate_tables_{cond}', cond=cond, **{k: v for k, v in c.items() if k.startswith(('i2t', 't2i'))}))
    for name, path in (('prep', PREP / 'PREP_MANIFEST.json'), ('gate', ROOT / 'GPU_GATE.json')):
        if path.exists():
            j = read_json(path); cost.append(dict(unit=name, gpu=j.get('env', {}).get('gpu', j.get('gpu')), job=j.get('env', {}).get('job', j.get('job')),
                                                  train_seconds=j.get('elapsed_seconds')))
    for name, rows in (('RESULTS_STEP_FIXED.csv', fixed), ('CURVES.csv', curves), ('CALIBRATION_RAW_VS_AFFINE.csv', calib),
                       ('RANK_TRANSITIONS.csv', trans), ('CANDIDATE_COVERAGE.csv', cover), ('COST_LEDGER.csv', cost), ('FAILURES.csv', fails)):
        wcsv(name, rows)
    # RESULTS.md
    L = [f'# R35 EXPANDED results (numbers only)', f'generated {utc()}; {len(fixed)}/42 units complete; fixed final step (A/B/C 8000, D 4000); full DEV 1014 img / 5070 cap; TEST not read.', '']
    if missing:
        L += [f'Missing (not complete): {", ".join(missing)}', '']
    L += ['## Fixed-step DEV retrieval (A/B: direct full-DEV ranking by peer-mean g; C/D: frozen-H top-100 reranked by g, rest in H order; upstream IN2R i2t/t2i)', '',
          '| unit | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δ vs H | tie-opt | tie-pess | J DEV-clean raw | J DEV-clean affine(logistic) | GPU |', '|' + '---|' * 14]
    for r in fixed:
        L.append('| ' + ' | '.join([r['unit']] + [f"{r[m]:.2f}" for m in R6] + [f"{r['rsum']:.2f}", f"{r['delta_rsum_vs_H']:.2f}", f"{r['rsum_tie_optimistic']:.2f}",
                                    f"{r['rsum_tie_pessimistic']:.2f}", f"{r['J_dev_clean_raw']:.4f}", f"{r['J_dev_clean_affine_logistic']:.4f}", r['gpu']]) + ' |')
    nat = {r['cond']: r['native_H_rsum'] for r in fixed}
    L += ['', 'Frozen H native DEV rSum: ' + ', '.join(f'{c} {v:.2f}' for c, v in sorted(nat.items())), '',
          'J readouts: A/B record U-statistic on the full DEV matrix (peer-mean score); C/D candidate-list J on DEV lists (clean annotation, 32 slots). The two objects differ and are not comparable.', '']
    if cover:
        L += ['## Candidate coverage (DEV, frozen-H top-100)', '', '| unit | cov i2t | cov t2i | cond R@1 i2t | cond R@1 t2i |', '|---|---|---|---|---|']
        for r in cover:
            if 'coverage_i2t' in r:
                L.append(f"| {r['unit']} | {r['coverage_i2t']:.4f} | {r['coverage_t2i']:.4f} | {r['cond_r1_i2t_covered']:.2f} | {r['cond_r1_t2i_covered']:.2f} |")
    L += ['', '## Cost (formal logs)', '', '| unit | GPU | updates | train s | eval s | s/update | peak GB |', '|---|---|---|---|---|---|---|']
    for r in cost:
        if 'updates' in r:
            L.append(f"| {r['unit']} | {r['gpu']} | {r['updates']} | {r['train_seconds']:.0f} | {r['eval_seconds']:.0f} | {r['train_seconds_per_update']:.4f} | {r['peak_gpu_gb']:.1f} |")
    L += ['', f'Failures / retries: {len(fails)} (FAILURES.csv). Other tables: CURVES.csv, CALIBRATION_RAW_VS_AFFINE.csv (incl. R34 §6.1), RANK_TRANSITIONS.csv (incl. R34 §6.3), CANDIDATE_COVERAGE.csv, COST_LEDGER.csv.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(L) + '\n')
    print('R35_SUMMARY', len(fixed), 'complete;', len(missing), 'missing;', len(fails), 'failure records')


if __name__ == '__main__':
    main()
