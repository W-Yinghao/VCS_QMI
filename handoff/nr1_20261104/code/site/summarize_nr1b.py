#!/usr/bin/env python3
"""NR1B summarizer: numbers only. Writes RESULTS_NR1B.md + CSVs in handoff/nr1_20261104/nr1b."""
import csv, json
from pathlib import Path
import numpy as np
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104'); O = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'); B = H/'nr1b'
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
K = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']
base = rj(O/'baseline/float64/METRICS.json')['metrics']['rSum']
def unit_dir(c, a, o, s): return (O/'runs'/f'NR1_{c}_{a}_{o}_s1001') if s == 1001 else (O/'runs_nr1b'/f'NR1B_{c}_{a}_{o}_s{s}')
def wcsv(p, rs):
    keys = list(dict.fromkeys(k for r in rs for k in r))
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rs)
L = ['# NR1B_20261104 results (numbers only; DEV; TEST not read)', '', f'Base rSum {base:.2f}. Protocol: nr1b/PROTOCOL_FROZEN_NR1B.json.', '']
# (1) seeds
seed_rows = []; L += ['## (1) Flickr00 training-seed replication, step 8000 rSum', '', '| recipe | s1001 | s1002 | s1003 | mean | sd | mean ΔrSum | status |', '|---|---|---|---|---|---|---|---|']
seeds = [1001, 1002, 1003]; per = {}
for a in ['G', 'GL']:
    for o in ['J', 'BCE', 'NCE']:
        v = []
        for s in seeds:
            m = rj(unit_dir('Flickr00', a, o, s)/'eval_008000/METRICS.json')
            v.append(m['metrics']['rSum'] if m else None)
            if m: seed_rows.append(dict(condition='Flickr00', architecture=a, objective=o, seed=s, **{k: round(m['metrics'][k], 2) for k in K}, delta_rSum=round(m['delta_rSum'], 2),
                                        step0_order_equals_base=rj(unit_dir('Flickr00', a, o, s)/'eval_000000/METRICS.json')['order_vs_base']))
        per[(a, o)] = v; ok = [x for x in v if x is not None]
        st = 'complete' if len(ok) == 3 else f'{len(ok)}/3'
        L.append(f"| {a} {o} | " + ' | '.join(f'{x:.2f}' if x is not None else 'NA' for x in v) +
                 (f" | {np.mean(ok):.2f} | {np.std(ok, ddof=1):.2f} | {np.mean(ok)-base:+.2f} | {st} |" if len(ok) > 1 else f" | NA | NA | NA | {st} |"))
L += ['', 'Paired differences per seed (same architecture):', '', '| arch | pair | s1001 | s1002 | s1003 | mean |', '|---|---|---|---|---|---|']
for a in ['G', 'GL']:
    for x, y in [('J', 'BCE'), ('J', 'NCE'), ('BCE', 'NCE')]:
        d = [(p - q) if p is not None and q is not None else None for p, q in zip(per[(a, x)], per[(a, y)])]
        ok = [z for z in d if z is not None]
        L.append(f"| {a} | {x}−{y} | " + ' | '.join(f'{z:+.2f}' if z is not None else 'NA' for z in d) + f" | {np.mean(ok):+.2f} |" if ok else '')
for o in ['J', 'BCE', 'NCE']:
    d = [(p - q) if p is not None and q is not None else None for p, q in zip(per[('GL', o)], per[('G', o)])]; ok = [z for z in d if z is not None]
    L.append(f"| GL−G | {o} | " + ' | '.join(f'{z:+.2f}' if z is not None else 'NA' for z in d) + f" | {np.mean(ok):+.2f} |")
wcsv(B/'RESULTS_NR1B_SEEDS.csv', seed_rows)
# read-only results
ro = {p.stem: rj(p) for p in sorted((O/'nr1b/readonly').glob('*.json'))}
order = [f'NR1_{c}_{a}_{o}_s1001' for c in ['Flickr00', 'Flickr60', 'Flickr80'] for a in ['G', 'GL'] for o in ['J', 'BCE', 'NCE']] + \
        [f'NR1B_Flickr00_{a}_{o}_s{s}' for a in ['G', 'GL'] for o in ['J', 'BCE', 'NCE'] for s in (1002, 1003)]
units = [u for u in order if u in ro]
# (2) ablation
L += ['', '## (2) GL local ablation (local_readout zeroed, read-only), step 8000', '',
      '| unit | full GL rSum | ablated rSum | G rSum (same obj) | readout w | i2t full-R1/abl-not | i2t abl-R1/full-not | t2i full-R1/abl-not | t2i abl-R1/full-not | t2i GL-R1/G-not | t2i G-R1/GL-not |', '|' + '---|'*11]
abl_rows = []
for u in units:
    r = ro[u]; la = r.get('local_ablation')
    if not la: continue
    pq = la['per_query']; w = la['readout_weight'][0]
    L.append(f"| {u} | {la['full_GL_rSum']:.2f} | {la['metrics']['rSum']:.2f} | {la['G_rSum']:.2f} | [{w[0]:.3f}, {w[1]:.3f}] | {pq['i2t']['full_R1_ablated_not']} | {pq['i2t']['ablated_R1_full_not']} | {pq['t2i']['full_R1_ablated_not']} | {pq['t2i']['ablated_R1_full_not']} | {pq['t2i'].get('GL_R1_G_not','')} | {pq['t2i'].get('G_R1_GL_not','')} |")
    abl_rows.append(dict(unit_id=u, full_GL_rSum=la['full_GL_rSum'], ablated_rSum=la['metrics']['rSum'], G_rSum=la['G_rSum'], readout_t2i_evidence=w[0], readout_i2t_evidence=w[1],
                         **{f'ablated_{k}': round(v, 2) for k, v in la['metrics'].items()}, **{f'{d}_{k}': v for d in ('i2t', 't2i') for k, v in pq[d].items() if not isinstance(v, dict)}))
wcsv(B/'RESULTS_NR1B_ABLATION.csv', abl_rows)
# (3) DEV categories
L += ['', '## (3a) DEV instance categories at step 8000 (counts; i2t n=1014, t2i n=5070)', '',
      '| unit | i2t base✓head✗ | i2t base✗head✓ | i2t in-top100 still ✗ | i2t coverage fail | t2i base✓head✗ | t2i base✗head✓ | t2i in-top100 still ✗ | t2i coverage fail |', '|' + '---|'*9]
cat_rows = []
for u in units:
    c = ro[u]['dev_categories_step8000']
    L.append(f"| {u} | " + ' | '.join(str(c[d][k]) for d in ('i2t', 't2i') for k in ('base_correct_head_lost', 'base_wrong_head_rescued', 'in_top100_still_failed', 'coverage_failure')) + ' |')
    cat_rows.append(dict(unit_id=u, **{f'{d}_{k}': v for d in ('i2t', 't2i') for k, v in c[d].items()}))
wcsv(B/'RESULTS_NR1B_DEV_CATEGORIES.csv', cat_rows)
# (3b) FIT noise strata
L += ['', '## (3b) FIT post-hoc noise strata (evaluator-only truth; independent FIT sample B, 32768 observed records)', '',
      '| unit | step | n clean | n mismatched | mean g clean | mean g mismatched | AUC clean vs mismatched | AUC mismatched vs Q |', '|---|---|---|---|---|---|---|---|']
fit_rows = []
for u in units:
    for row in ro[u]['fit_read']:
        fit_rows.append(dict(unit_id=u, step=row['step'], raw_J_B=row['raw_J_B'], common_cal_a=row['common_cal']['a'], common_cal_b=row['common_cal']['b'],
                             common_cal_J_B=row['common_cal_J_B'], PQ_auc=row['PQ_auc'], P_sat=row['P_sat'], Q_sat=row['Q_sat'],
                             **{f'P_q{k}': v for k, v in row['P_quantiles'].items()}, **{f'Q_q{k}': v for k, v in row['Q_quantiles'].items()},
                             **({f'strata_{k}': v for k, v in row['fit_noise_strata'].items()} if 'fit_noise_strata' in row else {})))
        if 'fit_noise_strata' in row and row['step'] in (0, 8000):
            s = row['fit_noise_strata']
            L.append(f"| {u} | {row['step']} | {s['n_clean']} | {s['n_mismatched']} | {s['mean_g_clean']:.3f} | {s['mean_g_mismatched']:.3f} | {s['auc_clean_vs_mismatched']:.4f} | {s['auc_mismatched_vs_Q']:.4f} |")
wcsv(B/'RESULTS_NR1B_FIT_READ.csv', fit_rows)
# (4) training signal
L += ['', '## (4) Training signal on independent FIT sample B (step 0 → 8000)', '',
      '| unit | raw J s0 | raw J s8000 | common-cal J s0 | common-cal J s8000 | cal a,b s8000 | P/Q AUC s0 | P/Q AUC s8000 | P sat s8000 | Q sat s8000 | grad-norm mean | frac logged >clip |', '|' + '---|'*12]
tr_rows = []
for u in units:
    f = {r['step']: r for r in ro[u]['fit_read']}; g = ro[u]['grad_norm_summary']
    L.append(f"| {u} | {f[0]['raw_J_B']:.4f} | {f[8000]['raw_J_B']:.4f} | {f[0]['common_cal_J_B']:.4f} | {f[8000]['common_cal_J_B']:.4f} | {f[8000]['common_cal']['a']:.2f},{f[8000]['common_cal']['b']:.2f} | {f[0]['PQ_auc']:.4f} | {f[8000]['PQ_auc']:.4f} | {f[8000]['P_sat']:.3f} | {f[8000]['Q_sat']:.3f} | {g['mean']:.3f} | {g['frac_logged_steps_clipped']:.3f} |")
    for r in ro[u]['train_log']: tr_rows.append(dict(unit_id=u, **r))
wcsv(B/'RESULTS_NR1B_TRAIN_LOG.csv', tr_rows)
missing = [u for u in order if u not in ro]
L += ['', f'Read-only coverage: {len(units)}/{len(order)} units' + (f'; missing: {", ".join(missing)}' if missing else '')]
(B/'RESULTS_NR1B.md').write_text('\n'.join(x for x in L if x is not None) + '\n'); print('\n'.join(L[:20]))
