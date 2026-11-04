"""Export (verbatim, no new computation) the CR2 audit numbers not yet in a public table:
support crosstabs and raw/calibrated candidate-rank seams. Reads only the existing CR2 audit JSONs."""
import csv, json, sys
from pathlib import Path
A = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
CONDS = ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104')
rows_x, rows_s, rows_e, src = [], [], [], {}
for c in CONDS:
    p = A/'audit'/f'AUDIT_LABELFREE_{c}.json'; d = json.loads(p.read_text()); src[str(p)] = __import__('hashlib').sha256(p.read_bytes()).hexdigest()
    for k, v in d['support_crosstab_label_free'].items(): rows_x.append(dict(condition=c, table='label_free', cell=k, value=v))
    for k, v in d['rows_with_k_experts'].items(): rows_x.append(dict(condition=c, table='rows_with_k_experts', cell=k, value=v))
    for s, v in d['scores'].items():
        r = dict(condition=c, source=s, score_sha256=v['score_sha256'])
        for k in ('rows_top1_changed_raw_vs_calibrated', 'rows_top1_changed_among_multi_expert_rows', 'mean_abs_rank_change', 'rows_any_rank_change',
                  'top1_equals_base_top1_calibrated', 'observed_is_calibrated_top1_H_rows'): r[k] = v[k]
        for k in ('observed_rank_in_C_raw_H_rows', 'observed_rank_in_C_calibrated_H_rows'):
            q = v[k]; r[k+'_q50'] = q[4]; r[k+'_q90'] = q[6];
        rows_s.append(r)
        for e, ev in v['per_expert_calibration'].items():
            cq = ev['calibrated_q']
            rows_e.append(dict(condition=c, source=s, expert=e, edges=ev['edges'], cal_a=ev['a'], cal_b=ev['b'], calibrated_q01=cq[1], calibrated_q50=cq[4],
                               calibrated_q99=cq[7], raw_min=ev['raw_q_0_50_100'][0], raw_median=ev['raw_q_0_50_100'][1], raw_max=ev['raw_q_0_50_100'][2]))
for c in ('Flickr00', 'Flickr60'):
    p = A/'evaluator_only'/f'AUDIT_TRUTH_{c}.json'; d = json.loads(p.read_text()); src[str(p)] = __import__('hashlib').sha256(p.read_bytes()).hexdigest()
    for k, v in d['crosstab'].items(): rows_x.append(dict(condition=c, table='evaluator_only_truth', cell=k, value=v))
    for k in ('H_rows_obs_correct', 'H0_rows_obs_correct', 'H0_rows_truth_in_C'): rows_x.append(dict(condition=c, table='evaluator_only_truth', cell=k, value=d[k]))
for name, rows in (('CR2_SUPPORT_CROSSTAB.csv', rows_x), ('CR2_RANK_SEAMS.csv', rows_s), ('CR2_EXPERT_SCORE_SCALES.csv', rows_e)):
    keys = []; [keys.append(k) for r in rows for k in r if k not in keys]
    with open(OUT/name, 'w', newline='') as f: w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
(OUT/'CR2_AUDIT_EXPORT_SOURCES.json').write_text(json.dumps({'sources_sha256': src, 'note': 'verbatim export of CR2 audit JSON fields; no new computation'}, indent=1))
print(len(rows_x), len(rows_s), len(rows_e))
