"""R40 section 7 summary (numbers only): diag7/READOUTS_<tag>.json (items 1, 2, 3, 5, 6), diag7/EXPOSURE_*.csv (item 4), diag7/TRUTH_f60_*.json
(isolated Flickr truth evaluator). Writes diag7/DIAG7_ITEM1.csv, DIAG7_ITEM2.csv, DIAG7_ITEM3.csv, DIAG7_TRUTH.csv and appends nothing elsewhere.
Usage: python3 -I -B tools/summarize_diag7.py"""
import csv, glob, json
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]; D = W / 'diag7'


def wcsv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(D / name, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


i1, i2, i3 = [], [], []
for f in sorted(D.glob('READOUTS_*.json')):
    d = json.loads(f.read_text()); tag = d['tag']
    for u, x in d['units'].items():
        a = x['item1']; b = x['item5']
        i1.append(dict(tag=tag, unit=u, objective=x['objective'], supervision=x['supervision'], H_native=d['native_H']['rsum'], final_g_two_stage=a['final_g_two_stage']['rsum'],
                       student_cos_two_stage=a['student_cosine_two_stage_H_top100']['rsum'], student_cos_full=a['student_cosine_full_gallery']['rsum'],
                       uncentred_student_cos_full=b['uncentred_student_cosine_full_gallery']['rsum'], uncentred_student_cos_two_stage=b['uncentred_student_cosine_two_stage']['rsum'],
                       reorder_chunk64_max_abs=b['reorder_chunk64_max_abs_vs_standard']))
        g = x['item2']['grad_norms_fixed_fit_batches']; fc = x['item2']['fit_feature_change']; pdist = x['item2']['param_distance']
        row = dict(tag=tag, unit=u, objective=x['objective'])
        for part in ('total', 'present_rows', 'missing_rows'):
            for grp in ('head', 'null', 'img', 'txt'):
                row[f'{part}_{grp}_grad'] = float(np.mean([g[p][f'{part}_{grp}'] for p in ('A', 'B')]))
        row['clip_multiplier_mean'] = float(np.mean([g[p]['clip_multiplier_mean'] for p in ('A', 'B')]))
        for k in ('img', 'cap'):
            for s in ('present', 'missing'):
                v = [fc[f'{k}_{p}'][s] for p in ('A', 'B') if fc[f'{k}_{p}'][s] is not None]
                row[f'fit_cos_{k}_{s}'] = float(np.mean(v)) if v else None
        for m in ('img', 'txt'):
            row[f'param_rel_l2_{m}'] = float(np.mean([pdist[f'{m}_{p}']['rel_l2'] for p in ('A', 'B')]))
        i2.append(row)
    for s in d.get('item3', []):
        for o, r in s['per_objective'].items():
            row = dict(tag=tag, state=s['state'], objective=o, J_BCE_identity_max_abs=s['J_BCE_identity_max_abs'])
            for kind in ('dLdg', 'dg'):
                for cl, v in r[kind].items():
                    row[f'{kind}_{cl}_mean'] = v['mean']; row[f'{kind}_{cl}_frac_down'] = v['frac_pushed_down']
            row.update({f'gradcos_{k}': v for k, v in s['param_grad_cosine'].items()})
            i3.append(row)
wcsv('DIAG7_ITEM1.csv', i1); wcsv('DIAG7_ITEM2.csv', i2); wcsv('DIAG7_ITEM3.csv', i3)
tr = []
for f in sorted(D.glob('TRUTH_*.json')):
    d = json.loads(f.read_text())
    for st, r in d['item3_slot_effects_by_truth'].items():
        for o, cls in r.items():
            for c, v in cls.items():
                tr.append(dict(tag=d['tag'], state=st, objective=o, slot_class=c, **v))
    for u, r in d['item2_caption_feature_change_by_truth'].items():
        tr.append(dict(tag=d['tag'], state=u, objective='-', slot_class='caption_feature_cos', observed_correct=np.mean([r[p]['caption_cos_observed_correct'] for p in ('A', 'B')]),
                       observed_wrong=np.mean([r[p]['caption_cos_observed_wrong'] for p in ('A', 'B')])))
wcsv('DIAG7_TRUTH.csv', tr)
print('diag7 rows', len(i1), len(i2), len(i3), len(tr))
