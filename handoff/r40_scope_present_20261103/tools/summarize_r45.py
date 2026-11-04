"""R45 summary (numbers only): COCO IN2R-native N-JOINT with a 16000-update budget vs the R43-A 8000-update N-JOINT and N-frozen
cells on the same parents (c60 s1001, c80 s1001). DEV = native 1K split, two-stage (frozen-H top-100 reranked). TEST not read.
Writes RESULTS_R45.md / R45_RESULTS.csv next to this tools/ dir. Usage: python3 -I -B tools/summarize_r45.py"""
import csv, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103'
PARENTS = {'C60_IN2R_H1001': 538.94, 'C80_IN2R_H1001': 526.52}
OBJ = ('J', 'BCE', 'NCE')


def rsum(d, step):
    f = os.path.join(d, f'eval_step{step}.json')
    if not os.path.exists(f):
        return None
    return json.load(open(f))['dev_retrieval']['rsum']


rows = []
for p, h in PARENTS.items():
    for o in OBJ:
        u16 = os.path.join(O, 'runs_r45', f'R45_{p}_N_joint_16k_{o}')
        j8 = os.path.join(O, 'runs_r43', f'R43_A_{p}_N_joint_{o}')
        fz = os.path.join(O, 'runs_r43', f'R43_A_{p}_N_frozen_{o}')
        r = dict(parent=p, H=h, obj=o, done_16k=os.path.exists(os.path.join(u16, 'DONE.json')),
                 frozen_8k=rsum(fz, 8000), joint_8k_R43=rsum(j8, 8000),
                 r16k_at_8000=rsum(u16, 8000), r16k_at_12000=rsum(u16, 12000), r16k_at_16000=rsum(u16, 16000))
        d = lambda a, b: None if a is None or b is None else a - b
        r['16k_minus_8k'] = d(r['r16k_at_16000'], r['joint_8k_R43'])
        r['16k_minus_frozen'] = d(r['r16k_at_16000'], r['frozen_8k'])
        rows.append(r)

f = lambda x: 'NA' if x is None else f'{x:.2f}'
L = ['# R45 results (numbers only)', '', '> **Readout note (2026-11-03, R44 audit):** numbers in this file use the native IN²R per-batch centring over the sequential native layout (true image-caption pairs share a batch mean); JOINT students since R39 were also trained with that layout. Layout-free re-evaluations of the same checkpoints: CORRECTED_READOUTS.md; audit: R44_AUDIT.md; layout-free retraining: RESULTS_R47.md.', '',
     'COCO IN²R-native parents, N-JOINT 16000 updates (cosine over 16000, from the original H + fresh zero-output head, training seed 1001) vs the R43-A '
     '8000-update N-JOINT and N-frozen cells on the same parent. DEV = native 1K split (1000 images × 5000 captions), two-stage (frozen-H top-100 reranked). TEST not read.', '',
     '| parent | H | obj | N-frozen (R43-A) | N-JOINT 8k (R43-A) | 16k @8000 | 16k @12000 | 16k @16000 | 16k@16000 − 8k | 16k@16000 − frozen |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['parent']} | {r['H']:.2f} | {r['obj']} | {f(r['frozen_8k'])} | {f(r['joint_8k_R43'])} | {f(r['r16k_at_8000'])} | {f(r['r16k_at_12000'])} | "
             f"{f(r['r16k_at_16000'])} | {f(r['16k_minus_8k'])} | {f(r['16k_minus_frozen'])} |")
L += ['', 'J−BCE / J−NCE at 16000:', '', '| parent | J−BCE | J−NCE |', '|---|---|---|']
for p in PARENTS:
    v = {r['obj']: r['r16k_at_16000'] for r in rows if r['parent'] == p}
    L.append(f"| {p} | {f(None if None in (v['J'], v['BCE']) else v['J'] - v['BCE'])} | {f(None if None in (v['J'], v['NCE']) else v['J'] - v['NCE'])} |")
open(os.path.join(HERE, 'RESULTS_R45.md'), 'w').write('\n'.join(L) + '\n')
with open(os.path.join(HERE, 'R45_RESULTS.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"R45 summary {sum(r['done_16k'] for r in rows)} / {len(rows)}")
print('\n'.join(L[4:]))
