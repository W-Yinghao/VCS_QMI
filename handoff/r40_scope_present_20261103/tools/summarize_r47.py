"""R47 summary (numbers only): N-JOINT trained with layout-free centring (G: one parent mean per split/modality/peer;
S: student centred by its own training-batch mean, frozen parent cache global). Primary DEV readout = the unit's own
layout-free readout; H = the unit's step-0 readout (layout-free parent H). References from CORRECTED_READOUTS.csv
(zero-training layout-free re-evaluation, column full_global): the matched N-frozen and the native-centring N-JOINT
(R39 / R39X / R41-C / R43-A) on the same parent. TEST not read.
Writes RESULTS_R47.md / R47_RESULTS.csv / R47_CURVES.csv. Usage: python3 -I -B tools/summarize_r47.py"""
import csv, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/runs_r47'
OBJ = ('J', 'BCE', 'NCE')
STEPS = (250, 500, 1000, 2000, 4000, 8000)
CONDS = {  # cond tag in R47 unit name: (label, N-frozen ref pattern, native-centring JOINT ref pattern)
    'F60_R30_H1001': ('Flickr60 R30-H s1001', 'r38_B_f60_R30_s1001_N_shared_{o}_cosine', 'R39_Flickr60_N_joint_global_{o}'),
    'F80_IN2R_H1001': ('Flickr80 IN2R f80 s1001', 'R41_C_F80_IN2R_H1001_N_frozen_{o}', 'R41_C_F80_IN2R_H1001_N_joint_{o}'),
    'F00_R30_H1001': ('Flickr0 R30-H f00 s1001', 'R39X_Flickr00_R30H_s1001_N_frozen_shared_{o}', 'R39X_Flickr00_R30H_s1001_N_joint_global_{o}'),
    'C60_IN2R_H1001': ('COCO60 IN2R c60 s1001', 'R43_A_C60_IN2R_H1001_N_frozen_{o}', 'R43_A_C60_IN2R_H1001_N_joint_{o}'),
    'CC31_H1001': ('CC152K R31-H0 s1001', 'r38_B_cc_R31_s1001_N_shared_{o}_cosine', 'R39_CC152K_N_joint_global_{o}'),
}
CR = {r['unit']: r for r in csv.DictReader(open(os.path.join(HERE, 'CORRECTED_READOUTS.csv')))}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ref(pat, o, col='full_global'):
    r = CR.get(pat.format(o=o))
    return None if r is None or r.get('status') != 'EVALUATED' else fnum(r.get(col))


def ev(d, s):
    f = os.path.join(d, f'eval_step{s}.json')
    return json.load(open(f))['dev_retrieval']['rsum'] if os.path.exists(f) else None


rows, curves = [], []
for c, (lab, pf, pj) in CONDS.items():
    for v in ('G', 'S'):
        for o in OBJ:
            d = os.path.join(O, f'R47_{v}_{c}_N_joint_{o}')
            done = os.path.exists(os.path.join(d, 'DONE.json'))
            H = ev(d, 0)
            r8 = ev(d, 8000) if done else None
            fz, jn = ref(pf, o), ref(pj, o)
            sub = lambda a, b: None if a is None or b is None else a - b
            rows.append(dict(cond=c, label=lab, variant=v, obj=o, done=done, H_layoutfree=H, rsum_8000=r8,
                             minus_H=sub(r8, H), Nfrozen_global=fz, minus_Nfrozen=sub(r8, fz),
                             nativeJOINT_global=jn, minus_nativeJOINT=sub(r8, jn)))
            for s in (0,) + STEPS:
                x = ev(d, s)
                if x is not None:
                    curves.append(dict(cond=c, variant=v, obj=o, step=s, rsum=x))

f = lambda x: 'NA' if x is None else f'{x:.2f}'
L = ['# R47 results (numbers only)', '',
     'N-JOINT, 8000 updates, R39 JOINT recipe, training seed 1001, trained and read out with layout-free centring. '
     'G = JOINT student and frozen parent cache centred by one parent mean per split/modality/peer; S = student centred by its own '
     'mean over the unique items of each training batch (DEV: fixed random 512-item batches), frozen parent cache global. '
     'H = each unit\'s step-0 readout (layout-free parent H). References (zero-training layout-free re-evaluation, CORRECTED_READOUTS.csv '
     'column full_global): N-frozen and the native-centring N-JOINT on the same parent. TEST not read.', '',
     f"units complete: {sum(r['done'] for r in rows)} / {len(rows)}", '',
     '| condition | variant | obj | H (layout-free) | R47 @8000 | −H | N-frozen (global) | −N-frozen | native-centring JOINT (global) | −native JOINT |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['label']} | {r['variant']} | {r['obj']} | {f(r['H_layoutfree'])} | {f(r['rsum_8000'])} | {f(r['minus_H'])} | "
             f"{f(r['Nfrozen_global'])} | {f(r['minus_Nfrozen'])} | {f(r['nativeJOINT_global'])} | {f(r['minus_nativeJOINT'])} |")
L += ['', '## DEV rSum curves (layout-free readout)', '', '| condition | variant | obj | ' + ' | '.join(str(s) for s in (0,) + STEPS) + ' |',
      '|---|---|---|' + '---|' * (len(STEPS) + 1)]
for c in CONDS:
    for v in ('G', 'S'):
        for o in OBJ:
            m = {x['step']: x['rsum'] for x in curves if x['cond'] == c and x['variant'] == v and x['obj'] == o}
            L.append(f"| {c} | {v} | {o} | " + ' | '.join(f(m.get(s)) for s in (0,) + STEPS) + ' |')
open(os.path.join(HERE, 'RESULTS_R47.md'), 'w').write('\n'.join(L) + '\n')
for name, data in (('R47_RESULTS.csv', rows), ('R47_CURVES.csv', curves)):
    with open(os.path.join(HERE, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print(f"R47 summary {sum(r['done'] for r in rows)} / {len(rows)}")
print('\n'.join(L[6:6 + 2 + len(rows)]))
