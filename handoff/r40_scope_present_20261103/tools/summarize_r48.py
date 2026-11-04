"""R48 summary (numbers only): N-frozen trained and read out on the R47 layout-free G parent cache, paired with R47 G N-JOINT (same init / sampler
stream / cache / readout). H = each unit's step-0 readout (layout-free parent H). Also listed: the native-centring N-frozen of the same parent re-read
layout-free (CORRECTED_READOUTS.csv, full_global). TEST not read. Writes RESULTS_R48.md / R48_RESULTS.csv / R48_CURVES.csv.
Usage: python3 -I -B tools/summarize_r48.py"""
import csv, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103'
OBJ = ('J', 'BCE', 'NCE')
STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
CONDS = {
    'F60_R30_H1001': ('Flickr60 R30-H s1001', 'r38_B_f60_R30_s1001_N_shared_{o}_cosine'),
    'F80_IN2R_H1001': ('Flickr80 IN2R f80 s1001', 'R41_C_F80_IN2R_H1001_N_frozen_{o}'),
    'F00_R30_H1001': ('Flickr0 R30-H f00 s1001', 'R39X_Flickr00_R30H_s1001_N_frozen_shared_{o}'),
    'C60_IN2R_H1001': ('COCO60 IN2R c60 s1001', 'R43_A_C60_IN2R_H1001_N_frozen_{o}'),
    'CC31_H1001': ('CC152K R31-H0 s1001', 'r38_B_cc_R31_s1001_N_shared_{o}_cosine'),
}
CR = {r['unit']: r for r in csv.DictReader(open(os.path.join(HERE, 'CORRECTED_READOUTS.csv')))}


def ev(d, s):
    f = os.path.join(d, f'eval_step{s}.json')
    return json.load(open(f))['dev_retrieval']['rsum'] if os.path.exists(f) else None


def cr(u):
    r = CR.get(u)
    try:
        return float(r['full_global']) if r and r.get('status') == 'EVALUATED' else None
    except (TypeError, ValueError):
        return None


sub = lambda a, b: None if a is None or b is None else a - b
rows, curves = [], []
for c, (lab, oldf) in CONDS.items():
    for o in OBJ:
        fd = os.path.join(OUT, 'runs_r48', f'R48_{c}_N_frozen_lfG_{o}')
        jd = os.path.join(OUT, 'runs_r47', f'R47_G_{c}_N_joint_{o}')
        done = os.path.exists(os.path.join(fd, 'DONE.json')); jdone = os.path.exists(os.path.join(jd, 'DONE.json'))
        H = ev(fd, 0); fz = ev(fd, 8000) if done else None; jt = ev(jd, 8000) if jdone else None; old = cr(oldf.format(o=o))
        rows.append(dict(cond=c, label=lab, obj=o, done=done, H_layoutfree=H, frozen_lfG_8000=fz, frozen_minus_H=sub(fz, H),
                         joint_lfG_8000_R47G=jt, joint_minus_H=sub(jt, H), joint_minus_frozen=sub(jt, fz),
                         old_frozen_relaid_global=old, R48_minus_old_relaid=sub(fz, old)))
        for s in STEPS:
            x = ev(fd, s)
            if x is not None:
                curves.append(dict(cond=c, obj=o, step=s, rsum=x))

f = lambda x: 'NA' if x is None else f'{x:.2f}'
L = ['# R48 results (numbers only)', '',
     'N-frozen (shared G head, cosine schedule, frozen parent encoders), 8000 updates, trained and read out on the R47 layout-free G parent cache '
     '(one parent mean per split / modality / peer). Paired with R47 G N-JOINT (same init, sampler stream, cache and readout). H = step-0 readout '
     '(layout-free parent H). "old frozen re-read" = the native-centring N-frozen of the same parent (R38 / R39X / R41-C / R43-A), re-read layout-free '
     '(CORRECTED_READOUTS.csv, full_global). DEV two-stage (frozen-H top-100 reranked). TEST not read.', '',
     f"units complete: {sum(r['done'] for r in rows)} / {len(rows)}", '',
     '| condition | obj | H (layout-free) | R48 N-frozen @8000 | −H | R47 G N-JOINT @8000 | JOINT−H | JOINT−FROZEN | old frozen re-read | R48−old |',
     '|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['label']} | {r['obj']} | {f(r['H_layoutfree'])} | {f(r['frozen_lfG_8000'])} | {f(r['frozen_minus_H'])} | {f(r['joint_lfG_8000_R47G'])} | "
             f"{f(r['joint_minus_H'])} | {f(r['joint_minus_frozen'])} | {f(r['old_frozen_relaid_global'])} | {f(r['R48_minus_old_relaid'])} |")
L += ['', '## R48 DEV rSum curves', '', '| condition | obj | ' + ' | '.join(str(s) for s in STEPS) + ' |', '|---|---|' + '---|' * len(STEPS)]
for c in CONDS:
    for o in OBJ:
        m = {x['step']: x['rsum'] for x in curves if x['cond'] == c and x['obj'] == o}
        L.append(f"| {c} | {o} | " + ' | '.join(f(m.get(s)) for s in STEPS) + ' |')
open(os.path.join(HERE, 'RESULTS_R48.md'), 'w').write('\n'.join(L) + '\n')
for name, data in (('R48_RESULTS.csv', rows), ('R48_CURVES.csv', curves)):
    with open(os.path.join(HERE, name), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print(f"R48 summary {sum(r['done'] for r in rows)} / {len(rows)}")
print('\n'.join(L[6:8 + len(rows)]))
