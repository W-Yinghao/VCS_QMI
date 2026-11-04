"""R49 summary (numbers only): layout-free N-frozen over all 24 parents (R48: 5 parents, R49: 19 parents), each trained and read out on its
R47-style layout-free G parent cache, 8000 updates, J/BCE/NCE. H = each unit's step-0 readout (layout-free parent H). "old frozen re-read" = the
native-centring N-frozen of the same parent re-read layout-free (CORRECTED_READOUTS.csv, full_global). TEST not read.
Writes RESULTS_R49.md / R49_RESULTS.csv / R49_CURVES.csv. Usage: python3 -I -B tools/summarize_r49.py"""
import csv, json, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103'
OBJ = ('J', 'BCE', 'NCE')
STEPS = (0, 250, 500, 1000, 2000, 4000, 8000)
CR = {r['unit']: r for r in csv.DictReader(open(os.path.join(HERE, 'CORRECTED_READOUTS.csv')))}
GROUP = lambda pk: ('Flickr30K', 'COCO', 'CC152K')[0 if pk.startswith('F') else 1 if pk.startswith('C') and not pk.startswith('CC') else 2]


def ev(d, s):
    f = os.path.join(d, f'eval_step{s}.json')
    return json.load(open(f))['dev_retrieval']['rsum'] if os.path.exists(f) else None


def cr(u):
    r = CR.get(u)
    try:
        return float(r['full_global']) if r and r.get('status') == 'EVALUATED' else None
    except (TypeError, ValueError):
        return None


units = []
for line, runs, ufile in (('R48', 'runs_r48', 'UNITS_R48.json'), ('R49', 'runs_r49', 'UNITS_R49.json')):
    for u in json.load(open(os.path.join(HERE, 'configs', ufile))):
        c = json.load(open(os.path.join(HERE, 'configs', f'{u}.json')))
        old = c.get('native_frozen_unit') or (c.get('frozen_template') or {}).get('unit')
        pk = u.split('_N_frozen_')[0].split('_', 1)[1]
        units.append((line, pk, c['objective'], c['cond'], os.path.join(OUT, runs, u), old))
sub = lambda a, b: None if a is None or b is None else a - b
rows, curves = [], []
for line, pk, o, cond, d, old in units:
    done = os.path.exists(os.path.join(d, 'DONE.json'))
    H = ev(d, 0); fz = ev(d, 8000) if done else None; ol = cr(old) if old else None
    rows.append(dict(line=line, dataset=GROUP(pk), parent=pk, obj=o, cond=cond, done=done, H_layoutfree=H, frozen_lfG_8000=fz, minus_H=sub(fz, H),
                     old_frozen_unit=old, old_frozen_relaid_global=ol, minus_old=sub(fz, ol)))
    for s in STEPS:
        x = ev(d, s)
        if x is not None:
            curves.append(dict(line=line, parent=pk, obj=o, step=s, rsum=x))
order = {'Flickr30K': 0, 'COCO': 1, 'CC152K': 2}
rows.sort(key=lambda r: (order[r['dataset']], r['parent'], OBJ.index(r['obj'])))
f = lambda x: 'NA' if x is None else f'{x:.2f}'
L = ['# R49 results (numbers only)', '',
     'Layout-free N-frozen (shared G head, cosine schedule, frozen parent encoders), 8000 updates, trained and read out on each parent\'s layout-free G '
     'cache (one parent mean per split / modality / peer); R48 (5 parents) and R49 (19 parents). H = step-0 readout (layout-free parent H). '
     '"old frozen re-read" = the native-centring N-frozen of the same parent, re-read layout-free (CORRECTED_READOUTS.csv, full_global). '
     'DEV two-stage (frozen-H top-100 reranked). TEST not read.', '',
     f"units complete: {sum(r['done'] for r in rows)} / {len(rows)}", '',
     '| dataset | parent | line | obj | H (layout-free) | N-frozen @8000 | −H | old frozen re-read | −old |', '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['dataset']} | {r['parent']} | {r['line']} | {r['obj']} | {f(r['H_layoutfree'])} | {f(r['frozen_lfG_8000'])} | {f(r['minus_H'])} | "
             f"{f(r['old_frozen_relaid_global'])} | {f(r['minus_old'])} |")
L += ['', '## N-frozen − H by dataset and objective (mean / min / max over parents; complete units only)', '',
      '| dataset | obj | parents | mean | min | max | parents > 0 |', '|---|---|---|---|---|---|---|']
for ds in order:
    for o in OBJ:
        v = [r['minus_H'] for r in rows if r['dataset'] == ds and r['obj'] == o and r['minus_H'] is not None]
        if v:
            L.append(f"| {ds} | {o} | {len(v)} | {sum(v) / len(v):.2f} | {min(v):.2f} | {max(v):.2f} | {sum(x > 0 for x in v)} |")
L += ['', '## DEV rSum curves', '', '| parent | obj | ' + ' | '.join(str(s) for s in STEPS) + ' |', '|---|---|' + '---|' * len(STEPS)]
for r in rows:
    m = {x['step']: x['rsum'] for x in curves if x['parent'] == r['parent'] and x['obj'] == r['obj']}
    L.append(f"| {r['parent']} | {r['obj']} | " + ' | '.join(f(m.get(s)) for s in STEPS) + ' |')
open(os.path.join(HERE, 'RESULTS_R49.md'), 'w').write('\n'.join(L) + '\n')
for name, data in (('R49_RESULTS.csv', rows), ('R49_CURVES.csv', curves)):
    if data:
        with open(os.path.join(HERE, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print(f"R49 summary {sum(r['done'] for r in rows)} / {len(rows)}")
i = L.index('## N-frozen − H by dataset and objective (mean / min / max over parents; complete units only)')
print('\n'.join(L[i:i + 3 + 9]))
