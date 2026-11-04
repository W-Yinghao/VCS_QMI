"""CORRECTED_READOUTS.csv / .md (numbers only) from corrected/results/*.json + corrected/UNITS_SCAN.json: per unit the logged DEV rSum and the layout-free
readouts (tools/corrected_readouts.py), grouped by round; gains relative to the parent H under the SAME centring. Units not evaluated are listed with
their status. Usage: python3 -I -B tools/corrected_aggregate.py"""
import csv, glob, json, re
from pathlib import Path
W = Path(__file__).resolve().parents[1]
scan = json.loads((W / 'corrected' / 'UNITS_SCAN.json').read_text())
res = {}
for f in sorted(glob.glob(str(W / 'corrected' / 'results' / '*.json'))):
    for u, r in json.loads(Path(f).read_text())['units'].items():
        res[u] = r


def rnd(u):
    m = re.match(r'(?i)(r\d+[a-z]?x?)', u); k = m.group(1).upper() if m else '?'
    return {'R37C': 'R37C', 'R39X': 'R39X'}.get(k, k)


def sub(a, b):
    return None if a is None or b is None else a - b


rows = []
for e in scan:
    r = res.get(e['unit'], {})
    rows.append(dict(round=rnd(e['unit']), unit=e['unit'], cond=e['cond'], encoder=e['encoder'], supervision=e['supervision'], objective=e['objective'], steps=e['steps'], oracle=e['oracle'],
                     status=r.get('status', e['status'] if e['status'] != 'TO_EVALUATE' else 'PENDING'), logged=e['logged_rsum'], native_recomputed=r.get('native'),
                     native_minus_logged=r.get('native_minus_logged'), H_native=r.get('H_native'), H_global=r.get('H_global'), H_decoupled=r.get('H_decoupled'),
                     student_global=r.get('student_global'), full_global=r.get('full_global'), full_decoupled=r.get('full_decoupled'),
                     logged_minus_H_native=sub(e['logged_rsum'], r.get('H_native')), full_global_minus_H_global=sub(r.get('full_global'), r.get('H_global')),
                     full_decoupled_minus_H_decoupled=sub(r.get('full_decoupled'), r.get('H_decoupled')),
                     student_cos_native=r.get('student_cos_native'), student_cos_global=r.get('student_cos_global'), gpu=r.get('gpu', e.get('gpu'))))
keys = list(rows[0])
with open(W / 'CORRECTED_READOUTS.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
f = lambda x: 'NA' if x is None else (f'{x:.2f}' if isinstance(x, float) else str(x))
order = ['R37', 'R37C', 'R37S', 'R38', 'R39', 'R39X', 'R40', 'R41', 'R42', 'R43', 'R44', 'R45', 'R46']
from collections import Counter
L = ['# Corrected (layout-free) DEV readouts (numbers only; zero training; TEST not read)', '',
     'logged = trainer readout (native per-batch centring over the sequential native layout). full_global / full_decoupled = parent cache, s_H, stage-1 and (JOINT) '
     'student centring recomputed with one global DEV mean / independent random batches. Gains are relative to the parent H under the same centring. '
     'See R44_AUDIT.md.', '', f"units: {len(rows)}; status: {dict(Counter(r['status'] for r in rows))}", '']
for rd in order + sorted({r['round'] for r in rows} - set(order)):
    R = [r for r in rows if r['round'] == rd]
    if not R:
        continue
    L += [f'## {rd}', '', '| unit | enc | logged | logged-H | full_global | global-H(global) | full_decoupled | decoupled-H(decoupled) | student_global (JOINT) | status |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in R:
        L.append(f"| {r['unit']} | {r['encoder']} | {f(r['logged'])} | {f(r['logged_minus_H_native'])} | {f(r['full_global'])} | {f(r['full_global_minus_H_global'])} | {f(r['full_decoupled'])} | "
                 f"{f(r['full_decoupled_minus_H_decoupled'])} | {f(r['student_global'])} | {r['status']} |")
    L.append('')
(W / 'CORRECTED_READOUTS.md').write_text('\n'.join(L) + '\n')
print('corrected rows', len(rows), dict(Counter(r['status'] for r in rows)))
