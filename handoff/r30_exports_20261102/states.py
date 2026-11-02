#!/usr/bin/env python3
"""List R30 saved states for the exports: v3 coverage items that still exist (R30 parents, fixed E10, fixed E40 of the
30-row merged matrix) + the 6 A1 units of MATRIX_A1.csv not in that matrix (E10 + E40). Prints one line per (cond, seed):
  COND SEED REF_STATE REF_ID LABEL=PATH=UNIT ...      with --a1-only: only the A1 states.   (R29 states were deleted.)"""
import csv, json, os, subprocess, sys
D = '/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101'
OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101'
SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'COVERAGE_NOW.json')
os.makedirs(os.path.dirname(SP), exist_ok=True)
subprocess.run(['python3', '-B', f'{D}/tools/r30_states.py', f'{D}/states_out_v3', SP], check=True, stdout=subprocess.DEVNULL)
cov = json.load(open(SP))
items = [i for i in cov['items'] if i['exists'] and i['kind'] != 'R29_saved_state']
names = {r['unit_id'] for r in csv.DictReader(open('/home/infres/yinwang/CS_QMI/r30_v11/VCS_R30_v1_1_Merged_Server_20261101/MATRIX.csv'))}
a1 = []
for r in csv.DictReader(open('/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/MATRIX_A1.csv')):
    if r['unit_id'] in names:
        continue
    d = f"{OUT}/runs/{r['output_subdir']}"
    c = 'f00' if r['noise_percent'] == '0' else 'f60'
    for kind, lab, p, ok in (('R30_fixed_E10', 'e10', f'{d}/state_completed10.pt', os.path.isfile(f'{d}/E10_RECORD.json')),
                             ('R30_fixed_E40', 'e40', f'{d}/state_last.pt', os.path.isfile(f'{d}/DONE.json'))):
        if ok and os.path.isfile(p):
            a1.append(dict(kind=kind + '_A1', unit_id=r['unit_id'], cond=c, seed=int(r['seed']), label=f"r30_{lab}_{r['unit_id']}", path=p))
sel = a1 if '--a1-only' in sys.argv else items + a1
refs = {(i['cond'], i['seed']): i for i in cov['items'] if i['kind'] == 'R30_parent_E0' and i['unit_id'].endswith('_H_e5')}
for (c, s), ref in sorted(refs.items()):
    g = [i for i in sel if i['cond'] == c and i['seed'] == s]
    if g:
        print(c, s, ref['path'], ref['unit_id'], ' '.join(f"{i['label']}={i['path']}={i['unit_id']}" for i in g))
