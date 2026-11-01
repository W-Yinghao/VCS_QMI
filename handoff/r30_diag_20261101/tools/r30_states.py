#!/usr/bin/env python3
"""Coverage of the v3 probe (DIAG_PROTOCOL_V3.json): R29 saved states (seed 1001, f00/f60), the 7 R30 v1.1 warmup parents
(E0; parents/<id>/parent_state.pt + PARENT.json), the fixed completed-epoch-10 states of the 30 matrix units
(runs/<unit>/state_completed10.pt + E10_RECORD.json) and the fixed E40 of the 12 FULL40 units (runs/<unit>/state_last.pt with
DONE.json, raw epoch 39). Writes COVERAGE_V3.json and, with --jobs, prints one line per (cond, seed) job group:
  COND SEED REF_STATE REF_ID LABEL=PATH=UNIT_ID ...   (existing, not yet measured states; empty groups omitted)
Usage: r30_states.py STATES_DIR COVERAGE.json [--jobs]"""
import csv, glob, json, os, sys

OUT = '/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101'
PKG = '/home/infres/yinwang/CS_QMI/r30_v11/VCS_R30_v1_1_Merged_Server_20261101'
R29 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r29_listwise_20261101/runs'
states_dir, cov_path = sys.argv[1], sys.argv[2]
jobs = '--jobs' in sys.argv[3:]
measured = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(states_dir, '*.json'))}
items = []
for c in ('f00', 'f60'):
    for a in ('h_native', 'n_infonce', 'l_listwise_vcs', 'b_listwise_bce'):
        for e, t, raw in (('epoch10', 'e10', 10), ('epoch20', 'e20', 20), ('last', 'last', 39)):
            p = f'{R29}/{c}_s1001/{a}/state_{e}.pt'
            items.append(dict(kind='R29_saved_state', unit_id=f'r29_{c}_s1001__{a}@raw{raw}', cond=c, seed=1001, label=f'{a}_{t}', path=p,
                              completed_cotrain_epochs=raw + 1, exists=os.path.isfile(p)))
for r in csv.DictReader(open(f'{PKG}/configs/WARMUP_PARENTS.csv')):
    d = f"{OUT}/parents/{r['parent_id']}"
    items.append(dict(kind='R30_parent_E0', unit_id=r['parent_id'], cond='f00' if r['noise_percent'] == '0' else 'f60', seed=int(r['seed']),
                      label=f"r30_parent_{r['parent_id']}", path=f'{d}/parent_state.pt', completed_cotrain_epochs=0,
                      exists=os.path.isfile(f'{d}/PARENT.json') and os.path.isfile(f'{d}/parent_state.pt')))
for r in csv.DictReader(open(f'{PKG}/MATRIX.csv')):
    d = f"{OUT}/runs/{r['output_subdir']}"
    c = 'f00' if r['noise_percent'] == '0' else 'f60'
    items.append(dict(kind='R30_fixed_E10', unit_id=r['unit_id'], cond=c, seed=int(r['seed']), label=f"r30_e10_{r['unit_id']}",
                      path=f'{d}/state_completed10.pt', completed_cotrain_epochs=10,
                      exists=os.path.isfile(f'{d}/state_completed10.pt') and os.path.isfile(f'{d}/E10_RECORD.json')))
    if r['execution_class'] == 'FULL40_REQUIRED':
        items.append(dict(kind='R30_fixed_E40', unit_id=r['unit_id'], cond=c, seed=int(r['seed']), label=f"r30_e40_{r['unit_id']}",
                          path=f'{d}/state_last.pt', completed_cotrain_epochs=40,
                          exists=os.path.isfile(f'{d}/DONE.json') and os.path.isfile(f'{d}/state_last.pt')))
for it in items:
    it['measured'] = f"{it['cond']}_s{it['seed']}__{it['label']}" in measured
refs = {(it['cond'], it['seed']): it for it in items if it['kind'] == 'R30_parent_E0' and it['unit_id'].endswith('_H_e5')}
json.dump({'items': items, 'reference_parent_per_condition_seed': {f'{c}_s{s}': v['unit_id'] for (c, s), v in refs.items()},
           'note': 'only states that exist are measured; groups and the fixed list need the H parent of the same condition/seed; R29 states use the s1001 H parent of their condition'},
          open(cov_path, 'w'), indent=1)
if jobs:
    for (c, s), ref in sorted(refs.items()):
        if not ref['exists']:
            continue
        sel = [it for it in items if it['cond'] == c and it['seed'] == s and it['exists'] and not it['measured']]
        if sel:
            print(c, s, ref['path'], ref['unit_id'], ' '.join(f"{it['label']}={it['path']}={it['unit_id']}" for it in sel))
else:
    for k in ('R29_saved_state', 'R30_parent_E0', 'R30_fixed_E10', 'R30_fixed_E40'):
        sel = [it for it in items if it['kind'] == k]
        print(k, 'expected', len(sel), 'exist', sum(it['exists'] for it in sel), 'measured', sum(it['measured'] for it in sel))
