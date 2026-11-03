"""R38 §6 zero-training readouts (diag6): paths and the read-only unit index.
Nothing here writes into any R30-R39 frozen file or output tree; outputs go to OUT6 (projects) and W6 (this dir)."""
from __future__ import annotations
import csv, json
from pathlib import Path

W = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102')
W6 = W / 'diag6'
OUTV = Path('/projects/EEG-foundation-model/yinghao/outputs_v9')
OUT6 = OUTV / 'r38_direction_20261102' / 'diag6'
R38_RUNS = OUTV / 'r38_direction_20261102' / 'runs'
R37_RUNS = OUTV / 'r37_confirm_20261102' / 'runs'
R37S_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102')
R37S_RUNS = OUTV / 'r37s_cc152k_r30h_20261102' / 'critic' / 'runs'
PY = '/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python'

# parent key -> code tree (directory with common.py/train_unit.py), cond name in that tree, native DEV ranks file
TREES = {'r38f': W / 'r38f', 'r38c': W / 'r38c', 'r37s_critic': R37S_ROOT / 'r37s_critic'}
PARENTS = {
    'f60_R30_s1001': dict(tree='r38f', cond='n60', per=5, n_img=1014, native=OUTV / 'r38_direction_20261102/summary/native_ranks_n60.npz'),
    'f60_R30_s1002': dict(tree='r38f', cond='n60s1002', per=5, n_img=1014, native=OUTV / 'r38_direction_20261102/summary/native_ranks_n60s1002.npz'),
    'f60_R30_s1003': dict(tree='r38f', cond='n60s1003', per=5, n_img=1014, native=OUTV / 'r38_direction_20261102/summary/native_ranks_n60s1003.npz'),
    'cc_R31_s1001': dict(tree='r38c', cond='cc', per=1, n_img=1000, native=OUTV / 'r38_direction_20261102/summary/native_ranks_cc.npz'),
    'cc_R31_s1002': dict(tree='r38c', cond='cc_s1002', per=1, n_img=1000, native=OUTV / 'r38_direction_20261102/summary/native_ranks_cc_s1002.npz'),
    'cc_R31_s1003': dict(tree='r38c', cond='cc_s1003', per=1, n_img=1000, native=OUTV / 'r38_direction_20261102/summary/native_ranks_cc_s1003.npz'),
    'cc_R37S_s1001': dict(tree='r37s_critic', cond='cc_r37s', per=1, n_img=1000, native=OUTV / 'r37s_cc152k_r30h_20261102/critic/summary/native_ranks_cc_r37s.npz'),
}
CC_PARENTS = ('cc_R31_s1001', 'cc_R31_s1002', 'cc_R31_s1003', 'cc_R37S_s1001')
FLICKR = ('f60_R30_s1001', 'f60_R30_s1002', 'f60_R30_s1003')


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, default=float)); tmp.replace(p)


def wcsv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ['empty']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def unit_index():
    """every critic unit on the 7 parents with its run dir: R38 units + R36/R37/R37C/R38-A shared-constant references
    (R38 RESULTS_FIXED.csv + REUSE_REFERENCES.json), the R37 B-block factor units (Flickr s1001), the R37S critic units."""
    refs = {r['unit_id']: Path(r['dir']) for r in read_json(W / 'REUSE_REFERENCES.json')['references']}
    U = []
    for r in csv.DictReader(open(W / 'RESULTS_FIXED.csv')):
        d = R38_RUNS / r['unit'] if r['source'] == 'R38 new' else refs.get(r['unit'])
        if d is None:   # R38 A-block (cc s1002/s1003 shared constant) are R38 runs
            d = R38_RUNS / r['unit']
        U.append(dict(unit=r['unit'], dir=d, parent=r['parent'], route=r['route'], architecture=r['architecture'], schedule=r['schedule'],
                      objective=r['objective'], source=r['source'], line=r['unit'].split('_')[0].upper()))
    for f in sorted(R37_RUNS.glob('r37_factor_f60_h1001_*')):
        c = read_json(f / 'RESULT.json')['config']
        route = {'TRUST_NATIVE_NULL': 'TRUST+N', 'N_PRESENT_ONLY': 'N_PRESENT_ONLY'}[c['supervision']]
        U.append(dict(unit=f.name, dir=f, parent='f60_R30_s1001', route=route, architecture='shared', schedule='constant', objective=c['objective'],
                      source='R37 B-block (read-only)', line='R37'))
    for f in sorted(R37S_RUNS.iterdir()):
        c = read_json(R37S_ROOT / 'configs' / f'{f.name}.json')
        U.append(dict(unit=f.name, dir=f, parent=c['parent_key'], route=c['route'], architecture=c['architecture'], schedule=c['optimizer']['schedule'],
                      objective=c['objective'], source='R37S critic (read-only)', line='R37S'))
    for u in U:
        u['tree'] = PARENTS[u['parent']]['tree']
    return U
