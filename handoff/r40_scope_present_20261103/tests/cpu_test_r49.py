"""R49 CPU check: (1) r49f/r49o/r49c = byte copies of r48f/r48o/r47cca except common.py, whose prefix is the source common.py byte-for-byte and whose
tail only rebinds RUNS / ORACLE_RUNS / FROZEN_FILE(S) and adds the R49 layout-free conds; (2) each R49 config differs from its source only in allowed keys
(direct: the parent's own N-frozen config; via_R48: the R48 config of the dataset plus the parent fields of the parent's R39X JOINT config and the frozen
fields of its R38 / R37S N-frozen config), is frozen, and keeps init seed / sampler / steps / objective / supervision; (3) in each tree every unit cond
resolves to an R47-style layout-free cache / prep with existing manifests (built by tools/r47_build_lf.py), RUNS = runs_r49, freeze file R49.
Writes CPU_TEST_R49.json. Usage: python -I -B tests/cpu_test_r49.py"""
import json, os, subprocess, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
PY = '/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python'
rd = lambda p: json.loads(Path(p).read_text())
CFG_DIRS = [W / 'configs', Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs'),
            Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs'), Path('/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/configs')]
find = lambda stem: rd(next(d / f'{stem}.json' for d in CFG_DIRS if (d / f'{stem}.json').exists()))
BOOK = {'authorisation', 'block', 'centring_rule', 'centring_variant', 'code_tree', 'cond', 'derivation', 'line', 'native_frozen_unit', 'reference_note', 'runner',
        'unit_id', 'oracle', 'test_read', 'reference_unit', 'reference_joint', 'paired_joint_unit', 'frozen_template'}
PARENT = {'parent_H', 'parent_key', 'parent_seed'}
FROZEN_FIELDS = ('adaptation', 'architecture_rule', 'encoder', 'evaluation', 'joint_rule', 'optimizer', 'student_modalities')
KEEP = ('init_seed', 'sampler_seed', 'steps', 'objective', 'supervision', 'ckpt_steps')
res = dict(job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})
chk = lambda name, ok, **info: res['checks'].__setitem__(name, dict(passed=bool(ok), **info))

for dst, src in (('r49f', 'r48f'), ('r49o', 'r48o'), ('r49c', 'r47cca')):
    a, b = W / src, W / dst
    fa = sorted(p.name for p in a.glob('*.py')); fb = sorted(p.name for p in b.glob('*.py'))
    same = [n for n in fa if n != 'common.py' and (a / n).read_bytes() == (b / n).read_bytes()]
    ca, cb = (a / 'common.py').read_bytes(), (b / 'common.py').read_bytes()
    tail = cb[len(ca):].decode()
    top = [l.split('=')[0].strip() for l in tail.splitlines() if l and not l.startswith((' ', '#', 'for ')) and '=' in l]
    chk(f'tree_{dst}', fa == fb and len(same) == len(fa) - 1 and cb.startswith(ca) and set(top) == {'RUNS', 'ORACLE_RUNS', 'FROZEN_FILE', 'FROZEN_FILES_R49', 'FROZEN_FILES_R41', 'R49_LF_BASES'},
        source=src, files=len(fb), identical_non_common=len(same), tail_top_level=top)

units = rd(W / 'configs' / 'UNITS_R49.json'); bad = {}
for u in units:
    c = rd(W / 'configs' / f'{u}.json'); d = c['derivation']; errs = []
    if d['kind'] == 'direct':
        s = find(d['frozen_config']); allowed = BOOK
        if s['cond'] + '_lfG' != c['cond']: errs.append('cond mismatch')
    else:
        s = find(d['r48_config']); j = find(d['parent_fields_from']); f = find(d['frozen_fields_from']); allowed = BOOK | PARENT | set(FROZEN_FIELDS)
        for k in PARENT:
            if c.get(k) != j.get(k): errs.append(f'{k} != R39X JOINT parent field')
        if j['cond'] + '_lfG' != c['cond']: errs.append('cond mismatch')
        for k in FROZEN_FIELDS:
            if k in f and f[k] != c[k]: errs.append(f'frozen field {k} != R38/R37S template')
    diff = {k for k in set(c) | set(s) if c.get(k) != s.get(k)}
    if not diff <= allowed: errs.append(f'unexpected keys {sorted(diff - allowed)}')
    for k in KEEP:
        if c.get(k) != s.get(k): errs.append(f'{k} differs from source')
    if not (c['encoder'] == 'frozen' and c['adaptation'] == 'frozen' and c['student_modalities'] == [] and c['joint_rule'] is None): errs.append('not frozen')
    if errs: bad[u] = errs
chk('configs', len(units) == 57 and not bad, units=len(units), errors=bad)

probe = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import common as C
out = {}
for c in json.loads(sys.argv[2]):
    d = C.CONDS[c]; cd, pd = Path(d['cache_dir']), Path(d['prep_dir'])
    out[c] = dict(cache_manifest=(cd / 'CACHE_MANIFEST.json').exists(), prep_manifest=(pd / 'PREP_MANIFEST.json').exists(), r47_parents='r47_parents' in str(cd) and 'r47_parents' in str(pd))
print(json.dumps(dict(RUNS=str(C.RUNS), FROZEN_FILES_R41=list(C.FROZEN_FILES_R41), conds=out)))
'''
for t in ('r49f', 'r49o', 'r49c'):
    conds = sorted({rd(W / 'configs' / f'{u}.json')['cond'] for u in units if rd(W / 'configs' / f'{u}.json')['code_tree'] == t})
    p = subprocess.run([PY, '-B', '-c', probe, str(W / t), json.dumps(conds)], capture_output=True, text=True, cwd='/tmp')
    try:
        o = json.loads(p.stdout.strip().splitlines()[-1])
        ok = o['RUNS'].endswith('runs_r49') and o['FROZEN_FILES_R41'] == ['PROTOCOL_FROZEN_R49.json'] and all(all(v.values()) for v in o['conds'].values())
    except Exception as e:
        o, ok = dict(error=repr(e), stderr=p.stderr[-2000:]), False
    chk(f'common_{t}', ok, n_conds=len(conds), **o)

res['ALL_PASS'] = all(v['passed'] for v in res['checks'].values())
(W / 'CPU_TEST_R49.json').write_text(json.dumps(res, indent=1))
print('R49 CPU ALL_PASS', res['ALL_PASS'], {k: v['passed'] for k, v in res['checks'].items()})
sys.exit(0 if res['ALL_PASS'] else 1)
