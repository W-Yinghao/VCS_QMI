"""R48 CPU check: (1) r48f/r48o/r48c = byte copies of r47f/r47o/r47c except common.py, whose prefix is the R47 common.py byte-for-byte and whose
tail only rebinds RUNS / ORACLE_RUNS / FROZEN_FILE(S); (2) each R48 config differs from its paired R47 G N-JOINT config only in the allowed keys, its
frozen fields equal its frozen template (where present) and it is frozen (encoder frozen, no student modalities, no joint rule); (3) in each tree the
unit's cond resolves to the R47 layout-free cache / prep with existing manifests, RUNS = runs_r48 and the freeze file is PROTOCOL_FROZEN_R48.json.
Writes CPU_TEST_R48.json. Usage: python -I -B tests/cpu_test_r48.py"""
import json, os, subprocess, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
PY = '/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python'
rd = lambda p: json.loads(Path(p).read_text())
ALLOWED = {'adaptation', 'architecture_rule', 'authorisation', 'block', 'centring_rule', 'code_tree', 'encoder', 'evaluation', 'frozen_template', 'joint_rule',
           'line', 'optimizer', 'paired_joint_unit', 'reference_note', 'runner', 'student_modalities', 'unit_id'}
FROZEN_FIELDS = ('adaptation', 'architecture_rule', 'encoder', 'evaluation', 'joint_rule', 'optimizer', 'student_modalities')
res = dict(job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks={})
def chk(name, ok, **info):
    res['checks'][name] = dict(passed=bool(ok), **info)

for t in 'foc':
    a, b = W / f'r47{t}', W / f'r48{t}'
    fa = sorted(p.name for p in a.glob('*.py')); fb = sorted(p.name for p in b.glob('*.py'))
    same = [n for n in fa if n != 'common.py' and (a / n).read_bytes() == (b / n).read_bytes()]
    ca, cb = (a / 'common.py').read_bytes(), (b / 'common.py').read_bytes()
    tail = cb[len(ca):].decode()
    assign = [l.split('=')[0].strip() for l in tail.splitlines() if l and not l.startswith('#') and '=' in l]
    chk(f'tree_r48{t}', fa == fb and len(same) == len(fa) - 1 and cb.startswith(ca) and set(assign) == {'RUNS', 'ORACLE_RUNS', 'FROZEN_FILE', 'FROZEN_FILES_R48', 'FROZEN_FILES_R41'},
        files=len(fb), identical_non_common=len(same), tail_assignments=assign)

units = rd(W / 'configs' / 'UNITS_R48.json')
bad = {}
for u in units:
    c = rd(W / 'configs' / f'{u}.json'); g = rd(W / 'configs' / f"{c['paired_joint_unit']}.json")
    tp = Path(next(str(p) for p in [W / 'configs' / f"{c['frozen_template']['unit']}.json",
                                    Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs') / f"{c['frozen_template']['unit']}.json",
                                    Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs') / f"{c['frozen_template']['unit']}.json"] if p.exists()))
    t = rd(tp)
    diff = {k for k in set(c) | set(g) if c.get(k) != g.get(k)}
    errs = []
    if not diff <= ALLOWED: errs.append(f'unexpected keys {sorted(diff - ALLOWED)}')
    for k in FROZEN_FIELDS:
        if k in t and t[k] != c[k]: errs.append(f'frozen field {k} != template')
    if not (c['encoder'] == 'frozen' and c['adaptation'] == 'frozen' and c['student_modalities'] == [] and c['joint_rule'] is None): errs.append('not frozen')
    if c['optimizer'] == g['optimizer']: errs.append('optimizer equals JOINT optimizer')
    for k in ('cond', 'init_seed', 'sampler_seed', 'steps', 'objective', 'supervision', 'ckpt_steps'):
        if c.get(k) != g.get(k): errs.append(f'{k} differs from paired JOINT')
    if errs: bad[u] = errs
chk('configs', len(units) == 15 and not bad, units=len(units), errors=bad)

probe = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import common as C
conds = json.loads(sys.argv[2]); out = {}
for c in conds:
    d = C.CONDS[c]; cd, pd = Path(d['cache_dir']), Path(d['prep_dir'])
    out[c] = dict(cache_dir=str(cd), prep_dir=str(pd), cache_manifest=(cd / 'CACHE_MANIFEST.json').exists(), prep_manifest=(pd / 'PREP_MANIFEST.json').exists(),
                  r47_parents='r47_parents' in str(cd) and 'r47_parents' in str(pd))
print(json.dumps(dict(RUNS=str(C.RUNS), FROZEN_FILE=C.FROZEN_FILE, FROZEN_FILES_R41=list(C.FROZEN_FILES_R41), conds=out)))
'''
for t in ('r48f', 'r48o', 'r48c'):
    conds = sorted({rd(W / 'configs' / f'{u}.json')['cond'] for u in units if rd(W / 'configs' / f'{u}.json')['code_tree'] == t})
    p = subprocess.run([PY, '-I', '-B', '-c', probe, str(W / t), json.dumps(conds)], capture_output=True, text=True, cwd='/tmp')
    try:
        o = json.loads(p.stdout.strip().splitlines()[-1])
        ok = o['RUNS'].endswith('runs_r48') and o['FROZEN_FILES_R41'] == ['PROTOCOL_FROZEN_R48.json'] and all(v['cache_manifest'] and v['prep_manifest'] and v['r47_parents'] for v in o['conds'].values())
    except Exception as e:
        o, ok = dict(error=repr(e), stderr=p.stderr[-2000:]), False
    chk(f'common_{t}', ok, **o)

res['ALL_PASS'] = all(v['passed'] for v in res['checks'].values())
(W / 'CPU_TEST_R48.json').write_text(json.dumps(res, indent=1))
print('R48 CPU ALL_PASS', res['ALL_PASS'], {k: v['passed'] for k, v in res['checks'].items()})
sys.exit(0 if res['ALL_PASS'] else 1)
