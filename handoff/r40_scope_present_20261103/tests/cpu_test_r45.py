"""R45 CPU checks on real data before PROTOCOL_FROZEN_R45.json (writes CPU_TEST_R45.json):
  code_derivation  r45o == r43o byte for byte except common.py, whose diff equals R45_DIFF_VS_R43O.patch
  configs          each unit = its R43-A 8000-update N-JOINT config except the declared R45 fields; 16000 updates, 9 readouts, seed 1001,
                   optimizer = R41-A; prefix reference hash-bound
  lr_schedule      16000-update stretched cosine: m(0) = 1, m(15999) = 0.1, m(8000) on the stretched curve
  e2e_r45o         2-update CPU production run_unit (tests/e2e_r44.py, generic): c60 J (prefix reference init / student init / chain at 0)
Usage: python -I -B tests/cpu_test_r45.py"""
import difflib, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
W = Path(__file__).resolve().parents[1]; PY = sys.executable; RES = {}
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
STRIP = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++', '@@'))]


def check(name, fn):
    t0 = time.time()
    try:
        v = fn(); RES[name] = dict(passed=bool(v.get('passed')), detail=v, seconds=time.time() - t0)
    except Exception:
        RES[name] = dict(passed=False, error=traceback.format_exc(), seconds=time.time() - t0)
    print(name, RES[name]['passed'], round(time.time() - t0, 1), flush=True)
    if not RES[name]['passed']:
        print(RES[name].get('error') or json.dumps(RES[name].get('detail'), default=str)[:6000], flush=True)


def t_code():
    same = {f.name: sh(f) == sh(W / 'r43o' / f.name) for f in sorted((W / 'r45o').glob('*.py')) if f.name != 'common.py'}
    fs = sorted(p.name for p in (W / 'r45o').glob('*.py')) == sorted(p.name for p in (W / 'r43o').glob('*.py'))
    d = ''.join(difflib.unified_diff((W / 'r43o/common.py').read_text().splitlines(True), (W / 'r45o/common.py').read_text().splitlines(True)))
    ok = STRIP(d) == STRIP((W / 'R45_DIFF_VS_R43O.patch').read_text())
    return dict(passed=all(same.values()) and fs and ok, byte_identical=same, file_set=fs, diff_registered=ok)


def t_configs():
    allowed = {'unit_id', 'block', 'line', 'code_tree', 'runner', 'steps', 'ckpt_steps', 'primary_endpoint', 'schedule_rule', 'budget_rule', 'reference_unit', 'reference_note',
               'reference_joint', 'authorisation'}
    out = dict(bad=[]); units = rj(W / 'configs/UNITS_R45.json')
    for u in units:
        c = rj(W / 'configs' / f'{u}.json'); pk = '_'.join(u.split('_')[1:4]); t = rj(W / 'configs' / f"R43_A_{pk}_N_joint_{c['objective']}.json")
        a41 = rj(W / 'configs' / f"R41_A_F60_R30_H1001_N_joint_16k_{c['objective']}.json")
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - allowed
        r = c['reference_joint']
        ok = not diff and c['steps'] == 16000 and c['ckpt_steps'] == [0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000] and c['init_seed'] == 1001 and c['sampler_seed'] == [1001, 3]
        ok &= c['optimizer'] == a41['optimizer'] and c['schedule_rule'] == a41['schedule_rule'] and r['mode'] == 'prefix' and r['unit_id'] == t['unit_id']
        ok &= sh(r['result_json']) == r['result_json_sha256'] and sh(r['ckpt_step0']) == r['ckpt_step0_sha256'] and c['code_tree'] == 'r45o' and c['test_read'] is False
        if not ok:
            out['bad'].append((u, sorted(diff)))
    out['passed'] = len(units) == 6 and not out['bad']
    return out


def t_lr():
    sys.path.insert(0, str(W / 'r45o')); import r38ext as X; sys.path.pop(0)
    m = lambda t, T: X.lr_multiplier(t, T, 'cosine', 0.1)
    out = dict(m0=m(0, 16000), mlast=m(15999, 16000), m8000_16k=m(8000, 16000), m4000_8k=m(4000, 8000))
    out['passed'] = out['m0'] == 1.0 and abs(out['mlast'] - .1) < 1e-15 and abs(out['m8000_16k'] - out['m4000_8k']) < 1e-3
    return out


def t_e2e():
    r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'e2e_r44.py'), 'r45o', 'R45_C60_IN2R_H1001_N_joint_16k_J'], capture_output=True, text=True, timeout=10800, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    line = [l for l in r.stdout.splitlines() if l.startswith('R44_E2E ')]
    out = json.loads(line[-1].split(' ', 1)[1]) if line else dict(failed=dict(passed=False, stdout=r.stdout[-3000:], stderr=r.stderr[-6000:]))
    out['passed'] = r.returncode == 0 and all(v['passed'] for v in out.values() if isinstance(v, dict))
    return out


for n, f in (('code_derivation', t_code), ('configs', t_configs), ('lr_schedule', t_lr), ('e2e_r45o', t_e2e)):
    check(n, f)
out = dict(part='R45', job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()))
(W / 'CPU_TEST_R45.json').write_text(json.dumps(out, indent=1, default=str)); print('R45_CPU_TEST ALL_PASS', out['ALL_PASS'])
