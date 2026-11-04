"""R46 CPU checks on real data before PROTOCOL_FROZEN_R46.json (writes CPU_TEST_R46.json):
  code_derivation  r46f == r42f (minus oracle runner / oracle_truth) and r46o == r43o byte for byte except common.py / train_unit.py (+ data35.py in
                   r46o); train_unit.py = tools/r46_patch.py output on the source (re-applied, byte-equal); diffs equal R46_DIFF_R46F.patch / R46_DIFF_R46O.patch
  prep             subset pools: 29k / 58k images nested, 5 observed captions per kept image, pool records' observed image in the subset, candidate rows
                   of the pool inside the pool, manifests hash-bound; round-2 tables: shapes, sha = manifest, round-1 checkpoint sha = config
  configs          A = R43-A c80 config except the declared R46 fields; B = round-1 config + init_from / sampler [1001, 46] / candidate_file (REBUILT)
  e2e_r46o / e2e_r46f  2-update CPU production run_unit (tests/e2e_r46.py) incl. the round-2 init (no readout on CPU: the step-0 round-1 DEV
                   rank equality runs in the GPU smoke)
Amendment (after CPU check 1085184, record CPU_TEST_R46_1085184_failed.json): the r46o CPU e2e uses the 29k pool FROZEN unit instead of the JOINT one:
on CPU fp32 kernels the R39 step-0 student-vs-native-cache equivalence of COCO captions is 1.3e-4 > the frozen 1e-4 tolerance (the cache was
built on GPU); the same check passed at the frozen tolerance in the GPU smoke 1085185 (R46_A_..._P29K_N_joint_J), which is the runtime path.
Usage: python -I -B tests/cpu_test_r46.py"""
import difflib, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]; PY = sys.executable; RES = {}
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())
STRIP = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++', '@@'))]
ud = lambda a, b: ''.join(difflib.unified_diff(Path(a).read_text().splitlines(True), Path(b).read_text().splitlines(True)))


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
    out = {}
    for new, old, changed in (('r46f', 'r42f', ('common.py', 'train_unit.py')), ('r46o', 'r43o', ('common.py', 'train_unit.py', 'data35.py'))):
        files = sorted(p.name for p in (W / new).glob('*.py')); exp = sorted(p.name for p in (W / old).glob('*.py') if p.name not in ('train_oracle.py', 'oracle_truth.py'))
        same = {f: sh(W / new / f) == sh(W / old / f) for f in files if f not in changed}
        tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f'r46_patch_check_{new}_{os.getpid()}.py'; tmp.write_text((W / old / 'train_unit.py').read_text())
        r = subprocess.run([PY, '-I', '-B', str(W / 'tools' / 'r46_patch.py'), str(tmp)], capture_output=True, text=True)
        cur = ''.join(ud(W / old / f, W / new / f) for f in changed)
        out[new] = dict(file_set=files == exp, byte_identical=same, train_unit_is_patch_output=r.returncode == 0 and sh(tmp) == sh(W / new / 'train_unit.py'),
                        diff_registered=STRIP(cur) == STRIP((W / f'R46_DIFF_{new.upper()}.patch').read_text()))
        out[new]['passed'] = out[new]['file_set'] and all(same.values()) and out[new]['train_unit_is_patch_output'] and out[new]['diff_registered']
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_prep():
    out = {}; obs = np.load('/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/noise/coco_precomp_0.8.npy').astype(np.int64)
    pools = {}
    for n in (29, 58):
        cond = f'c80in_s1001_p{n}k'; pdir = OUT / 'r46_parents' / 'prep' / cond; pm = rj(pdir / 'PREP_MANIFEST.json')
        P = np.load(pdir / f'pool_{cond}.npz'); S, R = P['images'], P['records']; Z = np.load(pdir / f'candidates_{cond}.npz'); pools[n] = S
        r = dict(n_images=len(S), n_records=len(R), five_per_image=bool((np.bincount(obs[R], minlength=113287)[S] == 5).all()), obs_in_subset=bool(np.isin(obs[R], S).all()),
                 i2t_rows_in_pool=bool(np.isin(Z['i2t_top32'][S], R).all()), t2i_rows_in_pool=bool(np.isin(Z['t2i_top32'][R], S).all()),
                 hashes=all(sh(pdir / f) == v['sha256'] for f, v in pm['files'].items()), a0=pm['calibration_init'][cond]['a0'], coverage=pm['candidates'][cond])
        jm = rj(OUT / 'r46_parents' / 'joint' / cond / 'JOINT_PREP_MANIFEST.json'); r['joint_copy'] = jm['cond'] == cond and all(sh(OUT / 'r46_parents' / 'joint' / cond / f) == v['sha256'] for f, v in jm['files'].items())
        r['passed'] = len(S) == n * 1000 and len(R) == 5 * n * 1000 and all(r[k] for k in ('five_per_image', 'obs_in_subset', 'i2t_rows_in_pool', 't2i_rows_in_pool', 'hashes', 'joint_copy')) and r['a0'] > 0
        out[cond] = r
    out['nested'] = dict(passed=bool(np.isin(pools[29], pools[58]).all()))
    for cond in ('n60', 'f80in', 'c80in_s1001'):
        m = rj(OUT / 'r46_parents' / 'r2cand' / f'R2CAND_MANIFEST_{cond}.json')
        out[f'r2cand_{cond}'] = dict(passed=sorted(m['tables']) == ['BCE', 'J', 'NCE'] and all(sh(v['path']) == v['sha256'] for v in m['tables'].values()),
                                     parent=m['parent_tables']['coverage'], rebuilt={o: v['coverage'] for o, v in m['tables'].items()})
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_configs():
    units = rj(W / 'configs' / 'UNITS_R46.json'); out = dict(n=len(units), bad=[])
    allowed = {'unit_id', 'block', 'line', 'code_tree', 'runner', 'oracle', 'test_read', 'label', 'authorisation', 'reference_unit', 'reference_joint', 'reference_note',
               'adaptation', 'student_modalities', 'encoder', 'job_bundle', 'package_unit', 'package_block', 'joint_rule', 'r39_template', 'teacher_parents', 'extension', 'matrix_source'}
    R39C = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs')
    for u in units:
        c = rj(W / 'configs' / f'{u}.json'); o = c['objective']
        if c['block'] == 'A':
            kind = 'frozen' if c['adaptation'] == 'frozen' else 'joint'; t = rj(W / 'configs' / f'R43_A_C80_IN2R_H1001_N_{kind}_{o}.json'); extra = {'cond', 'fit_pool'}
            ok = c['cond'] in ('c80in_s1001_p29k', 'c80in_s1001_p58k') and c['code_tree'] == 'r46o' and c['sampler_seed'] == [1001, 3]
        else:
            pk = '_'.join(u.split('_')[2:5])
            t = rj({'F60_R30_H1001': R39C / f'R39_Flickr60_N_joint_global_{o}.json', 'F80_IN2R_H1001': W / 'configs' / f'R41_C_F80_IN2R_H1001_N_joint_{o}.json',
                    'C80_IN2R_H1001': W / 'configs' / f'R43_A_C80_IN2R_H1001_N_joint_{o}.json'}[pk])
            extra = {'sampler_seed', 'init_from', 'round2', 'seed_rule', 'head_init', 'schedule_rule', 'candidate_file', 'supervision_rule', 'evaluation', 'architecture_rule'}
            inf = c['init_from']
            ok = c['sampler_seed'] == [1001, 46] and sh(inf['ckpt']) == inf['ckpt_sha256'] and sh(inf['dev_ranks']) == inf['dev_ranks_sha256'] and inf['step'] == 8000
            ok &= (c['round2'] == 'REBUILT') == bool(c.get('candidate_file')) and (not c.get('candidate_file') or sh(c['candidate_file']['path']) == c['candidate_file']['sha256'])
            ok &= c['code_tree'] == ('r46o' if pk.startswith('C80') else 'r46f') and c['encoder'] == 'joint_global'
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - allowed - extra
        ok &= not diff and c['steps'] == 8000 and c['test_read'] is False and c['optimizer'] == t['optimizer'] and c['init_seed'] == 1001
        if not ok:
            out['bad'].append((u, sorted(diff)))
    out['passed'] = out['n'] == 30 and not out['bad']
    return out


def t_e2e(tree, units):
    r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'e2e_r46.py'), tree, ','.join(units)], capture_output=True, text=True, timeout=14400, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    line = [l for l in r.stdout.splitlines() if l.startswith('R46_E2E ')]
    out = json.loads(line[-1].split(' ', 1)[1]) if line else dict(failed=dict(passed=False, stdout=r.stdout[-3000:], stderr=r.stderr[-6000:]))
    out['passed'] = r.returncode == 0 and all(v['passed'] for v in out.values() if isinstance(v, dict))
    return out


for n, f in (('code_derivation', t_code), ('prep', t_prep), ('configs', t_configs),
             ('e2e_r46f', lambda: t_e2e('r46f', ['R46_B_F60_R30_H1001_R2_REBUILT_J', 'R46_B_F80_IN2R_H1001_R2_CONTINUED_NCE'])),
             ('e2e_r46o', lambda: t_e2e('r46o', ['R46_A_C80_IN2R_H1001_P29K_N_frozen_J', 'R46_B_C80_IN2R_H1001_R2_REBUILT_BCE']))):
    check(n, f)
out = dict(part='R46', job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()))
(W / 'CPU_TEST_R46.json').write_text(json.dumps(out, indent=1, default=str)); print('R46_CPU_TEST ALL_PASS', out['ALL_PASS'])
