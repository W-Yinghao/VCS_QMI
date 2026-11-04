"""R43 CPU checks on real data (Slurm CPU job), one part per block, before PROTOCOL_FROZEN_R43.json. Writes CPU_TEST_R43_<part>.json.
part B (TRUST+N on the 80% Flickr parents, tree r43f; on CC152K CC31, tree r43c):
  code_derivation  r43f == r42f and r43c == r41c byte for byte except common.py, whose diffs equal R43F_DIFF_VS_R42F.patch / R43C_DIFF_VS_R41C.patch
  configs          every B unit = its N reference config (R41 C / R42 f80, R38 CC frozen, R39 CC joint) except the declared R43 fields;
                   supervision TRUST_NATIVE_NULL with the R37 B1 rule text; 8000 updates; seeds 1001 / [1001, 3]; no in-unit reference
  trust_prep       R43 prep dirs: candidate table byte-identical to the R41 / R42 source (sha), calibration entry identical, B-TRUST subset
                   non-empty, inside the record range, rule = R37 trust_cond; CC: R37C trust_cc.npz hash-bound in the R37C prep manifest
  e2e              2-update CPU production run_unit: f80in TRUST frozen J, f80in_s1002 TRUST joint NCE (r43f), CC TRUST joint BCE (r43c)
part A (MS-COCO, tree r43o):
  code_derivation  r43o files = r42f (heads, objectives, native_eval, r38ext, r39ext, r40ext, r39_targets) / r41c (evals) byte for byte;
                   common / data35 (vs r41c) and train_unit / data36 (vs r42f) diffs equal R43O_DIFF.patch
  configs          every A unit = the R41 C f80 config of the same cell except the declared R43 fields
  parents          7 COCO parents: exact native DEV reproduction (1000 x 5000), float32 re-score identical, a0 > 0, locks shapes,
                   JOINT equivalence max |diff| <= 1e-4, DEV ranks from student vectors = native; tokens vocab bound
  e2e              2-update CPU production run_unit: c20 frozen J, c60 s1001 joint BCE
Usage: python -I -B tests/cpu_test_r43.py B|A"""
from __future__ import annotations
import difflib, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PY = sys.executable
RES = {}
R38C = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs')
R39C = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs')
R37C_ = Path('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/configs')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')


def check(name, fn):
    t0 = time.time()
    try:
        v = fn(); ok = bool(v if not isinstance(v, dict) else v.get('passed', True))
        RES[name] = dict(passed=ok, detail=v if isinstance(v, dict) else None, seconds=time.time() - t0)
    except Exception:
        RES[name] = dict(passed=False, error=traceback.format_exc(), seconds=time.time() - t0)
    print(name, RES[name]['passed'], round(time.time() - t0, 1), flush=True)
    if not RES[name]['passed']:
        print(RES[name].get('error') or json.dumps(RES[name].get('detail'), default=str)[:6000], flush=True)


def sh(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rj(p):
    return json.loads(Path(p).read_text())


STRIP = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++', '@@'))]


def udiff(a, b):
    return ''.join(difflib.unified_diff(Path(a).read_text().splitlines(True), Path(b).read_text().splitlines(True)))


def t_code_B():
    out = {}
    for new, old, patch in (('r43f', 'r42f', 'R43F_DIFF_VS_R42F.patch'), ('r43c', 'r41c', 'R43C_DIFF_VS_R41C.patch')):
        same = {f.name: sh(f) == sh(W / old / f.name) for f in sorted((W / new).glob('*.py')) if f.name != 'common.py'}
        fs = sorted(p.name for p in (W / new).glob('*.py')) == sorted(p.name for p in (W / old).glob('*.py'))
        ok = STRIP(udiff(W / old / 'common.py', W / new / 'common.py')) == STRIP((W / patch).read_text())
        out[new] = dict(byte_identical=same, file_set_equal=fs, diff_equals_registered_patch=ok, passed=all(same.values()) and fs and ok)
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_code_A():
    core = ('heads.py', 'objectives.py', 'native_eval.py', 'r38ext.py', 'r39ext.py', 'r40ext.py', 'r39_targets.py')
    same = {f: sh(W / 'r43o' / f) == sh(W / 'r42f' / f) for f in core}
    same['evals.py'] = sh(W / 'r43o' / 'evals.py') == sh(W / 'r41c' / 'evals.py')
    fs = sorted(p.name for p in (W / 'r43o').glob('*.py')) == sorted(core + ('evals.py', 'common.py', 'data35.py', 'train_unit.py', 'data36.py'))
    patch = ''.join(udiff(W / 'r41c' / f, W / 'r43o' / f) for f in ('common.py', 'data35.py', 'evals.py'))
    patch += ''.join(udiff(W / 'r42f' / f, W / 'r43o' / f) for f in ('train_unit.py', 'data36.py', 'heads.py', 'objectives.py', 'native_eval.py', 'r38ext.py', 'r39ext.py', 'r40ext.py', 'r39_targets.py'))
    ok = STRIP(patch) == STRIP((W / 'R43O_DIFF.patch').read_text())
    return dict(passed=all(same.values()) and fs and ok, byte_identical=same, file_set=fs, diff_equals_registered_patch=ok)


ALLOWED = {'unit_id', 'block', 'line', 'code_tree', 'runner', 'oracle', 'test_read', 'label', 'authorisation', 'reference_unit', 'reference_joint', 'reference_note',
           'supervision', 'supervision_label', 'supervision_rule', 'adaptation', 'student_modalities', 'encoder', 'job_bundle', 'package_unit', 'package_block', 'joint_rule'}


def t_configs(part):
    units = [u for u in rj(W / 'configs' / 'UNITS_R43.json') if u.startswith(f'R43_{part}_')]
    out = dict(n=len(units), bad=[])
    t37 = {o: rj(R37C_ / f'r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_{o}.json') for o in ('J', 'BCE', 'NCE')}
    for u in units:
        c = rj(W / 'configs' / f'{u}.json'); o = c['objective']; kind = 'frozen' if c['adaptation'] == 'frozen' else 'joint'
        extra = set()
        if part == 'A':
            t = rj(W / 'configs' / f'R41_C_F80_IN2R_H1001_N_{kind}_{o}.json'); extra = {'cond', 'noise_percent', 'parent_seed', 'parent_H', 'parent_key', 'dataset', 'evaluation'}
            ok = c['code_tree'] == 'r43o' and c['supervision'] == 'OBS_MISSING_NULL' and c['cond'].startswith('c') and c['dataset'].startswith('MS-COCO')
        elif c['cond'] == 'cc':
            t = rj(R38C / f'r38_B_cc_R31_s1001_N_shared_{o}_cosine.json') if kind == 'frozen' else rj(R39C / f'R39_CC152K_N_joint_global_{o}.json')
            ok = c['code_tree'] == 'r43c'
            if kind == 'joint':
                ok &= c['optimizer'] == t['optimizer']
        else:
            nm = {'f80in': 'R41_C_F80_IN2R_H1001', 'f80in_s1002': 'R42_N_F80_IN2R_H1002', 'f80in_s1003': 'R42_N_F80_IN2R_H1003'}[c['cond']]
            t = rj(W / 'configs' / f'{nm}_N_{kind}_{o}.json'); ok = c['code_tree'] == 'r43f'
        if part == 'B':
            ok &= c['supervision'] == 'TRUST_NATIVE_NULL' and c['supervision_rule'].startswith(t37[o]['supervision_rule'])
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - ALLOWED - extra
        ok &= not diff and c['steps'] == 8000 and c['init_seed'] == 1001 and c['sampler_seed'] == [1001, 3] and c['test_read'] is False and not c['oracle']
        ok &= c['runner'] == f"{c['code_tree']}/train_unit.py" and c.get('reference_unit') is None and c.get('reference_joint') is None
        ok &= (c['encoder'] == 'joint_global') == ('lr_encoder' in c['optimizer']) and c['optimizer'] == t['optimizer']
        ok &= c['architecture'] == 'shared' and c['optimizer']['schedule'] == 'cosine' and c['ckpt_steps'] == [0, 250, 500, 1000, 2000, 4000, 8000]
        if not ok:
            out['bad'].append((u, sorted(diff)))
    out['passed'] = out['n'] == (42 if part == 'A' else 24) and not out['bad']
    return out


def t_trust():
    import numpy as np
    sys.path.insert(0, str(W / 'r43f'))
    import common as C
    out = {}
    src = {'f80in': C.R41_PARENTS / 'prep', 'f80in_s1002': C.R42_PARENTS / 'prep' / 'f80in_s1002', 'f80in_s1003': C.R42_PARENTS / 'prep' / 'f80in_s1003'}
    for cond, s in src.items():
        pdir = Path(C.CONDS[cond]['prep_dir']); pm = rj(pdir / 'PREP_MANIFEST.json'); sm = rj(s / 'PREP_MANIFEST.json'); cn = f'candidates_{cond}.npz'
        T = np.load(pdir / C.CONDS[cond]['trust_file'])['subset']
        r = dict(candidates_identical=sh(pdir / cn) == sh(s / cn) == sm['files'][cn]['sha256'] == pm['files'][cn]['sha256'],
                 calibration_identical=pm['calibration_init'][cond] == sm['calibration_init'][cond], source_manifest_sha=pm['source_prep_manifest']['sha256'] == sh(s / 'PREP_MANIFEST.json'),
                 r35=pm['r35_prep_manifest_sha256'] == sh('/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json'),
                 trust_sha=sh(pdir / C.CONDS[cond]['trust_file']) == pm['files'][C.CONDS[cond]['trust_file']]['sha256'],
                 n_subset=int(len(T)), retention=pm['trust'][cond]['retention'], sorted_unique=bool((np.diff(T) > 0).all()), in_range=bool(T.min() >= 0 and T.max() < C.N_TRAIN_CAP),
                 rule=pm['trust'][cond]['rule'], gpu=pm['trust'][cond]['gpu'])
        r['passed'] = all(r[k] for k in ('candidates_identical', 'calibration_identical', 'source_manifest_sha', 'r35', 'trust_sha', 'sorted_unique', 'in_range')) and 0 < r['n_subset'] < C.N_TRAIN_CAP
        out[cond] = r
    sys.path.remove(str(W / 'r43f')); sys.modules.pop('common', None)
    cpm = rj('/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/prep/PREP_MANIFEST.json')
    tc = '/projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/prep/trust_cc.npz'
    out['cc'] = dict(trust_sha=sh(tc) == cpm['files']['trust_cc.npz']['sha256'], retention=cpm['trust']['cc']['retention'], rule=cpm['trust']['cc']['rule'])
    out['cc']['passed'] = out['cc']['trust_sha']
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_parents_A():
    import numpy as np
    sys.path.insert(0, str(W / 'r43o'))
    import common as C
    out = {}
    tm = rj(C.TOKENS_FILE.with_name('TOKENS_MANIFEST.json'))
    for cond in C.CONDS:
        cm = rj(Path(C.CONDS[cond]['cache_dir']) / 'CACHE_MANIFEST.json'); pm = rj(Path(C.CONDS[cond]['prep_dir']) / 'PREP_MANIFEST.json')
        jm = rj(Path(C.JOINT_PARENTS[cond]['prep_dir']) / 'JOINT_PREP_MANIFEST.json')
        L = np.load(Path(C.CONDS[cond]['prep_dir']) / 'locks_records.npz')
        r = dict(native_rsum=cm['native_dev']['rsum'], checkpoint_best_rsum=cm['native_dev']['checkpoint_best_rsum'], exact=cm['native_dev']['reproduces_checkpoint_best_rsum_exactly'],
                 rescore=cm['native_dev']['cache_rescore_identical'], gpu=cm['env']['gpu'], noise_file=cm['observed_map']['path'], realized_mismatch=cm['observed_map']['realized_mismatch'],
                 dev_shapes=(cm['files']['dev_img_peerA.npy']['shape'], cm['files']['dev_cap_peerA.npy']['shape']), fit_shapes=(cm['files']['fit_img_peerA.npy']['shape'], cm['files']['fit_cap_peerA.npy']['shape']),
                 a0=pm['calibration_init'][cond]['a0'], candidates=pm['candidates'][cond], locks={k: list(L[k].shape) for k in L.files},
                 joint_max_abs=jm['equivalence_summary']['max_abs_over_all'], dev_from_student=jm['equivalence']['dev_native_from_student_vectors'],
                 tokens_bound=jm['tokens']['sha256'] == tm['sha256'] == sh(C.TOKENS_FILE), vocab_size=(jm['parent']['vocab_size'], tm['vocab_size']))
        r['passed'] = (r['exact'] and r['rescore'] and r['dev_shapes'] == ([1000, 1024], [5000, 1024]) and r['fit_shapes'] == ([113287, 1024], [566435, 1024]) and r['a0'] > 0
                       and r['joint_max_abs'] <= 1e-4 and r['dev_from_student']['n_i2t_rank_diff'] == 0 and r['dev_from_student']['n_t2i_rank_diff'] == 0 and r['tokens_bound']
                       and r['vocab_size'][0] == r['vocab_size'][1] and r['locks']['cal_p'] == [5000] and r['locks']['cal_q'] == [50000, 2] and r['noise_file'] == C.CONDS[cond]['noise_file'])
        out[cond] = r
    sys.path.remove(str(W / 'r43o')); sys.modules.pop('common', None)
    out['passed'] = all(v['passed'] for v in out.values()) and len(out) == 7
    return out


def t_e2e(tree, units):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'e2e_r43.py'), tree, ','.join(units)], capture_output=True, text=True, timeout=10800, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('R43_E2E ')]
    out = json.loads(line[-1].split(' ', 1)[1]) if line else dict(failed=dict(passed=False, stdout=r.stdout[-3000:], stderr=r.stderr[-6000:]))
    out['passed'] = r.returncode == 0 and all(v['passed'] for v in out.values() if isinstance(v, dict))
    return out


PART = sys.argv[1] if len(sys.argv) > 1 else 'B'


def main():
    if PART == 'B':
        checks = (('code_derivation', t_code_B), ('configs', lambda: t_configs('B')), ('trust_prep', t_trust),
                  ('e2e_r43f', lambda: t_e2e('r43f', ['R43_B_F80_IN2R_H1001_TRUST_frozen_J', 'R43_B_F80_IN2R_H1002_TRUST_joint_NCE'])),
                  ('e2e_r43c', lambda: t_e2e('r43c', ['R43_B_CC31_H1001_TRUST_joint_BCE'])))
    else:
        checks = (('code_derivation', t_code_A), ('configs', lambda: t_configs('A')), ('parents', t_parents_A),
                  ('e2e_r43o', lambda: t_e2e('r43o', ['R43_A_C20_IN2R_H1001_N_frozen_J', 'R43_A_C60_IN2R_H1001_N_joint_BCE'])))
    for n, f in checks:
        check(n, f)
    out = dict(part='R43-' + PART, job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES,
               ALL_PASS=all(v['passed'] for v in RES.values()))
    (W / f'CPU_TEST_R43_{PART}.json').write_text(json.dumps(out, indent=1, default=str))
    print('R43_CPU_TEST ALL_PASS', out['ALL_PASS'])


if __name__ == '__main__':
    main()
