"""R50 CPU checks on real data (before any formal R50 unit; writes CPU_TEST_R50.json). The package's own local tests are run too but are NOT a real-data gate.
  package_reference  package tests/test_reference.py (11 local CPU tests on synthetic data: centring / loss / PRESENT reference)
  code_derivation    r50f / r50c = byte copies of r48f / r48c + exactly the tools/r50_patch.py output (re-applied to fresh copies and compared byte for byte);
                     the registered diffs R50_DIFF_R50F.patch / R50_DIFF_R50C.patch match
  builds             R50 caches (protocol id, FIT files = R47 G FIT files by sha256, FIT mean recomputed == R47 G file bitwise, full-scale check A), prep identical
                     to R47 G (candidates + a0/b0), JOINT prep (global mode, center_dev == center_fit, equivalence), H-only builds (n00, c60)
  configs            24 units vs their N references (R48 frozen / R47 G JOINT): only declared keys differ; same init / sampler / steps / optimizer / readouts;
                     supervision / oracle / runner from MATRIX.csv; 18 replay configs differ from the reference only in the declared keys
  static_truth_scan  the training-path code (sampler, list builder, objectives, heads, student encoder, step_lists, r50ext) never uses //5, PER_CAP, truth,
                     DEV_GT or dev_clean; JointInputs uses batch_index only in its native_batch branch (R50 asserts mode global in-unit)
  real_<cond>        tests/r50_cpu_real.py: checks A-G on real data (CPU)
  e2e_*              H: production run_unit 2 updates on CPU (non-oracle r50f / r50c; oracle runner r50f)
Usage: python -I -B tests/cpu_test_r50.py"""
import ast, csv, difflib, glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
W = Path(__file__).resolve().parents[1]; PY = sys.executable; RES = {}
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
PKG = Path('/home/infres/yinwang/CS_QMI/r50_layout_audit/VCS_Layout_Audit_R50_20261104')
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); rj = lambda p: json.loads(Path(p).read_text())


def check(name, fn):
    t0 = time.time()
    try:
        v = fn(); RES[name] = dict(passed=bool(v.get('passed')), detail=v, seconds=time.time() - t0)
    except Exception:
        RES[name] = dict(passed=False, error=traceback.format_exc(), seconds=time.time() - t0)
    print(name, RES[name]['passed'], round(time.time() - t0, 1), flush=True)
    if not RES[name]['passed']:
        print(RES[name].get('error') or json.dumps(RES[name].get('detail'), default=str)[:8000], flush=True)


def t_pkg():
    r = subprocess.run([PY, '-I', '-B', str(PKG / 'tests' / 'test_reference.py')], capture_output=True, text=True, cwd=tempfile.gettempdir())
    n = len(re.findall(r'\.\.\. ok', r.stderr))
    return dict(passed=r.returncode == 0 and n == 11 and 'OK' in r.stderr, tests_ok=n, tail=r.stderr[-600:], package_file_sha256=sh(PKG / 'tests' / 'test_reference.py'))


def t_code():
    out = {}
    for new, old, kind in (('r50f', 'r48f', 'f'), ('r50c', 'r48c', 'c')):
        tmp = Path(tempfile.mkdtemp(prefix=f'r50chk_{new}_')) / new; tmp.mkdir()
        for f in (W / old).glob('*.py'):
            shutil.copyfile(f, tmp / f.name)
        r = subprocess.run([PY, '-I', '-B', str(W / 'tools' / 'r50_patch.py'), str(tmp), kind], capture_output=True, text=True)
        a = sorted(p.name for p in (W / new).glob('*.py')); b = sorted(p.name for p in tmp.glob('*.py'))
        same = {f: sh(W / new / f) == sh(tmp / f) for f in a if (tmp / f).exists()}
        unchanged = {f: sh(W / new / f) == sh(W / old / f) for f in a if (W / old / f).exists() and f not in ('common.py', 'train_unit.py')}
        cur = ''.join(''.join(difflib.unified_diff((W / old / f).read_text().splitlines(True) if (W / old / f).exists() else [], (W / new / f).read_text().splitlines(True),
                                                   f'{old}/{f}', f'{new}/{f}')) for f in a if not ((W / old / f).exists() and sh(W / old / f) == sh(W / new / f)))
        reg = W / f'R50_DIFF_{new.upper()}.patch'
        out[new] = dict(patch_rc=r.returncode, file_set=a == b, patch_output_identical=same, other_files_byte_identical_to_r48=unchanged,
                        oracle_truth_is_r42f=(sh(W / new / 'oracle_truth.py') == sh(W / 'r42f' / 'oracle_truth.py')) if kind == 'f' else None,
                        diff_registered=reg.exists() and reg.read_text() == cur)
        out[new]['passed'] = bool(r.returncode == 0 and a == b and all(same.values()) and all(unchanged.values()) and out[new]['diff_registered'] and out[new]['oracle_truth_is_r42f'] in (None, True))
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def check_a(nm, tv=1e-4, tm=1e-6):
    """check A re-evaluated under amendment A1 (vector 1e-4, mean 1e-6) from the recorded full-scale numbers"""
    A = nm['check_A_full_scale']
    ok = all(v['mean_perm_vs_natural']['max_abs'] <= tm and v['centred_perm_vs_natural']['max_abs'] <= tv for k, v in A.items() if isinstance(v, dict) and 'mean_perm_vs_natural' in v)
    return bool(ok and A['dev_scores_perm_fit_mean']['max_abs'] <= tv and A['dev_scores_perm_fit_mean']['n_i2t_rank_diff'] == 0 and A['dev_scores_perm_fit_mean']['n_t2i_rank_diff'] == 0)


def t_builds():
    out = {}
    for b in ('n60', 'f80in', 'cc'):
        cd, pd, jd = OUT / 'r50_parents/cache' / b, OUT / 'r50_parents/prep' / b, OUT / 'r50_parents/joint' / f'{b}_F'
        cm, pm, jm, nm = rj(cd / 'CACHE_MANIFEST.json'), rj(pd / 'PREP_MANIFEST.json'), rj(jd / 'JOINT_PREP_MANIFEST.json'), rj(cd / f'NORMALIZATION_{b}.json')
        gm = rj(OUT / 'r47_parents/cache' / b / 'CACHE_MANIFEST.json'); gp = rj(OUT / 'r47_parents/prep' / b / 'PREP_MANIFEST.json')
        files_ok = all(sh(cd / n) == v['sha256'] for n, v in cm['files'].items())
        fit_r47 = all(cm['files'][n]['sha256'] == gm['files'][n]['sha256'] for n in gm['files'] if n.startswith('fit_') or n == 'observed_map.npy')
        dev_new = all(cm['files'][n]['sha256'] != gm['files'][n]['sha256'] for n in gm['files'] if n.startswith('dev_') or n.startswith('native_dev'))
        cf = {f: rj(jd / 'JOINT_PREP_MANIFEST.json')['files'][f]['sha256'] for f in jm['files']}
        jc = all(cf[f'center_dev_{k}_peer{p}.npy'] == cf[f'center_fit_{k}_peer{p}.npy'] for k in ('img', 'cap') for p in 'AB')
        r = dict(H_primary=cm['native_dev']['rsum'], H_secondary_r47G=gm['native_dev']['rsum'], protocol=cm['protocol_id'], files_ok=files_ok, fit_files_are_r47G=fit_r47,
                 dev_files_differ_from_r47G=dev_new, fit_mean_equal=all(v['fit_mean_recomputed_equal_bitwise'] for v in nm['per_peer_modality'].values()),
                 fit_cache_recomputed_max_abs=max(v['fit_cache_vs_r47G']['max_abs'] for v in nm['per_peer_modality'].values()), check_A=check_a(nm), check_A_original_tol_1e5=nm['check_A_full_scale']['passed'],
                 prep_identical=pm['identical_to_r47G'], prep_files=all(sh(pd / n) == v['sha256'] for n, v in pm['files'].items()),
                 candidates_sha_equal_r47G=pm['files'][f'candidates_{b}.npz']['sha256'] == gp['files'][f'candidates_{b}.npz']['sha256'],
                 joint_mode=jm['centring_mode'], joint_dev_center_eq_fit=jc, joint_equivalence_max_abs=jm['equivalence_summary']['max_abs_over_all'],
                 secondary_dev_reproduced_max_abs=max(v['secondary_transductive_dev_reproduces_r47G_dev_cache']['max_abs'] for v in nm['per_peer_modality'].values()))
        r['passed'] = bool(r['protocol'] == 'FIT_FIXED_CENTER_V1' and files_ok and fit_r47 and dev_new and r['fit_mean_equal'] and r['check_A'] and r['prep_identical'] and r['prep_files']
                           and r['candidates_sha_equal_r47G'] and r['joint_mode'] == 'global' and jc and r['joint_equivalence_max_abs'] <= 1e-5)
        out[b] = r
    for b in ('n00', 'c60in_s1001'):
        cd = OUT / 'r50_parents/cache' / b; cm = rj(cd / 'CACHE_MANIFEST.json'); nm = rj(cd / f'NORMALIZATION_{b}.json')
        out[b] = dict(H_primary=cm['native_dev']['rsum'], H_secondary_r47G=cm['native_dev']['secondary_transductive_rsum_r47G'], h_only=cm['h_only'], check_A=check_a(nm), check_A_original_tol_1e5=nm['check_A_full_scale']['passed'],
                      fit_mean_equal=all(v['fit_mean_recomputed_equal_bitwise'] for v in nm['per_peer_modality'].values()),
                      files_ok=all(sh(cd / n) == v['sha256'] for n, v in cm['files'].items()))
        out[b]['passed'] = bool(out[b]['h_only'] and out[b]['check_A'] and out[b]['fit_mean_equal'] and out[b]['files_ok'])
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_configs():
    units = rj(W / 'configs/UNITS_R50.json'); out = dict(n=len(units), bad={})
    allowed = {'unit_id', 'line', 'block', 'cond', 'code_tree', 'runner', 'supervision', 'supervision_label', 'supervision_rule', 'oracle', 'protocol_id', 'centring_variant',
               'centring_rule', 'reference_unit', 'reference_joint', 'dataset', 'test_read', 'n_reference', 'package', 'label', 'authorisation', 'paired_joint_unit', 'frozen_template',
               'reference_note'}
    mat = {r['unit']: r for r in csv.DictReader(open(PKG / 'MATRIX.csv'))}
    for u in units:
        c = rj(W / 'configs' / f'{u}.json'); t = rj(W / 'configs' / f"{c['n_reference']['unit_id']}.json"); m = mat[u]
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)}
        e = []
        if diff - allowed: e.append(f'unexpected {sorted(diff - allowed)}')
        for k in ('init_seed', 'sampler_seed', 'steps', 'ckpt_steps', 'optimizer', 'objective', 'n_queries', 'clip_per_peer', 'adaptation', 'encoder', 'structure', 'joint_rule'):
            if c.get(k) != t.get(k): e.append(f'{k} differs')
        if c['cond'] != t['cond'][:-len('_lfG')] + '_lfF': e.append('cond')
        if c['supervision'] != {'PRESENT_ONLY': 'N_PRESENT_ONLY', 'ORACLE_MISSING': 'ORACLE_MISSING'}[m['supervision']] or c['oracle'] != (m['oracle'] == 'True'): e.append('supervision/oracle')
        if c['runner'].endswith('train_oracle.py') != c['oracle'] or c['objective'] != m['objective'] or (c['adaptation'] == 'frozen') != (m['encoder'] == 'frozen'): e.append('runner/objective/encoder')
        if c['protocol_id'] != 'FIT_FIXED_CENTER_V1' or c['steps'] != 8000 or c['test_read']: e.append('protocol/steps/test')
        if e: out['bad'][u] = e
    rp = rj(W / 'configs/UNITS_R50_REPLAY.json'); out['replay'] = len(rp)
    for u in rp:
        c = rj(W / 'configs' / f'{u}.json'); t = rj(W / 'configs' / f"{c['replay_of']['unit_id']}.json")
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)}
        if diff - {'unit_id', 'line', 'block', 'cond', 'code_tree', 'runner', 'replay_of', 'label'} or c['cond'] != t['cond'][:-4] + '_lfF':
            out['bad'][u] = sorted(diff)
    out['passed'] = len(units) == 24 and len(rp) == 18 and not out['bad']
    return out


def t_static():
    pat = re.compile(r'//\s*5\b|PER_CAP|truth|DEV_GT|dev_clean')
    out = {}
    targets = {'train_unit.py': ('step_lists', '_pairs', 'make_opt', 'peer_params', 'stu_pairs', 'g_pairs_lists'), 'data35.py': ('build_lists', 'sample_lists'),
               'data36.py': None, 'objectives.py': None, 'heads.py': None, 'r39ext.py': ('raw_encode', 'pairs', 'build_students', 'Student')}
    for t in ('r50f', 'r50c'):
        hits = {}
        for fn, names in targets.items():
            tree = ast.parse((W / t / fn).read_text()); src = (W / t / fn).read_text()
            nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (names is None or n.name in names)]
            for n in nodes:
                for x in ast.walk(n):   # drop docstrings; ast.unparse drops comments
                    if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.body and isinstance(x.body[0], ast.Expr) and isinstance(getattr(x.body[0], 'value', None), ast.Constant) and isinstance(x.body[0].value.value, str):
                        x.body = x.body[1:] or [ast.Pass()]
                code = ast.unparse(n)
                m = sorted(set(pat.findall(code)))
                if m: hits[f'{fn}:{n.name}'] = m
        src = (W / t / 'r39ext.py').read_text()
        enc = ast.get_source_segment(src, next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == 'encode'))
        bi_guarded = "if self.mode == 'native_batch':\n                    mu = self.mu[(split, kind, p)][batch_index(kind" in enc and enc.count('batch_index') == 1
        out[t] = dict(hits=hits, batch_index_only_in_native_batch_branch=bi_guarded, passed=not hits and bi_guarded)
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_sub(cmd, tag, timeout=14400):
    r = subprocess.run([PY, '-I', '-B'] + cmd, capture_output=True, text=True, timeout=timeout, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), cwd=tempfile.gettempdir())
    line = [l for l in r.stdout.splitlines() if l.startswith(tag + ' ')]
    if not line:
        return dict(passed=False, rc=r.returncode, stdout=r.stdout[-3000:], stderr=r.stderr[-6000:])
    o = json.loads(line[-1].split(' ', 1)[1])
    ok = r.returncode == 0 and (o['ALL_PASS'] if 'ALL_PASS' in o else all(v['passed'] for v in o.values()))
    return dict(o, passed=bool(ok), rc=r.returncode)


T = W / 'tests'
for n, f in (('package_reference', t_pkg), ('code_derivation', t_code), ('builds', t_builds), ('configs', t_configs), ('static_truth_scan', t_static),
             ('real_n60', lambda: t_sub([str(T / 'r50_cpu_real.py'), 'r50f', 'n60_lfF'], 'R50_CPU_REAL')),
             ('real_f80in', lambda: t_sub([str(T / 'r50_cpu_real.py'), 'r50f', 'f80in_lfF'], 'R50_CPU_REAL')),
             ('e2e_r50f', lambda: t_sub([str(T / 'r50_e2e.py'), 'r50f', 'R50_A_F60_R30H1001_PRESENT_ONLY_frozen_J,R50_A_F60_R30H1001_PRESENT_ONLY_both_joint_BCE'], 'R50_E2E')),
             ('e2e_oracle', lambda: t_sub([str(T / 'train_oracle_e2e_r50.py'), 'R50_B_F80_IN2RH1001_ORACLE_MISSING_frozen_NCE,R50_B_F80_IN2RH1001_ORACLE_MISSING_both_joint_J'], 'R50_ORACLE_E2E')),
             ('real_cc', lambda: t_sub([str(T / 'r50_cpu_real.py'), 'r50c', 'cc_lfF'], 'R50_CPU_REAL')),
             ('e2e_r50c', lambda: t_sub([str(T / 'r50_e2e.py'), 'r50c', 'R50_A_CC_R31H1001_PRESENT_ONLY_frozen_NCE,R50_A_CC_R31H1001_PRESENT_ONLY_both_joint_J'], 'R50_E2E'))):
    check(n, f)
out = dict(part='R50', job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()),
           tolerances=dict(vector_same_device=1e-4, vector_cpu_vs_gpu_cache=5e-4, mean=1e-6, score=1e-4, near_tie_score=1e-6, amendment='R50_AMENDMENT_A1.json'))
(W / 'CPU_TEST_R50.json').write_text(json.dumps(out, indent=1, default=str)); print('R50_CPU_TEST ALL_PASS', out['ALL_PASS'])
sys.exit(0 if out['ALL_PASS'] else 1)
