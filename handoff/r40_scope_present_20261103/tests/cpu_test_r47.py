"""R47 CPU checks on real data (before any formal R47 unit; writes CPU_TEST_R47.json):
  code_derivation  r47f / r47o / r47c = r42f / r43o / r41c byte for byte except common.py, r39ext.py, train_unit.py; r39ext / train_unit = tools/r47_patch.py
                   output on the source (r47c: r42f per-cond r39ext + the two per-cond lineage lines of r42f in train_unit); diffs = R47_DIFF_*.patch
  lf_prep          layout-free caches: H_lf (cache manifest) vs the corrected-readout parent H under global centring (corrected/results, same parent);
                   prep / JOINT manifests hash-bound; G JOINT centring = global means; candidate coverage
  configs          each unit = the condition's N-JOINT config except the declared R47 fields
  e2e              2-update CPU production run_unit: n60 G J, n60 S BCE (r47f); cc G NCE, cc S J (r47c)
Usage: python -I -B tests/cpu_test_r47.py"""
import difflib, glob, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
W = Path(__file__).resolve().parents[1]; PY = sys.executable; RES = {}
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
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
    out = {}
    for new, old in (('r47f', 'r42f'), ('r47o', 'r43o'), ('r47c', 'r41c')):
        ch = ('common.py', 'r39ext.py', 'train_unit.py')
        files = sorted(p.name for p in (W / new).glob('*.py')); exp = sorted(p.name for p in (W / old).glob('*.py') if p.name not in ('train_oracle.py', 'oracle_truth.py'))
        same = {f: sh(W / new / f) == sh(W / old / f) for f in files if f not in ch}
        tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f'r47chk_{new}_{os.getpid()}'; tmp.mkdir(parents=True, exist_ok=True)
        (tmp / 'r39ext.py').write_text((W / ('r42f' if new == 'r47c' else old) / 'r39ext.py').read_text())
        tu = (W / old / 'train_unit.py').read_text()
        if new == 'r47c':
            tu = tu.replace("parent_ckpt=str(R9.C.PARENT_CKPT), parent_sha256", "parent_ckpt=str(ctx.joint.ckpt_path), parent_sha256").replace(
                "parent_ckpt_sha256_unchanged=sha256(R9.C.PARENT_CKPT) == lineage", "parent_ckpt_sha256_unchanged=sha256(ctx.joint.ckpt_path) == lineage")
        (tmp / 'train_unit.py').write_text(tu)
        r = subprocess.run([PY, '-I', '-B', str(W / 'tools' / 'r47_patch.py'), str(tmp)], capture_output=True, text=True)
        po = r.returncode == 0 and all(sh(tmp / f) == sh(W / new / f) for f in ('r39ext.py', 'train_unit.py'))
        cur = ''.join(''.join(difflib.unified_diff((W / old / f).read_text().splitlines(True), (W / new / f).read_text().splitlines(True))) for f in ch)
        out[new] = dict(file_set=files == exp, byte_identical=same, patch_output=po, diff_registered=STRIP(cur) == STRIP((W / f'R47_DIFF_{new.upper()}.patch').read_text()))
        out[new]['passed'] = out[new]['file_set'] and all(same.values()) and po and out[new]['diff_registered']
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_lf():
    # reference parent-H (global centring) values keyed by (cond, GPU name): corrected-readout groups and the R44 audit (parent fresh student, global
    # student cosine). The layout-free H must equal a SAME-GPU-type reference exactly (CPU 1085587: n60 RTX build 496.627 vs L40S corrected 496.529,
    # one near-tie query; the RTX R44 audit gives 496.627); with only other-GPU references it must agree within 0.2 rSum (recorded cross_gpu_only)
    cr = {}
    for f in glob.glob(str(W / 'corrected' / 'results' / '*.json')):
        x = rj(f); cr.setdefault(x['cond'], {})[x['gpu']] = x['parent_H']['global']
    for b in ('n60', 'n00', 'f80in'):
        a = rj(W / f'R44_AUDIT_{b}.json'); cr.setdefault(b, {}).setdefault(a['gpu'], a['units']['parent_fresh_student']['global']['student_cosine_rsum'])
    out = {}
    for b in ('n60', 'f80in', 'n00', 'c60in_s1001', 'cc'):
        cm = rj(OUT / 'r47_parents/cache' / b / 'CACHE_MANIFEST.json'); pm = rj(OUT / 'r47_parents/prep' / b / 'PREP_MANIFEST.json')
        jg = rj(OUT / 'r47_parents/joint' / f'{b}_G' / 'JOINT_PREP_MANIFEST.json'); js = rj(OUT / 'r47_parents/joint' / f'{b}_S' / 'JOINT_PREP_MANIFEST.json')
        ref = cr.get(b) or {}; g = cm['env']['gpu']
        r = dict(H_lf=cm['native_dev']['rsum'], H_native=cm['native_dev']['native_layout_rsum'], build_gpu=g, references_global=ref, corrected_parent_H_global=ref.get(g),
                 files_ok=all(sh(OUT / 'r47_parents/cache' / b / n) == v['sha256'] for n, v in cm['files'].items()) and all(sh(OUT / 'r47_parents/prep' / b / n) == v['sha256'] for n, v in pm['files'].items()),
                 G_mode=jg['centring_mode'], S_mode=js['centring_mode'], G_eq=jg['equivalence_summary']['max_abs_over_all'], a0=pm['calibration_init'][b]['a0'], coverage=pm['candidates'][b])
        if g in ref:
            r['H_matches_corrected'] = abs(r['H_lf'] - ref[g]) < 1e-6
        elif ref:
            r['cross_gpu_only'] = True; r['H_matches_corrected'] = all(abs(r['H_lf'] - v) <= 0.2 for v in ref.values())
        else:
            r['H_matches_corrected'] = None
        r['passed'] = r['files_ok'] and r['G_mode'] == 'global' and r['S_mode'] == 'train_batch' and r['G_eq'] == 0.0 and r['a0'] > 0 and r['H_matches_corrected'] in (None, True)
        out[b] = r
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_configs():
    units = rj(W / 'configs/UNITS_R47.json'); out = dict(n=len(units), bad=[])
    allowed = {'unit_id', 'block', 'line', 'code_tree', 'runner', 'oracle', 'test_read', 'label', 'authorisation', 'reference_unit', 'reference_joint', 'reference_note', 'adaptation',
               'student_modalities', 'encoder', 'job_bundle', 'package_unit', 'package_block', 'joint_rule', 'r39_template', 'teacher_parents', 'extension', 'matrix_source', 'cond',
               'centring_variant', 'centring_rule', 'evaluation', 'architecture_rule'}
    for row in __import__('csv').DictReader(open(W / 'RESOLVED_MATRIX_R47.csv')):
        c = rj(W / 'configs' / f"{row['unit_id']}.json"); bd = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs') / row['base_config']
        t = rj(bd if bd.exists() else W / 'configs' / row['base_config'])
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - allowed
        ok = not diff and c['cond'] == f"{t['cond']}_lf{row['variant']}" and c['steps'] == 8000 and c['optimizer'] == t['optimizer'] and c['sampler_seed'] == [1001, 3] and c['init_seed'] == 1001
        if not ok:
            out['bad'].append((row['unit_id'], sorted(diff)))
    out['passed'] = out['n'] == 30 and not out['bad']
    return out


def t_e2e(tree, units):
    r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'e2e_r47.py'), tree, ','.join(units)], capture_output=True, text=True, timeout=14400, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    line = [l for l in r.stdout.splitlines() if l.startswith('R47_E2E ')]
    out = json.loads(line[-1].split(' ', 1)[1]) if line else dict(failed=dict(passed=False, stdout=r.stdout[-3000:], stderr=r.stderr[-6000:]))
    out['passed'] = r.returncode == 0 and all(v['passed'] for v in out.values() if isinstance(v, dict))
    return out


for n, f in (('code_derivation', t_code), ('lf_prep', t_lf), ('configs', t_configs),
             ('e2e_r47f', lambda: t_e2e('r47f', ['R47_G_F60_R30_H1001_N_joint_J', 'R47_S_F60_R30_H1001_N_joint_BCE'])),
             ('e2e_r47c', lambda: t_e2e('r47c', ['R47_G_CC31_H1001_N_joint_NCE', 'R47_S_CC31_H1001_N_joint_J']))):
    check(n, f)
out = dict(part='R47', job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()))
(W / 'CPU_TEST_R47.json').write_text(json.dumps(out, indent=1, default=str)); print('R47_CPU_TEST ALL_PASS', out['ALL_PASS'])
