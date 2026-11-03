"""R41 CPU checks on real data (Slurm CPU job). part 1 (A/B/D/E, before PROTOCOL_FROZEN_R41.json); part 2 (C, after the new-parent prep,
before PROTOCOL_FROZEN_R41C.json). Writes CPU_TEST_R41_<part>.json.
  code_derivation  r41f == r40f byte for byte except common/data35/r39ext/r40ext/train_unit, whose diff equals the registered
                   R41_DIFF_VS_R40.patch; data35.py == R39X r39xf/data35.py; oracle_truth.py == R39
  oracle_isolation R39 oracle selftest from r41f; oracle_truth refused in a trainer-role process
  configs          per block: recipe fields equal to the R39 / R39X / R38 template except the declared R41 fields; references hash-bound;
                   prefix references only on the 16000-update cells; B seeds 2002
  lr_schedule      16000-update stretched cosine: lr(0) = base, lr(15999) = 0.1 base, lr(8000) on the stretched curve; 8000-update unchanged
  e2e_part1        CPU 2-update production run_unit of A (n00 16k J, prefix reference), B (seed 2002 NCE), E (s1003 image_only BCE)
  e2e_oracle_part1 same inside a separate oracle-runner process: D joint J and D frozen NCE on s1002
  part 2: new-parent manifests (exact native DEV reproduction, prep, JOINT equivalence <= 1e-4) and e2e of C frozen J / joint BCE (f40in),
          C joint NCE (f80in)
Usage: python -I -B tests/cpu_test_r41.py 1|2"""
from __future__ import annotations
import difflib, json, math, os, subprocess, sys, time, traceback
from pathlib import Path

W = Path(__file__).resolve().parents[1]
R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
SCRATCH = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/cpu_scratch_r41')
PY = sys.executable
RES = {}


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
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


sys.path.insert(0, str(W / 'r41f'))


def t_code_derivation():
    changed = ('common.py', 'data35.py', 'r39ext.py', 'r40ext.py', 'train_unit.py')
    same = {f.name: sh(f) == sh(W / 'r41f' / f.name) for f in sorted((W / 'r40f').glob('*.py')) if f.name not in changed}
    files_equal = sorted(f.name for f in (W / 'r41f').glob('*.py')) == sorted(f.name for f in (W / 'r40f').glob('*.py'))
    patch = ''
    for f in ('common', 'r39ext', 'train_unit', 'r40ext', 'data35'):
        a = (W / 'r40f' / f'{f}.py').read_text().splitlines(True); b = (W / 'r41f' / f'{f}.py').read_text().splitlines(True)
        patch += ''.join(difflib.unified_diff(a, b, f'r40f/{f}.py', f'r41f/{f}.py'))
    reg = (W / 'R41_DIFF_VS_R40.patch').read_text()
    strip = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++', '@@'))]
    out = dict(byte_identical=same, file_set_equal=files_equal, diff_equals_registered_patch=strip(patch) == strip(reg),
               data35_equals_r39xf=sh(W / 'r41f' / 'data35.py') == sh(R39 / 'r39xf' / 'data35.py'), oracle_truth_equals_r39=sh(W / 'r41f' / 'oracle_truth.py') == sh(R39 / 'r39f' / 'oracle_truth.py'))
    out['passed'] = all(same.values()) and files_equal and out['diff_equals_registered_patch'] and out['data35_equals_r39xf'] and out['oracle_truth_equals_r39']
    return out


def t_oracle_isolation():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([PY, '-I', '-B', str(W / 'r41f' / 'train_oracle.py'), '--selftest'], capture_output=True, text=True, timeout=600, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('R39_ORACLE_SELFTEST')]
    d = json.loads(line[0].split(' ', 1)[1]) if line else {}
    blocked = subprocess.run([PY, '-I', '-B', '-c', f"import sys; sys.path.insert(0,'{W / 'r41f'}'); import common; common.install_guards('trainer'); import oracle_truth"], capture_output=True, text=True, env=env)
    return dict(passed=bool(d.get('ALL_PASS')) and blocked.returncode != 0 and 'oracle isolation' in blocked.stderr, selftest=d)


def t_configs(part):
    import common as C
    units = json.loads((W / 'configs' / ('UNITS_R41.json' if part == '1' else 'UNITS_R41C.json')).read_text())
    allowed = {'unit_id', 'package_unit', 'block', 'line', 'code_tree', 'runner', 'oracle', 'adaptation', 'student_modalities', 'encoder', 'test_read', 'label', 'job_bundle',
               'optimizer', 'joint_rule', 'evaluation', 'reference_unit', 'reference_joint', 'supervision', 'supervision_label', 'supervision_rule', 'extension', 'teacher_parents',
               'package_block', 'matrix_source', 'r39_template', 'reference_note', 'architecture_rule'}
    out = dict(n=len(units), bad=[])
    for u in units:
        c = json.loads((W / 'configs' / f'{u}.json').read_text()); b = c['block']
        extra = dict(A={'steps', 'ckpt_steps', 'primary_endpoint', 'schedule_rule', 'budget_rule'}, B={'init_seed', 'sampler_seed', 'seed_rule', 'head_init'},
                     C={'cond', 'noise_percent', 'parent_seed', 'parent_H', 'parent_key', 'dataset'}, D=set(), E=set())[b]
        if b in ('A', 'B', 'E') or (b == 'D' and c['adaptation'] != 'frozen'):
            tid = (c.get('reference_joint') or {}).get('unit_id') or {'n60': f"R39_Flickr60_N_joint_global_{c['objective']}"}.get(c['cond'])
            t = json.loads((R39 / 'configs' / f'{tid}.json').read_text())
        else:
            t = json.loads((Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/configs') / f"r38_B_f60_R30_s{'1002' if b == 'D' else '1001'}_N_shared_{c['objective']}_cosine.json").read_text())
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - allowed - extra
        ok = not diff and c['code_tree'] == 'r41f' and c['test_read'] is False and (c['encoder'] == 'joint_global') == ('lr_encoder' in c['optimizer'])
        ok &= (c['steps'] == 16000 and c['ckpt_steps'][-1] == 16000 and c['reference_joint']['mode'] == 'prefix') if b == 'A' else (c['steps'] == 8000 and 'mode' not in (c.get('reference_joint') or {}))
        ok &= (c['init_seed'] == 2002 and c['sampler_seed'] == [2002, 3]) if b == 'B' else (c['init_seed'] == 1001 and c['sampler_seed'] == [1001, 3])
        ok &= c['oracle'] == (b == 'D') and (c['supervision'] == 'ORACLE_MISSING') == (b == 'D')
        for k in ('reference_unit', 'reference_joint'):
            if c.get(k):
                ok &= sh(c[k]['result_json']) == c[k]['result_json_sha256'] and sh(c[k]['train_history_json']) == c[k]['train_history_json_sha256']
        if not ok:
            out['bad'].append((u, sorted(diff)))
    out['passed'] = out['n'] == (27 if part == '1' else 12) and not out['bad']
    return out


def t_lr():
    import r38ext as X
    m = lambda t, T: X.lr_multiplier(t, T, 'cosine', 0.1)
    out = dict(m0_16k=m(0, 16000), mlast_16k=m(15999, 16000), m8000_16k=m(8000, 16000), mlast_8k=m(7999, 8000), m4000_8k=m(4000, 8000))
    out['passed'] = out['m0_16k'] == 1.0 and abs(out['mlast_16k'] - .1) < 1e-15 and abs(out['m8000_16k'] - out['m4000_8k']) < 1e-3 and abs(out['mlast_8k'] - .1) < 1e-15
    return out


def t_e2e(units):
    import torch
    import common as C; C.install_guards('trainer'); import train_unit as TU
    TU.set_determinism()
    dev = torch.device('cpu'); ctxs = {}; out = {}
    for u in units:
        cfg = TU.cfg_of(u); cfg['_config_sha256'] = sh(W / 'configs' / f'{u}.json')
        if cfg['cond'] not in ctxs:
            ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], dev, need_local=False)}
        ctx = ctxs[cfg['cond']]
        if cfg['encoder'] == 'joint_global':
            ctx.ensure_joint()
        od = SCRATCH / f'{u}_{os.environ.get("SLURM_JOB_ID", "local")}_{int(time.time())}'
        res = TU.run_unit(dict(cfg, phase=1), ctx, pool=None, out_dir=od, stop_at=2, no_readout=True, oracle=None)
        h = res['hist']; r40 = h['r40']
        rec = dict(one_step_r40=r40.get('one_step_check'), references=r40.get('references'), r38_chain=h['r38']['reference_chain_equal_at'],
                   step0_cache_equivalence=(h['r39'].get('step0_cache_equivalence') or {}).get('passed'), step1_graph_check=h['r38'].get('step1_graph_check', {}).get('passed'),
                   lr_update0=h['r38']['step1_graph_check']['lr_update0'], lr_last_update=h['r38']['step1_graph_check']['lr_last_update'])
        rec['passed'] = bool(rec['one_step_r40'] and rec['one_step_r40']['passed'] and rec['step1_graph_check'] and rec['r38_chain'] == ([0] if cfg.get('reference_unit') else [])
                             and all(v['init_equal'] and v['student_init_equal'] in (None, True) and v['chain_equal_at'] == [0] for v in (rec['references'] or {}).values())
                             and (cfg['encoder'] != 'joint_global' or rec['step0_cache_equivalence']))
        out[u] = rec
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_e2e_oracle(units):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=''); out = {}
    for u in units:
        r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'train_oracle_cpu_e2e_r41.py'), u], capture_output=True, text=True, timeout=7200, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith('R41_ORACLE_E2E ')]
        out[u] = json.loads(line[-1].split(' ', 1)[1]) if line else dict(passed=False, stdout=r.stdout[-2000:], stderr=r.stderr[-4000:])
        out[u]['passed'] = bool(out[u].get('passed')) and r.returncode == 0
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_parents():
    import common as C
    out = {}
    for cond in ('f40in', 'f80in'):
        cm = json.loads((Path(C.CONDS[cond]['cache_dir']) / 'CACHE_MANIFEST.json').read_text())
        pm = json.loads((Path(C.CONDS[cond]['prep_dir']) / 'PREP_MANIFEST.json').read_text())
        jm = json.loads((Path(C.JOINT_PARENTS[cond]['prep_dir']) / 'JOINT_PREP_MANIFEST.json').read_text())
        out[cond] = dict(native_rsum=cm['native_dev']['rsum'], checkpoint_best_rsum=cm['native_dev']['checkpoint_best_rsum'], exact=cm['native_dev']['reproduces_checkpoint_best_rsum_exactly'],
                         cache_rescore_identical=cm['native_dev']['cache_rescore_identical'], a0b0=pm['calibration_init'][cond], candidates=pm['candidates'][cond],
                         joint_equivalence_max_abs=jm['equivalence_summary']['max_abs_over_all'], dev_from_student=jm['equivalence']['dev_native_from_student_vectors'])
        out[cond]['passed'] = out[cond]['exact'] and out[cond]['cache_rescore_identical'] and out[cond]['joint_equivalence_max_abs'] <= 1e-4 and out[cond]['a0b0']['a0'] > 0
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def main():
    part = sys.argv[1]
    if part == '1':
        for n, f in (('code_derivation', t_code_derivation), ('oracle_isolation', t_oracle_isolation), ('configs', lambda: t_configs('1')), ('lr_schedule', t_lr),
                     ('e2e_oracle_part1', lambda: t_e2e_oracle(['R41_D_F60_R30_H1002_ORACLE_joint_J', 'R41_D_F60_R30_H1002_ORACLE_frozen_NCE'])),
                     ('e2e_part1', lambda: t_e2e(['R41_A_F00_R30_H1001_N_joint_16k_J', 'R41_B_F60_R30_H1001_seed2002_N_joint_NCE', 'R41_E_F60_R30_H1003_image_only_BCE']))):
            check(n, f)
    else:
        for n, f in (('code_derivation', t_code_derivation), ('configs', lambda: t_configs('2')), ('new_parents', t_parents),
                     ('e2e_part2', lambda: t_e2e(['R41_C_F40_IN2R_H1001_N_frozen_J', 'R41_C_F40_IN2R_H1001_N_joint_BCE', 'R41_C_F80_IN2R_H1001_N_joint_NCE']))):
            check(n, f)
    out = dict(part=part, job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()))
    (W / f'CPU_TEST_R41_{part}.json').write_text(json.dumps(out, indent=1, default=str))
    print('R41_CPU_TEST', part, 'ALL_PASS', out['ALL_PASS'])


if __name__ == '__main__':
    main()
