"""R41-F CPU checks (real data): code derivation r41c vs r40c (registered R41F_DIFF_VS_R40C.patch, rest byte-identical); configs equal to the R39
CC N-JOINT configs except the declared fields and lr_encoder 1e-6; CPU 2-update production run_unit of the J unit (step-0 student/cache equivalence,
R38 graph check, R40 one-step rule, R38 + R39 reference identity). Writes CPU_TEST_R41F.json. Usage: python -I -B tests/cpu_test_r41f.py"""
import difflib, json, os, sys, time, traceback
from pathlib import Path
W = Path(__file__).resolve().parents[1]; R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
sys.path.insert(0, str(W / 'r41c'))
RES = {}


def sh(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(n, fn):
    t0 = time.time()
    try:
        v = fn(); RES[n] = dict(passed=bool(v.get('passed')), detail=v, seconds=time.time() - t0)
    except Exception:
        RES[n] = dict(passed=False, error=traceback.format_exc(), seconds=time.time() - t0)
    print(n, RES[n]['passed'], flush=True)
    if not RES[n]['passed']:
        print(RES[n].get('error') or json.dumps(RES[n].get('detail'), default=str)[:4000], flush=True)


def t_code():
    ch = ('common.py', 'train_unit.py', 'r40ext.py')
    same = {f.name: sh(f) == sh(W / 'r41c' / f.name) for f in (W / 'r40c').glob('*.py') if f.name not in ch}
    patch = ''.join(''.join(difflib.unified_diff((W / 'r40c' / f).read_text().splitlines(True), (W / 'r41c' / f).read_text().splitlines(True))) for f in ch)
    strip = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++', '@@'))]
    ok = strip(patch) == strip((W / 'R41F_DIFF_VS_R40C.patch').read_text())
    files = sorted(f.name for f in (W / 'r41c').glob('*.py')) == sorted(f.name for f in (W / 'r40c').glob('*.py'))
    return dict(passed=all(same.values()) and ok and files, byte_identical=same, diff_equals_registered=ok)


def t_configs():
    out = dict(bad=[])
    for u in json.loads((W / 'configs/UNITS_R41F.json').read_text()):
        c = json.loads((W / 'configs' / f'{u}.json').read_text()); t = json.loads((R39 / 'configs' / f"{c['r39_template']}.json").read_text())
        allowed = {'unit_id', 'package_unit', 'block', 'line', 'code_tree', 'runner', 'adaptation', 'student_modalities', 'r39_template', 'optimizer', 'joint_rule', 'amendment',
                   'reference_joint', 'teacher_parents', 'job_bundle'}
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - allowed
        ok = not diff and c['optimizer'] == dict(t['optimizer'], lr_encoder=1e-6) and t['optimizer']['lr_encoder'] == 1e-5 and c['reference_unit'] == t['reference_unit']
        ok &= {k: v for k, v in c['joint_rule'].items() if k != 'lr'} == {k: v for k, v in t['joint_rule'].items() if k != 'lr'}
        if not ok:
            out['bad'].append((u, sorted(diff)))
    out['passed'] = not out['bad']
    return out


def t_e2e():
    import torch
    import common as C; C.install_guards('trainer'); import train_unit as TU
    TU.set_determinism()
    u = 'R41_F_CC31_H1001_N_joint_enclr0p1_J'; cfg = TU.cfg_of(u); cfg['_config_sha256'] = sh(W / 'configs' / f'{u}.json')
    ctx = TU.Ctx(cfg['cond'], torch.device('cpu'), need_local=False); ctx.ensure_joint()
    od = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/cpu_scratch_r41') / f"{u}_{os.environ.get('SLURM_JOB_ID', 'local')}_{int(time.time())}"
    r = TU.run_unit(dict(cfg, phase=1), ctx, out_dir=od, stop_at=2, no_readout=True)
    h = r['hist']; o40 = h['r40']['one_step_check']
    lr_enc = [g['lr'] for g in TU.make_opt(cfg, r['heads'], r['nulls'], r['stud']).param_groups]
    rec = dict(one_step=o40, references=h['r40']['references'], r38_chain=h['r38']['reference_chain_equal_at'], eq=h['r39']['step0_cache_equivalence']['passed'],
               graph=h['r38']['step1_graph_check']['passed'], opt_group_lrs=lr_enc)
    rec['passed'] = bool(o40['passed'] and rec['eq'] and rec['graph'] and rec['r38_chain'] == [0] and lr_enc == [1e-4, 1e-6]
                         and all(v['init_equal'] and v['student_init_equal'] and v['chain_equal_at'] == [0] for v in rec['references'].values()))
    return rec


check('code_derivation', t_code); check('configs', t_configs); check('e2e_J', t_e2e)
out = dict(job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()))
(W / 'CPU_TEST_R41F.json').write_text(json.dumps(out, indent=1, default=str)); print('R41F_CPU_TEST ALL_PASS', out['ALL_PASS'])
