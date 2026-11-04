"""R44 CPU checks on real data (Slurm CPU job) before PROTOCOL_FROZEN_R44.json. Writes CPU_TEST_R44.json.
  package_reference_tests  the package tests/test_routing.py (12 reference checks) run unchanged (python -I, package path as argument)
  reference_vs_production  on small synthetic problems (J / BCE / NCE, 2 directions, 33 slots): the production loss functions (objectives.loss_list /
                           loss_list_masked, the r44f route_backward weighting) give the same head / encoder gradients as reference/gradient_routing.py
                           (routed_gradients), max abs <= 1e-12 in float64; random-mask expectation: exact enumeration of all 2^n Bernoulli(p) masks of the
                           production masked loss recovers the ALL encoder gradient (<= 1e-12)
  code_derivation          r44f == r42f (minus the oracle runner / oracle_truth) and r44c == r41c byte for byte except common.py / train_unit.py; the
                           train_unit diffs are the tools/r44_patch.py output (re-applied to the source and compared byte for byte); diffs registered
  configs                  every unit = its base config except the declared R44 fields; A seeds 2002 / [2002, 3] / 16000 / 9 readouts; B routing; C
                           N_PRESENT_ONLY; in-unit references hash-bound; unit ids = package MATRIX
  e2e_r44f / e2e_r44c      2-update CPU production run_unit (see tests/e2e_r44.py)
Usage: python -I -B tests/cpu_test_r44.py"""
from __future__ import annotations
import difflib, hashlib, itertools, json, os, subprocess, sys, time, traceback, csv
from pathlib import Path

W = Path(__file__).resolve().parents[1]
PKG = Path('/home/infres/yinwang/CS_QMI/r44_plan/VCS_R39X_R42_Review_R44_Plan_20261103')
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


sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rj = lambda p: json.loads(Path(p).read_text())
STRIP = lambda t: [l for l in t.splitlines() if not l.startswith(('---', '+++', '@@'))]


def t_pkg_tests():
    r = subprocess.run([PY, '-I', '-B', str(PKG / 'tests' / 'test_routing.py')], capture_output=True, text=True, timeout=600, cwd=str(W))
    return dict(passed=r.returncode == 0 and 'OK' in r.stderr.splitlines()[-1], tail=r.stderr[-600:], manifest_sha=sh(PKG / 'MANIFEST.sha256'))


def t_ref_vs_prod():
    import torch
    sys.path.insert(0, str(PKG / 'reference')); import gradient_routing as GR; sys.path.pop(0)
    sys.path.insert(0, str(W / 'r44f')); from objectives import loss_list, loss_list_masked, list_rows; sys.path.pop(0)
    out = {}
    torch.manual_seed(0)
    for obj in ('J', 'BCE', 'NCE'):
        n, K, De = 6, 33, 5
        e = torch.randn(De, dtype=torch.float64, requires_grad=True); hw = torch.randn(De, dtype=torch.float64, requires_grad=True)
        X = [torch.randn(n, K, De, dtype=torch.float64) for _ in range(2)]
        A = []
        for d in range(2):
            a = torch.zeros(n, K, dtype=torch.float64)
            for i in range(n):
                a[i, (i * 7 + d) % K] = 1.0
            A.append(a)
        G = [(torch.tanh(x @ e) * hw.sum() + x @ hw * .3) for x in X]          # g depends on both blocks (encoder e feeds the head)
        M = [torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.float64), torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.float64)]
        P = [0.7, 0.45]
        # production weighting (one peer): L_h = .5 loss_list(i2t) + .5 loss_list(t2i); L_e = .5 masked(i2t) + .5 masked(t2i)
        Lh = sum(.5 * loss_list(obj, g, a) for g, a in zip(G, A)); Le = sum(.5 * loss_list_masked(obj, g, a, m, p) for g, a, m, p in zip(G, A, M, P))
        ph = torch.autograd.grad(Lh, [hw], retain_graph=True)[0]; pe = torch.autograd.grad(Le, [e], retain_graph=True)[0]
        rows = [(-list_rows(g, a, obj) if obj == 'J' else list_rows(g, a, obj)) for g, a in zip(G, A)]
        ref_rows = [GR.row_losses(g, a, obj) for g, a in zip(G, A)]
        # random-mask expectation: exact enumeration over all 2^n masks per direction (production masked loss)
        full_e = torch.autograd.grad(sum(.5 * loss_list(obj, g, a) for g, a in zip(G, A)), [e], retain_graph=True)[0]
        exp_e = torch.zeros_like(e)
        for d in range(2):
            acc = torch.zeros_like(e)
            for bits in itertools.product((0, 1), repeat=n):
                m = torch.tensor(bits, dtype=torch.float64); w = P[d] ** m.sum() * (1 - P[d]) ** (n - m.sum())
                acc = acc + w * torch.autograd.grad(.5 * loss_list_masked(obj, G[d], A[d], m, P[d]), [e], retain_graph=True)[0]
            exp_e = exp_e + acc
        rh, re_ = GR.routed_gradients([GR.DirectionRows(r, m, p, .5) for r, m, p in zip(ref_rows, M, P)], [hw], [e])   # last: it frees the graph
        out[obj] = dict(head_max_abs=float((ph - rh[0]).abs().max()), enc_max_abs=float((pe - re_[0]).abs().max()),
                        rows_max_abs=max(float((a - b).abs().max()) for a, b in zip(rows, ref_rows)), random_expectation_max_abs=float((exp_e - full_e).abs().max()))
        out[obj]['passed'] = all(v <= 1e-12 for v in out[obj].values())
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def udiff(a, b):
    return ''.join(difflib.unified_diff(Path(a).read_text().splitlines(True), Path(b).read_text().splitlines(True)))


def t_code():
    out = {}
    for new, old in (('r44f', 'r42f'), ('r44c', 'r41c')):
        files = sorted(p.name for p in (W / new).glob('*.py'))
        exp = sorted(p.name for p in (W / old).glob('*.py') if p.name not in ('train_oracle.py', 'oracle_truth.py'))
        same = {f: sh(W / new / f) == sh(W / old / f) for f in files if f not in ('common.py', 'train_unit.py')}
        tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f'r44_patch_check_{new}_{os.getpid()}.py'
        tmp.write_text((W / old / 'train_unit.py').read_text())
        r = subprocess.run([PY, '-I', '-B', str(W / 'tools' / 'r44_patch.py'), str(tmp)], capture_output=True, text=True)
        patch_ok = r.returncode == 0 and sh(tmp) == sh(W / new / 'train_unit.py')
        reg = (W / f'R44_DIFF_{new.upper()}.patch').read_text()
        cur = udiff(W / old / 'common.py', W / new / 'common.py') + udiff(W / old / 'train_unit.py', W / new / 'train_unit.py')
        out[new] = dict(file_set=files == exp, byte_identical=same, train_unit_is_patch_output=patch_ok, diff_registered=STRIP(cur) == STRIP(reg))
        out[new]['passed'] = out[new]['file_set'] and all(same.values()) and patch_ok and out[new]['diff_registered']
    out['passed'] = all(v['passed'] for v in out.values())
    return out


ALLOWED = {'unit_id', 'block', 'line', 'code_tree', 'runner', 'oracle', 'test_read', 'label', 'authorisation', 'reference_unit', 'reference_joint', 'reference_note',
           'adaptation', 'student_modalities', 'encoder', 'job_bundle', 'package_unit', 'package_block', 'joint_rule', 'r39_template', 'teacher_parents', 'extension', 'matrix_source',
           'evaluation', 'architecture_rule'}


def t_configs():
    mat = {r['unit_id']: r for r in csv.DictReader(open(PKG / 'configs' / 'MATRIX.csv'))}
    units = rj(W / 'configs' / 'UNITS_R44.json'); out = dict(n=len(units), bad=[])
    R39C = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/configs')
    base_of = {'n00': R39C / 'R39X_Flickr00_R30H_s1001_N_joint_global_{o}.json', 'n60': R39C / 'R39_Flickr60_N_joint_global_{o}.json',
               'f80in': W / 'configs' / 'R41_C_F80_IN2R_H1001_N_joint_{o}.json', 'cc': R39C / 'R39_CC152K_N_joint_global_{o}.json'}
    for u in units:
        c = rj(W / 'configs' / f'{u}.json'); o = c['objective']; m = mat[u]
        if c['block'] == 'A':
            pk = '_'.join(u.split('_')[2:5]); t = rj(W / 'configs' / f'R41_A_{pk}_N_joint_16k_{o}.json')
            extra = {'init_seed', 'sampler_seed', 'seed_rule', 'head_init', 'ckpt_steps', 'primary_endpoint', 'budget_rule'}
            ok = c['init_seed'] == 2002 and c['sampler_seed'] == [2002, 3] and c['steps'] == 16000 and c['ckpt_steps'] == [0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000]
            ok &= ((c['reference_joint'] or {}).get('mode') == 'prefix') == (pk == 'F60_R30_H1001') and c['supervision'] == 'OBS_MISSING_NULL' and not c.get('routing')
        else:
            t = rj(Path(str(base_of[c['cond']]).format(o=o)))
            extra = {'supervision', 'supervision_label', 'supervision_rule', 'routing'}
            ok = c['steps'] == 8000 and c['init_seed'] == 1001 and c['sampler_seed'] == [1001, 3] and c['ckpt_steps'] == [0, 250, 500, 1000, 2000, 4000, 8000]
            ok &= (c.get('routing') == m['arm']) if c['block'] == 'B' else (c['supervision'] == 'N_PRESENT_ONLY' and not c.get('routing'))
            ok &= c['reference_joint']['unit_id'] == Path(t['unit_id']).name and 'mode' not in c['reference_joint']
            ok &= c['code_tree'] == ('r44c' if c['cond'] == 'cc' else 'r44f')
        diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)} - ALLOWED - extra
        ok &= not diff and c['objective'] == m['objective'] and c['steps'] == int(m['steps']) and c['init_seed'] == int(m['training_seed']) and c['test_read'] is False
        ok &= c['adaptation'] == 'both_joint' and c['encoder'] == 'joint_global' and c['optimizer'] == t['optimizer'] and c['optimizer']['schedule'] == 'cosine'
        if c.get('reference_joint'):
            r = c['reference_joint']; ok &= sh(r['result_json']) == r['result_json_sha256'] and sh(r['train_history_json']) == r['train_history_json_sha256'] and sh(r['ckpt_step0']) == r['ckpt_step0_sha256']
        if not ok:
            out['bad'].append((u, sorted(diff)))
    out['passed'] = out['n'] == 39 and sorted(units) == sorted(mat) and not out['bad']
    return out


def t_e2e(tree, units):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'e2e_r44.py'), tree, ','.join(units)], capture_output=True, text=True, timeout=14400, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('R44_E2E ')]
    out = json.loads(line[-1].split(' ', 1)[1]) if line else dict(failed=dict(passed=False, stdout=r.stdout[-3000:], stderr=r.stderr[-6000:]))
    out['passed'] = r.returncode == 0 and all(v['passed'] for v in out.values() if isinstance(v, dict))
    return out


def main():
    for n, f in (('package_reference_tests', t_pkg_tests), ('reference_vs_production', t_ref_vs_prod), ('code_derivation', t_code), ('configs', t_configs),
                 ('e2e_r44f', lambda: t_e2e('r44f', ['R44_B_F80_IN2R_H1001_ROUTE_PRESENT_J_train1001', 'R44_B_F00_R30_H1001_ROUTE_RANDOM_BCE_train1001',
                                                    'R44_C_F00_R30_H1001_PRESENT_ALL_NCE_train1001', 'R44_A_F60_R30_H1001_ALL_J_train2002'])),
                 ('e2e_r44c', lambda: t_e2e('r44c', ['R44_B_CC_R31_H1001_ROUTE_PRESENT_NCE_train1001']))):
        check(n, f)
    out = dict(part='R44', job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES,
               ALL_PASS=all(v['passed'] for v in RES.values()))
    (W / 'CPU_TEST_R44.json').write_text(json.dumps(out, indent=1, default=str))
    print('R44_CPU_TEST ALL_PASS', out['ALL_PASS'])


if __name__ == '__main__':
    main()
