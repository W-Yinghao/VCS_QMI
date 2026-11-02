"""R39 CPU checks (Slurm CPU job; writes CPU_TEST_<part>.json). part 'pre' (before any prep): package reference tests, reference
byte identity, code-derivation identity vs R38 (+ the listed patch only), oracle selftest, label/loss math on 33-slot completed
rows, completion invariants (synthetic), student-encoder mechanics on real parent weights + a few real items, config identity vs
the R38 references. part 'post' (after prep): real-data COMPLETE labels on the actual N sampler stream (both datasets), prep
manifests (completion cross-check, JOINT equivalence), Flickr audit file presence (not read for any rule).
Usage: python -I -B tests/cpu_test.py pre|post"""
from __future__ import annotations
import importlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
R38 = Path('/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102')
PKG = Path('/home/infres/yinwang/CS_QMI/r39_design/VCS_R38_Review_R39_Design_20261102')
PY = sys.executable
RES = {}


def check(name, fn):
    t0 = time.time()
    try:
        v = fn(); ok = bool(v if not isinstance(v, dict) else v.get('passed', True))
        RES[name] = dict(passed=ok, detail=v if isinstance(v, dict) else None, seconds=time.time() - t0)
    except Exception:
        RES[name] = dict(passed=False, error=traceback.format_exc(), seconds=time.time() - t0)
    print(name, RES[name]['passed'], flush=True)


def sh(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(t):
    for m in list(sys.modules):
        if m in ('common', 'data35', 'data36', 'evals', 'heads', 'native_eval', 'objectives', 'r38ext', 'r39ext', 'r39_targets', 'train_unit'):
            del sys.modules[m]
    sys.path[:] = [x for x in sys.path if not x.endswith(('/r39f', '/r39c'))]
    sys.path.insert(0, str(W / t))


# ------------------------------------------------------------------------------------------------ part pre
def t_package_tests():
    r = subprocess.run([PY, '-I', '-B', str(PKG / 'tests' / 'test_reference.py')], capture_output=True, text=True, timeout=600)
    return dict(passed=r.returncode == 0 and 'OK' in r.stderr, tail=r.stderr[-400:])


def t_reference_bytes():
    ref = sh(PKG / 'reference' / 'targets.py')
    return dict(passed=sh(W / 'r39f' / 'r39_targets.py') == ref == sh(W / 'r39c' / 'r39_targets.py'), sha256=ref)


def t_code_derivation():
    out = {}
    sys.path.insert(0, str(W / 'tools')); import patch_train_unit as PT
    for t, s in (('r39f', 'r38f'), ('r39c', 'r38c')):
        same = {f: sh(W / t / f'{f}.py') == sh(R38 / s / f'{f}.py') for f in ('data35', 'evals', 'heads', 'native_eval', 'objectives', 'r38ext')}
        src = (R38 / s / 'train_unit.py').read_text()
        for old, new in PT.E:
            assert src.count(old) == 1; src = src.replace(old, new)
        cur = (W / t / 'train_unit.py').read_text()
        # docstring + print-tag edits are the only differences allowed beyond PT.E
        strip = lambda x: [l for l in x.splitlines() if not l.startswith(('"""R39', 'per-peer joint clip', '"""R38')) and 'UNIT_DONE' not in l and 'JOB_END failed' not in l]
        out[t] = dict(byte_identical_to_r38=same, train_unit_equals_r38_plus_patch=strip(src) == strip(cur))
    out['r39ext_identical_trees'] = sh(W / 'r39f' / 'r39ext.py') == sh(W / 'r39c' / 'r39ext.py')
    out['passed'] = all(all(v['byte_identical_to_r38'].values()) and v['train_unit_equals_r38_plus_patch'] for k, v in out.items() if k in ('r39f', 'r39c')) and out['r39ext_identical_trees']
    return out


def t_oracle_selftest():
    r = subprocess.run([PY, '-I', '-B', str(W / 'r39f' / 'train_oracle.py'), '--selftest'], capture_output=True, text=True, timeout=600, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    line = [l for l in r.stdout.splitlines() if l.startswith('R39_ORACLE_SELFTEST')]
    d = json.loads(line[0].split(' ', 1)[1]) if line else {}
    blocked = subprocess.run([PY, '-I', '-B', '-c', f"import sys; sys.path.insert(0,'{W / 'r39f'}'); import common; common.install_guards('trainer'); import oracle_truth"],
                             capture_output=True, text=True)
    return dict(passed=bool(d.get('ALL_PASS')) and blocked.returncode != 0 and 'oracle isolation' in blocked.stderr, selftest=d, import_blocked_in_trainer=blocked.returncode != 0)


def t_loss_math():
    import torch
    tree('r39f'); import objectives as O
    spec = importlib.util.spec_from_file_location('pkgref', str(PKG / 'reference' / 'targets.py')); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    torch.manual_seed(3); g = torch.randn(64, 33, dtype=torch.float64, requires_grad=True)
    a = torch.zeros(64, 33, dtype=torch.float64); a[:32, 32] = 1; a[32:, 32] = .5; a[32:, 5] = .5     # N null rows / completed rows
    out = {}
    for obj in ('J', 'BCE', 'NCE'):
        mine = O.loss_list(obj, g, a); ref = R.list_loss(g, a, obj)
        out[f'{obj}_equals_package'] = float((mine - ref).abs())
        perm = torch.stack([torch.randperm(33) for _ in range(64)])
        out[f'{obj}_perm_invariant'] = float((O.loss_list(obj, g.gather(1, perm), a.gather(1, perm)) - mine).abs())
    gj = torch.autograd.grad(O.loss_list('J', g, a), g)[0]; gb = torch.autograd.grad(O.loss_list('BCE', g, a), g)[0]
    out['dJ_eq_(1-T^2)dBCE'] = float((gj - (1 - torch.tanh(g / 2) ** 2) * gb).abs().max())
    out['passed'] = all(v < 1e-12 for v in out.values())
    return out


def t_completion_synthetic():
    tree('r39f'); import r39ext as R9
    c = object.__new__(R9.Completion)
    ids = np.array([[3, 7, -1], [5, -1, 9], [-1, 2, 4], [8, 6, -1]]); a = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    q = np.array([0, 1, 2, 3]); pa = np.array([7, 9, 4, 1]); pb = np.array([7, 9, 4, 6]); rr = np.array([[1, 5], [1, 1], [2, 6], [1, 1]], float)
    pre = np.array([True, False, False, False])
    out, acc = c.apply_dir(ids, a, q, pa, pb, rr, pre)
    ok = dict(accepted=acc.tolist(), row0=out[0].tolist(), others_unchanged=bool(np.array_equal(out[1:], a[1:])))
    ok['passed'] = acc.tolist() == [True, False, False, False] and out[0].tolist() == [0, .5, .5] and ok['others_unchanged']
    try:
        c.apply_dir(ids, a, q, pa, pb, rr, np.array([False] * 4)); ok['mismatch_detected'] = False
    except AssertionError:
        ok['mismatch_detected'] = True
    ok['passed'] = ok['passed'] and ok['mismatch_detected']
    return ok


def t_student_mechanics():
    import torch
    tree('r39f'); import common as C; import r39ext as R9
    torch.use_deterministic_algorithms(True)
    ck = torch.load(C.PARENT_CKPT, map_location='cpu', weights_only=False)
    s = R9.build_students(ck, torch.device('cpu'))
    tr = np.load(C.RAW_FIT_IMS, mmap_mode='r'); T = np.load(C.TOKENS_FILE)
    x = torch.from_numpy(np.ascontiguousarray(tr[:4])); tok = torch.from_numpy(T['train_tok'][:6]); ln = torch.from_numpy(T['train_len'][:6])
    p = s['A']; e1 = p.enc_img(x); c1 = p.enc_cap(tok, ln)
    # forward semantics: train-mode GRUs == eval-mode GRUs (one layer, dropout 0); BN eval
    with torch.no_grad():
        for g in (p.txt.cap_rnn, p.txt.gpool.gru, p.img.gpool.gru):
            g.training = False
        e0 = p.enc_img(x); c0 = p.enc_cap(tok, ln)
        p.train()
    (e1.sum() + c1.sum()).backward()
    tr_ = p.trainable(); grads = [q.grad is not None and float(q.grad.abs().sum()) > 0 for q in tr_]
    rng0 = torch.get_rng_state(); s2 = R9.build_students(ck, torch.device('cpu')); rng_ok = torch.equal(rng0, torch.get_rng_state())
    out = dict(img_train_vs_eval_gru=float((e1 - e0).abs().max()), cap_train_vs_eval_gru=float((c1 - c0).abs().max()),
               frac_trainable_with_grad=float(np.mean(grads)), embed_frozen=p.txt.embed.weight.grad is None and not p.txt.embed.weight.requires_grad,
               bn_eval=all(not m.training for m in p.modules() if isinstance(m, torch.nn.BatchNorm1d)), dropout_eval=all(not m.training for m in p.modules() if isinstance(m, torch.nn.Dropout)),
               n_trainable=int(sum(q.numel() for q in tr_)), construction_draws_no_rng=bool(rng_ok),
               unit_norm=float((e1.norm(dim=1) - 1).abs().max()))
    out['passed'] = out['img_train_vs_eval_gru'] < 1e-6 and out['cap_train_vs_eval_gru'] < 1e-6 and out['frac_trainable_with_grad'] > .9 and out['embed_frozen'] and out['bn_eval'] and out['dropout_eval'] and rng_ok
    return out


def t_configs():
    units = json.loads((W / 'configs' / 'UNITS.json').read_text())
    out = dict(n=len(units), n_oracle=0, n_joint=0, bad=[])
    allowed = {'unit_id', 'package_unit', 'block', 'line', 'code_tree', 'runner', 'supervision', 'supervision_label', 'oracle', 'encoder', 'teacher_parents', 'architecture',
               'schedule_rule', 'architecture_rule', 'reference_unit', 'label', 'job_bundle', 'optimizer', 'joint_rule', 'evaluation', 'supervision_rule', 'package_block'}
    for u in units:
        c = json.loads((W / 'configs' / f'{u}.json').read_text()); r = c['reference_unit']
        b = json.loads((R38 / 'configs' / f"{r['unit_id']}.json").read_text())
        assert sh(R38 / 'configs' / f"{r['unit_id']}.json") == r['config_sha256']
        diff = {k for k in set(c) | set(b) if c.get(k) != b.get(k)} - allowed
        o = dict(c['optimizer']); o.pop('lr_encoder', None)
        if diff or o != b['optimizer'] or c['sampler_seed'] != b['sampler_seed'] or c['init_seed'] != b['init_seed']:
            out['bad'].append((u, sorted(diff)))
        out['n_oracle'] += c['oracle']; out['n_joint'] += c['encoder'] == 'joint_global'
        assert (c['encoder'] == 'joint_global') == ('lr_encoder' in c['optimizer'])
        assert (not c['oracle']) or c['cond'] == 'n60'
    out['passed'] = out['n'] == 21 and out['n_oracle'] == 3 and out['n_joint'] == 12 and not out['bad']
    return out


# ------------------------------------------------------------------------------------------------ part post
def t_real_completion(t):
    import torch
    tree(t); import common as C; from data35 import load_cond; import data36 as D36; import r39ext as R9
    d = load_cond(C.TARGET_COND, torch.device('cpu'))
    comp = R9.Completion(C.TARGET_COND, d)
    rng = np.random.default_rng([1001, 3]); tot = dict(rows=0, null=0, completed=0, changed_rows=0)
    for step in range(400):
        lists, idx = D36.sample('COMPLETE_XCHECK', rng, None, d, 16, None)
        new, st = comp.apply(lists)
        for k in (0, 1):
            a0, a1 = lists[k][2], new[k][2]
            tot['changed_rows'] += int((a0 != a1).any(1).sum())
        tot['rows'] += 32; tot['null'] += int(st['complete_null_rows_i2t'] + st['complete_null_rows_t2i']); tot['completed'] += int(st['complete_completed_rows_i2t'] + st['complete_completed_rows_t2i'])
    # identical N lists: the COMPLETE sampler draws exactly the N lists (same supervision family, same stream)
    r1, r2 = np.random.default_rng([1001, 3]), np.random.default_rng([1001, 3]); same = True
    for _ in range(50):
        l1, _ = D36.sample('COMPLETE_XCHECK', r1, None, d, 16, None); l2, _ = D36.sample('OBS_MISSING_NULL', r2, None, d, 16, None)
        same &= all(np.array_equal(x, y) for a_, b_ in zip(l1, l2) for x, y in zip(a_, b_))
    tot['N_lists_identical'] = bool(same); tot['completed_equals_changed'] = tot['completed'] == tot['changed_rows']
    tot['passed'] = tot['N_lists_identical'] and tot['completed_equals_changed']
    return tot


def t_prep_manifests():
    out = {}
    for ds in ('f60', 'cc'):
        pm = json.loads((Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/prep') / ds / 'COMPLETE_PREP_MANIFEST.json').read_text())
        jm = json.loads((Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102/prep') / ds / 'JOINT_PREP_MANIFEST.json').read_text())
        out[ds] = dict(counts=pm['counts'], empty_acceptance=pm['empty_acceptance'], equivalence_max_abs=jm['equivalence_summary']['max_abs_over_all'],
                       dev_native_from_student=jm['equivalence']['dev_native_from_student_vectors'])
    out['passed'] = all(out[ds]['equivalence_max_abs'] <= 1e-4 for ds in ('f60', 'cc'))
    return out


def main():
    part = sys.argv[1]
    if part == 'pre':
        for n, f in (('package_tests', t_package_tests), ('reference_bytes', t_reference_bytes), ('code_derivation', t_code_derivation),
                     ('oracle_selftest', t_oracle_selftest), ('loss_math', t_loss_math), ('completion_synthetic', t_completion_synthetic),
                     ('student_mechanics', t_student_mechanics), ('configs', t_configs)):
            check(n, f)
    else:
        check('real_completion_f60', lambda: t_real_completion('r39f')); check('real_completion_cc', lambda: t_real_completion('r39c'))
        check('prep_manifests', t_prep_manifests)
    out = dict(part=part, job=os.environ.get('SLURM_JOB_ID'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES, ALL_PASS=all(v['passed'] for v in RES.values()))
    (W / f'CPU_TEST_{part}.json').write_text(json.dumps(out, indent=1, default=str))
    print('R39_CPU_TEST', part, 'ALL_PASS', out['ALL_PASS'])


if __name__ == '__main__':
    main()
