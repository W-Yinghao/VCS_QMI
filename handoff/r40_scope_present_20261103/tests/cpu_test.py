"""R40 CPU checks on real data (Slurm CPU job; writes CPU_TEST.json). No standalone GPU gate (user waiver); these checks + the in-unit
fail-fast checks of every formal unit.
  package_tests        package tests/test_reference.py (unchanged package)
  reference_vs_prod    package reference/objectives.py (row_loss, present_only_loss) == production objectives (list_rows / loss_list /
                       loss_list_masked) in value and gradient, fp64 and fp32; masked rows zero gradient; empty batch; J/BCE identity
  code_derivation      r40f/r40c == r39f/r39c byte for byte except common.py (line allowlist), r39ext.py / train_unit.py (= R39 + tools/r40_patch.py
                       edits) and the new r40ext.py (identical in both trees); oracle_truth.py / train_oracle.py byte-identical to R39
  oracle_isolation     R39-identical oracle selftest; oracle_truth import refused in a trainer-role process; trainer guard refuses R39/R40 oracle dirs
  one_step_rule        synthetic AdamW: changed / zero-grad / below-fp32-spacing tensors pass, a reverted tensor and a frozen tensor in the optimizer fail
  student_modalities   real parent checkpoints (both datasets): per adaptation the frozen modality is requires_grad False, outside make_opt, gets no grad,
                       is bit-unchanged after AdamW steps; the active modality changes; state hash == the R39 N-JOINT student init (config value)
  present_real_<ds>    real FIT data: p_d recomputed from the observed map + candidate tables only (== production, == from a dict holding only those
                       arrays); PRESENT / N sampler streams draw identical lists; valid-row / unique-query / empty-batch statistics over 2000 updates
  configs              27 configs == R39 N-JOINT template except the R40 fields; frozen B optimizer == R39 frozen; references hash-bound and loadable
  e2e_<ds>             real-data CPU run of production run_unit (2 updates, readouts off, scratch dir) for representative A/B units: step-0 student vs
                       cache equivalence, R38 graph check, R40 one-step rule, reference init/student/chain identity, PRESENT p_d + masked-gradient checks
  e2e_oracle           same for one C unit inside a separate oracle-runner process (tests/train_oracle_cpu_e2e.py)
Usage: python -I -B tests/cpu_test.py"""
from __future__ import annotations
import difflib, importlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[1]
R39 = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')
PKG = Path('/home/infres/yinwang/CS_QMI/r40_plan/VCS_R39_Review_R40_Plan_20261103')
SCRATCH = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/cpu_scratch')
PY = sys.executable
RES = {}
MODS = ('common', 'data35', 'data36', 'evals', 'heads', 'native_eval', 'objectives', 'r38ext', 'r39ext', 'r40ext', 'r39_targets', 'train_unit')


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


def tree(t):
    for m in list(sys.modules):
        if m in MODS:
            del sys.modules[m]
    sys.path[:] = [x for x in sys.path if not x.endswith(('/r40f', '/r40c'))]
    sys.path.insert(0, str(W / t))


def pkg_reference():
    spec = importlib.util.spec_from_file_location('r40pkgref', str(PKG / 'reference' / 'objectives.py'))
    R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R); return R


# ------------------------------------------------------------------------------------------------
def t_package_tests():
    r = subprocess.run([PY, '-I', '-B', str(PKG / 'tests' / 'test_reference.py')], capture_output=True, text=True, timeout=600)
    return dict(passed=r.returncode == 0 and 'OK' in r.stderr, tail=r.stderr[-500:])


def t_reference_vs_prod():
    import torch
    tree('r40f'); import objectives as O
    R = pkg_reference(); out = {}
    for dt in (torch.float64, torch.float32):
        torch.manual_seed(40)
        g = torch.randn(48, 33, dtype=dt)
        a = torch.zeros(48, 33, dtype=dt); a[:16, 32] = 1; a[16:32, 3] = 1; a[32:, 4] = 1 / 3; a[32:, 9] = 1 / 3; a[32:, 20] = 1 / 3   # null / one / three positives
        perm = torch.stack([torch.randperm(33) for _ in range(48)]); g = g.gather(1, perm); a = a.gather(1, perm)
        present = torch.zeros(48, dtype=torch.bool); present[16:] = True
        tol = 1e-12 if dt == torch.float64 else 2e-6
        for obj in ('J', 'BCE', 'NCE'):
            g1 = g.clone().requires_grad_(); g2 = g.clone().requires_grad_()
            ref = R.row_loss(g1, a, obj); prod = O.list_rows(g2, a, obj); prod_loss = -prod if obj == 'J' else prod
            out[f'{dt}_{obj}_row'] = float((ref - prod_loss).abs().max())
            l1 = R.present_only_loss(g1, a, present, .7, obj); l2 = O.loss_list_masked(obj, g2, a, present.to(dt), .7)
            out[f'{dt}_{obj}_present_loss'] = float((l1 - l2).abs())
            G1, = torch.autograd.grad(l1, g1); G2, = torch.autograd.grad(l2, g2)
            out[f'{dt}_{obj}_present_grad'] = float((G1 - G2).abs().max())
            out[f'{dt}_{obj}_masked_rows_grad_zero'] = bool(torch.equal(G2[~present], torch.zeros_like(G2[~present]))) and float(G2[present].abs().sum()) > 0
            out[f'{dt}_{obj}_all_present_equals_loss_list'] = float((O.loss_list_masked(obj, g, a, torch.ones(48, dtype=dt), 1.0) - O.loss_list(obj, g, a)).abs())
            ge = g.clone().requires_grad_(); le = O.loss_list_masked(obj, ge, a, torch.zeros(48, dtype=dt), .4); Ge, = torch.autograd.grad(le, ge)
            out[f'{dt}_{obj}_empty_batch_zero'] = float(le.detach()) == 0.0 and float(Ge.abs().sum()) == 0.0
            for k in (f'{dt}_{obj}_row', f'{dt}_{obj}_present_loss', f'{dt}_{obj}_present_grad', f'{dt}_{obj}_all_present_equals_loss_list'):
                out[k + '_ok'] = out[k] <= tol
        gg = g.double().clone().requires_grad_()
        gj, = torch.autograd.grad(O.loss_list('J', gg, a.double()), gg); gb, = torch.autograd.grad(O.loss_list('BCE', gg, a.double()), gg)
        out['dJ_eq_(1-T^2)dBCE_max_abs'] = float((gj - (1 - torch.tanh(gg / 2) ** 2) * gb).abs().max())
    out['passed'] = all(v for k, v in out.items() if isinstance(v, bool)) and out['dJ_eq_(1-T^2)dBCE_max_abs'] < 1e-12
    return out


def t_code_derivation():
    sys.path.insert(0, str(W / 'tools')); import r40_patch as P
    out = {}
    for t, s in (('r40f', 'r39f'), ('r40c', 'r39c')):
        same = {f.name: sh(f) == sh(W / t / f.name) for f in sorted((R39 / s).glob('*.py')) if f.name not in ('common.py', 'r39ext.py', 'train_unit.py')}
        tu = P.derive_train_unit((R39 / s / 'train_unit.py').read_text()) == (W / t / 'train_unit.py').read_text()
        rx = P.apply((R39 / s / 'r39ext.py').read_text(), P.R39EXT_REP) == (W / t / 'r39ext.py').read_text()
        a = (R39 / s / 'common.py').read_text().splitlines(); b = (W / t / 'common.py').read_text().splitlines()
        added = [l for l in difflib.ndiff(a, b) if l.startswith('+ ')]; removed = [l for l in difflib.ndiff(a, b) if l.startswith('- ')]
        allow = ('R40', 'R39_ROOT', 'R39_OUT', "ROOT = Path", "OUT = Path", 'R39_PREP', '_FORBIDDEN_PREFIXES', 'ADAPTATIONS', 'FROZEN_FILE', '# ---')
        bad_add = [l for l in added if l.strip('+ ').strip() and not any(x in l for x in allow)]
        bad_rem = [l for l in removed if not any(x in l for x in ('"""R39', "ROOT = Path", "OUT = Path", 'R39_PREP', '_FORBIDDEN_PREFIXES'))]
        files = sorted(f.name for f in (W / t).glob('*.py')); files39 = sorted(f.name for f in (R39 / s).glob('*.py'))
        out[t] = dict(byte_identical=same, train_unit_derived=tu, r39ext_derived=rx, common_bad_added=bad_add, common_bad_removed=bad_rem,
                      extra_files=sorted(set(files) - set(files39)), missing_files=sorted(set(files39) - set(files)))
        out[t]['passed'] = all(same.values()) and tu and rx and not bad_add and not bad_rem and out[t]['extra_files'] == ['r40ext.py'] and not out[t]['missing_files']
    out['r40ext_identical_trees'] = sh(W / 'r40f' / 'r40ext.py') == sh(W / 'r40c' / 'r40ext.py')
    out['r39ext_identical_trees'] = sh(W / 'r40f' / 'r39ext.py') == sh(W / 'r40c' / 'r39ext.py')
    out['oracle_truth_bit_identical_r39'] = sh(W / 'r40f' / 'oracle_truth.py') == sh(R39 / 'r39f' / 'oracle_truth.py')
    out['passed'] = out['r40f']['passed'] and out['r40c']['passed'] and out['r40ext_identical_trees'] and out['r39ext_identical_trees'] and out['oracle_truth_bit_identical_r39']
    return out


def t_oracle_isolation():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([PY, '-I', '-B', str(W / 'r40f' / 'train_oracle.py'), '--selftest'], capture_output=True, text=True, timeout=600, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('R39_ORACLE_SELFTEST')]
    d = json.loads(line[0].split(' ', 1)[1]) if line else {}
    blocked = subprocess.run([PY, '-I', '-B', '-c', f"import sys; sys.path.insert(0,'{W / 'r40f'}'); import common; common.install_guards('trainer'); import oracle_truth"],
                             capture_output=True, text=True, env=env)
    guard = subprocess.run([PY, '-I', '-B', '-c', f"import sys; sys.path.insert(0,'{W / 'r40f'}'); import common as C; C.install_guards('trainer')\n"
                            "bad=[]\nfor p in (C.R39_OUT/'oracle_runs'/'ORACLE_DIAGNOSTICS.json', C.ORACLE_RUNS/'x.json'):\n"
                            "  try:\n    open(p)\n    bad.append(str(p))\n  except PermissionError:\n    pass\n  except FileNotFoundError:\n    bad.append('notfound:'+str(p))\n"
                            "print('GUARD_BAD', bad)"], capture_output=True, text=True, env=env)
    return dict(passed=bool(d.get('ALL_PASS')) and blocked.returncode != 0 and 'oracle isolation' in blocked.stderr and 'GUARD_BAD []' in guard.stdout,
                selftest=d, import_blocked_in_trainer=blocked.returncode != 0, trainer_guard=guard.stdout.strip()[-300:] + guard.stderr[-300:])


def t_one_step_rule():
    import torch
    tree('r40f'); import r40ext as X
    torch.manual_seed(1)

    def run(sabotage=False, frozen_in_opt=False):
        p_norm = torch.nn.Parameter(torch.randn(10)); p_zero = torch.nn.Parameter(torch.randn(10)); p_tiny = torch.nn.Parameter(torch.randn(10))
        p_big = torch.nn.Parameter(torch.full((10,), 1e4)); fz = torch.nn.Parameter(torch.randn(10), requires_grad=False)
        ps = [p_norm, p_zero, p_tiny, p_big] + ([fz] if frozen_in_opt else [])
        opt = torch.optim.AdamW([dict(params=ps, lr=1e-5)], weight_decay=1e-4)
        named = [('head.A.norm', p_norm), ('head.A.zero', p_zero), ('head.A.tiny', p_tiny), ('head.A.big', p_big)]
        fz_ref = {'student.A.txt.w': fz.detach().clone()}
        for step in range(2):
            opt.zero_grad()
            p_norm.grad = torch.randn(10); p_zero.grad = torch.zeros(10); p_tiny.grad = torch.full((10,), 1e-30); p_big.grad = torch.randn(10)
            if step == 1:
                g1 = X.grad_record(named); before = X.snapshot(named)
            opt.step()
        if sabotage:
            with torch.no_grad():
                p_norm.copy_(before['head.A.norm'])
        r = X.one_step_rule(opt, named, before, g1, [('student.A.txt.w', fz)], fz_ref)
        return r
    peers = X.PEERS; X.PEERS = ('A',)   # synthetic single-peer setting
    try:
        ok = run(); sab = run(sabotage=True); fin = run(frozen_in_opt=True)
    finally:
        X.PEERS = peers
    out = dict(normal_passed=ok['passed'], normal_unchanged_explained=ok['unchanged_explained'], sabotage_flagged=not sab['passed'] and 'head.A.norm' in sab['unexplained_unchanged'],
               frozen_in_optimizer_flagged=not fin['passed'] and not fin['frozen_modality_ok'])
    out['passed'] = bool(out['normal_passed'] and set(ok['unchanged_explained']) == {'head.A.zero', 'head.A.tiny', 'head.A.big'} and out['sabotage_flagged'] and out['frozen_in_optimizer_flagged'])
    return out


def t_student_modalities(t):
    import torch
    tree(t); import common as C; import r39ext as R9; from common import state_hash
    torch.use_deterministic_algorithms(True)
    ck = torch.load(C.PARENT_CKPT, map_location='cpu', weights_only=False)
    tr = np.load(C.RAW_FIT_IMS, mmap_mode='r'); T = np.load(C.TOKENS_FILE)
    x = torch.from_numpy(np.ascontiguousarray(tr[:6])); tok = torch.from_numpy(T['train_tok'][:8]); ln = torch.from_numpy(T['train_len'][:8])
    cfgs = {json.loads((W / 'configs' / f).read_text())['adaptation']: json.loads((W / 'configs' / f).read_text())
            for f in [f.name for f in (W / 'configs').glob('R40_*.json')] if json.loads((W / 'configs' / f).read_text())['code_tree'] == t}
    ref_hash = {c['adaptation']: c['reference_joint']['student_init_state_sha256'] for c in cfgs.values() if c.get('reference_joint')}
    out = {}
    for ad, mods in C.ADAPTATIONS.items():
        if not mods:
            continue
        s = R9.build_students(ck, torch.device('cpu'), mods)
        p = s['A']
        opt = torch.optim.AdamW([q for pp in C.PEERS for q in s[pp].trainable()], lr=1e-5, weight_decay=1e-4)
        ref0 = {n: q.detach().clone() for n, q in p.named_parameters()}; fz0 = R9.state_digest(p.frozen_state())
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = (p.enc_img(x) * torch.linspace(-1, 1, 1024)).sum() + (p.enc_cap(tok, ln) * torch.linspace(1, -1, 1024)).sum()
            loss.backward(); opt.step()
        opt_ids = {id(q) for g in opt.param_groups for q in g['params']}
        rec = {}
        for m in ('img', 'txt'):
            ps = [(n, q) for n, q in p.named_parameters() if n.startswith(m + '.')]
            if m in mods:
                act = [(n, q) for n, q in ps if q.requires_grad]
                rec[m] = dict(active=True, n_trainable=len(act), frac_changed=float(np.mean([not torch.equal(q.detach(), ref0[n]) for n, q in act])),
                              all_in_optimizer=all(id(q) in opt_ids for _, q in act))
            else:
                rec[m] = dict(active=False, any_requires_grad=any(q.requires_grad for _, q in ps), any_grad=any(q.grad is not None for _, q in ps),
                              any_in_optimizer=any(id(q) in opt_ids for _, q in ps), bit_unchanged=all(torch.equal(q.detach(), ref0[n]) for n, q in ps))
        rec['frozen_state_unchanged'] = R9.state_digest(p.frozen_state()) == fz0
        s0 = R9.build_students(ck, torch.device('cpu'), mods)
        rec['init_state_hash_equals_R39_N_JOINT'] = state_hash(s0) == ref_hash.get(ad, ref_hash.get('both_joint'))
        rec['passed'] = bool(rec['frozen_state_unchanged'] and rec['init_state_hash_equals_R39_N_JOINT'] and
                             all((v['frac_changed'] > .9 and v['all_in_optimizer']) if v['active'] else (not v['any_requires_grad'] and not v['any_grad'] and not v['any_in_optimizer'] and v['bit_unchanged'])
                                 for k, v in rec.items() if k in ('img', 'txt')))
        out[ad] = rec
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_present_real(t):
    import torch
    tree(t); import common as C; from data35 import load_cond; import data36 as D36; import r40ext as X
    d = load_cond(C.TARGET_COND, torch.device('cpu'))
    P = D36.presence_probs(d)
    P_only = D36.presence_probs(dict(obs_np=d['obs_np'].copy(), cand_i2t=d['cand_i2t'], cand_t2i=d['cand_t2i']))
    rc = X.present_recheck(d, P, D36.null_lists, D36.valid_rows)
    r1, r2 = np.random.default_rng([1001, 3]), np.random.default_rng([1001, 3])
    same = True; vi = vt = 0; ei = et = 0; ui = []; ut = []; n = 0
    for _ in range(2000):
        l1, i1 = D36.sample('N_PRESENT_ONLY', r1, None, d, 16, None); l2, i2 = D36.sample('OBS_MISSING_NULL', r2, None, d, 16, None)
        same &= all(np.array_equal(x, y) for a_, b_ in zip(l1, l2) for x, y in zip(a_, b_)) and all(np.array_equal(x, y) for x, y in zip(i1, i2))
        mi, mt = D36.valid_rows(l1)
        vi += int(mi.sum()); vt += int(mt.sum()); ei += int(mi.sum() == 0); et += int(mt.sum() == 0)
        ui.append(len(np.unique(l1[0][0][mi]))); ut.append(len(np.unique(l1[1][0][mt]))); n += 16
    out = dict(p_production=P, p_from_observed_and_candidates_only=P_only, recheck=rc, sampler_lists_identical_to_N=bool(same),
               valid_frac_i2t=vi / n, valid_frac_t2i=vt / n, empty_valid_batches_i2t=ei, empty_valid_batches_t2i=et,
               valid_unique_queries_mean_i2t=float(np.mean(ui)), valid_unique_queries_mean_t2i=float(np.mean(ut)), updates=2000)
    out['passed'] = bool(rc['passed'] and same and P_only['p_i2t'] == P['p_i2t'] and P_only['p_t2i'] == P['p_t2i'])
    return out


def t_configs():
    units = json.loads((W / 'configs' / 'UNITS.json').read_text())
    allowed = {'unit_id', 'package_unit', 'block', 'line', 'code_tree', 'runner', 'supervision', 'supervision_label', 'oracle', 'adaptation', 'student_modalities',
               'encoder', 'parent_key', 'label', 'job_bundle', 'r39_template', 'supervision_rule', 'joint_rule', 'reference_unit', 'reference_joint', 'reference_oracle',
               'optimizer', 'evaluation'}
    out = dict(n=len(units), bad=[], counts={})
    for u in units:
        c = json.loads((W / 'configs' / f'{u}.json').read_text()); tm = json.loads((R39 / 'configs' / f"{c['r39_template']}.json").read_text())
        diff = {k for k in set(c) | set(tm) if c.get(k) != tm.get(k)} - allowed
        frozen = c['adaptation'] == 'frozen'
        o = dict(c['optimizer']); ot = dict(tm['optimizer'])
        if frozen:
            ot.pop('lr_encoder')
        ok = not diff and o == ot and c['reference_unit']['unit_id'] == tm['reference_unit']['unit_id'] and c['steps'] == 8000 and c['ckpt_steps'] == tm['ckpt_steps']
        ok &= (c.get('reference_joint') is not None) == (not frozen) and (c.get('reference_oracle') is not None) == c['oracle']
        ok &= (c['encoder'] == 'joint_global') == (not frozen) and ('lr_encoder' in c['optimizer']) == (not frozen)
        for k in ('reference_unit', 'reference_joint', 'reference_oracle'):
            if c.get(k):
                ok &= sh(c[k]['result_json']) == c[k]['result_json_sha256'] and sh(c[k]['train_history_json']) == c[k]['train_history_json_sha256']
        if not ok:
            out['bad'].append((u, sorted(diff)))
        kk = (c['block'], c['adaptation'], c['supervision']); out['counts'][str(kk)] = out['counts'].get(str(kk), 0) + 1
    out['passed'] = out['n'] == 27 and not out['bad']
    return out


def t_e2e(t, units):
    import torch
    tree(t); import common as C; C.install_guards('trainer'); import train_unit as TU
    TU.set_determinism()
    dev = torch.device('cpu'); ctxs = {}; out = {}
    for u in units:
        cfg = TU.cfg_of(u); cfg['_config_sha256'] = sh(W / 'configs' / f'{u}.json')
        key = cfg['cond']
        if key not in ctxs:
            ctxs[key] = TU.Ctx(cfg['cond'], dev, need_local=False)
        ctx = ctxs[key]
        if cfg['encoder'] == 'joint_global':
            ctx.ensure_joint()
        od = SCRATCH / t / u
        if od.exists():
            od = SCRATCH / t / f'{u}_{os.environ.get("SLURM_JOB_ID", "local")}_{int(time.time())}'
        res = TU.run_unit(dict(cfg, phase=1), ctx, pool=None, out_dir=od, stop_at=2, no_readout=True, oracle=None)
        h = res['hist']; r40 = h['r40']; r39 = h['r39']
        rec = dict(one_step_r40=r40.get('one_step_check'), references=r40.get('references'), present_check=r40.get('present_check'), masked_grad=r40.get('masked_grad'),
                   step0_cache_equivalence=r39.get('step0_cache_equivalence', {}).get('passed') if r39.get('step0_cache_equivalence') else None,
                   step1_graph_check=h['r38'].get('step1_graph_check', {}).get('passed'), r38_reference_chain_equal_at=h['r38']['reference_chain_equal_at'])
        rec['passed'] = bool(rec['one_step_r40'] and rec['one_step_r40']['passed'] and rec['step1_graph_check'] and rec['r38_reference_chain_equal_at'] == [0]
                             and all(v['init_equal'] and v['student_init_equal'] in (None, True) and v['chain_equal_at'] == [0] for v in (rec['references'] or {}).values())
                             and (cfg['encoder'] != 'joint_global' or rec['step0_cache_equivalence'])
                             and (cfg['supervision'] != 'N_PRESENT_ONLY' or (rec['present_check']['passed'] and rec['masked_grad']['violations'] == 0 and rec['masked_grad']['updates_checked'] == 2)))
        out[u] = rec
    out['passed'] = all(v['passed'] for v in out.values())
    return out


def t_e2e_oracle():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run([PY, '-I', '-B', str(W / 'tests' / 'train_oracle_cpu_e2e.py'), 'R40_C_F60_R30_H1001_ORACLE_joint_J'], capture_output=True, text=True, timeout=7200, env=env)
    line = [l for l in r.stdout.splitlines() if l.startswith('R40_ORACLE_E2E ')]
    d = json.loads(line[-1].split(' ', 1)[1]) if line else dict(stdout=r.stdout[-2000:], stderr=r.stderr[-4000:])
    d['passed'] = bool(d.get('passed')) and r.returncode == 0
    return d


def main():
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    for n, f in (('package_tests', t_package_tests), ('reference_vs_prod', t_reference_vs_prod), ('code_derivation', t_code_derivation),
                 ('oracle_isolation', t_oracle_isolation), ('one_step_rule', t_one_step_rule), ('configs', t_configs),
                 ('student_modalities_f60', lambda: t_student_modalities('r40f')), ('student_modalities_cc', lambda: t_student_modalities('r40c')),
                 ('present_real_f60', lambda: t_present_real('r40f')), ('present_real_cc', lambda: t_present_real('r40c')),
                 ('e2e_f60', lambda: t_e2e('r40f', ['R40_A_F60_R30_H1001_image_only_J', 'R40_A_F60_R30_H1001_text_only_BCE', 'R40_B_F60_R30_H1001_PRESENT_frozen_NCE',
                                                    'R40_B_F60_R30_H1001_PRESENT_both_joint_J'])),
                 ('e2e_oracle', t_e2e_oracle),
                 ('e2e_cc', lambda: t_e2e('r40c', ['R40_B_CC31_H1001_PRESENT_frozen_BCE', 'R40_A_CC31_H1001_text_only_J', 'R40_A_CC31_H1001_image_only_NCE',
                                                   'R40_B_CC31_H1001_PRESENT_both_joint_NCE']))):
        if only is None or n in only:
            check(n, f)
    out = dict(job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), checks=RES,
               ALL_PASS=all(v['passed'] for v in RES.values()))
    out['only'] = only
    (W / ('CPU_TEST.json' if only is None else f"CPU_TEST_partial_{os.environ.get('SLURM_JOB_ID', 'local')}.json")).write_text(json.dumps(out, indent=1, default=str))
    print('R40_CPU_TEST ALL_PASS', out['ALL_PASS'])


if __name__ == '__main__':
    main()
