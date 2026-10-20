"""Short K8/K32 CUDA gate of the hardware amendment, one Slurm job per GPU type (RTX6000PRO, H100; optional L40S).

Why: the cross-hardware T01 gates (xhw_t01, core 4336bb6) ran the MF core at widths (3,2), (7,6), (16,16) and the host adapter /
registration geometry at K = 16 only; the K8 / K32 head, its M1 solver, the K-wide registration energy and the CPU-first checkpoint
restore (never exercised on any GPU) were not executed on RTX PRO 6000 or H100.  This gate runs them on the allocated GPU with the
real FIT data (FIT role only, through the BF1 DataGuard), the frozen BF1 tree and the unchanged K override, at K = 8, 16, 32 for the
three MF families with the inherited lr / cap:
  * hardware-amended runtime check on this GPU (exact name, partition, node; every BF1 version / precision / env check)
  * head shapes, optimizer groups, paired encoder initialisation and batch stream across K
  * one real training step: finite; perm1 / M0 |W| <= 1; M1 zero-head reference, threshold, stop rule and final FW gap against
    independent fp64 recomputations on the same features
  * moments == dense V reference and perm1 == explicit pairs at a random box W, on the step's CUDA features
  * BF1 save_checkpoint -> step -> CPU-first restore -> step reproduces the step (BF1 GPU-gate tolerance atol 1e-5, rtol 1e-4),
    AdamW step counters back on CPU, wrong-K identity / width refused
  * registration energy (MFRigidEnergy) rows == dense / explicit references, geometry gradient finite
  * run_hw.main end to end in a temporary bundle with only run_unit stubbed (real runtime check on this GPU) and its refusals.
Writes hw_amendment_20261019/gpu_gates/<partition>_<job>.json.  An execution / numerical gate, never scientific evidence.
"""
import argparse
import copy
import json
import math
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import capacity_common as C
import hw_amendment as H
import run as R
import run_hw as RH
from smoke_k import Recorder, fp64_head_reference

RESUME_TOL = (1e-5, 1e-4)                 # the BF1 GPU gate's resume comparison (gpu_gate.py), atol / rtol


def close_all(xs, ys, torch):
    atol, rtol = RESUME_TOL
    return all(torch.allclose(x, y, atol=atol, rtol=rtol) for x, y in zip(xs, ys))


def config_checks(ctx, family, K, check, tmp, device):
    bf1, A, torch, np, MF = ctx['bf1'], ctx['A'], ctx['torch'], ctx['np'], ctx['A'].MF
    cand = ctx['prereg']['families'][family]['candidate']
    method = C.METHODS[family]
    tag = f'{family}_K{K}'
    row = {'family': family, 'K': K, 'lr': cand['lr'], 'cap': cand['cap']}
    ident = {**bf1.identity(family, 1001, cand['lr'], cand['cap'], 'HW_GATE'), 'schema': 'v10.capacity_ext.hw_gate.v1', 'K': K}
    bf1.config = ctx['original_config']
    resolved = R.require_only_k_differs(bf1, ctx['config'], ident)
    bf1.config = ctx['config']
    t = bf1.new_trainer(ident, ctx['guard'], device, ctx['imgs'])
    h = t.head
    check(f'{tag}:patched_config_and_device', t.cfg.K == K and resolved['K'] == K and h.W.device.type == torch.device(device).type)
    check(f'{tag}:head_shapes', (h.left.in_features, h.left.out_features, h.right.out_features) == (16, K, K)
          and tuple(h.W.shape) == (K, K) and not bool(h.W.any()))
    groups = {g['name']: g for g in t.opt.param_groups}
    if family == 'MF_VCS_M1':
        check(f'{tag}:m1_W_not_optimised', set(groups) == {'encoders', 'feature_maps'} and not h.W.requires_grad)
    else:
        check(f'{tag}:W_group', set(groups) == {'encoders', 'feature_maps', 'W'} and groups['W']['params'] == [h.W]
              and groups['W']['weight_decay'] == 0.0 and groups['W']['lr'] == cand['lr'])
    check(f'{tag}:tied_lr', all(g['lr'] == cand['lr'] for g in groups.values()))
    ctx['paired']['encoder_init'].add(A.state_hash(t.eU, t.eV))
    ctx['paired']['sampler_before'].add(json.dumps(t.sampler.g.bit_generator.state, sort_keys=True))
    t0 = time.time()
    rec = t.step(keep=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    row['first_step_seconds'] = time.time() - t0
    hb, o = rec['_batch'][0], rec['_out']
    ctx['paired']['first_batch'].add(json.dumps([hb.cores, hb.y0.tolist(), hb.x0.tolist(), hb.k.tolist(), hb.flip.tolist(),
                                                 float(hb.yx.sum()), hb.perm.tolist()]))
    scenes = o['scenes']
    check(f'{tag}:finite_step', all(math.isfinite(rec[k]) for k in ('loss', 'J', 'grad_norm_preclip')), loss=rec['loss'])
    check(f'{tag}:feature_width', tuple(o['A'].shape) == (2048, K) and all(tuple(s.a.shape) == (128, K) for s in scenes))
    if family == 'MF_VCS_M1':
        g32, tau, it = rec['m1_gap_ref_zero'], rec['m1_tolerance_effective'], rec['m1_iterations']
        _, g64, _ = fp64_head_reference(scenes, torch.zeros(K, K, device=o['A'].device))
        _, _, gap64 = fp64_head_reference(scenes, o['info']['m1_W'])
        unit = A.m1_resolution_unit(torch.float32)
        check(f'{tag}:m1_reference', g32 > 0 and abs(g32 - g64) <= 16 * unit, g32=g32, g64=g64)
        check(f'{tag}:m1_threshold', math.isclose(tau, 0.01 * g32, rel_tol=1e-12, abs_tol=0.0))
        stop = rec['m1_stop_reason']
        check(f'{tag}:m1_stop_rule', (stop == 'gap_tolerance' and rec['m1_gap_final'] <= tau and it <= cand['cap'])
              or (stop == 'iteration_budget_exhausted' and it == cand['cap']), stop=stop, iterations=it)
        check(f'{tag}:m1_gap_final', abs(rec['m1_gap_final'] - gap64) <= 16 * unit + 1e-3 * gap64, gap32=rec['m1_gap_final'], gap64=gap64)
        diag = rec['m1_fp64_diagnostic']
        if diag.get('triggered'):
            check(f'{tag}:m1_fp64_diag_agrees', math.isclose(diag['gap_ref_zero_fp64'], g64, rel_tol=1e-9)
                  and math.isclose(diag['gap_final_fp64'], gap64, rel_tol=1e-6, abs_tol=1e-15))
        check(f'{tag}:m1_W', float(o['info']['m1_W'].abs().max()) <= 1 and torch.equal(h.W.detach(), o['info']['m1_W'].detach()))
        row['m1'] = {'gap_ref_zero': g32, 'gap_ref_zero_fp64': g64, 'iterations': it, 'stop': stop}
    else:
        check(f'{tag}:W_projected', rec['w_absmax'] <= 1 and float(h.W.detach().abs().max()) > 0, w_absmax=rec['w_absmax'])
    gen = torch.Generator().manual_seed(1000 + K)
    Wr = torch.empty(K, K).uniform_(-1, 1, generator=gen).to(o['A'].device)
    weights = A.host_weights(16)
    mom = MF.objective_terms(scenes, Wr, weights=weights)
    dense = MF.dense_objective_terms(scenes, Wr, weights=weights)
    cmp = A.compare_many([mom.p1, mom.p2, mom.q1, mom.q2, mom.j], [dense.p1, dense.p2, dense.q1, dense.q2, dense.j], 'float32')
    check(f'{tag}:moments_equal_dense', cmp['allclose'], max_abs_err=cmp['max_abs_err'])
    perm = o['perm']
    sp = MF.sampled_permutation_terms(scenes, Wr, weights=weights, permutations=[perm[s][None] for s in range(16)])
    q1 = sum(torch.einsum('ik,kl,il->i', s.a.double(), Wr.double(), s.b[perm[i]].double()).mean() / 16 for i, s in enumerate(scenes))
    check(f'{tag}:perm1_explicit', abs(float(sp.q1) - float(q1)) <= 1e-5 + 1e-4 * abs(float(q1)))
    cp = tmp / tag / 'step_0001.pt'
    bf1.save_checkpoint(t, cp, ident)
    t.step()
    expected_modules = [[v.detach().clone() for v in m.state_dict().values()] for m in t.modules()]
    expected_opt = copy.deepcopy(t.opt.state_dict())
    expected_sampler = copy.deepcopy(t.sampler.g.bit_generator.state)
    real_load, calls = torch.load, []

    def spy(*a, **k):
        calls.append(k.get('map_location'))
        return real_load(*a, **k)
    torch.load = spy
    try:
        ctx['loader'](t, cp, ident)
    finally:
        torch.load = real_load
    st0 = t.opt.state_dict()['state']
    steps_cpu = all(s['step'].device.type == 'cpu' for s in st0.values())
    moments_on_device = all(s['exp_avg'].device.type == torch.device(device).type for s in st0.values())
    check(f'{tag}:cpu_first_restore', calls[-1] == 'cpu' and t.n_steps == 1 and steps_cpu and moments_on_device,
          adamw_step_on_cpu=steps_cpu)
    t.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    got = [[v for v in m.state_dict().values()] for m in t.modules()]
    exact = all(torch.equal(a, b) for xs, ys in zip(got, expected_modules) for a, b in zip(xs, ys))
    modules_close = all(close_all(xs, ys, torch) for xs, ys in zip(got, expected_modules))
    st = t.opt.state_dict()['state']
    opt_close = all(close_all([st[i][n]], [v], torch) for i, s in expected_opt['state'].items() for n, v in s.items() if n != 'step') \
        and all(float(st[i]['step']) == float(s['step']) for i, s in expected_opt['state'].items())
    check(f'{tag}:resume_reproduces_step', modules_close and opt_close and t.sampler.g.bit_generator.state == expected_sampler,
          bitwise=exact, tolerance=RESUME_TOL)
    other_K = 32 if K != 32 else 8
    try:
        ctx['loader'](t, cp, {**ident, 'K': other_K})
        refused_identity = False
    except RuntimeError:
        refused_identity = True
    other = bf1.new_trainer({**ident, 'K': other_K}, ctx['guard'], device, ctx['imgs'])
    try:
        ctx['loader'](other, cp, ident)
        refused_width = False
    except RuntimeError:
        refused_width = True
    check(f'{tag}:wrong_K_refused', refused_identity and refused_width)
    del other
    core = ctx['fit'][0]
    ctx['guard'].log_rule(core, {'FIT'}, 'BF1 capacity hardware gate: perturbation rule (non-scientific)')
    inputs = A.GuardedInputs(ctx['guard'], {'FIT'}, 'BF1 capacity hardware gate energy (non-scientific)',
                             {core: A.Q.perturbations_p2(core)}, device)
    estimator = 'perm1' if method == 'MF_perm1' else 'moments'
    with bf1.evaluation_state(t):
        ref = A.reference_descriptors(t.eU, inputs, core)
        rng = np.random.default_rng(A.Q.seed_of(f'R6-REG-P2|pos|{core}|0'))
        energy = A.MFRigidEnergy(ref, inputs.moving(core, 0)[1], rng, eV=t.eV, head=t.head, estimator=estimator, record=True)
        _, ts = A.Q.grid_points()
        tt = torch.tensor(ts[:64], dtype=torch.float32, device=ref.device)
        with torch.no_grad():
            vals = energy.grid_values(0.0, tt)
            Bc, Ar = energy.last['B'], energy.A
            check(f'{tag}:energy_shapes', tuple(vals.shape) == (64,) and bool(torch.isfinite(vals).all())
                  and tuple(Bc.shape) == (64, 4096, K) and tuple(Ar.shape) == (4096, K))
            for label, Wx in (('head_W', energy.W), ('random_W', Wr)):
                rows = A.mf_energy_rows(Ar, Bc, Wx, estimator, perm=energy.perm)
                if estimator == 'moments':
                    refrows = A.mf_energy_rows(Ar, Bc, Wx, 'dense')
                else:
                    tp = torch.einsum('ik,kl,cil->ci', Ar.double(), Wx.double(), Bc.double())
                    tq = torch.einsum('ik,kl,cil->ci', Ar.double(), Wx.double(), Bc[:, energy.perm].double())
                    refrows = (tp.mean(-1) - tq.mean(-1) - 0.5 * (tp.square().mean(-1) + tq.square().mean(-1))).float()
                c2 = A.compare(rows, refrows, 'float32')
                check(f'{tag}:energy_{label}_matches_reference', c2['allclose'], max_abs_err=c2['max_abs_err'])
            again = A.compare(vals, A.mf_energy_rows(Ar, Bc, energy.W, estimator, perm=energy.perm), 'float32')
            check(f'{tag}:energy_rows_are_grid_values', again['allclose'], max_abs_err=again['max_abs_err'])
        p = torch.tensor([0.1, 1.0, -2.0], requires_grad=True, device=ref.device)
        val = energy(p)
        val.backward()
        check(f'{tag}:geometry_gradient', bool(torch.isfinite(val)) and p.grad is not None and bool(torch.isfinite(p.grad).all()))
    del energy, inputs, t
    if device == 'cuda':
        torch.cuda.empty_cache()
    return row


def runner_dry_run(check, base_freeze, hw_freeze, partition, stub_env=None):
    """run_hw.main end to end in a temporary bundle / output root; only BF1 run_unit is stubbed (and, on CPU only, the CUDA
    environment record).  Exercises the amended gates, admission, identity, both patches, resume truncation and the refusals."""
    from unittest import mock
    prereg = C.read(C.BUNDLE / 'PREREGISTRATION.json')
    unit = C.unit_by_key(prereg, 'MF_VCS_M1_K32_seed1003')
    job = os.environ['SLURM_JOB_ID']
    captured = {}

    class StopBeforeTraining(Exception):
        pass
    real_import = R.import_bf1
    real_validate, real_out_hw = H.validate_record, H.OUT_HW

    def validate_against_the_real_root(rec, prereg_):
        with mock.patch.object(H, 'OUT_HW', real_out_hw):           # the record names the real root; only the dry run writes elsewhere
            return real_validate(rec, prereg_)

    def fake_import():
        bf1 = real_import()

        def run_unit(ident, directory, allocation, env):
            captured.update(ident=ident, directory=Path(directory), K=bf1.config(ident).K, loader=bf1.load_checkpoint.__qualname__,
                            steps=[r['step'] for r in C.lines(Path(directory) / 'TRAIN_STEPS.jsonl')], gpu=env['gpu_name'],
                            partition=env['slurm_partition'], allocation=type(allocation).__name__)
            raise StopBeforeTraining()
        bf1.run_unit = run_unit
        return bf1

    def env_for_cpu(bf1):
        env = bf1.A.environment('cpu')
        env.update(stub_env)
        if H.hardware_problem(env):
            raise RuntimeError('Hardware outside the amendment: ' + H.hardware_problem(env))
        env['hardware_amendment'] = {'runtime_status': 'PASS (CPU dry-run stub)'}
        return env

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        bundle, out, out_hw = t / 'bundle', t / 'out', t / 'out_hw'
        hwdir = bundle / 'hw_amendment_20261019'
        hwdir.mkdir(parents=True)
        for name in ('FREEZE.json', 'PREREGISTRATION.json', 'CHECKS.json'):
            shutil.copy2(C.BUNDLE / name, bundle / name)
        for name in ('FREEZE.json', 'HARDWARE_AMENDMENT.json', 'CHECKS.json'):
            shutil.copy2(H.DIR / name, hwdir / name)
        C.write(hwdir / 'gpu_gates' / f'{partition}_dryrun.json', {'status': 'PASS', 'hw_freeze_sha256': hw_freeze, 'partition': partition,
                                                                     'gpu_name': H.GPUS[partition], 'note': 'dry-run stand-in'})
        env = {'SLURM_JOB_NAME': unit['job_name'], 'SLURM_JOB_PARTITION': partition, 'V10K_TIME_LIMIT_RAW': '3:00:00',
               'V10K_SCHED_START_RAW': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - 5)),
               'V10K_BATCH_START': str(time.time() - 2), 'V10_CAPACITY_FREEZE': base_freeze, 'V10_CAPACITY_HW_FREEZE': hw_freeze}
        patches = [mock.patch.object(C, 'BUNDLE', bundle), mock.patch.object(C, 'OUT', out), mock.patch.object(H, 'OUT_HW', out_hw),
                   mock.patch.object(H, 'DIR', hwdir), mock.patch.object(H, 'validate_record', validate_against_the_real_root),
                   mock.patch.object(R, 'import_bf1', fake_import), mock.patch.dict(os.environ, env)]
        if stub_env is not None:
            patches.append(mock.patch.object(H, 'runtime_environment', env_for_cpu))
        for p in patches:
            p.start()
        try:
            prereg_t = C.read(bundle / 'PREREGISTRATION.json')
            ident = H.expected_identity_hw(unit, prereg_t, base_freeze, hw_freeze)
            d = out_hw / unit['key']
            C.write(d / 'IDENTITY.json', ident)
            C.write(d / 'checkpoints' / 'step_0250.json', {'step': 250})
            (d / 'TRAIN_STEPS.jsonl').write_text(''.join(json.dumps({'step': s}) + '\n' for s in range(1, 301)))
            try:
                RH.main(unit['key'])
                stopped = False
            except StopBeforeTraining:
                stopped = True
            events = [r['event'] for r in C.lines(bundle / 'COST_LEDGER.jsonl')]
            runtime = C.read(bundle / 'runtime' / f'{job}.json') if (bundle / 'runtime' / f'{job}.json').exists() else {}
            check('dry_run:reached_run_unit', stopped and captured.get('K') == 32 and captured.get('ident') == ident
                  and captured.get('directory') == d and captured.get('loader', '').startswith('cpu_first_loader')
                  and captured.get('gpu') == H.GPUS[partition] and captured.get('partition') == partition,
                  captured={k: str(v) for k, v in captured.items() if k not in ('ident', 'steps')})
            check('dry_run:resume_cut_to_checkpoint', captured.get('steps') == list(range(1, 251)))
            check('dry_run:records', events == ['START', 'RESUME_PREPARED', 'RUNTIME_VERIFIED', 'END']
                  and runtime.get('identity') == ident and runtime.get('gpu_name') == H.GPUS[partition]
                  and (hwdir / 'jobs' / f'{job}.json').exists(), events=events)
            refusals = {}
            (out / unit['key']).mkdir(parents=True)
            refusals['base_unit_started'] = RH.main(unit['key'])
            (out / unit['key']).rmdir()
            if partition in H.GATED:
                for g in (hwdir / 'gpu_gates').glob('*.json'):
                    g.unlink()
                refusals['gate_missing'] = RH.main(unit['key'])
            check('dry_run:refusals_exit_3', all(rc == C.EXIT_REFUSED for rc in refusals.values()), refusals=refusals)
            for label, change, key in (('job_name', {'SLURM_JOB_NAME': 'v10k_wrong'}, unit['key']),
                                       ('partition', {'SLURM_JOB_PARTITION': 'A100'}, unit['key']),
                                       ('time_limit', {'V10K_TIME_LIMIT_RAW': '1:30:00'}, unit['key']),
                                       ('completed_base_unit', {'SLURM_JOB_NAME': 'v10k_m1_k32_1001'}, 'MF_VCS_M1_K32_seed1001')):
                with mock.patch.dict(os.environ, change):
                    try:
                        RH.main(key)
                        refused = False
                    except RuntimeError:
                        refused = True
                check(f'dry_run:refuses_{label}', refused)
        finally:
            for p in reversed(patches):
                p.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    ap.add_argument('--quick', action='store_true', help='CPU code test only: one family, K 8, dry run with stubbed GPU records')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    check = Recorder()
    base_freeze, hw_freeze = os.environ['V10_CAPACITY_FREEZE'], os.environ['V10_CAPACITY_HW_FREEZE']
    prereg = C.validate_prereg(C.frozen(base_freeze))
    H.validate_record(H.frozen_hw(hw_freeze), prereg)
    os.environ['V10_BF1_EXPECT_FREEZE'] = prereg['bf1_basis']['freeze']['sha256']
    bf1 = R.import_bf1()
    bf1.freeze()
    bf1.require_gpu_gate()
    A, torch, np = bf1.A, bf1.torch, bf1.np
    check('core_and_host_hash', A.code_hash() == C.CORE_HASH and A.host_code_hash() == C.HOST_HASH)
    check('bf1_job_pin', A.check_job_pin()['status'] == 'match')
    if a.device == 'cuda':
        env = H.runtime_environment(bf1)
        partition = env['slurm_partition']
    else:
        env = A.environment('cpu')
        partition = None
    check('hardware_amended_runtime_check', a.device == 'cpu' or (env['gpu_name'] == H.GPUS[partition] and env['host'] not in H.EXCLUDED),
          gpu=env.get('gpu_name'), node=env.get('host'), partition=partition)
    bf1.check_data({'FIT'})
    guard = A.DataGuard()
    fit = A.fit_cores(guard, 1.0, 1001)
    imgs = A.load_learned_inputs(guard, fit, {'FIT'}, 'BF1 capacity hardware gate (non-scientific)', a.device)
    ctx = {'bf1': bf1, 'A': A, 'torch': torch, 'np': np, 'prereg': prereg, 'guard': guard, 'fit': fit, 'imgs': imgs,
           'original_config': bf1.config, 'config': R.capacity_config_factory(bf1), 'loader': R.cpu_first_loader(bf1),
           'paired': {'encoder_init': set(), 'sampler_before': set(), 'first_batch': set()}}
    bf1.config, bf1.load_checkpoint = ctx['config'], ctx['loader']
    per = {}
    tmp = Path(tempfile.mkdtemp(prefix='v10k_gate_', dir=out))
    plan = [('MF_VCS_M1', 8)] if a.quick else [(f, K) for f in C.FAMILIES for K in (8, 16, 32)]
    for family, K in plan:
        try:
            per[f'{family}_K{K}'] = config_checks(ctx, family, K, check, tmp, a.device)
        except Exception as error:
            check(f'{family}_K{K}:exception', False, error=repr(error), traceback=traceback.format_exc())
        print('GATE', family, K, flush=True)
    try:
        if a.device == 'cuda':
            runner_dry_run(check, base_freeze, hw_freeze, partition)
        else:
            for part, gpu in H.GPUS.items():
                runner_dry_run(check, base_freeze, hw_freeze, part, stub_env={'gpu_name': gpu, 'slurm_partition': part, 'host': 'nodeX'})
    except Exception as error:
        check('dry_run:exception', False, error=repr(error), traceback=traceback.format_exc())
    if not a.quick:
        check('paired_encoder_init_identical', len(ctx['paired']['encoder_init']) == 1)
        check('paired_sampler_state_identical', len(ctx['paired']['sampler_before']) == 1)
        check('paired_first_batch_identical', len(ctx['paired']['first_batch']) == 1)
    check('data_roles_FIT_only', {r['role'] for r in guard.reads} == {'FIT'}, n_reads=len(guard.reads))
    shutil.rmtree(tmp, ignore_errors=True)
    status = 'PASS' if not check.failures else 'FAIL'
    result = {'schema': 'v10.capacity_ext.hw_gate.v1', 'status': status, 'utc': C.utc(), 'job': os.environ.get('SLURM_JOB_ID'),
              'device': a.device, 'quick': a.quick, 'partition': partition, 'gpu_name': env.get('gpu_name'), 'node': env.get('host'),
              'hw_freeze_sha256': hw_freeze, 'base_freeze_sha256': base_freeze, 'seconds': time.time() - t_start,
              'scope': 'execution / numerical gate of the K8/K32 path on this GPU type; not scientific evidence',
              'n_checks': len(check.checks), 'n_failed': len(check.failures), 'per_config': per, 'checks': check.checks,
              'environment': env, 'data_reads': guard.reads}
    C.write(out / 'HW_GATE.json', result)
    if a.device == 'cuda' and not a.quick:
        C.write(H.DIR / 'gpu_gates' / f"{partition}_{os.environ['SLURM_JOB_ID']}.json", result)
    print('HW_GATE_STATUS', status, partition, env.get('gpu_name'), len(check.checks), 'checks', len(check.failures), 'failed', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
