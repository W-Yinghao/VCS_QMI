"""CPU smoke of the K override on the real BF1 host (Slurm CPU job only; execution check, never scientific evidence).

Real FIT data through the BF1 DataGuard (role FIT only), the frozen BF1 runner functions and host encoders, seed 1001 with each
family's inherited lr / cap, K in (8, 16, 32):
  * TrainConfig == BF1 config with only K replaced; MFHead widths, W shape and zeros; AdamW groups (perm1/M0: W group with the tied
    lr and decay 0; M1: W solver-owned, not optimised)
  * paired design: identical encoder initialisation and identical host batch stream at every K
  * one real training step: finite loss / J / gradient; perm1/M0 |W| <= 1 after projection; M1 zero-head reference ||Delta||_1
    recomputed independently in fp64, threshold = 0.01 g_ref, stop rule vs cap, FW gap at the returned W recomputed in fp64
  * moments (M0/M1 objective) == dense V reference and perm1 == explicit permutation pairs, at a random box W on the step's features
  * BF1 save_checkpoint -> step -> CPU-first load (torch.load spied: map_location='cpu') -> step reproduces the uninterrupted step;
    a different-K identity and a different-width trainer are refused
  * registration energy (MFRigidEnergy, frozen modules) on one FIT core: one grid angle x 64 shifts; rows == moments/perm1
    functions == dense / explicit references; geometry gradient finite
"""
import argparse
import copy
import dataclasses
import json
import math
import os
import tempfile
import time
import traceback
from pathlib import Path

import capacity_common as C
import run as R


class Recorder:
    def __init__(self):
        self.checks, self.failures = [], []

    def __call__(self, name, ok, **detail):
        rec = {'check': name, 'pass': bool(ok), **json.loads(json.dumps(detail, default=str))}
        self.checks.append(rec)
        if not ok:
            self.failures.append(rec)
            print('FAIL', name, detail, flush=True)
        return ok


def fp64_head_reference(scenes, W):
    """Independent fp64 Delta, ||Delta||_1 and FW gap at W for V-mode scenes with weights 1/n_sc (dense formulas)."""
    import torch
    n_sc = len(scenes)
    delta = H = None
    W = W.detach().double()
    for s in scenes:
        a, b = s.a.detach().double(), s.b.detach().double()
        n = a.shape[0]
        d = a.T @ b / n - torch.outer(a.mean(0), b.mean(0))
        t = ((a @ W) * b).sum(-1)
        h = (a * t[:, None]).T @ b / n + (a.T @ a) @ W @ (b.T @ b) / (n * n)
        delta = d / n_sc if delta is None else delta + d / n_sc
        H = h / n_sc if H is None else H + h / n_sc
    g = delta - H
    gap = torch.where(g >= 0, g * (1 - W), -g * (1 + W)).sum()
    return delta, float(delta.abs().sum()), float(gap)


def runner_glue_dry_run(prereg, freeze_sha, check):
    """run.py main() end to end in a temporary bundle/output root, with only BF1 runtime_check (needs CUDA) and run_unit (the
    3000-step science) stubbed: freeze/prereg/CHECKS gates, Slurm name/partition/limit checks, admission, BF1 imports, freeze,
    GPU gate, code hashes, job pin, identity, the two patches, resume truncation, runtime record, refusal and failure paths."""
    from unittest import mock
    unit = C.unit_by_key(prereg, 'MF_VCS_M1_K32_seed1002')
    job = os.environ['SLURM_JOB_ID']
    captured = {}

    class StopBeforeTraining(Exception):
        pass

    real_import = R.import_bf1

    def fake_import():
        bf1 = real_import()

        def runtime_check(device):
            env = bf1.A.environment('cpu')
            env.update(gpu_name='NVIDIA L40S', host='node52', dry_run_stub=True)
            return env

        def run_unit(ident, directory, allocation, env):
            captured.update(ident=ident, directory=Path(directory), K=bf1.config(ident).K, loader=bf1.load_checkpoint.__qualname__,
                            steps=[r['step'] for r in C.lines(Path(directory) / 'TRAIN_STEPS.jsonl')],
                            allocation=type(allocation).__name__)
            raise StopBeforeTraining()
        bf1.runtime_check, bf1.run_unit = runtime_check, run_unit
        return bf1

    with tempfile.TemporaryDirectory() as tmp:
        bundle, out = Path(tmp) / 'bundle', Path(tmp) / 'out'
        bundle.mkdir()
        for name in ('FREEZE.json', 'PREREGISTRATION.json'):
            (bundle / name).write_bytes((C.BUNDLE / name).read_bytes())
        C.write(bundle / 'CHECKS.json', {'status': 'PASS', 'freeze_sha256': freeze_sha, 'note': 'dry-run stand-in'})
        env = {'SLURM_JOB_NAME': unit['job_name'], 'SLURM_JOB_PARTITION': 'L40S', 'V10K_TIME_LIMIT_RAW': '3:00:00',
               'V10K_SCHED_START_RAW': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - 5)),
               'V10K_BATCH_START': str(time.time() - 2), 'V10_CAPACITY_FREEZE': freeze_sha}
        with mock.patch.object(C, 'BUNDLE', bundle), mock.patch.object(C, 'OUT', out), \
                mock.patch.object(R, 'import_bf1', fake_import), mock.patch.dict(os.environ, env):
            ident = C.expected_identity(unit, prereg, freeze_sha)
            d = out / unit['key']
            C.write(d / 'IDENTITY.json', ident)
            C.write(d / 'checkpoints' / 'step_0250.json', {'step': 250})
            (d / 'TRAIN_STEPS.jsonl').write_text(''.join(json.dumps({'step': s}) + '\n' for s in range(1, 301)))
            try:
                R.main(unit['key'])
                stopped = False
            except StopBeforeTraining:
                stopped = True
            events = [r['event'] for r in C.lines(bundle / 'COST_LEDGER.jsonl')]
            adm = C.lines(bundle / 'ADMISSIONS.jsonl')
            runtime = C.read(bundle / 'runtime' / f'{job}.json') if (bundle / 'runtime' / f'{job}.json').exists() else {}
            check('dry_run:reached_run_unit_with_K32_and_both_patches', stopped and captured.get('K') == 32
                  and captured.get('ident') == ident and captured.get('directory') == d
                  and captured.get('loader', '').startswith('cpu_first_loader') and captured.get('allocation') == 'Allocation',
                  captured={k: (str(v) if isinstance(v, Path) else v) for k, v in captured.items() if k not in ('ident', 'steps')})
            check('dry_run:resume_cut_to_checkpoint', captured.get('steps') == list(range(1, 251))
                  and (d / f'TRAIN_STEPS.pre_resume_{job}.jsonl').exists())
            check('dry_run:records', events == ['START', 'RESUME_PREPARED', 'RUNTIME_VERIFIED', 'END']
                  and [a['decision'] for a in adm] == ['ADMITTED'] and adm[0]['reserved_seconds'] == 10800 + C.RESERVE_MARGIN_SECONDS
                  and runtime.get('identity') == ident and runtime.get('train_config', {}).get('K') == 32
                  and (bundle / 'failures' / f'{job}.json').exists(), events=events)
            for j in range(10):
                C.append(bundle / 'ADMISSIONS.jsonl', {'job': f'x{j}', 'unit': f'other{j}', 'decision': 'ADMITTED', 'reserved_seconds': 10920})
            rc = R.main(unit['key'])
            check('dry_run:budget_refusal', rc == C.EXIT_REFUSED and (bundle / 'refusals' / f'{job}.json').exists()
                  and (bundle / 'budget' / f"{unit['key']}.json").exists(), rc=rc)
            for label, change in (('job_name', {'SLURM_JOB_NAME': 'v10k_wrong'}), ('partition', {'SLURM_JOB_PARTITION': 'H100'}),
                                  ('time_limit', {'V10K_TIME_LIMIT_RAW': '1:30:00'})):
                with mock.patch.dict(os.environ, change):
                    try:
                        R.main(unit['key'])
                        refused = False
                    except RuntimeError:
                        refused = True
                check(f'dry_run:refuses_wrong_{label}', refused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    check = Recorder()
    freeze_sha = os.environ['V10_CAPACITY_FREEZE']
    prereg = C.validate_prereg(C.frozen(freeze_sha))
    os.environ['V10_BF1_EXPECT_FREEZE'] = prereg['bf1_basis']['freeze']['sha256']
    bf1 = R.import_bf1()
    bf1.freeze()
    bf1.require_gpu_gate()
    A, torch, np = bf1.A, bf1.torch, bf1.np
    MF = A.MF
    check('core_and_host_hash', A.code_hash() == C.CORE_HASH and A.host_code_hash() == C.HOST_HASH)
    pin = A.check_job_pin()
    check('bf1_job_pin', pin['status'] == 'match', pin=pin)
    env = A.environment('cpu')
    bf1.check_data({'FIT'})
    guard = A.DataGuard()
    seed = 1001
    fit = A.fit_cores(guard, 1.0, seed)
    check('fit_cores', len(fit) == 32)
    imgs = A.load_learned_inputs(guard, fit, {'FIT'}, 'BF1 capacity K smoke (CPU, non-scientific)', 'cpu')
    original_config = bf1.config
    config = R.capacity_config_factory(bf1)
    bf1.config = config                                  # the same two patches run.py applies; BF1 new_trainer / run_unit use them
    bf1.load_checkpoint = R.cpu_first_loader(bf1)
    loader = bf1.load_checkpoint
    real_load = torch.load
    load_calls = []

    def spy_load(*a, **k):
        load_calls.append(k.get('map_location'))
        return real_load(*a, **k)

    paired = {'encoder_init': set(), 'sampler_before': set(), 'first_batch': set()}
    per = {}
    tmp = Path(tempfile.mkdtemp(prefix='v10k_smoke_', dir=out))
    for family in C.FAMILIES:
        cand = prereg['families'][family]['candidate']
        method = C.METHODS[family]
        for K in (8, 16, 32):
            tag = f'{family}_K{K}'
            row = {'family': family, 'K': K, 'lr': cand['lr'], 'cap': cand['cap']}
            try:
                ident = {**bf1.identity(family, seed, cand['lr'], cand['cap'], 'SMOKE'), 'schema': 'v10.capacity_ext.smoke.v1', 'K': K}
                bf1.config = original_config
                resolved = R.require_only_k_differs(bf1, config, ident)
                bf1.config = config
                check(f'{tag}:config_only_K', resolved['K'] == K and resolved['method'] == method)
                t = bf1.new_trainer(ident, guard, 'cpu', imgs)   # BF1's own constructor path through the patched config
                check(f'{tag}:patched_config_used', t.cfg.K == K and dataclasses.asdict(t.cfg) == resolved)
                h = t.head
                check(f'{tag}:head_shapes', (h.left.in_features, h.left.out_features, h.right.in_features, h.right.out_features)
                      == (16, K, 16, K) and tuple(h.W.shape) == (K, K) and not bool(h.W.any()))
                groups = {g['name']: g for g in t.opt.param_groups}
                if family == 'MF_VCS_M1':
                    check(f'{tag}:m1_W_not_optimised', set(groups) == {'encoders', 'feature_maps'} and not h.W.requires_grad
                          and all(p is not h.W for g in groups.values() for p in g['params']))
                else:
                    check(f'{tag}:W_group', set(groups) == {'encoders', 'feature_maps', 'W'} and groups['W']['params'] == [h.W]
                          and groups['W']['weight_decay'] == 0.0 and groups['W']['lr'] == cand['lr'])
                check(f'{tag}:tied_lr_and_decay', all(g['lr'] == cand['lr'] for g in groups.values())
                      and groups['encoders']['weight_decay'] == 1e-4 and groups['feature_maps']['weight_decay'] == 1e-4)
                paired['encoder_init'].add(A.state_hash(t.eU, t.eV))
                paired['sampler_before'].add(json.dumps(t.sampler.g.bit_generator.state, sort_keys=True))
                t0 = time.time()
                rec = t.step(keep=True)
                row['first_step_seconds'] = time.time() - t0
                hb = rec['_batch'][0]
                paired['first_batch'].add(json.dumps([hb.cores, hb.y0.tolist(), hb.x0.tolist(), hb.k.tolist(), hb.flip.tolist(),
                                                      float(hb.yx.sum()), hb.perm.tolist()]))
                o = rec['_out']
                scenes = o['scenes']
                check(f'{tag}:finite_step', all(math.isfinite(rec[k]) for k in ('loss', 'J', 'grad_norm_preclip')),
                      loss=rec['loss'], J=rec['J'], grad=rec['grad_norm_preclip'])
                check(f'{tag}:feature_width', tuple(o['A'].shape) == (2048, K) and tuple(o['B'].shape) == (2048, K)
                      and len(scenes) == 16 and all(tuple(s.a.shape) == (128, K) for s in scenes))
                if family == 'MF_VCS_M1':
                    g32, tau, it = rec['m1_gap_ref_zero'], rec['m1_tolerance_effective'], rec['m1_iterations']
                    delta64, g64, _ = fp64_head_reference(scenes, torch.zeros(K, K))
                    _, _, gap64 = fp64_head_reference(scenes, o['info']['m1_W'])
                    unit = A.m1_resolution_unit(torch.float32)
                    diag = rec['m1_fp64_diagnostic']
                    check(f'{tag}:m1_reference', g32 > 0 and abs(g32 - g64) <= 16 * unit, g32=g32, g64=g64)
                    check(f'{tag}:m1_threshold', math.isclose(tau, 0.01 * g32, rel_tol=1e-12, abs_tol=0.0), tau=tau)
                    stop = rec['m1_stop_reason']
                    check(f'{tag}:m1_stop_rule', (stop == 'gap_tolerance' and rec['m1_gap_final'] <= tau and it <= cand['cap'])
                          or (stop == 'iteration_budget_exhausted' and it == cand['cap']), stop=stop, iterations=it)
                    check(f'{tag}:m1_gap_final', abs(rec['m1_gap_final'] - gap64) <= 16 * unit + 1e-3 * gap64,
                          gap32=rec['m1_gap_final'], gap64=gap64)
                    if diag.get('triggered'):
                        check(f'{tag}:m1_fp64_diag_agrees', math.isclose(diag['gap_ref_zero_fp64'], g64, rel_tol=1e-9)
                              and math.isclose(diag['gap_final_fp64'], gap64, rel_tol=1e-6, abs_tol=1e-15), diag_g=diag['gap_ref_zero_fp64'])
                    check(f'{tag}:m1_W', tuple(o['info']['m1_W'].shape) == (K, K) and float(o['info']['m1_W'].abs().max()) <= 1
                          and torch.equal(h.W.detach(), o['info']['m1_W'].detach()))
                    row['m1'] = {'gap_ref_zero': g32, 'gap_ref_zero_fp64': g64, 'iterations': it, 'stop': stop}
                else:
                    check(f'{tag}:W_projected', rec['w_absmax'] <= 1 and float(h.W.abs().max()) > 0, w_absmax=rec['w_absmax'])
                gen = torch.Generator().manual_seed(1000 + K)
                Wr = torch.empty(K, K).uniform_(-1, 1, generator=gen)
                weights = A.host_weights(16)
                mom = MF.objective_terms(scenes, Wr, weights=weights)
                dense = MF.dense_objective_terms(scenes, Wr, weights=weights)
                cmp = A.compare_many([mom.p1, mom.p2, mom.q1, mom.q2, mom.j], [dense.p1, dense.p2, dense.q1, dense.q2, dense.j], 'float32')
                check(f'{tag}:moments_equal_dense', cmp['allclose'], max_abs_err=cmp['max_abs_err'])
                perm = o['perm']
                sp = MF.sampled_permutation_terms(scenes, Wr, weights=weights, permutations=[perm[s][None] for s in range(16)])
                q1 = sum(torch.einsum('ik,kl,il->i', s.a.double(), Wr.double(), s.b[perm[i]].double()).mean() / 16
                         for i, s in enumerate(scenes))
                check(f'{tag}:perm1_explicit', abs(float(sp.q1) - float(q1)) <= 1e-5 + 1e-4 * abs(float(q1)))
                # checkpoint roundtrip with the CPU-first loader
                cp = tmp / tag / 'step_0001.pt'
                bf1.save_checkpoint(t, cp, ident)
                t.step()
                expected_modules = [copy.deepcopy({k: v.detach().clone() for k, v in m.state_dict().items()}) for m in t.modules()]
                expected_opt = copy.deepcopy(t.opt.state_dict())
                expected_sampler = copy.deepcopy(t.sampler.g.bit_generator.state)
                torch.load = spy_load
                try:
                    meta = loader(t, cp, ident)
                finally:
                    torch.load = real_load
                check(f'{tag}:cpu_first_map_location', load_calls[-1] == 'cpu' and t.n_steps == 1 and meta['step'] == 1)
                t.step()
                exact = all(torch.equal(v, ref[k]) for m, ref in zip(t.modules(), expected_modules) for k, v in m.state_dict().items())
                close = all(torch.allclose(v, ref[k], atol=1e-6, rtol=1e-5)
                            for m, ref in zip(t.modules(), expected_modules) for k, v in m.state_dict().items())
                st = t.opt.state_dict()['state']
                opt_close = all(torch.allclose(st[i][n], v, atol=1e-6, rtol=1e-5) and st[i][n].device.type == 'cpu'
                                for i, s in expected_opt['state'].items() for n, v in s.items())
                check(f'{tag}:resume_reproduces_step', close and opt_close and t.sampler.g.bit_generator.state == expected_sampler,
                      bitwise=exact)
                try:
                    loader(t, cp, {**ident, 'K': 32 if K != 32 else 8})
                    refused_identity = False
                except RuntimeError:
                    refused_identity = True
                other = bf1.new_trainer({**ident, 'K': 32 if K != 32 else 8}, guard, 'cpu', imgs)
                try:
                    loader(other, cp, ident)
                    refused_width = False
                except RuntimeError:
                    refused_width = True
                check(f'{tag}:wrong_K_refused', refused_identity and refused_width)
                del other
                # registration energy on one FIT core
                core = fit[0]
                guard.log_rule(core, {'FIT'}, 'BF1 capacity K smoke: perturbation rule (non-scientific)')
                inputs = A.GuardedInputs(guard, {'FIT'}, 'BF1 capacity K smoke energy (non-scientific)', {core: A.Q.perturbations_p2(core)}, 'cpu')
                estimator = 'perm1' if method == 'MF_perm1' else 'moments'
                with bf1.evaluation_state(t):
                    ref = A.reference_descriptors(t.eU, inputs, core)
                    rng = np.random.default_rng(A.Q.seed_of(f'R6-REG-P2|pos|{core}|0'))
                    energy = A.MFRigidEnergy(ref, inputs.moving(core, 0)[1], rng, eV=t.eV, head=t.head, estimator=estimator, record=True)
                    _, ts = A.Q.grid_points()
                    tt = torch.tensor(ts[:64], dtype=torch.float32)
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
                        check(f'{tag}:energy_rows_are_grid_values', torch.equal(vals, A.mf_energy_rows(Ar, Bc, energy.W, estimator, perm=energy.perm)))
                    p = torch.tensor([0.1, 1.0, -2.0], requires_grad=True)
                    val = energy(p)
                    val.backward()
                    check(f'{tag}:geometry_gradient', bool(torch.isfinite(val)) and p.grad is not None and bool(torch.isfinite(p.grad).all()))
                del energy, inputs, t
                row['status'] = 'DONE'
            except Exception as error:
                check(f'{tag}:exception', False, error=repr(error), traceback=traceback.format_exc())
                row['status'] = 'EXCEPTION'
            per[tag] = row
            print('SMOKE', tag, row.get('status'), f"{row.get('first_step_seconds', float('nan')):.1f}s", flush=True)
    try:
        runner_glue_dry_run(prereg, freeze_sha, check)
    except Exception as error:
        check('dry_run:exception', False, error=repr(error), traceback=traceback.format_exc())
    check('paired_encoder_init_identical_across_K_and_family', len(paired['encoder_init']) == 1)
    check('paired_sampler_state_identical', len(paired['sampler_before']) == 1)
    check('paired_first_batch_identical', len(paired['first_batch']) == 1)
    check('data_roles_FIT_only', {r['role'] for r in guard.reads} == {'FIT'}, n_reads=len(guard.reads))
    status = 'PASS' if not check.failures else 'FAIL'
    result = {'schema': 'v10.capacity_ext.smoke.v1', 'status': status, 'utc': C.utc(), 'job': os.environ.get('SLURM_JOB_ID'),
              'freeze_sha256': freeze_sha, 'seconds': time.time() - t_start, 'scope': 'execution check on CPU; not scientific evidence',
              'n_checks': len(check.checks), 'n_failed': len(check.failures), 'per_config': per, 'checks': check.checks,
              'environment': env, 'data_reads': guard.reads}
    C.write(out / 'SMOKE.json', result)
    print('SMOKE_STATUS', status, len(check.checks), 'checks', len(check.failures), 'failed', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
