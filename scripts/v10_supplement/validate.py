"""Pinned Slurm validation only; full CAL outputs never become scientific seeds."""
from __future__ import annotations

import argparse
import copy
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from v10.supplement.contract import FAMILIES, PARENT, candidates, digest, read, schedule, sha, source_inventory
from v10.supplement.records import Allocation, verified_complete, write, utc


def require(ok, why):
    if not ok: raise RuntimeError(why)


def verify_manifest():
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Validation requires Slurm')
    path = Path(os.environ['V10_VALIDATION_MANIFEST']).resolve()
    require(sha(path) == os.environ['V10_VALIDATION_SHA256'], 'Validation submission pin changed')
    manifest = read(path)
    require(manifest['status'] == 'VALIDATION_ONLY_NOT_A_SCIENTIFIC_RELEASE', 'Wrong validation scope')
    inventory = source_inventory()
    require(manifest['source_files'] == inventory and manifest['code_sha256'] == digest(inventory), 'Source changed')
    require(str(REPO.resolve()) == manifest['source_root'], 'Different source checkout')
    for key in ('data_lock', 'parent_protocol'):
        require(sha(manifest[key]['path']) == manifest[key]['sha256'], 'Changed pinned input: ' + key)
    return path, manifest


def report_base(manifest, family=None):
    return dict(utc=utc(), job=os.environ['SLURM_JOB_ID'], family=family,
                code_sha256=manifest['code_sha256'], validation_manifest_sha256=os.environ['V10_VALIDATION_SHA256'],
                purpose='Implementation validation only; excluded from method selection and scientific results',
                held_test_read=False, dev_read=False)


def expect_failure(call, errors=(RuntimeError,)):
    try: call()
    except errors: return
    raise AssertionError('Invalid input was silently accepted')


def gate(root, manifest, family=None):
    import numpy as np
    import torch
    import run as R
    from v10.supplement import checkpoints as CP
    from v10.supplement.learned import MLPRigidEnergy
    from v10.supplement.registration import RegistrationSession
    from v10.supplement.geometry_validation import native_gradient_check
    from v10.supplement.training import new_trainer, rng_state
    R.initialize_science(); A = R.A
    environment = R.runtime_environment()
    env = dict(os.environ, V10_TEST_DEVICE='cuda')
    output = root / 'CUDA_NUMERICS.txt'
    with output.open('w') as stream:
        rc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                             'tests/v10/test_supplement.py', 'tests/v10/test_rulsif_v2.py'], cwd=REPO, env=env,
                             stdout=stream, stderr=subprocess.STDOUT).returncode
    require(rc == 0, 'CUDA numerical identities failed; see ' + str(output))
    data = read(manifest['data_lock']['path'])
    reports = []
    selected_families = (family,) if family is not None else FAMILIES
    for family in selected_families:
        directory = root / family; directory.mkdir(parents=True, exist_ok=True)
        guard = A.DataGuard(); R.check_data(data, guard, {'FIT', 'CAL'})
        candidate = candidates(family)[0]
        ident = R.identity(os.environ['V10_VALIDATION_SHA256'], family, candidate, 1117,
                           'VALIDATION_GATE', manifest['data_lock']['sha256'])
        trainer = new_trainer(family, candidate, 1117, guard, 'cuda')
        torch.cuda.reset_peak_memory_stats()
        gradient = schedule(family)['kind'] in ('gradient_training', 'native_gradient_training')
        cp = directory / 'resume.pt'
        resume = {'kind': schedule(family)['kind']}
        if gradient:
            first = R.plain(trainer.step())
            CP.save(trainer, cp, ident)
            expected = R.plain(trainer.step())
            model = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                     for m in trainer.modules()]
            optimizers = copy.deepcopy([o.state_dict() for o in trainer.optimizers])
            local_rng = copy.deepcopy(rng_state(trainer))
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
            CP.load(trainer, cp, ident)
            actual = R.plain(trainer.step())
            for module, states in zip(trainer.modules(), model, strict=True):
                for k, v in module.state_dict().items():
                    torch.testing.assert_close(v.detach().cpu(), states[k], atol=1e-5, rtol=1e-4)
            def check_tree(x, y):
                if isinstance(x, torch.Tensor): torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
                elif isinstance(x, dict):
                    require(x.keys() == y.keys(), 'Optimizer keys changed')
                    for k in x: check_tree(x[k], y[k])
                elif isinstance(x, (list, tuple)):
                    require(len(x) == len(y), 'Optimizer length changed')
                    for a, b in zip(x, y): check_tree(a, b)
                else: require(x == y, 'Optimizer scalar changed')
            check_tree([o.state_dict() for o in trainer.optimizers], optimizers)
            require(rng_state(trainer) == local_rng and torch.equal(cpu_rng, torch.get_rng_state()), 'RNG resume mismatch')
            require(all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all())), 'CUDA RNG mismatch')
            require(trainer.n_steps == 2, 'Resume step mismatch')
            if family == 'InfoNCE_host[J_perm1]':
                require('J' not in first and first['training_objective'] == 'InfoNCE_host', 'Incorrect InfoNCE log label')
            resume.update(first=first, expected_second=expected, resumed_second=actual,
                          optimizer_and_rng_restored=True, full_fit_cores=32)
            del model, optimizers
        elif family in ('RuLSIF_exact', 'RuLSIF_exact_v2'):
            trainer.step(); CP.save(trainer, cp, ident); CP.load(trainer, cp, ident)
            require(len(trainer.fit_metadata['fit_cores']) == 32, 'Kernel FIT incomplete')
            resume.update(kernel_fit=trainer.fit_metadata)
        else:
            CP.save(trainer, cp, ident); CP.load(trainer, cp, ident)
            require(trainer.n_steps == 0, 'Training-free method has fake updates')
        expect_failure(lambda: CP.load(trainer, cp, {**ident, 'seed': 999}))
        # Resume complete state, then use the full common searcher on a CAL image.
        before_rng = copy.deepcopy(rng_state(trainer))
        flags = [(m.training, [p.requires_grad for p in m.parameters()]) for m in trainer.modules()]
        with CP.evaluation_state(trainer):
            core = guard.included('CAL')[0]
            session = RegistrationSession(family, trainer, guard, 'CAL', [core], 'cuda', candidate)
            energy, prep = session.energy(core, 0)
            p = torch.tensor([.37, .23, -.41], device='cuda', requires_grad=True)
            value = energy(p); grad = torch.autograd.grad(value, p)[0]
            require(torch.isfinite(grad).all().item(), 'Non-finite geometry gradient')
            if family in ('MI_native', 'NMI_native', 'CoMIR_native'):
                geometry_check = native_gradient_check(energy.energy, p, grad)
                fd = p.new_tensor(geometry_check['fd_half_fp64'])
            else:
                h = .01; eye = torch.eye(3, device='cuda') * h
                with torch.no_grad():
                    fd = torch.stack([(energy(p.detach()+e) - energy(p.detach()-e))/(2*h) for e in eye])
                torch.testing.assert_close(grad, fd, atol=5e-4, rtol=.05)
                geometry_check = {'method': 'fp32_central_difference', 'step': .01, 'atol': 5e-4, 'rtol': .05}
            if family in ('MLP_VCS_perm8', 'InfoNCE_host[J_perm1]'):
                raw = energy.energy
                rng1 = np.random.default_rng(8117); rng2 = np.random.default_rng(8117)
                ref = session.reference
                other = MLPRigidEnergy(ref, raw.mov, rng1, encoder=trainer.eV, critic=trainer.critic,
                                       draws=1)
                native = A.Q.RigidEnergy(torch, torch.nn.functional, 'critic', ref, raw.mov, rng2,
                                        eV=trainer.eV, critic=trainer.critic)
                torch.testing.assert_close(other(p), native(p), atol=1e-5, rtol=1e-4)
                require(rng1.bit_generator.state == rng2.bit_generator.state, 'Host permutation RNG changed')
                extra = MLPRigidEnergy(ref, raw.mov, np.random.default_rng(8117), encoder=trainer.eV,
                                      critic=trainer.critic, draws=8, extra_seed=917)
                require(torch.equal(other.perm, extra.permutations[0]), 'perm8 first draw differs from host')
            timing = A.register_timed(energy, 'cuda')
            require(timing['n_grid'] == 21 * 508, 'Common search grid changed')
        require(rng_state(trainer) == before_rng, 'Evaluation consumed training RNG')
        require(flags == [(m.training, [p.requires_grad for p in m.parameters()]) for m in trainer.modules()],
                'Evaluation did not restore training modes/grad flags')
        if gradient:
            trainer.step(); require(trainer.n_steps == 3, 'Training failed after evaluation')
        expect_failure(lambda: guard.check(guard.held()[0], {'FIT', 'CAL'}, 'negative held access'), (PermissionError,))
        require({r['role'] for r in guard.reads} <= {'FIT', 'CAL'}, 'Unexpected data role')
        report = {**report_base(manifest, family), 'status': 'PASS', 'stage': 'T01_AND_EXECUTION_GATE',
                  'environment': environment, 'resume': resume, 'full_search_timing': timing, 'preparation': prep,
                  'geometry_gradient': grad.detach().cpu().tolist(), 'finite_difference': fd.cpu().tolist(),
                  'geometry_fd_tolerance': geometry_check,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'data_reads': guard.reads}
        write(directory / 'GATE.json', report); reports.append({'family': family, 'path': str(directory/'GATE.json'),
                                                               'sha256': sha(directory/'GATE.json')})
        print('GATE_PASS', family, flush=True)
        del trainer, session, energy, value, grad
        gc.collect(); torch.cuda.empty_cache()
    write(root / 'GATE.json', {**report_base(manifest), 'status': 'PASS', 'families': reports,
                              'requested_families': list(selected_families),
                              'cuda_numerics_sha256': sha(output)})


def full(root, manifest, family):
    import torch
    import run as R
    R.initialize_science()
    environment = R.runtime_environment()
    data_sha = manifest['data_lock']['sha256']
    candidate = candidates(family)[0]  # Prespecified first candidate, never picked using results.
    ident = R.identity(os.environ['V10_VALIDATION_SHA256'], family, candidate, 1117, 'VALIDATION_FULL_CAL', data_sha)
    allocation = Allocation(root, family, 'VALIDATION_FULL_CAL')
    directory = root / 'unit'
    status = 'FAIL'; started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        result = R.run_unit(directory, ident, read(manifest['data_lock']['path']), allocation, environment)
        require(result['steps'] == schedule(family)['steps'], 'Full training incomplete')
        require(all(row['metrics']['n'] == 80 for row in result['cal_checkpoints']), 'Not complete CAL8 x 10')
        require({r['role'] for r in read(directory/'DATA_READS.json')} <= {'FIT', 'CAL'}, 'Unexpected data roles')
        require(verified_complete(directory, ident) == result, 'Reuse verification mismatch')
        expect_failure(lambda: verified_complete(directory, {**ident, 'seed': 1}))
        # Reject a false COMPLETE without changing the actual result or checkpoint.
        negative = root / 'negative_fixture'; negative.mkdir(exist_ok=True)
        fake = copy.deepcopy(result); fake['steps'] += 1
        write(negative/'COMPLETE.json', fake)
        expect_failure(lambda: verified_complete(negative, ident))
        status = 'PASS'
        common = {**report_base(manifest, family), 'status': status, 'candidate': candidate,
                  'full_training_steps': result['steps'], 'cal_checkpoints': len(result['cal_checkpoints']),
                  'registrations': 80 * len(result['cal_checkpoints']),
                  'seconds': time.monotonic()-started, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                  'full_complete': str(directory/'COMPLETE.json'), 'full_complete_sha256': sha(directory/'COMPLETE.json'),
                  'scientific_release_created': False, 'environment': environment}
        write(root/'T02.json', {**common, 'stage': 'T02'})
        write(root/'RUNNER.json', {**common, 'stage': 'RUNNER', 'reuse_and_false_complete_rejection': 'PASS'})
    finally:
        allocation.record('END', status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('gate', 'full'))
    parser.add_argument('--family', choices=FAMILIES)
    args = parser.parse_args()
    pin, manifest = verify_manifest()
    root = pin.parent/'runs'/os.environ['SLURM_JOB_ID']
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == 'gate': gate(root, manifest, args.family)
        else:
            require(args.family is not None, 'Full validation requires family')
            full(root, manifest, args.family)
    except BaseException as exc:
        write(root/'FAILURE.json', {**report_base(manifest, args.family), 'status': 'FAIL', 'stage': args.stage,
                                   'error': repr(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__': main()
