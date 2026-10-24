#!/usr/bin/env python3
"""C3: fixed-checkpoint, same-job readout ablation; CAL/DEV only."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
HOST = BASE / 'ssl_pilot_v10_r22c/v10/r22_pmi'
VARIANTS = ('J', 'PMI', 'PMI_C2', 'EPT', 'POS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_freeze():
    lock = json.loads((HERE / 'FREEZE.json').read_text())
    for path, digest in lock['files'].items():
        if sha(BASE / path) != digest:
            raise RuntimeError('Frozen source changed: ' + path)
    return sha(HERE / 'FREEZE.json')


def setup():
    sys.path.insert(0, str(HOST))
    import run_unit as R
    import torch
    PE = R.PE
    # Per-position clipping precedes averaging. This is an application readout,
    # not a replacement variational bound or an exact PMI estimator.
    def clipped(T):
        return (2 * torch.atanh(T.clamp(-1 + PE.EPS, 1 - PE.EPS))).clamp(-2., 2.).mean(-1)

    def positive(T):
        return (T - .5 * T.square()).mean(-1)

    PE._ROWS.update(PMI_C2=clipped, POS=positive)
    PE.VARIANTS = VARIANTS
    original_register = PE.register_variants
    PE.register_variants = lambda energy, device, truth: original_register(energy, device, truth, variants=VARIANTS)
    original_environment = R.environment

    def environment():
        return {**original_environment(), 'c3_freeze_sha256': sha(HERE / 'FREEZE.json'),
                'c3_runner_sha256': sha(__file__), 'clip_nats': 2.0}
    R.environment = environment
    return R, torch


def cpu_check(R, torch):
    """Exercise both host energy classes, preserving J values and gradients."""
    from v10.supplement.learned import MLPRigidEnergy
    torch.manual_seed(20261023)
    checks = []
    for cls in (R.A.Q.RigidEnergy, MLPRigidEnergy):
        e = cls.__new__(cls)
        e.torch, e.kind = torch, 'critic'
        e.u = torch.randn(11, 16, dtype=torch.float64)
        e.kw = {'critic': torch.nn.Sequential(torch.nn.Linear(32, 7), torch.nn.Tanh(), torch.nn.Linear(7, 1)).double()}
        e.perm = torch.randperm(11)
        e.permutations = e.perm[None]
        dv = torch.randn(3, 11, 16, dtype=torch.float64, requires_grad=True)
        original = e.j_critic(dv)
        grad = torch.autograd.grad(original.sum(), dv)[0]
        T = R.PE.positive_scores(e, dv)
        for variant in VARIANTS[1:]:
            R.PE.set_variant(e, variant)
            out = e.j_critic(dv)
            assert out.shape == (3,) and torch.isfinite(out).all()
            assert torch.isfinite(torch.autograd.grad(out.sum(), dv, retain_graph=True)[0]).all()
            if variant == 'POS':
                assert torch.equal(out, (T - .5 * T.square()).mean(-1))
            if variant == 'PMI_C2':
                expected = [[max(-2., min(2., 2 * __import__('math').atanh(max(-1+1e-4, min(1-1e-4, float(v)))))) for v in row] for row in T.detach()]
                assert torch.allclose(out, torch.tensor(expected, dtype=torch.float64).mean(-1), atol=1e-12, rtol=1e-12)
        R.PE.set_variant(e, 'J')
        restored = e.j_critic(dv)
        assert 'j_critic' not in e.__dict__ and torch.equal(original, restored)
        assert torch.equal(grad, torch.autograd.grad(restored.sum(), dv)[0])
        # POS - J is exactly the removed Q first + second moment, up to rounding.
        U = e.u[None].expand(3, -1, -1)
        tq = torch.tanh(e.kw['critic'](torch.cat((U, dv[:, e.perm]), -1))[..., 0])
        assert torch.allclose((T - .5*T.square()).mean(-1) - original, (tq + .5*tq.square()).mean(-1), atol=1e-12)
        checks.append(cls.__name__ + ': readouts, gradients, J restoration, Q decomposition')
    edge = torch.tensor([[-1., -.9, -.2, 0., .4, .99, 1.]], dtype=torch.float64, requires_grad=True)
    value = R.PE._ROWS['PMI_C2'](edge)
    assert abs(value.item()) <= 2 and torch.isfinite(torch.autograd.grad(value.sum(), edge)[0]).all()
    torch.autograd.gradcheck(R.PE._ROWS['PMI_C2'], (torch.tensor([[-.9, -.3, .2, .95]], dtype=torch.float64, requires_grad=True),))
    checks.append('finite endpoints and finite-difference clip gradient away from kinks')
    units = []
    for path in sorted((HERE / 'units').glob('*.json')):
        u = R.load_unit(path)
        assert u['key'] in ('MLPp1', 'InfoNCE') and u['c3_protocol_sha256'] == sha(HERE / 'PROTOCOL.json')
        units.append(u['job_name'])
    assert len(units) == 12 and len(set(units)) == 12
    R.write(HERE / 'CPU_CHECK.json', {'status': 'PASS', 'checks': checks, 'units': units,
                                    'freeze_sha256': sha(HERE / 'FREEZE.json'), 'job': os.environ['SLURM_JOB_ID'], 'utc': R.utc()})
    print('CPU_CHECK PASS', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'run'))
    ap.add_argument('--unit')
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical work must run through Slurm')
    freeze = verify_freeze()
    R, torch = setup()
    if args.mode == 'check':
        return cpu_check(R, torch)
    gate = json.loads((HERE / 'CPU_CHECK.json').read_text())
    assert gate['status'] == 'PASS' and gate['freeze_sha256'] == freeze
    unit_path = Path(args.unit).resolve()
    assert unit_path.parent == HERE / 'units'
    u = R.load_unit(unit_path)
    assert u['c3_protocol_sha256'] == sha(HERE / 'PROTOCOL.json')
    out = Path(u['out']); out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    receipt = {'job': os.environ['SLURM_JOB_ID'], 'name': u['job_name'], 'node': os.environ.get('SLURMD_NODENAME'),
               'partition': os.environ.get('SLURM_JOB_PARTITION'), 'freeze_sha256': freeze, 'start_unix': start,
               'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None, 'exit_code': None}
    receipt_path = HERE / 'receipts' / (receipt['job'] + '.json')
    R.write(receipt_path, receipt)
    try:
        with (out / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def stop(signum, frame):
                raise InterruptedError('signal ' + str(signum) + '; completed rows retained')
            signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGUSR1, stop)
            R.run_unit(u, out, deadline=time.monotonic() + 3.85 * 3600)
            receipt['exit_code'] = 0
            print('COMPLETE', u['job_name'], flush=True)
    except BaseException as exc:
        receipt['exit_code'] = 1
        R.write(out / ('FAIL_' + receipt['job'] + '.json'), {'error': repr(exc), 'traceback': traceback.format_exc(), 'utc': R.utc()})
        raise
    finally:
        receipt.update(end_unix=time.time(), allocated_seconds=time.time() - start)
        R.write(receipt_path, receipt)
        # Scientific aggregation also stays inside this Slurm allocation.
        if receipt['exit_code'] == 0:
            import summarize
            summarize.main()


if __name__ == '__main__':
    main()
