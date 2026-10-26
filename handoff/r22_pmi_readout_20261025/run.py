"""PMI readout on the fixed 30000-step shared-cosine geometry models (eval only, DEV, same-job 2x2 COS/PMI factorial).

Theory (PDF eq. 35): the optimal bounded critic is T* = tanh(PMI/2). In this family T = cos for every arm; vcs_exact, vcs_perm
and the (1+T)/2 BCE share that optimum, so mean 2 atanh(T) is the summed log-likelihood-ratio pose score. NCE scores cos/0.07,
for which atanh(cos) is not its PMI: it is the control. PMI clip EPS = 1e-4 as in R22 C (fixed before any run).
Host searcher, grid, 3 starts and 100 Adam steps unchanged; only coarse (grid_values) and local (__call__/value) readouts differ.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import types

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE / 'handoff/r22_geometry_train_20261023'))
import train as T
R, A, torch = T.R, T.A, T.torch
_spec = importlib.util.spec_from_file_location('stage_v2', BASE / 'handoff/r22_stage_readout_v2_20261024/run.py')
S = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S)
LONG = BASE / 'handoff/r22_geometry_long_v2_20261024'
NOISY = BASE / 'handoff/r22_geometry_noise_long_20261024'
ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
POLICIES = (('COS', 'COS'), ('PMI', 'COS'), ('COS', 'PMI'), ('PMI', 'PMI'))
EPS = 1e-4


def verify_freeze():
    for rel, digest in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        assert R.sha(BASE / rel) == digest, rel
    return R.sha(HERE / 'FREEZE.json')


def readout_energy(self, dv):
    if self.geometry_readout == 'PMI':
        u, v = T.normalized(self.u[None], dv)
        return (2 * torch.atanh((u * v).sum(-1).clamp(-1 + EPS, 1 - EPS))).mean(-1)
    return T.cosine_energy(self, dv)


def unit_dir(ds, arm, seed, noise):
    return (LONG / 'runs' / f'{ds}_{arm}_{seed}') if noise is None else (NOISY / 'runs' / f'{ds}_{arm}_{seed}_n{round(noise*100)}')


def build(ds, arm, seed, noise=None):
    src = unit_dir(ds, arm, seed, noise)
    done = json.loads((src / 'COMPLETE.json').read_text())
    assert done['status'] == 'COMPLETE' and done['final_steps'] == 30000 and done['identity']['arm'] == arm
    assert done['identity'].get('noise') == noise
    ckpt = src / 'checkpoints/step_30000.pt'
    assert R.sha(ckpt) == done['final_checkpoint_sha256']
    u = R.load_unit(BASE / 'handoff/r22_c3_codex_20261023/units' / f'r22_pmi3_{ds}_MLPp1_s{seed}.json')
    guard, H = R.make_guard(u)
    fit = A.fit_cores(guard, 1., seed) if H is None else H.fit_windows(guard)
    cfg = A.TrainConfig(method='MLP_perm1', seed=seed, encoder_lr=3e-4, head_lr=3e-4,
                        n_sc=16, n_pos=128, crop=160, grad_spot_every=100)
    if noise is None:
        t = T.GeometryTrainer(cfg, {}, fit, 'cuda', arm)
    else:  # mismatch-trained models: original class, which snapshots NOISE at construction (corruption only affects training draws)
        spec = importlib.util.spec_from_file_location('noise_train', BASE / 'handoff/r22_geometry_noise_20261023/train.py')
        N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N); N.NOISE = noise
        t = N.GeometryTrainer(cfg, {}, fit, 'cuda', arm); assert t.noise == noise
    R.CP.load(t, ckpt, done['identity'])
    assert t.n_steps == 30000
    return t, u, guard, H, done


def smoke_checks(energy, inner, truth):
    """Real-stack checks: COS path is the original function; PMI equals the manual formula; values/gradients finite."""
    p = torch.tensor([math.radians(truth[0]) * A.Q.RHO, truth[1], truth[2]], device='cuda', requires_grad=True)
    inner.geometry_readout = 'COS'
    inner.j_critic = types.MethodType(T.cosine_energy, inner); z0 = energy(p); g0 = torch.autograd.grad(z0, p)[0]
    inner.j_critic = types.MethodType(readout_energy, inner); z = energy(p); g = torch.autograd.grad(z, p)[0]
    # Repeated GPU encoder passes are not bitwise reproducible on every card (deterministic algorithms off, as in the host);
    # the COS branch is the original function, so equality is checked at the stage-readout gate tolerance.
    assert torch.allclose(z, z0, atol=1e-5, rtol=1e-4) and torch.allclose(g, g0, atol=1e-5, rtol=1e-4)
    cos_rerun_diff = dict(value=float((z - z0).abs().detach()), grad=float((g - g0).abs().max()))
    inner.geometry_readout = 'PMI'; zp = energy(p); gp = torch.autograd.grad(zp, p)[0]
    assert torch.isfinite(zp) and torch.isfinite(gp).all()
    ts = torch.tensor([[0., 0.], [1., -1.]], device='cuda')
    with torch.no_grad():
        inner.geometry_readout = 'PMI'; grid = energy.grid_values(0., ts)
        assert torch.isfinite(grid).all()
    return dict(cos_equals_original_within_tolerance=True, cos_rerun_abs_diff=cos_rerun_diff, pmi_value=float(zp.detach()), pmi_grad_norm=float(gp.norm()), cos_value=float(z.detach()))


def run(ds, arm, seed, noise=None, smoke=False):
    freeze = verify_freeze()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4); started = time.time(); job = os.environ['SLURM_JOB_ID']
    name = f'{ds}_{arm}_{seed}' + ('' if noise is None else f'_n{round(noise*100)}') + ('_smoke' if smoke else '')
    out = HERE / ('smoke' if smoke else 'runs') / name; out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out / 'COMPLETE.json').exists(), 'Never rerun completed unit'
        assert not list(out.glob('ATTEMPT_*.json')), 'Failed attempts are reviewed, not silently retried'
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', job=job, start=started))
        t, u, guard, H, done = build(ds, arm, seed, noise)
        before = A.state_hash(t.eU, t.eV)
        cores = sorted(guard.included('DEV')); cores = cores[:2] if smoke else cores
        expected = [(c, i) for c in cores for i in (range(1) if smoke else range(10))]
        rows, technical = [], []
        with R.CP.evaluation_state(t):
            session = R.make_session(u, t, guard, H, 'DEV', cores, 'cuda')
            for index, (core, i) in enumerate(expected):
                energy, _ = session.energy(core, i); inner = R.PE.inner(energy)
                truth = session.perturbations[core][i]
                if smoke: technical.append(dict(core=core, checks=smoke_checks(energy, inner, truth)))
                inner.j_critic = types.MethodType(readout_energy, inner)
                values = {}; shift = (seed + index) % len(POLICIES)
                order = POLICIES[shift:] + POLICIES[:shift]
                for coarse, local in order:
                    wrapped = S.StageEnergy(energy, inner, coarse, local)
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    row = R.plain(A.registration_record(wrapped, 'cuda', truth))
                    torch.cuda.synchronize(); row['synchronized_wall_seconds'] = time.perf_counter() - tick
                    assert all(math.isfinite(row[k]) for k in ('err_full', 'F_hat', 'F_true'))
                    assert wrapped.grid_calls == len(A.Q.grid_points()[0])
                    assert wrapped.local_calls == A.Q.P.N_STARTS * (A.Q.P.N_ADAM + 1) and wrapped.value_calls == 1
                    row.update(grid_calls=wrapped.grid_calls, local_calls=wrapped.local_calls,
                               grid_values_sha256=hashlib.sha256(''.join(wrapped.grid_digests).encode()).hexdigest())
                    values[coarse + '__' + local] = row
                assert len({r['n_grid'] for r in values.values()}) == 1
                record = dict(arm=arm, core=core, scene=guard.unit_of(core) if H else core, perturbation=i,
                              policies=values, order=[a + '__' + b for a, b in order])
                rows.append(record); R.append(out / 'ROWS.jsonl', record)
                R.write(out / 'PROGRESS.json', dict(rows=len(rows), expected=len(expected), utc=datetime.now(timezone.utc).isoformat()))
                if (index + 1) % 20 == 0: print(name, index + 1, '/', len(expected), flush=True)
                del energy
        assert A.state_hash(t.eU, t.eV) == before
        assert {x['role'] for x in guard.reads} <= {'FIT', 'CAL', 'DEV'}
        summaries = {}
        for policy in [a + '__' + b for a, b in POLICIES]:
            scenes = defaultdict(list)
            for r in rows: scenes[r['scene']].append(r['policies'][policy]['err_full'])
            summaries[policy] = dict(
                success9=statistics.mean(statistics.mean(x < 9 for x in v) for v in scenes.values()),
                success42=statistics.mean(statistics.mean(x < 42 for x in v) for v in scenes.values()),
                median_err=statistics.median(x for v in scenes.values() for x in v), n=len(rows), scenes=len(scenes),
                failures42=sum(x >= 42 for v in scenes.values() for x in v),
                seconds=sum(r['policies'][policy]['synchronized_wall_seconds'] for r in rows))
        R.write(out / 'INPUT_TRACE.json', guard.reads)
        result = dict(status='COMPLETE', freeze_sha256=freeze, smoke=smoke, ds=ds, arm=arm, seed=seed, noise=noise, job=job,
                      seconds=time.time() - started, summaries=summaries, technical_checks=technical,
                      source=dict(completion=str(unit_dir(ds, arm, seed, noise) / 'COMPLETE.json'),
                                  checkpoint_sha256=done['final_checkpoint_sha256'], dev_long_COS=done['dev_long']['COS']),
                      model_hash=before, weights_unchanged=True, rows_sha256=R.sha(out / 'ROWS.jsonl'),
                      environment=R.environment(), gpu=torch.cuda.get_device_name(),
                      note='Fixed 30000-step checkpoints, no selection; DEV reused (exploratory); search non-deterministic, paired in job.')
        R.write(out / 'COMPLETE.json', result)
        R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', job=job, seconds=time.time() - started))
        print('PMI_READOUT_COMPLETE', name, json.dumps({k: round(v['success9'], 4) for k, v in summaries.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--ds', choices=('bf', 'epfl'), required=True)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--noise', type=float, choices=(.2, .4)); p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    run(a.ds, a.arm, a.seed, a.noise, a.smoke)
