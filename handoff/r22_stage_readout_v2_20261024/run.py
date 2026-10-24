"""Fixed-checkpoint factorial: coarse search readout versus local refinement readout."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
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
ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
POLICIES = (('COS', 'COS'), ('J_EXACT', 'COS'), ('COS', 'J_EXACT'), ('J_EXACT', 'J_EXACT'))


def verify_freeze():
    for rel, digest in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        assert R.sha(BASE / rel) == digest, rel
    return R.sha(HERE / 'FREEZE.json')


class StageEnergy:
    """Host optimiser unchanged: grid_values uses coarse; __call__/value use local."""
    def __init__(self, energy, inner, coarse, local):
        self.energy, self.inner, self.coarse, self.local = energy, inner, coarse, local
        self.grid_calls = self.local_calls = self.value_calls = 0
        self.grid_digests = []

    def grid_values(self, theta, translations):
        self.inner.geometry_readout = self.coarse
        value = self.energy.grid_values(theta, translations)
        self.grid_calls += 1
        self.grid_digests.append(hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        return value

    def __call__(self, p):
        self.inner.geometry_readout = self.local
        self.local_calls += 1
        return self.energy(p)

    def value(self, theta, translation):
        self.inner.geometry_readout = self.local
        self.value_calls += 1
        return self.energy.value(theta, translation)


def cpu_check():
    freeze = verify_freeze()
    torch.set_num_threads(2)

    class Analytic:
        geometry_readout = 'COS'
        def center(self, device, dtype):
            return torch.tensor([2., 4., -3.] if self.geometry_readout == 'COS' else [-4., -5., 6.], device=device, dtype=dtype)
        def __call__(self, p):
            return -((p - self.center(p.device, p.dtype))**2).sum()
        def grid_values(self, theta, translations):
            u = translations.new_full((len(translations), 1), math.radians(theta) * A.Q.RHO)
            return -((torch.cat((u, translations), -1) - self.center(translations.device, translations.dtype))**2).sum(-1)
        def value(self, theta, translation):
            return float(self(torch.tensor([math.radians(theta)*A.Q.RHO, *translation])))

    checks = []
    for coarse, local in POLICIES:
        e = Analytic(); wrapped = StageEnergy(e, e, coarse, local)
        ts = torch.tensor([[1., 2.], [3., 4.]])
        actual = wrapped.grid_values(3., ts)
        e.geometry_readout = coarse
        assert torch.equal(actual, e.grid_values(3., ts))
        p = torch.tensor([1., 2., 3.], requires_grad=True)
        v = wrapped(p); g = torch.autograd.grad(v, p)[0]
        e.geometry_readout = local
        ref = e(p); gr = torch.autograd.grad(ref, p)[0]
        assert torch.equal(v, ref) and torch.equal(g, gr)
        assert wrapped.value(2., [3., 4.]) == e.value(2., [3., 4.])
        if coarse == local:
            e.geometry_readout = local
            native = A.Q.optimise_rigid(torch, e, torch.device('cpu'))
            delegated = A.Q.optimise_rigid(torch, StageEnergy(e, e, coarse, local), torch.device('cpu'))
            assert native[0] == delegated[0] and R.np.array_equal(native[1], delegated[1]) and native[2:] == delegated[2:]
        checks.append(dict(coarse=coarse, local=local, grid_value_and_local_gradient=True, pure_host_equivalence=coarse == local))
    R.write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, checks=checks))
    print('STAGE_CPU_CHECK PASS', flush=True)


def build(ds, arm, seed, device):
    source = json.loads((HERE / 'SOURCE_LOCK.json').read_text())[f'{ds}_{arm}_{seed}']
    assert R.sha(source['completion']) == source['completion_sha256']
    assert R.sha(source['checkpoint']) == source['checkpoint_sha256']
    done = json.loads(Path(source['completion']).read_text())
    assert done['status'] == 'COMPLETE' and done['identity']['arm'] == arm and done['identity']['seed'] == seed
    u = R.load_unit(BASE / 'handoff/r22_c3_codex_20261023/units' / f'r22_pmi3_{ds}_MLPp1_s{seed}.json')
    guard, H = R.make_guard(u)
    fit = A.fit_cores(guard, 1., seed) if H is None else H.fit_windows(guard)
    cfg = A.TrainConfig(method='MLP_perm1', seed=seed, encoder_lr=3e-4, head_lr=3e-4,
                        n_sc=16, n_pos=128, crop=160, grad_spot_every=100)
    trainer = T.GeometryTrainer(cfg, {}, fit, device, arm)
    R.CP.load(trainer, Path(source['checkpoint']), done['identity'])
    return trainer, u, guard, H, source


def compare_gate(energy, inner, truth):
    results = []
    for coarse, local in POLICIES:
        wrap = StageEnergy(energy, inner, coarse, local)
        ts = torch.tensor([[0., 0.], [1., -1.]], device='cuda')
        with torch.no_grad():
            z = wrap.grid_values(0., ts)
            inner.geometry_readout = coarse
            ref = energy.grid_values(0., ts)
            assert torch.allclose(z, ref, atol=1e-5, rtol=1e-4)
            grid_error = float((z-ref).abs().max())
        p = torch.tensor([math.radians(truth[0])*A.Q.RHO, truth[1], truth[2]], device='cuda', requires_grad=True)
        z = wrap(p); g = torch.autograd.grad(z, p)[0]
        inner.geometry_readout = local
        ref = energy(p); gr = torch.autograd.grad(ref, p)[0]
        assert torch.isfinite(g).all() and torch.allclose(z, ref, atol=1e-5, rtol=1e-4)
        assert torch.allclose(g, gr, atol=1e-5, rtol=1e-4)
        results.append(dict(coarse=coarse, local=local, value_error=float(abs(z-ref).detach()),
                            gradient_max_error=float((g-gr).abs().max()), grid_max_error=grid_error))
    return results


def run(ds, arm, seed, gate=False):
    freeze = verify_freeze(); cpu = json.loads((HERE / 'CPU_CHECK.json').read_text())
    assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4); started = time.time(); job = os.environ['SLURM_JOB_ID']; run_environment = R.environment()
    name = f'{ds}_gate' if gate else f'{ds}_{arm}_{seed}'
    out = HERE / ('gates' if gate else 'runs') / name; out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out / 'COMPLETE.json').exists(), 'Never rerun completed unit'
        assert not list(out.glob('ATTEMPT_*.json')), 'No automatic retries; failed attempts require review'
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', job=job, start=time.time()))
        role = 'CAL' if gate else 'DEV'
        summaries = {}; technical = []; input_records = []; model_records = []
        for trained_arm in (ARMS if gate else (arm,)):
            t,u,guard,H,source = build(ds, trained_arm, seed, 'cuda')
            before = A.state_hash(t.eU, t.eV)
            cores = sorted(guard.included(role)); cores = cores[:2] if gate else cores
            expected = [(c, i) for c in cores for i in (range(1) if gate else range(10))]
            historical = {}
            if gate:
                assert R.sha(source['cal_rows']) == source['cal_rows_sha256']
                historical = {(r['core'],r['perturbation']):r for r in R.rows(source['cal_rows'])}
            arm_rows = []
            with R.CP.evaluation_state(t):
                session = R.make_session(u, t, guard, H, role, cores, 'cuda')
                for index,(core,i) in enumerate(expected):
                    energy,_ = session.energy(core,i); inner = R.PE.inner(energy)
                    inner.j_critic = types.MethodType(T.cosine_energy, inner)
                    truth = session.perturbations[core][i]
                    if gate: technical.append(dict(arm=trained_arm, core=core, checks=compare_gate(energy,inner,truth)))
                    values = {}; shift = (seed + index) % len(POLICIES)
                    order = POLICIES[shift:] + POLICIES[:shift]
                    for coarse,local in order:
                        wrapped = StageEnergy(energy, inner, coarse, local)
                        torch.cuda.synchronize(); tick = time.perf_counter()
                        row = R.plain(A.registration_record(wrapped, 'cuda', truth))
                        torch.cuda.synchronize(); row['synchronized_wall_seconds'] = time.perf_counter()-tick
                        assert all(math.isfinite(row[k]) for k in ('err_full','F_hat','F_true'))
                        assert wrapped.grid_calls == len(A.Q.grid_points()[0])
                        assert wrapped.local_calls == A.Q.P.N_STARTS * (A.Q.P.N_ADAM+1) and wrapped.value_calls == 1
                        row.update(grid_calls=wrapped.grid_calls, local_calls=wrapped.local_calls,
                                   grid_values_sha256=hashlib.sha256(''.join(wrapped.grid_digests).encode()).hexdigest())
                        values[coarse+'__'+local] = row
                    assert len({r['n_grid'] for r in values.values()}) == 1
                    if gate:
                        old = historical[core,i]['E']['COS']; new = values['COS__COS']
                        assert old['n_grid'] == new['n_grid'] and old['theta'] == new['theta'] and old['t'] == new['t']
                        inner.geometry_readout = 'COS'
                        native_true = energy.value(truth[0], truth[1:])
                        assert abs(native_true-new['F_true']) <= 1e-5 + 1e-4*abs(native_true)
                        source_done = json.loads(Path(source['completion']).read_text())
                        source_driver = source_done['environment']['precision']['nvidia_smi_name_driver']
                        current_driver = run_environment['precision']['nvidia_smi_name_driver']
                        matched_hardware = source_driver == current_driver
                        if matched_hardware:
                            assert abs(old['F_true']-new['F_true']) <= 1e-5 + 1e-4*abs(old['F_true'])
                        technical[-1]['historical_check'] = dict(matched_gpu_and_driver=matched_hardware,
                            source_gpu_driver=source_driver, current_gpu_driver=current_driver,
                            historical_F_true=old['F_true'], current_F_true=new['F_true'],
                            native_F_true=native_true, difference=new['F_true']-old['F_true'],
                            treatment='gated unchanged tolerance' if matched_hardware else 'descriptive cross-hardware difference')
                    record = dict(arm=trained_arm, core=core, scene=guard.unit_of(core) if H else core,
                                  perturbation=i, role=role, policies=values, order=[a+'__'+b for a,b in order])
                    arm_rows.append(record); R.append(out/'ROWS.jsonl', record)
                    R.write(out/'PROGRESS.json', dict(arm=trained_arm, rows=len(arm_rows), expected=len(expected), utc=datetime.now(timezone.utc).isoformat()))
                    if (index+1) % 20 == 0: print(name, trained_arm, index+1, '/', len(expected), flush=True)
                    del energy
            assert A.state_hash(t.eU,t.eV) == before
            assert {x['role'] for x in guard.reads} <= {'FIT','CAL','DEV'}
            for policy in [a+'__'+b for a,b in POLICIES]:
                scenes = defaultdict(list)
                for r in arm_rows: scenes[r['scene']].append(r['policies'][policy]['err_full'])
                summaries.setdefault(trained_arm,{})[policy] = dict(
                    success9=statistics.mean(statistics.mean(x<9 for x in v) for v in scenes.values()),
                    success42=statistics.mean(statistics.mean(x<42 for x in v) for v in scenes.values()),
                    median_err=statistics.median(x for v in scenes.values() for x in v), n=len(arm_rows), scenes=len(scenes),
                    failures42=sum(x>=42 for v in scenes.values() for x in v),
                    seconds=sum(r['policies'][policy]['synchronized_wall_seconds'] for r in arm_rows))
            input_records.append(dict(arm=trained_arm, reads=guard.reads))
            model_records.append(dict(arm=trained_arm, source=source, model_hash=before, unchanged=True))
            del t; torch.cuda.empty_cache()
        R.write(out/'INPUT_TRACE.json', input_records)
        result = dict(status='COMPLETE', freeze_sha256=freeze, gate=gate, ds=ds, arm=arm, seed=seed, job=job,
            seconds=time.time()-started, summaries=summaries, technical_checks=technical, models=model_records,
            rows_sha256=R.sha(out/'ROWS.jsonl'), input_trace_sha256=R.sha(out/'INPUT_TRACE.json'), numerical_roles=[role],
            environment=R.environment(), gpu=torch.cuda.get_device_name(), weights_unchanged=True,
            note='Same host search budget, stage-specific energy readouts; all DEV reused exploratory. No model learning or readout selection.')
        R.write(out/'COMPLETE.json', result)
        R.write(out/f'ATTEMPT_{job}.json', dict(status='COMPLETE', job=job, seconds=time.time()-started))
        print('STAGE_COMPLETE',name,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-check',action='store_true');p.add_argument('--gate',action='store_true')
    p.add_argument('--ds',choices=('bf','epfl'));p.add_argument('--arm',choices=ARMS,default='vcs_exact');p.add_argument('--seed',type=int,default=1001)
    a=p.parse_args()
    if a.cpu_check: cpu_check()
    else: run(a.ds,a.arm,a.seed,a.gate)
