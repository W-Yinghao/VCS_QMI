"""Fixed CAL inputs, common COS starts, three complete host Adam trajectories."""
import argparse
import copy
import fcntl
import hashlib
import math
import os
import time
import traceback
import types
from common import BASE, HERE, read, write, sha, verify, geometry

L = geometry()
T, R, A, torch = L.T, L.R, L.A, L.torch
TERMS = ('P1', 'negative_P2_half', 'negative_Q1', 'negative_Q2_half', 'POS_QUAD', 'J')
OBJECTIVES = ('P1', 'POS_QUAD', 'J')
RECORD = (0, 1, 5, 10, 25, 50, 100)


def terms(u, v):
    u, v = T.normalized(u, v)
    p = (u*v).sum(-1)
    n = u.shape[-2]
    d = dict(P1=p.mean(-1), negative_P2_half=-.5*p.square().mean(-1),
             negative_Q1=-(u.mean(-2)*v.mean(-2)).sum(-1),
             negative_Q2_half=-.5*((u.transpose(-1, -2)@u/n)*(v.transpose(-1, -2)@v/n)).sum((-1, -2)))
    d['POS_QUAD'] = d['P1'] + d['negative_P2_half']
    d['J'] = sum(d[k] for k in TERMS[:4])
    return d


def install(energy):
    inner = R.PE.inner(energy)
    def evaluate(self, v):
        self.trajectory_terms = terms(self.u[None], v)
        return self.trajectory_terms[self.trajectory_objective]
    inner.j_critic = types.MethodType(evaluate, inner)
    inner.trajectory_objective = 'P1'
    return inner


def project(p):
    with torch.no_grad():
        p[0].clamp_(-A.Q.UMAX, A.Q.UMAX)
        n = p[1:].norm()
        if n > A.Q.TMAX:
            p[1:].mul_(A.Q.TMAX/n)


def pose_error(p, truth):
    q = p.detach().cpu().tolist()
    return A.Q.corner_error_full(math.degrees(q[0]/A.Q.RHO), q[1:], truth[0], truth[1:])


def context(ds, seed, device):
    # Read metadata directly. Do not use the generic loader's CAL+DEV cache/hash pass.
    u = read(BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_MLPp1_s{seed}.json')
    assert u['ds'] == ds and u['seed'] == seed and u['family'] == 'MLP_VCS_perm1'
    assert sha(u['data_lock']['path']) == u['data_lock']['sha256']
    lock = read(u['data_lock']['path'])
    if ds == 'bf':
        guard, H = A.DataGuard(), None
        files = {r['core']: r for r in lock['files'] if r['role'] == 'CAL'}
    else:
        from v10.epfl import guard as GD, host as H
        g = u['guard']
        guard = GD.EPFLDataGuard(record=g['record'], expect_record_sha256=g['record_sha256'], data_lock=lock)
        assert guard.stage == 'ACTIVATED' and guard.group_sha256 == g['group_record_sha256']
        files = {r['window']: dict(path=r['cache_path'], sha256=r['sha256']) for r in lock['windows'] if r['role'] == 'CAL'}
    expected = read(HERE/'INPUT_LOCK.json')[ds]
    cores = sorted(guard.included('CAL'))
    assert cores == expected['cores'] == sorted(files)
    for core in cores:
        path = guard.check(core, {'CAL'}, 'trajectory CAL cache hash verification')
        assert str(path) == files[core]['path'] and sha(path) == files[core]['sha256']
        assert files[core]['sha256'] == expected['cache_hashes'][core]
        scene = guard.unit_of(core) if H else core
        assert scene == expected['scenes'][core]
    fit = A.fit_cores(guard, 1., seed) if H is None else H.fit_windows(guard)  # IDs only
    assert {r['role'] for r in guard.reads} == {'CAL'}
    return u, guard, H, fit, cores


def restored(ds, arm, seed, fit, device):
    src = read(HERE/'SOURCE_LOCK.json')[f'{ds}_{arm}_{seed}']
    assert sha(src['checkpoint']) == src['sha256'] and sha(src['completion']) == src['completion_sha256']
    rec = read(src['completion'])
    assert rec['status'] == 'COMPLETE' and rec['identity']['ds'] == ds
    assert rec['identity']['arm'] == arm and rec['identity']['seed'] == seed
    trainer = T.GeometryTrainer(L.config(seed), {}, fit, device, arm)
    R.CP.load(trainer, src['checkpoint'], rec['identity'])
    assert trainer.n_steps == 30000
    return trainer, src


def starts(energy, inner, ds, arm, seed, core, perturbation):
    cached = read(HERE/'CACHED_STARTS.json').get(f'{ds}_{arm}_{seed}', {}).get(f'{core}|{perturbation}')
    inner.trajectory_objective = 'P1'
    if cached is not None:
        return copy.deepcopy(cached), 'frozen_prior_CAL_P1_top3'
    ths, ts = A.Q.grid_points()
    tt = torch.tensor(ts, dtype=torch.float32, device=inner.x.device)
    with torch.no_grad():
        values = R.np.concatenate([energy.grid_values(float(th), tt).float().cpu().numpy() for th in ths])
    assert R.np.isfinite(values).all()
    out = []
    for rank, i in enumerate(R.np.argsort(values)[::-1][:A.Q.P.N_STARTS]):
        out.append(dict(coarse_rank=rank, theta=float(ths[i//len(ts)]),
                        translation=ts[i%len(ts)].tolist(), coarse_value=float(values[i])))
    return out, 'one_new_CAL_COS_coarse_pass_no_exact_reranking'


def run_path(energy, inner, initial, objective, truth, ds):
    inner.trajectory_objective = objective
    p = initial.detach().clone().requires_grad_()
    opt = torch.optim.Adam([p], lr=A.Q.P.ADAM_LR)
    rows = []
    for step in range(A.Q.P.N_ADAM+1):
        opt.zero_grad()
        value = energy(p)
        assert torch.isfinite(value)
        if step in RECORD:
            z = inner.trajectory_terms
            grads = {k: torch.autograd.grad(z[k].sum(), p, retain_graph=True)[0] for k in TERMS}
            assert all(torch.isfinite(g).all() for g in grads.values())
            assert torch.allclose(grads['J'], sum(grads[k] for k in TERMS[:4]), atol=2e-5, rtol=1e-3)
            assert torch.allclose(value, z[objective].sum(), atol=1e-7, rtol=1e-5)
            state = opt.state.get(p, {})
            err = pose_error(p, truth)
            rows.append(dict(step=step, pose=p.detach().cpu().tolist(), error_full=err,
                             error_native=err/4 if ds == 'epfl' else err,
                             values={k: float(z[k].detach().sum()) for k in TERMS},
                             gradients={k: g.detach().cpu().tolist() for k, g in grads.items()},
                             adam_step=int(state['step']) if state else 0,
                             adam_exp_avg=state['exp_avg'].detach().cpu().tolist() if state else [0.]*3,
                             adam_exp_avg_sq=state['exp_avg_sq'].detach().cpu().tolist() if state else [0.]*3))
            if step == 50:
                # Check a copied real optimizer state round-trip, without an extra update.
                clone = p.detach().clone().requires_grad_()
                opt2 = torch.optim.Adam([clone], lr=A.Q.P.ADAM_LR)
                opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
                assert L.tree_equal(opt.state_dict(), opt2.state_dict())
        if step == A.Q.P.N_ADAM:
            break
        (-value).backward()
        assert torch.isfinite(p.grad).all()
        opt.step()
        project(p)
        assert abs(float(p[0])) <= A.Q.UMAX+1e-4 and float(p[1:].norm()) <= A.Q.TMAX+1e-4
    assert [r['step'] for r in rows] == list(RECORD)
    return dict(objective=objective, initial_pose=initial.detach().cpu().tolist(), trajectory=rows,
                final_pose=rows[-1]['pose'], final_energy=rows[-1]['values'][objective],
                final_error_full=rows[-1]['error_full'], final_error_native=rows[-1]['error_native'])


def cpu_check():
    assert os.environ.get('SLURM_JOB_ID')
    f = verify(); torch.set_num_threads(2); torch.manual_seed(20261025)
    assert (A.Q.P.N_ADAM, A.Q.P.ADAM_LR, A.Q.P.N_STARTS, A.Q.P.N_POS) == (100, .5, 3, 4096)
    u = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(u, requires_grad=True)
    z = terms(u, v); direct = T.exact_j(*T.normalized(u, v))
    assert torch.allclose(z['J'], direct, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(z['J'].sum(), (u, v), retain_graph=True)
    gb = torch.autograd.grad(direct.sum(), (u, v), retain_graph=True)
    assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-12) for a, b in zip(ga, gb))
    assert torch.autograd.gradcheck(lambda x, y: torch.stack(list(terms(x, y).values())), (u, v))
    # Independently execute the literal host 100-step loop vs the instrumented one.
    class Toy:
        trajectory_objective = 'P1'
        def __call__(self, p):
            a = -((p-torch.tensor([4., -3., 8.], dtype=p.dtype))**2).sum()/100
            b = -.02*(p**2).sum()/100
            c = .001*p.sum(); e = -.001*(p**2).sum()/100
            self.trajectory_terms = dict(zip(TERMS, (a, b, c, e, a+b, a+b+c+e)))
            return self.trajectory_terms[self.trajectory_objective]
    e = Toy(); init = torch.tensor([A.Q.UMAX, 50., 10.], dtype=torch.float64)
    for objective in OBJECTIVES:
        got = run_path(e, e, init, objective, [0., 0., 0.], 'bf')
        p = init.clone().requires_grad_(); opt = torch.optim.Adam([p], lr=.5)
        for _ in range(100):
            opt.zero_grad(); (-e(p)).backward(); opt.step()
            with torch.no_grad():
                p[0].clamp_(-A.Q.UMAX, A.Q.UMAX); nrm = p[1:].norm()
                if nrm > A.Q.TMAX: p[1:].mul_(A.Q.TMAX/nrm)
        assert p.detach().tolist() == got['final_pose']
    rows = []
    for ds in ('bf', 'epfl'):
        for seed in (1001, 1002, 1003):
            _, guard, _, fit, cores = context(ds, seed, 'cpu')
            for arm in ('vcs_exact', 'bce', 'nce'):
                t, src = restored(ds, arm, seed, fit, 'cpu')
                rows.append(dict(ds=ds, seed=seed, arm=arm, step=t.n_steps, source_sha256=src['sha256'], cal_windows=len(cores)))
                del t
            assert {r['role'] for r in guard.reads} == {'CAL'}
    write(HERE/'CPU_CHECK.json', dict(status='PASS', freeze_sha256=f, sources=rows,
        checks=['float64 term values/gradients and finite differences', 'instrumented100 Adam steps bitwise equal literal host CPU loop for all3 objectives', 'real optimizer state50 roundtrip', 'all18 fixed30k source restores', 'strict CAL cache identity and unchanged split/scene metadata']))
    print('TRAJECTORY_CPU PASS', flush=True)


def run(ds, arm, seed, gate=False):
    assert os.environ.get('SLURM_JOB_ID')
    f = verify(); c = read(HERE/'CPU_CHECK.json')
    assert c['status'] == 'PASS' and c['freeze_sha256'] == f
    if not gate:
        for data in ('bf', 'epfl'):
            g = read(HERE/'runs'/f'gate_{data}'/'COMPLETE.json')
            assert g['status'] == 'COMPLETE' and g['freeze_sha256'] == f and g['gate']
    torch.set_num_threads(4); assert torch.cuda.is_available()
    name = f'gate_{ds}' if gate else f'{ds}_{arm}_{seed}'
    out = HERE/'runs'/name; out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (out/'COMPLETE.json').exists() and not list(out.glob('ATTEMPT_*.json')), 'No duplicate or implicit retry'
        start = time.time(); job = os.environ['SLURM_JOB_ID']
        write(out/f'ATTEMPT_{job}.json', dict(status='STARTED', unix=start, job=job, freeze_sha256=f))
        try:
            u, guard, H, fit, cores = context(ds, seed, 'cuda')
            t, src = restored(ds, arm, seed, fit, 'cuda')
            active = cores[:1] if gate else cores
            perturbations = (0,) if gate else (0, 1)
            problems = []
            with R.CP.evaluation_state(t):
                session = R.make_session(u, t, guard, H, 'CAL', active, 'cuda')
                for core in active:
                    for i in perturbations:
                        energy, _ = session.energy(core, i); inner = install(energy)
                        assert inner.u.shape[0] == A.Q.P.N_POS == 4096
                        truth = session.perturbations[core][i]
                        candidates, origin = starts(energy, inner, ds, arm, seed, core, i)
                        assert len(candidates) == 3
                        start_hash = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode()).hexdigest()
                        result = dict(core=core, scene=guard.unit_of(core) if H else core, perturbation=i,
                                      truth=list(truth), candidates=candidates, start_origin=origin, start_sha256=start_hash,
                                      n_pos=int(inner.u.shape[0]),
                                      position_sha256=hashlib.sha256(inner.x.detach().cpu().numpy().tobytes()).hexdigest(),
                                      objectives={})
                        for objective in OBJECTIVES:
                            paths = []
                            for cand in candidates:
                                pose = torch.tensor([math.radians(cand['theta'])*A.Q.RHO, *cand['translation']], dtype=torch.float32, device='cuda')
                                paths.append(run_path(energy, inner, pose, objective, truth, ds))
                            selected = max(range(3), key=lambda k: paths[k]['final_energy'])
                            result['objectives'][objective] = dict(paths=paths, selected_rank=selected,
                                selected_error_native=paths[selected]['final_error_native'],
                                mean_three_start_error_native=sum(x['final_error_native'] for x in paths)/3)
                        for k in range(3):
                            assert all(result['objectives'][a]['paths'][k]['initial_pose'] == result['objectives']['P1']['paths'][k]['initial_pose'] for a in OBJECTIVES)
                        R.append(out/'TRAJECTORIES.jsonl', result); problems.append((core, i))
                        print(name, core, i, 'COMPLETE', flush=True)
            assert len(problems) == len(active)*len(perturbations)
            assert {r['role'] for r in guard.reads} == {'CAL'}
            assert sha(src['checkpoint']) == src['sha256']
            write(out/'INPUT_TRACE.json', guard.reads)
            write(out/'COMPLETE.json', dict(status='COMPLETE', freeze_sha256=f, ds=ds, arm=arm, seed=seed, gate=gate,
                job=job, seconds=time.time()-start, source=src, source_unchanged=True,
                problems=problems, n_problems=len(problems), objectives=list(OBJECTIVES), starts=3, adam_steps=100,
                adam_lr=.5, record_steps=list(RECORD), roles_read=['CAL'], environment=R.environment(),
                trajectory_sha256=sha(out/'TRAJECTORIES.jsonl'), input_trace_sha256=sha(out/'INPUT_TRACE.json'),
                interpretation='Matched fixed-CAL optimization paths, not new training or independent confirmation. Truth only logged; final rank uses each objective energy.'))
        except Exception as exc:
            write(out/f'FAILURE_{job}.json', dict(status='FAILED', seconds=time.time()-start, error=repr(exc), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'gate', 'run'])
    p.add_argument('--ds', choices=['bf', 'epfl']); p.add_argument('--arm', choices=['vcs_exact', 'bce', 'nce'], default='vcs_exact')
    p.add_argument('--seed', type=int, default=1001); a = p.parse_args()
    if a.mode == 'check': cpu_check()
    else: run(a.ds, a.arm, a.seed, a.mode == 'gate')
