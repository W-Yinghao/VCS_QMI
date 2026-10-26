"""Shared-cosine geometry training with a deployed-energy pose-ranking term (R22 registration, external review 2026-10-25).

L_pose = log(1 + sum_d exp((S(theta*+d) - S(theta*) + m_d) / tau)), S = deployed COS energy: BF/EPFL host warp of the whole perturbed
moving image into BOX1 -> eV encode -> sample at 4096 disk positions, reference eU-encoded once, S = mean cos. theta* = a known
FIT perturbation from the host rule Q.perturbations_p2 (same rule as CAL/DEV, here on FIT cores/windows only). Gradients flow
through eU and eV. Arms: pose_only; pose + lambda*(-J_exact); pose + lambda*BCE (train.py objectives). Same init, same host
sampler stream and the long-v2 LR schedule (3e-4 <=15000, 3e-5 <=24000, 3e-6 <=30000) from scratch, so long-v2 vcs_exact/bce/nce
30000-step runs are the matched J-only/BCE-only/NCE-only references. Pose randomness uses a separate RNG.
"""
import argparse, fcntl, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('long_v2', BASE / 'handoff/r22_geometry_long_v2_20261024/run.py')
L = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(L)
T, R, A, torch = L.T, L.R, L.A, L.torch
F, Q, P = T.F, A.Q, A.P
ARMS = ('pose_only', 'pose_vcs', 'pose_bce')
BASE_OBJECTIVE = {'pose_only': None, 'pose_vcs': 'vcs_exact', 'pose_bce': 'bce'}
REFERENCE = {'pose_only': 'vcs_exact', 'pose_vcs': 'vcs_exact', 'pose_bce': 'bce'}   # long-v2 run whose stream/init we must match
# Wrong-pose set relative to the truth (dtheta deg, dty, dtx at 512 px; err_full = 4 x 512-px corner error, corner radius 147.25 px).
# near ring ~12 err_full (3 px; 1.2 deg -> 3.08 px): 1.33x the strict 9 threshold, the observed DEV near-miss band 9-42.
# far ring ~32-33 err_full (8 px; 3.2 deg -> 8.2 px): inside the loose 42 threshold. Revised after a FIT-only calibration showed
# 4 px / 12 px candidates already lose to the truth in ~93% / 94-100% of FIT problems (CALIBRATION_V0.json); no DEV/CAL outcome used.
NEAR = [(0., 3., 0.), (0., -3., 0.), (0., 0., 3.), (0., 0., -3.), (1.2, 0., 0.), (-1.2, 0., 0.)]
FAR = [(0., 8 / math.sqrt(2), 8 / math.sqrt(2)), (0., -8 / math.sqrt(2), -8 / math.sqrt(2)), (3.2, 0., 0.), (-3.2, 0., 0.)]
D = NEAR + FAR
CONST = json.loads((HERE / 'CONSTANTS.json').read_text()) if (HERE / 'CONSTANTS.json').exists() else None   # tau, m_near, m_far, lam


def verify():
    for p, h in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        assert R.sha(BASE / p) == h, p
    return R.sha(HERE / 'FREEZE.json')


class PoseProblems:
    """Known FIT perturbations through the deployed host inputs (perturbed moving image, robust-norm learned path, BF RGB reference)."""
    def __init__(self, ds, guard, H, fit, device, seed):
        self.fit, self.device = sorted(fit), device
        purpose = 'R22 pose-ranking FIT known perturbations'
        if H is None:
            pert = {}
            for c in self.fit: guard.log_rule(c, {'FIT'}, purpose); pert[c] = Q.perturbations_p2(c)
            self.inputs = A.GuardedInputs(guard, {'FIT'}, purpose, pert, device)
        else:
            pert = H.perturbations_for(guard, self.fit, 'FIT', purpose)
            self.inputs = H.EPFLGuardedInputs(guard, {'FIT'}, purpose, pert, device)
        self.pert = pert; self.ref, self.mov = {}, {}
        self.rng = np.random.default_rng(Q.seed_of(f'R22-POSE|{ds}|{seed}'))   # separate stream: host sampler untouched

    def get(self, c, i):
        if c not in self.ref:
            self.ref[c] = self.inputs.reference(c)[1].cpu(); self.inputs.ref.clear()
        if (c, i) not in self.mov:
            self.mov[(c, i)] = self.inputs.moving(c, i)[1].cpu(); self.inputs.mov.clear()
        return self.ref[c].to(self.device), self.mov[(c, i)].to(self.device), self.pert[c][i]

    def draw(self):
        c = self.fit[int(self.rng.integers(len(self.fit)))]; i = int(self.rng.integers(Q.P.N_PERT))
        r = Q.R_OMEGA * np.sqrt(self.rng.uniform(0., 1., P.N_POS)); ang = self.rng.uniform(0., 2 * math.pi, P.N_POS)
        x = np.stack([Q.CEN + r * np.sin(ang), Q.CEN + r * np.cos(ang)], -1)
        return c, i, x


def candidates(truth, device):
    th, ty, tx = truth
    rows = [(th, ty, tx)] + [(th + a, ty + b, tx + c) for a, b, c in D]
    th = torch.tensor([math.radians(r[0]) for r in rows], dtype=torch.float32, device=device)
    t = torch.tensor([[r[1], r[2]] for r in rows], dtype=torch.float32, device=device)
    return th, t


def pose_energy(eU, eV, ref, mov, x, th, t):
    """Host critic path (Q.RigidEnergy 'critic' + T.cosine_energy COS) for K candidates, differentiable in eU and eV -> S [K]."""
    refmap = eU(ref[None])[0]; u = P.sample_map(torch, F, refmap, x)
    box = Q.grid_coords(torch, *Q.BOX1, mov.device); n = Q.BOX1[1] - Q.BOX1[0] + 1
    W = Q.sample_pts(torch, F, mov, Q.apply_T(torch, th, t, box))[..., 0].reshape(len(th), 1, n, n)
    maps = eV(W)
    dv = torch.stack([P.sample_map(torch, F, maps[k], x - Q.BOX1[0]) for k in range(len(th))])
    un, vn = T.normalized(u[None], dv)
    return (un * vn).sum(-1).mean(-1)


def pose_loss(S, const):
    m = torch.tensor([const['m_near']] * len(NEAR) + [const['m_far']] * len(FAR), dtype=S.dtype, device=S.device)
    z = torch.cat([S.new_zeros(1), (S[1:] - S[0] + m) / const['tau']])
    # tau * logsumexp: dL/dS are softmax weights in [0, 1] (energy units), the same scale as the base objectives' dependence on cos
    return const['tau'] * torch.logsumexp(z, 0)


def train_step(t, arm, problems, const):
    lr = L.lr_at(t.n_steps + 1)
    for g in t.opt.param_groups: g['lr'] = lr
    hb, bu, bv = t.draw(); digest = L.batch_sha(hb); row = dict(step=hb.step, lr=lr, batch_sha256=digest)
    start = time.perf_counter()
    c, i, xnp = problems.draw(); ref, mov, truth = problems.get(c, i)
    x = torch.tensor(xnp, dtype=torch.float32, device=t.device); th, tt = candidates(truth, t.device)
    S = pose_energy(t.eU, t.eV, ref, mov, x, th, tt); lp = pose_loss(S, const); total = lp
    if BASE_OBJECTIVE[arm] is not None:
        du, dv = A.batch_descriptors(t.eU, t.eV, bu, bv, hb.yx)
        base, j = T.objective(du, dv, BASE_OBJECTIVE[arm], torch.as_tensor(hb.perm, device=t.device))
        total = lp + const['lam'] * base; row['base_loss'] = float(base.detach())
        if j is not None: row['J'] = float(j.detach())
    t.opt.zero_grad(set_to_none=True); total.backward()
    row['grad_norm_preclip'] = float(torch.nn.utils.clip_grad_norm_(t.clip_params, t.cfg.clip_norm)); t.opt.step(); t.n_steps += 1
    if t.device.type == 'cuda': torch.cuda.synchronize()
    with torch.no_grad():
        gap = (S[1:] - S[0]).cpu().numpy()
    row.update(loss=float(total.detach()), pose_loss=float(lp.detach()), S_true=float(S[0].detach()), problem=[c, i],
               near_gap_mean=float(gap[:len(NEAR)].mean()), far_gap_mean=float(gap[len(NEAR):].mean()),
               truth_beats_all=bool((gap < 0).all()), synchronized_step_seconds=time.perf_counter() - start)
    assert row['step'] == t.n_steps and math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])
    return row


def reference_rows(ds, arm, seed):
    return [json.loads(l) for _, l in zip(range(5), open(BASE / f'handoff/r22_geometry_long_v2_20261024/runs/{ds}_{REFERENCE[arm]}_{seed}/TRAIN.jsonl'))]


def cpu_check():
    freeze = verify(); torch.set_num_threads(4); checks = []
    assert CONST is not None and set(CONST) >= {'tau', 'm_near', 'm_far', 'lam'}
    # candidate support: every sampled source location of V stays inside the image for every candidate of every FIT perturbation
    worst = Q.R_OMEGA + Q.RF_BOX * math.sqrt(2) + max(math.hypot(b, c) for _, b, c in D)
    assert worst < Q.CEN, worst
    for ds in ('bf', 'epfl'):
        u, guard, H, fit, imgs = L.context(ds, 1001, 'cpu', images=True)
        # sampler stream: fast-forward 3000 draws, steps 3001-3005 equal the long-v2 continuation stream for every arm reference
        for arm in ARMS:
            t = T.GeometryTrainer(L.config(1001), {}, fit, 'cpu', 'vcs_exact')
            for _ in range(3000): t.sampler.draw()
            mine = [L.batch_sha(t.sampler.draw()) for _ in range(5)]
            assert mine == [r['batch_sha256'] for r in reference_rows(ds, arm, 1001)], (ds, arm)
        orig = json.loads((BASE / f'handoff/r22_geometry_train_20261023/runs/{ds}_vcs_exact_1001/COMPLETE.json').read_text())
        t = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', 'vcs_exact')
        assert A.state_hash(t.eU, t.eV) == orig['initial_encoder_sha256']
        # deployed-energy equivalence: pose_energy == host RigidEnergy('critic') with the COS readout, same positions
        problems = PoseProblems(ds, guard, H, fit, 'cpu', 1001); c, i, _ = problems.draw(); ref, mov, truth = problems.get(c, i)
        with torch.no_grad():
            refmap = t.eU(ref[None])[0]
            energy = Q.RigidEnergy(torch, F, 'critic', refmap, mov, np.random.default_rng(7), eV=t.eV, critic=t.critic)
            energy.j_critic = T.types.MethodType(T.cosine_energy, energy); energy.geometry_readout = 'COS'
            th, tt = candidates(truth, 'cpu')
            host = energy.batch(th, tt); mine = pose_energy(t.eU, t.eV, ref, mov, energy.x, th, tt)
        assert torch.allclose(host, mine, atol=1e-6, rtol=1e-5), float((host - mine).abs().max())
        # base step with the pose term removed equals the original GeometryTrainer.step bitwise (same stream, optimizer, clip)
        for arm in ('pose_vcs', 'pose_bce'):
            a = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', BASE_OBJECTIVE[arm]); b = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', BASE_OBJECTIVE[arm])
            for g in b.opt.param_groups: g['lr'] = 3e-4
            ra = a.step(); hb, bu, bv = b.draw(); du, dv = A.batch_descriptors(b.eU, b.eV, bu, bv, hb.yx)
            base, _ = T.objective(du, dv, BASE_OBJECTIVE[arm], torch.as_tensor(hb.perm)); b.opt.zero_grad(set_to_none=True); base.backward()
            torch.nn.utils.clip_grad_norm_(b.clip_params, b.cfg.clip_norm); b.opt.step()
            assert ra['loss'] == float(base.detach()) and A.state_hash(a.eU, a.eV) == A.state_hash(b.eU, b.eV), (ds, arm)
        # one full pose step per arm is finite; pose RNG does not touch the host sampler
        for arm in ARMS:
            t = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', 'vcs_exact'); pr = PoseProblems(ds, guard, H, fit, 'cpu', 1001)
            row = train_step(t, arm, pr, CONST); assert row['batch_sha256'] == L.batch_sha(T.GeometryTrainer(L.config(1001), {}, fit, 'cpu', 'vcs_exact').sampler.draw())
            checks.append(dict(ds=ds, arm=arm, first_step={k: row[k] for k in ('loss', 'pose_loss', 'S_true', 'near_gap_mean', 'far_gap_mean', 'grad_norm_preclip')}))
        assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
    R.write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, worst_support_radius=worst, checks=checks, verified=[
        'host sampler steps 3001-3005 equal long-v2 continuation rows (all arm references, both datasets)',
        'initial encoder hash equals the original 3000-step training init', 'pose_energy equals host RigidEnergy critic+COS (atol1e-6)',
        'pose-free base step bitwise equals GeometryTrainer.step', 'all candidate samples inside the moving image',
        'one finite pose step per arm; pose RNG separate from the host sampler']))
    print('POSE_CPU PASS', flush=True)


def calibrate():
    """Energy-gap scale on FIT only, from existing checkpoints (fixes tau/margins before any pose training)."""
    torch.set_num_threads(4); out = {}
    for ds in ('bf', 'epfl'):
        u, guard, H, fit, imgs = L.context(ds, 1001, 'cuda', images=False)
        for label, kind in (('init', None), ('step3000', 'orig'), ('step30000', 'long')):
            t = T.GeometryTrainer(L.config(1001), {}, fit, 'cuda', 'vcs_exact')
            if kind == 'orig':
                d = BASE / f'handoff/r22_geometry_train_20261023/runs/{ds}_vcs_exact_1001'; R.CP.load(t, d / 'checkpoints/step_3000.pt', json.loads((d / 'COMPLETE.json').read_text())['identity'])
            if kind == 'long':
                d = BASE / f'handoff/r22_geometry_long_v2_20261024/runs/{ds}_vcs_exact_1001'; R.CP.load(t, d / 'checkpoints/step_30000.pt', json.loads((d / 'COMPLETE.json').read_text())['identity'])
            pr = PoseProblems(ds, guard, H, fit, 'cuda', 99); gaps = []
            with torch.no_grad():
                for _ in range(48):
                    c, i, xnp = pr.draw(); ref, mov, truth = pr.get(c, i); th, tt = candidates(truth, 'cuda')
                    S = pose_energy(t.eU, t.eV, ref, mov, torch.tensor(xnp, dtype=torch.float32, device='cuda'), th, tt)
                    gaps.append((S[1:] - S[0]).cpu().numpy())
            g = np.stack(gaps); out[f'{ds}_{label}'] = dict(near_abs_median=float(np.median(np.abs(g[:, :len(NEAR)]))), far_abs_median=float(np.median(np.abs(g[:, len(NEAR):]))),
                near_mean=float(g[:, :len(NEAR)].mean()), far_mean=float(g[:, len(NEAR):].mean()), near_frac_wrong_higher=float((g[:, :len(NEAR)] > 0).mean()),
                far_frac_wrong_higher=float((g[:, len(NEAR):] > 0).mean()))
            print(ds, label, out[f'{ds}_{label}'], flush=True)
        assert all(r['role'] == 'FIT' for r in guard.reads if 'pose-ranking' in r.get('purpose', ''))
    R.write(HERE / 'CALIBRATION.json', dict(status='COMPLETE', job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), problems_per_cell=48, gaps=out))


def run(ds, arm, seed, smoke_steps=0):
    freeze = verify(); cpu = json.loads((HERE / 'CPU_CHECK.json').read_text()); assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'); torch.set_num_threads(4); started = time.time()
    name = f'{ds}_{arm}_{seed}' + ('_smoke' if smoke_steps else ''); out = HERE / ('smoke' if smoke_steps else 'runs') / name
    out.mkdir(parents=True, exist_ok=True); ident = dict(ds=ds, arm=arm, seed=seed, freeze_sha256=freeze, smoke=bool(smoke_steps))
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out / 'COMPLETE.json').exists() and not list(out.glob('ATTEMPT_*.json')), 'No silent rerun or retry'
        job = os.environ['SLURM_JOB_ID']; R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', identity=ident, start=started, environment=R.environment()))
        u, guard, H, fit, imgs = L.context(ds, seed, 'cuda', images=True)
        t = T.GeometryTrainer(L.config(seed), imgs, fit, 'cuda', BASE_OBJECTIVE[arm] or 'vcs_exact'); initial = A.state_hash(t.eU, t.eV)
        orig = json.loads((BASE / f'handoff/r22_geometry_train_20261023/runs/{ds}_{REFERENCE[arm]}_{seed}/COMPLETE.json').read_text())
        assert initial == orig['initial_encoder_sha256'], 'Initial encoders differ from the matched reference'
        problems = PoseProblems(ds, guard, H, fit, 'cuda', seed); ref_rows = reference_rows(ds, arm, seed); cal = {}
        stops = (smoke_steps,) if smoke_steps else (10000, 20000, 30000)
        for stop in stops:
            with (out / 'TRAIN.jsonl').open('a') as trace:
                while t.n_steps < stop:
                    row = train_step(t, arm, problems, CONST)
                    if 3001 <= row['step'] <= 3005: assert row['batch_sha256'] == ref_rows[row['step'] - 3001]['batch_sha256'], 'Host stream differs'
                    trace.write(json.dumps(R.plain(row), allow_nan=False) + '\n')
                    if t.n_steps % 250 == 0: trace.flush(); print(name, 'step', t.n_steps, row['loss'], flush=True)
            if not smoke_steps:
                R.CP.save(t, out / 'checkpoints' / f'step_{stop}.pt', ident)
                d = out / f'cal_{stop}'; d.mkdir(exist_ok=True); cal[stop] = T.evaluate(t, u, guard, H, 'CAL', d, ('COS',))
        rows = [json.loads(l) for l in open(out / 'TRAIN.jsonl')]
        assert [r['step'] for r in rows] == list(range(1, t.n_steps + 1)) and all(r['lr'] == L.lr_at(r['step']) for r in rows)
        result = dict(status='COMPLETE', identity=ident, freeze_sha256=freeze, ds=ds, arm=arm, seed=seed, job=job, constants=CONST,
                      initial_encoder_sha256=initial, steps=t.n_steps, gpu=torch.cuda.get_device_name(), environment=R.environment(),
                      synchronized_training_seconds=sum(r['synchronized_step_seconds'] for r in rows), train_sha256=R.sha(out / 'TRAIN.jsonl'),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if smoke_steps:
            last = rows[-50:]; result.update(mean_step_ms=1000 * sum(r['synchronized_step_seconds'] for r in rows[10:]) / max(1, len(rows) - 10),
                last50=dict(pose_loss=float(np.mean([r['pose_loss'] for r in last])), near_gap=float(np.mean([r['near_gap_mean'] for r in last])),
                            truth_beats_all=float(np.mean([r['truth_beats_all'] for r in last]))))
        else:
            d = out / 'dev_30000'; d.mkdir(exist_ok=True); dev = T.evaluate(t, u, guard, H, 'DEV', d, ('COS',))
            assert len(R.rows(d / 'DEV.jsonl')) == 10 * len(guard.included('DEV'))
            result.update(cal=cal, dev_30000=dev, dev_sha256=R.sha(d / 'DEV.jsonl'), final_checkpoint_sha256=R.sha(out / 'checkpoints/step_30000.pt'))
        assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
        assert all(r['role'] == 'FIT' for r in guard.reads if 'pose-ranking' in r.get('purpose', '')); result['roles_read'] = sorted({r['role'] for r in guard.reads})
        result['seconds'] = time.time() - started
        R.write(out / 'COMPLETE.json', result); R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', identity=ident, seconds=result['seconds']))
        print('POSE_COMPLETE', name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'calibrate', 'smoke', 'run'])
    p.add_argument('--ds', choices=('bf', 'epfl')); p.add_argument('--arm', choices=ARMS); p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--steps', type=int, default=300); a = p.parse_args()
    if a.mode == 'check': cpu_check()
    elif a.mode == 'calibrate': calibrate()
    elif a.mode == 'smoke':
        for arm in ARMS: run(a.ds, arm, a.seed, a.steps)
    else: run(a.ds, a.arm, a.seed)
