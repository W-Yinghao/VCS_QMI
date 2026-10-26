"""Registration pose term v2: v1 local ring + a global term through the host COARSE-GRID energy path (R22, 2026-10-26).

L = L_local + L_global (+ lambda * base). L_local = v1's near-ring ranking on S_loc (host warp-then-encode BOX1 path).
L_global ranks the host grid nodes enclosing the truth (basin B: 2 bracketing grid rotations x up-to-4 enclosing grid translations,
soft max S_B) above 24 far grid nodes (err_full >= 42) on S_grid, the exact Q.RigidEnergy.grid_values 'critic' path with the COS
readout (rotation-only warp into BOX2, eV once per rotation, shifted sampling). Same init, host sampler stream, LR schedule and
pose-problem stream as v1; negatives use a separate RNG. v1 (handoff/r22_reg_pose_20261025/run.py) is imported, never edited.
"""
import argparse, fcntl, importlib.util, json, math, os, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
V1DIR = BASE / 'handoff/r22_reg_pose_20261025'
_spec = importlib.util.spec_from_file_location('reg_pose_v1', V1DIR / 'run.py')
V1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V1)
L, T, R, A, torch = V1.L, V1.T, V1.R, V1.A, V1.torch
F, Q, P = V1.F, V1.Q, V1.P
ARMS = ('pose2_only', 'pose2_vcs', 'pose2_bce')
V1_ARM = {'pose2_only': 'pose_only', 'pose2_vcs': 'pose_vcs', 'pose2_bce': 'pose_bce'}
BASE_OBJECTIVE = {a: V1.BASE_OBJECTIVE[V1_ARM[a]] for a in ARMS}
REFERENCE = {a: V1.REFERENCE[V1_ARM[a]] for a in ARMS}
NEAR = V1.NEAR
CONST = json.loads((HERE / 'CONSTANTS.json').read_text())
THS, TS = Q.grid_points()                        # 21 grid rotations (deg), 508 grid translations (y, x) at 512 px
TINDEX = {(int(round(y)), int(round(x))): j for j, (y, x) in enumerate(TS)}
assert len(TINDEX) == len(TS) and all(float(v).is_integer() for v in TS.ravel())
CORNERS = np.array([[-Q.HALF834, -Q.HALF834], [-Q.HALF834, Q.HALF834], [Q.HALF834, -Q.HALF834], [Q.HALF834, Q.HALF834]])


def verify():
    for p, h in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        assert R.sha(BASE / p) == h, p
    return R.sha(HERE / 'FREEZE.json')


def rot(d):
    r = math.radians(d); return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])


def err_all(th_grid, truth):
    """Q.corner_error_full(th_grid, t, theta*, t*) for every grid translation t (vectorised; equality checked on CPU)."""
    th, ty, tx = truth
    a = CORNERS @ rot(th_grid).T - CORNERS @ rot(th).T
    e = a[None] + (TS - np.array([ty, tx]))[:, None, :]
    return np.linalg.norm(e, axis=2).mean(1) * Q.P.DOWN


def grid_sets(truth, rng, const=CONST):
    """Basin and negative grid nodes for one problem -> (rotation indices [4], per-rotation translation index lists, n_basin per bracket)."""
    th, ty, tx = truth; step = Q.P.GRID_STEP
    k = min(int(math.floor((th + Q.THMAX) / Q.TH_STEP)), len(THS) - 2); brackets = [k, k + 1]
    assert THS[k] <= th <= THS[k + 1]
    gy = -Q.TMAX + step * math.floor((ty + Q.TMAX) / step); gx = -Q.TMAX + step * math.floor((tx + Q.TMAX) / step)
    basin_t = sorted({TINDEX[(int(round(gy + a)), int(round(gx + b)))] for a in (0, step) for b in (0, step)
                      if (int(round(gy + a)), int(round(gx + b))) in TINDEX})
    assert basin_t, truth
    thr = const['neg_min_err_full']; trans = []
    for r in brackets:
        e = err_all(THS[r], truth); pool = [j for j in np.flatnonzero(e >= thr) if j not in basin_t]
        trans.append(basin_t + [int(j) for j in rng.choice(pool, const['n_neg_bracket_per_rotation'], replace=False)])
    others = [int(r) for r in rng.choice([r for r in range(len(THS)) if r not in brackets], const['n_other_rotations'], replace=False)]
    for r in others:
        pool = np.flatnonzero(err_all(THS[r], truth) >= thr)
        trans.append([int(j) for j in rng.choice(pool, const['n_neg_other_per_rotation'], replace=False)])
    return brackets + others, trans, len(basin_t)


def local_candidates(truth, device):
    th, ty, tx = truth
    rows = [(th, ty, tx)] + [(th + a, ty + b, tx + c) for a, b, c in NEAR]
    return (torch.tensor([math.radians(r[0]) for r in rows], dtype=torch.float32, device=device),
            torch.tensor([[r[1], r[2]] for r in rows], dtype=torch.float32, device=device))


def energies(eU, eV, ref, mov, x, th, t, rots, trans):
    """S_loc [7] (v1 pose_energy body) and S_grid per rotation (Q.RigidEnergy.grid_values 'critic' + COS), one reference encode."""
    dev = mov.device
    refmap = eU(ref[None])[0]; u = P.sample_map(torch, F, refmap, x)
    box = Q.grid_coords(torch, *Q.BOX1, dev); n = Q.BOX1[1] - Q.BOX1[0] + 1
    W = Q.sample_pts(torch, F, mov, Q.apply_T(torch, th, t, box))[..., 0].reshape(len(th), 1, n, n)
    maps = eV(W)
    dv = torch.stack([P.sample_map(torch, F, maps[k], x - Q.BOX1[0]) for k in range(len(th))])
    un, vn = T.normalized(u[None], dv); s_loc = (un * vn).sum(-1).mean(-1)
    box2 = Q.grid_coords(torch, *Q.BOX2, dev); n2 = Q.BOX2[1] - Q.BOX2[0] + 1
    thg = torch.tensor([math.radians(THS[r]) for r in rots], dtype=torch.float32, device=dev)
    W2 = Q.sample_pts(torch, F, mov, Q.apply_T(torch, thg, torch.zeros(len(rots), 2, device=dev), box2))[..., 0].reshape(len(rots), 1, n2, n2)
    D = eV(W2); s_grid = []
    for k, r in enumerate(rots):
        ts = torch.tensor(TS[trans[k]], dtype=torch.float32, device=dev)
        c, s = math.cos(math.radians(THS[r])), math.sin(math.radians(THS[r]))
        sh = torch.stack([c * ts[:, 0] + s * ts[:, 1], -s * ts[:, 0] + c * ts[:, 1]], -1)
        g = Q.sample_pts(torch, F, D[k], x[None] + sh[:, None] - Q.BOX2[0])
        ug, vg = T.normalized(u[None], g); s_grid.append((ug * vg).sum(-1).mean(-1))
    return s_loc, s_grid


def losses(s_loc, s_grid, nb, const):
    tau = const['tau']
    z = torch.cat([s_loc.new_zeros(1), (s_loc[1:] - s_loc[0] + const['m_near']) / tau]); l_loc = tau * torch.logsumexp(z, 0)
    basin = torch.cat([s_grid[0][:nb], s_grid[1][:nb]]); neg = torch.cat([s_grid[0][nb:], s_grid[1][nb:]] + s_grid[2:])
    s_b = tau * torch.logsumexp(basin / tau, 0)
    zg = torch.cat([neg.new_zeros(1), (neg - s_b + const['m_glob']) / tau]); l_glob = tau * torch.logsumexp(zg, 0)
    return l_loc, l_glob, basin, neg


class Problems(V1.PoseProblems):
    """v1 pose-problem stream (unchanged) plus a separate RNG for the grid negatives."""
    def __init__(self, ds, guard, H, fit, device, seed):
        super().__init__(ds, guard, H, fit, device, seed)
        self.grng = np.random.default_rng(Q.seed_of(f'R22-POSE2|{ds}|{seed}'))


def train_step(t, arm, problems, const):
    lr = L.lr_at(t.n_steps + 1)
    for g in t.opt.param_groups: g['lr'] = lr
    hb, bu, bv = t.draw(); row = dict(step=hb.step, lr=lr, batch_sha256=L.batch_sha(hb))
    start = time.perf_counter()
    c, i, xnp = problems.draw(); ref, mov, truth = problems.get(c, i)
    x = torch.tensor(xnp, dtype=torch.float32, device=t.device); th, tt = local_candidates(truth, t.device)
    rots, trans, nb = grid_sets(truth, problems.grng, const)
    s_loc, s_grid = energies(t.eU, t.eV, ref, mov, x, th, tt, rots, trans)
    l_loc, l_glob, basin, neg = losses(s_loc, s_grid, nb, const); total = l_loc + l_glob
    if BASE_OBJECTIVE[arm] is not None:
        du, dv = A.batch_descriptors(t.eU, t.eV, bu, bv, hb.yx)
        base, j = T.objective(du, dv, BASE_OBJECTIVE[arm], torch.as_tensor(hb.perm, device=t.device))
        total = total + const['lam'] * base; row['base_loss'] = float(base.detach())
        if j is not None: row['J'] = float(j.detach())
    t.opt.zero_grad(set_to_none=True); total.backward()
    row['grad_norm_preclip'] = float(torch.nn.utils.clip_grad_norm_(t.clip_params, t.cfg.clip_norm)); t.opt.step(); t.n_steps += 1
    if t.device.type == 'cuda': torch.cuda.synchronize()
    with torch.no_grad():
        gap = (s_loc[1:] - s_loc[0]).cpu().numpy(); bmax, nmax = float(basin.max()), float(neg.max())
    row.update(loss=float(total.detach()), local_loss=float(l_loc.detach()), global_loss=float(l_glob.detach()), S_true=float(s_loc[0].detach()),
               problem=[c, i], near_gap_mean=float(gap.mean()), truth_beats_ring=bool((gap < 0).all()), n_basin=nb,
               basin_max=bmax, neg_max=nmax, basin_beats_all=bmax > nmax, rotations=[float(THS[r]) for r in rots],
               synchronized_step_seconds=time.perf_counter() - start)
    assert row['step'] == t.n_steps and math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])
    return row


def v1_problems(ds, arm, seed, n=5):
    return [json.loads(l)['problem'] for _, l in zip(range(n), open(V1DIR / f'runs/{ds}_{V1_ARM[arm]}_{seed}/TRAIN.jsonl'))]


def reference_rows(ds, arm, seed):
    return [json.loads(l) for _, l in zip(range(5), open(BASE / f'handoff/r22_geometry_long_v2_20261024/runs/{ds}_{REFERENCE[arm]}_{seed}/TRAIN.jsonl'))]


def host_energy(t, ref, mov):
    refmap = t.eU(ref[None])[0]
    e = Q.RigidEnergy(torch, F, 'critic', refmap, mov, np.random.default_rng(7), eV=t.eV, critic=t.critic)
    e.j_critic = T.types.MethodType(T.cosine_energy, e); e.geometry_readout = 'COS'
    return e


def cpu_check():
    freeze = verify(); torch.set_num_threads(4); checks = []
    assert {'tau', 'm_near', 'm_glob', 'lam'} <= set(CONST)
    # vectorised corner error equals the host function
    rng = np.random.default_rng(3)
    for _ in range(20):
        truth = (float(rng.uniform(-30, 30)), float(rng.uniform(-40, 40)), float(rng.uniform(-30, 30))); r = int(rng.integers(len(THS)))
        e = err_all(THS[r], truth); js = rng.integers(len(TS), size=8)
        assert np.allclose(e[js], [Q.corner_error_full(THS[r], TS[j], truth[0], truth[1:]) for j in js], atol=1e-9, rtol=1e-12)
    coverage = {}
    for ds in ('bf', 'epfl'):
        u, guard, H, fit, imgs = L.context(ds, 1001, 'cpu', images=True)
        # basin coverage and negative pools for every FIT perturbation
        pr = Problems(ds, guard, H, fit, 'cpu', 1001); nbs, berr = [], []
        for c in pr.fit:
            for truth in pr.pert[c]:
                rots, trans, nb = grid_sets(truth, np.random.default_rng(0)); nbs.append(nb)
                berr += [float(err_all(THS[r], truth)[j]) for r in rots[:2] for j in trans[rots.index(r)][:nb]]
                for k, r in enumerate(rots):
                    e = err_all(THS[r], truth)[trans[k][nb if k < 2 else 0:]]; assert (e >= CONST['neg_min_err_full']).all()
                assert len(set(trans[0][nb:])) == 4 and len(set(trans[2])) == 8 and len(set(rots)) == 4
        coverage[ds] = dict(fit_problems=len(nbs), basin_size_min=min(nbs), basin_size_counts={int(k): nbs.count(k) for k in sorted(set(nbs))},
                            basin_err_full_max=max(berr), basin_err_full_median=float(np.median(berr)))
        # sampler stream: fast-forward 3000 draws, steps 3001-3005 equal long-v2 for every arm reference
        for arm in ARMS:
            t = T.GeometryTrainer(L.config(1001), {}, fit, 'cpu', 'vcs_exact')
            for _ in range(3000): t.sampler.draw()
            assert [L.batch_sha(t.sampler.draw()) for _ in range(5)] == [r['batch_sha256'] for r in reference_rows(ds, arm, 1001)], (ds, arm)
        orig = json.loads((BASE / f'handoff/r22_geometry_train_20261023/runs/{ds}_vcs_exact_1001/COMPLETE.json').read_text())
        t = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', 'vcs_exact')
        assert A.state_hash(t.eU, t.eV) == orig['initial_encoder_sha256']
        # deployed-energy equivalence: local == host batch == v1 pose_energy; grid == host grid_values (COS)
        pr = Problems(ds, guard, H, fit, 'cpu', 1001); c, i, _ = pr.draw(); ref, mov, truth = pr.get(c, i)
        with torch.no_grad():
            host = host_energy(t, ref, mov); th, tt = local_candidates(truth, 'cpu'); rots, trans, nb = grid_sets(truth, pr.grng)
            s_loc, s_grid = energies(t.eU, t.eV, ref, mov, host.x, th, tt, rots, trans)
            assert torch.allclose(s_loc, host.batch(th, tt), atol=1e-6, rtol=1e-5)
            assert torch.equal(s_loc, V1.pose_energy(t.eU, t.eV, ref, mov, host.x, th, tt))
            gdiff = 0.
            for k, r in enumerate(rots):
                h = host.grid_values(float(THS[r]), torch.tensor(TS, dtype=torch.float32))[trans[k]]
                assert torch.allclose(s_grid[k], h, atol=1e-6, rtol=1e-5), (ds, k, float((s_grid[k] - h).abs().max()))
                gdiff = max(gdiff, float((s_grid[k] - h).abs().max()))
        # base step with the pose terms removed equals GeometryTrainer.step bitwise
        for arm in ('pose2_vcs', 'pose2_bce'):
            a = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', BASE_OBJECTIVE[arm]); b = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', BASE_OBJECTIVE[arm])
            for g in b.opt.param_groups: g['lr'] = 3e-4
            ra = a.step(); hb, bu, bv = b.draw(); du, dv = A.batch_descriptors(b.eU, b.eV, bu, bv, hb.yx)
            base, _ = T.objective(du, dv, BASE_OBJECTIVE[arm], torch.as_tensor(hb.perm)); b.opt.zero_grad(set_to_none=True); base.backward()
            torch.nn.utils.clip_grad_norm_(b.clip_params, b.cfg.clip_norm); b.opt.step()
            assert ra['loss'] == float(base.detach()) and A.state_hash(a.eU, a.eV) == A.state_hash(b.eU, b.eV), (ds, arm)
        # gradients of L_global reach both encoders
        t = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', 'vcs_exact'); s_loc, s_grid = energies(t.eU, t.eV, ref, mov, host.x, th, tt, rots, trans)
        _, l_glob, _, _ = losses(s_loc, s_grid, nb, CONST); l_glob.backward()
        gu = sum(float(p.grad.norm()) for p in t.eU.parameters() if p.grad is not None); gv = sum(float(p.grad.norm()) for p in t.eV.parameters() if p.grad is not None)
        assert gu > 0 and gv > 0 and math.isfinite(gu + gv)
        # one full step per arm: finite, host sampler first draw, same problem as v1 step 1
        for arm in ARMS:
            t = T.GeometryTrainer(L.config(1001), imgs, fit, 'cpu', 'vcs_exact'); pr = Problems(ds, guard, H, fit, 'cpu', 1001)
            row = train_step(t, arm, pr, CONST)
            assert row['batch_sha256'] == L.batch_sha(T.GeometryTrainer(L.config(1001), {}, fit, 'cpu', 'vcs_exact').sampler.draw())
            assert row['problem'] == v1_problems(ds, arm, 1001, 1)[0], (ds, arm)
            checks.append(dict(ds=ds, arm=arm, first_step={k: row[k] for k in ('loss', 'local_loss', 'global_loss', 'S_true', 'near_gap_mean',
                                                                                  'basin_max', 'neg_max', 'n_basin', 'grad_norm_preclip')}))
        checks.append(dict(ds=ds, grid_equivalence_max_abs=gdiff, global_grad_norm_sum=dict(eU=gu, eV=gv)))
        assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
    R.write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, coverage=coverage, checks=checks, verified=[
        'vectorised corner error equals Q.corner_error_full', 'basin non-empty and all negatives err_full>=42 for every FIT perturbation (both datasets)',
        'host sampler steps 3001-3005 equal long-v2 rows (all arm references)', 'initial encoder hash equals the original 3000-step init',
        'S_loc equals host RigidEnergy.batch (COS, atol1e-6) and bitwise equals v1 pose_energy', 'S_grid equals host grid_values (COS, atol1e-6)',
        'pose-free base step bitwise equals GeometryTrainer.step', 'L_global gradients reach eU and eV',
        'one finite step per arm; host sampler untouched; pose problem equals v1 step 1']))
    print('POSE2_CPU PASS', json.dumps(coverage), flush=True)


def diagnose():
    """FIT-only, report-only: local/global gaps and the full-grid rank of the basin on existing seed-1001 vcs_exact checkpoints."""
    torch.set_num_threads(4); out = {}
    for ds in ('bf', 'epfl'):
        u, guard, H, fit, imgs = L.context(ds, 1001, 'cuda', images=False)
        for label, kind in (('init', None), ('step3000', 'orig'), ('step30000', 'long')):
            t = T.GeometryTrainer(L.config(1001), {}, fit, 'cuda', 'vcs_exact')
            if kind == 'orig':
                d = BASE / f'handoff/r22_geometry_train_20261023/runs/{ds}_vcs_exact_1001'; R.CP.load(t, d / 'checkpoints/step_3000.pt', json.loads((d / 'COMPLETE.json').read_text())['identity'])
            if kind == 'long':
                d = BASE / f'handoff/r22_geometry_long_v2_20261024/runs/{ds}_vcs_exact_1001'; R.CP.load(t, d / 'checkpoints/step_30000.pt', json.loads((d / 'COMPLETE.json').read_text())['identity'])
            pr = Problems(ds, guard, H, fit, 'cuda', 99); rec = []
            with torch.no_grad():
                for _ in range(48):
                    c, i, xnp = pr.draw(); ref, mov, truth = pr.get(c, i); x = torch.tensor(xnp, dtype=torch.float32, device='cuda')
                    th, tt = local_candidates(truth, 'cuda'); rots, trans, nb = grid_sets(truth, pr.grng)
                    s_loc, s_grid = energies(t.eU, t.eV, ref, mov, x, th, tt, rots, trans); l_loc, l_glob, basin, neg = losses(s_loc, s_grid, nb, CONST)
                    host = host_energy(t, ref, mov); tsd = torch.tensor(TS, dtype=torch.float32, device='cuda')
                    full = torch.stack([host.grid_values(float(r), tsd) for r in THS]).cpu().numpy()      # [21, 508]
                    bset = {(rr, j) for rr in rots[:2] for j in trans[rots.index(rr)][:nb]}
                    bvals = [full[rr, j] for rr, j in bset]; others = np.array([full[rr, j] for rr in range(len(THS)) for j in range(len(TS)) if (rr, j) not in bset])
                    good = np.stack([err_all(THS[rr], truth) < 42 for rr in range(len(THS))])
                    top3 = np.argsort(full.ravel())[::-1][:3]
                    rec.append(dict(local_gap_max=float((s_loc[1:] - s_loc[0]).max()), truth_beats_ring=bool((s_loc[1:] < s_loc[0]).all()),
                                    sampled_global_gap=float(basin.max() - neg.max()), l_loc=float(l_loc), l_glob=float(l_glob),
                                    basin_rank_full_grid=int((others > max(bvals)).sum()), top3_has_err42=bool(good.ravel()[top3].any())))
            out[f'{ds}_{label}'] = dict(problems=len(rec), truth_beats_ring=float(np.mean([r['truth_beats_ring'] for r in rec])),
                local_gap_max_median=float(np.median([r['local_gap_max'] for r in rec])), sampled_global_gap_median=float(np.median([r['sampled_global_gap'] for r in rec])),
                sampled_basin_beats_all=float(np.mean([r['sampled_global_gap'] > 0 for r in rec])), l_loc_mean=float(np.mean([r['l_loc'] for r in rec])),
                l_glob_mean=float(np.mean([r['l_glob'] for r in rec])), basin_rank_full_grid_median=float(np.median([r['basin_rank_full_grid'] for r in rec])),
                basin_top3_full_grid=float(np.mean([r['basin_rank_full_grid'] < 3 for r in rec])), top3_has_err42=float(np.mean([r['top3_has_err42'] for r in rec])))
            print(ds, label, out[f'{ds}_{label}'], flush=True)
        assert all(r['role'] == 'FIT' for r in guard.reads if 'pose-ranking' in r.get('purpose', ''))
    R.write(HERE / 'DIAGNOSTIC.json', dict(status='COMPLETE', report_only=True, job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
                                          problems_per_cell=48, constants_sha256=R.sha(HERE / 'CONSTANTS.json'), cells=out))


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
        problems = Problems(ds, guard, H, fit, 'cuda', seed); ref_rows = reference_rows(ds, arm, seed); v1p = v1_problems(ds, arm, seed); cal = {}
        stops = (smoke_steps,) if smoke_steps else (10000, 20000, 30000)
        for stop in stops:
            with (out / 'TRAIN.jsonl').open('a') as trace:
                while t.n_steps < stop:
                    row = train_step(t, arm, problems, CONST)
                    if row['step'] <= 5: assert row['problem'] == v1p[row['step'] - 1], 'Pose-problem stream differs from v1'
                    if 3001 <= row['step'] <= 3005: assert row['batch_sha256'] == ref_rows[row['step'] - 3001]['batch_sha256'], 'Host stream differs'
                    trace.write(json.dumps(R.plain(row), allow_nan=False) + '\n')
                    if t.n_steps % 250 == 0: trace.flush(); print(name, 'step', t.n_steps, row['loss'], flush=True)
            if not smoke_steps:
                R.CP.save(t, out / 'checkpoints' / f'step_{stop}.pt', ident)
                d = out / f'cal_{stop}'; d.mkdir(exist_ok=True); cal[stop] = T.evaluate(t, u, guard, H, 'CAL', d, ('COS',))
        rows = [json.loads(l) for l in open(out / 'TRAIN.jsonl')]
        assert [r['step'] for r in rows] == list(range(1, t.n_steps + 1)) and all(r['lr'] == L.lr_at(r['step']) for r in rows)
        last = rows[-1000:]
        result = dict(status='COMPLETE', identity=ident, freeze_sha256=freeze, ds=ds, arm=arm, seed=seed, job=job, constants=CONST,
                      initial_encoder_sha256=initial, steps=t.n_steps, gpu=torch.cuda.get_device_name(), environment=R.environment(),
                      synchronized_training_seconds=sum(r['synchronized_step_seconds'] for r in rows), train_sha256=R.sha(out / 'TRAIN.jsonl'),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      last_rates=dict(steps=len(last), truth_beats_ring=float(np.mean([r['truth_beats_ring'] for r in last])),
                                      basin_beats_all=float(np.mean([r['basin_beats_all'] for r in last])),
                                      local_loss=float(np.mean([r['local_loss'] for r in last])), global_loss=float(np.mean([r['global_loss'] for r in last]))))
        if smoke_steps:
            result['mean_step_ms'] = 1000 * sum(r['synchronized_step_seconds'] for r in rows[10:]) / max(1, len(rows) - 10)
        else:
            d = out / 'dev_30000'; d.mkdir(exist_ok=True); dev = T.evaluate(t, u, guard, H, 'DEV', d, ('COS',))
            assert len(R.rows(d / 'DEV.jsonl')) == 10 * len(guard.included('DEV'))
            result.update(cal=cal, dev_30000=dev, dev_sha256=R.sha(d / 'DEV.jsonl'), final_checkpoint_sha256=R.sha(out / 'checkpoints/step_30000.pt'))
        assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
        assert all(r['role'] == 'FIT' for r in guard.reads if 'pose-ranking' in r.get('purpose', '')); result['roles_read'] = sorted({r['role'] for r in guard.reads})
        result['seconds'] = time.time() - started
        R.write(out / 'COMPLETE.json', result); R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', identity=ident, seconds=result['seconds']))
        print('POSE2_COMPLETE', name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'diagnose', 'smoke', 'run'])
    p.add_argument('--ds', choices=('bf', 'epfl')); p.add_argument('--arm', choices=ARMS); p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--steps', type=int, default=300); a = p.parse_args()
    if a.mode == 'check': cpu_check()
    elif a.mode == 'diagnose': diagnose()
    elif a.mode == 'smoke':
        for arm in ARMS: run(a.ds, arm, a.seed, a.steps)
    else: run(a.ds, a.arm, a.seed)
