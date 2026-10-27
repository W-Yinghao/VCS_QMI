"""R24 CoMIR closing experiments (PROTOCOL.json). Imports the frozen VCS-CoMIR runner (handoff/vcs_comir_20261026/run.py) unchanged.

  train    cross-modal-only InfoNCE (nce_mse_x / nce_cos_x): the original 2B-point expression with the same-modality off-diagonal
           entries additionally masked to -inf; every other part of the step is the base ObjectiveMixin.step. DEV via the base evaluator.
  readout  an existing base final checkpoint evaluated on DEV with ncc / cos (and mse) readouts of the CoMIR latents; same session,
           positions, perturbations and searcher; only RigidEnergy.batch changes.
  diag     FIT-only: positive / negative d/tau of the exp arms and ||grad contrastive|| / ||grad L1+L2|| per arm (final and step 0).
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('r24_base_run', BASE / 'handoff/vcs_comir_20261026/run.py')
V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V)
R, A, torch, F = V.R, V.A, V.torch, V.F
CP = R.CP
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/vcs_comir_close_20261027')
BASE_RUNS = V.OUT / 'runs'
SEEDS = V.FINAL_SEEDS
X_CONFIGS = {'nce_mse_x_t2.0_lr0.01': dict(obj='nce_mse_x', tau=2.0, lr=0.01), 'nce_cos_x_t0.1_lr0.01': dict(obj='nce_cos_x', tau=0.1, lr=0.01)}
ARMS = ('nce_mse_t2.0_lr0.01', 'nce_cos_t0.1_lr0.01', 'vcs_exp_t0.5_lr0.01', 'vcs_cos_lr0.003', 'bce_exp_t0.5_lr0.01', 'bce_cos_lr0.003')
READOUTS = ('mse', 'ncc', 'cos')
RESTORE_SAME = False     # CPU check only: True restores the same-modality negatives (mask = identity), i.e. the original InfoNCE


def verify():
    V.verify()
    for rel, h in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        if R.sha(BASE / rel) != h:
            raise RuntimeError('Frozen file changed: ' + rel)
    return R.sha(HERE / 'FREEZE.json')


# ------------------------------------------------------------------------------------------------------------ cross-modal NCE
_base_contrastive = V.contrastive


def contrastive(obj, la, lb, tau):
    if not obj.endswith('_x'):
        return _base_contrastive(obj, la, lb, tau)
    batch = la.shape[0]
    points = torch.cat((la, lb)).reshape(2 * batch, -1)
    if obj == 'nce_mse_x':             # the original CoMIRTrainer.step expression, same operation order
        similarity = -torch.cdist(points, points).square() / points.shape[1] / tau
    else:
        unit = F.normalize(points, dim=1)
        similarity = unit @ unit.T / tau
    ids = torch.arange(2 * batch, device=la.device)
    positives = similarity[ids, (ids + batch) % (2 * batch)]
    mask = torch.eye(2 * batch, device=la.device, dtype=torch.bool)
    if not RESTORE_SAME:
        side = ids < batch
        mask = mask | (side[:, None] == side[None, :])      # same-modality pairs (diagonal included) out of every denominator
    negatives = similarity.masked_fill(mask, -float('inf'))
    return (-positives + torch.logsumexp(negatives, 1)).mean(), {}


V.contrastive = contrastive    # ObjectiveMixin.step resolves this module global at call time


# ------------------------------------------------------------------------------------------------------------ readouts
class ReadoutEnergy(A.Q.RigidEnergy):
    """The host 'comir' energy on the same positions; mse is the unmodified host method."""
    readout = 'mse'

    def batch(self, th, t):
        if self.readout == 'mse':
            return super().batch(th, t)
        torch, F = self.torch, self.F
        v = A.Q.sample_pts(torch, F, self.mov, A.Q.apply_T(torch, th, t, self.x))[..., 0]
        u = self.u[None].expand_as(v)
        if self.readout == 'ncc':
            u, v = u - u.mean(1, keepdim=True), v - v.mean(1, keepdim=True)
        return (u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1)).clamp_min(1e-12)


def readout_energy(finite, readout):
    inner = copy.copy(finite.energy)
    if type(inner) is not A.Q.RigidEnergy or inner.kind != 'comir':
        raise TypeError('Expected the host comir RigidEnergy')
    inner.__class__ = ReadoutEnergy; inner.readout = readout
    return type(finite)(inner)


def session_for(t, ds, guard, H, role, cores):
    return (V.RegistrationSession('CoMIR_native', t, guard, role, cores, t.device, {}) if ds == 'bf'
            else H.EPFLRegistrationSession('CoMIR_native', t, guard, role, cores, t.device, {}))


def evaluate_readouts(t, ds, guard, H, readouts, out):
    cores = sorted(guard.included('DEV'))
    done = {r: {(x['core'], x['perturbation']) for x in R.rows(out / f'DEV_{r}.jsonl')} for r in readouts}
    with CP.evaluation_state(t):
        session = session_for(t, ds, guard, H, 'DEV', cores)
        for core in cores:
            for i in range(R.N_PERT):
                todo = [r for r in readouts if (core, i) not in done[r]]
                if not todo:
                    continue
                finite, prep = session.energy(core, i)
                for r in todo:
                    rec = R.plain(A.registration_record(readout_energy(finite, r), t.device, session.perturbations[core][i]))
                    if not all(math.isfinite(rec[k]) for k in ('err_full', 'F_hat', 'F_true')):
                        raise FloatingPointError('Non-finite registration record')
                    rec.update(core=core, perturbation=i, role='DEV', readout=r, scene=guard.unit_of(core) if H else core, **prep)
                    R.append(out / f'DEV_{r}.jsonl', rec); done[r].add((core, i))
    result = {}
    for r in readouts:
        recs = R.rows(out / f'DEV_{r}.jsonl')
        if len(recs) != len(cores) * R.N_PERT:
            raise RuntimeError('Incomplete readout block ' + r)
        result[r] = V.summarize(recs)
    return result


# ------------------------------------------------------------------------------------------------------------ helpers
def base_final(ds, name, seed):
    d = BASE_RUNS / f'{ds}_final_{name}_s{seed}'
    c = json.loads((d / 'COMPLETE.json').read_text())
    if c['status'] != 'COMPLETE' or c['identity']['seed'] != seed or c['identity']['phase'] != 'final':
        raise RuntimeError('Base final not COMPLETE: ' + d.name)
    return d, c


def base_config(name):
    return {V.cfg_name(c): c for c in V.cal_grid()}[name]


def gpu_job(freeze):
    gate = json.loads((HERE / 'CPU_CHECK.json').read_text())
    if gate['status'] != 'PASS' or gate['freeze_sha256'] != freeze:
        raise RuntimeError('CPU check is not PASS for this freeze')
    if not (torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')):
        raise RuntimeError('GPU Slurm job required')
    torch.set_num_threads(4)
    return os.environ['SLURM_JOB_ID'], V.environment_check()


def finish(out, guard, result, started, job, ident):
    roles = {r['role'] for r in guard.reads}
    if not roles <= {'FIT', 'CAL', 'DEV'}:
        raise PermissionError('Unexpected data role read: ' + str(roles))
    result.update(roles_read=sorted(roles), seconds=time.time() - started, job=job, gpu=torch.cuda.get_device_name())
    R.write(out / 'COMPLETE.json', result)
    R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', identity=ident, seconds=result['seconds']))


# ------------------------------------------------------------------------------------------------------------ units
def train_unit(ds, name, seed):
    freeze = verify(); job, env = gpu_job(freeze); c = X_CONFIGS[name]
    out = OUT / 'train' / f'{ds}_{name}_s{seed}'; out.mkdir(parents=True, exist_ok=True)
    ident = dict(kind='train', ds=ds, config=c, seed=seed, steps=V.STEPS, freeze_sha256=freeze); started = time.time()
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'COMPLETE.json').exists() or list(out.glob('ATTEMPT_*.json')):
            raise RuntimeError('Unit has a previous attempt; review it before any rerun')
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', identity=ident, start=started, environment=env))
        u, guard, H = V.context(ds)
        t = V.make(ds, c, seed, 'cuda', guard, H)
        initial = A.state_hash(*t.modules())
        with (out / 'TRAIN.jsonl').open('a') as trace:
            while t.n_steps < V.STEPS:
                row = t.step()
                if not (math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])):
                    raise FloatingPointError('Non-finite training step')
                trace.write(json.dumps(R.plain(row), allow_nan=False) + '\n')
        torch.cuda.synchronize(); train_seconds = time.time() - started
        meta = CP.save(t, out / 'checkpoints' / f'step_{V.STEPS:04d}.pt', ident)
        _, ref = base_final(ds, name.replace('_x_', '_'), seed)
        result = dict(status='COMPLETE', identity=ident, environment=env, initial_model_sha256=initial,
                      initial_equals_full_nce=initial == ref['initial_model_sha256'], checkpoint=meta, train_seconds=train_seconds,
                      DEV=V.evaluate(t, ds, guard, H, 'DEV', out / 'DEV.jsonl'))
        finish(out, guard, result, started, job, ident)
        print('R24_CLOSE_TRAIN', out.name, result['DEV']['success9'], flush=True)


def readout_unit(ds, name, seed, readouts):
    freeze = verify(); job, env = gpu_job(freeze)
    src, ref = base_final(ds, name, seed)
    out = OUT / 'readout' / f'{ds}_{name}_s{seed}'; out.mkdir(parents=True, exist_ok=True)
    ident = dict(kind='readout', ds=ds, arm=name, seed=seed, readouts=list(readouts), freeze_sha256=freeze); started = time.time()
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'COMPLETE.json').exists():
            raise RuntimeError('Readout unit already COMPLETE')
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', identity=ident, start=started, environment=env))
        u, guard, H = V.context(ds)
        t = V.make(ds, base_config(name), seed, 'cuda', guard, H)
        meta = CP.load(t, src / 'checkpoints' / f'step_{V.STEPS:04d}.pt', ref['identity'])
        if meta['sha256'] != ref['checkpoint']['sha256']:
            raise RuntimeError('Checkpoint differs from the base COMPLETE record')
        result = dict(status='COMPLETE', identity=ident, environment=env, checkpoint_sha256=meta['sha256'],
                      DEV=evaluate_readouts(t, ds, guard, H, readouts, out))
        finish(out, guard, result, started, job, ident)
        print('R24_CLOSE_READOUT', out.name, {r: v['success9'] for r, v in result['DEV'].items()}, flush=True)


# ------------------------------------------------------------------------------------------------------------ FIT diagnostics
def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(zip(('q05', 'q25', 'q50', 'q75', 'q95'), np.quantile(x, [.05, .25, .5, .75, .95]).tolist()), mean=float(x.mean()))


def diag_model(t, obj, tau, n_batches=16, seed=777):
    """Gradients of the contrastive and the regulariser terms on fixed FIT batches (train mode, as the optimiser step)."""
    g_saved = t.g; t.g = np.random.default_rng(seed); params = [p for m in t.modules() for p in m.parameters()]
    stats = dict(main=[], reg=[], cos=[], pos_x=[], neg_x=[], vcs_pos_factor=[], vcs_neg_factor=[], T_pos=[], T_neg=[])
    for m in t.modules():
        m.train()
    for b in range(n_batches):
        xa, xb = t.draw(); batch = t.cfg.batch_size
        ka, kb = t.g.integers(4, size=batch), t.g.integers(4, size=batch)

        def rotate(values, rotations):
            return torch.stack([torch.rot90(values[i], int(rotations[i]), (1, 2)) for i in range(len(values))])

        def term(which):
            # the official DenseUNet uses reentrant checkpointing (no autograd.grad): one forward + backward per term, same dropout seed
            torch.manual_seed(seed + b)
            la = rotate(t.moving_encoder(rotate(xa, ka)), -ka); lb = rotate(t.reference_encoder(rotate(xb, kb)), -kb)
            if which == 'main':
                value = V.contrastive(obj, la, lb, tau if tau is not None else t.cfg.temperature)[0]
            else:
                size = la.numel() + lb.numel()
                value = (t.cfg.regularization_l1 * (1 + la.abs().sum() + lb.abs().sum()) / size
                         + t.cfg.regularization_l2 * (1 + la.square().sum() + lb.square().sum()) / size)
            for q in params:
                q.grad = None
            value.backward()
            g = torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1) for q in params]).clone()
            for q in params:
                q.grad = None
            return g, la.detach(), lb.detach()
        vm, la, lb = term('main'); vr, _, _ = term('reg')
        stats['main'].append(float(vm.norm())); stats['reg'].append(float(vr.norm()))
        stats['cos'].append(float(vm @ vr / (vm.norm() * vr.norm()).clamp_min(1e-30)))
        with torch.no_grad():
            a, bb = la.reshape(batch, -1), lb.reshape(batch, -1); eye = torch.eye(batch, dtype=torch.bool, device=la.device)
            if obj.endswith('_exp') or obj.startswith('nce_mse'):
                x = (torch.cdist(a, bb).square() / a.shape[1] / tau).double()
                stats['pos_x'] += x[eye].tolist(); stats['neg_x'] += x[~eye].tolist()
                e = torch.exp(-x)
                stats['vcs_pos_factor'] += (4 / tau * e * (1 - e))[eye].tolist(); stats['vcs_neg_factor'] += (4 / tau * e * e)[~eye].tolist()
            else:
                T = F.normalize(a, dim=1) @ F.normalize(bb, dim=1).T
                stats['T_pos'] += T[eye].tolist(); stats['T_neg'] += T[~eye].tolist()
    t.g = g_saved
    out = dict(main_grad_norm=quantiles(stats['main']), reg_grad_norm=quantiles(stats['reg']),
               ratio_main_over_reg=quantiles(np.array(stats['main']) / np.array(stats['reg'])), cos_main_reg=quantiles(stats['cos']))
    for k in ('pos_x', 'neg_x', 'vcs_pos_factor', 'vcs_neg_factor', 'T_pos', 'T_neg'):
        if stats[k]:
            out[k] = quantiles(stats[k])
    if stats['pos_x']:
        px = np.array(stats['pos_x']); out['pos_x_fraction_below'] = {str(v): float((px < v).mean()) for v in (0.05, 0.1, 0.25, 0.5, 1.0)}
    return out


def diag_unit(ds, device='cuda', arms=ARMS, seeds=SEEDS, n_batches=16):
    freeze = verify()
    if device == 'cuda':
        job, env = gpu_job(freeze)
    out = OUT / 'diag'; out.mkdir(parents=True, exist_ok=True); started = time.time()
    u, guard, H = V.context(ds); res = {}
    for name in arms:
        c = base_config(name)
        for seed in seeds:
            src, ref = base_final(ds, name, seed)
            t0 = V.make(ds, c, seed, device, guard, H)
            if A.state_hash(*t0.modules()) != ref['initial_model_sha256']:
                raise RuntimeError('Initial model differs from the base record')
            init = diag_model(t0, c['obj'], c['tau'], n_batches); del t0
            t = V.make(ds, c, seed, device, guard, H)
            CP.load(t, src / 'checkpoints' / f'step_{V.STEPS:04d}.pt', ref['identity'])
            res[f'{name}_s{seed}'] = dict(arm=name, seed=seed, step0=init, final=diag_model(t, c['obj'], c['tau'], n_batches))
            print('R24_CLOSE_DIAG', ds, name, seed, res[f'{name}_s{seed}']['final']['ratio_main_over_reg']['q50'], flush=True)
    roles = {r['role'] for r in guard.reads}
    if not roles <= {'FIT', 'CAL', 'DEV'}:
        raise PermissionError('Unexpected data role read: ' + str(roles))
    result = dict(status='COMPLETE', ds=ds, freeze_sha256=freeze, n_batches=n_batches, batch_seed=777, results=res,
                  seconds=time.time() - started, roles_read=sorted(roles))
    if device == 'cuda':
        result.update(job=job, gpu=torch.cuda.get_device_name(), environment=env)
        R.write(out / f'DIAG_{ds}.json', result)
    return result


# ------------------------------------------------------------------------------------------------------------ CPU check
def cpu_check():
    global RESTORE_SAME
    freeze = verify(); torch.set_num_threads(4); torch.manual_seed(0); checks = []
    # 1. cross-modal-only formulas vs an fp64 dense loop; restored version equals the base objective; gradcheck
    la = torch.randn(5, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    lb = torch.randn(5, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    a, b = la.reshape(5, -1), lb.reshape(5, -1)
    for obj, base in (('nce_mse_x', 'nce_mse'), ('nce_cos_x', 'nce_cos')):
        tau = 0.7
        s = (lambda x, y: -float(((x - y) ** 2).sum()) / a.shape[1] / tau) if obj == 'nce_mse_x' else \
            (lambda x, y: float(x @ y / (x.norm() * y.norm())) / tau)
        ref = 0.0
        for i in range(5):
            ref += -s(a[i], b[i]) + math.log(sum(math.exp(s(a[i], b[j])) for j in range(5)))
            ref += -s(b[i], a[i]) + math.log(sum(math.exp(s(b[i], a[j])) for j in range(5)))
        ref /= 10
        val = float(contrastive(obj, la, lb, tau)[0])
        assert abs(val - ref) < 1e-10, (obj, val, ref)
        assert torch.autograd.gradcheck(lambda x, y: contrastive(obj, x, y, tau)[0], (la, lb)), obj
        RESTORE_SAME = True
        try:
            assert torch.equal(contrastive(obj, la, lb, tau)[0], _base_contrastive(base, la, lb, tau)[0]), obj
        finally:
            RESTORE_SAME = False
        assert not torch.equal(contrastive(obj, la, lb, tau)[0], _base_contrastive(base, la, lb, tau)[0]), obj
        checks.append(f'{obj}: fp64 dense-loop reference (both directions, B cross-modal terms per denominator), gradcheck, '
                      f'restored same-modality terms == base {base} exactly')
    # 2. real FIT steps: restored x-arm == original CoMIRTrainer.step bitwise; cross-only arm finite with the same sample stream
    u, guard, H = V.context('bf')
    cfg = V.replace(V.CoMIRConfig(seed=1001, temperature=2.0), batch_size=4)
    for obj, orig_obj in (('nce_mse_x', None), ('nce_cos_x', 'nce_cos')):
        orig = V.CoMIRTrainer(cfg, guard, 'cpu') if orig_obj is None else V.BFTrainer(cfg, guard, 'cpu')
        if orig_obj:
            orig.objective = orig_obj
        mine = V.BFTrainer(cfg, guard, 'cpu'); mine.objective = obj
        assert A.state_hash(*orig.modules()) == A.state_hash(*mine.modules())
        RESTORE_SAME = True
        try:
            for _ in range(2):
                state = torch.get_rng_state(); r1 = orig.step(); torch.set_rng_state(state); r2 = mine.step()
                assert r1['loss'] == r2['loss'] and r1['grad_norm_preclip'] == r2['grad_norm_preclip'], (r1, r2)
        finally:
            RESTORE_SAME = False
        assert A.state_hash(*orig.modules()) == A.state_hash(*mine.modules()), obj
        checks.append(f'{obj} with same-modality terms restored: two real FIT steps bitwise equal to '
                      + ('the original CoMIRTrainer.step' if orig_obj is None else 'the base nce_cos step'))
        x = V.BFTrainer(cfg, guard, 'cpu'); x.objective = obj; ref_t = V.CoMIRTrainer(cfg, guard, 'cpu')
        row = x.step(); ref_t.step()
        assert math.isfinite(row['loss']) and x.g.bit_generator.state == ref_t.g.bit_generator.state
        checks.append(f'{obj} cross-modal only: finite real FIT step, sample stream identical to the original')
    # 3. readouts on one CAL problem: mse copy bitwise equal to the host energy; ncc / cos equal numpy references; scale invariance
    with CP.evaluation_state(mine):
        core = sorted(guard.included('CAL'))[0]
        session = V.RegistrationSession('CoMIR_native', mine, guard, 'CAL', [core], mine.device, {})
        finite, _ = session.energy(core, 0)
        th = torch.tensor([0.0, 0.05, -0.1]); tt = torch.tensor([[0.0, 0.0], [1.5, -2.0], [-3.0, 0.5]])
        host = finite.energy.batch(th, tt)
        assert torch.equal(readout_energy(finite, 'mse').energy.batch(th, tt), host)
        v = A.Q.sample_pts(torch, F, finite.energy.mov, A.Q.apply_T(torch, th, tt, finite.energy.x))[..., 0].double().numpy()
        uu = finite.energy.u.double().numpy()
        for r in ('ncc', 'cos'):
            e = readout_energy(finite, r)
            got = e.energy.batch(th, tt).double().numpy()
            for k in range(3):
                x1, x2 = (uu - uu.mean(), v[k] - v[k].mean()) if r == 'ncc' else (uu, v[k])
                assert abs(got[k] - x1 @ x2 / np.linalg.norm(x1) / np.linalg.norm(x2)) < 1e-5, (r, got[k])
            assert e.energy.x is finite.energy.x and torch.isfinite(e.grid_values(0.0, tt)).all()
            scaled = copy.copy(e.energy); scaled.mov = finite.energy.mov * 3.0 + (0.7 if r == 'ncc' else 0.0)
            assert torch.allclose(scaled.batch(th, tt), e.energy.batch(th, tt), atol=1e-5), r
            assert math.isfinite(e.value(*session.perturbations[core][0][:1], session.perturbations[core][0][1:]))
        checks.append('readouts: mse copy bitwise equal to the host comir energy; ncc / cos equal numpy references on 3 poses, '
                      'share the host positions, invariant to moving-latent scale (ncc also to offset), finite grid / truth values')
    # 4. diagnostics run on CPU (one forward + backward per term)
    t = V.BFTrainer(cfg, guard, 'cpu'); t.objective = 'vcs_exp'
    d = diag_model(t, 'vcs_exp', 0.5, n_batches=1)
    assert all(math.isfinite(d[k]['mean']) for k in ('main_grad_norm', 'reg_grad_norm', 'pos_x', 'vcs_pos_factor'))
    checks.append('diagnostics run on a real FIT batch (vcs_exp, CPU), finite')
    assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
    # 5. every checkpoint the GPU jobs will read exists and matches its base record
    for ds in ('bf', 'epfl'):
        for name in ARMS:
            for seed in SEEDS:
                src, ref = base_final(ds, name, seed)
                assert R.sha(src / 'checkpoints' / f'step_{V.STEPS:04d}.pt') == ref['checkpoint']['sha256'], (ds, name, seed)
    checks.append('36 base final checkpoints present, sha256 equal to their COMPLETE records')
    R.write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, checks=checks, utc=R.utc()))
    print('R24_CLOSE_CPU PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('check', 'train', 'readout', 'diag'))
    p.add_argument('--ds', choices=('bf', 'epfl'), default='bf')
    p.add_argument('--name'); p.add_argument('--seed', type=int); p.add_argument('--readouts', default='ncc,cos')
    a = p.parse_args()
    if a.mode == 'check':
        cpu_check()
    elif a.mode == 'train':
        if a.seed not in SEEDS:
            raise SystemExit('seed must be one of ' + str(SEEDS))
        train_unit(a.ds, a.name, a.seed)
    elif a.mode == 'readout':
        readouts = tuple(a.readouts.split(','))
        assert set(readouts) <= set(READOUTS) and a.name in ARMS and a.seed in SEEDS
        readout_unit(a.ds, a.name, a.seed, readouts)
    else:
        diag_unit(a.ds)
