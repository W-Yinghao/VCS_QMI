"""R24: VCS / BCE drop-in replacements for CoMIR's InfoNCE (BF-SHG primary, EPFL RGB-NIR secondary).

Only the contrastive term of the V10 official-CoMIR recipe changes (v10.supplement.comir.CoMIRTrainer: official DenseUNet,
23 x 32 steps, batch 32 rotated 128 px pairs, C4 rot90 equivariance, SGD .01 / momentum .9, clip 1, L1/L2 regularisers).
Data stream, initialisation, augmentation, optimiser, steps and the registration (V10 RegistrationSession 'CoMIR_native':
host -MSE latent energy, unchanged grid + 3 starts x 100 Adam) are identical for every arm. The nce_mse arm runs the
original code path (bitwise equal to CoMIRTrainer.step, checked on CPU).

  nce_mse  original InfoNCE, 2B points, s = -||x_i - x_j||^2 / D / tau, same-modality negatives included
  nce_cos  same InfoNCE with s = cos(x_i, x_j) / tau
  vcs_exp  T_ij = 2 exp(-||a_i - b_j||^2 / D / tau) - 1 on cross-modal pairs; J = E_P[T - T^2/2] - E_Q[T + T^2/2],
           P = diagonal (a_i, b_i), Q = off-diagonal cross-modal pairs (empirical product of marginals); loss -J
  vcs_cos  T_ij = cos(a_i, b_j), same J
  bce_exp / bce_cos  balanced BCE with P(+) = (1 + T)/2 on the same T, P = diagonal, Q = off-diagonal cross-modal
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v10/vcs_comir_20261026')
sys.path.insert(0, str(BASE / 'ssl_pilot_v10_r22c/v10/r22_pmi'))
import run_unit as R  # noqa: E402  (pins the host files, puts ssl_pilot_v10_r22c/src on sys.path)
torch, A, F = R.torch, R.A, R.F
from v10.supplement import comir as _comir  # noqa: E402
from v10.supplement.comir import CoMIRConfig, CoMIRTrainer  # noqa: E402
from v10.supplement.registration import RegistrationSession  # noqa: E402

# Same process-local fix as TEST_REPEAT v6 (scripts/v10_test_repeat/evaluate.py): a second CoMIRTrainer in one process sees the official
# 'models.layers' namespace package (__file__ None) and the frozen check misreads it as foreign. Hide only namespace packages whose
# __path__ lies inside the pinned official root, run the frozen check (all file hashes still verified), restore them.
_frozen_verify = _comir.verify_official_source


def _verify_official_source():
    root = Path(json.loads((A.REPO / "v10/supplement/COMIR_SOURCE.json").read_text())["root"]).resolve()
    hidden = {}
    for name, module in tuple(sys.modules.items()):
        if (name == "models" or name.startswith("models.")) and getattr(module, "__file__", None) is None:
            paths = [Path(x).resolve() for x in (getattr(module, "__path__", None) or [])]
            if paths and all(x.is_relative_to(root) for x in paths):
                hidden[name] = sys.modules.pop(name)
    try:
        return _frozen_verify()
    finally:
        sys.modules.update(hidden)


_comir.verify_official_source = _verify_official_source

OBJECTIVES = ('nce_mse', 'nce_cos', 'vcs_exp', 'vcs_cos', 'bce_exp', 'bce_cos')
STEPS = 23 * 32
CAL_SEED = 1017                      # V10 CAL seed convention ("CAL seed is fixed at 1017")
FINAL_SEEDS = (1001, 1002, 1003)
UNIT_JSON = BASE / 'handoff/r22_c3_codex_20261023/units/r22_pmi3_{ds}_MLPp1_s1001.json'   # guard / data-lock source only
V10_COMIR = {'bf': BASE / 'handoff/v10_native_geometry_recovery_20261019/families/4/results/FINAL_DEV/CoMIR_native',
             'epfl': BASE / 'handoff/v10_epfl_production_20261019/results/FINAL_DEV/CoMIR_native'}


def cal_grid():
    rows = []
    for obj in ('nce_mse', 'nce_cos', 'vcs_exp', 'bce_exp'):
        rows += [dict(obj=obj, tau=tau, lr=0.01) for tau in (0.1, 0.5, 2.0)]
    for obj in ('vcs_cos', 'bce_cos'):
        rows += [dict(obj=obj, tau=None, lr=lr) for lr in (0.01, 0.003)]
    return rows


def cfg_name(c):
    return c['obj'] + (f"_t{c['tau']}" if c['tau'] is not None else '') + f"_lr{c['lr']}"


def sha(p):
    return R.sha(p)


def verify():
    for rel, h in json.loads((HERE / 'FREEZE.json').read_text())['files'].items():
        if sha(BASE / rel) != h:
            raise RuntimeError('Frozen file changed: ' + rel)
    return sha(HERE / 'FREEZE.json')


# ------------------------------------------------------------------------------------------------------------ objectives
def contrastive(obj, la, lb, tau):
    batch = la.shape[0]
    if obj in ('nce_mse', 'nce_cos'):
        points = torch.cat((la, lb)).reshape(2 * batch, -1)
        if obj == 'nce_mse':           # the original CoMIRTrainer.step expression, same operation order
            similarity = -torch.cdist(points, points).square() / points.shape[1] / tau
        else:
            unit = F.normalize(points, dim=1)
            similarity = unit @ unit.T / tau
        ids = torch.arange(2 * batch, device=la.device)
        positives = similarity[ids, (ids + batch) % (2 * batch)]
        negatives = similarity.masked_fill(torch.eye(2 * batch, device=la.device, dtype=torch.bool), -float('inf'))
        return (-positives + torch.logsumexp(negatives, 1)).mean(), {}
    a, b = la.reshape(batch, -1), lb.reshape(batch, -1)
    if obj.endswith('_exp'):
        T = 2 * torch.exp(-torch.cdist(a, b).square() / a.shape[1] / tau) - 1
    else:
        T = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T
    eye = torch.eye(batch, device=la.device, dtype=torch.bool)
    tp, tq = T[eye], T[~eye]
    info = dict(T_pos_mean=float(tp.detach().mean()), T_neg_mean=float(tq.detach().mean()))
    if obj.startswith('vcs'):
        j = (tp - 0.5 * tp.square()).mean() - (tq + 0.5 * tq.square()).mean()
        info['J'] = float(j.detach())
        return -j, info
    loss = (-torch.log(((1 + tp) / 2).clamp(1e-6, 1.0)).mean()
            - torch.log(((1 - tq) / 2).clamp(1e-6, 1.0)).mean())
    return loss, info


class ObjectiveMixin:
    """CoMIRTrainer.step with only the contrastive term replaced (copied verbatim otherwise)."""
    objective = 'nce_mse'

    def step(self):
        for model in self.modules():
            model.train()
        xa, xb = self.draw()
        batch = self.cfg.batch_size
        ka, kb = self.g.integers(4, size=batch), self.g.integers(4, size=batch)

        def rotate(values, rotations):
            return torch.stack([torch.rot90(values[i], int(rotations[i]), (1, 2))
                                for i in range(len(values))])

        la = rotate(self.moving_encoder(rotate(xa, ka)), -ka)
        lb = rotate(self.reference_encoder(rotate(xb, kb)), -kb)
        contrastive_value, info = contrastive(self.objective, la, lb, self.cfg.temperature)
        size = la.numel() + lb.numel()
        loss = (contrastive_value + self.cfg.regularization_l1 * (1 + la.abs().sum() + lb.abs().sum()) / size
                + self.cfg.regularization_l2 * (1 + la.square().sum() + lb.square().sum()) / size)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms = [float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.clip_norm))
                 for model in self.modules()]
        for optimizer in self.optimizers:
            optimizer.step()
        self.n_steps += 1
        return {"step": self.n_steps, "epoch": (self.n_steps - 1) // self.cfg.steps_per_epoch + 1,
                "loss": float(loss.detach()), "contrastive_loss": float(contrastive_value.detach()),
                "grad_norm_preclip": max(norms), "training_objective": self.objective, **info}


class BFTrainer(ObjectiveMixin, CoMIRTrainer):
    pass


def trainer_class(ds, H):
    if ds == 'bf':
        return BFTrainer
    return type('EPFLTrainer', (ObjectiveMixin, H.EPFLCoMIRTrainer), {})


def make(ds, c, seed, device, guard, H):
    # tau=None (cosine VCS/BCE) keeps the dataclass default; it is not read by those objectives.
    cfg = CoMIRConfig(seed=seed, learning_rate=c['lr'], **({'temperature': c['tau']} if c['tau'] is not None else {}))
    t = trainer_class(ds, H)(cfg, guard, device)
    t.objective = c['obj']
    return t


def context(ds):
    u = R.load_unit(str(UNIT_JSON).format(ds=ds))
    guard, H = R.make_guard(u)
    if H is not None:
        H.verify_official_source = _verify_official_source   # the EPFL host imported the name into its own namespace
    return u, guard, H


# ------------------------------------------------------------------------------------------------------------ evaluation
def evaluate(t, ds, guard, H, role, path):
    cores = sorted(guard.included(role))
    done = {(r['core'], r['perturbation']): r for r in R.rows(path)}
    with R.CP.evaluation_state(t):
        if ds == 'bf':
            session = RegistrationSession('CoMIR_native', t, guard, role, cores, t.device, {})
        else:
            session = H.EPFLRegistrationSession('CoMIR_native', t, guard, role, cores, t.device, {})
        for core in cores:
            for i in range(R.N_PERT):
                if (core, i) in done:
                    continue
                energy, prep = session.energy(core, i)
                rec = R.plain(A.registration_record(energy, t.device, session.perturbations[core][i]))
                if not all(math.isfinite(rec[k]) for k in ('err_full', 'F_hat', 'F_true')):
                    raise FloatingPointError('Non-finite registration record')
                rec.update(core=core, perturbation=i, role=role, scene=guard.unit_of(core) if H else core, **prep)
                R.append(path, rec); done[(core, i)] = rec
    recs = list(done.values())
    if len(recs) != len(cores) * R.N_PERT:
        raise RuntimeError('Incomplete evaluation block')
    return summarize(recs, guard if (H and role == 'DEV') else None)


def summarize(recs, guard=None):
    errs = [r['err_full'] for r in recs]
    scenes = defaultdict(list)
    for r in recs:
        scenes[r['scene']].append(r['err_full'])
    out = dict(n=len(errs), success42=sum(e < 42 for e in errs) / len(errs), success9=sum(e < 9 for e in errs) / len(errs),
               median_err=statistics.median(errs), p95_err=float(sorted(errs)[min(len(errs) - 1, int(math.ceil(0.95 * len(errs))) - 1)]),
               scene_macro_success42=statistics.fmean(sum(e < 42 for e in v) / len(v) for v in scenes.values()),
               scene_macro_success9=statistics.fmean(sum(e < 9 for e in v) / len(v) for v in scenes.values()), n_scenes=len(scenes))
    if guard is not None:
        from v10.epfl import production as PR
        out['DEV_statistics'] = PR.dev_statistics([{'window': r['core'], 'err_full': r['err_full']} for r in recs], guard.group_record)
    return out


def environment_check():
    env = A.environment('cuda')
    expected = json.loads((BASE / 'handoff/v10_bf1_20261017/PROTOCOL.json').read_text())['precision']
    for key in ('torch', 'cuda_runtime', 'cudnn', 'torch_threads'):
        if env[key] != expected['environment'][key]:
            raise RuntimeError('Runtime environment differs from the V10 protocol: ' + key)
    if A.precision_settings(env['precision']) != expected['settings'] or env['precision']['env'] != expected['env']:
        raise RuntimeError('Precision settings differ from the V10 protocol')
    return env


# ------------------------------------------------------------------------------------------------------------ units
def run_unit(ds, c, seed, phase, smoke=False):
    freeze = verify()
    gate = json.loads((HERE / 'CPU_CHECK.json').read_text())
    if gate['status'] != 'PASS' or gate['freeze_sha256'] != freeze:
        raise RuntimeError('CPU check is not PASS for this freeze')
    if not (torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')):
        raise RuntimeError('GPU Slurm job required')
    torch.set_num_threads(4)
    name = f"{ds}_{phase}_{cfg_name(c)}_s{seed}" + ('_smoke' if smoke else '')
    out = OUT / ('smoke' if smoke else 'runs') / name
    out.mkdir(parents=True, exist_ok=True)
    ident = dict(ds=ds, config=c, seed=seed, phase=phase, smoke=smoke, steps=STEPS if not smoke else 32, freeze_sha256=freeze)
    job = os.environ['SLURM_JOB_ID']; started = time.time()
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'COMPLETE.json').exists():
            raise RuntimeError('Unit already COMPLETE; no silent rerun')
        if list(out.glob('ATTEMPT_*.json')):
            raise RuntimeError('A previous attempt exists; review it before any rerun')
        env = environment_check()
        R.write(out / f'ATTEMPT_{job}.json', dict(status='RUNNING', identity=ident, start=started, environment=env))
        u, guard, H = context(ds)
        t = make(ds, c, seed, 'cuda', guard, H)
        initial = A.state_hash(*t.modules())
        steps = ident['steps']
        with (out / 'TRAIN.jsonl').open('a') as trace:
            while t.n_steps < steps:
                row = t.step()
                if not (math.isfinite(row['loss']) and math.isfinite(row['grad_norm_preclip'])):
                    raise FloatingPointError('Non-finite training step')
                trace.write(json.dumps(R.plain(row), allow_nan=False) + '\n')
        torch.cuda.synchronize(); train_seconds = time.time() - started
        meta = R.CP.save(t, out / 'checkpoints' / f'step_{steps:04d}.pt', ident)
        result = dict(status='COMPLETE', identity=ident, job=job, gpu=torch.cuda.get_device_name(), environment=env,
                      initial_model_sha256=initial, checkpoint=meta, train_seconds=train_seconds)
        if smoke:
            core = sorted(guard.included('CAL'))[0]
            with R.CP.evaluation_state(t):
                session = (RegistrationSession('CoMIR_native', t, guard, 'CAL', [core], t.device, {}) if ds == 'bf'
                           else H.EPFLRegistrationSession('CoMIR_native', t, guard, 'CAL', [core], t.device, {}))
                energy, _ = session.energy(core, 0)
                rec = R.plain(A.registration_record(energy, t.device, session.perturbations[core][0]))
            result['smoke_registration'] = {k: rec[k] for k in ('err_full', 'F_hat', 'F_true', 'seconds')}
            result['last_rows'] = [json.loads(l) for l in open(out / 'TRAIN.jsonl')][-3:]
        else:
            result['CAL'] = evaluate(t, ds, guard, H, 'CAL', out / 'CAL.jsonl')
            if phase == 'final':
                result['DEV'] = evaluate(t, ds, guard, H, 'DEV', out / 'DEV.jsonl')
        roles = {r['role'] for r in guard.reads}
        if not roles <= {'FIT', 'CAL', 'DEV'}:
            raise PermissionError('Unexpected data role read: ' + str(roles))
        result['roles_read'] = sorted(roles); result['seconds'] = time.time() - started
        R.write(out / 'COMPLETE.json', result)
        R.write(out / f'ATTEMPT_{job}.json', dict(status='COMPLETE', identity=ident, seconds=result['seconds']))
        print('R24_COMPLETE', name, json.dumps({k: result.get(k, {}).get('success9') if isinstance(result.get(k), dict) else None
                                                 for k in ('CAL', 'DEV')}), flush=True)


# ------------------------------------------------------------------------------------------------------------ CPU check
def cpu_check():
    freeze = verify(); torch.set_num_threads(4); torch.manual_seed(0); checks = []
    # 1. objective formulas vs dense loops (fp64) and gradcheck
    la = torch.randn(5, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    lb = torch.randn(5, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    a, b = la.reshape(5, -1), lb.reshape(5, -1)
    for obj in OBJECTIVES:
        tau = 0.5
        val, _ = contrastive(obj, la, lb, tau)
        if obj.startswith(('vcs', 'bce')):
            T = [[(2 * math.exp(-float(((a[i] - b[j]) ** 2).sum()) / a.shape[1] / tau) - 1) if obj.endswith('exp') else
                  float((a[i] @ b[j]) / (a[i].norm() * b[j].norm())) for j in range(5)] for i in range(5)]
            tp = [T[i][i] for i in range(5)]; tq = [T[i][j] for i in range(5) for j in range(5) if i != j]
            if obj.startswith('vcs'):
                ref = -(sum(x - x * x / 2 for x in tp) / 5 - sum(x + x * x / 2 for x in tq) / 20)
            else:
                ref = -sum(math.log((1 + x) / 2) for x in tp) / 5 - sum(math.log((1 - x) / 2) for x in tq) / 20
            assert abs(float(val) - ref) < 1e-10, (obj, float(val), ref)
        assert torch.autograd.gradcheck(lambda x, y: contrastive(obj, x, y, tau)[0], (la, lb)), obj
        checks.append(f'{obj}: dense reference + gradcheck')
    # VCS-exp bound and optimum sanity: T in (-1, 1]
    T = 2 * torch.exp(-torch.cdist(a, b).square() / a.shape[1] / 0.1) - 1
    assert float(T.max()) <= 1 and float(T.min()) > -1
    # 2. nce_mse step equals the original CoMIRTrainer.step bitwise (CPU, batch 4, two steps), real FIT data
    u, guard, H = context('bf')
    small = dict(obj='nce_mse', tau=0.5, lr=0.01)
    cfg = replace(CoMIRConfig(seed=1001), batch_size=4)
    orig = CoMIRTrainer(cfg, guard, 'cpu'); mine = BFTrainer(cfg, guard, 'cpu'); mine.objective = 'nce_mse'
    assert A.state_hash(*orig.modules()) == A.state_hash(*mine.modules()), 'initial models differ'
    for _ in range(2):
        # DenseUNet dropout draws from the global torch RNG: give both trainers the same state (one trainer per process in real runs)
        state = torch.get_rng_state()
        r1 = orig.step()
        torch.set_rng_state(state)
        r2 = mine.step()
        assert r1['loss'] == r2['loss'] and r1['grad_norm_preclip'] == r2['grad_norm_preclip'], (r1, r2)
    assert A.state_hash(*orig.modules()) == A.state_hash(*mine.modules()), 'nce_mse step is not bitwise original'
    checks.append('nce_mse arm: two real FIT steps bitwise equal to CoMIRTrainer.step (losses, grad norms, model hash)')
    # 3. every other arm: one real FIT step finite; same data stream as the original (sample RNG state equal after a step)
    for obj in OBJECTIVES[1:]:
        t = BFTrainer(replace(CoMIRConfig(seed=1001), batch_size=4), guard, 'cpu'); t.objective = obj
        ref = CoMIRTrainer(replace(CoMIRConfig(seed=1001), batch_size=4), guard, 'cpu')
        row = t.step(); ref.step()
        assert math.isfinite(row['loss']) and t.g.bit_generator.state == ref.g.bit_generator.state, obj
        checks.append(f'{obj}: finite real FIT step, sample stream identical to the original')
    # 4. registration path for a CoMIR trainer (one CAL problem, CPU) is the V10 CoMIR_native session
    with R.CP.evaluation_state(mine):
        core = sorted(guard.included('CAL'))[0]
        session = RegistrationSession('CoMIR_native', mine, guard, 'CAL', [core], mine.device, {})
        energy, _ = session.energy(core, 0)
        value = energy.value(*session.perturbations[core][0][:1], session.perturbations[core][0][1:])
        assert math.isfinite(value)
    checks.append('V10 CoMIR_native registration energy evaluates for the patched trainer')
    assert {r['role'] for r in guard.reads} <= {'FIT', 'CAL', 'DEV'}
    R.write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, checks=checks, utc=R.utc()))
    print('R24_CPU PASS', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('check', 'smoke', 'cal', 'final'))
    p.add_argument('--ds', choices=('bf', 'epfl'), default='bf')
    p.add_argument('--name'); p.add_argument('--seed', type=int)
    a = p.parse_args()
    if a.mode == 'check':
        cpu_check()
    elif a.mode == 'smoke':
        for c in [dict(obj=o, tau=(0.5 if o in ('nce_mse', 'nce_cos', 'vcs_exp', 'bce_exp') else None), lr=0.01) for o in OBJECTIVES]:
            run_unit(a.ds, c, CAL_SEED, 'cal', smoke=True)
    else:
        grid = {cfg_name(c): c for c in cal_grid()}
        c = grid[a.name]
        if a.mode == 'cal':
            run_unit(a.ds, c, CAL_SEED, 'cal')
        else:
            if a.seed not in FINAL_SEEDS:
                raise SystemExit('final seed must be one of ' + str(FINAL_SEEDS))
            run_unit(a.ds, c, a.seed, 'final')
