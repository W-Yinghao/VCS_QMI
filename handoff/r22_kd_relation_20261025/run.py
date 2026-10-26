"""DKD + relation term (direct / VCS / BCE): CPU check, 1-epoch technical smoke, full 240-epoch runs matched to the reused
DKD-only and DKD+logit-cosine controls (same student init, teacher, data order, GPU model and software/precision)."""
import argparse, copy, json, os, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OLD = BASE / 'handoff/r22_other_apps_20261023'
sys.path.insert(0, str(OLD / 'engine'))
import guard  # noqa: E402
import torch  # noqa: E402
from kd import runner as K, decision as D  # noqa: E402
from kd.losses import DistillationLoss  # noqa: E402
sys.path.insert(0, str(HERE))
import relation as RL  # noqa: E402
import hashlib  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def write(p, d):
    p = Path(p); tmp = Path(str(p) + f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + '\n'); tmp.replace(p)
RL.install()
SEEDS = (31415, 16180, 57721)
CONTROLS = {s: {'dkd': OLD / f'runs/kd_full_dkd_none_l0_s{s}/COMPLETE.json', 'dkd_cosine': OLD / f'runs/kd_full_dkd_cosine_l10_s{s}/COMPLETE.json'} for s in SEEDS}


def read(p): return json.loads(Path(p).read_text())


def verify():
    for rel, h in read(HERE / 'FREEZE.json')['files'].items(): assert sha(BASE / rel) == h, rel
    return sha(HERE / 'FREEZE.json')


def check():
    freeze = verify(); torch.set_num_threads(4); torch.manual_seed(5)
    u0, v0 = torch.randn(9, 13, dtype=torch.float64, requires_grad=True), torch.randn(9, 13, dtype=torch.float64)
    u, v = RL.coordinates(u0), RL.coordinates(v0)
    for m in RL.METHODS:
        val = RL.relation_loss(u, v, m); g = torch.autograd.grad(val, u0, retain_graph=True)[0]; assert torch.isfinite(val) and torch.isfinite(g).all()
        assert torch.autograd.gradcheck(lambda x: RL.relation_loss(RL.coordinates(x), v, m), (u0,))
    # relation 'vcs'/'bce' are the original decision objectives applied to relation coordinates
    S = u @ v.T; n = 9; mask = ~torch.eye(n, dtype=torch.bool); p = S.diagonal()
    dense = (p - .5 * p.square()).mean() - (S[mask] + .5 * S[mask].square()).mean()
    assert torch.allclose(-RL.relation_loss(u, v, 'vcs'), dense, atol=1e-12)
    cfg = read(HERE / 'configs/kd_rel_vcs_dkd_s31415.json'); guard.install(guard.kd_roots(cfg)); manifest = K.read_manifest(cfg['data']['manifest'])
    ds = K.RoleDataset(cfg['data']['root'], manifest, 'fit', augment=False, seed=0); items = RL.reference_items(ds)
    ids = [ds.ids[i] for i in items]; labels = sorted(int(ds.labels[i]) for i in ids)
    assert len(items) == 200 and labels == sorted(list(range(100)) * 2)
    # same reference set as the headroom check
    assert hashlib.sha256(torch.tensor(sorted(ids)).numpy().tobytes()).hexdigest() == read(HERE / 'COMPLETE.json')['anchor_ids_sha256']
    # one real FIT CPU step through the patched loss; anchor-only path unchanged when relation is off
    model = K.create_model(cfg['student'], manifest['num_classes'], cfg['seed']); dim = K.feature_dim(model)
    teacher, tdim, _ = RL.teacher(cfg, manifest, torch.device('cpu'))
    RL.STATE['student'] = model
    fit = K.RoleDataset(cfg['data']['root'], manifest, 'fit', augment=True, seed=cfg['seed'])
    images, labels_b, _ = next(iter(K._loader(fit, cfg, True, 0))); model.train()
    with torch.no_grad(): lt, ft = teacher(images)
    ls, fs = model(images)
    base_cfg = copy.deepcopy(cfg['loss']); base_cfg.pop('relation')
    a, _ = DistillationLoss(dim, tdim, base_cfg)(ls, lt, fs, ft, labels_b, 30)
    b, _ = RL.ORIGINAL_FORWARD(DistillationLoss(dim, tdim, base_cfg), ls, lt, fs, ft, labels_b, 30); assert torch.equal(a, b)
    for m in RL.METHODS:
        c = copy.deepcopy(cfg['loss']); c['relation']['method'] = m
        tot, parts = DistillationLoss(dim, tdim, c)(ls, lt, fs, ft, labels_b, 30)
        assert torch.isfinite(tot) and 'relation' in parts and float(tot) != float(a)
        g = torch.autograd.grad(tot, list(model.parameters()), retain_graph=True, allow_unused=True)
        assert all(x is None or torch.isfinite(x).all() for x in g) and all(p.grad is None for p in teacher.parameters())
    assert model.training
    write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, checks=[
        'gradcheck direct/vcs/bce relation losses', 'relation vcs equals dense J on relation coordinates',
        '200 references = 2 per class, identical id set to the headroom check', 'anchor path bitwise unchanged when relation off',
        'one real FIT CPU step per method finite, teacher stop-gradient, student back in train mode']))
    print('KD_REL_CPU PASS', flush=True)


def execute(name, technical=False):
    freeze = verify(); cpu = read(HERE / 'CPU_CHECK.json'); assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    if not technical:
        sm = read(HERE / 'smoke/COMPLETE.json'); assert sm['status'] == 'COMPLETE' and sm['freeze_sha256'] == freeze
    config = HERE / 'configs' / f'{name}.json'; cfg = K.read_config(config); out = HERE / ('smoke' if technical else 'runs/' + name)
    assert not (out / 'COMPLETE.json').exists() and not (out / 'ATTEMPT.json').exists(), 'No implicit resume/retry or duplicate run'
    guard.install(guard.kd_roots(cfg)); start = time.time(); out.mkdir(parents=True, exist_ok=True)
    write(out / 'ATTEMPT.json', dict(status='RUNNING', job=os.environ.get('SLURM_JOB_ID'), started=start, freeze_sha256=freeze))
    try:
        manifest = K.read_manifest(cfg['data']['manifest']); K._device(cfg); seed = cfg['seed']
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        probe = RL.ORIGINAL_CREATE(cfg['student'], manifest['num_classes'], seed); K.feature_dim(probe); probe_sha = K.state_digest(probe); del probe
        for kind, rp in CONTROLS[seed].items():
            c = read(rp); cr = read(Path(rp).parent / 'run.json')
            assert probe_sha == c['student_initial_sha256'], 'Initial state mismatch before training'
            assert torch.cuda.get_device_name() == c['gpu'] and K.environment() == cr['environment'], kind
        write(out / 'PREFLIGHT.json', dict(status='PASS', initial_sha256=probe_sha, gpu=torch.cuda.get_device_name(), environment=K.environment()))
        r = K.train(config, out); assert r['status'] == 'complete' and r['teacher_frozen_verified'] and not r['test_loaded_during_train']
        hist = read(out / 'history.json'); assert len(hist) == cfg['training']['epochs']
        for kind, rp in CONTROLS[seed].items():
            c = read(rp); ch = read(Path(rp).parent / 'history.json'); cr = read(Path(rp).parent / 'run.json')
            assert r['student_initial_state_sha256'] == c['student_initial_sha256'] and r['teacher_initial_state_sha256'] == cr['teacher_initial_state_sha256']
            assert r['environment'] == cr['environment'] and torch.cuda.get_device_name() == c['gpu']
            for x, y in zip(hist, ch):
                assert x['sample_order_sha256'] == y['sample_order_sha256'] and x['fit_n_seen'] == y['fit_n_seen'] == 44992 and x['next_lr'] == y['next_lr']
        write(out / 'COMPLETE.json', dict(status='COMPLETE', freeze_sha256=freeze, technical=technical, seed=seed, anchor='dkd',
              relation=cfg['loss']['relation'], epochs=len(hist), job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - start,
              config_sha256=sha(config), student_initial_sha256=r['student_initial_state_sha256'], gpu=torch.cuda.get_device_name(),
              environment=K.environment(), history_sha256=sha(out / 'history.json'), last_checkpoint_sha256=sha(out / 'last.pt'),
              final_top1=hist[-1]['cal']['top1'], guard=guard.status(), test_read=False, label='DKD + relation term; development CAL'))
        write(out / 'ATTEMPT.json', dict(status='COMPLETE', job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - start, freeze_sha256=freeze))
    except BaseException as e:
        write(out / 'ATTEMPT.json', dict(status='FAILED', job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - start, error=repr(e),
                                         traceback=traceback.format_exc(), freeze_sha256=freeze)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'smoke', 'run']); p.add_argument('--name'); a = p.parse_args()
    if a.mode == 'check': check()
    elif a.mode == 'smoke': execute('kd_rel_smoke_vcs_dkd_s31415', technical=True)
    else: execute(a.name)
