"""CE KD decision-coordinate aux with a bounded critic link T = tanh(a*cos + b): CPU check, stage-1 fixed-checkpoint diagnostic,
stage-2 full 240-epoch runs matched to the reused identity-VCS and cosine controls (same init/teacher/data order/GPU model)."""
import argparse, copy, hashlib, os, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'r22_kd_positive_control_v3_20261025'))
from common import BASE, OLD, read, write, sha  # noqa: E402  (v3 helpers; verify() below uses this package's FREEZE)
sys.path.insert(0, str(OLD / 'engine'))
import guard  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from kd import runner as K, decision as D  # noqa: E402
from kd.losses import DistillationLoss  # noqa: E402
sys.path.insert(0, str(HERE))
import link as LK  # noqa: E402
LK.install()
SEEDS = (31415, 16180, 57721)
WEIGHT = 10
CONTROLS = read(BASE / 'handoff/r22_kd_positive_control_v3_20261025/SOURCE_LOCK.json')['controls']
CKPTS = read(BASE / 'handoff/r22_mechanism_kd_repair_20261024/SOURCE_LOCK.json')['kd']


def verify():
    for rel, h in read(HERE / 'FREEZE.json')['files'].items(): assert sha(BASE / rel) == h, rel
    return sha(HERE / 'FREEZE.json')


def identity_j(s):
    """J with T = s on one n x n score matrix or a stack [B, n, n] (diagonal P, ordered off-diagonal Q, averaged over batches)."""
    s = s if s.ndim == 3 else s[None]; n = s.shape[-1]; mask = ~torch.eye(n, dtype=torch.bool, device=s.device)
    p, q = s.diagonal(dim1=-2, dim2=-1), s[:, mask]
    return ((p - .5 * p.square()).mean(-1) - (q + .5 * q.square()).mean(-1)).mean()


def stacked_terms(S, a, b, method):
    """Same objective as link.link_terms, on a stack of fixed score matrices S [B, 64, 64]; per-batch means then batch mean."""
    n = S.shape[-1]; mask = ~torch.eye(n, dtype=torch.bool, device=S.device); f = a * S + b
    fp, fq = f.diagonal(dim1=-2, dim2=-1), f[:, mask]
    if method == 'bce_link': return (F.softplus(-2 * fp).mean(-1) + F.softplus(2 * fq).mean(-1)).mean()
    tp, tq = torch.tanh(fp), torch.tanh(fq)
    return -((tp - .5 * tp.square()).mean(-1) - (tq + .5 * tq.square()).mean(-1)).mean()


def check():
    freeze = verify(); torch.set_num_threads(4); torch.manual_seed(25)
    s = torch.randn(9, 11, dtype=torch.float64, requires_grad=True); t = torch.randn_like(s, requires_grad=True)
    u, v = D.coordinates(s), D.coordinates(t.detach()); S = u @ v.T
    # identity map through the explicit n x n path equals the original exact-moment VCS
    assert torch.allclose(identity_j(S), D.quadratic_u(u, v), atol=1e-12, rtol=1e-12)
    # stacked objective equals the per-batch link objective
    LK.CURRENT['link'] = LK.Link(True).double()
    with torch.no_grad(): LK.CURRENT['link'].a.fill_(3.); LK.CURRENT['link'].b.fill_(-.2)
    for m in ('vcs_link', 'bce_link'):
        assert torch.allclose(stacked_terms(S.detach()[None], 3., -.2, m), LK.link_terms(u.detach(), v, LK.CURRENT['link'], m)[0].detach(), atol=1e-12, rtol=1e-12)
    for method in LK.METHODS:
        LK.CURRENT['link'] = LK.Link(method != 'vcs_tanh').double()
        val, diag = LK.decision_loss(s, t, method)
        gs, gt = torch.autograd.grad(val, (s, t), retain_graph=True, allow_unused=True); assert gt is None and torch.isfinite(gs).all()
        assert torch.autograd.gradcheck(lambda x: LK.decision_loss(x, t, method)[0], (s,))
        shifted, _ = LK.decision_loss(s + torch.randn(9, 1, dtype=torch.float64), t + torch.randn(9, 1, dtype=torch.float64), method)
        assert torch.allclose(val, shifted, atol=1e-12, rtol=1e-12)
        perm = torch.randperm(11); permuted, _ = LK.decision_loss(s[:, perm], t[:, perm], method)
        assert torch.allclose(val, permuted, atol=1e-12, rtol=1e-12)
        if method != 'vcs_tanh':
            ga = torch.autograd.grad(val, [LK.CURRENT['link'].a, LK.CURRENT['link'].b], retain_graph=True)
            assert all(torch.isfinite(g) for g in ga)
    # vcs_link at init equals vcs_tanh
    LK.CURRENT['link'] = LK.Link(True).double(); a, _ = LK.decision_loss(s, t, 'vcs_link')
    LK.CURRENT['link'] = LK.Link(False).double(); b, _ = LK.decision_loss(s, t, 'vcs_tanh'); assert torch.equal(a, b)
    # bce_link with f = atanh(s) reproduces the original bounded-score BCE ((1+T)/2 probability, T = s)
    class AtanhLink(torch.nn.Module):
        a = torch.tensor(1.); b = torch.tensor(0.)
        def forward(self, x): return torch.atanh(x.clamp(-1 + 1e-6, 1 - 1e-6))
    ours, _ = LK.link_terms(u, v, AtanhLink(), 'bce_link'); orig, _ = LK.ORIGINAL_DECISION(s, t, 'bce')
    assert torch.allclose(ours, orig, atol=1e-9, rtol=1e-9), (float(ours), float(orig))
    # original objectives unchanged
    for method in ('cosine', 'vcs', 'bce'):
        x, _ = LK.decision_loss(s, t, method); y, _ = LK.ORIGINAL_DECISION(s, t, method); assert torch.equal(x, y)
    # learnable link registers exactly two head parameters; fixed tanh registers none; anchor-only path untouched
    cfg = read(HERE / 'configs/kd_link_vcs_link_ce_s31415.json')['loss']
    m = DistillationLoss(11, 11, cfg); assert sorted(n for n, _ in m.named_parameters()) == ['decision_link.a', 'decision_link.b']
    cfg2 = read(HERE / 'configs/kd_link_vcs_tanh_ce_s31415.json')['loss']; assert not list(DistillationLoss(11, 11, cfg2).parameters())
    write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=freeze, checks=[
        'explicit n x n identity J equals exact-moment VCS', 'gradcheck/teacher detached/translation+class-permutation invariance for 3 link methods',
        'vcs_link at init equals vcs_tanh', 'bce_link with atanh link equals original bounded-score BCE', 'original cosine/vcs/bce unchanged',
        'learnable link = 2 head params; fixed tanh = 0 params']))
    print('KD_LINK_CPU PASS', flush=True)


def load_checkpoint(seed, method, device):
    src = CKPTS[f'ce_{method}_{seed}']; p = BASE / src['directory']; cfg = read(p / 'resolved_config.json'); rec = read(p / 'run.json')
    assert rec['status'] == 'complete' and rec['epochs_completed'] == 240 and not rec['test_loaded_during_train']
    assert sha(p / 'last.pt') == src['sha256']
    guard.install(guard.kd_roots(cfg)); manifest = K.read_manifest(cfg['data']['manifest'])
    ckpt = torch.load(p / 'last.pt', map_location='cpu', weights_only=True); assert ckpt['epoch'] == 239
    teacher, tdim, _ = K._teacher(cfg, manifest, device)
    model = K.create_model(cfg['student'], manifest['num_classes'], seed); sdim = K.feature_dim(model)
    model = model.to(device); model.load_state_dict(ckpt['student'])
    return src, cfg, manifest, ckpt, teacher, tdim, model, sdim


def fit_link(S, method, inits=((1., 0.), (5., 0.), (20., 0.))):
    """Fit (a, b) in fp64 on fixed score stacks: maximise mean J (vcs_link) or minimise balanced BCE (bce_link); best of 3 inits."""
    best = None
    for a0, b0 in inits:
        a = torch.tensor(a0, dtype=torch.float64, requires_grad=True); b = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([a, b], lr=1, max_iter=500, line_search_fn='strong_wolfe', tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            opt.zero_grad(); loss = stacked_terms(S, a, b, method); loss.backward(); return loss
        opt.step(closure)
        with torch.no_grad(): loss = float(stacked_terms(S, a, b, method))
        if best is None or loss < best[0]: best = (loss, float(a), float(b), (a0, b0))
    return dict(objective=best[0], a=best[1], b=best[2], init=best[3])


def t_stats(S, a, b):
    n = S.shape[-1]; mask = ~torch.eye(n, dtype=torch.bool); t = torch.tanh(a * S + b)
    tp, tq = t.diagonal(dim1=-2, dim2=-1).reshape(-1), t[:, mask].reshape(-1)
    pick = lambda x: x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:200000]]
    q = lambda x: [float(z) for z in torch.quantile(pick(x), torch.tensor([.05, .5, .95], dtype=x.dtype))]
    return dict(P_mean=float(tp.mean()), P_sd=float(tp.std()), P_q05_50_95=q(tp), Q_mean=float(tq.mean()), Q_sd=float(tq.std()), Q_q05_50_95=q(tq),
                P_abs_ge099=float((tp.abs() >= .99).double().mean()), Q_abs_ge099=float((tq.abs() >= .99).double().mean()),
                P_grad_factor_mean=float((1 - tp.square()).mean()), J=-float(stacked_terms(S, a, b, 'vcs_link')))


def margin(logits, labels):
    other = logits.clone(); other.scatter_(1, labels[:, None], -torch.inf)
    return logits.gather(1, labels[:, None])[:, 0] - other.max(1).values


def diag(device_name='cuda', max_batches=8):
    freeze = verify(); c = read(HERE / 'CPU_CHECK.json'); assert c['status'] == 'PASS' and c['freeze_sha256'] == freeze
    torch.set_num_threads(4); started = time.time(); device = torch.device(device_name); out = HERE / 'diag'
    assert not (out / 'COMPLETE.json').exists(); results = {}
    for seed in SEEDS:
        for source in ('cosine', 'vcs'):
            src, cfg, manifest, ckpt, teacher, tdim, model, sdim = load_checkpoint(seed, source, device)
            fit = K.RoleDataset(cfg['data']['root'], manifest, 'fit', augment=True, seed=seed)
            # 1) coordinates of all FIT samples, one augmented epoch-0 pass, eval-mode BN (no buffer writes)
            model.eval(); U, V, I = [], [], []
            with torch.no_grad():
                for images, labels, ids in K._loader(fit, cfg, True, 0):
                    images = images.to(device); ls, _ = model(images); lt, _ = teacher(images)
                    U.append(D.coordinates(ls).double().cpu()); V.append(D.coordinates(lt).double().cpu()); I.append(ids)
            U, V, I = torch.cat(U), torch.cat(V), torch.cat(I)
            halves = {}
            for h in (0, 1):
                idx = torch.nonzero(I % 2 == h)[:, 0]; idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed + h))]
                m = (len(idx) // 64) * 64; idx = idx[:m].reshape(-1, 64)
                halves[h] = U[idx] @ V[idx].transpose(-1, -2)  # [B, 64, 64] fixed score matrices
            fits = {f'{m}_{h}': fit_link(halves[h], m) for m in ('vcs_link', 'bce_link') for h in (0, 1)}
            heldout = dict(identity_J=float(identity_j(halves[1])), n_batches=[len(halves[0]), len(halves[1])],
                           tanh_fixed=t_stats(halves[1], 1., 0.),
                           vcs_link_fitA=t_stats(halves[1], fits['vcs_link_0']['a'], fits['vcs_link_0']['b']),
                           bce_link_fitA=t_stats(halves[1], fits['bce_link_0']['a'], fits['bce_link_0']['b']))
            ta = torch.tanh(fits['vcs_link_0']['a'] * halves[1] + fits['vcs_link_0']['b']); tb = torch.tanh(fits['bce_link_0']['a'] * halves[1] + fits['bce_link_0']['b'])
            heldout['vcs_vs_bce_link_mean_abs_T_diff'] = float((ta - tb).abs().mean())
            # 2) disposable real updates (historical momentum copied; BN buffers restored), equal lambda 10
            links = {'vcs_tanh': (1., 0.), 'vcs_link': (fits['vcs_link_0']['a'], fits['vcs_link_0']['b']), 'bce_link': (fits['bce_link_0']['a'], fits['bce_link_0']['b'])}
            lc = copy.deepcopy(cfg['loss']); lc.pop('decision', None); losses = DistillationLoss(sdim, tdim, lc).to(device)
            rows = []
            for batch, (images, labels, ids) in enumerate(K._loader(fit, cfg, True, 0)):
                if batch == max_batches: break
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad(): lt, ft = teacher(images)
                model.load_state_dict(ckpt['student']); model.eval()
                with torch.no_grad(): before = margin(model(images)[0], labels)
                subsets = {'all': torch.ones_like(labels, dtype=torch.bool), 'teacher_correct_student_wrong': (lt.argmax(1) == labels) & (before < 0),
                           'teacher_correct_student_margin_le1': (lt.argmax(1) == labels) & (before <= 1)}
                updates = {}
                for arm in ('anchor_only', 'cosine', 'vcs', 'vcs_tanh', 'vcs_link', 'bce_link'):
                    model.load_state_dict(ckpt['student']); model.train()
                    opt = torch.optim.SGD([{'params': list(model.parameters()), 'weight_decay': cfg['optimizer']['weight_decay']}], lr=cfg['optimizer']['lr'],
                                          momentum=cfg['optimizer'].get('momentum', .9), nesterov=False)
                    opt.load_state_dict(copy.deepcopy(ckpt['optimizer'])); opt.zero_grad(set_to_none=True)
                    s, f = model(images); a, _ = losses(s, lt, f, ft, labels, 239)
                    if arm == 'anchor_only': total = a
                    elif arm in ('cosine', 'vcs'): total = a + WEIGHT * LK.ORIGINAL_DECISION(s, lt, arm)[0]
                    else:
                        lk = LK.Link(False).to(device)
                        with torch.no_grad(): lk.a.fill_(links[arm][0]); lk.b.fill_(links[arm][1])
                        total = a + WEIGHT * LK.link_terms(D.coordinates(s), D.coordinates(lt.detach()), lk, 'bce_link' if arm == 'bce_link' else 'vcs_link')[0]
                    total.backward(); assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()); opt.step()
                    with torch.no_grad():
                        for n, b in model.named_buffers(): b.copy_(ckpt['student'][n].to(device))
                    model.eval()
                    with torch.no_grad(): delta = margin(model(images)[0], labels) - before
                    updates[arm] = {k: dict(n=int(m.sum()), mean_margin_delta=float(delta[m].mean()) if m.any() else None) for k, m in subsets.items()}
                rows.append(dict(batch=batch, updates=updates))
            results[f'{source}_{seed}'] = dict(source=src, fits=fits, heldout=heldout, disposable_updates=rows, links_used=links)
            print('KD_LINK_DIAG', source, seed, {k: (round(v['a'], 3), round(v['b'], 3)) for k, v in fits.items()}, flush=True)
            del model, teacher, losses, ckpt; torch.cuda.empty_cache()
    # stage-2 technical gate (pre-registered in PROTOCOL.json): non-degenerate, stable, non-saturated fitted VCS link on every checkpoint
    gate = {}
    for k, r in results.items():
        fa, fb = r['fits']['vcs_link_0'], r['fits']['vcs_link_1']; h = r['heldout']['vcs_link_fitA']
        gate[k] = dict(a_in_range=all(.1 < x['a'] < 100 for x in (fa, fb)), a_stable=abs(fa['a'] - fb['a']) <= .25 * max(abs(fa['a']), abs(fb['a'])),
                       b_stable=abs(fa['b'] - fb['b']) <= .25, grad_factor_ok=h['P_grad_factor_mean'] >= .05, saturation_ok=h['P_abs_ge099'] <= .5)
    passed = all(all(v.values()) for v in gate.values())
    status = guard.status(); assert not status.get('forbidden_hits', []), status
    write(out / 'COMPLETE.json', dict(status='COMPLETE', freeze_sha256=freeze, job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - started,
          gpu=torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu', results=results, stage2_gate=gate, stage2_gate_pass=passed, guard=status,
          test_read=False, note='Fixed epoch-240 CE checkpoints trained with T=cos; FIT halves split by sample-id parity (fit on even, evaluate on odd). '
                                'Margin deltas are descriptive local evidence, not a task-gain criterion.'))
    print('KD_LINK_DIAG COMPLETE gate_pass=', passed, flush=True)


def execute(name):
    freeze = verify(); cpu = read(HERE / 'CPU_CHECK.json'); assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    d = read(HERE / 'diag/COMPLETE.json'); assert d['stage2_gate_pass'] and d['freeze_sha256'] == freeze
    config = HERE / 'configs' / f'{name}.json'; cfg = K.read_config(config); out = HERE / 'runs' / name
    assert not out.exists(), 'No implicit resume/retry or duplicate successful run'
    guard.install(guard.kd_roots(cfg)); start = time.time(); out.mkdir(parents=True)
    write(out / 'ATTEMPT.json', dict(status='RUNNING', job=os.environ.get('SLURM_JOB_ID'), started=start, freeze_sha256=freeze))
    try:
        manifest = K.read_manifest(cfg['data']['manifest']); K._device(cfg); seed = cfg['seed']
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        probe = K.create_model(cfg['student'], manifest['num_classes'], seed); K.feature_dim(probe); probe_sha = K.state_digest(probe); del probe
        refs = CONTROLS[f'ce_{seed}']
        for method in ('cosine', 'vcs'):
            c = read(refs[method]['receipt']); cr = read(Path(refs[method]['receipt']).parent / 'run.json')
            assert probe_sha == c['student_initial_sha256'], 'Initial state mismatch before training'
            assert torch.cuda.get_device_name() == c['gpu'] and K.environment() == cr['environment']
        write(out / 'PREFLIGHT.json', dict(status='PASS', initial_sha256=probe_sha, gpu=torch.cuda.get_device_name(), environment=K.environment()))
        r = K.train(config, out); assert r['status'] == 'complete' and r['teacher_frozen_verified'] and not r['test_loaded_during_train']
        hist = read(out / 'history.json'); assert len(hist) == cfg['training']['epochs']
        for method in ('cosine', 'vcs'):
            cp = Path(refs[method]['receipt']); c = read(cp); ch = read(cp.parent / 'history.json'); cr = read(cp.parent / 'run.json')
            assert r['student_initial_state_sha256'] == c['student_initial_sha256'] and r['teacher_initial_state_sha256'] == cr['teacher_initial_state_sha256']
            assert r['environment'] == cr['environment'] and torch.cuda.get_device_name() == c['gpu']
            for x, y in zip(hist, ch):
                assert x['sample_order_sha256'] == y['sample_order_sha256'] and x['fit_n_seen'] == y['fit_n_seen'] == 44992 and x['next_lr'] == y['next_lr']
        write(out / 'COMPLETE.json', dict(status='COMPLETE', freeze_sha256=freeze, seed=seed, anchor='ce', method=cfg['loss']['decision']['method'],
              epochs=len(hist), job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - start, config_sha256=sha(config),
              student_initial_sha256=r['student_initial_state_sha256'], gpu=torch.cuda.get_device_name(), environment=K.environment(),
              history_sha256=sha(out / 'history.json'), last_checkpoint_sha256=sha(out / 'last.pt'), final_top1=hist[-1]['cal']['top1'],
              final_link=hist[-1]['dependence'], guard=guard.status(), test_read=False, label='CE KD bounded-link arm; development CAL'))
        write(out / 'ATTEMPT.json', dict(status='COMPLETE', job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - start, freeze_sha256=freeze))
    except BaseException as e:
        write(out / 'ATTEMPT.json', dict(status='FAILED', job=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - start, error=repr(e),
                                         traceback=traceback.format_exc(), freeze_sha256=freeze)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'diag', 'run']); p.add_argument('--name'); a = p.parse_args()
    if a.mode == 'check': check()
    elif a.mode == 'diag': diag()
    else: execute(a.name)
