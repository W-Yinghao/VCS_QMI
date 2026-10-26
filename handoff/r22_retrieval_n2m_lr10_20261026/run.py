"""N2M hinge host (SGR, sum-scale weighted max-violation hinge, 40 epochs) with lr_update 30 -> 10, everything else unchanged.

Modes: check (CPU, Slurm), gate (one RTX job, seed 901, both arms, sparse real updates), run --seed --arm (full 40 epochs).
The frozen N2M trainer is used as is; process-local wrappers only observe (initial state hash, applied Adam LR, per-epoch
comparison with the reused lr_update-30 N2M run). Data order/weights do not depend on the LR, so batch_mass/batch_ess at every
logged step must equal the control in all 40 epochs; epochs 0-9 have identical schedules, so their losses/DEV are compared too.
"""
import argparse
import itertools
import math
import os
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from common import (HERE, OUT, SEEDS, ARMS, PERIOD, UPDATES_PER_EPOCH, EPOCHS, read, write, sha, verify, lr_vector, trace_module,
                    n2m, runtime_src, experiment, training_inputs, control, state_sha256, manifest_matches)

LOG_EVERY = 20


def setup():
    finish = trace_module().install('c1')          # forbids TEST, eval_only, noise manifests/indices; DEV allowed
    M = n2m(); M['runtime'].configure_precision()   # deterministic algorithms, no TF32, threads = SLURM_CPUS_PER_TASK (as N2M)
    return finish, M


def config(cfg, period):
    assert cfg.lr_update == 30 and cfg.num_epochs == EPOCHS
    new = replace(cfg, lr_update=period)
    assert {k for k, v in asdict(cfg).items() if asdict(new)[k] != v} == ({'lr_update'} if period != 30 else set())
    return new


def replay_order(train, REF, v, seed, batch_size):
    """Logged-step batch mass/ESS of the trainer's own permutation stream (independent of the LR)."""
    out, step = [], 0
    for epoch in range(EPOCHS):
        for k, idx in enumerate(train.iterate_batches(len(v), batch_size, train.np.random.default_rng([seed, 9_020, epoch])), 1):
            step += 1
            if step % LOG_EVERY == 0:
                vb = v[idx]; out.append(dict(step=step, epoch=epoch, step_in_epoch=k, batch_mass=float(vb.sum()), batch_ess=REF.effective_sample_size(vb)))
        assert k == UPDATES_PER_EPOCH
    return out


def order_equal(mine, ref_steps):
    ref = {s['step']: s for s in ref_steps}; assert len(ref) == len(ref_steps)
    return all(all(m[k] == ref[m['step']][k] for k in ('epoch', 'step_in_epoch', 'batch_mass', 'batch_ess')) for m in mine) and len(mine) == len(ref)


def rel(a, b): return abs(a - b) / max(abs(b), 1e-12)


def check():
    f = verify(); finish, M = setup(); started = time.time()
    exp = experiment(); runtime_src(exp)
    import torch
    import v9.n2.train as train
    from v9.n2.data import load_vocab
    from v9.n2.teacher import load_host_lock
    from v9.n2.train import WeightedStudent
    from v9.n2.weights import REF
    cfg = load_host_lock(Path(exp['runtime_dir']) / 'HOST_LOCK.yaml', 'f30k_precomp', len(load_vocab(exp['inputs']['vocab'])))
    M['runner'].check_config(cfg); new = config(cfg, PERIOD); keys = M['runtime'].ENVIRONMENT_KEYS
    envs, units = set(), {}
    for seed in SEEDS:
        for arm in ARMS + ('ShuffleVCS',):
            d, r, log, man = control(seed, arm)
            assert sha(d / 'best.pt') == r['artifact_sha256']['best.pt']
            assert [e['epoch'] for e in log['epochs']] == list(range(EPOCHS)) and all(e['steps_in_epoch'] == UPDATES_PER_EPOCH for e in log['epochs'])
            assert [e['lr'] for e in log['epochs']] == lr_vector(30) and r['done']['global_step'] == 45320
            assert r['attempts_summary']['attempts'] == 1 and r['hardware']['gpu'].startswith('NVIDIA RTX PRO 6000 Blackwell')
            envs.add(tuple((k, r['hardware'][k]) for k in keys))
            if arm == 'ShuffleVCS': continue
            u, rows, weights, lock = training_inputs(exp, seed, arm)
            st = WeightedStudent(new, rows, weights, None, None, None, HERE / 'checks' / f'manifest_{arm}_s{seed}', seed, 1.0, device='cpu',
                                 weight_lock=lock, formal=False, loss_scale='sum', grad_log_window=(200, 200))
            assert manifest_matches(M, man, st.manifest, PERIOD), (seed, arm)
            mine = replay_order(train, REF, st.v, seed, new.batch_size)
            assert order_equal(mine, log['steps']), ('data order / weights differ from the control', seed, arm)
            units[f'{arm}_{seed}'] = dict(control_host=r['hardware']['host'], logged_steps_equal=len(mine), weights_sha256=u['weights_sha256'],
                                          control_best=r['done']['best'])
    assert len(envs) == 1, 'controls do not share one environment'
    # Exact production fit code on a toy host: the LR applied at every actual Adam.step for both periods.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w = torch.nn.Parameter(torch.randn(4, 4) * .01)
        def forward(self, ims, caps, lens): return self.w
    class Student(WeightedStudent):
        def validate(self, host): return {'rsum': 0.0}
    original_build, original_step = train.build_host, torch.optim.Adam.step; schedules = {}
    try:
        train.build_host = lambda c: Toy()
        for period in (30, PERIOD):
            applied = []
            def step(opt, *a, **kw):
                applied.append([g['lr'] for g in opt.param_groups]); return original_step(opt, *a, **kw)
            torch.optim.Adam.step = step
            toy = Student(replace(cfg, batch_size=4, lr_update=period), [{'pair_id': str(i), 'role': 'U'} for i in range(4)], {str(i): .5 for i in range(4)},
                          lambda rows: (torch.zeros(4, 1), torch.zeros(4, 1), [1] * 4), None, None,
                          HERE / 'checks' / f'toy_period{period}_{os.environ["SLURM_JOB_ID"]}', 901, 1.0, loss_scale='sum')
            toy.fit(EPOCHS, log_every=1); log = read(toy.out / 'train_log.json')
            assert applied == [[x] for x in lr_vector(period)] and [x['lr'] for x in log['epochs']] == lr_vector(period)
            assert log['epochs'][4]['phase'] == 'warmup_mean_hinge' and log['epochs'][5]['phase'] == 'max_violation'
            schedules[str(period)] = [x[0] for x in applied]
    finally:
        train.build_host, torch.optim.Adam.step = original_build, original_step
    trace = finish(HERE / 'checks/CPU_TRACE.json'); assert trace['status'] == 'PASS', trace['forbidden_hits']
    write(HERE / 'CPU_CHECK.json', dict(status='PASS', freeze_sha256=f, job=os.environ['SLURM_JOB_ID'], seconds=time.time() - started,
          shared_control_environment=dict(next(iter(envs))), units=units, actual_adam_lr=schedules, trace_sha256=sha(HERE / 'checks/CPU_TRACE.json'),
          verified=['N2M experiment receipt, package code and runtime file identity', '9 control receipts (train_log/run_manifest/identity/best.pt hashes, 40x1133 updates, lr_update-30 vector, one attempt, one shared RTX environment)',
                    '6 training inputs (observed pool, control weights + lock, vocab, token cache, feature size/mtime); noise manifests not opened',
                    'config differs from N2M only in lr_update', 'run manifest compatible with the control except lr_update',
                    'trainer permutation stream reproduces batch_mass/batch_ess exactly at all 2266 logged control steps (all 40 epochs)',
                    'actual Adam LR per update for periods 30 and 10 through the unmodified fit loop (toy host)']))
    print('N2MLR10_CPU PASS', flush=True)


def gate():
    """Seed 901, both arms: real host, real data, 60 updates in epoch 0 then 1 update in each later epoch (99 per arm), DEV 16 images."""
    f = verify(); cpu = read(HERE / 'CPU_CHECK.json'); assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == f
    finish, M = setup(); started = time.time(); import torch
    exp = experiment(); seed = 901
    cfg, _, dev_images, dev_captions, collate = M['runner'].load_training_data(exp); new = config(cfg, PERIOD)
    import v9.n2.train as train
    from v9.n2.train import WeightedStudent
    hw = M['runtime'].collect_environment(); rows_out, inits = [], {}
    for arm in ARMS:
        d, r, clog, man = control(seed, arm)
        diffs = M['runtime'].environment_differences(r['hardware'], hw); assert not diffs, diffs
        u, rows, weights, lock = training_inputs(exp, seed, arm)
        out = OUT / 'gates' / arm; assert not out.exists(), 'No hidden gate restart'
        st = WeightedStudent(new, rows, weights, collate, dev_images[:16], dev_captions[:80], out, seed, 1.0, device='cuda',
                             weight_lock=lock, formal=False, loss_scale='sum', grad_log_window=None)
        assert manifest_matches(M, man, st.manifest, PERIOD)
        it, stp, build = train.iterate_batches, torch.optim.Adam.step, train.build_host; calls = [0]; applied = []; init = {}
        def limited(*a, **kw):
            e = calls[0]; calls[0] += 1; return itertools.islice(it(*a, **kw), 60 if e == 0 else 1)
        def step(opt, *a, **kw):
            applied.append([g['lr'] for g in opt.param_groups]); return stp(opt, *a, **kw)
        def built(c):
            m = build(c); init['sha256'] = state_sha256(m); return m
        try:
            train.iterate_batches, torch.optim.Adam.step, train.build_host = limited, step, built
            st.fit(EPOCHS, log_every=LOG_EVERY)
        finally:
            train.iterate_batches, torch.optim.Adam.step, train.build_host = it, stp, build
        log = read(out / 'train_log.json'); per = [60] + [1] * (EPOCHS - 1)
        assert calls[0] == EPOCHS and [e['steps_in_epoch'] for e in log['epochs']] == per
        assert [e['lr'] for e in log['epochs']] == lr_vector(PERIOD)
        assert applied == [[lr] for lr, n in zip(lr_vector(PERIOD), per) for _ in range(n)]
        assert all(math.isfinite(s['loss']) for s in log['steps']) and all(math.isfinite(e['mean_loss']) for e in log['epochs'])
        ref = {s['step']: s for s in clog['steps']}; cmp = []
        for s in log['steps']:
            if s['epoch'] != 0: continue
            c = ref[s['step']]
            assert (s['epoch'], s['step_in_epoch'], s['batch_mass'], s['batch_ess'], s['lr']) == (c['epoch'], c['step_in_epoch'], c['batch_mass'], c['batch_ess'], c['lr'])
            cmp.append(dict(step=s['step'], loss=s['loss'], control_loss=c['loss'], loss_rel_diff=rel(s['loss'], c['loss']),
                            u_risk_rel_diff=rel(s['u_risk_unnormalised'], c['u_risk_unnormalised']), exact=s['loss'] == c['loss'] and s['u_risk_unnormalised'] == c['u_risk_unnormalised']))
        assert [x['step'] for x in cmp] == [20, 40, 60] and max(x['loss_rel_diff'] for x in cmp) <= 1e-3, cmp
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['epoch'] == EPOCHS - 1 and all(g['lr'] == lr_vector(PERIOD)[-1] for g in ck['opt']['param_groups']) and len(ck['opt']['param_groups']) == 1
        h = __import__('hashlib').sha256()
        for k, v in sorted(ck['model'].items()): h.update(k.encode()); h.update(v.contiguous().numpy().tobytes())
        assert h.hexdigest() != init['sha256'], 'parameters did not change'
        inits[arm] = init['sha256']; del ck
        rows_out.append(dict(arm=arm, updates=sum(per), actual_adam_lr=[x[0] for x in applied], initial_model_sha256=init['sha256'], epoch0_vs_control=cmp,
                             control_host=r['hardware']['host'], log_sha256=sha(out / 'train_log.json')))
        print('GATE_ARM_PASS', arm, cmp, flush=True)
    assert inits['VCS'] == inits['ConstantU'], 'paired arms must share the seeded initial state'
    trace = finish(HERE / 'checks/GPU_TRACE.json'); assert trace['status'] == 'PASS', trace['forbidden_hits']
    write(HERE / 'GATE.json', dict(status='PASS', freeze_sha256=f, job=os.environ['SLURM_JOB_ID'], seconds=time.time() - started, host=hw['host'],
          environment=hw, rows=rows_out, trace_sha256=sha(HERE / 'checks/GPU_TRACE.json'),
          scope='Technical only: 2 arms x 99 real SGR updates on the full pool (epoch 0: 60, later epochs: 1), DEV subset 16 images/80 captions; not performance evidence.'))
    print('N2MLR10_GATE PASS', flush=True)


def run(seed, arm):
    f = verify(); cpu = read(HERE / 'CPU_CHECK.json'); g = read(HERE / 'GATE.json')
    assert cpu['status'] == 'PASS' and g['status'] == 'PASS' and cpu['freeze_sha256'] == g['freeze_sha256'] == f
    job = os.environ['SLURM_JOB_ID']; out = OUT / 'runs' / f'{arm}_s{seed}'; assert not out.exists(), 'No duplicate run or silent resume'
    finish, M = setup(); started = time.time(); import torch
    exp = experiment(); d, r, clog, man = control(seed, arm)
    hw = M['runtime'].collect_environment(); diffs = M['runtime'].environment_differences(r['hardware'], hw); assert not diffs, diffs
    out.mkdir(parents=True); attempt = out / f'ATTEMPT_{job}.json'
    write(attempt, dict(status='RUNNING', seed=seed, arm=arm, lr_update=PERIOD, job=job, host=hw['host'], start=started, freeze_sha256=f))
    try:
        u, rows, weights, lock = training_inputs(exp, seed, arm)
        cfg, _, dev_images, dev_captions, collate = M['runner'].load_training_data(exp); new = config(cfg, PERIOD)
        import v9.n2.train as train
        from v9.n2.train import WeightedStudent
        ref_steps = {}
        for s in clog['steps']: ref_steps.setdefault(s['epoch'], []).append(s)
        epochs_checked = []

        def epoch_check(epoch, log):
            mine = [s for s in log['steps'] if s['epoch'] == epoch]; ref = ref_steps[epoch]
            assert [(s['step'], s['step_in_epoch']) for s in mine] == [(s['step'], s['step_in_epoch']) for s in ref]
            assert all(s['lr'] == lr_vector(PERIOD)[epoch] for s in mine)
            assert all(rel(a['batch_mass'], b['batch_mass']) <= 1e-12 and rel(a['batch_ess'], b['batch_ess']) <= 1e-12 for a, b in zip(mine, ref)), 'data order'
            rec = dict(epoch=epoch, lr=lr_vector(PERIOD)[epoch], order_exact=all(a['batch_mass'] == b['batch_mass'] and a['batch_ess'] == b['batch_ess'] for a, b in zip(mine, ref)),
                       dev_rsum=log['epochs'][-1]['dev']['rsum'], control_dev_rsum=clog['epochs'][epoch]['dev']['rsum'])
            if epoch < PERIOD:        # identical schedule before the first divergence: compare the trajectory
                lr = [rel(a['loss'], b['loss']) for a, b in zip(mine, ref)]
                rec.update(max_step_loss_rel_diff=max(lr), steps_exact=all(a['loss'] == b['loss'] for a, b in zip(mine, ref)),
                           mean_loss_diff=log['epochs'][-1]['mean_loss'] - clog['epochs'][epoch]['mean_loss'],
                           dev_rsum_diff=rec['dev_rsum'] - rec['control_dev_rsum'])
                if epoch == 0: assert rec['max_step_loss_rel_diff'] <= 1e-3, ('epoch 0 trajectory differs from the matched control', rec)
            epochs_checked.append(rec); print('EPOCH', epoch, rec, flush=True)

        class Matched(WeightedStudent):
            def _save(self, host, opt, epoch, best, log, name='last.pt'):
                if name == 'last.pt': epoch_check(epoch, log)
                return super()._save(host, opt, epoch, best, log, name)
        build, init = train.build_host, {}
        def built(c):
            m = build(c); init['sha256'] = state_sha256(m); return m
        st = Matched(new, rows, weights, collate, dev_images, dev_captions, out, seed, 1.0, device='cuda', weight_lock=lock,
                     formal=False, loss_scale='sum', grad_log_window=(200, 200))
        assert manifest_matches(M, man, st.manifest, PERIOD)
        try:
            train.build_host = built; done = st.fit(EPOCHS, log_every=LOG_EVERY)
        finally:
            train.build_host = build
        log = read(out / 'train_log.json')
        assert [e['epoch'] for e in log['epochs']] == list(range(EPOCHS)) and all(e['steps_in_epoch'] == UPDATES_PER_EPOCH for e in log['epochs'])
        assert done['global_step'] == 45320 and [e['lr'] for e in log['epochs']] == lr_vector(PERIOD) and len(epochs_checked) == EPOCHS
        gref = [__import__('json').loads(x) for x in open(d / 'grad_norms.jsonl')] if (d / 'grad_norms.jsonl').exists() else []
        gmine = [__import__('json').loads(x) for x in open(out / 'grad_norms.jsonl')]
        grad_window = dict(n=len(gmine), control_n=len(gref), exact=gmine == gref,
                           max_loss_rel_diff=max((rel(a['loss'], b['loss']) for a, b in zip(gmine, gref)), default=None))
        trace = finish(out / f'OPEN_TRACE_{job}.json'); assert trace['status'] == 'PASS', trace['forbidden_hits']
        result = dict(status='COMPLETE', seed=seed, arm=arm, lr_update=PERIOD, lr_vector=lr_vector(PERIOD), freeze_sha256=f, job=job, host=hw['host'],
                      environment=hw, control=dict(dir=str(d), full_result_sha256=sha(d / 'full_result.json'), host=r['hardware']['host'], best=r['done']['best']),
                      initial_model_sha256=init['sha256'], done=done, epochs=epochs_checked, grad_window_vs_control=grad_window,
                      artifact_sha256={n: sha(out / n) for n in ('best.pt', 'last.pt', 'train_log.json', 'run_manifest.json')},
                      seconds=time.time() - started, trace_sha256=sha(out / f'OPEN_TRACE_{job}.json'), test_read=False,
                      gate_sha256=sha(HERE / 'GATE.json'), cpu_check_sha256=sha(HERE / 'CPU_CHECK.json'))
        write(out / 'COMPLETE.json', result); write(attempt, dict(status='COMPLETE', job=job, seconds=result['seconds'], freeze_sha256=f))
        print('N2MLR10_COMPLETE', seed, arm, done['best'], flush=True)
    except BaseException as e:
        write(attempt, dict(status='FAILED', job=job, seconds=time.time() - started, error=repr(e), traceback=traceback.format_exc(), freeze_sha256=f))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check', 'gate', 'run'])
    p.add_argument('--seed', type=int, choices=SEEDS); p.add_argument('--arm', choices=ARMS); a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Numerical work requires Slurm'
    if a.mode == 'check': check()
    elif a.mode == 'gate': gate()
    else: run(a.seed, a.arm)
