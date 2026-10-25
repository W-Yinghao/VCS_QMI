"""Only cfg.lr_update changes in full runs; original trainer/objective remain frozen."""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import math
import os
import sys
import time
from lr_common import BASE, HERE, OLD, read, write, sha, verify, load, reference, cpu_pass

sys.path.insert(0, str(OLD))
O = load('lr_protected_source', OLD / 'run_retrieval.py')
torch, np = O.torch, O.np
INPUTS, CONTEXT, SETUP = O.inputs, O.R.c1.load_context, O.R.gpu_setup

def inputs(seed, method):
    current = O.HERE
    try:
        O.HERE = OLD
        return INPUTS(seed, method)
    finally: O.HERE = current

def config(old, period):
    assert old.lr_update == 30 and old.num_epochs == 40
    new = replace(old, lr_update=period)
    assert {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]} == ({'lr_update'} if period != 30 else set())
    return new

def lr_vector(period): return [2e-4 * .1 ** (e // period) for e in range(40)]

def environment(seed, method):
    precision, env = SETUP(); _, ref, _ = reference(seed, method)
    assert precision == ref['precision'], ('precision', precision, ref['precision'])
    for key in O.R.c1.PAIRED_KEYS:
        assert env[key] == ref['environment'][key], ('environment', key, env[key], ref['environment'][key])
    return precision, env

def matching(seed, method, cfg, inp, student):
    _, ref, manifest = reference(seed, method)
    assert asdict(cfg) == {**manifest['cfg'], 'lr_update': cfg.lr_update}
    assert student.manifest == {**manifest, 'cfg': asdict(cfg)}, 'Only LR period may change manifest'
    assert inp['transformed'] == ref['transformed_weight']
    checks = O.R.c1.pool_checks(student, inp['rows'], inp['weights'], seed, cfg,
                               {**O.R.c1.PROTOCOL_POOL, 'updates': 45320})
    assert checks['ok'] and checks['data_order_sha256'] == ref['sample_order_sha256']
    init = O.R.c1.initial_model_sha256(cfg, seed)
    assert init == ref['initial_model_sha256'], ('init', seed, method)
    return checks, init, ref

def check():
    f = verify(); finish = O.R.trace_open.install('c0')
    from v9.n2.teacher import HostConfig
    import v9.n2.train as train
    from v9.n2.train import WeightedStudent
    weight_checks = {}
    for seed in (901, 902, 903):
        for method in ('vcs', 'constant'):
            _, ref, manifest = reference(seed, method)
            cfg = HostConfig(**manifest['cfg']); new = config(cfg, 10)
            assert asdict(new) == {**manifest['cfg'], 'lr_update': 10}
            inp = inputs(seed, method)
            assert inp['transformed'] == ref['transformed_weight']
            assert len(inp['rows']) == 145000 and len(inp['weights']) == 130500
            assert abs(math.fsum(inp['weights'].values()) / 130500 - .5) < 1e-12
            weight_checks[f'{method}_{seed}'] = inp['transformed']['sha256']
    # Exact production fit code, small synthetic host: all forty LR transitions
    # are observed at actual Adam.step, not merely calculated by this wrapper.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w = torch.nn.Parameter(torch.randn(4, 4) * .01)
        def forward(self, ims, caps, lens): return self.w
    class Student(WeightedStudent):
        def validate(self, host): return {'rsum': 0.0}
    original_build, original_step = train.build_host, torch.optim.Adam.step
    schedules = {}
    try:
        train.build_host = lambda cfg: Toy()
        for period in (30, 10):
            applied = []
            def step(opt, *a, **kw):
                applied.append([g['lr'] for g in opt.param_groups]); return original_step(opt, *a, **kw)
            torch.optim.Adam.step = step
            cfg = replace(HostConfig(**manifest['cfg']), batch_size=4, lr_update=period)
            rows = [{'pair_id': str(i), 'role': 'U'} for i in range(4)]
            student = Student(cfg, rows, {str(i): .5 for i in range(4)},
                lambda rows: (torch.zeros(4, 1), torch.zeros(4, 1), [1]*4), None, None,
                HERE/'checks'/f'toy_period{period}', 901, objective=O.Objective('constant'), loss_scale='sum')
            student.fit(40, log_every=1)
            log = read(student.out/'train_log.json')
            assert applied == [[x] for x in lr_vector(period)]
            assert [x['lr'] for x in log['epochs']] == lr_vector(period)
            assert all(x['steps_in_epoch'] == 1 and math.isfinite(x['mean_loss']) for x in log['epochs'])
            schedules[str(period)] = applied
    finally:
        train.build_host, torch.optim.Adam.step = original_build, original_step
    trace = finish(HERE/'checks/CPU_TRACE.json'); assert trace['status'] == 'PASS'
    write(HERE/'checks/CPU.json', dict(status='PASS', freeze_sha256=f,
        job=os.environ['SLURM_JOB_ID'], weights=weight_checks, actual_adam_lr=schedules,
        original_fit_source_unchanged=True, trace_sha256=sha(HERE/'checks/CPU_TRACE.json')))

def gate():
    f = verify(); cpu_pass(); started = time.time(); finish = O.R.trace_open.install('c1')
    import v9.n2.train as train
    from v9.n2.train import WeightedStudent
    precision, env = environment(901, 'vcs'); ctx = CONTEXT(); spec = O.R.c1.protocol()
    results = []
    for method in ('vcs', 'constant'):
        environment(901, method)
        for period in (30, 10):
            out = HERE/'gates'/f'{method}_period{period}'; assert not out.exists(), 'No hidden gate restart'
            cfg = config(ctx['cfg'], period); inp = inputs(901, method); batches = []
            def collate(rows):
                if len(batches) < 8: batches.append(hashlib.sha256('\n'.join(r['pair_id'] for r in rows).encode()).hexdigest())
                return ctx['collate'](rows)
            class Student(WeightedStudent):
                def validate(self, host):
                    val = super().validate(host); O.R.c1.require_clean_trace(finish.forbidden_hits); return val
            student = Student(cfg, inp['rows'], inp['weights'], collate,
                ctx['dev_images'][:16], ctx['dev_captions'][:80], out, 901, spec['lr_mult'],
                device='cuda', weight_lock=inp['lock'], formal=True, loss_scale=spec['loss_scale'], objective=O.Objective(method))
            checks, init, ref = matching(901, method, cfg, inp, student)
            original_iterator, original_step = train.iterate_batches, torch.optim.Adam.step
            epoch_calls = [0]; applied = []
            def limited(*a, **kw):
                e = epoch_calls[0]; epoch_calls[0] += 1
                return itertools.islice(original_iterator(*a, **kw), 8 if e == 0 else 1)
            def step(opt, *a, **kw):
                applied.append([g['lr'] for g in opt.param_groups]); return original_step(opt, *a, **kw)
            try:
                train.iterate_batches = limited; torch.optim.Adam.step = step
                student.fit(40, log_every=1)
            finally:
                train.iterate_batches, torch.optim.Adam.step = original_iterator, original_step
            assert epoch_calls[0] == 40
            log = read(out/'train_log.json'); expected = [8] + [1]*39
            assert [x['steps_in_epoch'] for x in log['epochs']] == expected
            assert [x['lr'] for x in log['epochs']] == lr_vector(period)
            assert applied == [[lr] for lr, n in zip(lr_vector(period), expected) for _ in range(n)]
            assert all(math.isfinite(x['loss']) for x in log['steps'])
            assert batches == ref['first_batch_sha256']
            ck = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            assert ck['epoch'] == 39 and all(g['lr'] == lr_vector(period)[-1] for g in ck['opt']['param_groups'])
            before = O.R.c1.initial_model_sha256(cfg, 901)
            train.seed_everything(901)
            initial = train.build_host(cfg).state_dict()
            assert any(not torch.equal(ck['model'][k], v) for k,v in initial.items())
            del initial
            del ck
            results.append(dict(method=method, period=period, epochs=40, updates=47, actual_adam_lr=applied,
                initial_model_sha256=init, first_batch_sha256=batches, pool_checks=checks,
                log_sha256=sha(out/'train_log.json'), last_checkpoint_sha256=sha(out/'last.pt'),
                initial_fingerprint_rechecked=before == init))
            print('GATE_CELL_PASS', method, period, flush=True)
    trace = finish(HERE/'checks/GPU_TRACE.json'); assert trace['status'] == 'PASS'
    write(HERE/'gates/COMPLETE.json', dict(status='COMPLETE', freeze_sha256=f, job=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-started, rows=results, precision=precision, environment=env,
        trace_sha256=sha(HERE/'checks/GPU_TRACE.json'), cpu_sha256=sha(HERE/'checks/CPU.json'),
        scope='Four real SGR cells x forty sparse epochs, 188 actual updates; first8 original batches, then firstbatch per epoch. Technical only, not performance evidence.'))

def run(seed, method):
    f = verify(); cpu_pass(); g = read(HERE/'gates/COMPLETE.json')
    assert g['status'] == 'COMPLETE' and g['freeze_sha256'] == f
    finish = O.R.trace_open.install('c1'); environment(seed, method)
    out = HERE/'runs'/f'retrieval_{method}_s{seed}'; assert not out.exists(), 'No duplicate or silent resume'
    ctx = CONTEXT(); ctx['cfg'] = config(ctx['cfg'], 10)
    inp = inputs(seed, method)
    # Preflight complete pool/order, source init and manifest before training.
    from v9.n2.train import WeightedStudent
    probe = WeightedStudent(ctx['cfg'], inp['rows'], inp['weights'], ctx['collate'],
        ctx['dev_images'], ctx['dev_captions'], HERE/'preflight'/f'{method}_{seed}', seed,
        O.R.c1.protocol()['lr_mult'], device='cuda', weight_lock=inp['lock'], formal=True,
        loss_scale='sum', objective=O.Objective(method))
    _, _, ref = matching(seed, method, ctx['cfg'], inp, probe); del probe
    O.HERE = HERE; O.inputs = inputs; O.R.c1.load_context = lambda: ctx
    O.R.gpu_setup = lambda: environment(seed, method)
    r = O.train_one(seed, method, False, finish)
    for k in ('initial_model_sha256','sample_order_sha256','first_batch_sha256','precision'):
        assert r[k] == ref[k], k
    log = read(out/'train_log.json'); assert [x['lr'] for x in log['epochs']] == lr_vector(10)
    trace = finish(out/f'OPEN_TRACE_{os.environ["SLURM_JOB_ID"]}.json'); assert trace['status'] == 'PASS'
    r.update(lr_update=10, lr_vector=lr_vector(10), reused_control=read(HERE/'SOURCE_LOCK.json')['controls'][f'{method}_{seed}'],
             technical_gate_sha256=sha(HERE/'gates/COMPLETE.json'), trace_sha256=sha(out/f'OPEN_TRACE_{os.environ["SLURM_JOB_ID"]}.json'))
    write(out/'COMPLETE.json', r)
    print('FULL_COMPLETE', seed, method, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['check','gate','run'])
    p.add_argument('--seed', type=int, choices=[901,902,903]); p.add_argument('--method', choices=['vcs','constant']); a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Numerical work requires Slurm'
    if a.mode == 'check': check()
    elif a.mode == 'gate': gate()
    else: run(a.seed,a.method)
