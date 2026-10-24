"""Audit a COMPLETE registered dense-geometry series using existing CAL/DEV records only."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
DENSE = BASE / 'handoff/r22_dense_geometry_20261023'
ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
SEEDS = (1001, 1002, 1003)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def quantile(v, q):
    v = sorted(v)
    i = q * (len(v) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (i - lo) * (v[hi] - v[lo])


def summarize(rr, view):
    scenes = defaultdict(list)
    for r in rr:
        scenes[r['scene']].append(r['E'][view]['err_full'])
    errors = [x for v in scenes.values() for x in v]
    out = {f'success{t}': st.mean(st.mean(x < t for x in v) for v in scenes.values()) for t in (9, 42)}
    out.update(median_err=st.median(errors), n=len(errors), p95_err=quantile(errors, .95),
               max_err=max(errors), failed42_count=sum(x >= 42 for x in errors),
               failed100_count=sum(x >= 100 for x in errors))
    return out


def audit(ds, n):
    jobs = [j for j in read(DENSE / 'JOBS.json') if j['name'].startswith(f'r22_dense{n}_{ds}_')]
    assert len(jobs) == 12
    missing = [j['name'] for j in jobs if not Path(j['completion']).exists()]
    if missing:
        raise SystemExit('Series incomplete; no review written: ' + ', '.join(missing))
    freeze_hash = sha(DENSE / 'FREEZE.json')
    freeze = read(DENSE / 'FREEZE.json')['files']
    for rel, digest in freeze.items():
        assert sha(BASE / rel) == digest, rel
    ledger = [x.split('\t') for x in (BASE / 'handoff/r22_task_diagnostics_20261023/SUBMITTED.tsv').read_text().splitlines()]
    automatic = read(BASE / 'handoff/r22_task_diagnostics_20261023/RESEARCH_SUMMARY.json')
    result, signatures, truth = {}, {}, {}
    for j in jobs:
        path = Path(j['completion']); r = read(path); p = path.parent
        a, s = r['identity']['arm'], r['identity']['seed']
        assert a in ARMS and s in SEEDS
        assert r['status'] == 'COMPLETE' and r['freeze_sha256'] == j['freeze_sha256'] == freeze_hash
        assert r['identity'] == dict(ds=ds, arm=a, seed=s, n_pos=n, freeze_sha256=freeze_hash, smoke=False)
        assert set(r['roles_read']) == {'FIT', 'CAL', 'DEV'}
        assert sha(p / 'TRAIN.jsonl') == r['train_rows_sha256']
        assert sha(p / 'dev/DEV.jsonl') == r['dev_rows_sha256']
        train = rows(p / 'TRAIN.jsonl'); dev = rows(p / 'dev/DEV.jsonl')
        assert [x['step'] for x in train] == list(range(1, 3001))
        assert all(x['n_pos'] == n and x['training_objective'] == a and
                   math.isfinite(x['loss']) and math.isfinite(x['grad_norm_preclip']) for x in train)
        sig = (r['initial_encoder_sha256'], tuple(x['batch_sha256'] for x in train))
        if s in signatures:
            assert signatures[s] == sig, ('input/init mismatch', s, a)
        signatures[s] = sig
        near(sum(x['train_step_seconds'] for x in train), r['train_seconds'])
        last = read(p / 'last.json'); assert last['step'] == 3000 and last['identity'] == r['identity']
        assert sha(p / 'last.pt') == last['sha256']
        selection = r['selection']; assert selection == read(p / 'SELECTION_BEFORE_DEV.json')
        assert set(selection['cal']) == {'1000', '2000', '3000'}
        for step, reported in selection['cal'].items():
            calrows = rows(p / f'cal_{step}/CAL.jsonl')
            assert all(x['role'] == 'CAL' for x in calrows)
            metrics = summarize(calrows, 'COS')
            for k, v in reported.items():
                near(metrics[k], v)
        chosen = max(selection['cal'], key=lambda k: (selection['cal'][k]['success42'],
                     selection['cal'][k]['success9'], -selection['cal'][k]['median_err'], -int(k)))
        assert int(chosen) == selection['step']
        assert sha(p / f'checkpoints/step_{chosen}.pt') == selection['checkpoint_sha256']
        keys = [(x['core'], x['perturbation']) for x in dev]
        assert len(keys) == len(set(keys)) and all(x['role'] == 'DEV' for x in dev)
        tsig = [(x['core'], x['scene'], x['perturbation'], x['E']['COS']['theta'], x['E']['COS']['t'], x['E']['COS']['n_grid']) for x in dev]
        if s in truth:
            assert tsig == truth[s]
        truth[s] = tsig
        metrics = {v: summarize(dev, v) for v in ('COS', 'J_EXACT')}
        for view, reported in r['dev'].items():
            for k, v in reported.items():
                near(metrics[view][k], v)
        existing = automatic['dense_geometry'][str(n)][ds][a][str(s)]
        assert existing['dev'] == r['dev'] and existing['selection'] == selection
        attempts = [read(f) for f in p.glob('ATTEMPT_*.json')]
        submits = [x for x in ledger if len(x) >= 3 and x[1] == j['name']]
        assert len(attempts) == len(submits) == 1 and submits[0][2] == str(r['job'])
        assert attempts[0]['status'] == 'COMPLETE'
        result.setdefault(a, {})[str(s)] = dict(dev=metrics, selection=selection, gpu=r['gpu'],
            host=r['environment']['host'], precision=r['environment']['precision'],
            initial_encoder_sha256=r['initial_encoder_sha256'],
            batch_trace_sha256=hashlib.sha256(''.join(x['batch_sha256'] for x in train).encode()).hexdigest(),
            training_seconds=r['train_seconds'], completion_seconds=r['seconds'],
            attempt_seconds=attempts[0]['seconds'], peak_allocated_bytes=r['peak_allocated_bytes'],
            loss_blocks=[st.mean(x['loss'] for x in train[k:k+1000]) for k in (0,1000,2000)],
            last100_loss=st.mean(x['loss'] for x in train[-100:]), known_attempts=len(attempts))
    means = {a: {v: {k: st.mean(result[a][str(s)]['dev'][v][k] for s in SEEDS)
              for k in ('success9','success42','median_err','p95_err','failed42_count','failed100_count')}
              for v in ('COS','J_EXACT')} for a in ARMS}
    paired = {}
    for a in ARMS[1:]:
        paired[a] = {}
        for v in ('COS','J_EXACT'):
            paired[a][v] = {}
            for k in ('success9','success42','median_err'):
                diffs = [result['vcs_exact'][str(s)]['dev'][v][k]-result[a][str(s)]['dev'][v][k] for s in SEEDS]
                auto = automatic['dense_pairs'][str(n)][ds]['vcs_exact-minus-'+a][v][k]
                assert auto['seeds'] == [str(s) for s in SEEDS]
                near(auto['mean_density_difference'], st.mean(diffs))
                paired[a][v][k] = dict(differences=diffs, mean=st.mean(diffs), positive=sum(x>0 for x in diffs))
    curves = {a: {step: {k: st.mean(result[a][str(s)]['selection']['cal'][step][k] for s in SEEDS)
              for k in ('success9','success42','median_err')} for step in ('1000','2000','3000')} for a in ARMS}
    by_hardware = defaultdict(list)
    for a in ARMS:
        for s, r in result[a].items():
            by_hardware[r['gpu']].append(dict(arm=a, seed=s, host=r['host'], training_seconds=r['training_seconds'],
                completion_seconds=r['completion_seconds'], peak_allocated_bytes=r['peak_allocated_bytes']))
    return dict(utc=datetime.now(timezone.utc).isoformat(), status='AUDIT_PASS', dataset=ds, n_positions=n,
        series=f'r22_dense_geometry_20261023:{n}_{ds}', source_freeze=freeze_hash,
        frozen_files_verified=len(freeze), units=12, seeds=list(SEEDS), fixed_updates=3000,
        units_by_arm=result, means=means, exact_minus=paired, cal_curves=curves,
        dense_minus128=automatic['dense_minus128'][str(n)][ds],
        cost=dict(completion_gpu_hours=sum(r['completion_seconds'] for a in result.values() for r in a.values())/3600,
            attempt_gpu_hours=sum(r['attempt_seconds'] for a in result.values() for r in a.values())/3600,
            training_gpu_hours=sum(r['training_seconds'] for a in result.values() for r in a.values())/3600,
            by_hardware=dict(by_hardware), failed_training_attempts=0,
            scope='Run receipts; excludes scheduler startup, gates and FIT microbenchmarks. No cross-node speedup claim.'),
        limitations=['Exploratory repeated DEV; no TEST or independent confirmation.',
            'Fixed3000 updates and three CAL candidates; not equal wall-clock budget.',
            'Hardware/driver/node differ; non-deterministic pose search adds evaluation variation.',
            'CAL curves describe training progress; no DEV curve or longer-training extrapolation.',
            'Peak allocation includes evaluation; objective-only microbenchmarks exclude encoder.',
            'More positions change P sampling too; correlated local points are not independent data.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--ds', choices=('bf','epfl'), required=True)
    p.add_argument('--n', type=int, choices=(512,2048), required=True); args = p.parse_args()
    r = audit(args.ds, args.n)
    out = HERE / f'ANALYSIS_DENSE_{args.ds.upper()}_{args.n}_20261024.json'
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','means','cal_curves','cost')},ensure_ascii=False))
