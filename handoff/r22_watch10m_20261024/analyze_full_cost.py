"""Audit frozen FIT cost records. No training, task evaluation or job submission."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PACKAGE = BASE / 'handoff/r22_cost_validation_20261024'
ARMS = ('vcs_exact', 'vcs_dense', 'vcs_perm', 'bce', 'bce_block', 'nce', 'nce_block')
PAIRS = {('vcs_exact', 'vcs_dense'), ('bce', 'bce_block'), ('nce', 'nce_block')}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_unit(job, gate=False):
    path = Path(job['completion'])
    r = read(path)
    assert r['status'] == 'COMPLETE' and r['gate'] is gate
    assert r['freeze_sha256'] == job['freeze_sha256'] == sha(PACKAGE / 'FREEZE.json')
    assert r['weights_unchanged'] is True and r['numerical_roles'] == ['FIT']
    assert set(r['integrity_roles']) <= {'FIT', 'CAL', 'DEV'}
    assert r['ds'] in ('bf', 'epfl') and r['seed'] in (1001, 1002, 1003)
    prefix = 'r22_cost_gate' if gate else 'r22_cost'
    assert job['name'] == f"{prefix}_{r['ds']}_s{r['seed']}"
    source = read(PACKAGE / 'SOURCE_LOCK.json')[f"{r['ds']}_{r['seed']}"]
    for kind in ('checkpoint', 'completion'):
        assert r[f'source_{kind}_sha256'] == source[f'{kind}_sha256'] == sha(source[kind])
    batches = range(1 if gate else 3)
    expected = {(b, n) for b in batches for n in (128, 512, 2048)}
    assert len(r['rows']) == len(expected)
    assert {(x['batch'], x['n_pos']) for x in r['rows']} == expected
    for row in r['rows']:
        assert set(row['costs']) == set(ARMS)
        shift = (r['seed'] + row['batch'] + row['n_pos']) % len(ARMS)
        assert row['order'] == list(ARMS[shift:] + ARMS[:shift])
        assert len(row['batch_sha256']) == 64 and int(row['batch_sha256'], 16) >= 0
        for c in row['costs'].values():
            assert len(c['all_seconds']) == (2 if gate else 10)
            assert all(math.isfinite(x) and x > 0 for x in c['all_seconds'])
            assert math.isclose(c['median_seconds'], st.median(c['all_seconds']), rel_tol=1e-12)
            assert c['peak_allocated_bytes'] >= c['baseline_bytes'] > 0
            assert c['peak_increment_bytes'] == c['peak_allocated_bytes'] - c['baseline_bytes']
    assert len(r['equivalence']) == len(expected) * len(PAIRS)
    assert {(x['batch'], x['n'], x['reference'], x['alternative']) for x in r['equivalence']} == {
        (b, n, ref, alt) for b, n in expected for ref, alt in PAIRS}
    for x in r['equivalence']:
        assert math.isfinite(x['encoder_gradient_relative_l2']) and 0 <= x['encoder_gradient_relative_l2'] < 2e-4
        assert math.isfinite(x['value_error']) and x['value_error'] >= 0
        # The relative value tolerance requires the reference value, not saved here.
        # Its assertion ran in the frozen producer; do not invent a reconstructed value.
    assert math.isfinite(r['seconds']) and r['seconds'] > 0
    attempts = [read(p) for p in path.parent.glob('ATTEMPT_*.json')]
    assert len(attempts) == 1 and attempts[0]['status'] == 'COMPLETE'
    assert str(attempts[0]['job']) == str(r['job']) == str(r['environment']['slurm_job_id'])
    ledger = (BASE / 'handoff/r22_task_diagnostics_20261023/SUBMITTED.tsv').read_text().splitlines()
    submits = [line.split('\t') for line in ledger if len(line.split('\t')) >= 3 and line.split('\t')[1] == job['name']]
    assert len(submits) == 1 and submits[0][2] == str(r['job'])
    return r


def aggregate(r):
    result = {}
    for n in (128, 512, 2048):
        rows = [x for x in r['rows'] if x['n_pos'] == n]
        assert len(rows) == 3
        # First median of 10 timed repetitions, then median across 3 fixed FIT batches.
        arms = {a: {k: st.median(x['costs'][a][k] for x in rows) for k in
                    ('median_seconds', 'peak_allocated_bytes', 'peak_increment_bytes', 'baseline_bytes')}
                for a in ARMS}
        comparisons = {}
        for family in ('bce', 'nce'):
            candidates = (family, family + '_block')
            fastest = min(candidates, key=lambda a: arms[a]['median_seconds'])
            smallest = min(candidates, key=lambda a: arms[a]['peak_allocated_bytes'])
            speedup = arms[fastest]['median_seconds'] / arms['vcs_exact']['median_seconds']
            no_higher = arms['vcs_exact']['peak_allocated_bytes'] <= arms[fastest]['peak_allocated_bytes']
            comparisons[family] = dict(fastest_variant=fastest, smallest_memory_variant=smallest,
                exact_speedup_vs_fastest=speedup, exact_no_higher_peak_than_fastest=no_higher,
                exact_no_higher_peak_than_smallest=arms['vcs_exact']['peak_allocated_bytes'] <= arms[smallest]['peak_allocated_bytes'],
                criterion_pass=speedup >= 1.2 and no_higher)
        result[str(n)] = dict(arms=arms, comparisons=comparisons,
            both_competitors_pass=all(c['criterion_pass'] for c in comparisons.values()),
            per_batch_comparisons=[dict(batch=x['batch'], **{family: {
                'speedup_vs_fastest': min(x['costs'][a]['median_seconds'] for a in (family, family + '_block')) / x['costs']['vcs_exact']['median_seconds']}
                for family in ('bce', 'nce')}) for x in rows])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-gates', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('cost_admission', PACKAGE / 'admit_full.py')
    admission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admission)
    gate_check = admission.verify()  # Read-only: never calls admission's CLI or submission lock.
    assert gate_check['status'] == 'GATES_VERIFIED_READY'
    if args.check_gates:
        for job in read(PACKAGE / 'GATE_JOBS.json'):
            verify_unit(job, gate=True)
        print('GATE_RECORD_AUDIT_PASS; no full-result report written')
        return
    jobs = read(PACKAGE / 'FULL_JOBS_PENDING_GATES.json')
    missing = [j['name'] for j in jobs if not Path(j['completion']).exists()]
    if missing:
        print(json.dumps(dict(status='WAITING_FOR_ALL6', missing=missing)))
        return
    assert len(jobs) == 6
    units = {}
    for job in jobs:
        r = verify_unit(job)
        key = f"{r['ds']}_{r['seed']}"
        assert key not in units
        units[key] = dict(ds=r['ds'], seed=r['seed'], job=r['job'], gpu=r['gpu'],
            environment=r['environment'], source=job['completion'], source_sha256=sha(job['completion']),
            seconds=r['seconds'], aggregate=aggregate(r),
            max_gradient_relative_l2=max(x['encoder_gradient_relative_l2'] for x in r['equivalence']),
            max_value_error=max(x['value_error'] for x in r['equivalence']))
    assert set(units) == {f'{d}_{s}' for d in ('bf', 'epfl') for s in (1001, 1002, 1003)}
    dataset_criteria = {}
    for ds in ('bf', 'epfl'):
        n_pass = sum(units[f'{ds}_{s}']['aggregate']['2048']['both_competitors_pass'] for s in (1001, 1002, 1003))
        dataset_criteria[ds] = dict(seeds_passing_both_competitors=n_pass, pass_at_least2=n_pass >= 2)
    result = dict(status='AUDIT_PASS', utc=datetime.now(timezone.utc).isoformat(),
        series='r22_cost_validation_20261024:other_cost', freeze_sha256=gate_check['freeze_sha256'],
        units=units, dataset_criteria=dataset_criteria,
        registered_cost_signal=all(x['pass_at_least2'] for x in dataset_criteria.values()),
        cost=dict(full_recorded_gpu_hours=sum(x['seconds'] for x in units.values()) / 3600,
                  gate_recorded_gpu_hours=sum(read(j['completion'])['seconds'] for j in read(PACKAGE / 'GATE_JOBS.json')) / 3600,
                  known_attempts_full=6, scope='Recorded producer elapsed time, excluding Python startup/queue/allocation overhead.'),
        aggregation='Median of10 timings per cell, then median of3 FIT batches for each arm/N/seed. Choose fastest measured BCE and NCE implementation separately, compare its peak; additionally report memory-minimizing variant. Require both competitors in the same at least2of3 seeds per dataset.',
        audit='Frozen files, source/checkpoint hashes, FIT role, exact grid/order, 10 positive finite times, medians/memory arithmetic, 27 numerical comparisons/unit, unchanged weights and one complete attempt/submission/unit verified. Reference-relative value tolerance asserted by frozen producer; reference values not serialized.',
        task_quality_status='Must be interpreted with all dense task series; cost criterion alone does not establish application advantage.',
        limitations=read(PACKAGE / 'PROTOCOL.json')['limitations'])
    path = HERE / 'ANALYSIS_FULL_COST_20261024.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), dataset_criteria=dataset_criteria,
                         registered_cost_signal=result['registered_cost_signal'], cost=result['cost'])))


if __name__ == '__main__':
    main()
