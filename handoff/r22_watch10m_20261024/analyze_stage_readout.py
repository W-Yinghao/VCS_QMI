"""Audit the frozen 24-unit stage-readout series; no conclusions from partial runs."""
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
BUNDLE = BASE / 'handoff/r22_stage_readout_v2_20261024'
ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
SEEDS = (1001, 1002, 1003)
POLICIES = ('COS__COS', 'J_EXACT__COS', 'COS__J_EXACT', 'J_EXACT__J_EXACT')
METRICS = ('success9', 'success42', 'median_err', 'p95_err', 'failure100_rate')


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def signature(rr, field):
    return [(r['core'], r['scene'], r['perturbation'],
             r[field]['COS__COS' if field == 'policies' else 'COS']['theta'],
             r[field]['COS__COS' if field == 'policies' else 'COS']['t']) for r in rr]


def metrics(rr, policy):
    scenes = defaultdict(list)
    for r in rr:
        scenes[r['scene']].append(r['policies'][policy]['err_full'])
    errors = sorted(x for v in scenes.values() for x in v)
    curve = {str(t): st.mean(st.mean(x < t for x in v) for v in scenes.values())
             for t in (3, 9, 21, 42, 100)}
    return dict(success9=curve['9'], success42=curve['42'],
                median_err=st.median(errors), p95_err=errors[math.ceil(.95 * len(errors))-1],
                failure100_rate=1-curve['100'], failed100=sum(x >= 100 for x in errors),
                failures42=sum(x >= 42 for x in errors), n=len(errors), scenes=len(scenes),
                success_curve=curve,
                seconds=sum(r['policies'][policy]['synchronized_wall_seconds'] for r in rr),
                scene_failure_counts={c: sum(x >= 42 for x in v) for c, v in scenes.items()})


def audit_unit(job, freeze, sources):
    p = Path(job['completion']).parent
    r = read(p / 'COMPLETE.json')
    ds, arm, seed = r['ds'], r['arm'], r['seed']
    key = f'{ds}_{arm}_{seed}'
    assert job['name'] == f'r22_stage2_{ds}_{arm}_s{seed}'
    assert r['status'] == 'COMPLETE' and not r['gate']
    assert r['freeze_sha256'] == job['freeze_sha256'] == freeze
    assert r['numerical_roles'] == ['DEV'] and r['weights_unchanged']
    assert len(r['models']) == 1 and r['models'][0]['unchanged']
    source = sources[key]
    assert r['models'][0]['source'] == source and r['models'][0]['arm'] == arm
    for k in ('completion', 'checkpoint', 'cal_rows'):
        assert sha(source[k]) == source[k + '_sha256']
    original = read(source['completion'])
    assert original['status'] == 'COMPLETE'
    assert all(original['identity'][k] == v for k, v in [('ds', ds), ('arm', arm), ('seed', seed)])
    assert original['selection']['step'] == source['step']
    assert original['selection']['checkpoint_sha256'] == source['checkpoint_sha256']
    original_rows = Path(source['completion']).parent / 'dev/DEV.jsonl'
    assert sha(original_rows) == original['dev_rows_sha256']
    assert sha(p / 'ROWS.jsonl') == r['rows_sha256']
    assert sha(p / 'INPUT_TRACE.json') == r['input_trace_sha256']
    trace = read(p / 'INPUT_TRACE.json')
    assert len(trace) == 1 and trace[0]['arm'] == arm
    assert {x['role'] for x in trace[0]['reads']} <= {'FIT', 'CAL', 'DEV'}
    rr = rows(p / 'ROWS.jsonl')
    assert rr and len({(x['core'], x['perturbation']) for x in rr}) == len(rr)
    assert signature(rr, 'policies') == signature(rows(original_rows), 'E')
    for c in {x['core'] for x in rr}:
        assert {x['perturbation'] for x in rr if x['core'] == c} == set(range(10))
    digest_disagreements = {coarse: 0 for coarse in ('COS', 'J_EXACT')}
    ftrue_gap = {local: 0. for local in ('COS', 'J_EXACT')}
    for index, row in enumerate(rr):
        assert row['role'] == 'DEV' and row['arm'] == arm
        assert set(row['policies']) == set(POLICIES)
        shift = (seed + index) % 4
        assert row['order'] == list(POLICIES[shift:] + POLICIES[:shift])
        ref = row['policies']['COS__COS']
        for value in row['policies'].values():
            assert value['theta'] == ref['theta'] and value['t'] == ref['t']
            assert (value['n_grid'], value['grid_calls'], value['local_calls']) == (10668, 21, 303)
            assert all(math.isfinite(value[k]) for k in ('err_full', 'F_true', 'F_hat', 'synchronized_wall_seconds'))
            assert value['synchronized_wall_seconds'] > 0 and value['err_full'] >= 0
        for coarse in ('COS', 'J_EXACT'):
            a, b = [row['policies'][coarse + '__' + local] for local in ('COS', 'J_EXACT')]
            digest_disagreements[coarse] += int(a['grid_values_sha256'] != b['grid_values_sha256'])
        for local in ('COS', 'J_EXACT'):
            a, b = [row['policies'][coarse + '__' + local] for coarse in ('COS', 'J_EXACT')]
            ftrue_gap[local] = max(ftrue_gap[local], abs(a['F_true'] - b['F_true']))
    computed = {policy: metrics(rr, policy) for policy in POLICIES}
    for policy, m in computed.items():
        stored = r['summaries'][arm][policy]
        for k in ('success9', 'success42', 'median_err', 'n', 'scenes', 'failures42', 'seconds'):
            assert math.isclose(m[k], stored[k], rel_tol=1e-12, abs_tol=1e-12), (key, policy, k)
    attempts = [read(x) for x in p.glob('ATTEMPT_*.json')]
    assert len(attempts) == 1 and attempts[0]['status'] == 'COMPLETE' and attempts[0]['job'] == r['job']
    assert r['seconds'] > 0 and math.isfinite(r['seconds'])
    assert sum(m['seconds'] for m in computed.values()) <= r['seconds'] + 1
    progress = read(p / 'PROGRESS.json')
    assert progress['rows'] == progress['expected'] == len(rr)
    id_field = 'core' if ds == 'bf' else 'window'
    dev_reads = [x for x in trace[0]['reads'] if x['role'] == 'DEV']
    assert all(id_field in x for x in dev_reads)
    assert {x[id_field] for x in dev_reads} == {x['core'] for x in rr}
    inputs = [(x[id_field], x.get('record_id'), x.get('path'), x.get('kind'),
               x.get('purpose'), x.get('stage')) for x in dev_reads]
    unit = dict(ds=ds, arm=arm, seed=seed, job=r['job'], policies=computed,
                seconds=r['seconds'], environment=r['environment'], source=source,
                input_trace_sha256=r['input_trace_sha256'], rows_sha256=r['rows_sha256'],
                same_coarse_grid_digest_disagreements=digest_disagreements,
                same_local_ftrue_max_gap=ftrue_gap,
                synchronized_time_ratios={p: computed[p]['seconds']/computed['COS__COS']['seconds'] for p in POLICIES},
                inherited_cal_curve=original['selection']['cal'])
    return key, unit, signature(rr, 'policies'), inputs


def delta(values):
    return dict(per_seed=values, mean=st.mean(values), positive=sum(x > 0 for x in values))


def effects(units, ds):
    factorial = {}
    comparisons = [('coarse_J_at_local_COS', 1, 0), ('coarse_J_at_local_J', 3, 2),
                   ('local_J_at_coarse_COS', 2, 0), ('local_J_at_coarse_J', 3, 1)]
    for arm in ARMS:
        seeds = [units[f'{ds}_{arm}_{s}'] for s in SEEDS]
        factorial[arm] = {label: {m: delta([u['policies'][POLICIES[a]][m] - u['policies'][POLICIES[b]][m]
                                           for u in seeds]) for m in METRICS}
                           for label, a, b in comparisons}
        factorial[arm]['interaction'] = {m: delta([
            u['policies'][POLICIES[3]][m] - u['policies'][POLICIES[2]][m]
            - u['policies'][POLICIES[1]][m] + u['policies'][POLICIES[0]][m] for u in seeds]) for m in METRICS}
        ratios = [u['synchronized_time_ratios'][POLICIES[1]] for u in seeds]
        factorial[arm]['primary_runtime_ratio'] = dict(per_seed=ratios, median=st.median(ratios))
    exact_minus = {p: {arm: {m: delta([units[f'{ds}_vcs_exact_{s}']['policies'][p][m]
                                       - units[f'{ds}_{arm}_{s}']['policies'][p][m] for s in SEEDS])
                                  for m in METRICS} for arm in ('vcs_perm', 'bce', 'nce')} for p in POLICIES}
    primary = factorial['vcs_exact']['coarse_J_at_local_COS']
    search = (primary['success9']['mean'] >= .01 and primary['success9']['positive'] >= 2
              and primary['success42']['mean'] >= -.0025
              and factorial['vcs_exact']['primary_runtime_ratio']['median'] <= 1.5)
    specific = all(exact_minus[POLICIES[1]][a]['success9']['mean'] >= .01
                   and exact_minus[POLICIES[1]][a]['success9']['positive'] >= 2
                   and exact_minus[POLICIES[1]][a]['success42']['mean'] >= -.0025 for a in ('bce', 'nce'))
    means = {a: {p: {m: st.mean(units[f'{ds}_{a}_{s}']['policies'][p][m] for s in SEEDS)
                      for m in METRICS} for p in POLICIES} for a in ARMS}
    return dict(means=means, factorial=factorial, exact_minus=exact_minus,
                useful_search_signal=search, matched_objective_signal=specific,
                vcs_specific_application_signal=search and specific)


def analyze(check_completed=False):
    jobs = read(BUNDLE / 'FULL_JOBS_PENDING_GATES.json')
    assert len(jobs) == len({j['name'] for j in jobs}) == 24
    ready = [j for j in jobs if Path(j['completion']).exists()]
    if len(ready) < 24 and not check_completed:
        return dict(status='WAITING_FOR_COMPLETE_SERIES',complete=len(ready),expected=24)
    freeze = sha(BUNDLE / 'FREEZE.json')
    for p, h in read(BUNDLE / 'FREEZE.json')['files'].items():
        assert sha(BASE / p) == h, p
    admission = read(BUNDLE / 'FULL_ADMISSION.json')
    assert admission['status'] == 'FULL24_ADMITTED' and admission['freeze_sha256'] == freeze
    assert admission['full_manifest_sha256'] == sha(BUNDLE / 'FULL_JOBS_PENDING_GATES.json')
    for gate in admission['gates']:
        assert sha(gate['path']) == gate['sha256']
    sources = read(BUNDLE / 'SOURCE_LOCK.json')
    units, truths, inputs = {}, {}, {}
    for job in ready:
        key, u, sig, inp = audit_unit(job, freeze, sources)
        pair = (u['ds'], u['seed'])
        if pair in truths:
            assert sig == truths[pair] and inp == inputs[pair], key
        truths[pair], inputs[pair], units[key] = sig, inp, u
    if check_completed:
        return dict(status='COMPLETED_UNIT_AUDITS_PASS', complete=len(ready), expected=24,
                    freeze_sha256=freeze, audited_jobs=[u['job'] for u in units.values()],
                    scientific_conclusion_generated=False)
    assert set(units) == {f'{d}_{a}_{s}' for d in ('bf','epfl') for a in ARMS for s in SEEDS}
    datasets = {ds: effects(units, ds) for ds in ('bf', 'epfl')}
    old = BASE / 'handoff/r22_stage_readout_20261024'
    failed = read(old / 'GATE_FAILURE_1062609.json')
    cost = [dict(kind='full_receipt', job=u['job'], seconds=u['seconds']) for u in units.values()]
    for gate in admission['gates']:
        r = read(gate['path']); cost.append(dict(kind='v2_gate_receipt', job=r['job'], seconds=r['seconds']))
    r = read(old / 'gates/epfl_gate/COMPLETE.json')
    cost.append(dict(kind='v1_successful_gate_receipt', job=r['job'], seconds=r['seconds']))
    cost.append(dict(kind='v1_failed_gate_scheduler', job=failed['job'], seconds=failed['scheduler_seconds']))
    r = read(old / 'DIAGNOSTIC_COMPLETE.json')
    cost.append(dict(kind='failure_diagnostic_receipt', job=r['job'], seconds=r['seconds']))
    result = dict(status='AUDIT_PASS', utc=datetime.now(timezone.utc).isoformat(),
                  freeze_sha256=freeze, analyzer_sha256=sha(__file__), units=units, datasets=datasets,
                  cross_dataset_useful_search_signal=all(v['useful_search_signal'] for v in datasets.values()),
                  cross_dataset_vcs_specific_signal=all(v['vcs_specific_application_signal'] for v in datasets.values()),
                  cost=dict(records=cost, known_seconds=sum(c['seconds'] for c in cost),
                            note='Receipts plus failed gate scheduler time; successful scheduler startup and inherited training excluded. Not total allocation cost.'),
                  limitations=['Reused DEV, no independent confirmation; three seeds, scene-macro success, perturbations not independent.',
                               'Source checkpoint hashes and logged input traces verified; runtime input bytes not independently rehashed by this analyzer.',
                               'Pure and hybrid policies paired within each job; training objectives may use different GPUs/hosts. No cross-host speedup claim.',
                               'Grid digest differences and F_true gaps are reported under frozen nondeterministic evaluation, not automatically excluded.',
                               'No new training curve: inherited CAL trajectories reported; readout success curves retained. Thresholds are reporting rules, not permanent stopping rules.'])
    out = HERE / 'ANALYSIS_STAGE_READOUT_ALL24_20261024.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(status='AUDIT_PASS',path=str(out),cross_dataset_useful_search_signal=result['cross_dataset_useful_search_signal'],
                cross_dataset_vcs_specific_signal=result['cross_dataset_vcs_specific_signal'])


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--check-completed',action='store_true')
    print(json.dumps(analyze(p.parse_args().check_completed),ensure_ascii=False))
