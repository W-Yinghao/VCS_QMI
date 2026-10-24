"""Offline complete-series review; reads BF FIT/CAL/DEV records only."""
import csv
import hashlib
import json
import math
import re
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PKG = BASE / 'handoff/r22_geometry_long_v2_20261024'
ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
SEEDS = (1001, 1002, 1003)


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines()]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source = HERE / 'ANALYSIS_LONG_BF_20261024.json'
    audit = read(source)
    assert audit['status'] == 'AUDIT_PASS' and len(audit['units']) == 12
    source_lock = read(PKG / 'SOURCE_LOCK.json')
    core_rows, sources, initial = {}, [], {}
    all_dev_cores, cal_signatures = None, {}
    for name, u in audit['units'].items():
        arm, seed = u['arm'], u['seed']
        folder = PKG / 'runs' / f'bf_{arm}_{seed}'
        r = read(folder / 'COMPLETE.json')
        assert r['identity'] == dict(ds='bf', arm=arm, seed=seed, freeze_sha256=audit['freeze_sha256'])
        old = source_lock[f'bf_{arm}_{seed}']
        prior = read(old['completion'])
        assert prior['status'] == 'COMPLETE' and prior['identity'] == old['identity']
        if seed in initial:
            assert initial[seed] == prior['initial_encoder_sha256']
        initial[seed] = prior['initial_encoder_sha256']
        tr = rows(old['source_train'])
        assert len(tr) == 3000 and [x['step'] for x in tr] == list(range(1, 3001))
        attempts = [read(p) for p in Path(old['completion']).parent.glob('ATTEMPT_*.json')]
        assert len(attempts) == 1 and attempts[0]['status'] == 'COMPLETE'
        sources.append(dict(unit=name, job=prior['job'], seconds=prior['seconds'],
                            device=prior['environment']['gpu_name'],
                            driver=prior['environment']['precision']['nvidia_smi_name_driver'],
                            attempts=attempts))
        old_dev = rows(Path(old['completion']).parent / 'dev/DEV.jsonl')
        expected_cores = {x['core'] for x in old_dev}
        assert len(expected_cores) == 32
        if all_dev_cores is not None:
            assert expected_cores == all_dev_cores
        all_dev_cores = expected_cores
        core_rows[name] = {}
        for budget in (3000, 30000):
            rr = rows(folder / f'dev_{budget}/DEV.jsonl')
            assert len(rr) == 320 and {x['core'] for x in rr} == expected_cores
            assert all(x['E']['COS']['n_grid'] == 10668 for x in rr)
            by_core = defaultdict(list)
            for x in rr:
                by_core[x['core']].append(x['E']['COS']['err_full'])
            core_rows[name][budget] = {
                c: dict(success9=sum(e < 9 for e in es), success42=sum(e < 42 for e in es),
                        failed100=sum(e >= 100 for e in es), median_err=st.median(es), n=len(es))
                for c, es in sorted(by_core.items())}
        for step in (10000, 20000, 30000):
            rr = rows(folder / f'cal_{step}/CAL.jsonl')
            assert len(rr) == 80 and len({x['core'] for x in rr}) == 8
            assert len({(x['core'], x['perturbation']) for x in rr}) == 80
            assert all(x['role'] == 'CAL' and set(x['E']) == {'COS'} and x['E']['COS']['n_grid'] == 10668 for x in rr)
            for c in {x['core'] for x in rr}:
                assert {x['perturbation'] for x in rr if x['core'] == c} == set(range(10))
            sig = [(x['core'], x['perturbation'], x['E']['COS']['theta'], x['E']['COS']['t']) for x in rr]
            if seed in cal_signatures:
                assert sig == cal_signatures[seed]
            cal_signatures[seed] = sig
            for k in ('success9', 'success42', 'median_err'):
                assert math.isclose(u['cal'][str(step)][k], r['cal'][str(step)]['COS'][k], abs_tol=1e-12)

    means, curves = {}, {}
    for arm in ARMS:
        uu = [audit['units'][f'{arm}_{s}'] for s in SEEDS]
        means[arm] = {end: {k: st.mean(x[end][k] for x in uu)
                           for k in ('success9', 'success42', 'median_err', 'p95_err', 'failed100')}
                      for end in ('short', 'long')}
        curves[arm] = dict(
            cal={step: {k: st.mean(x['cal'][step][k] for x in uu) for k in ('success9', 'success42', 'median_err', 'p95_err')}
                 for step in ('10000', '20000', '30000')},
            fit={step: {k: st.mean(x['fit_curve'][step][k] for x in uu) for k in ('loss', 'grad')}
                 for step in ('4000', '10000', '20000', '30000')})
    gates = []
    for directory, filename in (
        ('r22_geometry_long_20261024', 'PASSED_1062671_SCONTROL.txt'),
        ('r22_geometry_long_20261024', 'FAILED_1062696_SCONTROL.txt'),
        ('r22_geometry_long_v2_20261024', 'PASSED_1062765_SCONTROL.txt'),
        ('r22_geometry_long_v2_20261024', 'PASSED_1062771_SCONTROL.txt')):
        p = BASE / 'handoff' / directory / filename
        t = p.read_text()
        h, m, s = map(int, re.search(r'RunTime=(\d+):(\d+):(\d+)', t).groups())
        gates.append(dict(path=str(p), sha256=sha(p), job=re.search(r'JobId=(\d+)', t)[1],
                          state=re.search(r'JobState=(\w+)', t)[1], scheduler_seconds=h*3600+m*60+s))
    new_h = audit['cost']['receipt_gpu_hours']
    source_h = sum(x['seconds'] for x in sources) / 3600
    gates_h = sum(x['scheduler_seconds'] for x in gates) / 3600
    review = dict(
        status='COMPLETE_SERIES_REVIEW_PASS', utc=datetime.now(timezone.utc).isoformat(),
        audit_path=str(source), audit_sha256=sha(source), seeds=SEEDS, means=means,
        curves=curves, per_core_fixed_endpoints=core_rows,
        checks=dict(original3000_steps_and_initialization_paired=True,
                    original3000_batch_trace='Not saved by original trainer; no retrospective per-step hash equality claim. Source state hashes and original common sampler/CPU gate remain the evidence.',
                    new27000_update_and_input_audit='PASS in linked base audit',
                    dev_all32_original_cores_and10perturbations=True,
                    cal_all8cores_and10perturbations_and_same_truth=True,
                    same_fixed_search_grid10668=True,
                    each_original_and_continuation_one_successful_attempt=True,
                    full_run_access_evidence='Frozen runtime role assertion plus aggregated roles_read; full raw guard trace was not saved. Gate raw trace is saved.'),
        cost=dict(new_continuation_and_evaluation_receipt_gpu_hours=new_h,
                  new_synchronized_training_gpu_hours=sum(x['compute_seconds'] for x in audit['cost']['records'])/3600,
                  original12_train_and_evaluation_receipts_gpu_hours=source_h,
                  shared_clean_long_v1_v2_gates_scheduler_gpu_hours=gates_h,
                  new_plus_old_receipts_and_shared_gates_known_gpu_hours=new_h+source_h+gates_h,
                  original_records=sources, shared_gate_records=gates,
                  note='Original receipts include their CAL/DEV evaluation, not pure prefix training; shared gates counted once for the whole BF/EPFL long campaign, not again per dataset. New receipt times exclude batch startup/teardown. Original project gates and unrelated diagnostics are not in this scoped total.'),
        limitations=[
            'All twelve continuations use L40S; perm1001 and BCE1003 use driver595.84, other ten use580.178.04. Original3000prefix is mixed RTX/L40S and driver; not identical training environment.',
            'Samejob short/long evaluation still has nondeterministic pose search; effect is training-duration plus registered LR decay, not pure duration.',
            'Three seeds and reused DEV only; 320 perturbations are not320independent units. Per-core tail inspection is descriptive, no core removal.',
            'CAL strict/loose is saturated at100% from10k in every arm; medians fluctuate, late FIT flattening is not proof of converged DEV performance.',
            'Error>=100 is a severe geometric error count, not software job failures. No TEST reads or checkpoint selection.'],
        decision=dict(registered_practical_signal=audit['registered_practical_signal'],
                      interpretation='Longer fixed recipe improves exact VCS versus its short endpoint, including failure tails. At equal30k budget it trades lower strict success than NCE for better loose success; a uniformly superior VCS method is not established.',
                      action='Continue all admitted EPFL clean long,48noisy long and6protected retrieval jobs at fixed settings. Analyze each full matched series. No extra clean60k batch yet: existing noise arms test late wrong-pair memorization with matched clean30k controls. Reporting threshold is not a permanent research prohibition.'))
    out = HERE / 'ANALYSIS_LONG_BF_REVIEW_20261024.json'
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    with (HERE / 'ANALYSIS_LONG_BF_MEANS_20261024.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm','budget','success9_percent','success42_percent','mean_seed_median_error','mean_seed_p95_error','mean_seed_error_ge100_count'])
        for arm in ARMS:
            for end, budget in (('short',3000),('long',30000)):
                x = means[arm][end]
                w.writerow([arm,budget,100*x['success9'],100*x['success42'],x['median_err'],x['p95_err'],x['failed100']])
    print(json.dumps(dict(status=review['status'], path=str(out), new_gpu_hours=new_h,
                          original_gpu_hours=source_h, shared_gates_gpu_hours=gates_h,
                          scoped_known_gpu_hours=new_h+source_h+gates_h), ensure_ascii=False))


if __name__ == '__main__':
    main()
