"""Offline EPFL complete-series audit; no training or TEST access."""
import csv
import json
import math
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from review_long_bf_20261024 import read, rows, sha, HERE, BASE, PKG, ARMS, SEEDS


def main():
    source = HERE / 'ANALYSIS_LONG_EPFL_20261024.json'
    audit = read(source)
    assert audit['status'] == 'AUDIT_PASS' and len(audit['units']) == 12
    source_lock = read(PKG / 'SOURCE_LOCK.json')
    window_rows, original_cost, initial, cal_signatures = {}, [], {}, {}
    dev_membership = None
    for name, u in audit['units'].items():
        arm, seed = u['arm'], u['seed']
        folder = PKG / 'runs' / f'epfl_{arm}_{seed}'
        r = read(folder / 'COMPLETE.json')
        assert r['identity'] == dict(ds='epfl', arm=arm, seed=seed, freeze_sha256=audit['freeze_sha256'])
        old = source_lock[f'epfl_{arm}_{seed}']
        prior = read(old['completion'])
        assert prior['status'] == 'COMPLETE' and prior['identity'] == old['identity']
        if seed in initial:
            assert initial[seed] == prior['initial_encoder_sha256']
        initial[seed] = prior['initial_encoder_sha256']
        tr = rows(old['source_train'])
        assert len(tr) == 3000 and [x['step'] for x in tr] == list(range(1, 3001))
        attempts = [read(p) for p in Path(old['completion']).parent.glob('ATTEMPT_*.json')]
        assert len(attempts) == 1 and attempts[0]['status'] == 'COMPLETE'
        original_cost.append(dict(unit=name, job=prior['job'], seconds=prior['seconds'],
                                  device=prior['environment']['gpu_name'],
                                  driver=prior['environment']['precision']['nvidia_smi_name_driver'],
                                  attempts=attempts))
        old_dev = rows(Path(old['completion']).parent / 'dev/DEV.jsonl')
        expected = {(x['core'], x['scene']) for x in old_dev}
        assert len(expected) == 74 and len({s for _, s in expected}) == 72
        if dev_membership is not None:
            assert expected == dev_membership
        dev_membership = expected
        window_rows[name] = {}
        for budget in (3000, 30000):
            rr = rows(folder / f'dev_{budget}/DEV.jsonl')
            assert len(rr) == 740 and {(x['core'], x['scene']) for x in rr} == expected
            assert all(x['E']['COS']['n_grid'] == 10668 for x in rr)
            by_window = defaultdict(list)
            for x in rr:
                by_window[x['core']].append(x['E']['COS']['err_full'])
            window_rows[name][budget] = {
                c: dict(success9=sum(e < 9 for e in es), success42=sum(e < 42 for e in es),
                        failed100=sum(e >= 100 for e in es), median_err=st.median(es), n=len(es))
                for c, es in sorted(by_window.items())}
        for step in (10000, 20000, 30000):
            rr = rows(folder / f'cal_{step}/CAL.jsonl')
            assert len(rr) == 100 and len({x['core'] for x in rr}) == 10 and len({x['scene'] for x in rr}) == 9
            assert len({(x['core'], x['perturbation']) for x in rr}) == 100
            assert all(x['role'] == 'CAL' and set(x['E']) == {'COS'} and x['E']['COS']['n_grid'] == 10668 for x in rr)
            for c in {x['core'] for x in rr}:
                assert {x['perturbation'] for x in rr if x['core'] == c} == set(range(10))
            sig = [(x['core'], x['scene'], x['perturbation'], x['E']['COS']['theta'], x['E']['COS']['t']) for x in rr]
            if seed in cal_signatures:
                assert sig == cal_signatures[seed]
            cal_signatures[seed] = sig
            for k in ('success9', 'success42', 'median_err'):
                assert math.isclose(u['cal'][str(step)][k], r['cal'][str(step)]['COS'][k], abs_tol=1e-12)

    means, curves, hardware = {}, {}, {}
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
        for u in uu:
            hardware[f"{arm}_{u['seed']}"] = dict(device=u['environment']['gpu_name'],
                                                   driver=u['environment']['precision']['nvidia_smi_name_driver'],
                                                   host=u['environment']['host'])
    bf = read(HERE / 'ANALYSIS_LONG_BF_REVIEW_20261024.json')
    new_h = audit['cost']['receipt_gpu_hours']
    original_h = sum(x['seconds'] for x in original_cost) / 3600
    gates = bf['cost']['shared_gate_records']
    assert all(sha(g['path']) == g['sha256'] for g in gates)
    gate_h = sum(g['scheduler_seconds'] for g in gates)/3600
    total_new_h = new_h + bf['cost']['new_continuation_and_evaluation_receipt_gpu_hours']
    total_original_h = original_h + bf['cost']['original12_train_and_evaluation_receipts_gpu_hours']
    review = dict(status='COMPLETE_SERIES_REVIEW_PASS', utc=datetime.now(timezone.utc).isoformat(),
                  audit_path=str(source), audit_sha256=sha(source), seeds=SEEDS,
                  means=means, curves=curves, per_window_fixed_endpoints=window_rows,
                  hardware=hardware,
                  checks=dict(original3000_steps_and_initialization_paired=True,
                              original3000_batch_trace='Not saved in original trainer; source-state hashes, common frozen sampler and CPU gates are the evidence. No retrospective per-step equality claim.',
                              new27000_steps_lr_input_and_finite_checks='PASS in linked audit',
                              dev_all74_windows_72groups_10perturbations=True,
                              cal_all10_windows_9groups_10perturbations=True,
                              original_group_membership_and_fixed_grid10668=True,
                              each_original_and_continuation_one_successful_attempt=True,
                              role_evidence='Full run saved aggregate roles_read and frozen runtime assertion, not raw full guard trace; technical gates saved raw trace.'),
                  cost=dict(epfl_new_receipt_gpu_hours=new_h,
                            epfl_synchronized_training_gpu_hours=sum(x['compute_seconds'] for x in audit['cost']['records'])/3600,
                            epfl_original_train_and_evaluation_gpu_hours=original_h,
                            original_records=original_cost,
                            both_datasets_shared_long_gates_scheduler_gpu_hours=gate_h,
                            shared_gate_records=gates,
                            both_datasets24_new_receipt_gpu_hours=total_new_h,
                            both_datasets24_original_receipt_gpu_hours=total_original_h,
                            both_datasets_known_scoped_gpu_hours=total_new_h+total_original_h+gate_h,
                            note='37shared gate seconds counted once across BF/EPFL, including13seconds failedv1EPFL. Original receipts include CAL/DEV, not pure prefix training. New receipts exclude batch startup/teardown; unrelated original gates/diagnostics not included.'),
                  limitations=[
                      'ReusedDEV and3seeds; perturbations/windows are not independent replicates. Primary success is equal-group macro over72frozen groups, not an unweighted740trial mean.',
                      'NewBCE1002/NCE1002 useRTX; othersL40S. exact1002/BCE1001 onL40S595.84, otherL40S580.178.04; inherited3000prefix hardware also mixed.',
                      'Fixed short/long DEV evaluated samejob but pose search nondeterministic; continuation changes duration andLR decay together.',
                      'CAL strict fluctuates and all arms CAL medians rebound slightly20k→30k; no best-checkpoint substitution or inference of DEV convergence.',
                      'All windows retained. Severe-error counts describe geometry, not software failure. No TEST access.'],
                  decision=dict(registered_practical_signal=audit['registered_practical_signal'],
                                interpretation='Longer recipe reduces water tails for every arm, but exact VCS strict improvement is small and all3seeds trail BCE/NCE at30k; BFtail advantage overNCE does not generalize toEPFL.',
                                action='Finish48admitted noisy long runs with matched clean30k comparisons and protected retrieval6. No extra clean60k or readout grid now. Threshold is a reporting criterion, not permanent closure of a scientific hypothesis.'))
    out = HERE / 'ANALYSIS_LONG_EPFL_REVIEW_20261024.json'
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    with (HERE / 'ANALYSIS_LONG_EPFL_MEANS_20261024.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm','budget','group_macro_success9_percent','group_macro_success42_percent','mean_seed_median_error','mean_seed_p95_error','mean_seed_error_ge100_count'])
        for arm in ARMS:
            for end, budget in (('short',3000),('long',30000)):
                x = means[arm][end]
                w.writerow([arm,budget,100*x['success9'],100*x['success42'],x['median_err'],x['p95_err'],x['failed100']])
    print(json.dumps(dict(status=review['status'], path=str(out), epfl_new_gpu_hours=new_h,
                          epfl_original_gpu_hours=original_h, all24_new_gpu_hours=total_new_h,
                          both_datasets_known_scoped_gpu_hours=total_new_h+total_original_h+gate_h), ensure_ascii=False))


if __name__ == '__main__':
    main()
