"""Audit complete shuffled-U3 with six previously audited matched controls.

--check-ready verifies frozen prerequisites and matching, without interpreting
in-flight endpoints. The default requires all three COMPLETE receipts.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from analyze_protected_all_seeds import sha, close, RECALLS, SEEDS
import watch10m as w

PACK = w.BASE / 'handoff/r22_mechanism_review_20261024'
OLD = w.BASE / 'handoff/r22_other_apps_20261023'
CONTROL_REPORT = w.HERE / 'ANALYSIS_PROTECTED_ALL_SEEDS_20261024.json'
METRICS = ('best_rsum', 'last_rsum', 'mean_last5_rsum', 'mean_last10_rsum')


def prerequisites():
    frozen_counts = {}
    for package in (PACK, OLD):
        freeze = w.read(package / 'FREEZE.json')
        for rel, h in freeze['files'].items():
            assert sha(w.BASE / rel) == h, rel
        frozen_counts[package.name] = len(freeze['files'])
    fh = sha(PACK / 'FREEZE.json')
    cpu = w.read(PACK / 'checks/retrieval.json')
    assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == fh
    trace = PACK / 'checks/retrieval_read_trace.json'
    assert sha(trace) == cpu['trace_sha256']
    assert w.read(trace)['status'] == 'PASS' and not w.read(trace)['forbidden_hits']
    gatepath = PACK / 'gates/retrieval/COMPLETE.json'
    gate = w.read(gatepath)
    admission = w.read(PACK / 'ADMISSION.json')
    assert gate['status'] == 'COMPLETE' and gate['freeze_sha256'] == fh
    assert admission['gates']['retrieval'] == sha(gatepath) and 'retrieval_full' in admission['groups']
    trace = PACK / 'checks/retrieval_gpu_trace.json'
    assert sha(trace) == gate['trace_sha256']
    assert w.read(trace)['status'] == 'PASS' and not w.read(trace)['forbidden_hits']
    assert len(gate['rows']) == 2 and all(x['smoke'] for x in gate['rows'])
    for attr in ('initial_model_sha256', 'sample_order_sha256', 'first_batch_sha256', 'precision'):
        assert gate['rows'][0][attr] == gate['rows'][1][attr]
    controls = w.read(CONTROL_REPORT)
    assert controls['status'] == 'AUDIT_PASS' and controls['freeze_sha256'] == sha(OLD / 'FREEZE.json')
    receipts, manifests, weights = {}, {}, {}
    for seed in SEEDS:
        receipts[seed], manifests[seed] = {}, {}
        for method in ('vcs', 'constant'):
            p = OLD / 'runs' / f'retrieval_{method}_s{seed}'
            u = controls['units'][f'{method}_{seed}']
            assert sha(p / 'COMPLETE.json') == u['completion_sha256']
            r = w.read(p / 'COMPLETE.json')
            assert r['status'] == 'COMPLETE' and not r['smoke']
            assert sha(p / 'train_log.json') == r['train_log_sha256']
            assert sha(p / 'best.pt') == r['best_checkpoint_sha256']
            assert sha(p / f"OPEN_TRACE_{r['job']}.json") == u['trace_sha256']
            receipts[seed][method] = r
            manifests[seed][method] = w.read(p / 'run_manifest.json')
        real = receipts[seed]['vcs']['transformed_weight']
        item = cpu['weights'][str(seed)]
        assert sha(real['path']) == real['sha256'] == item['source_sha256']
        assert sha(item['path']) == item['sha256']
        a, b = w.read(real['path'])['weights'], w.read(item['path'])['weights']
        assert set(a) == set(b) and len(a) == 130500
        assert sorted(a.values()) == sorted(b.values()) and a != b
        assert all(math.isfinite(x) and .1 <= x <= .9 for x in b.values())
        close(mean(b.values()), .5)
        close(sum(b.values()) ** 2 / sum(x*x for x in b.values()), item['ess'])
        p = PACK / 'runs' / f'retrieval_shuffle_s{seed}'
        m = w.read(p / 'run_manifest.json')
        assert m['weight_lock_sha256'] == item['sha256']
        assert m['cfg'] == manifests[seed]['vcs']['cfg'] == manifests[seed]['constant']['cfg']
        assert m['objective']['tau'] == .05 and m['mean_v_full_pool'] == .55
        assert (m['n_pool'], m['n_A'], m['n_U']) == (145000, 14500, 130500)
        assert m['cfg']['num_epochs'] == 40 and m['cfg']['lr_update'] == 30
        weights[str(seed)] = dict(cpu_record=item, exact_multiset_preserved=True,
            changed_weight_value_count=sum(a[k] != b[k] for k in a),
            assignment_sha256=item['sha256'], original_sha256=real['sha256'])
    return fh, frozen_counts, gate, controls, receipts, weights


def audit_new(seed, fh, real, weight):
    p = PACK / 'runs' / f'retrieval_shuffle_s{seed}'
    r = w.read(p / 'COMPLETE.json')
    assert r['status'] == 'COMPLETE' and r['freeze_sha256'] == fh and not r['smoke']
    assert r['seed'] == seed and r['method'] == 'shuffle'
    assert r['pool_checks']['ok'] and r['log_checks']['ok']
    assert sha(p / 'train_log.json') == r['train_log_sha256']
    assert sha(p / 'best.pt') == r['best_checkpoint_sha256']
    trace = w.read(p / f"OPEN_TRACE_{r['job']}.json")
    assert trace['status'] == 'PASS' and not trace['forbidden_hits']
    assert r['transformed_weight'] == weight['cpu_record']
    rows = w.read(p / 'train_log.json')['epochs']
    assert [x['epoch'] for x in rows] == list(range(40))
    assert all(x['steps_in_epoch'] == 1133 for x in rows)
    for x in rows:
        assert math.isfinite(x['mean_loss']) and math.isfinite(x['dev']['rsum'])
        close(x['lr'], 2e-4 if x['epoch'] < 30 else 2e-5)
        close(x['dev']['rsum'], sum(x['dev'][k] for k in RECALLS))
    vals = [x['dev']['rsum'] for x in rows]
    ep = r['endpoints']
    assert ep['epochs'] == 40 and ep['best_epoch'] == vals.index(max(vals))
    close(ep['best_rsum'], max(vals)); close(ep['last_rsum'], vals[-1])
    close(ep['mean_last5_rsum'], mean(vals[-5:]))
    close(ep['best_rsum'], w.read(p / 'done.json')['best']['rsum'])
    for k in RECALLS:
        close(ep[k], rows[ep['best_epoch']]['dev'][k])
    attempts = [json.loads(x) for x in (p / 'attempts.jsonl').read_text().splitlines()]
    starts = [x for x in attempts if x['event'] == 'start']
    ends = [x for x in attempts if x['event'] == 'end']
    assert len(starts) == len(ends) == 1 and not starts[0]['resumes_from_last_pt']
    assert ends[0]['status'] == 'COMPLETE' and starts[0]['job_id'] == r['job']
    close(ends[0]['elapsed_s'], r['seconds'])
    submitted = [x for x in (w.POOL / 'SUBMITTED.tsv').read_text().splitlines() if f'\tr22_mech_shuffle_s{seed}\t' in x]
    assert len(submitted) == 1 and submitted[0].split('\t')[-1] == r['job']
    for attr in ('initial_model_sha256', 'sample_order_sha256', 'first_batch_sha256', 'precision'):
        assert r[attr] == real[attr], (seed, attr)
    for attr in ('gpu', 'torch', 'cuda', 'cudnn', 'python', 'numpy'):
        assert r['environment'][attr] == real['environment'][attr], (seed, attr)
    matching = dict(initialization=True, data_order=True, first8_batches=True, precision=True,
        gpu_model_software=True, driver_equal=r['environment']['driver'] == real['environment']['driver'],
        host_equal=r['environment']['host'] == real['environment']['host'])
    return dict(seed=seed, method='shuffle', endpoints={**ep, 'mean_last10_rsum':mean(vals[-10:])},
        curve_rsum=vals, curve_loss=[x['mean_loss'] for x in rows], best_epoch_one_based=ep['best_epoch']+1,
        job=r['job'], environment=r['environment'], precision=r['precision'], attempts=attempts,
        receipt_gpu_hours=r['seconds']/3600, completion_sha256=sha(p/'COMPLETE.json'),
        trace_sha256=sha(p/f"OPEN_TRACE_{r['job']}.json"), matching_to_real=matching)


def main(check_ready=False):
    missing = [s for s in SEEDS if not (PACK/'runs'/f'retrieval_shuffle_s{s}'/'COMPLETE.json').exists()]
    if missing and not check_ready:
        print(json.dumps(dict(status='WAITING_FOR_COMPLETE_SERIES', missing_seeds=missing)))
        return
    fh, counts, gate, old, receipts, weights = prerequisites()
    if check_ready:
        result = dict(status='PREREQUISITES_PASS', utc=w.utc(), freeze_sha256=fh,
            analyzer_sha256=sha(Path(__file__)), frozen_counts=counts,
            gate_job=gate['job'], weight_multisets_and_current_manifests_verified=True,
            reused_control_report_sha256=sha(CONTROL_REPORT), missing_seeds=missing,
            scope='No in-flight endpoints interpreted. Full audit still required after all3 COMPLETE.')
        w.write(w.HERE/'SHUFFLED_ANALYSIS_READY_20261025.json', result)
        print(json.dumps(result)); return
    units = {f'{m}_{s}':old['units'][f'{m}_{s}'] for s in SEEDS for m in ('vcs','constant')}
    for seed in SEEDS:
        units[f'shuffle_{seed}'] = audit_new(seed, fh, receipts[seed]['vcs'], weights[str(seed)])
    paired = {}
    for lhs, rhs in (('vcs','shuffle'), ('vcs','constant'), ('shuffle','constant')):
        paired[f'{lhs}_minus_{rhs}'] = {}
        for metric in METRICS:
            v = [units[f'{lhs}_{s}']['endpoints'][metric]-units[f'{rhs}_{s}']['endpoints'][metric] for s in SEEDS]
            paired[f'{lhs}_minus_{rhs}'][metric] = dict(seeds=list(SEEDS), differences=v,
                mean=mean(v), sd=stdev(v), positive=sum(x>0 for x in v))
    primary = paired['vcs_minus_shuffle']['mean_last5_rsum']
    result = dict(status='COMPLETE_AUDIT_PENDING_INTERPRETATION', utc=w.utc(),
        series='r22_mechanism_review_20261024:other_mechanism_retrieval', freeze_sha256=fh,
        analyzer_sha256=sha(Path(__file__)), frozen_counts=counts, units=units, weights=weights,
        means={m:{k:mean(units[f'{m}_{s}']['endpoints'][k] for s in SEEDS) for k in METRICS} for m in ('vcs','shuffle','constant')},
        paired=paired, criterion=dict(allocation_signal=primary['mean']>=3 and primary['positive']>=2,
            rule='mean real-shuffle last5>=3 rSum and >=2/3positive; no posthoc threshold changes'),
        cost=dict(new_three_receipt_gpu_hours=sum(units[f'shuffle_{s}']['receipt_gpu_hours'] for s in SEEDS),
            reused_six_receipt_gpu_hours=sum(units[f'{m}_{s}']['receipt_gpu_hours'] for s in SEEDS for m in ('vcs','constant')),
            shared_gate_seconds_counted_once=gate['seconds'], failed_or_resumed_full_attempts=0,
            scope='CPU, previous studies, scheduler/startup cost separate; no speed claim across nodes.'),
        limitations=['Three development seeds, reused DEV and selected pilot901; no TEST or confirmation.',
            'Last5 is the registered allocation endpoint; bestDEV and epoch40 retained, late stability is not selected-model improvement.',
            'One fixed weight permutation per seed; unchanged multiset separates allocation from distribution, not a VCS versus BCE advantage.',
            'Host/driver matches are disclosed individually; hardware/run variation remains possible. No raw-data access by this analyzer.'])
    w.write(w.HERE/'ANALYSIS_SHUFFLED_RETRIEVAL_ALL3_20261025.json',result)
    with (w.HERE/'SHUFFLED_RETRIEVAL_ALL9_CURVES_20261025.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=['seed','method','epoch_one_based','dev_rsum','fit_loss']);wr.writeheader()
        for u in units.values():
            for i,(score,loss) in enumerate(zip(u['curve_rsum'],u['curve_loss'])):
                wr.writerow(dict(seed=u['seed'],method=u['method'],epoch_one_based=i+1,dev_rsum=score,fit_loss=loss))
    print(json.dumps({k:result[k] for k in ('status','means','paired','criterion','cost')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-ready',action='store_true')
    main(parser.parse_args().check_ready)
