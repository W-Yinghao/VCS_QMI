"""Read-only supplemental review of the complete frozen BF40 continuation series."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PACK = BASE / 'handoff/r22_geometry_noise_long_20261024'
ARMS = ('vcs_exact', 'vcs_perm', 'bce', 'nce')
SEEDS = (1001, 1002, 1003)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(rr):
    return [(r['core'], r['scene'], r['perturbation'], r['E']['COS']['theta'],
             r['E']['COS']['t'], r['E']['COS']['n_grid']) for r in rr]


def main():
    audit_path = HERE / 'ANALYSIS_NOISE_LONG_N40_BF_20261024.json'
    audit = read(audit_path)
    assert audit['status'] == 'AUDIT_PASS' and audit['dataset'] == 'bf' and audit['noise'] == .4
    assert len(audit['units']) == 12
    reference_dev = {}
    reference_cal = {}
    tails = {}
    hardware = {}
    attempts = {}
    strict_by_core = {}
    inputs_checked = []
    for arm in ARMS:
        for seed in SEEDS:
            key = f'{arm}_{seed}'
            p = PACK / 'runs' / f'bf_{key}_n40'
            result = read(p / 'COMPLETE.json')
            assert result['identity'] == dict(ds='bf', arm=arm, seed=seed, noise=.4,
                                              freeze_sha256=audit['freeze_sha256'])
            long = rows(p / 'dev_30000/DEV.jsonl')
            short = rows(p / 'dev_3000/DEV.jsonl')
            assert len(long) == len(short) == 320
            assert len({r['core'] for r in long}) == 32
            sig = identities(long)
            assert sig == identities(short)
            if seed in reference_dev:
                assert sig == reference_dev[seed]
            reference_dev[seed] = sig
            for directory in ('r22_geometry_long_v2_20261024', 'r22_geometry_noise_long_20261024'):
                unit = f'bf_{key}' + ('_n20' if 'noise' in directory else '')
                comparator = BASE / 'handoff' / directory / 'runs' / unit / 'dev_30000/DEV.jsonl'
                assert sig == identities(rows(comparator)), (key, directory)
            for step in (10000, 20000, 30000):
                rr = rows(p / f'cal_{step}/CAL.jsonl')
                assert len(rr) == 80 and len({r['core'] for r in rr}) == 8
                assert all(r['role'] == 'CAL' and set(r['E']) == {'COS'} for r in rr)
                cs = identities(rr)
                if seed in reference_cal:
                    assert reference_cal[seed] == cs
                reference_cal[seed] = cs
            tail = {}
            for label, rr in [('short', short), ('long', long)]:
                tail[label] = {str(threshold):dict(Counter(
                    r['core'] for r in rr if r['E']['COS']['err_full'] >= threshold))
                    for threshold in (42, 100)}
                assert sum(tail[label]['100'].values()) == audit['units'][key][label]['failed100']
            tails[key] = tail
            strict_by_core[key] = {core:sum(r['E']['COS']['err_full'] < 9 for r in long if r['core'] == core)
                                   for core in sorted({r['core'] for r in long})}
            local_attempts = [read(q) for q in p.glob('ATTEMPT_*.json')]
            assert len(local_attempts) == 1 and local_attempts[0]['status'] == 'COMPLETE'
            assert local_attempts[0]['job'] == result['job']
            # Frozen run.py samples the attempt clock after environment capture and
            # writing COMPLETE.json, so the final attempt includes that overhead.
            receipt_overhead = local_attempts[0]['seconds'] - result['seconds']
            assert receipt_overhead >= 0
            prefix_dir = Path(result['source']['completion']).parent
            prefix_attempts = [read(q) for q in prefix_dir.glob('ATTEMPT_*.json')]
            assert all(a['status'] == 'COMPLETE' for a in prefix_attempts)
            attempts[key] = dict(new_attempts=local_attempts, prefix_attempt_records=prefix_attempts,
                                 completion_record_write_overhead_seconds=receipt_overhead)
            env = result['environment']
            hardware[key] = dict(gpu=env['gpu_name'], host=env['host'],
                                 driver=env['precision']['nvidia_smi_name_driver'])
            assert env['gpu_name'] == 'NVIDIA L40S'
            inputs_checked.append(key)

    reports = {
        'clean':read(HERE / 'ANALYSIS_LONG_BF_20261024.json'),
        'n20':read(HERE / 'ANALYSIS_NOISE_LONG_N20_BF_20261024.json'),
        'n40':audit,
    }
    table = []
    for rate, report in reports.items():
        assert report['status'] == 'AUDIT_PASS'
        for arm in ARMS:
            values = [report['units'][f'{arm}_{s}']['long'] for s in SEEDS]
            table.append(dict(condition=rate, arm=arm, seeds=3,
                              strict_percent=100*statistics.mean(v['success9'] for v in values),
                              loose_percent=100*statistics.mean(v['success42'] for v in values),
                              mean_seed_median_error=statistics.mean(v['median_err'] for v in values),
                              mean_seed_p95_error=statistics.mean(v['p95_err'] for v in values),
                              total_failed100=sum(v['failed100'] for v in values)))
    csv_path = HERE / 'BF_CLEAN_NOISE_LONG_ALL_RATES_20261024.csv'
    with csv_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    gate_records = {ds:read(PACK / 'gates' / ds / 'COMPLETE.json') for ds in ('bf','epfl')}
    review = dict(status='COMPLETE_REVIEW', utc=datetime.now(timezone.utc).isoformat(),
        audit_sha256=sha(audit_path), reviewer_sha256=sha(Path(__file__)),
        verified_complete_units=inputs_checked, paired_cal_dev_and_clean_n20_identity=True,
        expected_dev_per_unit=320, expected_cal_per_step=80, realized_noise=51/128,
        reviewer_check_correction='Initial review incorrectly required identical ATTEMPT/COMPLETE wall times. Frozen run.py lines207-216 samples them sequentially, with environment capture and receipt write in between. Nonnegative overhead is recorded per unit; no experiment, metric or frozen code changed.',
        tail_failures_by_core=tails, strict_success_counts_by_core=strict_by_core,
        hardware=hardware, attempts=attempts, all_rates_fixed30k_means=table,
        shared_gates=dict(seconds=sum(r['seconds'] for r in gate_records.values()),
                          note='Shared once across the four dataset/rate series; not charged again to each series.'),
        cost_scope='New12 continuation/evaluation receipts6.28578276846144 GPUh; source3000 prefixes2.2096215849452547 GPUh separately. CPU, gate and scheduler overhead are not silently folded into a full-history claim.',
        conclusion='Long training improves mean exact performance, but exact loses strict to BCE on all3 seeds and loses mean loose too. Versus NCE the smaller loose-tail benefit coexists with lower strict performance. No unique robustness at40% mismatch.',
        mechanism_boundary='No causal attribution to a specific J term from these endpoint contrasts; separate frozen term diagnostics and A1 remain the mechanism tests.',
        next='Complete all frozen EPFL40, retrieval allocation, KD and A1 work unchanged; no extra longer-training, dense-sampling or critic grid. The old practical signal remains descriptive, never a permanent stop.')
    out = HERE / 'ANALYSIS_NOISE_LONG_N40_BF_REVIEW_20261024.json'
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=review['status'], units=len(inputs_checked), path=str(out),
                         gate_seconds=review['shared_gates']['seconds'], table=table), ensure_ascii=False))


if __name__ == '__main__':
    main()
