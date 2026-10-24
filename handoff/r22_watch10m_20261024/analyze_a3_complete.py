"""Read-only full A3 audit; writes only the resulting analysis, never training state."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs_v9_r22/vcsloss'
SEEDS = (901, 902, 903)
ARMS = ('NCE_VCSW', 'NCE_BCEW', 'NCE')
RECALLS = ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), (a, b)


def endpoints(rows):
    rs = [r['dev']['rsum'] for r in rows]
    best = max(range(len(rs)), key=lambda k: (rs[k], -k))
    return dict(best_rsum=rs[best], best_epoch=rows[best]['epoch'],
                last_rsum=rs[-1], mean_last5_rsum=mean(rs[-5:]),
                mean_last10_rsum=mean(rs[-10:]), epochs=len(rows),
                best_to_last=rs[-1]-rs[best],
                **{k: rows[best]['dev'][k] for k in RECALLS})


def main():
    summary, identity, hinge = [read(OUT / n) for n in ('SUMMARY.json', 'A3_IDENTITY.json', 'HINGE_IDENTITY.json')]
    assert summary['test_read'] is False and summary['dev_only']
    assert identity['status'] == 'REUSE_IDENTICAL_EXCEPT_HARDWARE'
    assert identity['code']['same']
    assert all(x['ok'] for x in identity['weights'].values())
    assert all(x['pre_ok'] and all(y['ok'] for y in x['post_run'].values()) for x in identity['nce_reuse'].values())
    units, records, manifests, curve = {}, {}, {}, []
    for seed in SEEDS:
        for arm in ARMS:
            key = f's{seed}__{arm}'
            folder = OUT / 'runs' / key
            record, manifest = read(folder / 'unit_result.json'), read(folder / 'run_manifest.json')
            records[key], manifests[key] = record, manifest
            assert record['status'] == 'COMPLETE_VERIFIED' and record['test_read'] is False
            assert record['pool_checks']['ok'] and record['log_checks']['ok']
            for name, digest in record['artifacts_sha256'].items():
                assert sha(folder / name) == digest, (key, name)
            for name, digest in record['inputs_sha256'].items():
                assert sha(record['inputs'][name]) == digest, (key, name)
            assert record['done']['manifest_sha256'] == sha(folder / 'run_manifest.json')
            rows = read(folder / 'train_log.json')['epochs']
            assert [r['epoch'] for r in rows] == list(range(40))
            assert all(r['steps_in_epoch'] == 1133 for r in rows)
            assert record['log_checks']['updates'] == 45320
            assert record['pool_checks']['last_batch_sizes'] == [104]
            assert record['done']['epochs_run'] == 40
            for row in rows:
                close(row['lr'], 2e-4 if row['epoch'] < 30 else 2e-5)
                assert math.isfinite(row['mean_loss'])
                assert all(math.isfinite(row['dev'][k]) for k in (*RECALLS, 'rsum'))
                close(row['dev']['rsum'], sum(row['dev'][k] for k in RECALLS))
                curve.append(dict(seed=seed, arm=arm, epoch_one_based=row['epoch']+1,
                                  lr=row['lr'], fit_mean_loss=row['mean_loss'], dev_rsum=row['dev']['rsum'],
                                  cumulative_seconds=row['seconds']))
            ep = endpoints(rows)
            close(ep['best_rsum'], record['done']['best']['rsum'])
            assert ep['best_epoch'] == record['done']['best']['epoch']
            for name in ('best_rsum', 'last_rsum', 'mean_last5_rsum', *RECALLS):
                close(ep[name], summary['units'][key][name])
            attempts = [json.loads(x) for x in (folder / 'attempts.jsonl').read_text().splitlines() if x.strip()]
            assert len(attempts) == 2 and [a['event'] for a in attempts] == ['start', 'end']
            assert attempts[1]['status'] == 'COMPLETE' and not attempts[0]['resumes_from_last_pt']
            assert attempts[0]['job_id'] == attempts[1]['job_id'] == record['environment']['job_id']
            traces = [(p, read(p)) for p in (OUT / 'traces').glob(f'run_s{seed}_{arm}_*.json')]
            traces = [(p, t) for p, t in traces if t.get('arm') == arm and t.get('seed') == seed]
            assert len(traces) == 1
            path, trace = traces[0]
            assert trace['status'] == 'PASS' and not trace['forbidden_hits']
            assert str(trace['job_id']) == record['environment']['job_id']
            units[key] = dict(seed=seed, arm=arm, directory=str(folder), **ep,
                              endpoint_epochs_one_based={str(e): rows[e-1]['dev']['rsum'] for e in (10,20,30,40)},
                              fit_loss_first=rows[0]['mean_loss'], fit_loss_last=rows[-1]['mean_loss'],
                              environment=record['environment'], source=record['source'],
                              initial_model_sha256=record['initial_model_sha256'],
                              data_order_sha256=record['data_order_sha256'],
                              input_hashes=record['inputs_sha256'], artifact_hashes=record['artifacts_sha256'],
                              manifest_sha256=sha(folder / 'run_manifest.json'),
                              record_sha256=sha(folder / 'unit_result.json'),
                              trace=dict(path=str(path), sha256=sha(path), status=trace['status']),
                              attempts=attempts, attempts_sha256=sha(folder / 'attempts.jsonl'),
                              attempt_seconds=attempts[1]['elapsed_s'], audit='PASS')
        ref = records[f's{seed}__NCE']
        mref = manifests[f's{seed}__NCE']
        for arm in ARMS[:2]:
            key = f's{seed}__{arm}'
            r, m = records[key], manifests[key]
            for k in ('initial_model_sha256', 'data_order_sha256', 'precision_effective'):
                assert r[k] == ref[k], (key, k)
            for k in ('cfg', 'seed', 'lr_mult', 'n_pool', 'n_A', 'n_U', 'formal'):
                assert m[k] == mref[k], (key, k)
            for k in ('python', 'torch', 'cuda', 'cudnn', 'numpy'):
                assert r['environment'][k] == ref['environment'][k]
            assert r['inputs_sha256']['observed'] == ref['inputs_sha256']['observed']
            assert r['inputs_sha256']['weights'] == identity['weights'][key]['sha256']
            assert r['pool_checks']['A_weight_exactly_one'] and r['pool_checks']['U_weights_equal_lock']
            close(m['mean_v_full_pool'], (m['n_A'] + m['sum_w_U']) / m['n_pool'])
            close(r['objective']['tau'], .05)
    # Recompute reused hinge endpoints from full existing DEV logs. No baseline model evaluation.
    for seed in SEEDS:
        for arm in ('HINGE_VCSW_N2M', 'HINGE_VCSW_N2F', 'HINGE_BCEW_N2F'):
            key = f's{seed}__{arm}'
            cached = summary['units'][key]
            folder = Path(cached['dir'])
            rows = read(folder / 'train_log.json')['epochs']
            assert len(rows) == 40 and [r['epoch'] for r in rows] == list(range(40))
            ep = endpoints(rows)
            for k in ('best_rsum', 'last_rsum', 'mean_last5_rsum', *RECALLS):
                close(ep[k], cached[k])
            units[key] = dict(seed=seed, arm=arm, directory=str(folder), **ep,
                              train_log_sha256=sha(folder / 'train_log.json'),
                              audit='ENDPOINTS_RECOMPUTED; inherited baseline identity report retained')
    pairs = []
    for old in summary['pairs_a3']:
        a, b = old['a'], old['b']
        per = {str(s): {m: units[f's{s}__{a}'][m] - units[f's{s}__{b}'][m]
                      for m in ('best_rsum', 'last_rsum', 'mean_last5_rsum', 'mean_last10_rsum', *RECALLS)} for s in SEEDS}
        for s, row in per.items():
            for k in old['per_seed'][s]:
                close(row[k], old['per_seed'][s][k])
        pairs.append(dict(a=a, b=b, per_seed=per,
                          means={k:mean(r[k] for r in per.values()) for k in next(iter(per.values()))},
                          positive_seeds={k:sum(r[k]>0 for r in per.values()) for k in next(iter(per.values()))}))
    means = {a: {m:mean(units[f's{s}__{a}'][m] for s in SEEDS)
                 for m in ('best_rsum', 'last_rsum', 'mean_last5_rsum', 'mean_last10_rsum')}
             for a in (*ARMS, 'HINGE_VCSW_N2M', 'HINGE_VCSW_N2F', 'HINGE_BCEW_N2F')}
    formal = [u for u in units.values() if u['arm'] in ARMS[:2]]
    nce = [u for u in units.values() if u['arm']=='NCE']
    report = dict(schema='r22-a3-complete-review-v1', utc=datetime.now(timezone.utc).isoformat(),
                  series='original_A3_all_seeds', status='PASS', dev_only=True, test_read=False,
                  primary='best full DEV rSum among 40 epochs, earliest tie; 0–600 scale',
                  unit_of_analysis='3 paired training seeds, reused DEV; not independent held-out confirmation',
                  units=units, means=means, pairs=pairs,
                  original_preregistered_rule=summary['a3_rule'],
                  identity_reports=dict(a3=identity, hinge=hinge),
                  audit_scope='All 9 A3/NCE unit records, input hashes, all artifact hashes including best/last checkpoint bytes, 40-epoch curves, 45320 updates, LR, initial/order/config/software/precision identities, 9 access traces, 9 attempt ledgers. Reused hinge DEV endpoints recomputed; prior identity reports retained, hinge checkpoints not reloaded.',
                  cost=dict(a3_six_successful_attempt_gpu_hours=sum(u['attempt_seconds'] for u in formal)/3600,
                            reused_nce_gpu_hours=sum(u['attempt_seconds'] for u in nce)/3600,
                            a3_full_training_attempts=6, a3_failed_or_unfinished_attempts=0,
                            excludes='Slurm startup overhead, inherited weight fitting and hinge training. Historical two CPU fixture-test failures retained, not formal GPU attempts; cancelled-before-start smoke has no GPU runtime.'),
                  limitations=['Every VCS/BCE seed pair uses different GPU model; drivers also differ. Matched seeds/inputs do not remove hardware confounding.',
                               'Best DEV is selected from 40 candidates; last/last5/last10 are secondary and cannot replace primary.',
                               'No confidence claim from three seeds; development set reused across design iterations.',
                               'VCS is used upstream to estimate positive-pair reliability weights; this is not direct VCS-QMI representation training. Weighting changes empirical positive measure and normalization.'],
                  conclusion='No consistent VCS-specific best-DEV advantage. Both reliability weights reduce late degradation versus unweighted NCE, while neither improves mean best DEV. Original final criteria both NOT_MET.',
                  next_decision='Continue already-frozen protected-weight retrieval seeds 902/903 and registration long24. Do not extend unchanged A3 merely to occupy GPUs: all 40-epoch runs are complete and best DEV precedes final epoch with later decline. A future budget/optimization experiment requires separately frozen matched controls; old threshold is not a permanent prohibition.')
    dest = HERE / 'ANALYSIS_A3_ALL_SEEDS_20261024.json'
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (HERE / 'A3_ALL_SEEDS_CURVES_20261024.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(curve[0]));writer.writeheader();writer.writerows(curve)
    print(json.dumps(dict(status='PASS', file=str(dest), means=means, cost=report['cost'],
                          best_epochs_one_based={k:u['best_epoch']+1 for k,u in units.items() if u['arm'] in ARMS},
                          pair_summaries=[{k:p[k] for k in ('a','b','means','positive_seeds')} for p in pairs]),ensure_ascii=False))


if __name__ == '__main__':
    main()
