"""Analysis of the N2M hinge-host lr_update 10 study. WAITING until all 6 new runs are COMPLETE; never reads TEST."""
import csv
import json
import statistics
from common import HERE, OUT, SEEDS, ARMS, PERIOD, EPOCHS, UPDATES_PER_EPOCH, read, write, sha, lr_vector, control, n2m

NCE_HOST_CONSTANT10_BEST = 472.9454306377383   # protected NCE host, constant + lr_update 10 (ANALYSIS_RETRIEVAL_LR_ALL6_20261025.json)


def readout(log):
    r = [e['dev']['rsum'] for e in log['epochs']]
    b = max(range(len(r)), key=lambda i: (r[i], -i))            # earliest epoch on ties
    return dict(best_rsum=r[b], best_epoch=b, last_rsum=r[-1], mean_last5_rsum=statistics.fmean(r[-5:]), curve=r)


def contrast(a, b):
    v = [a[s] - b[s] for s in SEEDS]
    return dict(values=v, mean=statistics.fmean(v), positive=sum(x > 0 for x in v), signal=statistics.fmean(v) >= 3 and sum(x > 0 for x in v) >= 2)


def main():
    done = {(s, a): OUT / 'runs' / f'{a}_s{s}' / 'COMPLETE.json' for s in SEEDS for a in ARMS}
    missing = [f'{a}_s{s}' for (s, a), p in done.items() if not p.exists()]
    if missing:
        print('WAITING', missing); return
    M = n2m(); runs, curves, checks = {}, [], {}
    for (s, a), p in done.items():
        c = read(p); d = p.parent; log = read(d / 'train_log.json')
        assert c['status'] == 'COMPLETE' and c['lr_update'] == PERIOD and sha(d / 'train_log.json') == c['artifact_sha256']['train_log.json']
        assert [e['steps_in_epoch'] for e in log['epochs']] == [UPDATES_PER_EPOCH] * EPOCHS and c['done']['global_step'] == 45320
        assert [e['lr'] for e in log['epochs']] == lr_vector(PERIOD)
        cd, cr, clog, _ = control(s, a)
        assert c['control']['full_result_sha256'] == sha(cd / 'full_result.json')
        assert not M['runtime'].environment_differences(cr['hardware'], c['environment'])
        runs[(s, a, PERIOD)] = dict(**readout(log), collapse=M['aggregate'].collapse_diagnostic(log)['severe_collapse'], host=c['host'], seconds=c['seconds'])
        runs[(s, a, 30)] = dict(**readout(clog), collapse=M['aggregate'].collapse_diagnostic(clog)['severe_collapse'], host=cr['hardware']['host'],
                                seconds=cr['attempts_summary']['attempt_wall_s_known'])
        pre = [e for e in c['epochs'] if e['epoch'] < PERIOD]
        checks[f'{a}_s{s}'] = dict(host=c['host'], control_host=c['control']['host'], initial_model_sha256=c['initial_model_sha256'],
                                   order_exact_all_epochs=all(e['order_exact'] for e in c['epochs']),
                                   pre_divergence_steps_exact=all(e['steps_exact'] for e in pre),
                                   pre_divergence_max_step_loss_rel_diff=max(e['max_step_loss_rel_diff'] for e in pre),
                                   pre_divergence_max_abs_dev_rsum_diff=max(abs(e['dev_rsum_diff']) for e in pre),
                                   grad_window_exact=c['grad_window_vs_control']['exact'])
    for s in SEEDS:   # paired arms share the seeded initial state
        assert checks[f'VCS_s{s}']['initial_model_sha256'] == checks[f'ConstantU_s{s}']['initial_model_sha256']
    for s in SEEDS:
        _, _, slog, _ = control(s, 'ShuffleVCS'); runs[(s, 'ShuffleVCS', 30)] = dict(**readout(slog), collapse=None, host=None, seconds=None)
    for (s, a, per), r in sorted(runs.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        for e, v in enumerate(r['curve']): curves.append(dict(seed=s, arm=a, lr_update=per, epoch_one_based=e + 1, lr=lr_vector(per)[e], dev_rsum=v))
    m = lambda arm, per, k='best_rsum': {s: runs[(s, arm, per)][k] for s in SEEDS}
    out = {}
    for k in ('best_rsum', 'last_rsum', 'mean_last5_rsum'):
        out[k] = {'VCS10-ConstantU10': contrast(m('VCS', 10, k), m('ConstantU', 10, k)),
                  'VCS10-VCS30': contrast(m('VCS', 10, k), m('VCS', 30, k)),
                  'ConstantU10-ConstantU30': contrast(m('ConstantU', 10, k), m('ConstantU', 30, k)),
                  'VCS30-ConstantU30': contrast(m('VCS', 30, k), m('ConstantU', 30, k))}
        inter = [out[k]['VCS10-ConstantU10']['values'][i] - out[k]['VCS30-ConstantU30']['values'][i] for i in range(3)]
        out[k]['interaction'] = dict(values=inter, mean=statistics.fmean(inter), positive=sum(x > 0 for x in inter))
    means = {f'{a}{per}': {k: statistics.fmean(runs[(s, a, per)][k] for s in SEEDS) for k in ('best_rsum', 'last_rsum', 'mean_last5_rsum')}
             for a, per in (('VCS', 10), ('ConstantU', 10), ('VCS', 30), ('ConstantU', 30), ('ShuffleVCS', 30))}
    gpuh = sum(runs[(s, a, PERIOD)]['seconds'] for s in SEEDS for a in ARMS) / 3600
    with (HERE / 'CURVES.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    result = dict(status='COMPLETE_ANALYSIS', primary='best full DEV rSum (earliest tie)', contrasts=out, means=means,
                  per_run={f'{a}{per}_s{s}': {k: v for k, v in r.items() if k != 'curve'} for (s, a, per), r in runs.items()},
                  main_signal=out['best_rsum']['VCS10-ConstantU10']['signal'], matching=checks,
                  descriptive_vs_nce_host_constant10_best=dict(value=NCE_HOST_CONSTANT10_BEST, VCS10_mean_best=means['VCS10']['best_rsum'],
                                                                note='different host and objective; not a matched contrast'),
                  new_run_receipt_gpu_hours=gpuh, curves_sha256=sha(HERE / 'CURVES.csv'),
                  limitations=['DEV reused across N2F/N2M/R22 retrieval studies (development evidence)', 'n=3 seeds; descriptive rule, no significance',
                               'lr_update 10 changes decay timing, number of decays and cumulative LR together',
                               'controls ran Oct 19; new runs on possibly different RTX nodes (same asserted environment); bitwise reproduction of epochs 0-9 reported per run'])
    write(HERE / 'ANALYSIS.json', result)
    print(json.dumps(dict(means=means, main=out['best_rsum']['VCS10-ConstantU10'], interaction=out['best_rsum']['interaction']), indent=1))


if __name__ == '__main__':
    main()
