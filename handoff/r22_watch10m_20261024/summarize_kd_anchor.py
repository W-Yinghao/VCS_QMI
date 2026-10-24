"""Aggregate a fully reviewed three-seed KD anchor; reads existing CAL records only."""
import argparse
import json
import statistics
from pathlib import Path

import analyze_kd_seed as audit
import watch10m as w


def summarize(anchor):
    protocol = w.read(w.BASE / 'handoff/r22_kd_long_validation_20261024/PROTOCOL.json')
    frozen = w.read(w.BASE / 'handoff/r22_kd_long_validation_20261024/FREEZE.json')['files']
    jobs = w.read(w.BASE / 'handoff/r22_kd_long_validation_20261024/JOBS.json')
    jobs = [j for j in jobs if j['anchor'] == anchor]
    assert len(jobs) == 12
    ledger = (w.BASE / 'handoff/r22_task_diagnostics_20261023/SUBMITTED.tsv').read_text().splitlines()
    reports, report_evidence, units, identities = {}, {}, {}, {}
    metrics = ['final_top1_percent', 'last10_mean_percent', 'best_top1_percent']
    for seed in protocol['seeds']:
        path = w.HERE / f'ANALYSIS_KD_SEED{seed}_20261024.json'
        if not path.exists():
            path = w.HERE / f'ANALYSIS_KD_{anchor.upper()}_S{seed}_20261024.json'
        reports[seed] = w.read(path)
        assert reports[seed]['seed'] == seed
        report_evidence[str(seed)] = dict(path=str(path), sha256=audit.sha(path))
    for job in jobs:
        folder = Path(job['completion']).parent
        complete, run = w.read(folder / 'COMPLETE.json'), w.read(folder / 'run.json')
        assert complete['status'] == 'COMPLETE' and complete['freeze_sha256'] == job['freeze_sha256']
        assert run['status'] == 'complete' and run['epochs_completed'] == 240
        assert run['train_data_role'] == 'FIT' and run['selection_data_role'] == 'CAL'
        assert run['teacher_frozen_verified'] and not run['test_loaded_during_train']
        cfg_path = w.BASE / 'handoff/r22_other_apps_20261023/configs' / (job['config_name'] + '.json')
        assert audit.sha(cfg_path) == complete['config_sha256'] == frozen[str(cfg_path.relative_to(w.BASE))]
        cfg = w.read(cfg_path)
        for field in ['root', 'manifest']:
            cfg['data'][field] = str(Path(cfg['data'][field]).expanduser().resolve())
        for who in ['teacher', 'student']:
            for field in ['source_root', 'checkpoint', 'provenance']:
                if cfg[who].get(field):
                    cfg[who][field] = str(Path(cfg[who][field]).expanduser().resolve())
        assert cfg == run['config'] == w.read(folder / 'resolved_config.json')
        canonical = audit.hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert canonical == run['config_sha256']
        assert audit.sha(folder / 'history.json') == complete['history_sha256']
        assert audit.sha(folder / 'best_cal_predictions.csv') == complete['predictions_sha256']
        assert audit.sha(folder / 'teacher_cal_predictions.csv') == complete['teacher_predictions_sha256']
        history = w.read(folder / 'history.json')
        assert [x['epoch'] for x in history] == list(range(240))
        key = anchor + ':' + job['method']
        unit = reports[job['seed']]['units'][key]
        actual = [history[-1]['cal']['top1'] * 100,
                  statistics.mean(x['cal']['top1'] for x in history[-10:]) * 100,
                  max(x['cal']['top1'] for x in history) * 100]
        assert all(abs(unit[m] - value) < 1e-8 for m, value in zip(metrics, actual))
        assert unit['job'] == complete['job'] and unit['gpu'] == complete['gpu']
        attempts = [line.split('\t') for line in ledger if '\t' + job['name'] + '\t' in line]
        assert len(attempts) == 1 and attempts[0][-1] == complete['job'], 'Review additional attempts before aggregation'
        assert unit['method'] == job['method']
        units[job['config_name']] = dict(unit, seed=job['seed'],
            known_submissions=attempts, config_and_role_audit=True)
        identities.setdefault(job['seed'], []).append({
            k: complete[k] for k in ['student_initial_sha256', 'sample_order_sha256', 'gpu']}
            | {k: run[k] for k in ['source_sha256', 'split_sha256']})
    for seed, records in identities.items():
        assert len(records) == 4
        for key in records[0]:
            assert len({r[key] for r in records}) == 1, (seed, key)
    method_means = {
        method: {metric: dict(mean=statistics.mean(u[metric] for u in units.values() if u['method'] == method),
                             sd=statistics.stdev(u[metric] for u in units.values() if u['method'] == method))
                 for metric in metrics}
        for method in protocol['arms']}
    paired = {}
    for control in ['none', 'cosine', 'bce']:
        comparison = 'vcs-minus-' + control
        paired[comparison] = {}
        for metric in metrics:
            values = {str(seed): reports[seed]['primary_differences_pp'][anchor][comparison][metric]
                      for seed in protocol['seeds']}
            paired[comparison][metric] = dict(per_seed_pp=values, mean_pp=statistics.mean(values.values()),
                sd_pp=statistics.stdev(values.values()), positive_seeds=sum(v > 0 for v in values.values()))
    practical_gain = all(p['final_top1_percent']['mean_pp'] >= .2 and
                         p['final_top1_percent']['positive_seeds'] >= 2 and
                         p['last10_mean_percent']['mean_pp'] > 0 for p in paired.values())
    result = dict(utc=w.utc(), anchor=anchor, expected=12, completed=12,
        label='All three seeds for this anchor; exploratory CAL development evidence',
        primary=protocol['primary'], protocol_interpretation=protocol['interpretation'],
        reviewed_seed_reports=report_evidence, units=units, matching=identities,
        method_means=method_means, paired_differences_pp=paired,
        original_practical_gain_criterion_met=practical_gain,
        complete_run_receipt_gpu_hours=sum(u['seconds'] for u in units.values()) / 3600,
        cost_scope='Each unit has one known pool submission. Receipt times exclude unmeasured scheduler/startup overhead and prior gates/pilot; not total allocated GPU cost. Hardware differs across seeds; per-method wall times are not a causal speed comparison.',
        curves={str(seed): {method: reports[seed]['units'][anchor + ':' + method]['curve_top1_percent']
                            for method in protocol['arms']} for seed in protocol['seeds']},
        best_checkpoint_transfer={str(seed): reports[seed]['best_checkpoint_transfer'][anchor] for seed in protocol['seeds']},
        transfer_scope='Secondary CAL-selected best checkpoint predictions, not epoch240. No causal teacher-transfer claim.',
        limitations=protocol['limitations'],
        decision='Finish all24 registered units unchanged. Do not start a longer run or new lambda grid on this anchor; assess the complete CE/DKD evidence and remaining application series before allocating a new batch.')
    out = w.HERE / f'ANALYSIS_KD_{anchor.upper()}_ALL3_20261024.json'
    w.write(out, result)
    lines = [f'KD {anchor.upper()}三seed固定240轮验证：12/12完成（整体状态见24项汇总）', '',
             '各seed内配置/初始化/数据顺序/GPU型号匹配，历史与预测哈希、FIT/CAL角色、教师冻结通过；每臂一次已知提交。']
    for method, values in method_means.items():
        lines.append(f"{method}: 第240轮均值{values['final_top1_percent']['mean']:.3f}%，末10轮{values['last10_mean_percent']['mean']:.3f}%，最佳点{values['best_top1_percent']['mean']:.3f}%。")
    for comparison, p in paired.items():
        main = p['final_top1_percent']
        lines.append(f"{comparison}: 主指标均值{main['mean_pp']:+.3f}pp，seed间SD {main['sd_pp']:.3f}pp，{main['positive_seeds']}/3为正；末10轮{p['last10_mean_percent']['mean_pp']:+.3f}pp，最佳点{p['best_top1_percent']['mean_pp']:+.3f}pp。")
    lines += ['', f'原登记的实用增益判据通过：{practical_gain}。这不是显著性检验，也不是理论整体的结论。',
              '旧短pilot与长验证同时改变seed和学习率计划，不能将差异只归因于训练时长。结论以冻结变量、λ和实现为条件；不读TEST。',
              f"完成回执计时合计{result['complete_run_receipt_gpu_hours']:.4f} GPUh；不含全部调度启动成本、旧门禁/pilot，不能用不同节点的墙钟时长宣称目标函数加速。",
              '决策：继续完成全部24项，不改λ、不早停；本anchor不追加更长训练或λ网格。']
    out.with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(anchor=anchor, means=method_means, paired=paired,
                         criterion_met=practical_gain, receipt_gpu_hours=result['complete_run_receipt_gpu_hours']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', choices=['ce', 'dkd'], required=True)
    summarize(parser.parse_args().anchor)
