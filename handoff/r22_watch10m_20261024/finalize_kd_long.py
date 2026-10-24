"""Audit and combine all 24 registered long runs; existing FIT/CAL records only."""
import hashlib
import json
import statistics as st
from pathlib import Path

import watch10m as w


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bundle = w.BASE / 'handoff/r22_kd_long_validation_20261024'
    protocol = w.read(bundle / 'PROTOCOL.json')
    jobs = w.read(bundle / 'JOBS.json')
    assert len(jobs) == 24
    freeze_checks = {}
    for rel in ['handoff/r22_kd_long_validation_20261024/FREEZE.json',
                'handoff/r22_other_apps_20261023/FREEZE.json']:
        path = w.BASE / rel
        frozen = w.read(path)['files']
        assert all(sha(w.BASE / name) == digest for name, digest in frozen.items())
        freeze_checks[rel] = dict(sha256=sha(path), files_verified=len(frozen))
    summaries = {a: w.read(w.HERE / f'ANALYSIS_KD_{a.upper()}_ALL3_20261024.json')
                 for a in protocol['anchors']}
    units, curves, probes, shared_identity = {}, {}, {}, {}
    for anchor, summary in summaries.items():
        assert summary['completed'] == summary['expected'] == 12
        assert not summary['original_practical_gain_criterion_met']
        for name, unit in summary['units'].items():
            folder = w.BASE / 'handoff/r22_other_apps_20261023/runs' / name
            run = w.read(folder / 'run.json')
            receipt = w.read(folder / 'COMPLETE.json')
            history = w.read(folder / 'history.json')
            assert receipt['status'] == 'COMPLETE' and run['epochs_completed'] == 240
            assert sha(folder / 'history.json') == receipt['history_sha256']
            assert len(unit['known_submissions']) == 1
            assert all(row['fit_n_seen'] == 44992 for row in history)
            assert run['config']['loss']['decision']['weight'] == protocol['lambdas'][anchor][unit['method']]
            assert run['config']['optimizer']['milestones'] == protocol['milestones']
            identity = dict(teacher=run['teacher_initial_state_sha256'],
                            student=receipt['student_initial_sha256'],
                            order=receipt['sample_order_sha256'],
                            source=run['source_sha256'], split=run['split_sha256'],
                            environment=run['environment'], gpu=receipt['gpu'])
            shared_identity.setdefault(str(unit['seed']), []).append(identity)
            units[name] = dict(unit, complete_sha256=sha(folder / 'COMPLETE.json'),
                               fit_examples_per_epoch=44992, optimizer_steps=703 * 240)
            if unit['method'] != 'none':
                late = history[-10:]
                probes[name] = {
                    'last_batch_mean_positive_cosine': st.mean(h['dependence']['positive_cosine'] for h in late),
                    'last_batch_mean_abs_T_ge099': st.mean(h['dependence']['positive_abs_T_ge099'] for h in late),
                    'first_batch_mean_aux_over_anchor': st.mean(h['gradient_probe']['decision_space']['aux_over_anchor'] for h in late),
                    'first_batch_mean_gradient_cosine': st.mean(h['gradient_probe']['decision_space']['cosine'] for h in late)}
        curves[anchor] = {
            'vcs-minus-' + control: {
                e: st.mean(s['vcs'][e] - s[control][e] for s in summary['curves'].values())
                for e in ['40', '80', '120', '150', '180', '210', '240']}
            for control in ['none', 'cosine', 'bce']}
    assert set(units) == {j['config_name'] for j in jobs}
    for seed, identities in shared_identity.items():
        assert len(identities) == 8 and all(x == identities[0] for x in identities)
    pilot_jobs = [j for j in w.read(w.BASE / 'handoff/r22_other_apps_20261023/PILOT_JOBS.json') if j['app'] == 'kd']
    assert len(pilot_jobs) == 20
    pilot_seconds = sum(w.read(Path(j['completion']))['seconds'] for j in pilot_jobs)
    cost = dict(long_24_receipt_gpu_hours=sum(u['seconds'] for u in units.values()) / 3600,
                prior_20_pilot_receipt_gpu_hours=pilot_seconds / 3600,
                known_long_submissions=24, known_long_failed_attempts=0,
                prior_failed_gate_job='1061572', prior_repaired_gate_job='1061590',
                prior_gate_failure='CPU input / CUDA model in dimension probe; zero optimizer steps. Separate gate wrapper repaired before training admission.',
                gate_elapsed_seconds=None,
                limitations='Receipt timers exclude unmeasured scheduler/startup overhead; gate elapsed times unavailable. This is not total allocated research GPU time. Different nodes/models prevent a causal speed comparison.')
    evidence = {a: dict(path=str(w.HERE / f'ANALYSIS_KD_{a.upper()}_ALL3_20261024.json'),
                        sha256=sha(w.HERE / f'ANALYSIS_KD_{a.upper()}_ALL3_20261024.json'))
                for a in protocol['anchors']}
    result = dict(utc=w.utc(), status='COMPLETE_REVIEWED', expected=24, completed=24,
        series='r22_kd_long_validation_20261024:other_kd_full',
        label='Exploratory FIT/CAL development evidence; fixed epoch240 primary',
        protocol=str(bundle / 'PROTOCOL.json'), freeze_checks=freeze_checks,
        source_reports=evidence, units=units, within_seed_all8_identity_verified=shared_identity,
        method_means={a: s['method_means'] for a, s in summaries.items()},
        paired_differences_pp={a: s['paired_differences_pp'] for a, s in summaries.items()},
        practical_gain_criterion_met={a: s['original_practical_gain_criterion_met'] for a, s in summaries.items()},
        curve_mean_differences_pp=curves,
        best_checkpoint_transfer={a: s['best_checkpoint_transfer'] for a, s in summaries.items()},
        transfer_scope='CAL-selected best checkpoints, not epoch240. Repaired/damaged counts and probability margins are descriptive, not causal teacher-transfer effects.',
        last10_sparse_diagnostics=probes,
        diagnostic_scope='Dependence fields are each epoch final FIT batch; gradient fields are first FIT batch. Ten probes do not represent full-dataset distributions or establish gradient conflict/saturation.',
        cost=cost, limitations=protocol['limitations'] + [
            'Same seed uses the same GPU model/environment but not necessarily the same node; no bitwise training reproducibility claim.',
            'CE VCS and BCE have weight10; DKD VCS/cosine use10 and BCE uses1, selected by the same short pilot. DKD comparison is between frozen selected recipes, not solely the objective at equal weight.',
            'Cosine uses positive matching only; VCS/BCE use the same off-diagonal cross-instance reference. This comparison cannot attribute the gap solely to the reference term.',
            'No free tanh critic or inner fitting in this decision-coordinate recipe; optimizing those cannot be assumed to resolve this result.'],
        conclusions_cn=[
            '完整长验证未建立VCS独有收益。CE下依赖/匹配辅助项均有益，但VCS弱于余弦，与BCE接近；DKD下VCS三个seed均弱于anchor和两个辅助对照。',
            '固定240轮与末10轮同向；最佳点也没有改变均值结论。原实用增益判据两个anchor均未通过。',
            '长训练是必要的验证：早期排名可反转；但240轮后当前接法没有翻盘，不支持继续以训练不足为由追加预算。',
            '不否定理论本身；结论限于该教师/学生、CIFAR100划分、中心化归一化logits、固定λ与训练配方。'],
        decisions=dict(new_gpu_jobs=0, kd_recipe='Lower priority; no automatic longer training or lambda grid.',
            current_queue='Continue registered dense registration and retrieval/NCR series through the sole existing pool.',
            registration='Wait for matched dense48 task/curve/cost results; objective microbenchmark alone is insufficient for longer training.',
            retrieval='Finish protected three-arm40epoch comparison and original A3/NCR; apply frozen gates only after completeness.',
            future_kd='Require a new task-specific hypothesis with matched controls and a frozen finite budget before any new batch; no generic saturation or critic-width explanation is established here.'))
    out = w.HERE / 'ANALYSIS_KD_LONG_ALL24_20261024.json'
    w.write(out, result)
    lines = ['KD长验证完整结论：24/24，全部固定240轮、3seed、FIT/CAL。', '',
             '两个冻结包共282个文件核验通过。每seed八臂的初始化、输入顺序、教师、source/split、环境和GPU型号一致；每项168720更新，λ与学习率计划未变。',
             '主指标为固定240轮；每seed内配对后均值和SD，以下单位为百分比/百分点。']
    for anchor, s in summaries.items():
        lines += ['', anchor.upper()]
        for method, m in s['method_means'].items():
            lines.append(f"{method}: final {m['final_top1_percent']['mean']:.4f}%; last10 {m['last10_mean_percent']['mean']:.4f}%; best {m['best_top1_percent']['mean']:.4f}%.")
        for comparison, p in s['paired_differences_pp'].items():
            main_metric = p['final_top1_percent']
            lines.append(f"{comparison}: final {main_metric['mean_pp']:+.4f}pp, SD {main_metric['sd_pp']:.4f}, positive {main_metric['positive_seeds']}/3; per seed {main_metric['per_seed_pp']}; last10 {p['last10_mean_percent']['mean_pp']:+.4f}pp; best {p['best_top1_percent']['mean_pp']:+.4f}pp.")
        lines.append('原实用增益判据通过：False；这是描述性资源判据，不是显著性检验。')
        for control, values in curves[anchor].items():
            lines.append(f'第40/80/120/150/180/210/240轮 {control}: ' + ', '.join(f'{x:+.3f}' for x in values.values()) + ' pp。')
    lines += ['', '辅助预测分析（各臂CAL选择的最佳checkpoint，不是第240轮）：']
    for anchor, s in summaries.items():
        for method in ['vcs', 'cosine', 'bce']:
            values = [s['best_checkpoint_transfer'][str(seed)][method] for seed in protocol['seeds']]
            lines.append(f"{anchor}+{method}: 三seed平均修正{st.mean(v['repaired'] for v in values):.1f}例、新增错误{st.mean(v['damaged'] for v in values):.1f}例、净{st.mean(v['net_correct'] for v in values):+.1f}例/5000；不作因果机制判断。")
    lines += ['', '诊断保留每轮末batch依赖统计、首batch梯度；不是全训练分布，不能据此声称普遍饱和或梯度冲突。',
              f"24项回执计时{cost['long_24_receipt_gpu_hours']:.4f} GPUh；旧20项短pilot另计{cost['prior_20_pilot_receipt_gpu_hours']:.4f} GPUh。24项各一次已知提交、无已知失败尝试。旧门禁1061572设备不匹配失败，1061590修复通过，失败日志保留、耗时未知；回执不含完整调度/启动成本。", '',
              *result['conclusions_cn'], '',
              '决策：当前KD接法降优先级，不追加更长训练或λ网格，不改理论。继续唯一补位池中已登记的配准和retrieval/NCR；新批次须有任务机制、匹配对照和有限预算。',
              '局限：全为CAL开发证据；短pilot和长验证的seed/LR计划不同，不能隔离时长效应。DKD的BCE与VCS冻结λ不同；结论针对所选配方。三个seed不能证明小差值的统计优势。']
    out.with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(out), completed=24, cost=cost,
                         curve_mean_differences_pp=curves), ensure_ascii=False))


if __name__ == '__main__':
    main()
