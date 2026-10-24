"""Review the completed protected-weight pilot using existing FIT/DEV artifacts only."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import watch10m as w


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    base = w.BASE / 'handoff/r22_other_apps_20261023'
    frozen = w.read(base / 'FREEZE.json')
    assert all(sha(w.BASE / p) == h for p, h in frozen['files'].items())
    freeze = sha(base / 'FREEZE.json')
    protocol = w.read(base / 'PROTOCOL.json')['retrieval']
    cpu = w.read(base / 'RETRIEVAL_CPU_CHECK.json')
    assert cpu['status'] == 'PASS' and cpu['freeze_sha256'] == freeze
    ledger = (w.BASE / 'handoff/r22_task_diagnostics_20261023/SUBMITTED.tsv').read_text().splitlines()
    units, receipts, manifests, weights = {}, {}, {}, {}
    for method in ['vcs', 'bce', 'constant']:
        folder = base / 'runs' / f'retrieval_{method}_s901'
        r = w.read(folder / 'COMPLETE.json')
        assert r['status'] == 'COMPLETE' and r['freeze_sha256'] == freeze and not r['smoke']
        assert r['pool_checks']['ok'] and r['log_checks']['ok']
        assert sha(folder / 'train_log.json') == r['train_log_sha256']
        assert sha(folder / 'best.pt') == r['best_checkpoint_sha256']
        trace = w.read(folder / f"OPEN_TRACE_{r['job']}.json")
        assert trace['status'] == 'PASS' and not trace['forbidden_hits']
        log = w.read(folder / 'train_log.json')
        history = log['epochs']
        assert [x['epoch'] for x in history] == list(range(40))
        assert all(x['steps_in_epoch'] == 1133 for x in history)
        assert sum(x['steps_in_epoch'] for x in history) == 45320
        assert all(math.isfinite(x['mean_loss']) and math.isfinite(x['dev']['rsum']) for x in history)
        vals = [x['dev']['rsum'] for x in history]
        best = vals.index(max(vals))
        ep = r['endpoints']
        assert ep['epochs'] == 40 and ep['best_epoch'] == best
        assert ep['best_rsum'] == max(vals) == w.read(folder / 'done.json')['best']['rsum']
        assert abs(ep['mean_last5_rsum'] - st.mean(vals[-5:])) < 1e-10 and ep['last_rsum'] == vals[-1]
        recall_keys = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10']
        assert all(ep[k] == history[best]['dev'][k] for k in recall_keys)
        attempts = [json.loads(x) for x in (folder / 'attempts.jsonl').read_text().splitlines()]
        starts = [x for x in attempts if x['event'] == 'start']
        ends = [x for x in attempts if x['event'] == 'end']
        assert len(starts) == len(ends) == 1 and not starts[0]['resumes_from_last_pt']
        assert ends[0]['status'] == 'COMPLETE' and ends[0]['elapsed_s'] == r['seconds']
        submitted = [x for x in ledger if '\tr22_retrieval_' + method + '_s901\t' in x]
        assert len(submitted) == 1 and submitted[0].split('\t')[-1] == r['job']
        manifest = w.read(folder / 'run_manifest.json')
        assert manifest['cfg']['num_epochs'] == 40 and manifest['cfg']['lr_update'] == 30
        assert manifest['objective']['tau'] == .05 and manifest['mean_v_full_pool'] == .55
        assert manifest['n_pool'] == 145000 and manifest['n_U'] == 130500 and manifest['n_A'] == 14500
        meta = r['transformed_weight']
        assert sha(Path(meta['path'])) == meta['sha256'] == manifest['weight_lock_sha256']
        weights[method] = w.read(Path(meta['path']))
        ww = list(weights[method]['weights'].values())
        assert len(ww) == 130500 and abs(st.mean(ww) - .5) < 1e-12
        assert .1 <= min(ww) <= max(ww) <= .9
        assert abs(sum(ww) ** 2 / sum(x*x for x in ww) - meta['ess']) < 1e-5
        units[method] = dict(endpoints=ep, best_epoch_one_based=best + 1, curve_rsum=vals,
            mean_last10_rsum=st.mean(vals[-10:]), final_recalls={k: history[-1]['dev'][k] for k in recall_keys},
            last5_recalls={k: st.mean(x['dev'][k] for x in history[-5:]) for k in recall_keys},
            job=r['job'], environment=r['environment'], precision=r['precision'],
            receipt_gpu_hours=r['seconds']/3600, known_submissions=submitted, attempts=attempts,
            completion_sha256=sha(folder / 'COMPLETE.json'), best_checkpoint_sha256_verified=True,
            trace_status='PASS', trace_scope=trace['scope'], weights=meta,
            pool_checks=r['pool_checks'], log_checks=r['log_checks'])
        receipts[method], manifests[method] = r, manifest
    for attr in ['initial_model_sha256', 'sample_order_sha256', 'first_batch_sha256', 'precision']:
        assert len({json.dumps(r[attr], sort_keys=True) for r in receipts.values()}) == 1
    assert len({json.dumps(m['cfg'], sort_keys=True) for m in manifests.values()}) == 1
    assert len({tuple(sorted(x['weights'])) for x in weights.values()}) == 1
    assert weights['vcs']['source_sha256']['observed'] == weights['bce']['source_sha256']['observed']
    for method in ['vcs', 'bce']:
        for key, path in weights[method]['source'].items():
            assert sha(Path(path)) == weights[method]['source_sha256'][key]
    ids = sorted(weights['vcs']['weights'])
    a, b = weights['vcs']['weights'], weights['bce']['weights']
    difference = [abs(a[i] - b[i]) for i in ids]
    ordering_equal = sorted(ids, key=a.get) == sorted(ids, key=b.get)
    weight_audit = dict(mean_abs_difference=st.mean(difference), max_abs_difference=max(difference),
                        equal_transformed_stable_order=ordering_equal,
                        original_equal_order=cpu['weights']['901']['equal_stable_order'])
    baseline_folder = w.BASE / 'outputs_v9_r22/vcsloss/runs/s901__NCE'
    baseline = w.read(baseline_folder / 'unit_result.json')
    assert baseline['status'] == 'COMPLETE_VERIFIED' and not baseline['test_read']
    for name in ['train_log.json', 'run_manifest.json', 'done.json']:
        assert sha(baseline_folder / name) == baseline['artifacts_sha256'][name]
    baseline_best = baseline['done']['best']['rsum']
    scores = {m: u['endpoints']['best_rsum'] for m, u in units.items()}
    gate = scores['vcs'] >= max(scores['bce'], scores['constant']) + 3 and scores['vcs'] >= baseline_best
    paired = {m: {k: units['vcs']['endpoints'][k] - units[m]['endpoints'][k]
                  for k in ['best_rsum', 'last_rsum', 'mean_last5_rsum']}
              for m in ['bce', 'constant']}
    result = dict(utc=w.utc(), series='r22_other_apps_20261023:other_retrieval', status='COMPLETE_REVIEWED',
        label='Single seed901 PILOT/DEVELOPMENT;40epoch full schedule; existing FIT/DEV only',
        freeze_sha256=freeze, frozen_files_verified=len(frozen['files']), protocol=protocol,
        units=units, paired_vcs_minus_control=paired, weight_audit=weight_audit,
        matching=dict(initialization=True, sample_order=True, first8_batches=True, architecture_schedule=True,
            observed_pairs=True, transformed_weight_ids=True, all40epochs_45320updates=True,
            VCS_constant_gpu_and_driver=True, VCS_BCE_gpu_and_driver=False),
        unweighted_reference=dict(best_rsum=baseline_best, gpu=baseline['environment']['gpu'],
            vcs_difference=scores['vcs'] - baseline_best, role='Historical registered gate reference; hardware differs from protected VCS'),
        gate=dict(eligible=gate, required_gain_over_both=3, require_unweighted=True),
        receipt_gpu_hours=sum(u['receipt_gpu_hours'] for u in units.values()),
        cost_scope='Three successful first attempts, no known failed training attempts. Runner receipts include setup inside train_one; unmeasured scheduler startup, shared weight fitting, original NCE and technical gate costs are separate. Not an objective speed comparison across hardware.',
        interpretation=[
            'The frozen expansion gate fails: protected VCS loses on best DEV to both controls and unweighted NCE.',
            'VCS beats constant weights late in training, but BCE also does; improved late stability does not establish a better selected model or VCS-specific advantage.',
            'VCS and BCE weights share ordering and differ by only about0.004 on average. This is a calibration-shape comparison, not evidence for distinct noise identification.',
            'VCS versus constant uses the same GPU model/driver. BCE and historical NCE are L40S, VCS is RTX; do not attribute VCS-BCE differences solely to the objective.',
            'All best checkpoints are early; curves do not support blind extension beyond40epochs. Single seed and reused DEV are not independent confirmation.'],
        decision=dict(action='STOP_THIS_PROTECTED_RECIPE', new_gpu_jobs=0, extend_seeds902_903=False,
            new_floor_or_mean_grid=False, longer_training=False,
            continue_work='Finish original A3/NCR and dense matched registration. Require a new theoretical/task hypothesis before a new application batch.'))
    assert not gate
    out = w.HERE / 'ANALYSIS_PROTECTED_RETRIEVAL_20261024.json'
    w.write(out, result)
    lines = ['检索保护权重三臂完整分析：seed901，40轮，DEV开发证据。', '',
             '248个冻结文件、三臂初始化/输入顺序/前八批/训练配置/权重锁/历史与模型哈希通过；每臂45320更新。访问轨迹PASS（Python文件审计范围）。']
    for m, u in units.items():
        ep=u['endpoints']; lines.append(f"{m}: 最佳rSum {ep['best_rsum']:.4f}（第{u['best_epoch_one_based']}轮），最后{ep['last_rsum']:.4f}，末5轮{ep['mean_last5_rsum']:.4f}；回执{u['receipt_gpu_hours']:.3f}GPUh；{u['environment']['gpu']}。")
    for m, diffs in paired.items():
        lines.append(f'VCS−{m}: ' + ', '.join(f'{k} {v:+.4f}' for k,v in diffs.items()))
    lines += [f"不加权NCE参考{baseline_best:.4f}，VCS差{scores['vcs']-baseline_best:+.4f}。",
              f"VCS/BCE权重平均绝对差{weight_audit['mean_abs_difference']:.6f}，最大{weight_audit['max_abs_difference']:.6f}；变换后排序一致={ordering_equal}。", '',
              '结论：未过原扩展门槛，不交902/903六项，不另开floor/mean网格或更长训练。VCS后期比constant稳定，但BCE也有类似行为；最佳模型没有额外收益。',
              '硬件限制：VCS与constant同型RTX/同驱动，BCE为L40S；VCS−BCE不能作纯目标函数因果差异。单seed和DEV选择亦限制结论。',
              f"三臂回执合计{result['receipt_gpu_hours']:.4f}GPUh，三项各一次成功尝试；门禁/旧权重拟合/不加权NCE/未测调度启动成本另计。",
              '下一步：原A3/NCR和配准完整对照继续；新任务必须回答新的任务机制问题，不能为了维持排队数量重复已完成实验。']
    out.with_suffix('.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scores=scores, paired=paired, gate=gate, weights=weight_audit,
                         receipt_gpu_hours=result['receipt_gpu_hours']), ensure_ascii=False))


if __name__ == '__main__':
    main()
