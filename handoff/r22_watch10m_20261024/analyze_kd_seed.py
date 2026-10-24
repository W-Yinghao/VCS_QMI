"""Review a complete KD seed or anchor quartet using existing CAL records only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import watch10m as w


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(path):
    result = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            ident, label, pred = int(row['id']), int(row['label']), int(row['prediction'])
            correct = bool(int(row['correct']))
            prob = json.loads(row['probabilities'])
            assert ident not in result and len(prob) == 100
            assert all(math.isfinite(x) and x >= 0 for x in prob)
            assert abs(sum(prob) - 1) < 1e-4 and correct == (label == pred)
            assert pred == max(range(100), key=prob.__getitem__)
            result[ident] = dict(label=label, prediction=pred, correct=correct,
                probability_margin=prob[label] - max(x for i, x in enumerate(prob) if i != label))
    assert len(result) == 5000
    return result


def compare_predictions(baseline, student, teacher):
    assert baseline.keys() == student.keys() == teacher.keys()
    groups = {}
    for tc, bc, name in [(True, False, 'teacher_correct_anchor_wrong'),
                         (False, True, 'teacher_wrong_anchor_correct'),
                         (True, True, 'both_correct'), (False, False, 'both_wrong')]:
        ids = [i for i in baseline if teacher[i]['correct'] == tc and baseline[i]['correct'] == bc]
        repaired = sum(not baseline[i]['correct'] and student[i]['correct'] for i in ids)
        damaged = sum(baseline[i]['correct'] and not student[i]['correct'] for i in ids)
        groups[name] = dict(n=len(ids), repaired=repaired, damaged=damaged,
            mean_probability_margin_change=statistics.mean(student[i]['probability_margin'] - baseline[i]['probability_margin'] for i in ids) if ids else None,
            new_wrong_matching_teacher=sum(baseline[i]['correct'] and not student[i]['correct'] and
                student[i]['prediction'] == teacher[i]['prediction'] for i in ids))
    repaired = sum(x['repaired'] for x in groups.values())
    damaged = sum(x['damaged'] for x in groups.values())
    net = sum(x['correct'] for x in student.values()) - sum(x['correct'] for x in baseline.values())
    assert repaired - damaged == net
    return dict(repaired=repaired, damaged=damaged, net_correct=net, net_accuracy_pp=net / 50, groups=groups)


def analyze(seed, anchor=None, write_report=True):
    anchors = [anchor] if anchor else ['ce', 'dkd']
    jobs = w.read(w.BASE / 'handoff/r22_kd_long_validation_20261024/JOBS.json')
    jobs = [j for j in jobs if j['seed'] == seed and j['anchor'] in anchors]
    assert len(jobs) == 4 * len(anchors)
    units, receipts, preds, teachers = {}, {}, {}, {}
    for job in jobs:
        folder = Path(job['completion']).parent
        r = w.read(folder / 'COMPLETE.json')
        assert r['status'] == 'COMPLETE' and r['freeze_sha256'] == job['freeze_sha256']
        assert sha(folder / 'history.json') == r['history_sha256']
        history = w.read(folder / 'history.json')
        assert len(history) == r['epochs'] == 240 and [x['epoch'] for x in history] == list(range(240))
        assert sha(folder / 'best_cal_predictions.csv') == r['predictions_sha256']
        assert sha(folder / 'teacher_cal_predictions.csv') == r['teacher_predictions_sha256']
        key = job['anchor'] + ':' + job['method']
        preds[key] = predictions(folder / 'best_cal_predictions.csv')
        teachers[key] = predictions(folder / 'teacher_cal_predictions.csv')
        best = max(x['cal']['top1'] for x in history)
        assert abs(sum(x['correct'] for x in preds[key].values()) / 5000 - best) < 1e-6
        assert preds[key].keys() == teachers[key].keys()
        assert all(preds[key][i]['label'] == teachers[key][i]['label'] for i in preds[key])
        receipts[key] = r
        units[key] = dict(anchor=job['anchor'], method=job['method'], weight=job['weight'],
            final_top1_percent=history[-1]['cal']['top1'] * 100,
            last10_mean_percent=statistics.mean(x['cal']['top1'] for x in history[-10:]) * 100,
            best_top1_percent=best * 100, final_nll=history[-1]['cal']['nll'],
            curve_top1_percent={str(e): history[e - 1]['cal']['top1'] * 100 for e in [40, 80, 120, 150, 180, 210, 240]},
            gpu=r['gpu'], seconds=r['seconds'], job=r['job'], hashes_verified=True,
            best_prediction_accuracy_percent=sum(x['correct'] for x in preds[key].values()) / 50)
    primary, transfer = {}, {}
    for anchor in anchors:
        keys = [anchor + ':' + m for m in ['none', 'vcs', 'cosine', 'bce']]
        for attr in ['student_initial_sha256', 'sample_order_sha256', 'gpu']:
            assert len({receipts[k][attr] for k in keys}) == 1
        teacher = teachers[keys[0]]
        assert all(teachers[k].keys() == teacher.keys() for k in keys)
        assert all(teachers[k][i]['prediction'] == teacher[i]['prediction'] and
                   teachers[k][i]['label'] == teacher[i]['label'] for k in keys for i in teacher)
        primary[anchor], transfer[anchor] = {}, {}
        for method in ['none', 'cosine', 'bce']:
            primary[anchor]['vcs-minus-' + method] = {metric: units[anchor + ':vcs'][metric] - units[anchor + ':' + method][metric]
                for metric in ['final_top1_percent', 'last10_mean_percent', 'best_top1_percent']}
        for method in ['vcs', 'cosine', 'bce']:
            key = anchor + ':' + method
            assert all(preds[key][i]['label'] == preds[keys[0]][i]['label'] for i in teacher)
            transfer[anchor][method] = compare_predictions(preds[keys[0]], preds[key], teacher)
    interpretation = []
    for anchor in anchors:
        differences = primary[anchor]
        for method, diff in differences.items():
            interpretation.append(
                f"{anchor} {method}: epoch240 {diff['final_top1_percent']:+.3f}pp; "
                f"last10 {diff['last10_mean_percent']:+.3f}pp; best {diff['best_top1_percent']:+.3f}pp.")
        curve_v = units[anchor + ':vcs']['curve_top1_percent']
        curve_b = units[anchor + ':none']['curve_top1_percent']
        interpretation.append(
            f"{anchor} VCS-minus-anchor within the long schedule: epoch40 "
            f"{curve_v['40'] - curve_b['40']:+.3f}pp, epoch240 {curve_v['240'] - curve_b['240']:+.3f}pp.")
    interpretation += [
        'All endpoints above are descriptive. Retain fixed epoch240 as primary; do not choose a favorable endpoint.',
        'The separate short pilot uses another seed and LR schedule; differences cannot be attributed to duration alone.',
        'Best-checkpoint teacher transfer counts are descriptive and secondary; they cannot establish the mechanism of the fixed240 result.']
    record = dict(utc=w.utc(), seed=seed, anchors=anchors,
        label='One-seed component of the registered three-seed review, ' + '/'.join(anchors) + '; development CAL only',
        units=units, primary_differences_pp=primary,
        matching='Within each anchor: same initialization, data order, GPU model, fixed240 epochs; metadata and CAL prediction hashes verified.',
        best_checkpoint_transfer=transfer,
        transfer_scope='Secondary analysis of CAL-selected best checkpoints, not epoch240. Probability margin is p(true)-max p(other); no causal claim about teacher influence.',
        complete_run_gpu_hours=sum(u['seconds'] for u in units.values()) / 3600,
        cost_scope='Completed run receipt seconds; gate and prior pilot costs are separate.',
        interpretation=interpretation,
        decision='Complete all24 registered units; do not select lambda again, stop arms early, or add a new grid based on this one seed.')
    stem = f'ANALYSIS_KD_{anchors[0].upper()}_S{seed}' if len(anchors) == 1 else f'ANALYSIS_KD_SEED{seed}'
    out = w.HERE / f'{stem}_20261024.json'
    lines = [f"KD seed{seed} {'/'.join(anchors)}完整{len(units)}臂分析（整体状态见24项汇总）", '',
             '各anchor内同GPU、同初始化、同数据顺序、固定240轮，历史与预测文件哈希通过。']
    for key, u in units.items():
        lines.append(f"{key}: 第240轮CAL={u['final_top1_percent']:.2f}%；末10轮均值={u['last10_mean_percent']:.3f}%；最佳={u['best_top1_percent']:.2f}%。")
    for anchor in anchors:
        for method, diff in primary[anchor].items():
            lines.append(f"{anchor} {method}: 第240轮{diff['final_top1_percent']:+.3f}pp，末10轮{diff['last10_mean_percent']:+.3f}pp，最佳{diff['best_top1_percent']:+.3f}pp。")
        cv = units[anchor + ':vcs']['curve_top1_percent']
        cb = units[anchor + ':none']['curve_top1_percent']
        lines.append(f"{anchor}长训练轨迹中VCS−anchor：第40轮{cv['40'] - cb['40']:+.3f}pp，第240轮{cv['240'] - cb['240']:+.3f}pp。")
    lines += ['', '上述均为开发集阶段证据。保持固定240轮为主指标，不挑有利终点；独立短pilot还改变seed和学习率计划，不能据此归因于时长。', '',
              '以下仅为各臂CAL选中最佳checkpoint的辅助预测分析，非第240轮：']
    for anchor, methods in transfer.items():
        for method, t in methods.items():
            group = t['groups']['teacher_correct_anchor_wrong']
            harm = t['groups']['teacher_wrong_anchor_correct']
            lines.append(f"{anchor}+{method}: 共修正{t['repaired']}、新增错误{t['damaged']}、净{t['net_correct']:+d}例；教师对/anchor错的{group['n']}例中修正{group['repaired']}；教师错/anchor对的{harm['n']}例中新增错误{harm['damaged']}（其中{harm['new_wrong_matching_teacher']}例转为教师的错误类别）。")
    lines += ['', '决策：继续完成全部24项，保持原λ、主指标和停止条件。本seed仅提供阶段证据，暂不另开网格。']
    if write_report:
        w.write(out, record)
        out.with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(seed=seed, primary=primary,
        transfer={a: {m: {k: t[k] for k in ['repaired', 'damaged', 'net_correct']} for m, t in methods.items()} for a, methods in transfer.items()},
        complete_run_gpu_hours=record['complete_run_gpu_hours']), ensure_ascii=False))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--anchor', choices=['ce', 'dkd'])
    args = parser.parse_args()
    analyze(args.seed, args.anchor)
