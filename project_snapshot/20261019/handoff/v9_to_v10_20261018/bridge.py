"""Close out the frozen V9 matrix before a dependent V10 GPU allocation starts.

CPU Slurm only. No training, changes to scientific recipes, or V9 holds/cancels.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

BUNDLE = Path(__file__).resolve().parent
PROJECT = BUNDLE.parents[1]
V9 = PROJECT / 'handoff/v9_full_20261017'
V10 = PROJECT / 'handoff/v10_bf1_20261017'
V9_FREEZE = 'ae8d1e55d7ec5ae9439950d776a2c5efa122de45a58490aadcde0f6046fe1907'
V10_FREEZE = 'a03e88277ba61159d4cd70255d72bf13515b8cac9843465329f6f9bcb15f58a2'
SOURCE_COMMIT = 'b8f59ea825dc7a2064457d47a628bef09ca4e185'


def utc():
    return time.strftime('%FT%TZ', time.gmtime())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(args, timeout=45):
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f'{args[0]} failed: {p.stderr.strip()}')
    return p.stdout


def parse_queue(stdout):
    rows = []
    for line in stdout.splitlines():
        parts = line.strip().split('|')
        require(len(parts) == 8, 'Malformed squeue output')
        rows.append(dict(zip(('id', 'name', 'state', 'qos', 'tres', 'start', 'elapsed', 'reason'), parts)))
    return rows


def queue():
    return parse_queue(command(['squeue', '-u', 'yinwang', '-r', '-h', '-o', '%i|%j|%T|%q|%b|%S|%M|%R']))


def verify_bundle():
    for rel, value in read(BUNDLE / 'FREEZE.json')['files'].items():
        require(sha(BUNDLE / rel) == value, f'Bridge changed: {rel}')


def verify_science(check_data=False):
    require(sha(V9 / 'FROZEN.json') == V9_FREEZE, 'V9 freeze changed')
    require(sha(V10 / 'FROZEN.json') == V10_FREEZE, 'V10 freeze changed')
    v9, v10 = read(V9 / 'FROZEN.json'), read(V10 / 'FROZEN.json')
    for name, value in v9['code_sha256'].items():
        require(sha(V9 / name) == value, f'V9 scientific file changed: {name}')
    require(v10['source_commit'] == SOURCE_COMMIT, 'V10 source commit changed')
    for name, value in v10['source_files'].items():
        require(sha(V10 / 'source' / name) == value, f'V10 scientific file changed: {name}')
    for name, value in v10['run_inputs'].items():
        require(sha(V10 / name) == value, f'V10 run input changed: {name}')
    require((PROJECT / 'ssl_pilot_v9/reports/V9/V9_N2_units/wave1_paused.json').exists(), 'Original wave1 pause removed')
    if check_data:
        for name, value in v9['input_sha256'].items():
            require(sha(name) == value, f'V9 frozen input changed: {name}')
        for name, identity in v9['data_files'].items():
            st = Path(name).stat()
            require(st.st_size == identity['size'] and st.st_mtime_ns == identity['mtime_ns'], f'V9 data stat changed: {name}')
        lock = read(V10 / 'DATA_LOCK.json')
        require(lock['status'] == 'PASS' and not lock['held_test_read'], 'V10 data lock invalid')
        require(sha(lock['r6_manifest']['path']) == lock['r6_manifest']['sha256'], 'V10 role manifest changed')
        for item in lock['files']:
            require(item['role'] in ('FIT', 'CAL', 'DEV'), 'Held data must not be read')
            require(sha(item['path']) == item['sha256'], f'V10 active data changed: {item["path"]}')
    return {'status': 'PASS', 'utc': utc(), 'v9_freeze': V9_FREEZE, 'v10_freeze': V10_FREEZE,
            'v9_code_files': len(v9['code_sha256']), 'v10_source_files': len(v10['source_files']),
            'data_checked': check_data}


def readiness(rows, groups, launch, completed_groups, finished):
    known = {str(x['job']) for x in launch.values()} | {'1042382', '1045056'}
    live = [r for r in rows if r['id'] in known or r['name'].startswith('v9')]
    missing = [g['id'] for g in groups if g['id'] not in completed_groups]
    reasons = []
    if len(groups) != 18 or sum(len(g['units']) for g in groups) != 33:
        reasons.append('FROZEN_MATRIX_SIZE_MISMATCH')
    if set(launch) != {g['id'] for g in groups}:
        reasons.append('NOT_ALL_GROUPS_DISPATCHED')
    if missing:
        reasons.append('V9_GROUPS_INCOMPLETE')
    if live:
        reasons.append('V9_JOBS_STILL_LIVE')
    if not finished:
        reasons.append('FEEDER_NOT_FINISHED')
    elif finished.get('failed_groups') or finished.get('full_complete') != 33:
        reasons.append('FEEDER_FAILURE_OR_INCOMPLETE')
    return {'ready': not reasons, 'reasons': reasons, 'missing_groups': missing, 'live_v9': live}


def verify_unit(path, group, unit, binary=True):
    result = read(path / 'full_result.json')
    ident = result['identity']
    expected = {'group': group['id'], 'seed': group['seed'], **unit,
                'epochs': 40, 'freeze_sha256': V9_FREEZE}
    require(all(ident.get(k) == v for k, v in expected.items()), f'Unit identity mismatch: {path}')
    require(ident == read(path / 'identity.json'), f'Identity artifact mismatch: {path}')
    require(result.get('test_read') is False and result['full_epochs'] == 40, f'Invalid data role/length: {path}')
    names = ('best.pt', 'last.pt', 'train_log.json', 'run_manifest.json', 'identity.json')
    for name in names:
        if binary or not name.endswith('.pt'):
            require(sha(path / name) == result['artifact_sha256'][name], f'Artifact hash mismatch: {path / name}')
    require(result['done']['manifest_sha256'] == result['artifact_sha256']['run_manifest.json'], 'Done manifest mismatch')
    log = read(path / 'train_log.json')
    epochs = log['epochs']
    require([e['epoch'] for e in epochs] == list(range(40)), f'Incomplete epochs: {path}')
    require(all(math.isfinite(e['dev']['rsum']) for e in epochs), f'Nonfinite metric: {path}')
    best = max(epochs, key=lambda e: e['dev']['rsum'])
    require(best['epoch'] == result['done']['best']['epoch'] and abs(best['dev']['rsum'] - result['done']['best']['rsum']) < 1e-8,
            f'Checkpoint selection mismatch: {path}')
    # Each resumed fit resets its elapsed clock. Sum completed segments, not just the last attempt.
    boundaries = [0] + log.get('resumed_at', []) + [40]
    require(boundaries == sorted(boundaries) and all(0 < b <= 40 for b in boundaries[1:]), f'Invalid resume boundaries: {path}')
    measured = sum(epochs[end - 1]['seconds'] for start, end in zip(boundaries, boundaries[1:-1]) if end > start) + result['done']['seconds']
    return {'group': group['id'], 'seed': group['seed'], **unit, 'best_dev_rsum': best['dev']['rsum'],
            'best_epoch': best['epoch'], 'last_dev_rsum': epochs[-1]['dev']['rsum'],
            'gpu': result['hardware']['gpu'], 'host': result['hardware']['host'],
            'measured_training_seconds': measured, 'resumed_at': log.get('resumed_at', []),
            'full_result_sha256': sha(path / 'full_result.json')}


def load_summarizer():
    sys.path.insert(0, str(V9))
    spec = importlib.util.spec_from_file_location('frozen_v9_summarizer', V9 / 'summarize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_report(audit):
    report = audit['readout']
    lines = ['# V9 N2F 全量收尾与阶段研究结论', '', f"生成时间：{audit['utc']}。由CPU收尾作业生成，基于33次完整40 epoch训练；只使用完整DEV，未读取TEST。", '',
             '## 完整性与范围', '', '18组/33次结果、身份、训练日志、best/last checkpoint哈希和最佳DEV选择均通过核验。历史作业1042382已正常退出；原wave1暂停保留，未扩展V9网格。', '',
             'N1/G1/G2的既有结论保留在根目录PROJECT_MEMORY_CN.md；这里关闭本轮N2F，不把旧调参结果并入三seed矩阵。', '',
             '## 三seed结果', '', '| 方法/尺度 | seed901 | seed902 | seed903 | 均值 ± 样本标准差 | 严重崩塌次数 |', '|---|---:|---:|---:|---:|---:|']
    methods = ('VCS_Weight','BCE_Weight','RankMass','Unweighted_A','RuLSIF_Weight','OracleMask','OracleDrop')
    for scale in ('sum', 'mean'):
        for method in methods:
            rows = sorted([r for r in report['units'] if r['method'] == method and r['scale'] == scale], key=lambda r: r['seed'])
            if not rows:
                continue
            require(len(rows) == 3, 'Report requires all three seeds')
            values = [r['best_dev_rsum'] for r in rows]
            lines.append(f"| {method}/{scale.upper()} | " + ' | '.join(f'{x:.2f}' for x in values) +
                         f" | {statistics.mean(values):.2f} ± {statistics.stdev(values):.2f} | {sum(r['severe_collapse'] for r in rows)}/3 |")
    lines += ['', '## 配对差、尺度与Oracle', '', '| 比较 | seed901差 | seed902差 | seed903差 | 平均差 | 正方向seed数 |', '|---|---:|---:|---:|---:|---:|']
    for key, contrast in report['contrasts'].items():
        require(contrast['n'] == 3, f'Incomplete contrast: {key}')
        lines.append(f'| {key} | ' + ' | '.join(f"{p['delta']:.2f}" for p in contrast['pairs']) +
                     f" | {contrast['mean']:.2f} | {contrast['positive']}/3 |")
    decision = report['decision']
    lines += ['', '## 研究判断', '',
              'SUM为预先固定主尺度，MEAN为预定同方法同GPU敏感性配对；终末GPU/节点和续跑记录见CLOSEOUT.json。若中断续跑更换硬件，相应尺度差保留该混杂，不能声称严格同GPU。方法之间也可能使用不同型号GPU。以上三seed差值为描述性结果，不作统计显著性或独立TEST确认主张。', '',
              ('VCS相对BCE、RankMass、RuLSIF均达到原预设资源阈值（平均至少+3、至少2/3同方向且崩塌不更多）。这支持进一步独立确认的研究价值，仍不证明CS理论特有优势。'
               if decision['vcs_specific_confirmation_worth_testing'] else
               'VCS未同时达到相对BCE、RankMass、RuLSIF的原预设资源阈值。当前不建立稳定的VCS独有优势，不追加本路线网格。'), '',
              f"通用加权资源阈值：SUM={decision['generic_weighting_gate']['sum']}，MEAN={decision['generic_weighting_gate']['mean']}。阈值要求相对不加权平均至少+5、至少2/3同方向且崩塌不更多。", '',
              ('两个尺度均未通过通用加权阈值，停止当前full in-batch weighting路线扩展。' if decision['stop_current_route'] else
               '至少一个尺度支持通用加权的实际收益；按用户决定，保留此结论并结束本轮，不自动扩大V9。'), '',
              'OracleMask/Drop使用额外对应真值，不能作为可部署对照或严格上界。Drop同时改变负样本池、样本量和每epoch更新数，差值不能单独定位机制。', '',
              '理论等式S−J(T)=E_M[(T−eta)^2]刻画posterior contrast拟合误差；匹配BCE共享无限制最优解。它不直接保证依赖学生表示及batch内困难负样本的完整检索风险更低。后续转向已冻结V10的结构化表示、精确矩与优化比较，不根据这些DEV结果修改V10网格。', '',
              '## 失败、崩塌和成本', '',
              f"严重崩塌{sum(r['severe_collapse'] for r in report['units'])}/33；未解决失败0。调度续跑记录与各单位resumed_at保存在CLOSEOUT.json。", '',
              f"实际训练计时累计（含可恢复的已完成段）{audit['cost']['measured_training_gpu_hours']:.3f} GPU小时。它不包含所有初始化、被中断但尚未落盘的epoch及调度开销，不能冒充完整allocated GPU时长。", '',
              '控制器观察到的作业起始时间与SCHED_EXIT退出记录另列于CLOSEOUT.json。sacct会计服务当前不可用，完整分配时长存在缺口，明确保留该成本限制，不以估算补造实测值。', '',
              '## 执行后续', '', 'V9收尾核验通过后，本CPU作业退出0，Slurm释放依赖它的V10 GPU gate；gate成功后再运行冻结的四族数组。TEST134继续HOLD。']
    (BUNDLE / 'V9_CLOSEOUT_CN.md').write_text('\n'.join(lines) + '\n')


def audit_all():
    verify_science(check_data=True)
    legacy = PROJECT / 'slurm_logs/v9n2_t17_1042382.out'
    require(legacy.read_text().strip().endswith('exit=0 end=2026-10-17T19:38:21Z'), 'Legacy V9 completion is not verified')
    groups = read(V9 / 'GROUPS.json')
    launch = read(V9 / 'LAUNCH.json')
    units, exits = [], {}
    for group in groups:
        gd = V9 / 'results' / group['id']
        done = read(gd / 'DONE.json')
        require(done['freeze_sha256'] == V9_FREEZE, 'Group freeze mismatch')
        expected = {str(gd / f"{u['method']}_{u['scale']}" / 'full_result.json') for u in group['units']}
        require(set(done['units']) == expected, 'Group unit manifest mismatch')
        require(not (gd / 'FAILED.json').exists(), 'Unresolved scientific failure')
        exit_rec = read(gd / 'SCHED_EXIT.json')
        require(exit_rec['exit_code'] == 0 and exit_rec['job_id'] == launch[group['id']]['job'], 'Final attempt exit not successful')
        exits[group['id']] = exit_rec
        for unit in group['units']:
            units.append(verify_unit(gd / f"{unit['method']}_{unit['scale']}", group, unit))
    readout = load_summarizer().summarize()
    require(len(units) == 33 and not readout['failures'], 'Incomplete or failed matrix')
    require(readout['decision']['status'] == 'FULL_DEV_RESOURCE_DECISION', 'Scientific readout incomplete')
    observations = read(BUNDLE / 'JOB_OBSERVATIONS.json') if (BUNDLE / 'JOB_OBSERVATIONS.json').exists() else {}
    observed_allocations = []
    for group_id, record in exits.items():
        observation = observations.get(record['job_id'])
        if observation and observation['start'] not in ('N/A', 'Unknown'):
            start = datetime.fromisoformat(observation['start']).replace(tzinfo=datetime.fromisoformat(record['utc'].replace('Z', '+00:00')).tzinfo)
            end = datetime.fromisoformat(record['utc'].replace('Z', '+00:00'))
            observed_allocations.append({'group': group_id, 'job_id': record['job_id'], 'start_utc': start.isoformat(),
                                         'exit_utc': record['utc'], 'observed_allocation_seconds': (end - start).total_seconds(),
                                         'note': 'Controller start to exit trap; excludes post-trap scheduler epilog.'})
    audit = {'status': 'PASS', 'utc': utc(), 'job_id': os.environ.get('SLURM_JOB_ID'),
             'v9_freeze_sha256': V9_FREEZE, 'v10_freeze_sha256': V10_FREEZE,
             'units': units, 'readout': readout, 'final_scheduler_exits': exits,
             'launch': launch, 'legacy_log_sha256': sha(legacy),
             'cost': {'measured_training_gpu_hours': sum(r['measured_training_seconds'] for r in units) / 3600,
                      'complete_allocated_cost': False,
                      'observation_file': str(BUNDLE / 'JOB_OBSERVATIONS.json'),
                      'observed_final_allocations': observed_allocations,
                      'limitation': 'Measured training segment times are a lower bound, not full allocation time. Accounting service unavailable; interrupted partial epochs may be missing.'},
             'held_test_read': False, 'model_predictions_recomputed': False}
    final_report(audit)
    audit['report_sha256'] = sha(BUNDLE / 'V9_CLOSEOUT_CN.md')
    return audit


def monitor():
    verify_bundle()
    with (BUNDLE / 'MONITOR.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_science()
        started = time.monotonic()
        observations = read(BUNDLE / 'JOB_OBSERVATIONS.json') if (BUNDLE / 'JOB_OBSERVATIONS.json').exists() else {}
        while time.monotonic() - started < 94 * 3600:
            if (BUNDLE / 'PAUSE_AUTO_START.json').exists():
                write(BUNDLE / 'STATUS.json', {'utc': utc(), 'state': 'PAUSED_BY_RECORD'})
                time.sleep(30)
                continue
            try:
                rows = queue()
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                write(BUNDLE / 'STATUS.json', {'utc': utc(), 'state': 'WAIT_QUEUE_QUERY', 'error': repr(exc)})
                time.sleep(30)
                continue
            for row in rows:
                if row['name'].startswith('v9f_') and row['name'] != 'v9f_feed' and row['state'] == 'RUNNING':
                    observations.setdefault(row['id'], {'first_observed_utc': utc(), **row})
            write(BUNDLE / 'JOB_OBSERVATIONS.json', observations)
            groups, launch = read(V9 / 'GROUPS.json'), read(V9 / 'LAUNCH.json')
            completed = [g['id'] for g in groups if (V9 / 'results' / g['id'] / 'DONE.json').exists()]
            finished = read(V9 / 'FEED_FINISHED.json') if (V9 / 'FEED_FINISHED.json').exists() else None
            state = readiness(rows, groups, launch, completed, finished)
            write(BUNDLE / 'STATUS.json', {'utc': utc(), 'state': 'AUDITING' if state['ready'] else 'WAIT_V9',
                                         'job': os.environ.get('SLURM_JOB_ID'), 'completed_groups': len(completed), **state})
            if state['ready']:
                verify_bundle()
                audit = audit_all()
                # Recheck after hashing: a newly resumed V9 job or changed policy must block release.
                state = readiness(queue(), groups, read(V9 / 'LAUNCH.json'), completed, read(V9 / 'FEED_FINISHED.json'))
                require(state['ready'] and not (BUNDLE / 'PAUSE_AUTO_START.json').exists(), 'V9/policy changed during audit')
                verify_bundle()
                write(BUNDLE / 'CLOSEOUT.json', audit)
                state_path = PROJECT / 'handoff/V9_TAKEOVER_STATE.json'
                takeover = read(state_path)
                takeover.update(updated_utc=utc(), new_full_results_complete=33, groups_completed=18,
                                auto_handoff_status='V9_CLOSEOUT_PASS', closeout=str(BUNDLE / 'CLOSEOUT.json'),
                                final_report=str(BUNDLE / 'V9_CLOSEOUT_CN.md'))
                write(state_path, takeover)
                write(BUNDLE / 'STATUS.json', {'utc': utc(), 'state': 'PASS_V10_DEPENDENCY_RELEASE', 'closeout': str(BUNDLE / 'CLOSEOUT.json')})
                return
            time.sleep(30)
        raise RuntimeError('V9 handoff exceeded 94 hours')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        if args.preflight:
            verify_bundle()
            write(BUNDLE / 'PREFLIGHT.json', verify_science(check_data=True))
        else:
            monitor()
    except Exception as exc:
        write(BUNDLE / 'ERROR.json', {'utc': utc(), 'error': repr(exc), 'traceback': traceback.format_exc()})
        raise
