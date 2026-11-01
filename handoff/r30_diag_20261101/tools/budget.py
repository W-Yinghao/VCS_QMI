#!/usr/bin/env python3
"""Diagnostic budget ledger (reported separately from the R30 training budget): one row per submitted job of this package.
Wall time = first submission record -> last write of the job log (upper bound: includes queueing/staging; sacct unavailable).
GPU jobs counted as 1 GPU. Usage: budget.py  (writes BUDGET.json / BUDGET.md)"""
import csv, datetime as dt, glob, json, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.reader(open(os.path.join(D, 'SUBMITTED.tsv')), delimiter='\t'))[1:]
first = {}
for r in rows:
    if len(r) >= 3 and r[2].isdigit():
        first.setdefault(r[2], (r[0], r[1], r[3] if len(r) > 3 else ''))
out = []
for job, (t0, name, note) in first.items():
    logs = glob.glob(os.path.join(D, 'logs', f'*_{job}.out'))
    t0d = dt.datetime.strptime(t0, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc)
    end = dt.datetime.fromtimestamp(os.path.getmtime(logs[0]), dt.timezone.utc) if logs else None
    gpu = not name.startswith(('r30diag_cpu', 'r30diag_queue'))
    sec = (end - t0d).total_seconds() if end else None
    out.append(dict(job=job, name=name, kind='GPU runfill' if gpu else 'CPU', submitted_utc=t0, last_log_write_utc=end.strftime('%Y-%m-%dT%H:%M:%SZ') if end else None,
                    wall_seconds_upper_bound=sec, note=note))
gh = sum(r['wall_seconds_upper_bound'] or 0 for r in out if r['kind'].startswith('GPU')) / 3600
ch = sum(r['wall_seconds_upper_bound'] or 0 for r in out if r['kind'] == 'CPU') / 3600
json.dump({'jobs': out, 'gpu_hours_upper_bound': gh, 'cpu_job_hours_upper_bound': ch,
           'note': 'upper bounds (submit -> last log write); runfill GPU (V100/L40S/A100/A40); includes failed/smoke attempts'}, open(os.path.join(D, 'BUDGET.json'), 'w'), indent=1)
L = ['# 诊断预算（与 R30 训练预算分开）', '', f'GPU 小时上界合计：{gh:.2f}；CPU 作业小时上界合计：{ch:.2f}（提交→日志最后写入，含排队/拷贝；含失败与 smoke 尝试）', '',
     '| 作业 | 名称 | 类型 | 提交UTC | 日志最后写入UTC | 墙钟秒上界 |', '|---|---|---|---|---|---|']
L += [f"| {r['job']} | {r['name']} | {r['kind']} | {r['submitted_utc']} | {r['last_log_write_utc']} | {r['wall_seconds_upper_bound']:.0f} |" if r['wall_seconds_upper_bound'] is not None
      else f"| {r['job']} | {r['name']} | {r['kind']} | {r['submitted_utc']} | NA | NA |" for r in out]
open(os.path.join(D, 'BUDGET.md'), 'w').write('\n'.join(L) + '\n')
print(f'GPU h {gh:.2f} CPU h {ch:.2f} jobs {len(out)}')
