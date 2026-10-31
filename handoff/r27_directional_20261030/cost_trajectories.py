"""Full per-attempt cost and per-epoch DEV trajectories for the R27-S / R27-D / R27-O / R27-O3 / R27-CS / R27-A1X units.
sacct is not available on this cluster, so cost is reconstructed from the job logs: every Slurm log of a unit (first attempt
and every timeout continuation) gives one attempt; its span = last minus first logging timestamp (python logging, includes
data load and DEV evaluation; excludes node staging before the first log line), GPU = the nvidia-smi line at the top.
PEAK_*.json wall_seconds exist only for attempts that ended normally; timed-out attempts are counted from their logs here.
Trajectories: per-epoch full-DEV rSum from EPOCHS.json. No TEST. Writes COST_TRAJECTORIES.json and COST_TRAJECTORIES.md."""
import glob, json, os, re, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
H = '/home/infres/yinwang/CS_QMI/handoff'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9'
LINES = {  # line: (log dir, log-name prefix, output root, arms, cells)
    'R27-S': (f'{H}/r27_qtarget_20261029/logs', 'r27s_', f'{O}/r27_qtarget_20261029/runs',
              ['s1_balanced_dual', 's2_single_free_vcs', 's3_single_cap_delta', 's4_single_cap_vcs', 's5_single_cap_bce', 's6_single_cap_snce'], ['f40_s1001', 'f60_s1001']),
    'R27-D': (f'{H}/r27_directional_20261030/logs', 'r27d_', f'{O}/r27_directional_20261030/runs',
              ['d1_vcs_i2t', 'd2_vcs_t2i', 'd3_snce_i2t', 'd4_snce_t2i'], ['f40_s1001', 'f60_s1001']),
    'R27-O': (f'{H}/r27_oracle_20261030/logs', 'r27o_', f'{O}/r27_oracle_20261030/runs', ['o1_rescue_native_t2i', 'o2_rescue_vcs_t2i'], ['f40_s1001', 'f60_s1001']),
    'R27-O3': (f'{H}/r27_oracle3_20261031/logs', 'r27o3_', f'{O}/r27_oracle3_20261031/runs', ['o3_true_target_t2i'], ['f40_s1001', 'f60_s1001']),
    'R27-CS': (f'{H}/r27_candsrc_20261031/logs', 'r27cs_', f'{O}/r27_candsrc_20261031/runs', ['bn_cross_t2i', 'cn_mix_t2i'], ['f40_s1001', 'f60_s1001']),
    'R27-A1X': (f'{H}/r27_a1cross_20261030/logs', 'r27x_a1_', f'{O}/r27_a1cross_20261030/runs', ['a1_iface'],
                ['c40_s1001', 'c40_s1002', 'c40_s1003', 'cc_s1001', 'cc_s1002', 'cc_s1003']),
}
TS = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ')


def attempt(path):
    first = last = None; gpu = None; ended = None
    with open(path, errors='replace') as f:
        for ln in f:
            if gpu is None and ('NVIDIA' in ln and 'MiB' in ln):
                gpu = ln.split(',')[0].strip()
            m = TS.match(ln)
            if m:
                t = datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                first = first or t; last = t
            if 'continuation submitted' in ln: ended = 'timeout_continued'
            if 'UNIT_DONE' in ln: ended = 'done'
            if 'UNIT_FAILED' in ln: ended = 'failed'
    return dict(log=os.path.basename(path), job=re.search(r'_(\d+)\.out$', path).group(1), gpu=gpu, ended=ended or 'unknown',
                span_h=None if first is None else (last - first).total_seconds() / 3600)


out = {}
for line, (ld, pre, root, arms, cells) in LINES.items():
    out[line] = {}
    for c in cells:
        for a in arms:
            if line == 'R27-A1X':
                logs = sorted(glob.glob(f'{ld}/{pre}{c}_*.out'))
            else:
                logs = sorted(glob.glob(f'{ld}/{pre}*{a}_{c}_*.out'))
            logs = [p for p in logs if re.search(r'_\d+\.out$', p)]
            atts = sorted((attempt(p) for p in logs), key=lambda x: int(x['job']))
            d = f'{root}/{c}/{a}'
            traj = None
            if os.path.exists(f'{d}/EPOCHS.json'):
                traj = [round(e['dev']['rsum'], 3) for e in json.load(open(f'{d}/EPOCHS.json'))['cotrain']]
            peaks = [json.load(open(p)) for p in sorted(glob.glob(f'{d}/PEAK_*.json'))]
            out[line][f'{c}/{a}'] = dict(done=os.path.exists(f'{d}/DONE.json'), attempts=atts, n_attempts=len(atts),
                                          log_span_h_total=sum(x['span_h'] or 0 for x in atts),
                                          peak_wall_h_recorded=sum(p.get('wall_seconds', 0) for p in peaks) / 3600,
                                          gpus=sorted({x['gpu'] for x in atts if x['gpu']}), dev_rsum_per_epoch=traj)
json.dump({'note': __doc__, 'units': out}, open(os.path.join(HERE, 'COST_TRAJECTORIES.json'), 'w'), indent=1)
L = ['# R27 后续线：逐尝试成本与 DEV 轨迹（自动生成；只列数字）', '',
     'sacct 不可用；成本由每个尝试的作业日志首末 logging 时间戳之差重建（含数据载入与 DEV 评估，不含首条日志前的节点 staging），超时续跑的尝试也计入。PEAK 记录只覆盖正常结束的尝试，二者并列。GPU 为日志开头 nvidia-smi 所记型号；不同 GPU 的耗时不可直接比较速度。', '',
     '## 成本', '', '| 线 | 单元 | 完成 | 尝试数 | 日志跨度合计 h | PEAK 记录 h | GPU | 各尝试（作业: 结束方式, h） |', '|---|---|---|---|---|---|---|---|']
for line, us in out.items():
    for k, u in us.items():
        L.append(f"| {line} | {k} | {'是' if u['done'] else '否'} | {u['n_attempts']} | {u['log_span_h_total']:.2f} | {u['peak_wall_h_recorded']:.2f} | "
                 f"{'; '.join(u['gpus'])} | " + '; '.join(f"{x['job']}: {x['ended']}, {x['span_h']:.2f}" if x['span_h'] is not None else f"{x['job']}: {x['ended']}, NA" for x in u['attempts']) + ' |')
tot = sum(u['log_span_h_total'] for us in out.values() for u in us.values())
L += ['', f'日志跨度合计（全部单元、全部尝试）：{tot:.1f} GPU·h（不同 GPU 型号混合）。', '',
      '## DEV rSum 轨迹（每轮，co-train epoch 0–39）', '']
for line, us in out.items():
    for k, u in us.items():
        t = u['dev_rsum_per_epoch']
        L.append(f"- {line} {k}: " + ('NA' if not t else ', '.join(f'{x:.2f}' for x in t)))
open(os.path.join(HERE, 'COST_TRAJECTORIES.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:12])); print('total', round(tot, 1))
