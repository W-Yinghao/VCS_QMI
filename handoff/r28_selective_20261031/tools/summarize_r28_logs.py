"""R28 training-log summary (numbers only, no interpretation; no TEST). Reads EPOCHS.json of the 8 R28 units and reports,
per unit and peer: (1) the per-stage calibration records written when the stage fits run (rstage_*: anchor a, b, beta,
support-calibrator u means on fit/replay split with/without the observed pair, fallbacks, labelled-pair counts), one row per
distinct rstage_version; (2) the t2i selective-pooling statistics logged every co-train epoch (u mean, fraction u<0.1 /
u<0.5 / u>0.9, fallback mass = mean (1-u), final alpha max/entropy, argmax-differs-from-cosine fraction, residual stats),
as epoch 0 / epoch 39 / mean over the 40 co-train epochs. Writes R28_LOG_SUMMARY.json and R28_LOG_SUMMARY.md."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = '/projects/EEG-foundation-model/yinghao/outputs_v9/r28_selective_20261031/runs'
CELLS = ('f40_s1001', 'f60_s1001')
ARMS = ('g1_cos_sel', 'g2_vcs_sel', 'r1_res_vcs_sel', 'r2_res_bce_sel')
STAGE = ['rstage_kind', 'rstage_a', 'rstage_b', 'rstage_beta', 'rstage_thr', 'rstage_n_labeled', 'rstage_nP_fit', 'rstage_aff_fallback',
         'rstage_u_fallback', 'rstage_calfit_u_mean_with_pair', 'rstage_calfit_u_mean_without_pair', 'rstage_calrep_u_mean_with_pair',
         'rstage_calrep_u_mean_without_pair', 'rstage_val_top1_obs']
EPOCH = ['t2i_r28_u_mean', 't2i_r28_frac_u_lt_0.1_mean', 't2i_r28_frac_u_lt_0.5_mean', 't2i_r28_frac_u_above_0.9_mean',
         't2i_r28_fallback_mass_mean', 't2i_r28_alpha_final_max_mean', 't2i_r28_alpha_final_entropy_mean',
         't2i_r28_argmax_differs_from_cos_mean', 't2i_r28_res_rowstd_mean', 't2i_r28_res_absmax_mean']

out = {}
for c in CELLS:
    for a in ARMS:
        d = f'{RUNS}/{c}/{a}'
        assert os.path.exists(f'{d}/DONE.json'), d
        ep = json.load(open(f'{d}/EPOCHS.json'))['cotrain']
        assert [e['epoch'] for e in ep] == list(range(40)), d
        u = {}
        for peer in ('A', 'B'):
            stages, seen = [], set()
            for e in ep:
                t = e['train'][peer]
                v = t.get('rstage_version')
                if v is not None and v not in seen:
                    seen.add(v); stages.append({'first_epoch': e['epoch'], 'version': v, **{k: t.get(k) for k in STAGE}})
            per = {k: [e['train'][peer].get(k) for e in ep] for k in EPOCH}
            u[peer] = {'stages': stages,
                       'epoch_stats': {k: {'epoch0': v[0], 'epoch39': v[-1],
                                           'mean40': float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None,
                                           'n_epochs_logged': sum(x is not None for x in v)} for k, v in per.items()}}
        out[f'{c}/{a}'] = u
json.dump({'note': __doc__, 'units': out}, open(os.path.join(HERE, 'R28_LOG_SUMMARY.json'), 'w'), indent=1)
f = lambda v, p=3: 'NA' if v is None else (f'{v:.{p}f}' if isinstance(v, float) else str(v))
L = ['# R28 训练日志汇总（自动生成；只列数字，不含解读；未读 TEST）', '',
     '来源：各单元 EPOCHS.json。阶段校准记录（rstage_*）在每次阶段拟合时写入，每个不同 rstage_version 列一行；t2i 选择性加权统计每个 co-train epoch 记录（u 为支持校准器输出，回退质量 = mean(1−u)）。R2 f60 第 0–3 轮在 runfill A40 上运行，00:00UTC 被抢占后从 epoch 3 的 state_last 续跑于 RTX6000PRO（每轮秒数见 EPOCHS.json）。', '']
L += ['## 阶段校准记录', '', '| 单元 | peer | 首次 epoch | kind | a | b | β | 阈值 | 标注对数 | 拟合正例 | 锚回退 | u 回退 | u 拟合集 有/无配对 | u 复查集 有/无配对 | val top1(obs) |',
      '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for k, u in out.items():
    for peer, v in u.items():
        for s in v['stages']:
            L.append(f"| {k} | {peer} | {s['first_epoch']} | {s['rstage_kind']} | {f(s['rstage_a'])} | {f(s['rstage_b'])} | {f(s['rstage_beta'])} | {f(s['rstage_thr'])} | "
                     f"{f(s['rstage_n_labeled'])} | {f(s['rstage_nP_fit'])} | {f(s['rstage_aff_fallback'])} | {f(s['rstage_u_fallback'])} | "
                     f"{f(s['rstage_calfit_u_mean_with_pair'])} / {f(s['rstage_calfit_u_mean_without_pair'])} | "
                     f"{f(s['rstage_calrep_u_mean_with_pair'])} / {f(s['rstage_calrep_u_mean_without_pair'])} | {f(s['rstage_val_top1_obs'])} |")
L += ['', '## t2i 选择性加权统计（epoch 0 / epoch 39 / 40 轮均值）', '',
      '| 单元 | peer | ' + ' | '.join(k.replace('t2i_r28_', '').replace('_mean', '') for k in EPOCH) + ' |', '|---|---|' + '---|' * len(EPOCH)]
for k, u in out.items():
    for peer, v in u.items():
        L.append(f'| {k} | {peer} | ' + ' | '.join(f"{f(s['epoch0'])} / {f(s['epoch39'])} / {f(s['mean40'])}" for s in v['epoch_stats'].values()) + ' |')
open(os.path.join(HERE, 'R28_LOG_SUMMARY.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[:20]))
