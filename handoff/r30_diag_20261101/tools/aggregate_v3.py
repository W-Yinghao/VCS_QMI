#!/usr/bin/env python3
"""v3 aggregation (DIAG_PROTOCOL_V3.json, DIAG_EXPORT_CONTRACT.json): states_out_v3/<cond>_s<seed>__<label>.json (+ gradient
strength npz) -> MARGIN_DIAG.json (one record per unit_id, contract fields) + numbers-only MARGIN_DIAG.md.
Calibrated gradient quantiles use COEFFICIENTS.json of the R30 run (s1001 H parent of the condition) when it exists.
Usage: aggregate_v3.py STATES_DIR OUT_DIR COVERAGE_V3.json [COEFFICIENTS.json]"""
import glob, json, os, sys
import numpy as np

LAYERS = (('A_score_joint@update', 'A 独立 score-space（双向联合梯度，单位步，一阶），更新批'),
          ('A_score_own@update', 'A 独立 score-space（每个 query 只沿自身 loss 梯度，单位步，一阶），更新批'),
          ('B_main_inherited@update', 'B 仅主项 + 继承 AdamW（真实裁剪），更新批'),
          ('B_main_inherited@heldout', 'B 仅主项 + 继承 AdamW（真实裁剪），留出 FIT 批'),
          ('C_full@update', 'C 主项+intra+noisy + 继承 AdamW（真实裁剪），更新批'),
          ('C_full@heldout', 'C 主项+intra+noisy + 继承 AdamW（真实裁剪），留出 FIT 批'))
EXTRA = ('X_adam1_fresh@update', 'X_gd@update')
OBJS = ('H_hinge', 'N', 'L', 'B', 'PN', 'PV', 'PB')
SUBS = (('all', '全部'), ('ref_labelled_first', '参考态标注正例第一'), ('ref_labelled_not_first', '参考态标注正例未第一'),
        ('mismatch_truepair_absent', '错配·真对应不在批内'), ('mismatch_truepair_present', '错配·真对应在批内'),
        ('correct_ref_not_first', '正确困难'), ('correct_ref_first', '正确容易'))
AUD = (('correct_easy', '正确容易'), ('correct_hard', '正确困难'), ('mismatch&true_unavailable', '错配·真对应不在候选'),
       ('mismatch&true_available&competitor_is_true', '错配·竞争者即真对应'), ('mismatch&true_available&competitor_not_true', '错配·真对应在候选但非竞争者'))
QGRP = ('all', 'correct_ref_first', 'correct_ref_not_first', 'mismatch_truepair_absent')
COEF_OF = {'L': 'Ls', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}
THR = None
CN = [str(i) for i in range(1, 60)]


def f(x, nd=3):
    if x is None:
        return 'NA'
    if isinstance(x, bool):
        return '1' if x else '0'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f'{x:.{nd}f}'


def table(h, rows):
    return '\n'.join(['| ' + ' | '.join(h) + ' |', '|' + '---|' * len(h)] + ['| ' + ' | '.join(r) + ' |' for r in rows])


ORDER = ('r30_parent', 'r30_e10', 'r30_e40', 'h_native', 'n_infonce', 'l_listwise_vcs', 'b_listwise_bce')


def skey(r):
    m = r['meta']
    return (m['cond'], m['seed'], next((i for i, a in enumerate(ORDER) if m['label'].startswith(a)), 99), m['raw_epoch'], m['label'])


def main():
    global THR
    src, out, covp = sys.argv[1], sys.argv[2], sys.argv[3]
    coefp = sys.argv[4] if len(sys.argv) > 4 else None
    proto = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DIAG_PROTOCOL_V3.json')))
    THR = proto['near_zero_threshold']
    cov = json.load(open(covp))
    coefs = json.load(open(coefp)) if coefp and os.path.isfile(coefp) else None
    states = sorted((json.load(open(p)) for p in glob.glob(os.path.join(src, '*.json'))), key=skey)
    # gradient-strength quantiles (raw and calibrated)
    for s in states:
        m = s['meta']
        z = np.load(m['gradstrength_file'])
        pid = cov['reference_parent_per_condition_seed'].get(f"{m['cond']}_s1001")
        cset = (coefs or {}).get('parents', {}).get(pid, {}).get('coefficients') if coefs else None
        q = {}
        for o in OBJS:
            g = z[f'g_{o}'].astype(np.float64)
            c = None
            if cset is not None:
                c = 1.0 if o in ('N', 'B', 'H_hinge') else (cset.get(COEF_OF[o], {}).get('c') if o in COEF_OF else None)
            for grp in ('all', 'ref_labelled_first', 'ref_labelled_not_first', 'mismatch_truepair_absent', 'mismatch_truepair_present',
                        'correct_ref_not_first', 'correct_ref_first'):
                v = g[z[f'grp_{grp}']]
                if v.size == 0:
                    q[f'{o}|{grp}'] = {'n': 0}; continue
                qs = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
                q[f'{o}|{grp}'] = {'n': int(v.size), 'raw_quantiles_p5_p25_p50_p75_p95': [float(x) for x in qs], 'raw_frac_below_threshold': float((v < THR).mean()),
                                   'calibration_coefficient': c, 'calibrated_quantiles_p5_p25_p50_p75_p95': [float(x * c) for x in qs] if c else None,
                                   'calibrated_frac_below_threshold': float((v * c < THR).mean()) if c else None}
        s['grad_strength'] = {'threshold': THR, 'coefficients_source': coefp if cset is not None else None, 'by_objective_group': q}
    records = {s['meta']['unit_id']: {k: v for k, v in s.items() if k != 'grad_composition_per_batch'} for s in states}   # per-batch rows stay in the per-state files
    doc = {'contract': 'DIAG_EXPORT_CONTRACT.json', 'protocol': 'DIAG_PROTOCOL_V3.json', 'protocol_sha256': states[0]['meta']['protocol_sha256'] if states else None,
           'coverage': cov, 'records': records}
    json.dump(doc, open(os.path.join(out, 'MARGIN_DIAG.json'), 'w'))

    L = ['# R30 共同状态探针 v3：三层更新、竞争者身份审计、固定参考清单', '',
         '只测量保存状态上单步更新的局部行为，不是对训练失败的完整解释。注入噪声真值只进入评价器（分组、真对应 margin/排名、身份审计）。',
         '分组与固定参考清单在同条件同 seed 的共同 H 父状态上一次确定；所有状态与目标共用同一人群。协议：DIAG_PROTOCOL_V3.json；字段契约：DIAG_EXPORT_CONTRACT.json。',
         f"PN/PV/PB 实现来源：{states[0]['meta']['pair_source'] if states else 'NA'}。近零阈值（事先固定）：{THR}。", '']
    n = 0

    def H(t):
        nonlocal n
        n += 1
        return f'## 表{n} {t}'
    L += [H('覆盖范围'), '']
    L += [table(['类别', 'unit_id', '完成co-train轮', '存在', '已测量', '路径'],
                [[i['kind'], i['unit_id'], str(i['completed_cotrain_epochs']), f(i['exists']), f(i['measured']), i['path']] for i in cov['items']]), '']
    L += [H('参考父状态与固定参考清单'), '']
    rows, seen = [], set()
    for s in states:
        r = s['meta']['reference']
        k = (s['meta']['cond'], s['meta']['seed'])
        if k in seen:
            continue
        seen.add(k)
        c = r['counts']
        g = lambda t, x: str(c.get(f'{t}:i2t:{x}', 0) + c.get(f'{t}:t2i:{x}', 0))
        rows.append([f'{k[0]}_s{k[1]}', r['ref_id'], r['state_sha256'][:16], r['fixed_list_sha256'][:16]]
                    + [g('update', x) for x, _ in SUBS] + [g('heldout', x) for x, _ in SUBS[:1] + SUBS[3:]])
    L += [table(['条件', '参考父状态', '父状态sha256前16', '固定清单sha256前16'] + ['更新批 ' + h for _, h in SUBS] + ['留出批 ' + h for _, h in SUBS[:1] + SUBS[3:]], rows), '']
    L += [H('已测量状态与有效性回执'), '']
    L += [table(['unit_id', '完成co-train轮', 'full-DEV rSum', '下一步lr', 'noisy分支', '指纹不变', 'GRU模式前向差', '恢复后重算差', '非有限', 'GPU', '秒'],
                [[s['meta']['unit_id'], str(s['meta']['completed_cotrain_epochs']), f(s['meta']['full_dev_rsum'], 2), f(s['meta']['lr_next'], 6),
                  f(all(s['meta']['noisy_branch_available'].values())), f(s['meta']['checks']['fingerprint_unchanged']),
                  f"{s['meta']['checks']['gru_train_vs_eval_forward_maxabs']:.1e}", f"{s['meta']['checks']['restore_recompute_maxabs']:.1e}",
                  str(s['meta']['checks']['nonfinite_events']), s['meta']['env']['gpu'], f(s['meta']['seconds'], 0)] for s in states]), '']
    L += [H('固定参考清单上的拟合—排序（两 peer 平均分数，两方向合并，全部 query；S−J 与独立平方 gap 的最大差）'), '']
    rows = []
    for s in states:
        v = s['fixed_list']['both|all']
        rows.append([s['meta']['unit_id'], str(s['meta']['completed_cotrain_epochs']), f(s['meta']['full_dev_rsum'], 2),
                     f(v['mean_J_list'], 4), f(v['mean_S_list'], 4), f(v['mean_S_list'] - v['mean_J_list'], 4), f(v['mean_gap_list'], 4),
                     f"{v['max_abs_S_minus_J_minus_gap_list']:.1e}", f(v['mean_m'], 3), f(v['mean_J_pair'], 4), f(v['mean_S_pair'], 4),
                     f(v['mean_gap_pair'], 4), f"{v['max_abs_S_minus_J_minus_gap_pair']:.1e}", f(v['mean_margin_fixed'], 4), f(v['mean_pos_first'], 3), f(v['mean_rank_pos'], 2)])
    L += [table(['unit_id', '完成轮', 'full-DEV rSum', 'J列表', 'S列表=(K−m)/(K+m)', 'S−J列表', 'gap列表', 'max abs(S−J−gap) 列表', '平均m',
                 'J二选一', 'S二选一', 'gap二选一', 'max abs(S−J−gap) 二选一', '固定竞争者margin', '标注正例第一比例', '平均排名'], rows), '']
    L += [H('动态最难竞争者读数（与固定清单分表）'), '']
    L += [table(['unit_id', '完成轮', '动态margin', 'J二选一（动态）', 'gap二选一（动态）', '动态竞争者=固定竞争者比例'],
                [[s['meta']['unit_id'], str(s['meta']['completed_cotrain_epochs']), f(s['fixed_list']['both|all']['mean_margin_dyn'], 4),
                  f(s['fixed_list']['both|all']['mean_J_pair_dyn'], 4), f(s['fixed_list']['both|all']['mean_gap_pair_dyn'], 4),
                  f(s['fixed_list']['both|all']['mean_dyn_equals_fixed'], 3)] for s in states]), '']
    L += [H('B/C 层的梯度构成、裁剪与实际更新范数（每批每 peer 的均值；比值为中位数）'), '']
    rows = []
    for s in states:
        for o in OBJS:
            g = s['grad_composition'][o]
            rows.append([s['meta']['unit_id'], o, f(g['mean_g_main'], 3), f(g['mean_g_intra'], 3), f(g['mean_g_noisy'], 3),
                         f((g['ratio_main_intra_quantiles'] or {}).get('p50'), 2), f((g['ratio_main_noisy_quantiles'] or {}).get('p50'), 2),
                         f(g['mean_cos_main_intra'], 3), f(g['mean_cos_main_noisy'], 3), f(g['mean_cos_intra_noisy'], 3),
                         f(g['mean_B_preclip'], 3), f(g['mean_B_postclip'], 3), f(g['B_clip_rate'], 3), f(g['mean_B_update_norm'], 4),
                         f(g['mean_C_preclip'], 3), f(g['mean_C_postclip'], 3), f(g['C_clip_rate'], 3), f(g['mean_C_update_norm'], 4)])
    L += [table(['unit_id', '目标', '‖g主项‖', '‖g intra‖', '‖g noisy‖', '主/intra中位比', '主/noisy中位比', 'cos(主,intra)', 'cos(主,noisy)', 'cos(intra,noisy)',
                 'B裁剪前', 'B裁剪后', 'B裁剪率', 'B实际更新范数', 'C裁剪前', 'C裁剪后', 'C裁剪率', 'C实际更新范数'], rows), '']
    L += [H(f'每 query score-space 梯度强度分位数（原始 / 标定后；近零 = 小于 {THR}）'), '']
    rows = []
    for s in states:
        for o in OBJS:
            for grp in QGRP:
                v = s['grad_strength']['by_objective_group'][f'{o}|{grp}']
                if not v.get('n'):
                    continue
                rq = v['raw_quantiles_p5_p25_p50_p75_p95']; cq = v['calibrated_quantiles_p5_p25_p50_p75_p95']
                rows.append([s['meta']['unit_id'], o, grp, str(v['n']), ' / '.join(f(x, 4) for x in rq), f(v['raw_frac_below_threshold'], 4),
                             f(v['calibration_coefficient'], 4), ' / '.join(f(x, 4) for x in cq) if cq else 'NA', f(v['calibrated_frac_below_threshold'], 4)])
    L += [table(['unit_id', '目标', '分组', 'n', '原始 p5/p25/p50/p75/p95', '原始近零比例', '标定系数', '标定后 p5/p25/p50/p75/p95', '标定后近零比例'], rows), '']
    for layer, title in LAYERS:
        L += [H(f'{title}：P(loss↓ 且 观测margin↓) / 平均 Δ观测margin'), '']
        rows = []
        for s in states:
            for o in OBJS:
                v = s['results'][layer][o]['AB']['both']
                nd = 4 if layer.startswith('A') else 5
                rows.append([s['meta']['unit_id'], o] + [f"{f(v[k]['frac_loss_down_and_margin_down'])} / {f(v[k]['mean_dmargin'], nd)}" for k, _ in SUBS]
                            + [f"{f(v[k]['frac_loss_down_and_truepair_margin_down'])} / {f(v[k]['mean_dtruepair_margin'], nd)}" for k in ('correct_ref_not_first', 'correct_ref_first')]
                            + [str(v['all']['n']), f(v['all']['frac_loss_down']), f(v['all']['frac_margin_down'])])
        L += [table(['unit_id', '目标'] + [h for _, h in SUBS] + ['正确困难 真margin', '正确容易 真margin', 'n', 'P(loss↓)', 'P(观测margin↓)'], rows), '']
    for layer in ('B_main_inherited@update', 'B_main_inherited@heldout', 'C_full@update', 'C_full@heldout'):
        L += [H(f'竞争者身份审计 {layer}：n / Δ观测margin / Δ真margin / P(真排名变差) / P(竞争者换人) / 错误第一名率 前→后'), '']
        rows = []
        for s in states:
            for o in OBJS:
                a = s['audit'][layer][o]['both']
                cells = []
                for g, _ in AUD:
                    v = a.get(g, {'group_count': 0})
                    if not v['group_count']:
                        cells.append('0'); continue
                    cells.append(f"{v['group_count']} / {f(v['mean_d_observed_margin'], 5)} / {f(v['mean_d_true_margin'], 5)} / {f(v['frac_true_rank_worse'])} / "
                                 f"{f(v['frac_top_competitor_changed'])} / {f(v['wrong_top1_observed_before'])}→{f(v['wrong_top1_observed_after'])}")
                al = a['all']
                cells += [str(al['unavailable_count']), str(al['competitor_is_true_match_count'])]
                rows.append([s['meta']['unit_id'], o] + cells)
        L += [table(['unit_id', '目标'] + [h for _, h in AUD] + ['真对应不在候选数（全部）', '竞争者即真对应数（全部）'], rows), '']
    open(os.path.join(out, 'MARGIN_DIAG.md'), 'w').write('\n'.join(L) + '\n')
    print('records', len(records))


if __name__ == '__main__':
    main()
