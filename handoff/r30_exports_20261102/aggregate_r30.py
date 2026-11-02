#!/usr/bin/env python3
"""Numbers-only aggregation of the R30 exports -> EXPORT_R30_CN.md, out/EXPORT_R30.json, out/actualscale_layers.csv,
out/branch_classes.csv. Inputs: v3 probe outputs (handoff/r30_diag_20261101/states_out_v3 + out/states_v3_a1), v4
actual-scale outputs (out/states_v4), branch exports (out/branch_v1), out/LOGGED_SPLIT_COUNTS.csv, COEFFICIENTS.json.
Usage: python3 -B aggregate_r30.py"""
import csv, glob, json, math, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = '/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101'
COEF = json.load(open('/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/COEFFICIENTS.json'))
THR = json.load(open(f'{V3}/DIAG_PROTOCOL_V3.json'))['near_zero_threshold']
OBJS = ('H_hinge', 'N', 'L', 'B', 'PN', 'PV', 'PB')
COEF_KEY = {'L': 'Ls', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}
OBJ_OF_R30 = {'H': 'H_hinge', 'L': 'L', 'N': 'N', 'B': 'B', 'PN': 'PN', 'PV': 'PV', 'PB': 'PB'}
SUBS = ('all', 'ref_labelled_first', 'ref_labelled_not_first', 'mismatch_truepair_absent', 'mismatch_truepair_present', 'correct_ref_not_first', 'correct_ref_first')
MD_SUBS = (('all', '全部'), ('mismatch_truepair_absent', '错配·真对应不在批内'), ('correct_ref_not_first', '正确困难'))
LAYERS = (('B_main_inherited@update', 'B_main_actualscale@update'), ('B_main_inherited@heldout', 'B_main_actualscale@heldout'),
          ('C_full@update', 'C_full_actualscale@update'), ('C_full@heldout', 'C_full_actualscale@heldout'))
NA = 'NA'


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return NA
    if isinstance(x, bool):
        return '1' if x else '0'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f'{x:.{nd}f}'


def table(h, rows):
    return '\n'.join(['| ' + ' | '.join(h) + ' |', '|' + '---|' * len(h)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def load(pattern):
    out = {}
    for p in glob.glob(pattern):
        if p.endswith('.json') and not os.path.basename(p).startswith('REFLIST'):
            d = json.load(open(p)); out[(d['meta']['cond'], d['meta']['seed'], d['meta']['label'])] = d
    return out


def common_c(cond, o):
    """v3 'calibrated' column: s1001 H parent of the condition for every seed"""
    if o not in COEF_KEY:
        return 1.0
    return COEF['parents'][f'wp_{cond}_s1001_H_e5']['coefficients'][COEF_KEY[o]]['c']


def actual_c(meta, o):
    so = meta.get('state_opt_r30') or {}
    if so.get('r30_stage') == 'unit' and OBJ_OF_R30.get(so.get('r30_objective')) == o:
        return 1.0 if so.get('r30_coef') is None else float(so['r30_coef']), 'unit_own'
    if o not in COEF_KEY:
        return 1.0, 'fixed_1'
    return COEF['parents'][f"wp_{meta['cond']}_s{meta['seed']}_H_e5"]['coefficients'][COEF_KEY[o]]['c'], 'cell_H_parent'


def main():
    v3 = load(f'{V3}/states_out_v3/*.json'); v3.update(load(f'{HERE}/out/states_v3_a1/*.json'))
    v4 = load(f'{HERE}/out/states_v4/*.json')
    br = load(f'{HERE}/out/branch_v1/*.json')
    doc = {'v3_states': len(v3), 'v4_states': len(v4), 'branch_states': len(br)}
    L = ['# R30 只读导出：实际单元系数列与可信主分支 / noisy 修复分支分类统计（2026-11-02 复盘包 §1 R30）', '',
         '只测量保存状态；无新训练、无反馈 2×2、未读 TEST。注入噪声真值只进入评价器（分组、计数、hard-correct 判定）。数字表，无解读。', '',
         '## 表0 f60_s1003 HH 的 L40S DEV 复核（已由主会话完成，本次未重做）', '',
         '作业 1080903（L40S，DEV-only）：rsum 496.1 对日志 496.07，六个 Recall 全部相同；V100-16GB 门禁失败记录保留于 '
         'handoff/test_eval_20261101/results_superseded/ca77e4e6b58e5357.V100-16GB_gateFAIL.json；TEST 于 L40S 重做（1080912/1080913，见 handoff/test_eval_20261101/SUBMITTED.tsv）。', '']
    # ---------------- coefficients
    rows = []
    for k in sorted(set(v3) | set(v4)):
        m = (v4.get(k) or v3.get(k))['meta']
        so = m.get('state_opt_r30') or {}
        own = OBJ_OF_R30.get(so.get('r30_objective')) if so.get('r30_stage') == 'unit' else None
        rows.append([m['unit_id'], m['completed_cotrain_epochs'], own or '-', ('native=1' if so.get('r30_coef') is None else f(so.get('r30_coef'), 4)) if own else '-',
                     ' '.join(f"{o}:{f(common_c(m['cond'], o), 4)}" for o in ('L', 'PN', 'PV', 'PB')),
                     ' '.join(f"{o}:{f(actual_c(m, o)[0], 4)}" for o in ('L', 'PN', 'PV', 'PB')), 'yes' if k in v4 else ('NA (state deleted)' if m['unit_id'].startswith('r29_') else 'no')])
    L += ['## 表1 系数：共同参考列（v3：条件的 s1001 H 父状态）与实际单元列', '',
          '实际单元列：该状态所属单元自身目标用其 r30_coef（L0/HL/LL 为 1，Ls 为标定值，H 原生=1）；其余目标用本 (cond, seed) 的 H 父状态标定（L→Ls），N/B/H=1。'
          'R29 保存状态已在 11-02 清理中删除，B/C 层实际系数重算为 NA（梯度强度分位数仍可由 v3 的 npz 按系数缩放）。', '',
          table(['unit_id', '完成轮', '自身目标', '自身系数', '共同参考 c (v3)', '实际单元 c', 'B/C 实际系数已测'], rows), '']
    # ---------------- B/C layers: v3 raw vs v4 actual
    csv_rows, md_rows = [], []
    for k in sorted(v4):
        a = v4[k]; m = a['meta']; b = v3.get(k)
        for o in m['objectives_run']:
            for l3, l4 in LAYERS:
                for s in SUBS:
                    x = a['results'][l4][o]['AB']['both'][s]
                    y = b['results'][l3][o]['AB']['both'][s] if b else None
                    csv_rows.append({'unit_id': m['unit_id'], 'completed_cotrain_epochs': m['completed_cotrain_epochs'], 'cond': m['cond'], 'seed': m['seed'],
                                     'objective': o, 'c_actual': m['coefficients_actual'][o]['c'], 'c_source': m['coefficients_actual'][o]['source'],
                                     'layer': l4.split('@')[0].replace('_actualscale', ''), 'batch': l4.split('@')[1], 'subset': s, 'n': x['n'],
                                     **{f'actual_{q}': x.get(q) for q in ('frac_loss_down_and_margin_down', 'mean_dmargin', 'mean_dtruepair_margin', 'frac_loss_down', 'frac_margin_down')},
                                     **{f'v3raw_{q}': (y or {}).get(q) for q in ('frac_loss_down_and_margin_down', 'mean_dmargin', 'mean_dtruepair_margin', 'frac_loss_down', 'frac_margin_down')}})
            for l3, l4 in (LAYERS[0], LAYERS[3]):
                cells = []
                for s, _ in MD_SUBS:
                    x = a['results'][l4][o]['AB']['both'][s]; y = b['results'][l3][o]['AB']['both'][s] if b else {}
                    cells += [f"{f(y.get('frac_loss_down_and_margin_down'))} / {f(y.get('mean_dmargin'), 5)}",
                              f"{f(x.get('frac_loss_down_and_margin_down'))} / {f(x.get('mean_dmargin'), 5)}"]
                md_rows.append([m['unit_id'], m['completed_cotrain_epochs'], o, f(m['coefficients_actual'][o]['c'], 4), l4.split('@')[0][0] + '@' + l4.split('@')[1]] + cells)
    if csv_rows:
        with open(f'{HERE}/out/actualscale_layers.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    hdr = []
    for _, h in MD_SUBS:
        hdr += [f'{h} v3原始', f'{h} 实际系数']
    L += ['## 表2 B（仅主项）/ C（主项+intra+noisy）层：v3 原始梯度列 与 实际单元系数列', '',
          '每格 = P(loss↓ 且 观测 margin↓) / 平均 Δ观测 margin（两 peer、两方向合并）。B@update = 更新批；C@heldout = 留出 FIT 批。c=1 的目标与 v3 计算相同，未重跑（N 只在每作业第一个状态上作复现对照）。'
          '全部层（B/C × update/heldout）、全部分组与真对应 margin 见 out/actualscale_layers.csv。', '',
          table(['unit_id', '完成轮', '目标', 'c 实际', '层'] + hdr, md_rows), '']
    # ---------------- grad composition
    rows = []
    for k in sorted(v4):
        a = v4[k]; m = a['meta']; b = v3.get(k)
        for o in m['objectives_run']:
            g4 = a['grad_composition'][o]; g3 = (b or {}).get('grad_composition', {}).get(o, {})
            rows.append([m['unit_id'], m['completed_cotrain_epochs'], o, f(g4['c'], 4), f(g3.get('mean_g_main'), 2), f(g4['mean_g_main_scaled'], 2), f(g4['mean_g_intra'], 3),
                         f(g4['mean_g_noisy'], 3), f((g3.get('ratio_main_intra_quantiles') or {}).get('p50'), 2), f(g4['median_ratio_scaledmain_intra'], 2),
                         f((g3.get('ratio_main_noisy_quantiles') or {}).get('p50'), 2), f(g4['median_ratio_scaledmain_noisy'], 2),
                         f(g3.get('B_clip_rate')), f(g4['B_clip_rate']), f(g3.get('C_clip_rate')), f(g4['C_clip_rate']),
                         f(g3.get('mean_B_update_norm'), 4), f(g4['mean_B_update_norm'], 4), f(g3.get('mean_C_update_norm'), 4), f(g4['mean_C_update_norm'], 4)])
    L += ['## 表3 梯度构成、裁剪与实际更新范数：v3 原始 vs 实际系数', '',
          table(['unit_id', '完成轮', '目标', 'c', '‖g主‖ 原始(v3)', '‖c·g主‖', '‖g intra‖', '‖g noisy‖', '主/intra 中位(v3)', 'c·主/intra 中位', '主/noisy 中位(v3)', 'c·主/noisy 中位',
                 'B裁剪率 v3', 'B裁剪率 实际', 'C裁剪率 v3', 'C裁剪率 实际', 'B更新范数 v3', 'B更新范数 实际', 'C更新范数 v3', 'C更新范数 实际'], rows), '']
    # ---------------- N reproduction control
    rows = []
    for k in sorted(v4):
        a = v4[k]; m = a['meta']
        if not m['n_control_objective'] or k not in v3:
            continue
        for l3, l4 in LAYERS:
            x = a['results'][l4]['N']['AB']['both']['all']; y = v3[k]['results'][l3]['N']['AB']['both']['all']
            rows.append([m['unit_id'], l4.replace('_actualscale', ''), f(y['frac_loss_down_and_margin_down'], 4), f(x['frac_loss_down_and_margin_down'], 4),
                         f(y['mean_dmargin'], 6), f(x['mean_dmargin'], 6), v3[k]['meta']['env']['gpu'].replace('NVIDIA ', ''), m['env']['gpu'].replace('NVIDIA ', '')])
    L += ['## 表4 复现对照：c=1 的 N 目标，v4 重算 vs v3（同批、同参考分组；GPU 不同且 v4 的 S0 用 eval 模式 GRU）', '',
          table(['state', '层', 'P(loss↓&m↓) v3', 'P(loss↓&m↓) v4', 'Δm v3', 'Δm v4', 'GPU v3', 'GPU v4'], rows), '']
    # ---------------- gradient-strength quantiles (aggregation only; includes R29 states)
    rows, gsj = [], {}
    for k in sorted(v3):
        m = v3[k]['meta']
        npz = m.get('gradstrength_file')
        if not npz or not os.path.isfile(npz):
            continue
        z = np.load(npz)
        for o in OBJS:
            cc = common_c(m['cond'], o); ca, src = actual_c(m, o)
            for grp in ('all', 'correct_ref_first', 'correct_ref_not_first', 'mismatch_truepair_absent'):
                v = z[f'g_{o}'][z[f'grp_{grp}']].astype(np.float64)
                if not v.size:
                    continue
                q = lambda c: [float(x) for x in np.quantile(v * c, [0.05, 0.5, 0.95])]
                rec = {'n': int(v.size), 'raw_p5_p50_p95': q(1.0), 'common_c': cc, 'common_p5_p50_p95': q(cc), 'common_frac_below_thr': float((v * cc < THR).mean()),
                       'actual_c': ca, 'actual_c_source': src, 'actual_p5_p50_p95': q(ca), 'actual_frac_below_thr': float((v * ca < THR).mean())}
                gsj[f"{m['unit_id']}|{o}|{grp}"] = rec
                if o in ('PN', 'PV', 'PB', 'L') and grp in ('correct_ref_not_first', 'mismatch_truepair_absent'):
                    rows.append([m['unit_id'], o, grp, rec['n'], f(rec['raw_p5_p50_p95'][1], 4), f(cc, 4), f(rec['common_p5_p50_p95'][1], 4), f(rec['common_frac_below_thr'], 4),
                                 f(ca, 4), src, f(rec['actual_p5_p50_p95'][1], 4), f(rec['actual_frac_below_thr'], 4)])
    L += [f'## 表5 每 query score-space 梯度强度中位数：原始 / 共同参考系数（v3 列）/ 实际单元系数（近零阈值 {THR}）', '',
          '由 v3 已保存的梯度强度 npz 按系数缩放（score-space 梯度与系数成正比，不需 GPU 重算）。表中只列 L/PN/PV/PB 的正确困难与错配·真对应不在批内两组；全部目标与分组（含 p5/p95）在 out/EXPORT_R30.json gradient_strength。', '',
          table(['unit_id', '目标', '分组', 'n', '原始 p50', '共同 c', '共同 p50', '共同近零比例', '实际 c', '实际 c 来源', '实际 p50', '实际近零比例'], rows), '']
    doc['gradient_strength'] = gsj
    # ---------------- branch classes
    rows_s, rows_c, rows_b, bcsv = [], [], [], []
    for k in sorted(br):
        d = br[k]; m = d['meta']
        for peer in ('A', 'B'):
            fc = d['full_train_counts'][peer]
            rs = d['split_recomputed']['split_' + fc['split_from_peer']]
            lg = [v['split_' + fc['split_from_peer']]['n_pred_clean'][1] for v in d['split_logged_vs_recomputed'].values()]
            rows_s.append([m['unit_id'], m['split_epoch'], peer, rs['n_pred_clean'], f(rs['precision'], 5), f(rs['recall'], 5), ','.join(map(str, sorted(set(lg)))) or 'NA (epoch not logged)',
                           f(d['split_exact_match_logged'])])
            rows_c.append([m['unit_id'], m['completed_cotrain_epochs'], peer, fc['main_n'], fc['main_observed_clean'], fc['main_observed_mismatch'], fc['noisy_n'],
                           fc['noisy_observed_clean'], fc['noisy_observed_mismatch']])
        for key, v in d['batches'].items():
            peer, o, c, u, branch, cls = key.split('|')
            nb = d['batches'].get('|'.join([peer, o, c, u, branch, 'TOTAL']), {}).get('n_batches') or 1
            rec = {'state': m['unit_id'], 'completed_cotrain_epochs': m['completed_cotrain_epochs'], 'peer': peer, 'objective': o, 'c': float(c), 'unit': u,
                   'branch': branch, 'class': cls, 'n_batches': nb, **{q: v[q] / nb for q in v if q != 'n_batches'}}
            bcsv.append(rec)
        for o_ in m['objectives']:
            for branch in ('main', 'noisy'):
                for cls in ('TOTAL', 'observed_clean', 'observed_mismatch', 'hard_correct', 'easy_correct'):
                    vals = [r for r in bcsv if r['state'] == m['unit_id'] and r['objective'] == o_['objective'] and r['unit'] == o_['unit_id'] and r['branch'] == branch and r['class'] == cls]
                    if not vals:
                        continue
                    mean = lambda q: float(np.mean([r[q] for r in vals if q in r])) if any(q in r for r in vals) else None
                    rows_b.append([m['unit_id'], m['completed_cotrain_epochs'], o_['unit_id'] if o_['unit_id'] != m['unit_id'] else '-', o_['objective'], f(o_['coefficient'], 4),
                                   branch, cls, f(mean('rows'), 1), f(mean('gnorm_raw'), 3), f(mean('gnorm_scaled'), 3), f(mean('proj_on_branch_raw'), 3),
                                   f(mean('proj_on_branch_scaled'), 3), f(mean('proj_on_G'), 3), f(mean('proj_on_G_postclip'), 3), f(mean('G_norm'), 3), f(mean('clip_factor'), 4)])
    if bcsv:
        keys = []
        for r in bcsv:
            keys += [q for q in r if q not in keys]
        with open(f'{HERE}/out/branch_classes.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, keys, restval=NA); w.writeheader(); w.writerows(bcsv)
    L += ['## 表6 下一轮 GMM 划分：重算 vs 日志', '',
          '重算 = 恢复状态 RNG 后按原生 eval_train + GMM + split_prob 计算；日志 = 该单元（或从该父状态派生的各单元）EPOCHS.json 下一轮 split。peer A 用 pred_B 训练。E40 无下一轮日志。', '',
          table(['state', '划分所属 epoch', '训练 peer', '重算 n_pred_clean', '重算 precision', '重算 recall', '日志 n_pred_clean', '与日志完全一致'], rows_s), '']
    L += ['## 表7 全训练集分支计数（重算划分；观测正确 = 观测配对未被注入错配）', '',
          table(['state', '完成轮', 'peer', '主分支 n', '主·观测正确', '主·观测错配', 'noisy n', 'noisy·观测正确', 'noisy·观测错配'], rows_c), '',
          '训练全过程逐轮计数（由日志 precision × n_pred_clean 得到，36 单元 × 40 轮 × 2 peer）：out/LOGGED_SPLIT_COUNTS.csv。', '']
    L += ['## 表8 固定 FIT 批上的分类梯度贡献（主分支：标注批、单元目标；noisy 分支：未标注批、原生 GraphRefiner+SCE）', '',
          '每 peer 32 个标注批 + 32 个未标注批（各 128，取自该状态自己的划分）；两 peer 平均。rows = 每批该类的方向行数（每样本 i2t+t2i 两行）；hard_correct ⊂ observed_clean。'
          '‖g‖ 为编码器参数梯度范数；投影 = ⟨g_k, g_branch⟩/‖g_branch‖（各类之和 = 分支总梯度范数）；on G = ⟨c·g_k, G⟩/‖G‖，G = c·g主 + g_intra + g_noisy（裁剪前），postclip 乘 min(1, 2/‖G‖)。'
          'noisy 分支无系数（原始 = 缩放）。父状态无 refiner/队列，noisy 为 NA。逐批原始数在 out/branch_classes.csv。', '',
          table(['state', '完成轮', '派生单元', '目标', 'c', '分支', '类别', 'rows/批', '‖g‖原始', '‖g‖缩放', '投影(分支)原始', '投影(分支)缩放', '投影 on G', '投影 on G 裁剪后', '‖G‖', '裁剪因子'], rows_b), '']
    open(f'{HERE}/EXPORT_R30_CN.md', 'w').write('\n'.join(L) + '\n')
    doc['branch'] = {f"{k[0]}_s{k[1]}__{k[2]}": {q: br[k][q] for q in ('meta', 'split_recomputed', 'split_logged_vs_recomputed', 'split_exact_match_logged', 'full_train_counts', 'checks')} for k in br}
    doc['v4_meta'] = {f"{k[0]}_s{k[1]}__{k[2]}": {q: v4[k]['meta'][q] for q in ('unit_id', 'coefficients_actual', 'objectives_run', 'checks', 'env', 'seconds', 'reference')} for k in v4}
    json.dump(doc, open(f'{HERE}/out/EXPORT_R30.json', 'w'), default=float)
    print(json.dumps({k: v for k, v in doc.items() if isinstance(v, int)}))


if __name__ == '__main__':
    main()
