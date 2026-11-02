#!/usr/bin/env python3
"""Numbers-only markdown + compact JSON for the R32 exports (out/UNITS.json from export_logs.py, out/score/*.json from
score_states.py, out/curves_steps.csv). Writes EXPORT_R32_CN.md and out/EXPORT_R32.json. Usage: python3 -B build_md.py"""
import csv, glob, json, math, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NA = 'NA'


def f(x, nd=2):
    if x is None or x == NA or x == '' or (isinstance(x, float) and not math.isfinite(x)):
        return 'NA'
    if isinstance(x, bool):
        return '1' if x else '0'
    if isinstance(x, int):
        return str(x)
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)


def table(h, rows):
    return '\n'.join(['| ' + ' | '.join(h) + ' |', '|' + '---|' * len(h)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def short(u):
    return u.replace('nrcs_v1_', '')


def main():
    U = json.load(open(f'{HERE}/out/UNITS.json'))['units']
    SC = defaultdict(dict)
    for p in glob.glob(f'{HERE}/out/score/*.json'):
        d = json.load(open(p)); SC[d['unit_id']][d['tag']] = d
    steps = defaultdict(list)
    for r in csv.DictReader(open(f'{HERE}/out/curves_steps.csv')):
        if r.get('diag.main.encoder', NA) not in (NA, '') and r['duplicate_of_earlier_attempt'] == 'False':
            steps[r['unit_id']].append(r)
    L = ['# R32 NRCS-v1 既有产物导出（2026-11-02 复盘包 §1 R32；只读，无新训练，未读 TEST）', '',
         '数字表；未完成单元记 NA。来源：nrcs/report.py（原样运行，仅输出目录改到本目录 out/report/）、各单元 history.json / steps.jsonl / '
         'metrics.json / DONE.json，以及 score_states.py 在同一完整 DEV 池（1014 图 / 5070 caption，roles/dev.npz）上的只读重评分。', '']
    st = defaultdict(lambda: defaultdict(int))
    for x in U.values():
        st[x['stage']][x['status']] += 1
    L += ['## 表1 完成度', '', table(['阶段', '状态计数'], [[k, ', '.join(f'{a}={b}' for a, b in v.items())] for k, v in sorted(st.items())]), '',
          '未完成：' + (', '.join(short(u) for u, x in U.items() if x['status'] != 'COMPLETE') or '无（95/95 完成）') + '。', '']
    # S0
    rows = []
    for u, x in sorted(U.items()):
        r = x['report_record'] or {}
        if x['stage'] == 'S0':
            rows.append([short(u), r.get('arm'), r.get('noise_percent'), r.get('seed'), f(r.get('S0_exact_S'), 5), f(r.get('S0_selected_J'), 5),
                         f(r.get('S0_posterior_mse'), 7), r.get('S0_selected_step'), f(r.get('S0_last_J'), 5), f(r.get('S0_last_posterior_mse'), 7)])
    L += ['## 表2 S0 合成：exact S、选中 J、后验 MSE', '', table(['unit', 'arm', 'noise%', 'seed', 'exact S', 'selected J', 'posterior MSE', 'selected step', 'last J', 'last MSE'], rows), '']
    # W2
    rows = []
    for u, x in sorted(U.items()):
        if x['stage'] != 'W2':
            continue
        r = x['report_record'] or {}
        s = SC[u].get('W2_E5')
        g = lambda k, rule='package_stable_ties': f(s['scores'][k][rule]['rsum']) if s else NA
        rows.append([short(u), f(r.get('W2_e5_dev_i2t_r1')), f(r.get('W2_e5_dev_t2i_r1')), f(r.get('W2_e5_dev_rsum')), f(r.get('W2_e5_select_J'), 4),
                     g('h_cos'), g('head_cos'), g('formal'), g('h_cos', 'upstream_argsort'), g('head_cos', 'upstream_argsort'), g('formal', 'upstream_argsort')])
    L += ['## 表3 W2 第5轮原生检索（日志）与同一状态只读评分（DEV rSum）', '',
          'formal = 该单元正式读出（cosine_hinge：线性头输出的 cosine）。tie 规则：pkg = 包 retrieval_metrics（降序、同分按候选序号，训练读出所用）；up = 上游 IN2R i2t/t2i（argsort）。', '',
          table(['unit', 'log i2t R@1', 'log t2i R@1', 'log rSum', 'log SELECT J', 'h_cos pkg', 'head_cos pkg', 'formal pkg', 'h_cos up', 'head_cos up', 'formal up'], rows), '']
    # F1
    rows = []
    for u, x in sorted(U.items()):
        if x['stage'] != 'F1':
            continue
        r = x['report_record'] or {}
        rows.append([short(u), r.get('F1_selected_step'), f(r.get('F1_select_J'), 4), f(r.get('F1_audit_J'), 4), f(r.get('F1_dev_J'), 4), f(r.get('F1_dev_rsum')),
                     f(r.get('F1_last_dev_rsum')), f(r.get('F1_last_audit_J'), 4), f(r.get('F1_audit_ratio_norm_maxabs'), 4), f(r.get('F1_audit_pool_balance'), 4),
                     f(r.get('F1_dev_near_cap_P'), 4), f(r.get('F1_seconds'), 0)])
    L += ['## 表4 F1 固定特征：SELECT-J 选中步与末步', '', table(['unit', 'sel step', 'sel SELECT J', 'sel AUDIT J', 'sel DEV J', 'sel DEV rSum', 'last DEV rSum', 'last AUDIT J',
                                                      'sel AUDIT ratio-norm maxabs', 'sel AUDIT pool balance', 'sel DEV near-cap P', '秒'], rows),
          '', '完整 F1 轨迹：out/curves_f1.csv。', '']
    # E2 / C2 endpoints
    rows = []
    for u, x in sorted(U.items()):
        if x['stage'] not in ('E2', 'C2'):
            continue
        r = x['report_record'] or {}
        rows.append([short(u), x['status'], f(r.get('E20_dev_rsum')), f(r.get('E20_dev_i2t_r1')), f(r.get('E20_dev_t2i_r1')), f(r.get('E20_dev_J'), 4), f(r.get('E20_audit_J'), 4),
                     r.get('sel_epoch', NA), f(r.get('sel_select_J'), 4), f(r.get('sel_dev_rsum')), f(r.get('last5_dev_rsum')),
                     f(r.get('E20_audit_ratio_norm_maxabs'), 4), f(r.get('E20_audit_text_ratio_norm_maxabs'), 4), f(r.get('E20_audit_pool_balance'), 4),
                     f(r.get('E20_dev_near_cap_P'), 4), f(r.get('E20_dev_r_over_cap_max'), 4), r.get('gpu', NA)])
    L += ['## 表5 E2/C2：固定第20轮与 SELECT-J 选中', '',
          'ratio-norm maxabs = 条件 ratio 边缘归一化偏差 max|E_text[r|image]−1|（image）/ 对称（text）；pool balance 与 near-cap 为包 frozen.evaluate 的字段。完整逐轮曲线：out/curves_epoch.csv、out/report/learning_curves.csv。', '',
          table(['unit', '状态', 'E20 DEV rSum', 'E20 i2t R@1', 'E20 t2i R@1', 'E20 DEV J', 'E20 AUDIT J', 'sel epoch', 'sel SELECT J', 'sel DEV rSum', 'last5 DEV rSum',
                 'E20 AUDIT ratio-norm img', 'E20 AUDIT ratio-norm txt', 'E20 AUDIT pool balance', 'E20 DEV near-cap P', 'E20 DEV r/cap max', 'GPU'], rows), '']
    # read-only scores E2 / C2
    rows = []
    for u, x in sorted(U.items()):
        if x['stage'] not in ('E2', 'C2'):
            continue
        for tag in ('E2_zero_update', 'E20', 'selected'):
            if x['stage'] == 'C2' and tag == 'E2_zero_update':
                continue
            s = SC[u].get(tag)
            if not s:
                rows.append([short(u), tag] + [NA] * 10); continue
            g = lambda k, rule: f(s['scores'][k][rule]['rsum'])
            c = s['check_formal_vs_logged']
            rows.append([short(u), tag, g('h_cos', 'package_stable_ties'), g('head_cos', 'package_stable_ties'), g('formal', 'package_stable_ties'),
                         g('h_cos', 'upstream_argsort'), g('head_cos', 'upstream_argsort'), g('formal', 'upstream_argsort'),
                         s['scores']['formal']['unique_scores'], f(c['logged_rsum']), f(c['abs_diff'], 3), s['env']['gpu'].replace('NVIDIA ', '')])
    L += ['## 表6 只读评分：同一状态、同一完整 DEV 池与真值（rSum）', '',
          'E2_zero_update = 由 W2 父状态按 train_gpo.Runner.init_model 构造的子模型，零步更新；E20 = 固定第20轮；selected = SELECT-J 选中。'
          'h_cos：编码器 h 的 cosine；head_cos：同一线性头输出（logits）的 cosine；formal：正式读出。logged = 该检查点自身日志中的 DEV rSum（零更新对应 history 第0轮）；'
          'diff = |formal pkg − logged|（同算法重算，GPU 浮点与近似并列差异，未修正）。', '',
          table(['unit', '状态', 'h_cos pkg', 'head_cos pkg', 'formal pkg', 'h_cos up', 'head_cos up', 'formal up', 'formal 唯一分数数', 'logged', 'diff', 'GPU'], rows), '']
    # code / geometry
    rows = []
    for u, x in sorted(U.items()):
        for tag, s in sorted(SC[u].items()):
            ci, ct, gm = s['head_code_image'], s['head_code_text'], s['mean_direction_geometry']
            rows.append([short(u), tag, f(ci['mean_per_sample_entropy_nats'], 3), f(ci['entropy_of_mean_posterior_nats'], 3), f(ci['effective_components_of_mean_posterior'], 1),
                         ci['hard_occupied_components'], f(ci['logit_std'], 3), f(ct['mean_per_sample_entropy_nats'], 3), f(ct['entropy_of_mean_posterior_nats'], 3),
                         f(ct['effective_components_of_mean_posterior'], 1), ct['hard_occupied_components'], f(ct['logit_std'], 3),
                         f(gm['h']['image_mean_unit_vector_norm'], 3), f(gm['h']['text_mean_unit_vector_norm'], 3), f(gm['h']['cos_image_mean_text_mean'], 3),
                         f(gm['head']['image_mean_unit_vector_norm'], 3), f(gm['head']['text_mean_unit_vector_norm'], 3)])
    L += ['## 表7 DEV 池上的头输出 softmax 编码统计与均值方向几何', '',
          'softmax(线性头输出)：对 ratio 臂即正式单纯形编码；对 cosine 臂只是同一头输出的 softmax（不是其正式读出）。熵单位 nats，K=256（ln K = 5.545）。'
          '均值方向：单位化 h（或头输出）在 DEV 池上的平均向量范数。', '',
          table(['unit', '状态', 'img 样本熵', 'img 均值后验熵', 'img 有效分量', 'img 占用', 'img logit std', 'txt 样本熵', 'txt 均值后验熵', 'txt 有效分量', 'txt 占用',
                 'txt logit std', '|mean h| img', '|mean h| txt', 'cos(mean h img, txt)', '|mean head| img', '|mean head| txt'], rows), '']
    # gradient terms at logged diag steps
    rows = []
    for u, x in sorted(U.items()):
        if x['stage'] not in ('W2', 'E2', 'C2') or x['status'] != 'COMPLETE':
            continue
        rs = steps[u]
        sel = [r for r in rs if r['step'] in ('0', '1')] + ([rs[-1]] if rs else [])
        for r in sel:
            g = lambda k, nd=4: f(r.get(k, NA), nd)
            rows.append([short(u), r['step'], g('J'), g('balance_raw'), g('intra'), g('diag.main.encoder'), g('diag.main.head'), g('diag.balance_weighted.encoder'),
                         g('diag.balance_weighted.head'), g('diag.intra_weighted.encoder'), g('diag.dmain_dlogratio_norm_P', 6), g('diag.dmain_dlogratio_norm_Q', 6),
                         g('diag.T_abs_gt_0.99_fraction'), g('diag.r_over_K_gt_0.95_fraction'), g('diag.critic.image_mean_per_sample_effective_components', 2),
                         g('diag.critic.text_mean_per_sample_effective_components', 2), r.get('diag.critic.image_hard_occupied_components', NA),
                         r.get('diag.critic.text_hard_occupied_components', NA), g('E_P_T'), g('E_Q_T'), g('grad_norm_preclip', 2)])
    L += ['## 表8 训练日志中的 J / 平衡 / intra 与各项梯度（encoder / head 范数），第0、1步与最后一个诊断步', '',
          '诊断步为第0、1步与每200步（train_gpo.py）；全部诊断步与每50步记录见 out/curves_steps.csv（含由有效分量推得的编码熵 = ln(有效分量)）。', '',
          table(['unit', 'step', 'J', 'balance', 'intra', 'main enc', 'main head', 'bal·w enc', 'bal·w head', 'intra·w enc', '|dmain/dlogr| P', '|dmain/dlogr| Q',
                 'P(|T|>.99)', 'P(r>.95K)', 'img 有效分量', 'txt 有效分量', 'img 占用', 'txt 占用', 'E_P T', 'E_Q T', '裁剪前范数'], rows), '']
    open(f'{HERE}/EXPORT_R32_CN.md', 'w').write('\n'.join(L) + '\n')
    json.dump({'units': U, 'scores': {u: {t: {k: v for k, v in s.items() if k not in ('code_sha256',)} for t, s in d.items()} for u, d in SC.items()}},
              open(f'{HERE}/out/EXPORT_R32.json', 'w'), default=str)
    print('md lines', len(L), 'scored states', sum(len(v) for v in SC.values()))


if __name__ == '__main__':
    main()
