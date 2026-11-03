"""diag6 summary (CPU, numbers only; reads diag6 outputs, no TEST file opened). Writes W6/DIAG6_RESULTS.md, GRAD_FIXED_FIT.csv (peer mean),
CLIP_REPLAY.csv, NULL_INVARIANCE_SUMMARY.csv."""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import units6 as U6
import numpy as np

W6, O6 = U6.W6, U6.OUT6


def rd(p):
    return list(csv.DictReader(open(p)))


def f(x, fmt='{:.3f}'):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 'NA' if x in (None, '') else str(x)
    return 'NA' if not np.isfinite(v) else fmt.format(v)


def tbl(rows, cols, fmts=None):
    fmts = fmts or {}
    out = ['|' + '|'.join(cols) + '|', '|' + '|'.join('---' for _ in cols) + '|']
    for r in rows:
        out.append('|' + '|'.join(f(r.get(c), fmts.get(c, '{:.3f}')) if c not in ('parent', 'direction', 'subset', 'label', 'label_a', 'label_b', 'architecture', 'schedule', 'objective', 'family', 'kind', 'parent_a', 'parent_b', 'unit', 'source', 'train_gpu', 'replay_gpu') else str(r.get(c)) for c in cols) + '|')
    return '\n'.join(out)


L = ['# R38 §6 零训练补充读数（diag6；仅数字；开发实验，未读 TEST）', '',
     '代码 `handoff/r38_direction_20261102/diag6/`，原始输出 `outputs_v9/r38_direction_20261102/diag6/`。只读冻结 checkpoint / cache / prep / 排名导出；'
     '不训练新模型；不改任何 R30–R39 冻结文件。作业见 diag6/SUBMITTED.tsv。', '']
I = lambda c: {c: '{:.0f}'}

# ---------------------------------------------------------------- item 1
cr = rd(W6 / 'CC_SUPPORT_CROSSTAB6.csv')
L += ['## 1. CC152K 可信集合 × 原生 Top32 支持（观测关系；真实交集计数）', '',
      'trust = 父模型冻结的原生 GMM 子集（retention = |trust|/150000，是保留率，不是精度）；in_top32 = 记录的观测伙伴在冻结候选表 Top32 中（i2t：图像 k 的查询中 caption k；t2i：caption k 的查询中图像 k）。', '',
      tbl(cr, ['parent', 'direction', 'trust_n', 'trust_retention', 'trust_and_in_top32', 'trust_and_not_in_top32', 'notrust_and_in_top32', 'notrust_and_not_in_top32',
               'frac_of_trusted_not_in_top32', 'frac_of_untrusted_not_in_top32', 'frac_of_in_top32_trusted', 'frac_all_in_top32', 'frac_observed_at_rank1'],
          {**{k: '{:.0f}' for k in ('trust_n', 'trust_and_in_top32', 'trust_and_not_in_top32', 'notrust_and_in_top32', 'notrust_and_not_in_top32')}, **{k: '{:.4f}' for k in ('trust_retention', 'frac_of_trusted_not_in_top32', 'frac_of_untrusted_not_in_top32', 'frac_of_in_top32_trusted', 'frac_all_in_top32', 'frac_observed_at_rank1')}}), '',
      'R37 复盘用四舍五入边缘比例给出的下界（R31 s1001：i2t ≥21.65%、t2i ≥25.18% 的可信记录不在 Top32）对应的实测交集见上表 frac_of_trusted_not_in_top32。', '']
bins = rd(W6 / 'CC_SUPPORT_RANK_BINS.csv')
bc = [c for c in bins[0] if c.startswith('pct_rank_')]
L += ['### 观测伙伴的原生 H 排名分布（全池 150000，1 = 第一名；占该子集 %）', '', tbl(bins, ['parent', 'direction', 'subset', 'n'] + bc, {'n': '{:.0f}', **{c: '{:.2f}' for c in bc}}), '']
q = rd(W6 / 'CC_SUPPORT_RANK_QUANTILES.csv')
L += ['### 排名分位数（1-based）', '', tbl(q, ['parent', 'direction', 'subset', 'rank1based_median', 'rank1based_p75', 'rank1based_p90', 'rank1based_p99', 'rank1based_max', 'mean_log10_rank1based'],
                                       {**{c: '{:.0f}' for c in ('rank1based_median', 'rank1based_p75', 'rank1based_p90', 'rank1based_p99', 'rank1based_max')}}), '']
jt = rd(W6 / 'CC_SUPPORT_JOINT.csv')
L += ['### 记录级双向联合在场（in_top32）', '', tbl(jt, ['parent', 'subset', 'n', 'both_dirs_in_top32', 'only_i2t_in', 'only_t2i_in', 'neither_in'], {c: '{:.0f}' for c in ('n', 'both_dirs_in_top32', 'only_i2t_in', 'only_t2i_in', 'neither_in')}), '']
ss = U6.read_json(O6 / 'support' / 'SUPPORT_SUMMARY.json')['table_consistency']
L += ['排名重算与冻结候选表一致性（rank<32 与表内成员不一致的记录数；均为并列分数）：' + '；'.join(f"{k} i2t {v['i2t']['n_mismatch_rank_lt32_vs_table']}（并列 {v['i2t']['n_mismatch_with_ties']}）/ t2i {v['t2i']['n_mismatch_rank_lt32_vs_table']}（并列 {v['t2i']['n_mismatch_with_ties']}）" for k, v in ss.items()) + '。重算在 A100（runfill），候选表由各自 prep 作业在其 GPU 上生成。', '']

# ---------------------------------------------------------------- item 5
pp = rd(W6 / 'CC_PARENT_PAIRS.csv')
L += ['## 5. R37S（R30-style）父模型与 R31 父模型的候选支持 / 可信保留差（CC152K）', '', 'R31 种子间（同配方）配对作为参照。', '',
      tbl(pp, ['parent_a', 'parent_b', 'kind', 'trust_retention_a', 'trust_retention_b', 'trust_retention_diff_a_minus_b', 'trust_both', 'trust_only_a', 'trust_only_b', 'trust_neither', 'trust_jaccard'],
          {**{c: '{:.4f}' for c in ('trust_retention_a', 'trust_retention_b', 'trust_retention_diff_a_minus_b', 'trust_jaccard')}, **{c: '{:.0f}' for c in ('trust_both', 'trust_only_a', 'trust_only_b', 'trust_neither')}}), '']
for dn in ('i2t', 't2i'):
    cols = [f'{dn}_in_top32_a', f'{dn}_in_top32_b', f'{dn}_in_top32_diff_a_minus_b', f'{dn}_present_both', f'{dn}_present_only_a', f'{dn}_present_only_b', f'{dn}_present_neither',
            f'{dn}_top32_mean_overlap_frac', f'{dn}_top32_identical_set_frac', f'{dn}_top1_agree_frac', f'{dn}_observed_rank_spearman']
    L += [f'### {dn}', '', tbl(pp, ['parent_a', 'parent_b'] + cols, {**{c: '{:.4f}' for c in cols}, **{c: '{:.0f}' for c in cols if 'present_' in c}}), '']

# ---------------------------------------------------------------- item 2
eu = rd(W6 / 'ERR_DIRECTIONAL_RECALL.csv'); ep = rd(W6 / 'ERR_PAIRWISE.csv'); ef = rd(W6 / 'ERR_FAMILY_SUMMARY.csv')
L += ['## 2. Flickr60 三父状态：分方向 R@1/5/10、救回/丢失、方法间共同/互补错误', '',
      '来源：各单元训练时保存的 dev_ranks_step8000.npz 与 R38 summary 的原生排名（未重算分数）。t2i 按图像分组（同一 DEV 图像 5 条 caption 一组）。完整单元表 ERR_DIRECTIONAL_RECALL.csv，全部两两比较 ERR_PAIRWISE.csv（含图像簇 bootstrap 95% CI，B=2000）。', '']
key = [r for r in eu if (r['architecture'] in ('shared', '-') and r['schedule'] in ('constant', '-')) or r['line'] == 'R38']
L += ['### 单元（R@1/5/10 与相对 H 的第一名救回/丢失；t2i 另给图像级）', '',
      tbl(key, ['parent', 'label', 'direction', 'r1', 'r5', 'r10', 'd_r1_vs_H', 'rescued_r1', 'lost_r1', 'rescued_r5', 'lost_r5', 'rescued_r10', 'lost_r10', 'img_more_r1_than_H', 'img_fewer_r1_than_H', 'img_with_rescue', 'img_with_loss'],
          {**{c: '{:.2f}' for c in ('r1', 'r5', 'r10', 'd_r1_vs_H')}, **{c: '{:.0f}' for c in ('rescued_r1', 'lost_r1', 'rescued_r5', 'lost_r5', 'rescued_r10', 'lost_r10', 'img_more_r1_than_H', 'img_fewer_r1_than_H', 'img_with_rescue', 'img_with_loss')}}), '']
L += ['### 方法族：H 错误被任一/全部方法救回、无方法能修的错误、互补程度', '',
      tbl(ef, ['parent', 'family', 'n_methods', 'direction', 'H_errors', 'H_errors_rescued_by_any', 'H_errors_rescued_by_all', 'H_errors_no_method_fixes', 'H_correct_lost_by_any', 'H_correct_lost_by_all', 'H_r1', 'best_single_r1', 'oracle_union_r1', 'mean_pairwise_jaccard_errors'],
          {**{c: '{:.0f}' for c in ('n_methods', 'H_errors', 'H_errors_rescued_by_any', 'H_errors_rescued_by_all', 'H_errors_no_method_fixes', 'H_correct_lost_by_any', 'H_correct_lost_by_all')}, **{c: '{:.2f}' for c in ('H_r1', 'best_single_r1', 'oracle_union_r1')}}), '',
      'oracle_union_r1 = 任一方法第一名正确的比例（事后上界，非可部署方法）。shared_constant_N_T 族的各方法独有救回数：', '']
for r in ef:
    if r['family'] == 'shared_constant_N_T':
        L.append(f"- {r['parent']} {r['direction']}: {r['unique_rescues_per_method']}")
L.append('')


def pick(lab_a, lab_b):
    return [r for r in ep if (r['label_a'], r['label_b']) in ((lab_a, lab_b),) or (r['label_b'], r['label_a']) == (lab_a, lab_b)]


sel = []
for r in ep:
    la, lb = r['label_a'], r['label_b']
    sc = lambda s: s == 'H' or ('-sha-const' in s and (':N-' in s or ':T-' in s))
    if sc(la) and sc(lb):
        sel.append(r)
    elif ('R38:N-dir-' in la or 'R38:N-dir-' in lb) and ('-sha-' in la or '-sha-' in lb) and la.split('-')[-1] == lb.split('-')[-1] and la.split('-')[-2] == lb.split('-')[-2]:
        sel.append(r)
L += ['### 关键两两比较（H 与 shared-constant N/T 之间；R38 directional 与同目标同日程 shared）', '', 'd_r1_pp = R@1(a) − R@1(b)（百分点）；CI 为图像簇 bootstrap 95%；both_wrong_same_top1 = 两者都错且第一名是同一个错误项。', '',
      tbl(sel, ['parent', 'label_a', 'label_b', 'direction', 'both_correct', 'only_a_correct', 'only_b_correct', 'both_wrong', 'both_wrong_same_top1', 'jaccard_errors', 'd_r1_pp', 'd_r1_ci95_lo', 'd_r1_ci95_hi', 'img_a_better', 'img_b_better'],
          {**{c: '{:.0f}' for c in ('both_correct', 'only_a_correct', 'only_b_correct', 'both_wrong', 'both_wrong_same_top1', 'img_a_better', 'img_b_better')}, **{c: '{:.2f}' for c in ('d_r1_pp', 'd_r1_ci95_lo', 'd_r1_ci95_hi')}}), '']

# ---------------------------------------------------------------- item 3 / 4
grad_rows, clip_rows, null_rows, fid = [], [], [], []
for pk in U6.PARENTS:
    d = O6 / 'grad_replay' / pk
    if not (d / 'SUMMARY.json').exists():
        continue
    S = U6.read_json(d / 'SUMMARY.json')
    for u, rec in S['units'].items():
        r = rec.get('replay')
        if r:
            clip_rows.append(dict(parent=pk, unit=u, architecture=rec['architecture'], schedule=rec['schedule'], objective=rec['objective'], source=rec['source'],
                                  train_gpu=rec['train_gpu'], replay_gpu=S['gpu'], trigger_rate=r['trigger_rate'], trigger_rate_peerA=r['trigger_rate_by_peer'][0], trigger_rate_peerB=r['trigger_rate_by_peer'][1],
                                  trigger_rate_steps_lt250=r['trigger_rate_steps_lt250'], trigger_rate_steps_250_2000=r['trigger_rate_steps_250_2000'], trigger_rate_steps_2000_8000=r['trigger_rate_steps_2000_8000'],
                                  n_triggered_only_because_of_null=r['n_triggered_only_because_of_null'], mean_clip_coef_when_triggered=r['mean_clip_coef_when_triggered'], min_clip_coef=r['min_clip_coef'],
                                  preclip_norm_mean=r['preclip_norm_mean'], preclip_norm_p99=r['preclip_norm_p99'], preclip_norm_max=r['preclip_norm_max'], null_share_of_norm_sq_mean=r['null_share_of_norm_sq_mean'],
                                  chains_equal=r['fidelity']['all_chains_equal'], states_bitwise=r['fidelity']['all_states_bitwise'], max_head_state_abs_diff=r['fidelity']['max_head_state_abs_diff'],
                                  max_null_state_abs_diff=r['fidelity']['max_null_state_abs_diff'], max_window_norm_abs_diff=max([x['abs_diff'] for x in r['window_grad_norm_crosscheck']] or [float('nan')])))
    if (d / 'grad_fixed_fit.csv').exists():
        acc = defaultdict(list)
        for r in rd(d / 'grad_fixed_fit.csv'):
            acc[(r['unit'], r['architecture'], r['schedule'], r['objective'], r['source'], int(r['step']), r['quantity'])].append(r)
        for (u, a, s, o, src, st, qn), rs in acc.items():
            grad_rows.append(dict(parent=pk, unit=u, architecture=a, schedule=s, objective=o, source=src, step=st, quantity=qn,
                                  full_set_peer_mean=float(np.mean([float(x['full_set']) for x in rs])), batch_mean_peer_mean=float(np.mean([float(x['batch_mean']) for x in rs])),
                                  batch_frac_negative_peer_mean=float(np.mean([float(x['batch_frac_negative']) for x in rs]))))
    if (d / 'null_invariance.csv').exists():
        null_rows += [dict(r, parent=pk) for r in rd(d / 'null_invariance.csv')]
U6.wcsv(W6 / 'GRAD_FIXED_FIT.csv', grad_rows); U6.wcsv(W6 / 'CLIP_REPLAY.csv', clip_rows)
ns = []
by = defaultdict(list)
for r in null_rows:
    by[(r['objective'],)].append(r)
for (o,), rs in sorted(by.items()):
    ns.append(dict(objective=o, n_checks=len(rs), n_real_logit_grad_bitwise_equal=sum(r['real_logit_grad_bitwise_equal'] == 'True' for r in rs),
                   max_real_logit_grad_abs_diff=max(float(r['real_logit_grad_max_abs_diff']) for r in rs), max_real_logit_grad_abs=max(float(r['real_logit_grad_max_abs']) for r in rs),
                   n_head_param_grad_bitwise_equal=sum(r['head_param_grad_bitwise_equal'] == 'True' for r in rs), max_head_param_grad_abs_diff=max(float(r['head_param_grad_max_abs_diff']) for r in rs),
                   n_clip_coef_changed=sum(abs(float(r['clip_coef_base']) - float(r['clip_coef_shifted'])) > 0 for r in rs), n_clip_active_base=sum(float(r['clip_coef_base']) < 1 for r in rs),
                   max_rel_change_preclip_norm=max(abs(float(r['preclip_norm_shifted']) / float(r['preclip_norm_base']) - 1) for r in rs)))
U6.wcsv(W6 / 'NULL_INVARIANCE_SUMMARY.csv', ns)

L += ['## 3. shared head 固定 FIT 批次上的 i2t / t2i 梯度夹角与对完整更新的投影（诊断，不作性能原因）', '',
      '固定 FIT 批次 = R35/R37C 锁定的 list-eval 记录（lev_i / lev_t 各 1024），N 33 槽列表，按 16 行切成 64 批（与训练 n_queries=16 同）；梯度 g_d = ∂(0.25·L_d)/∂θ（训练损失的方向拆分），'
      '每 peer 分别算后取两 peer 均值。full = 64 批梯度平均后的夹角；batch = 逐批夹角的均值与负值比例。adam_share_d = ⟨g_d, Δ⟩/⟨G, Δ⟩，Δ = 保存的 AdamW 状态在该批 G=g_i2t+g_t2i 上（训练同款逐 peer 裁剪 2.0，含解耦 weight decay）将执行的更新；'
      'euclid_share_d = ⟨g_d, G⟩/|G|²。trunk = GlobalMLP（shared 与 directional 的共享部分）；head = 除 null 外全部头参数；null 两个分量分属两个方向，故 cos_null ≡ 0。完整表 GRAD_FIXED_FIT.csv（peer 均值；逐 peer 原始在 outputs diag6/grad_replay/<parent>/grad_fixed_fit.csv）。', '']
G = defaultdict(dict)
for r in grad_rows:
    G[(r['parent'], r['architecture'], r['schedule'], r['objective'], r['step'])][r['quantity']] = r
rows3 = []
for (pk, a, s, o, st), dq in sorted(G.items()):
    if st not in (0, 1000, 4000, 8000):
        continue
    g = lambda qn, k='full_set_peer_mean': dq.get(qn, {}).get(k)
    rows3.append(dict(parent=pk, architecture=a, schedule=s, objective=o, step=st, cos_trunk_full=g('cos_trunk'), cos_trunk_batch=g('cos_trunk', 'batch_mean_peer_mean'), frac_neg_trunk_batch=g('cos_trunk', 'batch_frac_negative_peer_mean'),
                      cos_head_full=g('cos_head'), cos_head_batch=g('cos_head', 'batch_mean_peer_mean'), adam_share_i2t_full=g('adam_share_i2t'), adam_share_i2t_batch=g('adam_share_i2t', 'batch_mean_peer_mean'),
                      euclid_share_i2t_batch=g('euclid_share_i2t', 'batch_mean_peer_mean'), norm_ratio_i2t_over_t2i_head_batch=(g('norm_i2t_head', 'batch_mean_peer_mean') / g('norm_t2i_head', 'batch_mean_peer_mean')) if g('norm_t2i_head', 'batch_mean_peer_mean') else None))
L += [tbl([r for r in rows3 if r['architecture'] == 'shared'], ['parent', 'schedule', 'objective', 'step', 'cos_trunk_full', 'cos_trunk_batch', 'frac_neg_trunk_batch', 'cos_head_full', 'cos_head_batch', 'adam_share_i2t_full', 'adam_share_i2t_batch', 'euclid_share_i2t_batch', 'norm_ratio_i2t_over_t2i_head_batch'],
          {'step': '{:.0f}'}), '',
      '### directional（同读数；trunk 为共享部分，输出层各方向独立）', '',
      tbl([r for r in rows3 if r['architecture'] == 'directional'], ['parent', 'schedule', 'objective', 'step', 'cos_trunk_full', 'cos_trunk_batch', 'frac_neg_trunk_batch', 'cos_head_full', 'cos_head_batch', 'adam_share_i2t_full', 'adam_share_i2t_batch'], {'step': '{:.0f}'}), '']
# compact pooled view: per dataset x architecture x schedule x objective, mean over parents of the batch-mean trunk/head cos at steps 1000..8000
pool = defaultdict(list)
for r in rows3:
    if r['step'] >= 1000:
        pool[('Flickr' if r['parent'].startswith('f60') else ('CC-R37S' if 'R37S' in r['parent'] else 'CC-R31'), r['architecture'], r['schedule'], r['objective'])].append(r)
pr = [dict(dataset=k[0], architecture=k[1], schedule=k[2], objective=k[3], n=len(v), cos_trunk_full_mean=float(np.nanmean([x['cos_trunk_full'] for x in v])),
           cos_head_full_mean=float(np.nanmean([x['cos_head_full'] for x in v])), cos_head_batch_mean=float(np.nanmean([x['cos_head_batch'] for x in v])),
           adam_share_i2t_batch_mean=float(np.nanmean([x['adam_share_i2t_batch'] for x in v]))) for k, v in sorted(pool.items())]
L += ['注：adam_share / euclid_share 是比值，分母（⟨G,Δ⟩ 或 |G|²）在两方向抵消时很小，逐批均值会出现 >1 或 <0 的值；以 full 行与逐批夹角为主读数。directional 的 cos_head 接近 0 是因为两方向的输出层/仿射参数块互不重叠（只有 trunk 共享）。', '']
L += ['### 合并（每数据集，父模型与 step∈{1000,4000,8000} 的均值）', '', tbl(pr, ['dataset', 'architecture', 'schedule', 'objective', 'n', 'cos_trunk_full_mean', 'cos_head_full_mean', 'cos_head_batch_mean', 'adam_share_i2t_batch_mean'], {'n': '{:.0f}'}), '']

L += ['## 4. null 标量与 real-logit 梯度；共享裁剪触发率', '',
      '### 4a 数值核验（裁剪前）', '', 'N 单元在 step 0/1000/8000 的 4 个固定 FIT 批次上，每 peer 把两个 null 标量同时平移 δ∈{−4,−1,+1,+4}：比较 real 槽的 ∂L/∂g 与头参数梯度（与未平移逐位比较），以及每 peer 裁剪系数（null 只能经共享裁剪间接耦合）。', '',
      tbl(ns, ['objective', 'n_checks', 'n_real_logit_grad_bitwise_equal', 'max_real_logit_grad_abs_diff', 'max_real_logit_grad_abs', 'n_head_param_grad_bitwise_equal', 'max_head_param_grad_abs_diff', 'n_clip_active_base', 'n_clip_coef_changed', 'max_rel_change_preclip_norm'],
          {**{c: '{:.0f}' for c in ('n_checks', 'n_real_logit_grad_bitwise_equal', 'n_head_param_grad_bitwise_equal', 'n_clip_active_base', 'n_clip_coef_changed')}, **{c: '{:.3g}' for c in ('max_real_logit_grad_abs_diff', 'max_real_logit_grad_abs', 'max_head_param_grad_abs_diff', 'max_rel_change_preclip_norm')}}), '',
      '### 4b 实际共享裁剪触发率（记录批次重放）', '',
      '对每个 N 单元，从每个保存的 checkpoint 按冻结续跑路径恢复（头、null、AdamW 状态、numpy/torch RNG），用冻结的 train_unit.step_lists / clip_grad_norm_ / 日程 / AdamW 重新执行记录的批次至下一个 checkpoint；'
      '批索引哈希链必须等于 TRAIN_HISTORY 记录；段末头/null 状态与保存的 checkpoint 比较。触发 = 每 peer 裁剪前范数（头+null 同组）+1e-6 > 2.0；only_null = 触发但仅头部分范数不触发。8000 次更新 × 2 peer。'
      'max_window_norm_abs_diff = 重放的窗口平均裁剪前范数与训练日志 train_window.grad_norm_mean 之差。重放在 A100（runfill）上，训练在 L40S/RTX PRO 6000：状态非逐位，差值列在表中。', '',
      tbl(clip_rows, ['parent', 'unit', 'train_gpu', 'trigger_rate', 'trigger_rate_steps_lt250', 'trigger_rate_steps_250_2000', 'trigger_rate_steps_2000_8000', 'n_triggered_only_because_of_null', 'min_clip_coef', 'preclip_norm_mean', 'preclip_norm_p99', 'preclip_norm_max', 'null_share_of_norm_sq_mean', 'chains_equal', 'max_head_state_abs_diff', 'max_null_state_abs_diff', 'max_window_norm_abs_diff'],
          {**{c: '{:.4f}' for c in ('trigger_rate', 'trigger_rate_steps_lt250', 'trigger_rate_steps_250_2000', 'trigger_rate_steps_2000_8000', 'min_clip_coef', 'preclip_norm_mean', 'preclip_norm_p99', 'preclip_norm_max', 'null_share_of_norm_sq_mean')}, 'n_triggered_only_because_of_null': '{:.0f}', **{c: '{:.2g}' for c in ('max_head_state_abs_diff', 'max_null_state_abs_diff', 'max_window_norm_abs_diff')}}), '']
agg = defaultdict(list)
for r in clip_rows:
    agg[('Flickr' if r['parent'].startswith('f60') else 'CC', r['objective'])].append(r)
L += ['合并：' + '；'.join(f"{k[0]} {k[1]}: 触发率均值 {np.mean([x['trigger_rate'] for x in v]):.5f}（最大 {max(x['trigger_rate'] for x in v):.5f}，n={len(v)}），only_null 合计 {sum(x['n_triggered_only_because_of_null'] for x in v)}" for k, v in sorted(agg.items())) + '。', '']
L += ['文件：CC_SUPPORT_CROSSTAB6.csv、CC_SUPPORT_RANK_BINS.csv、CC_SUPPORT_RANK_QUANTILES.csv、CC_SUPPORT_JOINT.csv、CC_PARENT_PAIRS.csv、ERR_DIRECTIONAL_RECALL.csv、ERR_PAIRWISE.csv、ERR_FAMILY_SUMMARY.csv、GRAD_FIXED_FIT.csv、CLIP_REPLAY.csv、NULL_INVARIANCE_SUMMARY.csv。', '']
(W6 / 'DIAG6_RESULTS.md').write_text('\n'.join(L))
print('DIAG6_SUMMARY_DONE', len(clip_rows), len(grad_rows), len(null_rows))
