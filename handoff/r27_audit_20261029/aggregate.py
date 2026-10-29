"""R27 audit aggregation -> AUDIT_SUMMARY.json + AUDIT_CN.md (numbers, settings, evidence paths; no interpretation).
Usage: aggregate.py [OUT_ROOT]   (default /projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029). CPU only.
"""
import os, sys, json, glob
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else '/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029'
HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(ROOT, 'audit1_graphs')
G3 = os.path.join(ROOT, 'audit3_ranks')
TGT = ('mean', 'delta', 'vcs', 'bce', 'snce')
CELLS = ('f40_s1001', 'f40_s1002', 'f40_s1003', 'f60_s1001', 'f60_s1002', 'f60_s1003')


def q(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    nf = int((~np.isfinite(a)).sum()); a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, n_nonfinite=nf)
    return dict(n=int(a.size), n_nonfinite=nf, mean=float(a.mean()), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                p99=float(np.quantile(a, .99)), max=float(a.max()), min=float(a.min()))


def f(x, p=4):
    if x is None:
        return '—'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f'{x:.2e}'
    return f'{x:.{p}f}'


def ckpt_summary(cell, label):
    d = os.path.join(G1, cell, label)
    r = json.load(open(os.path.join(d, 'RESULT.json')))
    if r.get('status') != 'OK':
        return r
    S = dict(cell=cell, label=label, status='OK', checkpoint=r['checkpoint'], state_epoch=r['state_epoch'],
             has_ids=r['state_has_queue_ids'], gpu=r['gpu'], job=r['job'], peers={k: {kk: vv for kk, vv in v.items() if kk != 'batch_capid_sha16'} for k, v in r['peers'].items()},
             n_graphs=0, by_branch={})
    for br in ('i2t', 't2i'):
        gs = [g for g in r['graphs'] if g['graph'].endswith(br) and g['status'] == 'ok']
        if not gs:
            continue
        S['n_graphs'] += len(gs)
        arr = {}
        for g in gs:
            z = np.load(os.path.join(d, 'graphs', g['graph'] + '.npz'))
            for k in z.files:
                arr.setdefault(k, []).append(z[k])
        arr = {k: np.concatenate(v) for k, v in arr.items()}
        B = dict(n_graphs=len(gs), n_rows=int(sum(g['B'] for g in gs)), M_mean=float(np.mean([g['M'] for g in gs])),
                 no_proto=int(sum(1 for g in r['graphs'] if g['graph'].endswith(br) and g['status'] != 'ok')))
        B['pair_angles'] = {f'{a}|{b}': q(arr[f'angle_{a}_{b}']) for i, a in enumerate(TGT) for b in TGT[i + 1:]}
        B['raw_norm'] = {t: q(arr[f'rawnorm_{t}']) for t in TGT}
        B['objectives'] = {o: {f'{kind}_{t}': q(arr[f'obj_{o}_{kind}_{t}']) for t in TGT for kind in ('raw', 'norm')} for o in ('vcs', 'bce', 'snce')}
        B['counts'] = {t: {k: int(sum(g[f'prod_{t}'].get(k, 0) for g in gs)) for k in ('n_rows', 'invalid_rows', 'fallback', 'cap', 'interior', 'near_log_boundary')} for t in TGT}
        B['iters'] = {t: dict(mean=float(np.mean([g[f'prod_{t}']['iters_mean'] for g in gs])), max=int(max(g[f'prod_{t}']['iters_max'] for g in gs)))
                      for t in ('vcs', 'bce', 'snce')}
        B['rel_max'] = {t: float(max(g[f'prod_{t}']['rel_max'] for g in gs)) for t in ('vcs', 'bce', 'snce')}
        B['fp32_vs_fp64_maxabs'] = {t: float(max((g[f'prod_{t}']['fp32_target_vs_fp64_maxabs'] or 0) for g in gs)) for t in TGT}
        ck = {}
        for t in TGT:
            ks = [k for k in gs[0]['collisions'][t]]
            ck[t] = {k: int(sum(g['collisions'][t][k] for g in gs)) for k in ks}
        B['collisions'] = ck
        B['tight'] = {'vcs_exact_tol1e-9_angle_max': float(max(g['tight_tolerance']['vcs_exact_tol1e-9']['angle_vs_prod']['max'] for g in gs)),
                      'vcs_spg_1e-9_vs_exact': q(arr['tight_angle_vcs_spg']),
                      'bce_spg_1e-9_vs_prod': q(arr['tight_angle_bce_spg']), 'snce_spg_1e-9_vs_prod': q(arr['tight_angle_snce_spg']),
                      'bce_spg_1e-9_cap': int(sum(g['tight_tolerance']['bce_spg_tol1e-9_vs_prod']['cap'] for g in gs)),
                      'snce_spg_1e-9_cap': int(sum(g['tight_tolerance']['snce_spg_tol1e-9_vs_prod']['cap'] for g in gs)),
                      'bce_obj_tight_minus_prod_max': float(max(g['tight_tolerance']['bce_spg_tol1e-9_vs_prod']['obj_tight_minus_prod']['max'] for g in gs)),
                      'snce_obj_tight_minus_prod_max': float(max(g['tight_tolerance']['snce_spg_tol1e-9_vs_prod']['obj_tight_minus_prod']['max'] for g in gs))}
        pairs = gs[0]['grad_x']['pairs'].keys()
        B['grad'] = {p: dict(cos_median=float(np.median([g['grad_x']['pairs'][p]['cos'] for g in gs if p in g['grad_x']['pairs']])),
                             cos_min=float(np.min([g['grad_x']['pairs'][p]['cos'] for g in gs if p in g['grad_x']['pairs']])),
                             rel_norm_median=float(np.median([g['grad_x']['pairs'][p]['rel_norm'] for g in gs if p in g['grad_x']['pairs']])))
                     for p in pairs}
        B['sce_loss_mean'] = {k: float(np.mean([g['grad_x']['sce_loss'][k] for g in gs if k in g['grad_x']['sce_loss']])) for k in gs[0]['grad_x']['sce_loss']}
        a2 = {}
        for gen in ('bce', 'snce'):
            for key in ('spg_max|1e-06', 'spg_min|1e-06', 'spg_max|1e-09', 'spg_min|1e-09', 'fista|1e-09'):
                rows = [g['audit2_spg'][f'{gen}|{key}'] for g in gs]
                e = dict(seconds_total=float(sum(x['seconds'] for x in rows)), seconds_mean=float(np.mean([x['seconds'] for x in rows])),
                         iters_mean=float(np.mean([x['iters_mean'] for x in rows])), iters_max=int(max(x['iters_max'] for x in rows)),
                         cap=int(sum(x['cap'] for x in rows)), rel_max=float(max(x['rel_max'] for x in rows)))
                if key != 'fista|1e-09':
                    rr, tol = key.split('|')
                    e['angle_vs_fista'] = q(arr[f'a2_angle_vs_fista_{gen}_{rr[4:]}_{tol}'])
                    e['obj_fista_minus_max'] = float(max(x['obj_fista_minus']['max'] for x in rows))
                    e['obj_fista_minus_mean'] = float(np.mean([x['obj_fista_minus']['mean'] for x in rows]))
                    e['cert_excess_max'] = float(max(x['cert_excess_max'] for x in rows))
                    e['decrease_accepts'] = int(sum(x['decrease_accepts'] for x in rows))
                    e['rejects'] = int(sum(x['rejects'] for x in rows))
                    if key == 'spg_max|1e-06':
                        e['audit_copy_bitwise_equal_production_all'] = bool(all(x['bitwise_equal_production'] for x in rows))
                a2[f'{gen}|{key}'] = e
            a2[f'{gen}|angle max_vs_min 1e-6'] = q(np.array([g['audit2_spg'][f'angle {gen}|spg_max|1e-06 vs {gen}|spg_min|1e-06']['max'] for g in gs]))
            if 'audit2_slsqp' in gs[0]:
                sl = [o for g in gs if 'audit2_slsqp' in g for o in g['audit2_slsqp'][gen]]
                a2[f'{gen}|slsqp'] = dict(n=len(sl), feasible=int(sum(o['feasible'] for o in sl)), success=int(sum(o['success'] for o in sl)),
                                          seconds_mean=float(np.mean([o['seconds'] for o in sl])), nit_max=int(max(o['nit'] for o in sl)),
                                          angle_vs_fista_max=max([o['angle_vs_fista'] for o in sl if o['feasible']], default=None),
                                          angle_vs_prod_max=max([o['angle_vs_prod'] for o in sl if o['feasible']], default=None),
                                          angle_vs_spg_min_max=max([o['angle_vs_spg_min'] for o in sl if o['feasible']], default=None),
                                          obj_fista_minus_slsqp_max=max([o['obj_fista_minus'] for o in sl if o['feasible']], default=None))
        for key in ('spg_max|1e-06', 'spg_min|1e-06', 'spg_max|1e-09', 'spg_min|1e-09'):
            rows = [g['audit2_spg'][f'vcs|{key}'] for g in gs]
            a2[f'vcs|{key}'] = dict(seconds_total=float(sum(x['seconds'] for x in rows)), iters_mean=float(np.mean([x['iters_mean'] for x in rows])),
                                    iters_max=int(max(x['iters_max'] for x in rows)), cap=int(sum(x['cap'] for x in rows)),
                                    angle_vs_exact_max=float(max(x['angle_vs_exact']['max'] for x in rows)),
                                    obj_exact_minus_max=float(max(x['obj_exact_minus']['max'] for x in rows)))
        a2['vcs|exact_solver_seconds_total'] = float(sum(g['prod_vcs']['seconds_solver'] for g in gs))
        B['audit2'] = a2
        S['by_branch'][br] = B
    return S


def audit3():
    rows = []
    for p in sorted(glob.glob(os.path.join(G3, '*', '*', 'RESULT.json'))):
        r = json.load(open(p))
        rows.append(dict(cell=r['cell'], arm=r['arm'], epoch=r['model_best_epoch'], epochs_best_epoch=r['epochs_best_epoch'],
                         logged_rsum=r['logged_dev_at_model_best_epoch']['rsum'] if r['logged_dev_at_model_best_epoch'] else None,
                         recomputed_rsum=r['recomputed_from_saved_ranks']['rsum'], max_abs_recall_diff_pp=r['max_abs_recall_diff_pp'],
                         rsum_diff=r['diff_recomputed_minus_logged']['rsum'] if r['diff_recomputed_minus_logged'] else None,
                         within_tolerance=r['within_tolerance'], recomputed_equals_host_now=r['recomputed_equals_host_now'],
                         gpu=r['gpu'], job=r['job'], dir=os.path.dirname(p)))
    return rows


def main():
    A1 = []
    for cell in CELLS:
        for d in sorted(glob.glob(os.path.join(G1, cell, '*', 'RESULT.json'))):
            A1.append(ckpt_summary(cell, os.path.basename(os.path.dirname(d))))
    A3 = audit3()
    checks = sorted(glob.glob(os.path.join(HERE, 'CHECK_*.json')))
    chk = json.load(open(checks[-1])) if checks else None
    out = dict(root=ROOT, audit1_2=A1, audit3=A3, check=os.path.basename(checks[-1]) if checks else None)
    json.dump(out, open(os.path.join(HERE, 'AUDIT_SUMMARY.json'), 'w'), indent=1, default=float)
    write_md(A1, A3, chk, checks[-1] if checks else None)


def write_md(A1, A3, chk, chkpath):
    L = []
    w = L.append
    w('# R27 只读审计（补充包 v1.1 SERVER_ADDENDUM「只读审计」）—— 数字与设置，无解释\n')
    w('日期：2026-10-29。由 `aggregate.py` 从结果文件生成；重跑作业后重新运行 `aggregate.py` 即可更新。')
    w('代码：`/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029/`（只读导入冻结树 `r27_qtarget_20261029/code_v11`，未修改 `code/`、`code_v11/`）。')
    w(f'结果根目录：`{ROOT}`。汇总 JSON：`AUDIT_SUMMARY.json`（同目录）。提交记录：`SUBMITTED.tsv`。\n')
    w('## 0. 门禁（CPU 小例）\n')
    if chk:
        w(f'文件 `{os.path.basename(chkpath)}`；all_pass = {chk["all_pass"]}。')
        w(f'- spg_max_equals_production：{chk["spg_max_equals_production"]["pass_"]}（{chk["spg_max_equals_production"]["n_cases"]} 例，Z/iters/rel 逐位相等）')
        sm = chk['spg_min_converges']
        w(f'- spg_min_converges：{sm["pass_"]}（min-history 全部 rel≤1e-6、无 cap、证书内；decrease-accepts 总数 {sm["total_decrease_accepts_min"]}）')
        w(f'- ranks：{chk["ranks"]["pass_"]}（host i2t/t2i 返回 ranks 复算 recall 差 0；与独立严格计数 rank 一致；并列情形差值见 JSON）')
        sl = chk.get('slsqp_cost', {})
        for g in ('bce', 'snce'):
            if g in sl:
                w(f'- slsqp_cost {g}（B={sl["B"]}, M={sl["M"]}, d={sl["d"]}）：' + '；'.join(
                    f'row{o["row"]} {f(o["seconds"],1)}s nit {o["nit"]} 可行 {o["feasible"]} 与FISTA夹角 {f(o["angle_vs_fista"])}' for o in sl[g]))
    else:
        w('（尚无）')
    w('\n## 1. 审计 1：固定 FIT 图上的目标方向\n')
    w('设置：每个检查点，GMM 划分由 eval_train 重建（seed=cell seed）；每 peer 取固定生成器（seed 20261029）的前 NB 个 unlabeled FIT batch（B≥2），'
      '两个方向（i2t、t2i）。query = 当前 peer train 模式 forward_emb_ctx（每 batch 固定 torch seed，前向后恢复 BN 状态），另一 peer val 模式搜索其队列（K=5），'
      '当前 peer eval 重编码候选（生产 R27 图 A、Y）。early = 共同前缀；前缀队列无 ID，**按 r27_calibrate.rebuild_queue 在内存中重建带 ID 队列（重构，非训练时队列）**；'
      'mid = MISSING（R27 未保存中期检查点）；final = 各 cell a3_vcs 的 state_last（f40_s1001、f60_s1001 另加 a2_mean、a4_bce）。'
      '五个目标：mean（generate mean）、delta（code_v11 r27_delta_control.generate_centered_mean）、vcs（solve_vcs）、bce / snce（生产 solve_spg，tol 1e-6）。'
      '角度 = 2·asin(‖u−v‖/2)（弧度）。无参数更新、无写回、只读 train split。\n')
    for S in A1:
        if S.get('status') != 'OK':
            w(f'- {S["cell"]} / {S["label"]}：**{S.get("status")}**（{S.get("reason", "")}）')
    w('')
    for S in [s for s in A1 if s.get('status') == 'OK']:
        w(f'### {S["cell"]} / {S["label"]}\n')
        w(f'检查点 `{S["checkpoint"]}`（state epoch {S["state_epoch"]}，含队列 ID：{S["has_ids"]}），GPU {S["gpu"]}，job {S["job"]}，图数 {S["n_graphs"]}。')
        w('队列：' + '；'.join(f'{k}: {v.get("queue", "")}，已填 {v.get("queue_filled")}，带 ID {v.get("queue_id_bearing")}' for k, v in S['peers'].items()) + '\n')
        for br, B in S['by_branch'].items():
            w(f'**{br}**：{B["n_graphs"]} 图，{B["n_rows"]} 行，平均原子数 M {f(B["M_mean"],1)}，no_proto {B["no_proto"]}。\n')
            w('| 目标对 | 夹角中位 | p90 | max |'); w('|---|---:|---:|---:|')
            for k, v in B['pair_angles'].items():
                w(f'| {k} | {f(v.get("median"))} | {f(v.get("p90"))} | {f(v.get("max"))} |')
            w('')
            w('| 目标 | ‖raw‖ 中位 | ‖raw‖ min | 行数 | invalid | fallback | cap | interior | 迭代均值/最大 | rel 最大 | fp32 vs fp64 |')
            w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for t in TGT:
                c = B['counts'][t]; it = B['iters'].get(t)
                w(f'| {t} | {f(B["raw_norm"][t].get("median"))} | {f(B["raw_norm"][t].get("min"))} | {c["n_rows"]} | {c["invalid_rows"]} | {c["fallback"]} | {c["cap"]} | {c["interior"]} | '
                  f'{(f(it["mean"],1) + "/" + str(it["max"])) if it else "—"} | {f(B["rel_max"].get(t))} | {f(B["fp32_vs_fp64_maxabs"][t])} |')
            w('')
            w('目标值（行均值，有限项；括号为非有限行数）：')
            w('| 评价目标 | ' + ' | '.join(f'{t} raw | {t} norm' for t in TGT) + ' |')
            w('|---|' + '---:|' * (2 * len(TGT)))
            for o in ('vcs', 'bce', 'snce'):
                cells = []
                for t in TGT:
                    for kind in ('raw', 'norm'):
                        v = B['objectives'][o][f'{kind}_{t}']
                        cells.append(f'{f(v.get("mean"))} ({v["n_nonfinite"]})')
                w(f'| f_{o} | ' + ' | '.join(cells) + ' |')
            w('')
            w('碰撞（不同行目标近同；行对总数 ' + str(B['collisions']['vcs']['n_pairs']) + f'，A 行完全相同的行对 {B["collisions"]["vcs"]["n_pairs_identical_A"]}）：')
            w('| 目标 | <1e-6 rad（A同/A异） | <1e-3 | <1e-2 | <5e-2 | 目标≈单原子 <1e-6 / <1e-3 |'); w('|---|---:|---:|---:|---:|---:|')
            for t in TGT:
                c = B['collisions'][t]
                w(f'| {t} | {c["pairs_lt_1e-06_identical_A"]}/{c["pairs_lt_1e-06_different_A"]} | {c["pairs_lt_0.001_identical_A"]}/{c["pairs_lt_0.001_different_A"]} | '
                  f'{c["pairs_lt_0.01_identical_A"]}/{c["pairs_lt_0.01_different_A"]} | {c["pairs_lt_0.05_identical_A"]}/{c["pairs_lt_0.05_different_A"]} | {c["rows_atom_lt_1e-06"]} / {c["rows_atom_lt_0.001"]} |')
            w('')
            t = B['tight']
            w(f'更紧容差（1e-9）：vcs solve_vcs(tol_rel=1e-9) 与生产夹角 max {f(t["vcs_exact_tol1e-9_angle_max"])}（solve_vcs 不使用 tol_rel）；'
              f'SPG-vcs 1e-9 对精确解夹角 中位/max {f(t["vcs_spg_1e-9_vs_exact"].get("median"))}/{f(t["vcs_spg_1e-9_vs_exact"].get("max"))}；'
              f'bce SPG 1e-9 对生产 1e-6 夹角 中位/p99/max {f(t["bce_spg_1e-9_vs_prod"].get("median"))}/{f(t["bce_spg_1e-9_vs_prod"].get("p99"))}/{f(t["bce_spg_1e-9_vs_prod"].get("max"))}（cap {t["bce_spg_1e-9_cap"]}，目标增量 max {f(t["bce_obj_tight_minus_prod_max"])}）；'
              f'snce 同 {f(t["snce_spg_1e-9_vs_prod"].get("median"))}/{f(t["snce_spg_1e-9_vs_prod"].get("p99"))}/{f(t["snce_spg_1e-9_vs_prod"].get("max"))}（cap {t["snce_spg_1e-9_cap"]}，目标增量 max {f(t["snce_obj_tight_minus_prod_max"])}）。\n')
            w('下游：对 query X 的 SCE 梯度（未乘 λ；生产每方向 0.5·SCE_native + 0.5·λ·SCE_aux）。按图计算整批展平梯度的 cos 与范数比（前者/后者），列跨图中位：')
            w('| 对 | cos 中位 | cos 最小 | 范数比中位 |'); w('|---|---:|---:|---:|')
            for p, v in B['grad'].items():
                w(f'| {p} | {f(v["cos_median"])} | {f(v["cos_min"])} | {f(v["rel_norm_median"])} |')
            w('')
            w('SCE 损失均值：' + '，'.join(f'{k} {f(v)}' for k, v in B['sce_loss_mean'].items()) + '\n')
            # audit 2 per branch
            a2 = B['audit2']
            w(f'**审计 2（{br}）SPG 历史参照**：')
            w('| 生成器 | 求解器 | 总秒 | 均秒/图 | 迭代均值 | 迭代最大 | cap | rel 最大 | 与 FISTA1e-9 夹角 中位/max | f_FISTA−f max | decrease-accepts | rejects |')
            w('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
            for gen in ('bce', 'snce'):
                for key in ('spg_max|1e-06', 'spg_min|1e-06', 'spg_max|1e-09', 'spg_min|1e-09', 'fista|1e-09'):
                    e = a2[f'{gen}|{key}']
                    ang = e.get('angle_vs_fista')
                    w(f'| {gen} | {key} | {f(e["seconds_total"],2)} | {f(e["seconds_mean"],3)} | {f(e["iters_mean"],1)} | {e["iters_max"]} | {e["cap"]} | {f(e["rel_max"])} | '
                      f'{(f(ang.get("median")) + "/" + f(ang.get("max"))) if ang else "—"} | {f(e.get("obj_fista_minus_max"))} | {e.get("decrease_accepts", "—")} | {e.get("rejects", "—")} |')
            w('')
            w('max-history 1e-6 vs min-history 1e-6 每图最大夹角（跨图）：' + '；'.join(f'{g} 中位 {f(a2[f"{g}|angle max_vs_min 1e-6"].get("median"))} max {f(a2[f"{g}|angle max_vs_min 1e-6"].get("max"))}' for g in ('bce', 'snce'))
              + f'。审计副本 ref=max 与生产 solve_spg 逐位相同：bce {a2["bce|spg_max|1e-06"].get("audit_copy_bitwise_equal_production_all")}，snce {a2["snce|spg_max|1e-06"].get("audit_copy_bitwise_equal_production_all")}。')
            w('VCS（对照：精确解 solve_vcs 总秒 ' + f(a2['vcs|exact_solver_seconds_total'], 2) + '）：' + '；'.join(
                f'{k} 迭代均值 {f(a2[f"vcs|{k}"]["iters_mean"],1)} 最大 {a2[f"vcs|{k}"]["iters_max"]} cap {a2[f"vcs|{k}"]["cap"]} 总秒 {f(a2[f"vcs|{k}"]["seconds_total"],2)} 对精确解夹角 max {f(a2[f"vcs|{k}"]["angle_vs_exact_max"])}'
                for k in ('spg_max|1e-06', 'spg_min|1e-06', 'spg_max|1e-09', 'spg_min|1e-09')) + '。')
            for g in ('bce', 'snce'):
                if f'{g}|slsqp' in a2:
                    s = a2[f'{g}|slsqp']
                    w(f'SLSQP（{g}，约化坐标，CPU，前 {s["n"]} 行/每图子集）：可行 {s["feasible"]}/{s["n"]}，success {s["success"]}，均秒 {f(s["seconds_mean"],1)}，nit 最大 {s["nit_max"]}，'
                      f'对 FISTA 夹角 max {f(s["angle_vs_fista_max"])}，对生产 max {f(s["angle_vs_prod_max"])}，对 min-history max {f(s["angle_vs_spg_min_max"])}，f_FISTA−f_SLSQP max {f(s["obj_fista_minus_slsqp_max"])}。')
            w('')
    w('## 2. 审计 3：逐 query DEV 排名\n')
    w('设置：每个完成单元（DONE.json）与 6 个 A0（R25 H0）的 model_best.pth.tar；Flickr full DEV 1,014 图 / 5,070 caption，固定编码顺序；host validate 路径'
      '（每 peer encode_data，图像每 5 行取一，shard_attn_scores shard 100，两 peer 相似度求和 /2）；host evaluation.i2t / t2i return_ranks。'
      f'保存 int32 .npy（i2t_rank、i2t_top1、t2i_rank、t2i_top1），索引说明 `{G3}/INDEX.json`。从保存的 rank 复算六项 Recall，对照 EPOCHS.json 中 model_best epoch 的记录值；'
      '容差 0.1 pp/项（一条 i2t query = 0.0986 pp）。只读 DEV，未读 TEST。\n')
    w('| cell | 臂 | model_best epoch | EPOCHS 最佳 epoch | 记录 rSum | 复算 rSum | rSum 差 | 六项最大绝对差 (pp) | 容差内 | GPU | job |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---|---|---|')
    for r in A3:
        w(f'| {r["cell"]} | {r["arm"]} | {r["epoch"]} | {r["epochs_best_epoch"]} | {f(r["logged_rsum"],3)} | {f(r["recomputed_rsum"],3)} | {f(r["rsum_diff"],4)} | {f(r["max_abs_recall_diff_pp"],4)} | {r["within_tolerance"]} | {r["gpu"]} | {r["job"]} |')
    if A3:
        w(f'\n单元数 {len(A3)}；六项最大绝对差的最大值 {f(max(r["max_abs_recall_diff_pp"] for r in A3),4)} pp；容差内 {sum(r["within_tolerance"] for r in A3)}/{len(A3)}。')
    w('\n## 3. 证据路径\n')
    w(f'- 审计 1/2 每检查点：`{G1}/<cell>/<label>/RESULT.json`（每图汇总）与 `graphs/<peer>_b<k>_<branch>.npz`（逐行数组）。')
    w(f'- 审计 3：`{G3}/<cell>/<arm>/{{RESULT.json,i2t_rank.npy,i2t_top1.npy,t2i_rank.npy,t2i_top1.npy}}`，`{G3}/INDEX.json`。')
    w('- SPG 副本与差异：`audit_spg.py`，`SPG_DIFF.patch`（相对生产 solve_spg 的唯一功能差异：hist.max → hist.min，另加诊断计数）。')
    w('- 门禁：`CHECK_<job>.json`；日志 `logs/`。')
    open(os.path.join(HERE, 'AUDIT_CN.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
