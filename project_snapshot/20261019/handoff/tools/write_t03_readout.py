#!/usr/bin/env python3
"""Render the human evaluator-side T03 readout from independently audited JSON."""
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff/evaluator_only"
AGG = ROOT / "ssl_pilot_v10_t03/v10/T03/formal/aggregate/evaluator_only/T03_AGGREGATE.json"


def main():
    audit = json.loads((ROOT / "handoff/evidence/V10_AUDIT.json").read_text())
    desc = json.loads((OUT / "T03_DESCRIPTIVE.json").read_text())
    assert audit["status"] == "PASS"
    assert hashlib.sha256(AGG.read_bytes()).hexdigest() == desc["aggregate_sha256"]
    agg = json.loads(AGG.read_text())
    selected = json.loads((ROOT / "ssl_pilot_v10_t03/v10/T03/CAL/selection/T03_CAL_SELECTION.json").read_text())["groups"]
    diagonal_mf = [g for g in selected if g["family"] == "diagonal_mixture" and g["rho"] > 0 and g["method"].startswith("MF")]
    boundary_cal = sum(g["selected"]["lr"] == 0.001 and g["selected"]["step"] == 2000 for g in diagonal_mf)
    boundary_main = total_diagonal_mf = 0
    for path in (ROOT / "ssl_pilot_v10_t03/v10/T03/formal/runs").glob("T03_MAIN_*/result.json"):
        r = json.loads(path.read_text())
        c = r["config"]
        if c["family"] == "diagonal_mixture" and c["rho"] > 0 and c["method"].startswith("MF"):
            total_diagonal_mf += 1
            boundary_main += r["selection"]["own_cal"]["step"] == 2000
    methods = ["MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1", "MLP_VCS_perm1", "MLP_VCS_perm8"]
    labels = ["MF-p1", "M0", "M1", "MLP-p1", "MLP-p8"]

    def case(c):
        return c.replace("diagonal_mixture", "diagonal").replace("two_block", "two-block").replace("__", " / ").replace("rho", "ρ=").replace("p", ".")

    lines = ["# V10 T03 结果读数与接手结论", "", "日期：2026-10-17。**Evaluator-only：本文件含真值与 EVAL 派生结果，不能作为训练、CAL 选择或已冻结 BF1 网格的输入。**", "",
             "T03 已完成并通过本次文件与记录核验。以下仅描述冻结网格的读数，区分结构、计算与优化；必需基线尚未补齐，不形成方法优劣或新颖性结论。BF1 的既定下一步保持不变。", "",
             "## 1. 结果来源和核验", "",
             "- 工作树 `ssl_pilot_v10_t03`，结果提交 `1c13749`；T03 code `54606921cdd2bf15`，formal pin `64478d332c0a115a`。",
             "- smoke 7；CAL 252（60/60 组完成选择）；formal 960 = main600 + extra-algo120 + capacity240。formal job1043864，CAL job1043862。",
             f"- [独立核验](../evidence/V10_AUDIT.json) 共 {audit['n_checks']:,} 个程序条件通过；导出文件及2,466个外部artifact条目哈希匹配。这个计数不是统计检验次数。",
             f"- 全部单元已记录的最大恒等式残差为 `{desc['max_abs_recorded_identity_residual']:.3g}`。本次复核记录内的代数一致性，没有重新执行 checkpoint 前向求值。",
             "- [原 aggregate](../../ssl_pilot_v10_t03/v10/T03/formal/aggregate/evaluator_only/T03_AGGREGATE.json) 的汇总器 SHA 与代码一致；本次独立复算了均值、方差、SD、RMSE、配对差值和成员归属，未重新计算全部 CI 端点。", "",
             "原项目在本次审计期间新增 [中文报告](../../ssl_pilot_v10_t03/v10/T03/V10_T03_REPORT_20261017.md)，提交0518984；其报告生成器也做完整性和复算检查。它是外部新增工作，本文件是独立接手读数，两者的训练代码和结果来源相同。", "",
             "## 2. 实验对象", "",
             "two_block / diagonal_mixture × m=8/32 × ρ=0/.25/.75，共12条件；FIT4096、CAL2048、EVAL8192。one-hot 输入。MF K16 主网格，MLP 两层64隐藏单元、GELU、tanh输出。每单元2,000 full-batch steps、每250步checkpoint。", "",
             "每条件在 tuning data/algo seed1017 上完成21个候选：3个MF-p1、3个M0、9个M1(lr×cap)、3个MLP-p1、3个MLP-p8；按独立CAL的完整U-J选lr/cap，formal每个重复再用自己的CAL选checkpoint。真值与EVAL不参与选择。", "",
             "主表：10个独立data roots2026101700–2026101709，algo seed固定1001。算法敏感性：第一个data root上algo1001/1002/1003，R=3；不能与10个数据重复混池。容量扩展：四个m32、非零ρ条件，三种MF、K8/16/32；K16复用主表，新增240单元。", "",
             "T03主目标为U：负对从均匀derangement得到；V只对同一个冻结critic作诊断。MF-p1/M0/M1共享MF初始化，MLP-p1/p8共享MLP初始化；对应数据与第一条permutation流在记录中一致。不同critic结构不声称相同参数量或表示能力。", "",
             "## 3. 主表：posterior总体平方误差", "",
             "每格为10个数据重复的 `E_M(T−η)²` 均值，越低越好；由恒等式等于 S−J_pop。MF-p1=MF_VCS_perm1，MLP-p8=MLP_VCS_perm8。", "",
             "| 条件 | S | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|---:|---:|"]
    cases = sorted(set(c["case_id"] for c in agg["main"]))
    lookup = {(c["case_id"], c["method"]): c for c in agg["main"]}
    for c in cases:
        cells = [lookup[c, m] for m in methods]
        lines.append("| " + case(c) + f" | {cells[0]['S']:.6g} | " + " | ".join(f"{x['pop_mse']['mean']:.6g}" for x in cells) + " |")
    lines += ["", "本表two-block的非零条件下MF误差读数较低，diagonal的四个非零条件下MLP误差读数较低。例如diagonal/m32/ρ=.75，M0约.2084，MLP-p1约.01833。该对照同时包含结构、参数化和优化的差异，不能把差距全部归因为纯表示容量。", "",
              f"边界限制：diagonal非零条件的MF调参组中，{boundary_cal}/{len(diagonal_mf)} 同时选中lr网格上端1e-3和最后2000步checkpoint；对应formal主单元有{boundary_main}/{total_diagonal_mf}选最后checkpoint。因此现有误差不是MF函数类已达到的最小误差。", "",
              "## 4. 主表：S估计的EVAL RMSE", "", "每格为 `sqrt(mean((EVAL_UJ−S)²))`。这与上表posterior MSE不是同一个量：EVAL还包含抽样误差。完整bias、SD见 [T03_main.csv](T03_main.csv)。", "",
              "| 条件 | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|---:|"]
    for c in cases:
        lines.append("| " + case(c) + " | " + " | ".join(f"{lookup[c,m]['EVAL_UJ_minus_S']['rmse']:.6g}" for m in methods) + " |")
    lines += ["", "### 配对方向的描述性计数", "", "以下按同一data root配对后求每条件均值，只数方向；没有把12条件当独立样本作新的显著性检验。", "",
              "| A vs B | A的posterior MSE更低 | A的EVAL平方误差更低 |", "|---|---:|---:|"]
    for x in desc["paired_direction_counts_descriptive_only"]:
        lines.append(f"| {x['a']} vs {x['b']} | {x['a_lower_pop_mse_conditions']}/{x['conditions']} | {x['a_lower_eval_squared_error_conditions']}/{x['conditions']} |")
    lines += ["", "M0相对MF-p1为6/12，M1相对M0的posterior误差仅1/12更低。消除固定critic的负对抽样噪声，是精确矩的性质；它不蕴含经过完整训练和CAL选择后总体误差一定更小。M1的固定feature头部优化证书也不等于posterior拟合或泛化保证。", "",
              "## 5. K敏感性", "", "每格仍为posterior MSE均值。沿用已选训练规则，仅改变K；这张表同时包含优化影响，不是测得函数类的不可约逼近误差。", "",
              "| 条件 | 方法 | K8 | K16 | K32 |", "|---|---|---:|---:|---:|"]
    for c in sorted(set(x["case_id"] for x in agg["capacity"])):
        for m in methods[:3]:
            cs = sorted((x for x in agg["capacity"] if x["case_id"] == c and x["method"] == m), key=lambda x:x["K"])
            lines.append("| " + case(c) + " | " + m + " | " + " | ".join(f"{x['pop_mse']['mean']:.6g}" for x in cs) + " |")
    lines += ["", "增大K常缓解diagonal结构的误差，但不保证单调，例如ρ=.75的M0在K32略差于K16。即使K达到m，也不能从2000步的实际训练推断已达到该函数类的最优值。FIT-count saturated表是另一个经验拟合对照，不能冒充真实posterior oracle或泛化下界。", "",
              "## 6. M1求解状态和成本", ""]
    m = desc["m1_main"]
    lines += [f"主表120个M1单元，所选cap分布为10:{m['caps']['10']}、30:{m['caps']['30']}、100:{m['caps']['100']}。每单元训练步内的已记录converged fraction范围 {m['converged_step_fraction_min']:.4f}–{m['converged_step_fraction_max']:.4f}，中位数 {m['converged_step_fraction_median']:.5f}。这对应inner-pre-outer的当前FIT特征，不是最终保存pair的全FIT最优性，更不是自然图像M1的状态。", "",
              "单元平均wall seconds包含该单元训练、CAL/checkpoint及evaluator诊断等开销；运行布局为CPU单线程进程并发，不应用来预测L40S自然配准加速比。", "", "| 方法 | 主表单元平均秒 |", "|---|---:|"]
    for method in methods:
        lines.append(f"| {method} | {desc['main_cost_seconds_mean'][method]:.3f} |")
    lines += ["", "正式batch记录约1127秒；相加所有并发单元wall time不等于batch elapsed或allocated CPU hours。原包phase1估计基于另一线程布局/单元数/solver规则，应保留为历史预测。", "",
              "## 7. 可以写和不能写的结论", "",
              "可以写：在冻结的两类有限支持设计和预算下，精确矩实现、CAL隔离与数据配对都有完整可追溯记录；two-block与diagonal的误差读数表现不同，后者的MF选择还落在预算边界；M0和M1读数没有统一改善。所有比较保持为本次格点描述。", "",
              "不能写：新的散度、普遍低方差/更准、M1全面优于M0、已证明自然配准更快或更准、已独立确认V10。exact-moment RuLSIF等后续对照尚缺。", "",
              "统计限制：主表R=10；算法表R=3且仅固定一个数据集。原t/chi-square区间有分布假设，尤其方差区间要求正态重复，bootstrap也不自动保证小样本覆盖；没有据此宣布48个配对比较均有已控制多重性的显著性。", "",
              "后续仍按已批准BF1协议执行，不根据这次evaluator读数改变既定lr/cap/K或自然任务目标。[主表CSV](T03_main.csv)、[算法CSV](T03_algo.csv)、[容量CSV](T03_capacity.csv)、[描述性摘要](T03_DESCRIPTIVE.json)保留供复核。", ""]
    (OUT / "V10_T03_READOUT_CN.md").write_text("\n".join(lines))
    print(json.dumps({"report": str((OUT / "V10_T03_READOUT_CN.md").relative_to(ROOT)), "main_conditions": len(cases)}))


if __name__ == "__main__":
    main()
