# 证据、来源与 r3/VL v1 → r4 的变化

## 1. 快照

本轮核查 main：`e19782a47a908a5fddea707d129ae0a830149c60`，提交时间 `2026-10-10T20:57:09Z`。

直接读取：main、VL1-22 完成报告/冻结协议、P151 关键代码段。未查询实时队列，未拉取全部仓库，未重新计算全部原始单元。用户上传的附录 4 完整保存在 sources；其原文中未修正的旧句子不因此进入执行规范。

## 2. 最新 Table C：报告原值摘录

下表直接抄录完成报告，不是从逐 query 记录重算。单位为 DEV image-macro Acc@IoU 0.5（%），mean over 3 seeds。

| 数据集 | 主干 | VCS | JS | Softmax |
|---|---|---:|---:|---:|
| RefCOCOg | CLIP B/16 | 46.27 | 47.36 | 47.07 |
| RefCOCOg | SigLIP 2 B/16 | 56.22 | 57.59 | 56.86 |
| RefCOCOg | CLIP L/14@336 | 50.75 | 52.28 | 51.91 |
| RefCOCOg | SigLIP 2 L/16 | 60.45 | 61.25 | 60.69 |
| RefCOCO | CLIP B/16 | 44.23 | 44.93 | 48.39 |
| RefCOCO | SigLIP 2 B/16 | 56.56 | 55.82 | 57.76 |
| RefCOCO | CLIP L/14@336 | 50.99 | 50.13 | 53.58 |
| RefCOCO | SigLIP 2 L/16 | 60.12 | 59.92 | 61.07 |
| RefCOCO+ | CLIP B/16 | 42.08 | 42.43 | 46.20 |
| RefCOCO+ | SigLIP 2 B/16 | 53.46 | 53.43 | 55.66 |
| RefCOCO+ | CLIP L/14@336 | 46.07 | 46.04 | 49.11 |
| RefCOCO+ | SigLIP 2 L/16 | 56.84 | 57.62 | 60.28 |

报告汇总：VCS-softmax −2.05 [−2.93,−1.16]；VCS-JS −0.39 [−0.89,+0.10]。各条件相关，汇总不是独立总体推断。图内约44–52候选；覆盖率 RefCOCOg/RefCOCO/RefCOCO+ =97.3/98.5/98.8%。四舍五入单格求均值可能与原始汇总有微小差异。[R1]

## 3. 修改来源与状态

| 来源建议 | r4 接受的内容 | r4 明确不继承的推论 |
|---|---|---|
| 上传综合分析附录 4 | RPC 定位、common字段、同目标 P134 重读、公平性 | “JS较好只能是优化慢”“预算增加必反转” |
| 原计划 | M/S/J、直接样本估计、有限样本与实际乘积采样 | 基础泛函首次性不由本包保证 |
| 上一轮最后讨论 | 统计问题做实；候选支持适配；少量验证 | 不自动变成已确定的全新估计器理论 |
| 最新 VL1-22 frozen | 完整 VL 实验、先保留真实候选主表 | 新冻结 F2 小头筛选不再默认执行 |
| 公共 incomplete-U 文献 | 配对计算与数据量有区别 | 不将“少 tuples”本身再次作为新方法 |

## 4. 第一轮需要作者确认的决定

| 决定 | 默认建议 | 不确认时怎样做 |
|---|---|---|
| 新统计路线的研究角色 | 小型假设/数值核对，不预先变主理论 | 只交已有关系与未解决点 |
| 新 VL 训练预算 | 两数据集、一个Base、三目标、三seed | 先交只读诊断和真实成本，不排长作业 |
| 使用哪块未读数据作确认 | 先查访问与预训练 exposure 台账 | 旧 DEV 只作探索，不自称确认 |
| 历史 in-flight 作业 | 沿用 owner 已批准规则 | 不擅自取消/重启 |
| 官方测试与改稿 | 后续明确一次性流程 | 本轮不改稿、不打开测试 |

## 5. 来源索引

### O：原计划

`Variational_CS_QMI_Research_Plan (1).pdf`，September 2026。§2–5：混合参考、S/J与critic；§10–11：稳定性/统计计划；§17：prior-art与实际乘积采样。包内提供用户已上传原文件，只读。

### A4：用户上传分析

`VCS_QMI_Progress_Problems_Literature_NextSteps_20261010(1).md`，附录4优先于其自相矛盾正文。本轮未独立审核它列出的约120篇文献；文件原样复制到 sources，并通过 SHA-256 记录。

### R：仓库

- [R0] main：<https://github.com/W-Yinghao/VCS_QMI/commit/e19782a47a908a5fddea707d129ae0a830149c60>
- [R1] Table C 完成：<https://github.com/W-Yinghao/VCS_QMI/blob/e19782a47a908a5fddea707d129ae0a830149c60/reports/VL1/VL1_22_TABLE_C_REPORT_20261010.md>
- [R2] Table C frozen：<https://github.com/W-Yinghao/VCS_QMI/blob/e19782a47a908a5fddea707d129ae0a830149c60/reports/VL1/VL1_22_TABLE_C_FROZEN_20261010.md>
- [R3] P151 代码：<https://github.com/W-Yinghao/VCS_QMI/blob/e19782a47a908a5fddea707d129ae0a830149c60/scripts/p151_e0_aggregate.py>，本轮重点为111–160行。
- [R4] VL3 完成报告（此前轮次已读，本轮由当前 Table C 引述衔接）：<https://github.com/W-Yinghao/VCS_QMI/blob/e19782a47a908a5fddea707d129ae0a830149c60/reports/VL3/VL3_COMPLETE_12CELL_REPORT_20261010.md>
- [R5] P134 原始目录（服务器需重汇总，本轮未重算）：<https://github.com/W-Yinghao/VCS_QMI/tree/e19782a47a908a5fddea707d129ae0a830149c60/reports/P134>

### L：本轮核查的原始文献入口

- [L1] Tsai et al., *Self-supervised Representation Learning with Relative Predictive Coding*, ICLR 2021。原始目标、最优函数和统计结果：<https://arxiv.org/html/2103.11275v3>。作者代码：<https://github.com/martinmamql/relative_predictive_coding>。
- [L2] Yamada et al., *Relative Density-Ratio Estimation for Robust Distribution Comparison*：<https://arxiv.org/html/1106.4729>。混合相对比值与最小二乘是先行结构。
- [L3] Reid and Williamson, *Composite Binary Losses*, JMLR 2010：<https://jmlr.org/papers/v11/reid10a.html>。用于界定loss/link、Bayes后验和有限函数类的不同角色。
- [L4] Clémençon, Colin and Bellet, *Scaling-up Empirical Risk Minimization: Optimization of Incomplete U-statistics*, JMLR 2016：<https://jmlr.csail.mit.edu/papers/v17/15-012.html>。用于提醒一般稀疏tuple学习已存在，不作为本文未证的新颖性。

上列文献用于具体定位，不复制它们的完整全文进包。统计规范里的核构造、保守常数和任务协议是本轮独立实例化 [D]；它们的数值自检不等同于新颖性证明或真实模型性能验证。
