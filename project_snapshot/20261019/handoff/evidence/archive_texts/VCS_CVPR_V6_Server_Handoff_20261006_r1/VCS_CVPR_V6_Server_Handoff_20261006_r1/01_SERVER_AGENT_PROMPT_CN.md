# 给服务器端 agent 的接入指令

请按随附 `CVPR-MV6-HANDOFF-20261006-r1` 材料包接入 VCS_QMI 项目。本轮以 **2026-10-06 的 CVPR 稿件 V6** 为论文基线，不是要求回退到旧 server-v6，也不是清空已有 server-v7 工作。

## 论文主线

这是 estimator 论文：保留合作者的 equal-mixture reference、原二次目标、bounded critic 和 exact regression gap。完整视觉 SSL 是主应用；冻结外部模型的条件属性审计是第二个直接测量应用。已知真值的压缩实验属于 estimator 验证，不另写一篇 saliency 论文。

正文统一一个名称 VCS。当前主 SSL 实现对应内部 A-P3：固定 `T=tanh[a(s-kappa)]`，当前 `(a,kappa)=(2,0.5)`，全视图配对，正负项分别归一化，两端完整梯度；无 queue、无 momentum。旧 recipe、G2 和自由评分器是内部消融，不作为多个“ours”占据外部比较表。

## 先阅读

1. 包内 `00_README_CN.md`；`reference_v6/Paper_Positioning_and_Interpretability_CN.md`；完整 V6 PDF 与 `source/supplement.tex`。
2. `docs/02_PAPER_AND_METHOD_CONTRACT_CN.md`、`03_EVIDENCE_REUSE_CN.md`、`04_EXECUTION_PLAN_CN.md`。
3. 做测量时读 `05_AUDIT_AND_COMPRESSION_SPEC_CN.md`；交付时按 `06_DELIVERY_AND_WRITING_CN.md`。

## 第一阶段：接入与复用，不盲目重跑

记录当前仓库 HEAD、branch、dirty 文件、现有实验编号、作业与 checkpoint 状态。与 V6 的固定快照 `bc28bf89c94d5331de811a83aa8bcbfb77d7d155` 对照。先读实际文件再判断完成状态；报告标题、作业退出码和历史汇总表都不能单独代替完整结果检查。

复用 P86 estimator 基准、P118 已知压缩、P107/P129 SSL 与选择结果、P116 条件测量、P109/P118 配对响应、P141 分层读出。已完成的格子不因章节变化重训。已有获批任务继续按原协议运行，不取消、不重启、不改其未完成日程。

将当前证据相对 V6 的新增、冲突、修正逐项列出。V6 本身保留原样；需要改稿的数字进入 change log，不能静默替换。尤其保留 CIFAR-100 选定 JS 领先 VCS、kNN 反向排序、强增强下 SimCLR 领先，以及短预算 JS 接近 VCS 的事实。

## 第二阶段：围绕论文缺口补实验

按 `docs/04` 衔接三类工作：

- 规模化 SSL：先查是否已有后续 ImageNet 授权和任务。V6 快照的 P132 只是预检查；解决并复核输入管线／多 GPU 配对，再提交明确的完整训练协议和预算。不得拿 preflight 或 short pilot 代替训练结果。
- 完整条件审计：固定外部 SimCLR，使用相同 split、N|Y、null、permutations 和预算比较 VCS、匹配 JS、同目标 kernel、条件 HSIC。原 P116 与 P109 是不同 fixture，分别保存；新联合审计单独建 fixture，不拼接其历史数字。再按库存和协议扩到另一个外部 encoder、另一个数据集。
- 真实压缩测量：仅使用可验证的 h→r→z、h→固定任务输出等映射。复用 nested_exact，对同一底层条件实验报告原始 J、带符号 Delta J、D_T 和拟合敏感性。不要假定 pooled layer3→h 是确定性压缩；不要把负增量截零或把小残差当成准确性证书。

不得自行恢复随机掩膜空间归因、RISE/RELAX、VOC 大基准、额外响应噪声、生成式反事实、依赖分数选模或“门控天然耐噪”主线。结果为零或 baseline 更好也应交付，不以结果必须有利为完成条件。

## 执行与交付要求

先跑包内只读检查，复用既有测试；新实现先小型代码验证。新 GPU 科学作业使用项目现有授权流程，材料包不赋予无限预算。正式训练从初始化按照冻结的完整日程执行；允许同一日程的故障续跑，不把短 pilot 续训冒充独立正式实验。

每个结果保留完整配置、commit、dataset/split/fixture/checkpoint hashes、独立单位、encoder seed 与重复采样 seed、选择角色、成本和状态。不同目标不比较原始 MSE；真实图像无 oracle 就不填 posterior MSE。对非拒绝写“给定样本和测量器预算下未检出”，不写“信息不存在”。

第一轮返回：`INTAKE_REPORT`、V6→HEAD reconciliation、复用／在跑／缺失清单、实际入口映射、待冻结的最小任务表与预算。之后每轮提交结构化结果、可复核图表和论文对应位置；禁止只写 PASS/FAIL、不交效应大小和局限。详见 templates。

现在先完成接入盘点与协议整理；对于已有明确授权的持续任务保留原安排。尚无新训练授权时，不因收到本包自动启动长训练。
