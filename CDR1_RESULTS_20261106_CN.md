# CDR1：污染感知标定 × 可拒绝检索（2026-11-06 第五十二版）

接在第五十一版（`CR6_NR6A_READOUT_20261106_CN.md`）之后。只发布数字表与代码，不附解读。用户：「把CDR1的push一版」。NR6-B（11/12）、NR6A_P1105 / NR6B_P1105 与收尾三表仍在运行，本版不含。

## CDR1（`handoff/cdr1_20261106/`，`RESULTS_CDR1.md` 与台账）

- 包 `VCS_CDR1_D1D2_Server_20261106`（协议 CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1，证据 a55332645）；用户「查看，准备，并提交这个」。冻结协议 `PUBLIC_PROTOCOL_FROZEN_CDR1.json`（冻结于任何真实 fit 之前）。
- 锁定未经本项目任务训练的 google/siglip2-base-patch16-224 原始 cosine 内容分数（沿 NR1 锁定版本，逐样本 L2，无批中心化，不加载 NR/CR checkpoint）；只拟合两个标量的单调标定 ell = exp(u)·z + b，排序不变（全部 181 个预测文件的 Top1 等于基础 argmax）。
- D1：在受控污染模型下比较 J 与 forward-BCE，读法 NAIVE / POST-PROJECTED / FORWARD；rho 已知、未知（部分区间与 epsilon = 0 / 0.05 / 0.1 锚点、profile 仅作敏感性）、clean oracle 参照。D2：text→image 可拒绝检索，固定阈值 0.90 / 0.95，图库 M = 32 / 128 / 512，存在先验 pi = 0.25 / 0.5 / 0.75，分开报告 p_present 与 p_best_correct；零覆盖时风险记 NA。
- 规模：24 个数据情形（Flickr / COCO × 独立污染 0 / 60 / 80% 与困难错配 60% × 数据 seed 1001 / 2002 / 3003），84 个唯一两参数 fit（均收敛），96 个 profile fit，24 个锚点 fit；跨数据迁移（Flickr↔COCO、→CC152K_RECOVERED_20261104）。CC 无真实 rho、不生成 clean oracle，本地上界只作模型条件下的敏感性。
- 任务划分按明确图像身份分组（字节相同原图合并为同组）；隔离 curator 构造受控污染与匹配图库，fit / predict 不读真值；评价目录在 fit / predict 期间关闭。全部为 CPU 作业，不读 TEST。
- 未完成项：DEDPUL 记 PENDING（未接入，不声称比较）；D3 / OT 未做；D4 为 NA。
- 代码：`site/`、`sbatch/`；S0 记录 `s0/`（文本输出）。设计包本身（`package/`）不发布。`PREDICTION_LEDGER.csv`（约 5.1 MB）超过发布单文件上限，本版不含；其余台账均发布。

## 范围说明

受控污染与合成压力模型下的结果不等于自然数据证据；配对差不给显著性结论。Flickr、COCO、CC152K_RECOVERED 分开报告。
