# 03｜V6 证据复用与接入核对

## 1. 两本账分别保存

**V6 ledger** 回答“这版稿件的文字与数字基于什么”；**HEAD reconciliation** 回答“服务器现在比 V6 多了什么、哪些需要修正”。前者原样保留在 `reference_v6/evidence/`；后者由服务器按 intake 模板生成。包内没有对当前 HEAD 或 checkpoint 完整性作任何声明。

来源优先级：实际配置、split 和逐 run 结果支持报告；报告支持稿件。发现报告与原记录冲突时列明冲突，不擅自选择更好数字。旧摘要和记忆不替代实际文件。

## 2. 复用映射 [V6]

以下路径均相对于 `W-Yinghao/VCS_QMI`，以 V6 固定提交为基准。本表不是启动这些实验的命令。

| 逻辑证据 | 原报告 | 稿件位置／复用动作 |
|---|---|---|
| E-DIM／E-N | `reports/P86_ESTIMATOR_BENCHMARK_REPORT_20260929.md` | Fig.1–2；从原聚合数据导出均值及可支持的误差条；保留 corrected SMILE addendum |
| E-COMP | `reports/P118_V5_NEST_REPORT_20261003.md` | Table1；moderate condition 的相同 sampled readout；VCS/JS × separate/nested |
| S-C10/C100 | `reports/P107_AP3_LAYER2_REPORT_20261003.md` | 主 SSL 表和 C10 五种子确认；保留 fresh seeds3–4 的标识 |
| S-JS-C10 | `reports/P129_ADDENDUM1_REPORT_20261005.md` | C10 选定 JS (2,.25)，三种子 |
| S-JS-C100 | `reports/P129_ADDENDUM3_REPORT_20261006.md` | C100 选定 JS (3,.5)，三种子；不只保留默认参数对照 |
| S-SHARED | `reports/P120_ADDENDUM1_REPORT_20261005.md` | 补充 shared-score 五种子；与 selected-score 比较分开 |
| S-AUG | P107 及 `reports/P115_V5_AUG_FACTORS_REPORT_20261004.md` | 强增强反向结果及因素分析；不重复跑以改变方向 |
| I-SOLVER | `reports/P116_results.md`、`reports/P116_V5_SOLVE_REPORT_20261003.md` | Table3；外部 SimCLR fixture；A1 对短预算 A3@50 |
| I-RESPONSE | `reports/P109_V4_X_I_REPORT_20261002.md` 与其 effect files | 配对图像响应；与 P116 各保留 fixture id |
| I-SEED/NEST | P118 report、`reports/P118_nest_results.md` | seeds1–2 检测与已有真实压缩测量；保留 null flags |
| S-SITE | `reports/P141_V7_C100_LAYER_READOUT_REPORT_20261006.md` | 分层可读性补充，不解释为 MI 真值 |
| S-SCALE | `reports/P132_V6_IMAGENET_PREFLIGHT_REPORT_20261005.md` | 规模化入口；快照只有 preflight，不能作为完整训练结果 |

`contracts/evidence_sources.json` 是可机读版本，`tools/intake_repo.py` 可读取这些路径在 baseline/HEAD/working-tree 的存在性与 hash。路径存在只是定位成功，不表示内容或实验已验证。

## 3. 数字锁只用来检查转录 [V6]

以下均为 V6 已报数字，不是本包重跑的结果。

| 项目 | V6 读数 | 关键限定 |
|---|---|---|
| 50维/侧 estimator | VCS .169，RFF .507，KDE .527，truth S=.540 | posterior MSE，同一目标 |
| N=256→16384 | VCS .364→.029，matched JS 最大 N 为 .026 | 固定分布，不推出 SSL 排序 |
| known-zero step | VCS separate −.0204，nested −.0001 | moderate Gaussian，signed bias |
| known-positive step | VCS RMSE .0105→.0070；JS .0109→.0087 | 真值约 .0441；同 sampled readout |
| C10 five-seed linear | VCS 89.02±.04；SimCLR 88.31±.21 | validation；seed0参与开发选择 |
| C100 selected-score linear | VCS60.00±.25；JS60.49±.21；SimCLR58.25±.35 | 三种子；JS selected score；不宣称全局最优 |
| C100 shared-score linear | VCS60.15±.38；JS59.35±.42 | 五种子、同 (2,.5)，另一协议 |
| C10 strong linear | VCS88.25；SimCLR89.49 | 三种子、strong augmentation |
| P116 SimCLR blur.5/n1000 | VCS .75；JS .82 | 100重复的拒绝率，不是分类精度 |
| P116 SimCLR blur.5/n2000 | VCS .99；JS1.00 | 同上 |
| P116 SimCLR colour.2/n2000 | VCS .97；JS .94 | VCS约.05s，JS约.07s；短 solver |
| P109 colour.2 response | accuracy约−.7点，true-class probability约−.007，flip约3.8% | 8000 paired images；不与 P116 混池 |

标准差和置信区间必须从相同 seed 集、相同 readout 和未舍入记录导出。包内原检查脚本使用报告中的舍入数组，只检查统计运算；不能代替 server raw-data verification。

## 4. 不重复做什么

P86 的全网格、P118 的已完成真值链、P107 的已完成 CIFAR 长实验、P116 solver continuation，都先复用。重跑理由必须是可复现性核查、缺失文件、确定实现错误或明确新条件，不是换了章节标题。

新正式训练从初始化按完整日程执行。中断后的同日程恢复保留同一 run id／resume lineage；短 pilot 和 confirmatory 正式 run 分开。历史配置中已有官方 test 访问，不能重新称“整个项目从未接触 test”；当前 45k/5k 的主表也不能重命名为 official-test。

## 5. 必须保留的反向或不充分证据

CIFAR-100 上所选 JS 的 linear 更高；SimCLR 的 kNN 更高；strong augmentation 反转 linear 排序；学习型 VCS 对 JS 不具有全格精度优势；短预算 JS 消除了长预算比较的大部分时间差。这些都属于当前论证边界，不是“噪声格子”。

non-rejection 不等于不存在依赖；probe 改善不是精确 MI 改善；嵌套拟合的残差很小不是 oracle 误差已小。新的核／HSIC／encoder／dataset 对照未完成时保持缺失状态，不用旧内部配置差异补位置。

## 6. 源包忠实性说明

`reference_v6/` 原样保留收到的文件。英文 Markdown 是正文导出，不含完整 supplement，个别交叉引用存在导出痕迹；方法的精确公式、章节关联以 `source/main_body.tex`、`source/supplement.tex` 和完整 PDF 核对。该说明不授权静默修正论文的科学内容。需要的格式修补单独记入 writing change log。
