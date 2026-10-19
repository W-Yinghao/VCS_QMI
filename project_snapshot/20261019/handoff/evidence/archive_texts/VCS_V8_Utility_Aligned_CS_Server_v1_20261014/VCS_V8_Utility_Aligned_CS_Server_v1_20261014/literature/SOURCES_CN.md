# 来源目录与使用范围
核对日期：2026-10-14。原始文献/作者页面与仓库事实分开。本包中新公式若注明“直接推导”，属于本轮分析，不冒充来源已证明的新定理。

## [F01] 合作者原始 CS-QMI 研究计划
来源：user-provided Research Plan, September 2026。
入口：conversation:turn599file3
核对范围：全文已在对话提供；本包使用§3–7、§10、§13。
解释边界：一般P/Q、S/J与gap；不复制论文全文。

## [F02] 历史审查附录4
来源：用户提供的2026-10-10勘误。
入口：conversation:turn599file0#L326-L359
核对范围：本次复核已提供内容。
解释边界：按最新勘误理解；不继承已撤回的不可能性和场馆断言。

## [G00] 当前主分支
来源：W-Yinghao/VCS_QMI。
入口：https://github.com/W-Yinghao/VCS_QMI/tree/c7296d2ebb15658fa93737803ff757c98f64ad4a
核对范围：GitHub分支API，2026-10-14。
解释边界：只读快照，不是实时队列。

## [G01] V7分支
来源：W-Yinghao/VCS_QMI。
入口：https://github.com/W-Yinghao/VCS_QMI/tree/891fb8a9948854b2ff4dc93e9212b0cf791b942e
核对范围：GitHub分支API，2026-10-14。
解释边界：分支与主分支分开锁定。

## [G02] V7冻结协议
来源：reports/V7/V7_03_PROTOCOL_FROZEN.yaml。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_03_PROTOCOL_FROZEN.yaml
核对范围：全文。
解释边界：12正式训练、E主读数、校准规则。

## [G03] V7外部主家族修订
来源：reports/V7/V7_03_AMENDMENT_V2_20261014.yaml。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_03_AMENDMENT_V2_20261014.yaml
核对范围：全文。
解释边界：8外部对比；BCE为另一个归因家族；封存标签权限。

## [G04] 实际数据审查
来源：reports/V7/V7_01_DATA_AUDIT.md。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_01_DATA_AUDIT.md
核对范围：全文。
解释边界：32717图、2689组件；3/6原始评分，不等于后验真值。

## [G05] 共同特征估计协议
来源：reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml
核对范围：全文。
解释边界：原生768维、PCA64/256、N0表征敏感性、核调参。

## [G06] 外部基线接入与状态
来源：reports/V7/V7X_30_EXTERNAL_BASELINES.md。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7X_30_EXTERNAL_BASELINES.md
核对范围：全文。
解释边界：本地TIFA/DSG/PN-VQA与API禁用；不把状态表当结果表。

## [G07] 全模型概率结果
来源：reports/V7/V7_TABLE_A_results/v7_table_a.json。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_TABLE_A_results/v7_table_a.json
核对范围：本次读取开头90行；完整比较继承上一轮核对。
解释边界：不宣称本次重算全部训练结果。

## [G08] 新项目体检
来源：reports/PROJECT_MAP_20261014.md。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/c7296d2ebb15658fa93737803ff757c98f64ad4a/reports/PROJECT_MAP_20261014.md
核对范围：本次重点核对理论定位与实验索引；长输出部分截断。
解释边界：事实与作者解释分开；不直接接受“逐点后验不可能排序”。

## [P01] EvalMuse-40K / FGA-BLIP2 / PN-VQA
来源：Han et al., arXiv:2412.18150v2;正式版AAAI2026。
入口：https://arxiv.org/html/2412.18150v2
核对范围：方法、数据及标注节。
解释边界：整体L1+元素监督已有；3–6次标注与重标注选择性。

## [P02] GenAI-Bench / GenAI-Rank
来源：Li et al., arXiv:2406.13743。
入口：https://arxiv.org/html/2406.13743
核对范围：数据和排名协议。
解释边界：固定候选的人类评价；不是新任务。

## [P03] VQAScore
来源：Lin et al., ECCV2024, arXiv:2404.01291。
入口：https://arxiv.org/abs/2404.01291
核对范围：论文摘要和项目在仓库中的协议。
解释边界：逐图概率分数可以完成best-of-N，不接受点式必败。

## [P04] NIMA: Neural Image Assessment
来源：Talebi and Milanfar, IEEE TIP 2018。
入口：https://doi.org/10.1109/TIP.2018.2831899
核对范围：公开方法段、EMD公式。
解释边界：评分分布、CDF损失和平均分预测已有；原任务不是图文对齐。

## [P05] Strictly Proper Scoring Rules, Prediction, and Estimation
来源：Gneiting and Raftery, JASA 2007。
入口：https://doi.org/10.1198/016214506000001437
核对范围：原始文章/作者技术报告摘要；标准RPS部分按定义推导。
解释边界：正规评分、RPS、Bayes风险；不声称新的概率评分原则。

## [P06] CORAL: Rank-consistent ordinal regression
来源：Cao, Mirjalili, Raschka, PRL 2020。
入口：https://arxiv.org/abs/1901.07884
核对范围：原始摘要、作者引用页。
解释边界：阈值分解、等级一致性已有；注意不同阈值先验。

## [P07] CORN: Rank-consistent ordinal regression based on conditional probabilities
来源：Shi, Cao, Raschka, arXiv:2111.08851。
入口：https://arxiv.org/abs/2111.08851
核对范围：原始摘要和作者实现入口。
解释边界：条件风险集与链式概率已有；不混为无条件阈值BCE。

## [P08] RankNet: Learning to rank using gradient descent
来源：Burges et al., ICML2005。
入口：https://doi.org/10.1145/1102351.1102363
核对范围：出版摘要。
解释边界：相同提示的配对排序强对照；不是概率校准基线。

## [P09] ListNet: From pairwise to listwise learning to rank
来源：Cao et al., ICML2007。
入口：https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/
核对范围：原作者研究页摘要。
解释边界：列表top-one分布已有；本包只是图文任务适配。

## [P10] Relative Predictive Coding
来源：Tsai et al., arXiv:2103.11275v3。
入口：https://arxiv.org/html/2103.11275v3
核对范围：基础目标与后验关系。
解释边界：平衡RPC与J同一；应用换成阈值不消除重合。

## [P11] RuLSIF: Relative Density-Ratio Estimation
来源：Yamada et al., NeurIPS2011。
入口：https://proceedings.neurips.cc/paper_files/paper/2011/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html
核对范围：原始论文摘要与项目已有同目标实现。
解释边界：强传统密度比路线，不只用弱KDE。

## [P12] Rich Human Feedback for Text-to-Image Generation
来源：Liang et al., CVPR2024。
入口：https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.html
核对范围：官方摘要与数据任务。
解释边界：外部对齐/丰富反馈；不强行将所有字段映射为1–5评级。

## [P13] HPSv3: Towards Wide-Spectrum Human Preference Score
来源：Ma et al., ICCV2025。
入口：https://arxiv.org/abs/2508.03789
核对范围：原始摘要。
解释边界：偏好与不确定性排序已有；审美/整体偏好不等于文字对齐。

## [P14] TIFA
来源：Hu et al., ICCV2023。
入口：https://arxiv.org/abs/2303.11897
核对范围：沿用V7已审官方接入；本次不重读全文。
解释边界：本地官方路由与API路由区分。

## [P15] Davidsonian Scene Graph (DSG)
来源：Cho et al., ICLR2024。
入口：https://proceedings.iclr.cc/paper_files/paper/2024/hash/446905abfec9db8856f7f7465bbd5552-Abstract-Conference.html
核对范围：沿用V7已审官方接入；本次不重读全文。
解释边界：依赖解析不是新方法；本地LLM替换单列。

## [D01] EvalMuse public data
来源：DY-Evalab/EvalMuse。
入口：https://huggingface.co/datasets/DY-Evalab/EvalMuse
核对范围：本次使用仓库数据审查，不重新下载标签。
解释边界：官方test的可评分性/许可按封存协议核验。

## [D02] GenAI-Rank public data
来源：zhiqiulin/GenAI-Image-Ranking-800。
入口：https://huggingface.co/datasets/zhiqiulin/GenAI-Image-Ranking-800
核对范围：仓库已固定图像/标签分离。
解释边界：800提示、两生成器各9图；本包未读人工标签。


## 交付增量来源
以下增量晚于上列设计快照。

## [G08] TIFA本地问题无标签统计
入口：https://github.com/W-Yinghao/VCS_QMI/commit/4f90d6dd68c3c8d0977b6eab30731abc03780c4f
核对范围：完整commit diff；31解析失败、33空问题提示。不是人工评价结果。

## [G09] REV9评价打包变更
入口：https://github.com/W-Yinghao/VCS_QMI/commit/2d38ea35e8e6e178f580e5bbb052fc37e05230c7
核对范围：完整commit diff；报告声明配置/shard/evaluator不变。不是实时队列。
