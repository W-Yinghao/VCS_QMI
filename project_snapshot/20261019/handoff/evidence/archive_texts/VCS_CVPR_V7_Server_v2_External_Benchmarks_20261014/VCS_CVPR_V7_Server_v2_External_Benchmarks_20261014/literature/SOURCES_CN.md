# 来源目录与核验深度（v2）

v1来源保留其原核验日期；本轮新核对项单独标明。来源条目不是独立论文计数，X10/X13为范围/复核记录。所有公式代码均为本包自行实现的参考，不附第三方全文、权重或字体。

## [F01] 合作者原计划

出处：用户提供：Variational Cauchy–Schwarz Quadratic Mutual Information — Research Plan, September 2026

入口：conversation:turn581file3

核验：全文；重点 §2–8、§10、§17；2026-10-13 (inherited from v1; not all reopened in v2)

注意：原始理论；不把原草稿全部创新表述当成已完成新颖性审计

## [F02] 历史审查的修订附录

出处：用户提供：VCS_QMI 项目综合文档，附录4

入口：conversation:turn581file0#L326-L359

核验：已给出的修订附录；2026-10-13 (inherited from v1; not all reopened in v2)

注意：只继承已明确的归属与勘误；不把10月10日旧状态当成当前状态

## [G01] 本项目快照

出处：W-Yinghao/VCS_QMI @ 3e2fe59c4861abae46537574c4c59f9bedfe91c6

入口：https://github.com/W-Yinghao/VCS_QMI/tree/3e2fe59c4861abae46537574c4c59f9bedfe91c6

核验：GitHub main API；2026-10-13 (inherited from v1; not all reopened in v2)

注意：截至2026-10-13 18:51:49 UTC；不是实时Slurm查询

## [P01] EvalMuse / FGA-BLIP2 / PN-VQA

出处：EvalMuse-40K, AAAI 2026；arXiv版本标题略不同

入口：https://ojs.aaai.org/index.php/AAAI/article/view/42458

核验：正式出版摘要；arXiv HTML方法和数据节；2026-10-13 (inherited from v1; not all reopened in v2)

注意：arXiv: https://arxiv.org/html/2412.18150v2

## [D01] EvalMuse公开数据

出处：DY-Evalab/EvalMuse

入口：https://huggingface.co/datasets/DY-Evalab/EvalMuse

核验：官方卡片、字段预览；2026-10-13 (inherited from v1; not all reopened in v2)

注意：卡片有test标注稍后公开说明；以实际访问为准，不能假定可下载测试标签

## [G02] FGA核心代码

出处：lavis/models/blip2_models/fga_blip2.py

入口：https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/models/blip2_models/fga_blip2.py

核验：完整源码；2026-10-13 (inherited from v1; not all reopened in v2)

注意：活跃itm分支是L1；variance分支注释且在return后

## [G03] FGA训练配置

出处：lavis/projects/blip2/train/fga_blip2.yaml

入口：https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/projects/blip2/train/fga_blip2.yaml

核验：完整源码；2026-10-13 (inherited from v1; not all reopened in v2)

注意：freeze_vit=True, 5 epochs, batch14, image364；有效运行配置仍须展开

## [G04] FGA数据读取

出处：align_datasets.py

入口：https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/datasets/datasets/align_datasets.py

核验：完整源码；2026-10-13 (inherited from v1; not all reopened in v2)

注意：32 token截断；第0位token_score是总体分数，不是语义元素标签

## [G05] FGA元素映射

出处：process/element2mask.py

入口：https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/process/element2mask.py

核验：完整源码；2026-10-13 (inherited from v1; not all reopened in v2)

注意：多个元素的重叠token标签被平均；失败样本会被过滤

## [G06] FGA标签预处理

出处：process/process_train.py

入口：https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/process/process_train.py

核验：完整源码；2026-10-13 (inherited from v1; not all reopened in v2)

注意：原始投票转均值；按prompt算总体评分方差；保留原始投票

## [P02] GenAI-Bench / GenAI-Rank

出处：GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation, 2024

入口：https://arxiv.org/html/2406.13743

核验：全文相关数据、排名协议节；2026-10-13 (inherited from v1; not all reopened in v2)

注意：2024 SynData workshop记录；不写成CVPR主会论文

## [D02] GenAI数据入口

出处：BaiqiL/GenAI-Bench

入口：https://huggingface.co/datasets/BaiqiL/GenAI-Bench

核验：官方卡片；2026-10-13 (inherited from v1; not all reopened in v2)

注意：Bench与Rank提示重叠，必须作为共同终末评测资源

## [G07] GenAI-Rank官方脚本

出处：genai_image_ranking.py

入口：https://github.com/linzhiqiu/t2v_metrics/blob/main/genai_image_ranking.py

核验：源码1–245行；blob 0e0fdb3e9550b1b1217c9feb78c46ed1095a7252；2026-10-13 (inherited from v1; not all reopened in v2)

注意：DALLE_3 / SDXL_Base；N=3按固定三个连续分组；服务器锁定commit

## [G08] VQAScore工具版本

出处：t2v_metrics README

入口：https://github.com/linzhiqiu/t2v_metrics

核验：官方README；2026-10-13 (inherited from v1; not all reopened in v2)

注意：原论文复现用legacy v3.0；现代宿主用另一个已锁定环境

## [P03] VQAScore

出处：Evaluating Text-to-Visual Generation with Image-to-Text Generation, ECCV 2024

入口：https://arxiv.org/abs/2404.01291

核验：原论文与官方项目说明；2026-10-13 (inherited from v1; not all reopened in v2)

注意：必须比较原生公开强模型，不只比较CLIPScore

## [P04] TIFA

出处：TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering, ICCV 2023

入口：https://arxiv.org/abs/2303.11897

核验：原论文摘要与EvalMuse关联协议；2026-10-13 (inherited from v1; not all reopened in v2)

注意：TIFA160是人工对齐子集，不等于4K prompt的TIFA v1.0

## [P05] DSG

出处：Davidsonian Scene Graph, ICLR 2024

入口：https://proceedings.iclr.cc/paper_files/paper/2024/hash/446905abfec9db8856f7f7465bbd5552-Abstract-Conference.html

核验：正式摘要；2026-10-13 (inherited from v1; not all reopened in v2)

注意：对象与属性依赖、唯一原子问题不是本文新发明

## [P06] RichHF

出处：Rich Human Feedback for Text-to-Image Generation, CVPR 2024

入口：https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.html

核验：正式摘要；2026-10-13 (inherited from v1; not all reopened in v2)

注意：有多种质量维度；只把对齐维度纳入本轮主要相关性

## [P07] HPSv3

出处：HPSv3: Towards Wide-Spectrum Human Preference Score, ICCV 2025

入口：https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html

核验：正式摘要；2026-10-13 (inherited from v1; not all reopened in v2)

注意：综合偏好，不是同定义的语义满足概率

## [P08] VisionReward

出处：VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning, AAAI 2026

入口：https://ojs.aaai.org/index.php/AAAI/article/view/38107

核验：正式摘要；2026-10-13 (inherited from v1; not all reopened in v2)

注意：多维偏好参考；记录实际输出维度与聚合方法

## [P09] DynEval

出处：DynEval: Holistic Evaluations of T2I Generative Models in the Wild, 2026 preprint

入口：https://arxiv.org/abs/2607.11199

核验：论文HTML与方法概述；2026-10-13 (inherited from v1; not all reopened in v2)

注意：模型/权重入口本轮未完成可执行复现核验；不得虚构官方数值

## [P10] DEPICT

出处：DEPICT: Scoring Text-to-Image Alignment by Answer Agreement, 2026 preprint

入口：https://arxiv.org/abs/2610.03617

核验：论文HTML；2026-10-13 (inherited from v1; not all reopened in v2)

注意：本轮未确认可用官方代码；论文复现必须独立命名

## [P11] NegBench

出处：Vision-Language Models Do Not Understand Negation

入口：https://arxiv.org/abs/2501.09425

核验：原始摘要与官方项目页；2026-10-13 (inherited from v1; not all reopened in v2)

注意：https://negbench.github.io/；选取图像子任务，不能跨18种任务混成总准确率

## [P12] Winoground

出处：Winoground, CVPR 2022

入口：https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.html

核验：正式摘要；2026-10-13 (inherited from v1; not all reopened in v2)

注意：官方text/image/group三类判据，严格不等式

## [T01] RPC

出处：Self-supervised Representation Learning with Relative Predictive Coding, ICLR 2021

入口：https://arxiv.org/abs/2103.11275

核验：原文关系＋项目已核对附录；2026-10-13 (inherited from v1; not all reopened in v2)

注意：平衡RPC与J完全重合，不另造重复算法行

## [T02] 正规复合损失

出处：Composite Binary Losses, JMLR 2010

入口：https://jmlr.org/papers/v11/reid10a.html

核验：原始论文页面；2026-10-13 (inherited from v1; not all reopened in v2)

注意：用于区分总体proper性质、受限函数类和有限优化

## [P13] 人类评价协议

出处：Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation, CVPR 2023

入口：https://openaccess.thecvf.com/content/CVPR2023/html/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.html

核验：正式摘要；2026-10-13 (inherited from v1; not all reopened in v2)

注意：仅在新生成人工确认时使用；首轮优先已有人工标签

## [X01] Cauchy-Schwarz Divergence Information Bottleneck for Regression

出处：Yu et al., ICLR 2024

入口：https://proceedings.iclr.cc/paper_files/paper/2024/hash/aecc44c30b1ea774c5d1d37da1a73b53-Abstract-Conference.html

核验：正式摘要、arXiv方法文本、作者README；2026-10-14

注意：https://arxiv.org/html/2404.17951v1 ; https://github.com/SJYuCNEL/Cauchy-Schwarz-Information-Bottleneck ; 不是EvalMuse原生方法

## [X02] The Conditional Cauchy-Schwarz Divergence With Applications to Time-Series Data and Sequential Decision Making

出处：Yu et al., TPAMI 47(7), 5901–5917, 2025

入口：https://ieeexplore.ieee.org/document/10930811/

核验：IEEE元数据/摘要、arXiv摘要、作者项目入口；2026-10-14

注意：DOI 10.1109/TPAMI.2025.3552434 ; https://arxiv.org/abs/2301.08970 ; https://github.com/SJYuCNEL/conditional_CS_divergence ; 本包未运行原生RL/聚类

## [X03] Relative Density-Ratio Estimation for Robust Distribution Comparison

出处：Yamada et al., NeurIPS 2011

入口：https://proceedings.neurips.cc/paper_files/paper/2011/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html

核验：正式摘要与arXiv方法文本；2026-10-14

注意：https://arxiv.org/html/1106.4729 ; 原生相对比与后验适配分别标记

## [X04] A Least-squares Approach to Direct Importance Estimation

出处：Kanamori, Hido and Sugiyama, JMLR 10:1391–1445, 2009

入口：https://jmlr.csail.mit.edu/beta/papers/v10/kanamori09a.html

核验：正式论文摘要与题录；2026-10-14

注意：uLSIF是普通密度比；不是默认S估计

## [X05] Mutual Information Neural Estimation

出处：Belghazi et al., ICML 2018

入口：https://proceedings.mlr.press/v80/belghazi18a.html

核验：正式摘要与题录；2026-10-14

注意：MINE原生Shannon MI；本包语义DV-KL桥接为新任务适配

## [X06] On Variational Bounds of Mutual Information

出处：Poole et al., ICML 2019

入口：https://proceedings.mlr.press/v97/poole19a.html

核验：正式摘要与题录；沿用项目已验证目标实现；2026-10-14

注意：各界的定义、正常数、K与原生/公共评价必须分开

## [X07] Understanding the Limitations of Variational Mutual Information Estimators

出处：Song and Ermon, ICLR 2020

入口：https://arxiv.org/abs/1910.06222

核验：原始摘要与题录；2026-10-14

注意：SMILE必须用纠正后实现；未重新运行原作者实验

## [X08] A Kernel Two-Sample Test

出处：Gretton et al., JMLR 13:723–773, 2012

入口：https://jmlr.org/papers/v13/gretton12a.html

核验：正式摘要与题录；2026-10-14

注意：MMD用于分布检验，不是Bernoulli概率

## [X09] A Kernel Statistical Test of Independence

出处：Gretton et al., NeurIPS 2007

入口：https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html

核验：正式摘要与题录；2026-10-14

注意：HSIC独立性检验；与两样本组标签置换区分

## [X10] General f-divergence / estimator comparison scope

出处：合作者原计划 §13；Poole et al. 2019作为此次核对入口

入口：https://proceedings.mlr.press/v97/poole19a.html

核验：本次方法分类与用户原计划对应；2026-10-14

注意：不是额外独立算法或新增论文计数；generic f-GAN关系按原稿引用

## [X11] Distributional Vision Language Alignment by Cauchy-Schwarz Divergence

出处：CS-Aligner, arXiv 2502.17028 v3（沿用项目发表定位）

入口：https://arxiv.org/html/2502.17028v3

核验：方法文本与摘要；2026-10-14

注意：比较模态边缘分布；不直接作为语义满足概率

## [X12] Sufficient Dimension Reduction via Squared-loss Mutual Information Estimation

出处：Suzuki and Sugiyama, AISTATS 2010

入口：https://proceedings.mlr.press/v9/suzuki10a.html

核验：正式摘要与题录；2026-10-14

注意：已核实的LSMI应用出处；不为未核实的2009题目补假DOI

## [X13] GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation

出处：Li et al., arXiv 2406.13743v3

入口：https://arxiv.org/abs/2406.13743

核验：原始摘要与版本；官方ranking代码沿用v1；2026-10-14

注意：同P02的复核，不算新增独立论文

## [X14] GenAI-Image-Ranking-800 dataset card

出处：zhiqiulin/GenAI-Image-Ranking-800

入口：https://huggingface.co/datasets/zhiqiulin/GenAI-Image-Ranking-800

核验：官方卡片元数据与本项目data_fetch对应；2026-10-14

注意：HF split=train不授权本研究使用其终末人工标签训练

## [G09] V7数据入口新提交

出处：W-Yinghao/VCS_QMI@aeb56b5c7b3c238565dcf9a43804cca130cec9f4

入口：https://github.com/W-Yinghao/VCS_QMI/commit/aeb56b5c7b3c238565dcf9a43804cca130cec9f4

核验：HEAD API及commit diff；2026-10-14

注意：提交时间2026-10-13 21:58:14UTC；脚本登记不证明下载完成；未读实际队列
