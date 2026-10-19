# 来源目录与核验范围


所有网络来源核验日期：2026-10-13。以下是原始论文/官方代码/卡片。不同来源包含不同层次的事实；未完成运行复现与全文核验的项明确说明。


原始理论全文已在对话提供；本包不复制他人全文或字体。


## [F01] 合作者原计划

**来源：** 用户提供：Variational Cauchy–Schwarz Quadratic Mutual Information — Research Plan, September 2026

**入口：** conversation:turn581file3

**本轮核验：** 全文；重点 §2–8、§10、§17

**注意：** 原始理论；不把原草稿全部创新表述当成已完成新颖性审计


## [F02] 历史审查的修订附录

**来源：** 用户提供：VCS_QMI 项目综合文档，附录4

**入口：** conversation:turn581file0#L326-L359

**本轮核验：** 已给出的修订附录

**注意：** 只继承已明确的归属与勘误；不把10月10日旧状态当成当前状态


## [G01] 本项目快照

**来源：** W-Yinghao/VCS_QMI @ 3e2fe59c4861abae46537574c4c59f9bedfe91c6

**入口：** https://github.com/W-Yinghao/VCS_QMI/tree/3e2fe59c4861abae46537574c4c59f9bedfe91c6

**本轮核验：** GitHub main API

**注意：** 截至2026-10-13 18:51:49 UTC；不是实时Slurm查询


## [P01] EvalMuse / FGA-BLIP2 / PN-VQA

**来源：** EvalMuse-40K, AAAI 2026；arXiv版本标题略不同

**入口：** https://ojs.aaai.org/index.php/AAAI/article/view/42458

**本轮核验：** 正式出版摘要；arXiv HTML方法和数据节

**注意：** arXiv: https://arxiv.org/html/2412.18150v2


## [D01] EvalMuse公开数据

**来源：** DY-Evalab/EvalMuse

**入口：** https://huggingface.co/datasets/DY-Evalab/EvalMuse

**本轮核验：** 官方卡片、字段预览

**注意：** 卡片有test标注稍后公开说明；以实际访问为准，不能假定可下载测试标签


## [G02] FGA核心代码

**来源：** lavis/models/blip2_models/fga_blip2.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/models/blip2_models/fga_blip2.py

**本轮核验：** 完整源码

**注意：** 活跃itm分支是L1；variance分支注释且在return后


## [G03] FGA训练配置

**来源：** lavis/projects/blip2/train/fga_blip2.yaml

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/projects/blip2/train/fga_blip2.yaml

**本轮核验：** 完整源码

**注意：** freeze_vit=True, 5 epochs, batch14, image364；有效运行配置仍须展开


## [G04] FGA数据读取

**来源：** align_datasets.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/datasets/datasets/align_datasets.py

**本轮核验：** 完整源码

**注意：** 32 token截断；第0位token_score是总体分数，不是语义元素标签


## [G05] FGA元素映射

**来源：** process/element2mask.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/process/element2mask.py

**本轮核验：** 完整源码

**注意：** 多个元素的重叠token标签被平均；失败样本会被过滤


## [G06] FGA标签预处理

**来源：** process/process_train.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/process/process_train.py

**本轮核验：** 完整源码

**注意：** 原始投票转均值；按prompt算总体评分方差；保留原始投票


## [P02] GenAI-Bench / GenAI-Rank

**来源：** GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation, 2024

**入口：** https://arxiv.org/html/2406.13743

**本轮核验：** 全文相关数据、排名协议节

**注意：** 2024 SynData workshop记录；不写成CVPR主会论文


## [D02] GenAI数据入口

**来源：** BaiqiL/GenAI-Bench

**入口：** https://huggingface.co/datasets/BaiqiL/GenAI-Bench

**本轮核验：** 官方卡片

**注意：** Bench与Rank提示重叠，必须作为共同终末评测资源


## [G07] GenAI-Rank官方脚本

**来源：** genai_image_ranking.py

**入口：** https://github.com/linzhiqiu/t2v_metrics/blob/main/genai_image_ranking.py

**本轮核验：** 源码1–245行；blob 0e0fdb3e9550b1b1217c9feb78c46ed1095a7252

**注意：** DALLE_3 / SDXL_Base；N=3按固定三个连续分组；服务器锁定commit


## [G08] VQAScore工具版本

**来源：** t2v_metrics README

**入口：** https://github.com/linzhiqiu/t2v_metrics

**本轮核验：** 官方README

**注意：** 原论文复现用legacy v3.0；现代宿主用另一个已锁定环境


## [P03] VQAScore

**来源：** Evaluating Text-to-Visual Generation with Image-to-Text Generation, ECCV 2024

**入口：** https://arxiv.org/abs/2404.01291

**本轮核验：** 原论文与官方项目说明

**注意：** 必须比较原生公开强模型，不只比较CLIPScore


## [P04] TIFA

**来源：** TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering, ICCV 2023

**入口：** https://arxiv.org/abs/2303.11897

**本轮核验：** 原论文摘要与EvalMuse关联协议

**注意：** TIFA160是人工对齐子集，不等于4K prompt的TIFA v1.0


## [P05] DSG

**来源：** Davidsonian Scene Graph, ICLR 2024

**入口：** https://proceedings.iclr.cc/paper_files/paper/2024/hash/446905abfec9db8856f7f7465bbd5552-Abstract-Conference.html

**本轮核验：** 正式摘要

**注意：** 对象与属性依赖、唯一原子问题不是本文新发明


## [P06] RichHF

**来源：** Rich Human Feedback for Text-to-Image Generation, CVPR 2024

**入口：** https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.html

**本轮核验：** 正式摘要

**注意：** 有多种质量维度；只把对齐维度纳入本轮主要相关性


## [P07] HPSv3

**来源：** HPSv3: Towards Wide-Spectrum Human Preference Score, ICCV 2025

**入口：** https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html

**本轮核验：** 正式摘要

**注意：** 综合偏好，不是同定义的语义满足概率


## [P08] VisionReward

**来源：** VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning, AAAI 2026

**入口：** https://ojs.aaai.org/index.php/AAAI/article/view/38107

**本轮核验：** 正式摘要

**注意：** 多维偏好参考；记录实际输出维度与聚合方法


## [P09] DynEval

**来源：** DynEval: Holistic Evaluations of T2I Generative Models in the Wild, 2026 preprint

**入口：** https://arxiv.org/abs/2607.11199

**本轮核验：** 论文HTML与方法概述

**注意：** 模型/权重入口本轮未完成可执行复现核验；不得虚构官方数值


## [P10] DEPICT

**来源：** DEPICT: Scoring Text-to-Image Alignment by Answer Agreement, 2026 preprint

**入口：** https://arxiv.org/abs/2610.03617

**本轮核验：** 论文HTML

**注意：** 本轮未确认可用官方代码；论文复现必须独立命名


## [P11] NegBench

**来源：** Vision-Language Models Do Not Understand Negation

**入口：** https://arxiv.org/abs/2501.09425

**本轮核验：** 原始摘要与官方项目页

**注意：** https://negbench.github.io/；选取图像子任务，不能跨18种任务混成总准确率


## [P12] Winoground

**来源：** Winoground, CVPR 2022

**入口：** https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.html

**本轮核验：** 正式摘要

**注意：** 官方text/image/group三类判据，严格不等式


## [T01] RPC

**来源：** Self-supervised Representation Learning with Relative Predictive Coding, ICLR 2021

**入口：** https://arxiv.org/abs/2103.11275

**本轮核验：** 原文关系＋项目已核对附录

**注意：** 平衡RPC与J完全重合，不另造重复算法行


## [T02] 正规复合损失

**来源：** Composite Binary Losses, JMLR 2010

**入口：** https://jmlr.org/papers/v11/reid10a.html

**本轮核验：** 原始论文页面

**注意：** 用于区分总体proper性质、受限函数类和有限优化


## [P13] 人类评价协议

**来源：** Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation, CVPR 2023

**入口：** https://openaccess.thecvf.com/content/CVPR2023/html/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.html

**本轮核验：** 正式摘要

**注意：** 仅在新生成人工确认时使用；首轮优先已有人工标签
