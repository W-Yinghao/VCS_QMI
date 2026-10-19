# D5 文献综述｜图文数据选择与再描述

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 这条线为什么需要完整学生训练？

数据过滤的目标不是“过滤分数好看”，而是用它选出的数据训练出更好的模型。DFN[D02]明确表明，筛选网络自己的识别能力与它构造训练集的能力可以不同。DataComp[D01]因此固定候选池、模型、训练预算和多任务评价，避免用数据规模/训练时长变化掩盖策略作用。

Sieve[D03]利用captioner提供另一个语义信号；Recap-DataComp[D04]将原文与重写描述混合；BLIP[D07]更早已经用captioner/filter清洗训练。新的VCS系统不能仅把“检测错配”作为评价终点，更不能只挑高分简单图文对然后假定下游必好。

## 2. 相关工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [D01] **DataComp: In Search of the Next Generation of Multimodal Datasets** — NeurIPS 2023 Datasets and Benchmarks | 固定池、模型及计算的图文数据策展基准。 | DataComp four scales; 38 downstream tasks；38任务平均及ImageNet/检索。 |
| [D02] **Data Filtering Networks** — ICLR 2024 | 筛选网络的效用不等于其自身下游准确率。 | DFN pools; 多项零样本评价；由筛选集训练的模型质量。 |
| [D03] **Sieve: Multimodal Dataset Pruning Using Image Captioning Models** — CVPR 2024 | 生成描述与原文字的语义一致性筛选，结合CLIP。 | DataComp medium/large; 38 tasks；下游质量; 筛选与caption成本。 |
| [D04] **What If We Recaption Billions of Web Images with LLaMA-3?** — ICML 2025 | 大规模再描述，原文与新caption混合用于CLIP及生成训练。 | Recap-DataComp-1B; 多项图文检索和生成；检索; 生成忠实性。 |
| [D05] **Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders** — ICLR 2026 | 稀疏自编码器的目标分布匹配用于图文过滤(SDM)。 | DataComp medium; 38 tasks；目标与整体迁移性能。 |
| [D06] **Multimodal Distribution Matching for Vision-Language Dataset Distillation** — arXiv preprint, May 2026 | 多模态分布匹配用于合成数据蒸馏。 | 图文数据蒸馏；由合成数据训练的下游效果。 |
| [D07] **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** — ICML 2022 | captioner/filter组合清理和补充图文预训练。 | 图文预训练/检索/生成任务；检索与生成质量。 |

## 3. 本轮检索对创新定位的直接改变

ICLR2026的SDM[D05]已经研究目标分布对齐的过滤。因此我们提出“让选出数据接近目标分布”并非新颖性本身。2026预印本MDM[D06]还研究多模态分布匹配的数据蒸馏，但它生成合成数据，不等同于从固定池过滤；不应为了凑强baseline把两个预算体系混合。

可以研究的具体增量是：**目标关系分布未知且样本有限时，如何用学习型CS估计筛选/加权一个关系覆盖更合适的子集，并用下游模型验证这份估计确实有用？** 需要与同目标核MMD/相对回归、Logistic ratio、SDM可用实现和普通CLIP/Sieve过滤比较。

## 4. 数据与评估

首个正式基准推荐DataComp-CLIP Small的完整官方训练协议，而非自动启动medium。其独立图片数、保留比例、examples seen和模型固定；同样数量的小子集反复采样可能与更大unique set不同，二者都要记录。38任务平均和细分任务完整报告，不能只报一个目标域。[D01]

若当前磁盘/许可不允许DataComp，使用CC3M固定可访问子集进行内部验证，但明确它不是官方DataComp结果。Recap子集要保留相同图片的新旧caption，才能隔离“文本变化”与“选图变化”。Caption生成成本与下游训练成本都要计入。[D03,D04]

## 5. 理论风险

从VCS输出反解重要性比r=(1+T)/(1−T)可能极大，所以有界critic并不自动给出稳定数据权重。目标分布在来源池外的部分，也无法通过过滤创造出来。最小化一个拟合不足的critic的J会造成伪匹配，需要独立refit、替代统计量和学生验证。

D5因此不是一条“便宜的备用”。过滤计算便宜，但每个策略要训练学生才能判断；反而可能比D2或D6贵。建议只在固定数据池和学生预算能得到保障时启用。

## 本方向主来源

- [D01] [DataComp: In Search of the Next Generation of Multimodal Datasets](https://arxiv.org/abs/2304.14108)。代码/项目：https://www.datacomp.ai/dcclip/
- [D02] [Data Filtering Networks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html)。代码/项目：https://github.com/apple/ml-dfn
- [D03] [Sieve: Multimodal Dataset Pruning Using Image Captioning Models](https://openaccess.thecvf.com/content/CVPR2024/html/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.html)。代码/项目：https://github.com/facebookresearch/SIEVE
- [D04] [What If We Recaption Billions of Web Images with LLaMA-3?](https://proceedings.mlr.press/v267/li25ch.html)。代码/项目：https://github.com/UCSC-VLAA/Recap-DataComp-1B
- [D05] [Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8291acb610ddc260d554da5a3d273ae9-Abstract-Conference.html)。本轮未确认可直接运行的官方实现。
- [D06] [Multimodal Distribution Matching for Vision-Language Dataset Distillation](https://arxiv.org/abs/2605.23482)。本轮未确认可直接运行的官方实现。
- [D07] [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n.html)。代码/项目：https://github.com/salesforce/BLIP
