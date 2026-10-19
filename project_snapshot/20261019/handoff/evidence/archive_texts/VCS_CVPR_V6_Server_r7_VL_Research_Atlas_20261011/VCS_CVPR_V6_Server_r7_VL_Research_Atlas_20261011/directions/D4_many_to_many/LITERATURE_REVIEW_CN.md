# D4 文献综述｜多对多、不完整标注与概率图文关系

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 任务本身是多义关系，而不只是错误配对

同一图可由不同文字正确描述，同一短句也可适用于多图。原始COCO给每图几条caption，不代表其他图文组合都错。CxC[M05]与ECCV Caption[M06]正是为检索漏标补充语义判断。它们表明传统Recall@1可能奖励寻找唯一标注出处，而不完整反映语义关系。

概率嵌入PCME[M01]、PCME++[M02]已经显式处理多义、不确定性及假负例；ProLIP[M03]把这条线扩到大规模预训练。因此，VCS的有界critic本身不构成“新概率模型”。需要把可训练分布表达、缺失标签假设和实际多正例决策一起定义。

## 2. 相关工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [F05] **Learning Transferable Visual Models From Natural Language Supervision** — ICML 2021 | 图文对比双编码器与零样本迁移。 | 大规模图文预训练; 多个分类和检索集；零样本准确率; 双向Recall@K。 |
| [F06] **Sigmoid Loss for Language Image Pre-Training** — ICCV 2023 | 逐对sigmoid替代批softmax；关注正负比例与配对预算。 | 图文预训练; 零样本分类与检索；准确率; 检索; batch/配对效率。 |
| [F07] **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features** — 2025 technical report | 多目标和数据策略增强图文表示；作为现代预训练参考。 | 多语言图文; 定位及稠密特征任务；分类/检索/定位。 |
| [M01] **Probabilistic Embeddings for Cross-Modal Retrieval** — CVPR 2021 | 高斯概率嵌入与不确定性，多对多检索。 | COCO Caption; CUB Captions；Recall@K; PMRP/类相关检索。 |
| [M02] **Improved Probabilistic Image-Text Representations (PCME++)** — ICLR 2024 | 闭式概率距离与伪正例、混合训练缓解假负例。 | COCO Caption; CxC; ECCV Caption；多正例检索; 噪声/不确定性。 |
| [M03] **ProLIP: Probabilistic Language-Image Pre-Training** — ICLR 2025 | 大规模概率预训练与语义包含/不确定性。 | DataComp; 零样本分类; 检索; 不确定性任务；检索/分类/包含和不确定性。 |
| [M04] **LongProLIP: A Probabilistic Vision-Language Model with Long Context Text** — ICLR 2025 uncertainty workshop tiny paper / arXiv, not main | 概率模型拓展长文本。 | Urban1k; DataComp任务等；长描述检索; 分类保持。 |
| [M05] **Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO** — EACL 2021 | 补充图图/文文/图文的人类语义相似度。 | CxC on COCO；语义相关性; retrieval。 |
| [M06] **ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO** — ECCV 2022 | 修复COCO检索漏标正例，引入mAP@R。 | ECCV Caption / COCO Karpathy test subset；mAP@R; R-Precision; Recall@K。 |
| [M07] **FLAIR: VLM with Fine-Grained Language-Informed Image Representations** — CVPR 2025 | 细描述与文本条件视觉池化支持局部关系。 | 细粒度/长描述检索与区域任务；检索/局部对应。 |
| [M08] **SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations** — 2024 arXiv / verify venue before manuscript | 同义正描述与困难负描述，分离句式和语义。 | SugarCrepe++；官方双正例组合指标。 |

## 3. 三个最重要的比较

PCME++是自然训练对照，因为它不仅替换距离，还使用伪正例/混合策略。若VCS有额外伪标签、PCME++没有，应分开同标签预算与完整官方recipe两张表。

ProLIP提供强预训练不确定性参考，其额外训练数据、包含目标和概率表示不能免费算成“BCE对照”。LongProLIP是workshop tiny paper，作用是提醒长文本context本身会改变检索与零样本取舍，不能让VCS输入256tokens、基线截断64后归因于散度。

ECCV Caption是主要外部评价，不是训练负例矿。其查询子集和人工judge范围有限；官方mAP@R结果不等于全集所有pair的概率准确率。CxC的0–5语义相似度也不能除以5后就当groundtruth Bernoulli后验。[M05,M06]

## 4. 两条不同应用选择

**A. 有标注或类相关的多对多关系。** 在CUB class-relevant协议中可以明确多个相关目标，作为受控任务，但需称“同类检索”，不称每条caption真实支持每张图。自然数据若有充分judged positives/negatives，再评价语义概率。

**B. 不完整自然图文训练。** 未观察到的关系不是负类。可以维持joint-vs-product机制、明确它估计的是采样关系；也可以假设正未标记模型并估计标注倾向，但需要额外的可识别条件和PU先例，不能直接以“软标签”修好。第一轮不强制增加新的PU理论。

## 5. 可能优势和最大风险

期待在同信息量下保留更多正确匹配、对歧义查询更合理地表达不确定性，而不仅R@1高一点。真正需要同时比较mAP@R、R-Precision、官方R@K、已judge子集风险、检索集合大小与准确率。

若只有一个标量T，uncertainty应称“匹配证据不确定”，不能称数据模态的概率分布。若使用K个原型或Gaussian embeddings，需要同样给Logistic/contrastive基线，以免把架构收益归给VCS。CUB positive定义、COCO漏标、CxC语义连续标签三个问题也不能混成一个目标。

这条最容易接续现有双塔，但也最容易退回“新配方vs旧检索”。所以本包把它列为自然多正例标签足够时的备选，而不是默认重新训练CLIP。

## 本方向主来源

- [F05] [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)。代码/项目：https://github.com/openai/CLIP
- [F06] [Sigmoid Loss for Language Image Pre-Training](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)。代码/项目：https://github.com/google-research/big_vision
- [F07] [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)。代码/项目：https://huggingface.co/google
- [M01] [Probabilistic Embeddings for Cross-Modal Retrieval](https://arxiv.org/abs/2101.05068)。代码/项目：https://github.com/naver-ai/pcme
- [M02] [Improved Probabilistic Image-Text Representations (PCME++)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ad9d6ab10446114cf5482d5e1f971a84-Abstract-Conference.html)。代码/项目：https://github.com/naver-ai/pcmepp
- [M03] [ProLIP: Probabilistic Language-Image Pre-Training](https://arxiv.org/abs/2410.18857)。代码/项目：https://github.com/naver-ai/prolip
- [M04] [LongProLIP: A Probabilistic Vision-Language Model with Long Context Text](https://arxiv.org/abs/2503.08048)。本轮未确认可直接运行的官方实现。
- [M05] [Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO](https://aclanthology.org/2021.eacl-main.249/)。代码/项目：https://github.com/google-research-datasets/Crisscrossed-Captions
- [M06] [ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO](https://arxiv.org/abs/2204.03359)。代码/项目：https://github.com/naver-ai/eccv-caption
- [M07] [FLAIR: VLM with Fine-Grained Language-Informed Image Representations](https://arxiv.org/abs/2412.03561)。代码/项目：https://github.com/ExplainableML/flair
- [M08] [SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations](https://arxiv.org/abs/2406.11171)。本轮未确认可直接运行的官方实现。
