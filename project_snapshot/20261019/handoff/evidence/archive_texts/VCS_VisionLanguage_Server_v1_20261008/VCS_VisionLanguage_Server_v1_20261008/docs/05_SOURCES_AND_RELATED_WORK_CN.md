# 来源、相关工作与新增建议的边界

核查日期2026-10-08。本文是源索引与简要综述，不是所有文献的完整重读。公开元数据、数据内容与官方入口已核查；图像包未下载，权重未评价。下载URL、许可与镜像仍由服务器接入时复核。

## 1. 原计划与项目证据

**[O1] 合作者原计划：** *Variational Cauchy–Schwarz Quadratic Mutual Information: A Stable Alternative to MINE*，September 2026，10页。重点：§2–5混合参考、J与最优critic；§7平方回归解释；§8已有散度与重参数化；§10稳定性的具体含义；§12实验指标。方案直接保留其S/J，不把Shannon MI作为S真值。

**[O2] 2026-09-28补充意见：** *对《VCS_QMI_Estimator_Research_and_Server_v1》的补充意见*。§1.1记录P72负结果与balanced-logistic后验；§2.6讨论平方增量；§3.7要求保留神经横向对照。此来源中的“VCS特有”等措辞需要结合后续实验和既有方法理解；本包不把它扩成所有VCS实现必然优越的结论。

**[R0] 本次确认HEAD：**
<https://github.com/W-Yinghao/VCS_QMI/commit/6f30f9b5cf85b8af1b4780a24c2103824d96c3cb>

**[R1] 精确配对修正的P80完成报告（本轮重读）：**
<https://github.com/W-Yinghao/VCS_QMI/blob/6f30f9b5cf85b8af1b4780a24c2103824d96c3cb/reports/P80_R2_EXACT_PAIRING_REPORT_20260928.md>

**[R2] 主题配对的P72完成报告（此前已读取，本轮沿用）：**
<https://github.com/W-Yinghao/VCS_QMI/blob/6f30f9b5cf85b8af1b4780a24c2103824d96c3cb/reports/P72_ROBUSTNESS_R2R3_REPORT_20260928.md>

**[R3] P79修正协议（此前已读取，本轮沿用）：**
<https://github.com/W-Yinghao/VCS_QMI/blob/6f30f9b5cf85b8af1b4780a24c2103824d96c3cb/reports/P79_R2_EXACT_PAIRING_PREREG_FROZEN_20260928.md>

这些是历史记录，不是要求重跑的目标数字。当前服务器后来变更的状态不由本包推断。

## 2. 数据来源

| ID | 官方／原作者来源 | 核查用于什么 |
|---|---|---|
| D1 | <https://github.com/BryanPlummer/flickr30k_entities> | 实体共指、框、split、图片条款 |
| D2 | <https://github.com/lichengunc/refer> | RefCOCOg、UMD和其他split、获取入口 |
| D3 | 同D2；另见L7 | RefCOCO+原生指代任务 |
| D4 | 同D2；另见L7 | RefCOCO原生指代任务 |
| D5 | <https://visualgenome.org/> | VG入口；本轮自动解析未成功，下载待核对 |
| D6 | <https://cs.stanford.edu/people/dorarad/gqa/download.html> | GQA下载、场景图／图像来源 |
| D7 | <https://github.com/facebookresearch/DCI> | 密集描述与区域；官方原描述/生成摘要区别；许可要求 |
| D8 | <https://google.github.io/docci/> | 15k长人工描述、官方数据与许可 |
| D9 | <https://github.com/ChenyunWu/PhraseCutDataset> | mask、区域表达、词类标签、划分 |
| D10 | <https://github.com/ccvl/clevr-refplus-dataset-gen> | 合成场景／表达程序、框/mask工具 |
| D11 | <https://cocodataset.org/#download> | COCO官方图文资源入口 |
| D12 | 同D1的原始Flickr图片链接 | 同源多caption和图片授权 |
| D13a | <https://data.caltech.edu/records/65de6-vp158> | CUB原始图片／属性／框/部位资源 |
| D13b | <https://github.com/reedscot/cvpr2016> | 细粒度图文描述资源与原评测 |
| D14 | <https://github.com/RAIVNLab/sugar-crepe> | 语义困难负例、COCO val2017来源 |
| D15 | <https://github.com/Sri-Harsha/scpp> | 同义正描述和困难负描述 |
| D16 | <https://github.com/mertyg/vision-language-models-are-bows> | ARO任务与VG/GQA/COCO/Flickr来源 |
| D17 | <https://github.com/IMirandaM/BiVLC> | 双向双图双文评测、COCO及生成图像来源 |
| D18 | 同D2的RefCLEF接口 | 非COCO指代表达备用 |

另外参考官方数据实现对split的说明：
<https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/ref_coco/README.md>

**重要重叠图：** RefCOCO/+/g与COCO共用图片；VG/GQA/PhraseCut/ARO可能共用图片；Flickr Entities与Flickr检索/ARO-Flickr共用图片；SugarCrepe/++与BiVLC需追溯COCO来源；DCI区域共享其基础图像。独立数据条目不等于独立测试图片集合。

## 3. 与本设计最直接相关的文献

### [L1] CLIP，ICML 2021

*Learning Transferable Visual Models From Natural Language Supervision*。
<https://proceedings.mlr.press/v139/radford21a.html>

CLIP提供图文双编码器和配对学习基础。本方案首先保留原始CLIP的任务表现，避免把适配损害初始能力误读成数据噪声的性质；不是从零预训练一个新CLIP。

### [L2] SigLIP，ICCV 2023

*Sigmoid Loss for Language Image Pre-Training*。
<https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html>

逐对sigmoid不依赖全局softmax归一化，并研究样本、配对数与正负比例。因而“逐对可加”“无需全局归一化”不是本应用独有优点。平衡JS用于同后验拟合比较；SigLIP或明确标记的sigmoid任务控制用于另一权重方案比较。

### [L3] FLAIR，CVPR 2025

*FLAIR: VLM with Fine-grained Language-informed Image Representations*。
<https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_FLAIR_VLM_with_Fine-grained_Language-informed_Image_Representations_CVPR_2025_paper.html>
<https://github.com/ExplainableML/flair>

工作用详细描述与子caption学习局部/文本条件化图像表示。它提醒我们：缺乏局部视觉证据不一定由损失修复。采用其局部特征时，VCS与全部主要对照须同时获得相同输入。

### [L4] FG-CLIP，ICML 2025；FG-CLIP 2，ICML 2026

*FG-CLIP: Fine-Grained Visual and Textual Alignment*；*FG-CLIP 2: A Bilingual Fine-grained Vision-language Alignment Model*。
<https://github.com/360CVGroup/FG-CLIP>

作者实现结合区域文本、较长描述与困难负例，并扩展到双语多粒度训练。这里作为强预训练参考，不把不同训练数据规模下的绝对结果用来隔离二次损失贡献。选定版本后保存权重与revision。

### [L5] CS-Aligner，ICLR 2026

*Distributional Vision-Language Alignment by Cauchy–Schwarz Divergence*。
<https://arxiv.org/abs/2502.17028>

它研究图像与文本的分布对齐，与配对学习结合。本文的主对象是条件联合关系对边缘乘积的S。不能仅因名字都含CS，就将两个对象或核CS损失互换。只有专门增加模态分布对齐消融时才引入其对应目标，不进入首轮。

### [L6] Relative Predictive Coding，ICLR 2021

*Self-supervised Representation Learning with Relative Predictive Coding*。
<https://arxiv.org/abs/2103.11275>

二次配对目标有先行工作；当前V6已经识别balanced RPC与原J的代数关系。VL应用不重新宣称发明一般平方配对损失。新增内容要落在明确关系、采样、有限样本行为及其视觉用途。

### [L7] 指代表达的上下文，ECCV 2016

*Modeling Context in Referring Expressions*。
<https://arxiv.org/abs/1608.00272>

同图其他对象的上下文对指代有作用。因此，从tight crop变为context是一项有根据的候选，但属于共同视觉信息的改动，不是VCS散度本身的理论保证。

### [L8] DCI，CVPR 2024；DOCCI，ECCV 2024

*A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions*；*DOCCI: Descriptions of Connected and Contrasting Images*。入口见D7/D8。

这两类数据支持从短caption扩展到细节描述，但DCI包含局部mask文本，DOCCI主要是整图长描述；二者不能共用一个未经更改的区域真值定义。

### [L9] 组合性与评价偏差

ARO/NegCLIP（ICLR 2023，D16），SugarCrepe（NeurIPS 2023，D14），SugarCrepe++（<https://arxiv.org/abs/2406.11171> ，D15），BiVLC（<https://arxiv.org/abs/2406.09952> ，D17）。

这些工作分别检查属性/词序、负文本捷径、同义改写与双向匹配。仅有高随机错配AUROC不足以代表细粒度理解。使用其测试时保留原输入与评价协议，不能用生成负例的风格分类替代图文匹配。

## 4. 哪些内容是本轮提出，而非来源已经证明？

本方案中的图内均匀区域采样、P/Q权重矩阵、低N网格、F2小型评分器、两类选择checkpoint、适配流程、失败分支和推进门槛，均为本轮为服务器制定的实施建议。它们不是已完成实验，也不保证VCS在所选数据集胜过JS或SigLIP。

排名联系、候选数量影响S、非均匀先验修正和输入碰撞的推前分布，是从原J/密度比结构出发的有限模型推导；本包用CPU代码独立核对，不称为未经文献排他性检查的新主定理。
