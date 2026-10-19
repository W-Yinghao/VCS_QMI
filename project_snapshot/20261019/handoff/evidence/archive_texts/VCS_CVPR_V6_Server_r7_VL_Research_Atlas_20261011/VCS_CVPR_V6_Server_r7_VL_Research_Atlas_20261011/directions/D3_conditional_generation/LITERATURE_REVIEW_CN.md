# D3 文献综述｜条件生成分布比较

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 这不是另一个CLIPScore

单张生成图与prompt是否相符、整组图是否逼真、合理结果是否覆盖充分，是三个不同问题。FID/CMMD主要比较边缘图像分布；TIFA/VQAScore/GenEval主要看prompt是否被满足；条件分布评价则希望区分“图像都很好但条件配错”与“条件正确但模式缺失”。

本轮特别核对到CFID[G02]和cFreD[G03]，后者已是WACV2026工作。**条件生成分布比较不是未被占据的空白。** 因而我们的潜在增量只能是有限每条件样本、高维/多模态非高斯关系或更可靠的分辨能力，不能简单声称首次把文本加到分布指标。

## 2. 核心工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [G01] **Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)** — CVPR 2024 | CLIP空间核MMD，不再使用FID的高斯假设。 | 多个图像生成模型/受控扰动；分布距离; 人评相关性; 样本效率。 |
| [G02] **Conditional Frechet Inception Distance** — 2021/2022 arXiv revision | 条件Wasserstein/FID的闭式高斯形式。 | 条件生成；CFID与FID/MSE对比。 |
| [G03] **Evaluating Text-to-Image and Text-to-Video Synthesis with a Conditional Fréchet Distance (cFreD)** — WACV 2026, author publication page verified | 联合视觉质量与条件文本的一致性，条件Fréchet距离。 | HPDv2; GenAI-Bench; PartiPrompts; COCO；人类相关性与模型排序。 |
| [G04] **GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment** — NeurIPS 2023 Datasets and Benchmarks | 基于对象检测的可组合T2I要求检查。 | GenEval prompts；单物体/多物体/数量/颜色/位置/颜色绑定及平均。 |
| [G05] **T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation** — NeurIPS 2023 | 属性绑定、对象关系及复杂组合的分项评价。 | 6000 prompts / 6 categories (original)；绑定/关系各官方评价器。 |
| [G06] **T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation** — TPAMI 2025 | 扩展组合类别和更细粒度诊断。 | 8000 prompts / 8 categories (++ release)；版本对应的官方子类分数。 |
| [G07] **ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty** — NeurIPS 2024 Datasets and Benchmarks | 按概念数量控制生成要求难度，自动问答并有人类验证。 | ConceptMix；各难度满足概念比例与全部满足率。 |
| [G08] **GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation** — 2024 arXiv / CVPR 2024 SynData workshop, not claimed main track | 复杂提示、图/视频生成的人类对齐评分；GenAI-Rank候选比较。 | GenAI-Bench / GenAI-Rank；人类相关性; 成对偏好; best-of-N。 |
| [G09] **Demystifying MMD GANs** — ICLR 2018 | KID的无偏MMD评价背景。 | 生成图像；KID; MMD。 |
| [G10] **Reliable Fidelity and Diversity Metrics for Generative Models** — ICML 2020 | Density/Coverage把保真和覆盖分开。 | 生成模型；Density; Coverage; precision/recall相关比较。 |
| [H10] **Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)** — ECCV 2024 | VQA模型对“是否显示该文字”的Yes概率作为配对评价。 | 多个图文/生成对齐benchmarks；人类偏好相关性/成对排序。 |

## 3. 三类基线各自负责什么？

CMMD[G01]使用CLIP表示的核MMD；它的“C”不是conditional。我们应给它同样的特征和每条件样本，同时增加强的条件/分组MMD控制，不能只跟无条件FID比来制造容易的条件错配优势。CFID/cFreD是直接比较对象；KID和Density/Coverage[G09/G10]分别提供采样与模式覆盖参照。

GenEval、T2I-CompBench与ConceptMix[G04–G07]不是“真实生成分布”，而是提示语义要求的评价器。它们提供外部任务检查，不提供自然S真值。GenAI-Bench[G08]和HPDv2的人评可以评价指标是否有用，但人类审美、语义忠实与多样性需要分项，不指望一个标量解释全部。

## 4. 最重要的数据难题：参考分布究竟是什么？

给每个自然prompt一张真实图，并不能定义该prompt下所有合理图像的总体。我们建议先用可控制的场景/渲染条件建立分布对象，再在有多实例的语义条件组上验证；自然细描述只有单例时，改称有限参考条件兼容性，不宣称准确估计未知总体。

同一prompt中的真实/生成图片如果有renderer、水印或来源风格差别，critic可能仅识别来源。跨生成器、风格控制和同边缘条件置乱可以检查这个问题。固定共同视觉特征的指标检验的是该特征观测中的分布，不等于原像素总体。

## 5. 可以争取什么优势？

希望VCS的学习型critic在有限样本下比固定核特征更善于捕获非线性条件差异，并通过共同读数说明误差；目标是识别受控条件破坏和模式缺失、可靠排序生成分布，而不是高J就好。拟合方式匹配Logistic后读取同J，若同样好则承认学习型函数类收益。

相比D1/D2，这条对estimator依赖最直接，完整“应用”可以是一套从样本到可靠评价的系统，不强制训练大扩散模型。但至少要有已知真值/可验证方向的受控结果与真实外部用途，不能只用一个对真实图像无oracle的J表自证。

## 6. 风险

小样本条件critic可能记住prompt或样本身份；最大化估计的有限经验差异产生偏高读数；强依赖又可能饱和。只要求conditional semantic discrepancy可检测，而没有比CFID/条件MMD更好的精度、成本或适用条件，则不够形成新评估方法。生成模型微调属于第二阶段，别让“又训练一个generator”掩盖指标可靠性问题。

## 本方向主来源

- [G01] [Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603)。代码/项目：https://github.com/google-research/google-research/tree/master/cmmd
- [G02] [Conditional Frechet Inception Distance](https://arxiv.org/abs/2103.11521)。代码/项目：https://github.com/Michael-Soloveitchik/CFID
- [G03] [Evaluating Text-to-Image and Text-to-Video Synthesis with a Conditional Fréchet Distance (cFreD)](https://arxiv.org/abs/2503.21721)。代码/项目：https://vislang.ai/publications/koo2026evaluating
- [G04] [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)。代码/项目：https://github.com/djghosh13/geneval
- [G05] [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350v2)。代码/项目：https://github.com/Karine-Huang/T2I-CompBench
- [G06] [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation](https://doi.org/10.1109/TPAMI.2025.3531907)。代码/项目：https://github.com/Karine-Huang/T2I-CompBench
- [G07] [ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty](https://arxiv.org/abs/2408.14339)。代码/项目：https://princetonvisualai.github.io/conceptmix/
- [G08] [GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation](https://arxiv.org/abs/2406.13743)。代码/项目：https://huggingface.co/datasets/BaiqiL/GenAI-Bench
- [G09] [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)。本轮未确认可直接运行的官方实现。
- [G10] [Reliable Fidelity and Diversity Metrics for Generative Models](https://proceedings.mlr.press/v119/naeem20a.html)。本轮未确认可直接运行的官方实现。
- [H10] [Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)](https://arxiv.org/abs/2404.01291)。代码/项目：https://github.com/linzhiqiu/t2v_metrics
