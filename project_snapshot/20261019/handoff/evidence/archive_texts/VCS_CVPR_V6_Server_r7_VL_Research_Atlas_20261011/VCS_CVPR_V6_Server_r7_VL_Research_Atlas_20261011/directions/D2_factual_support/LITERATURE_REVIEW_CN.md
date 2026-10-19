# D2 文献综述｜视觉事实支持与生成忠实性

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 三种研究问题不能混为“幻觉”

图文相似性衡量话题相关；语义支持判断图像是否支持一个声明；生成忠实性评价完整答案有没有错误且有没有回答问题。我们希望研究后两者，不能把原先joint-vs-product的配对概率直接改名为真实性。

CHAIR[H08]和POPE[H05]使对象幻觉可量化，但不覆盖所有关系与属性。HallusionBench[H06]用图像/问题配对揭示语言先验与视觉错觉；AMBER[H07]同时看生成与判别，特别要求保留信息覆盖。这些工作共同说明：总说“不确定”或生成极短答案不能算成功。

## 2. 核心文献

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [H01] **Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding** — CVPR 2024 | 原图和扰动图条件的输出对比，免训练减轻幻觉。 | POPE等对象幻觉与通用理解；幻觉指标; QA质量; 额外forward成本。 |
| [H02] **Detecting and Preventing Hallucinations in Large Vision Language Models** — AAAI 2024 | M-HalDetect细粒度标注；奖励/偏好优化与best-of-N。 | M-HalDetect; 多源模型回答；声明/片段正确性; 人评; 生成质量。 |
| [H03] **Woodpecker: Hallucination Correction for Multimodal Large Language Models** — 2023 arXiv; venue not re-verified here | 结构化核查与多步骤生成纠错。 | 图文问答/幻觉评测；纠错效果; 信息保持; 推理成本。 |
| [H04] **Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework** — CVPR 2026 | SCI多轮视觉/文本反事实；DRBench动态压力测试。 | DRBench及VLM鲁棒性任务；正确性与测试时计算。 |
| [H05] **Evaluating Object Hallucination in Large Vision-Language Models** — EMNLP 2023 | POPE对象存在性问答，random/popular/adversarial负例。 | COCO来源图像等发布子集；Accuracy; Precision; Recall; F1; Yes ratio。 |
| [H06] **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models** — CVPR 2024 | 成对控制图像/问题诊断语言幻觉与视觉错觉。 | HallusionBench；aAcc; fAcc; qAcc; 分组诊断。 |
| [H07] **AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation** — 2023 arXiv; venue not asserted | 生成和判别两种幻觉评价，不依赖生成式LLM裁判。 | AMBER；CHAIR; Cover; Hal; Cog; Acc/P/R/F1。 |
| [H08] **Object Hallucination in Image Captioning** — EMNLP 2018 | CHAIR用对象提及检测图像描述幻觉。 | COCO captions；CHAIR_i; CHAIR_s; caption质量。 |
| [H09] **TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering** — ICCV 2023 | 把提示词拆成问答，检查生成图是否满足要求。 | TIFA评测集; 多个T2I模型；VQA问答一致性; 人类相关性。 |
| [H10] **Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)** — ECCV 2024 | VQA模型对“是否显示该文字”的Yes概率作为配对评价。 | 多个图文/生成对齐benchmarks；人类偏好相关性/成对排序。 |
| [H11] **Visual Entailment: A Novel Task for Fine-Grained Image Understanding** — 2019 arXiv / SNLI-VE dataset | 图像前提与文本声明的entailment/neutral/contradiction。 | SNLI-VE (Flickr30k + SNLI)；3类accuracy; 本方案另加macro-F1/校准。 |

## 3. 方法线索与真正的竞争者

VCD[H01]利用原图/扰动图的输出差异，说明“视觉对比”不是新点。SCI[H04]进一步使用多轮视觉和文本反事实，意味着“自我核验”也不是空白。它们的多次推理需按同预算比较，而不能让我们的verifier多看几张图、多生成几个答案后与单次baseline比。

M-HalDetect[H02]提供细粒度人类标注并将它用于检测、奖励与偏好优化，是最直接的训练数据/方法参照。我们更有价值的提问是：**把语言本身可预测的支持倾向与视觉新增证据分开，会不会使未见模型回答的核验更准确，并转化为真实的生成改善？**

TIFA[H09]、VQAScore[H10]已把VQA概率用于生成图忠实性评价。采用VCS给候选图打分不是足够的新方法；需要在独立人类标签上证明不同于单张CLIP/VQA分的能力，并在best-of-N或有限生成适配中体现。D2首轮优先caption/VQA核验，T2I奖励保留为第二应用，避免两个生成系统同时搭建。

## 4. 数据应怎样选择？

训练优先M-HalDetect，必要时用SNLI-VE[H11]补充supported/neutral/contradicted语义结构。二者任务标签并不完全相同，不能直接拼接为一组二值数据。POPE、AMBER、HallusionBench只用于独立评价，不提供方便的训练负例。COCO生成caption上用CHAIR配合内容质量，而不是只看CHAIR下降。

AMBER判别P/R/F1以“真值no的幻觉问题”为正类，这是原文定义；POPE的yes/no统计不能原样搬过来。HallusionBench必须保留成对聚合fAcc/qAcc，不能把aAcc当整个benchmark。[H05–H08]

若走T2I评分，则使用BaiqiL版GenAI-Bench/Rank[G08]并留出prompt和generator。人类喜欢某张图不等于图像支持每个声明；偏好、细粒度事实与审美分表。

## 5. 创新与预期优势边界

候选新增点是：有限标注下，估计视觉相对纯文本的证据增益，并将可靠的支持分数用于纠错/生成，而不是简单的二元平方损失。原J保证的后验与平方风险结构可以帮助分析，但不保证生成偏好优化稳定或泛化。

期待的优势是较低的支持概率风险、更强的跨回答生成器泛化，以及在相同覆盖/计算预算下更少错误。必须保留同架构Logistic、直接错误预测、无图/错图控制、VCD与一项现有纠错/奖励方法。若收益消失于这些匹配控制，只能报告额外监督或系统结构收益。

## 6. 实际可行性

D2比D1少一些token内核兼容性风险，且存在人类错误标注；但语义unknown、文本偏差和生成评估很容易污染结论。首先审样本和标签，再训练。最危险的假成功是verifier学到回答风格、critic给自己的输出打高分、或者通过拒答降低幻觉。三个都需要独立外部终点才能排除。

## 本方向主来源

- [H01] [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)。代码/项目：https://github.com/DAMO-NLP-SG/VCD
- [H02] [Detecting and Preventing Hallucinations in Large Vision Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29771)。代码/项目：https://github.com/hendryx-scale/mhal-detect
- [H03] [Woodpecker: Hallucination Correction for Multimodal Large Language Models](https://arxiv.org/abs/2310.16045)。代码/项目：https://github.com/BradyFU/Woodpecker
- [H04] [Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Scaling_Test-Time_Robustness_of_Vision-Language_Models_via_Self-Critical_Inference_Framework_CVPR_2026_paper.html)。本轮未确认可直接运行的官方实现。
- [H05] [Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/)。代码/项目：https://github.com/AoiDragon/POPE
- [H06] [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://arxiv.org/abs/2310.14566)。代码/项目：https://github.com/tianyi-lab/HallusionBench
- [H07] [AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation](https://arxiv.org/abs/2311.07397)。代码/项目：https://github.com/junyangwang0410/AMBER
- [H08] [Object Hallucination in Image Captioning](https://aclanthology.org/D18-1437/)。代码/项目：https://github.com/LisaAnne/Hallucination
- [H09] [TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_TIFA_Accurate_and_Interpretable_Text-to-Image_Faithfulness_Evaluation_with_Question_Answering_ICCV_2023_paper.html)。代码/项目：https://github.com/Yushi-Hu/tifa
- [H10] [Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)](https://arxiv.org/abs/2404.01291)。代码/项目：https://github.com/linzhiqiu/t2v_metrics
- [H11] [Visual Entailment: A Novel Task for Fine-Grained Image Understanding](https://arxiv.org/abs/1901.06706)。代码/项目：https://github.com/necla-ml/SNLI-VE
