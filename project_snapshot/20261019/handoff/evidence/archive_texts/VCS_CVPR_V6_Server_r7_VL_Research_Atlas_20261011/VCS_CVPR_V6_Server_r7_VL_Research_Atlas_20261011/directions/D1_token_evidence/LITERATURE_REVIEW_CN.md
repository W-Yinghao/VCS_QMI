# D1 文献综述｜问题条件下的视觉证据压缩

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 领域是怎样发展的？

这条线研究长视觉序列的计算冗余，但“冗余”有三种不同定义：注意力较少、与当前问题不相关、删掉后任务风险几乎不变。三者不等价。我们希望研究第三种，不能仅凭换一种token重要性评分就认为完成了风险估计。

早期FastV[T01]利用模型内部注意力进行免训练裁减；SparseVLM[T02]引入文本raters、逐层预算和回收。FEATHER[T03]进一步说明：某些方法在通用QA均值上看似安全，却损害OCR/定位等视觉密集任务。这使最终比较必须同时看真正需要局部视觉信息的问题及整系统时延，而不是一张平均保持率表。

IF-Prune[T05]已经将小VLM和变分信息瓶颈用于问题条件化剪枝。2026年预印本DiffPrune[T06]又通过任务驱动可微节流训练选择器；ET-Prune[T07]明确使用“证据感知”和动态预算。因此，“信息论剪枝”“问题相关”“动态budget”都不是空白。最值得继续问的是：**能否预测压缩造成的实际任务风险，并在预算决策中比这些代理信号更有用？**

## 2. 核心文献与实验对象

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [T01] **An Image is Worth 1/2 Tokens After Layer 2: Plug-and-play Inference Acceleration for Large Vision-Language Models** — ECCV 2024 | FastV利用早期层后注意力裁减视觉token。 | LLaVA系; Qwen-VL; 视频任务；原任务表现; FLOPs; 推理延迟。 |
| [T02] **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference** — ICML 2025 | 文本raters、逐层稀疏与token回收；免训练。 | GQA; MMBench; MME; POPE; ScienceQA; SEED; TextVQA; MMVet；官方任务分数; 保持率; FLOPs; CUDA时间。 |
| [T03] **Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration** — ICCV 2025 | 分析视觉中心任务失效；RoPE-free注意力与两阶段裁减。 | 通用QA; TextVQA; 视觉定位等；任务质量与加速。 |
| [T04] **Conical Visual Concentration for Efficient Large Vision-Language Models (PyramidDrop)** — CVPR 2025 | 分阶段减少视觉token，训练/推理设定分别处理。 | 多项图像问答任务；任务准确率; 训练/推理成本。 |
| [T05] **IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models** — CVPR 2026 | 小VLM的变分信息瓶颈提供问题条件下的视觉裁减信号。 | 8项评测；官方README明确含ChartQA，Qwen2-VL/InternVL系；任务表现保持率; token/计算成本。 |
| [T06] **DiffPrune: Differentiable Information Throttling for Token Pruning in Vision-Language Models** — arXiv preprint, August 2026 | 可微信息节流训练token评分器；以任务损失学习裁减。 | LLaVA1.5-7B; LLaVA-NEXT7B; Qwen2.5-VL7B; GQA/TextVQA等10任务；固定预算任务质量; 实际效率。 |
| [T07] **ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs** — arXiv preprint, August 2026 | 问题条件证据评分、文本空间保护与动态预算下限。 | Qwen3-VL8B; InternVL3.5-8B; OCRBench-v2; MMBench1.1；OCR/QA; token数与时延。 |
| [F08] **Qwen2.5-VL Technical Report** — 2025 technical report / official model card | 动态视觉分辨率的开放VLM；作为D1/D2宿主。 | 文档/图表/OCR/通用视觉；各官方任务指标。 |

## 3. 应该重点读哪几篇，而不是全部平均用力？

**IF-Prune是方法定位的必读邻居。** 它使用VIB估计视觉token的信息信号，我们的方法若只是把KL改成CS，理论对象是否仍与“删掉后回答变差”对应，需要证明而不是命名。原摘要的“保留95%表现”是相对baseline，不是95%答案正确。

**DiffPrune是训练设计的直接邻居。** 它说明任务损失已经进入裁减训练。我们不能把“可微选择器+任务监督”据为己有。合理增量可能是：离线独立评价的风险损失、有限预测误差的校正，以及对跨问题预算的可靠分配。但这些都要比同监督量的直接风险回归更好，不能只赢注意力启发式。

**FEATHER决定评价不能太容易。** 文本细节和位置约束应作为预定任务，不是做完后再挑出来的特殊子集。TextVQA、ChartQA、SP-DocVQA都需要原生evaluator；图表数值容差和文档ANLS不能混成“VQA accuracy”。[B02–B04]

## 4. 基线分层

首轮必须有完整未剪枝模型、随机/均匀裁减、FastV、SparseVLM或FEATHER中的至少一条适配成功的强免训练方法；新增训练方法需要匹配的直接风险选择器和Logistic后验选择器。进入正文候选必须补IF-Prune，或明确给出无法复现的原因与同架构替代。T06/T07即便没有现成代码，也要纳入新颖性讨论，不必盲目重写大型未发布系统。

每种方法在同一host上的比较与原作者published host分开；层内attention读取可能改变FlashAttention执行路径，不能不计它的真实开销。更换更强7B宿主本身不算剪枝方法提升。

## 5. 数据与预期优势

GQA提供组合问答与结构化辅助标签，适合二元任务风险验证；TextVQA检验小文字；SP-DocVQA检验页面信息；ChartQA检验视觉加数值推理。MMBench/ScienceQA作为覆盖面补充，不能替代视觉密集任务。建议主宿主为固定revision的Qwen2.5-VL-7B，第二宿主选已经有强剪枝实现的LLaVA1.5-7B；模型规模不是“最新越大越好”。[F08,T01–T07]

**希望验证的优势：**同一token/时延预算下更少任务损失，或同一质量下使用更少计算；被预测的损失与实际损失在不同任务难度中相关且可校准。**不能预定：**CS必然比KL/BCE好，低J损失等于答案不变，或每种任务都能保留相同比例token。

## 6. 风险与转向

若直接风险回归与VCS没有区别，新增贡献可能来自选择器而非散度；若text-rich任务损害明显，优先定位token位置/分辨率处理和训练覆盖，不能改评测分母。若选择器成本吞掉节省，本方向应转为训练期证据分析，不能宣称部署加速。若无法获得超越IF-Prune/DiffPrune/ET-Prune的具体能力，不把它升为主贡献。

## 本方向主来源

- [T01] [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764)。代码/项目：https://github.com/pkunlp-icler/FastV
- [T02] [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://proceedings.mlr.press/v267/zhang25s.html)。代码/项目：https://github.com/Gumpest/SparseVLMs
- [T03] [Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](https://arxiv.org/abs/2412.13180)。代码/项目：https://web.stanford.edu/~markendo/projects/feather
- [T04] [Conical Visual Concentration for Efficient Large Vision-Language Models (PyramidDrop)](https://arxiv.org/abs/2410.17247)。代码/项目：https://github.com/Cooperx521/PyramidDrop
- [T05] [IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.html)。代码/项目：https://github.com/snap-research/EVLM-IF-Prune
- [T06] [DiffPrune: Differentiable Information Throttling for Token Pruning in Vision-Language Models](https://arxiv.org/abs/2608.01985)。本轮未确认可直接运行的官方实现。
- [T07] [ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs](https://arxiv.org/abs/2608.01979)。本轮未确认可直接运行的官方实现。
- [F08] [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)。代码/项目：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
