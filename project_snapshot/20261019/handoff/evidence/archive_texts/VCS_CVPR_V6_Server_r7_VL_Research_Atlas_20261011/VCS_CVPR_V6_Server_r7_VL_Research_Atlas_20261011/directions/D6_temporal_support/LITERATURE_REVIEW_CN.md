# D6 文献综述｜叙述视频的时间支持与定位

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 从弱叙述对齐到查询条件时间支持

视频中的一句话可能早于动作、晚于动作、概括几个步骤，或者与当前画面无关。MIL-NCE[V01]早已用多实例正袋处理叙述错位；因此“允许时间不对齐”不是一个新点。

QVHighlights[V02]将自然语言查询、多个相关时间段和片段显著性统一起来；QD-DETR[V03]、CG-DETR[V05]强调查询对视频表示的选择性；EaTR[V04]利用事件结构，BM-DETR[V06]处理背景。近年的Sim-DETR[V07]、SG-DETR[V08]继续改进完整定位与显著性系统。这里的机会不是把RefCOCO的框换成视频帧继续排Top-1，而是**学习随时间变化、能支持多段输出和不同持续时长的语言证据**。

## 2. 相关工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [V01] **End-to-End Learning of Visual Representations from Uncurated Instructional Videos (MIL-NCE)** — CVPR 2020 | 多实例NCE处理叙述视频固有时间错配。 | HowTo100M train; YouCook2/MSR-VTT/CrossTask/COIN等；检索; 动作分类/定位/分割。 |
| [V02] **QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries** — NeurIPS 2021 | 查询条件多时间段定位与显著性，Moment-DETR端到端头。 | QVHighlights；MR mAP; R@1; HD mAP/Hit@1。 |
| [V03] **Query-Dependent Video Representation for Moment Retrieval and Highlight Detection (QD-DETR)** — CVPR 2023 | 查询依赖表示与负查询训练。 | QVHighlights; TVSum; Charades-STA；moment retrieval/highlight官方指标。 |
| [V04] **Knowing Where to Focus: Event-aware Transformer for Video Grounding (EaTR)** — ICCV 2023 | 事件结构引导视频定位。 | 视频语言时间定位集；R@1 tIoU; mAP等官方任务指标。 |
| [V05] **Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding (CG-DETR)** — 2023/2024 arXiv revision; venue not re-verified | dummy tokens与跨模态相关性调节查询参与程度。 | QVHighlights; temporal grounding benchmarks；MR/HD指标。 |
| [V06] **Background-Aware Moment Detection for Video Moment Retrieval (BM-DETR)** — WACV 2025 | 显式处理相关片段与背景。 | QVHighlights等时间定位集；moment retrieval指标。 |
| [V07] **Sim-DETR: Unlock DETR for Temporal Sentence Grounding** — ICCV 2025 | 处理查询竞争与全局/局部定位。 | temporal sentence grounding benchmarks；官方定位指标。 |
| [V08] **Saliency-Guided DETR for Moment Retrieval and Highlight Detection (SG-DETR)** — WACV 2026 | 显著性引导MR/HD，使用更强视频预训练信息。 | QVHighlights及时间定位；MR/HD。 |
| [V09] **Hierarchical Video-Moment Retrieval and Step-Captioning (HiREST)** — CVPR 2023 | 长视频中的层级片段和步骤描述。 | HiREST；视频/片段retrieval; step captioning。 |
| [V10] **MomentSeeker: A Benchmark for Long-Video Moment Retrieval** — NeurIPS 2025 Datasets and Benchmarks | 更长视频和多种查询形态。 | MomentSeeker；长视频定位/检索。 |
| [V11] **NaQ: Leveraging Narrations as Queries to Supervise Episodic Memory** — CVPR 2023 | 将叙述转为Ego4D NLQ训练监督。 | Ego4D NLQ；R@1/5 at tIoU 0.3/0.5。 |

## 3. 哪个benchmark最合适？

首选QVHighlights，因为它同一视频/查询下有多相关moments与saliency，可以同时评价“是否支持”和“在哪里支持”。官方提供预提取特征与完整Moment-DETR训练/eval，适合在可控资源内比较。缓存特征训练完整定位头是该领域标准协议，但不是更新全部视频encoder，更不是实时原视频加速。[V02]

Charades-STA作为第二数据集，检验场景与表达变化；固定train/test及feature来源。ActivityNet Captions可扩展更长活动；HiREST有层级步骤；Ego4D NLQ的第一人称长视频和MomentSeeker长查询适合最后验证，不宜第一批全部下载。[V04,V09–V11]

## 4. 我们的方法可能做什么？

选择被标注为支持/不支持查询的片段分布P/Q，学习T并经过明确先验校准生成时间支持场。由该支持场指导片段预算、融合和区间头，同时保留标准span边界/匹配损失。原始S/J可以分析证据拟合，实际终点则是MR/HD和置信度，而不是只最大化时间文本依赖。

最接近的控制是同宿主Logistic saliency和标准QD-DETR/CG-DETR，而不是只比随机负例InfoNCE。若增加相邻时间的平滑或持续时间先验，每条方法得到同样后处理机会；这些结构也是经典时间建模，不归给CS散度。

## 5. 预期价值与限制

希望在多段支持、长背景以及规定时间抖动条件下，得到更准确的support风险和完整定位；或用较少片段取得相似定位质量。先检验噪声模型是否对应真实标注，不将人工时间抖动的成功推广为自然叙述鲁棒性。

QVHighlights官方当前已写明Codalab关闭、test真值发布；这只是评价方式变化，不允许训练用test。MR mAP和HD mAP不是同一个指标，不能用更容易的saliency峰值替代边界定位。

主要风险是未标注片段被误认为负、长视频训练靠位置先验、特征预训练/提取器不匹配，以及又退回只看最高帧分数。需要完整任务头和独立的时间支持校准来避免这些问题。

## 本方向主来源

- [V01] [End-to-End Learning of Visual Representations from Uncurated Instructional Videos (MIL-NCE)](https://arxiv.org/abs/1912.06430)。代码/项目：https://www.di.ens.fr/willow/research/mil-nce/
- [V02] [QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries](https://arxiv.org/abs/2107.09609)。代码/项目：https://github.com/jayleicn/moment_detr
- [V03] [Query-Dependent Video Representation for Moment Retrieval and Highlight Detection (QD-DETR)](https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Query-Dependent_Video_Representation_for_Moment_Retrieval_and_Highlight_Detection_CVPR_2023_paper.html)。代码/项目：https://github.com/wjun0830/QD-DETR
- [V04] [Knowing Where to Focus: Event-aware Transformer for Video Grounding (EaTR)](https://openaccess.thecvf.com/content/ICCV2023/html/Jang_Knowing_Where_to_Focus_Event-aware_Transformer_for_Video_Grounding_ICCV_2023_paper.html)。代码/项目：https://github.com/jinhyunj/EaTR
- [V05] [Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding (CG-DETR)](https://arxiv.org/abs/2311.08835)。代码/项目：https://github.com/wjun0830/CGDETR
- [V06] [Background-Aware Moment Detection for Video Moment Retrieval (BM-DETR)](https://openaccess.thecvf.com/content/WACV2025/html/Jung_Background-Aware_Moment_Detection_for_Video_Moment_Retrieval_WACV_2025_paper.html)。代码/项目：https://github.com/minjoong507/BM-DETR
- [V07] [Sim-DETR: Unlock DETR for Temporal Sentence Grounding](https://openaccess.thecvf.com/content/ICCV2025/html/Tang_Sim-DETR_Unlock_DETR_for_Temporal_Sentence_Grounding_ICCV_2025_paper.html)。本轮未确认可直接运行的官方实现。
- [V08] [Saliency-Guided DETR for Moment Retrieval and Highlight Detection (SG-DETR)](https://openaccess.thecvf.com/content/WACV2026/html/Gordeev_Saliency-Guided_DETR_for_Moment_Retrieval_and_Highlight_Detection_WACV_2026_paper.html)。本轮未确认可直接运行的官方实现。
- [V09] [Hierarchical Video-Moment Retrieval and Step-Captioning (HiREST)](https://openaccess.thecvf.com/content/CVPR2023/html/Zala_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_CVPR_2023_paper.html)。本轮未确认可直接运行的官方实现。
- [V10] [MomentSeeker: A Benchmark for Long-Video Moment Retrieval](https://proceedings.neurips.cc/paper_files/paper/2025/hash/281e0b9142763f2b6c944fedb8550ba9-Abstract-Datasets_and_Benchmarks_Track.html)。本轮未确认可直接运行的官方实现。
- [V11] [NaQ: Leveraging Narrations as Queries to Supervise Episodic Memory](https://arxiv.org/abs/2301.00746)。代码/项目：https://vision.cs.utexas.edu/projects/naq/
