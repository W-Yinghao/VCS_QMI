# 检索方法、覆盖与待核验项

检索日期：2026-10-11。问题导向的选择性综述，不宣称穷尽2024–2026全部VL论文。围绕六组关键词查原始会议页、arXiv、作者项目/代码和数据/evaluator。先查最新直接邻居，再追溯基础工作；不按“是否支持VCS”筛文献。

## 本轮检索范围

| 方向 | 核心检索词 | 重点出处 |
|---|---|---|
| D1 | visual token pruning; information bottleneck; evidence-aware; dynamic budget; DiffPrune | CVPR/ICCV/PMLR与arXiv/official repos |
| D2 | visual factual support; hallucination; contrastive decoding; visual entailment; VQAScore | CVF/ACL/AAAI/arXiv，官方hallucination evaluator |
| D3 | conditional generative evaluation; cFreD/CFID; CMMD; text-image faithfulness/coverage | CVF/PMLR/arXiv，生成评估项目 |
| D4 | probabilistic embeddings; missing positives; PCME/ProLIP; CxC/ECCV Caption | CVF/ICLR/ACL/official repos |
| D5 | DataComp; data filtering network; recaption; submodular distribution matching | DataComp官方、ICLR/PMLR/CVF |
| D6 | temporal support; narrated video; moment retrieval; QVHighlights | CVF/NeurIPS/作者代码与官方metrics |

## 必须影响设计的近期先例

T05 IF-Prune已经是信息瓶颈引导剪枝，T06 DiffPrune是任务驱动可微信息节流，T07 ET-Prune是证据感知动态预算。D1不能以这些词本身宣称首次。T06/T07按2026预印本处理，不虚构会议录用或可运行代码。

H04 SCI已有视觉/文本反事实迭代；G03 cFreD已有条件生成质量评价；D05 SDM已有目标分布过滤。它们需要实质对照，不应被从“最接近工作”表中省略。

M04 LongProLIP在arXiv备注中是ICLR2025不确定性workshop tiny paper，不是ICLR主会。D02 DFN确认是ICLR2024，D04再描述确认是ICML2025。

## 核验深度与限制

`papers.csv`逐项注明摘要、全文或官方README。能核对方法和任务，不等于已下载模型、重建训练或核实每张结果表。若只读到摘要，就不臆造全部数据集/训练预算/超参数；服务器须在进入该基线时补全文和evaluator。

IF-Prune的CVF全文本轮访问受限；论文摘要、作者页及官方仓库已核，README明确ChartQA/Qwen/InternVL接口，但“八项评测”完整清单尚需补。SCI、Sim-DETR、SG-DETR、MomentSeeker的全文/部署细节未逐项复现。DiffPrune/ET-Prune是可读预印本，代码可用性未保证。当前部分源码链接来自论文公开入口，尚未安装；不存在的/迁移的链接按原作者路线恢复，不猜镜像。

本包没有将发表论文的SOTA数值复制成服务器预期结果。只有相同host、split、输入和evaluator时才能建立reproduction anchor；其余先以原方法定义作为对照，实际跑出本地结果。

## 数据与评估的版本陷阱

- SP-DocVQA与DocVQA2026不同。
- T2I-CompBench原版6000/6类与++8000/8类不同。
- BaiqiL/GenAI-Bench与GenAI Arena命名不同团队；固定项目身份。
- HallusionBench full版与镜像image-only样本数不同。
- QVHighlights官方当前说明Codalab已关闭并公开test真值；保持最终评价封存，不向旧服务误提交。
- PCME的CUB类相关性不是声明真值；CxC分级相似度不是概率；ECCV Caption标注域不完整。

## 论文引用

本包BibTeX只包含已核对的最小metadata，不猜作者拼写、页码和DOI；成稿从出版社BibTeX补齐。第三方论文全文和代码未打包分发，避免无必要复制与许可证问题。
