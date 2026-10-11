# 最近相关工作的实验协议卡

本表只写本轮原始来源明确支持的内容，供服务器建立复现清单。没有核实的训练数量、版本和超参数不补猜。更完整的论文摘要、数据与脚本链接见`literature/papers.json`、各方向综述和数据图谱。原论文结果不能直接当成我们在新宿主上的baseline结果。

## D1：剪枝

**T02 SparseVLM（ICML2025）。**研究文本相关token驱动的视觉稀疏化；主结果是完整VLM问答/理解与效率，而不是每token标签预测。移植时要对齐所用宿主、剪枝发生位置、token保留层次与推理backend。数据/评价列表以其官方实现和补充为完整清单；本包选择GQA/TextVQA/ChartQA/DocVQA是新方案，不表示每篇baseline原文都跑了相同四项。

**T05 IF-Prune（CVPR2026）。**本轮核对作者页面、CVF标题摘要及官方仓库；仓库提供Qwen2-VL和InternVL训练/测试入口、ChartQA示例和lmms-eval集成。不同宿主使用不同transformers版本。全文不可访问的实验细节须服务器补读；不能因为repo里有一个脚本就断言某个全部benchmark已在原文验证。其VIB与小模型指导是D1必须逐项区分的先行结构。

**T06 DiffPrune（2026预印本）。**原文列LLaVA-1.5、LLaVA-NEXT与Qwen2.5-VL，并在GQA、MMBench/中文、MME、POPE、ScienceQA-IMG、VQAv2、TextVQA、SEED、VizWiz等任务评价。其训练弱化信息、部署hardtopK；所以soft训练到hard部署与额外模块成本都需单列。不能将其论文中的LLM-prefill倍数改称端到端倍数。

**T07 ET-Prune（2026预印本）。**研究文本密集视觉中的动态证据预算。它与“我们第一次按证据确定budget”直接重合；具体训练/测试集以全文实验与released-code revision为准，不能把RelatedWork里列举的DocVQA自动算成已完成主实验。它应先完成方法级比较，再决定D1何处有新增空间。

## D2：事实与幻觉

**H01 VCD。**原文采用LLaVA-1.5、InstructBLIP、Qwen-VL；POPE表包含COCO、A-OKVQA、GQA来源及random/popular/adversarial设置；另评价MME与LLaVA-Bench。无需训练但增加原图/扰动图两路计算。新宿主比较要同步temperature、decode策略和最大输出长度，不能拿其原文分数填我方Qwen2.5-VL表。

**H02 M-HalDetect。**人类细粒度标注用于检测/奖励与FDPO；原文best-of-n测试包含n=16、64，并研究InstructBLIP奖励对LLaVA/mPLUG-OWL的泛化。我们的首轮n=1/4是节约资源的新协议，不能说复现原64采样结果；每个baseline共享候选回答才能隔离重排序器。原文对analysis类的处理本身有消融，服务器须保留标签语义。

**H06/H07 HallusionBench、AMBER。**前者保留question/figure/pair家族聚合，后者分生成与判别两个任务。它们不是训练错误标签的默认来源；AMBER生成coverage与幻觉同报，判别P/R/F1的正类采用原定义。原图、编辑图及配对问题必须同组留出。

## D3：生成分布

**G01 CMMD。**原文方法是固定CLIP表示上的MMD；它既不是conditionalMMD，也不是以每张图语义真假训练的分类器。比较时所有分布统计使用相同样本数/featurepreprocess，单独报告自己学习critic所需的FIT样本。

**G03 cFreD。**正文及附录使用PartiPrompts、HPDv2与COCO提示，对多种生成器和图文特征组合评价，并检查人类判断关联。D3不能只对比FID和CLIPScore而省略它。原文的feature选择研究不表示我们可以在最终人评集选择最有利的feature；我方选择须留出。

**G04-G08 语义/偏好基准。**GenEval、CompBench原版与++、ConceptMix解决不同组合要求；BaiqiL的GenAI-Bench/Rank有发布版本差异，不能混用同名GenAI-Arena。它们评价的是语义或偏好，不提供自然条件分布的S真值。条件频率、每prompt生成数和judge版本在比较前固定。

## D4：多对多关系

**M01/M02/M03 PCME、PCME++、ProLIP。**它们已经研究概率嵌入与不确定关系。任务控制不仅换loss，还要给Logistic/VCS一致的概率头、采样次数和标签图。将官方大规模预训练checkpoint作为外部参考，与我们小样本完整微调的控制分表，避免归因错误。

**M06 ECCV Caption。**官方工具统一评价COCO1k、5k、CxC、ECCVRecall、R-Precision和mAP@R。额外标注覆盖Karpathy测试池的1,332个imagequery和1,261个captionquery子集，而不是每个query都有全部潜在真对应。mAP@R必须调用其发布的query列表。官方README示例API在数据图谱中给出。

**M07 FLAIR。**文本条件化池化是方法的一部分。只读取一个全局vector再算cosine属于削弱版，不代表论文的局部/细粒度能力；DCI/DOCCI扩展若采用它，先锁其原生score路径。

## D5：数据选择

**D01 DataComp。**原文固定四档pool与训练预算，通过从所选数据训练CLIP，再评价38个分类/检索任务判断数据质量。小档pool是12.8M，表中的samples-seen与选择后unique样本数是不同列。我们缩小成有限本地池时必须称customscale，不能套small标签。

**D02/D03/D04 DFN、Sieve、RecapDataComp。**分别引入过滤网络、caption模型或重新生成描述。filter/captioner的监督、训练来源与推理开销是系统的一部分。仅比较CLIPcosine而跳过这些已有信号，会让VCS的额外model调用看起来像损失优势。

**D05 SDM（ICLR2026）。**官方出版页明确目标对齐数据过滤及稀疏特征匹配；它是“选择与目标分布更接近的数据”的直接先行。服务器拟走D5时必须补读全文和实现，并给出同目标可执行对照；摘要核对不等于已经复现。

## D6：时间支持

**V02 Moment-DETR / QVHighlights。**官方仓库提供约8GB的CLIP+SlowFast缓存特征、200epochs训练入口与独立evaluator。这个时间只属于原作者的硬件记录，不作为我们的预算。注释是CCBY-NC-SA4.0，代码MIT；原始视频另有来源约束。先运行golden predictionfixture。

**V03/V05/V06 QD-DETR、CG-DETR、BM-DETR。**任务中已经存在query-dependent、相关性校准和background建模。新增VCS支持模块必须在共同feature与模型上与这些机制比较；不以额外ASR/更强feature来解释loss贡献。

**V07/V08 Sim-DETR、SG-DETR。**作为较新的强定位候选，服务器需补读原实验配置和数据预训练清单，至少选择一项与拟定任务真正可比的baseline。不可将只读摘要的SOTA数字直接写入共同协议表。

**官方服务状态。**QVHighlights standalone_eval README已说明Codalab停止、test标注发布；文档下方旧提交说明仍在，但不应建议使用已经停止的服务。公开test继续只作冻结后的最后评价，不能用于阈值/epoch选择。

## 接入完成的定义

每项最近方法只有在`baseline_reproduction.csv`具备代码revision、数据版本、宿主、preprocessing、evaluator和实际输出后，才标为“本地复现完成”。论文页/项目页存在只代表可查来源，不代表现有server已装好依赖或权重。
