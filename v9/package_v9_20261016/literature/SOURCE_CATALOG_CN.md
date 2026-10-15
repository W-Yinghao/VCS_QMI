# 来源目录

每项的核验深度限制其可支持的主张；未核实的运行细节交server补齐。

### R01 · P157 SEP-2：受限 critic 在污染后后验上的比较
- 来源：https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/P157_SEP2_REPORT_20261015.md
- 类型：repository；核验：本轮上下文含报告；此前已读取
- 范围：合成数据，评价污染后后验；不是干净恢复或真实学习结论。

### R02 · P159 SEP-3：冻结自然表示上的比较
- 来源：https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/P159_SEP3_NATURAL_REPORT_20261015.md
- 类型：repository；核验：本轮重新读取全文
- 范围：FIT-ONLY；32/32污染条件公共J较高；cos-2排序相同；quad-3有学习率边界。

### R03 · REG-P3：EPFL RGB–NIR 配准
- 来源：https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/R6/R6_REG_P3_REPORT_20261012.md
- 类型：repository；核验：上下文报告完整
- 范围：共享优化器、2D刚体；不是原生CoMIR系统；许可仍需确认。

### R04 · V8 最新执行提交
- 来源：https://github.com/W-Yinghao/VCS_QMI/commit/a59c8333e3f272829afd93dcee72d162f98f4285
- 类型：repository；核验：本轮读取提交diff
- 范围：v8_eval聚合修正；提交写6/24单元在场，不能当实时进度。

### R05 · 合作者原计划及10月10日综合意见附录4
- 来源：local-source://conversation/Variational-CS-QMI-and-supplement
- 类型：user-source；核验：当前对话给出完整文本
- 范围：保持S/J；旧正文冲突以附录4为准；本包只转述所引用部分。

### L01 · NCR: Learning with Noisy Correspondence for Cross-modal Matching, NeurIPS 2021
- 来源：https://papers.neurips.cc/paper_files/paper/2021/hash/f5e62af885293cf4d511ceef31e61c80-Abstract.html
- 类型：paper；核验：官方摘要+代码README
- 范围：记忆效应、co-teaching、对应关系修正；原数据与训练协议参照。

### L02 · NCR 官方代码
- 来源：https://github.com/XLearning-SCU/2021-NeurIPS-NCR
- 类型：code；核验：检索到官方README
- 范围：COCO testall/fold5需区分；README与许可证展示有差异，以实际LICENSE核对。

### L03 · L2RM: Learning to Rematch Mismatched Pairs for Robust Cross-Modal Retrieval, CVPR 2024
- 来源：https://openaccess.thecvf.com/content/CVPR2024/html/Han_Learning_to_Rematch_Mismatched_Pairs_for_Robust_Cross-Modal_Retrieval_CVPR_2024_paper.html
- 类型：paper；核验：CVF摘要+arXiv入口+官方README
- 范围：OT/partial OT重匹配；不是仅删除低分对。

### L04 · L2RM 官方代码与数据入口
- 来源：https://github.com/hhc1997/L2RM
- 类型：code；核验：本轮GitHub读取README
- 范围：README明确NCR特征数据与0.2/0.4/0.6/0.8索引；CC152脚本命名须检查目录。

### L05 · ESC: Robust Noisy Correspondence Learning with Equivariant Similarity Consistency, CVPR 2024
- 来源：https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Robust_Noisy_Correspondence_Learning_with_Equivariant_Similarity_Consistency_CVPR_2024_paper.html
- 类型：paper；核验：CVF搜索返回摘要；直接open一度403
- 范围：语义变化一致性；本轮没有核实官方可执行代码，服务器需核实/按论文复现。

### L06 · ASL: Noisy Correspondence Rectification via Asymmetric Similarity Learning, AAAI 2025
- 来源：https://ojs.aaai.org/index.php/AAAI/article/view/35439
- 类型：paper；核验：出版方摘要与元数据
- 范围：正式全称是Asymmetric Similarity Learning，不是Asymmetric Self-paced Learning。

### L07 · ReCon, CVPR 2025
- 来源：https://arxiv.org/abs/2502.19962
- 类型：paper；核验：arXiv摘要+作者机构正式出版记录
- 范围：关系一致性；作者代码README仍说coming soon，不能算已有可运行基线。

### L08 · ReCon 官方仓库
- 来源：https://github.com/qxzha/ReCon
- 类型：code；核验：本轮GitHub读取README；blob 89e0140b65ec555d706424fda42562e2744ac6c8
- 范围：本轮读取到的README称代码即将发布；不是已完成复现。

### L09 · RRSITR: Robust Remote Sensing Image-Text Retrieval with Noisy Correspondence, CVPR 2026
- 来源：https://openaccess.thecvf.com/content/CVPR2026/html/Song_Robust_Remote_Sensing_Image-Text_Retrieval_with_Noisy_Correspondence_CVPR_2026_paper.html
- 类型：paper；核验：CVF摘要
- 范围：遥感域噪声分组和可靠度加权；不因换遥感而获得新颖性。

### L10 · BBSE: Detecting and Correcting for Label Shift with Black Box Predictors, ICML 2018
- 来源：https://proceedings.mlr.press/v80/lipton18a.html
- 类型：paper；核验：PMLR摘要与方法定位
- 范围：先验变化、类条件不变；混淆矩阵可逆；校准不是一致性的必要条件。

### L11 · Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation, ICML 2020
- 来源：https://proceedings.mlr.press/v119/alexandari20a.html
- 类型：paper；核验：PMLR摘要与实现入口
- 范围：校准+MLE/EM必须作为强先验估计基线；不是仅比未校准softmax。

### L12 · Mixture Proportion Estimation via Kernel Embeddings of Distributions, ICML 2016
- 来源：https://proceedings.mlr.press/v48/ramaswamy16.html
- 类型：paper；核验：PMLR摘要
- 范围：通常给混合样本+一个成分；与双参考P/Q信息条件不同，需分表。

### L13 · Relative Density-Ratio Estimation for Robust Distribution Comparison, NeurIPS 2011
- 来源：https://arxiv.org/abs/1106.4729
- 类型：paper；核验：此前项目定位；本包保留原始论文入口，服务器补全文
- 范围：RuLSIF alpha=1/2与当前T的变换；主要传统基线。

### L14 · Relative Predictive Coding, ICLR 2021
- 来源：https://arxiv.org/abs/2103.11275
- 类型：paper；核验：当前提供材料附录4明确对应
- 范围：平衡RPC与基础J精确重合；不重复造独立算法行。

### L15 · CoMIR: Contrastive Multimodal Image Representation for Registration, NeurIPS 2020
- 来源：https://arxiv.org/abs/2006.06325
- 类型：paper；核验：本轮arXiv官方摘要
- 范围：InfoNCE、两模态稠密表示、RGB-NIR及BF-SHG。

### L16 · CoMIR 官方代码
- 来源：https://github.com/MIDA-group/CoMIR
- 类型：code；核验：前轮已读取；本轮不宣称重新核对运行代码
- 范围：原生配准流程和特征+共享优化器两表分开。

### L17 · MIND: Modality Independent Neighbourhood Descriptor, Medical Image Analysis 2012
- 来源：https://doi.org/10.1016/j.media.2012.05.008
- 类型：paper；核验：作者机构库+PubMed元数据
- 范围：基于局部自相似的模态独立描述子；2D适配须记录。

### L18 · Information Theoretic Shape Matching, TPAMI 2014
- 来源：https://pubmed.ncbi.nlm.nih.gov/26353150/
- 类型：paper；核验：本轮PubMed摘要
- 范围：包含CS点集分布匹配；不能冒称现成的稠密BF-SHG算法。

### L19 · GRES / gRefCOCO, CVPR 2023
- 来源：https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.html
- 类型：paper；核验：上轮分析所用原文；本轮仅保留备选入口
- 范围：无目标与多目标；第三方向不自动训练。

### L20 · SGRAF: Similarity Reasoning and Filtration for Image-Text Matching, AAAI 2021
- 来源：https://github.com/Paranioar/SGRAF
- 类型：code；核验：NCR/L2RM官方README共同上游
- 范围：服务器必须锁SGR单模型结构，不能把SGR与SGRAF双模型分数混同。

### L21 · CS-Aligner: distribution-level vision-language alignment
- 来源：https://arxiv.org/abs/2502.17028
- 类型：paper；核验：前轮资料来源；本轮未重读全文
- 范围：相关工作/条件适配，不是现成噪声后验评分器。
