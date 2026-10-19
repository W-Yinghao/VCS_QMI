# 文献综述 · V9所需的三组外部比较

**核验日期：2026-10-16。** 本综述区分原论文/官方README、仓库实证与本包设计。完整入口和核验深度在sources.csv/json；不声称已复现或已逐式读完全部论文。没有将第三方论文全文打包。

## 1. 噪声图文对应：从判别到纠错/修复
| 工作 | 其方法与原任务 | V9应怎样比较 |
|---|---|---|
| NCR, NeurIPS 2021 [L01,L02] | 依据训练中的记忆现象分离干净/错误，再相互监督修正对应 | 原生checkpoint和完整流程复现；A表原协议，B表相同可信锚点 |
| L2RM, CVPR 2024 [L03,L04] | 用OT/partial OT寻找可用重匹配，不只下调错误对 | 保留其修复能力与额外计算；不削成弱过滤对照 |
| ESC, CVPR 2024 [L05](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Robust_Noisy_Correspondence_Learning_with_Equivariant_Similarity_Consistency_CVPR_2024_paper.html) | 比较两模态在anchor附近的语义变化一致性 | 关系性可靠度的外部对照；代码/实现待核，不编现成结果 |
| ASL, AAAI 2025 [L06](https://ojs.aaai.org/index.php/AAAI/article/view/35439) | 正负对非对称学习并对正对自步加权 | 与单纯概率加权直接相关；正式名称是Asymmetric Similarity Learning |
| ReCon, CVPR 2025 [L07,L08] | 模态内外关系一致性帮助区分真假对应 | 不能只对比2021方法；但本次官方README称代码即将发布，需实际复现计划 |
| RRSITR, CVPR 2026 [L09](https://openaccess.thecvf.com/content/CVPR2026/html/Song_Robust_Remote_Sensing_Image-Text_Retrieval_with_Noisy_Correspondence_CVPR_2026_paper.html) | 遥感图文中的清洁/模糊/噪声分组与可靠度调节 | 说明此领域不是空白；不强行混进普通COCO原生表 |

这些工作把任务性能作为终点。V9的区别候选是额外提供明确定义的批次量化和可校验概率接口，但这还需要实验证明有价值。它不是首次做可靠度加权或噪声检索。

## 2. 比例估计、密度比与校准
BBSE [L10](https://proceedings.mlr.press/v80/lipton18a.html) 研究先验改变、类条件不变；预测器即使不完美或未校准，只要对应矩阵可辨认也能估比例。因此rawscore/BBSE不是应当输给VCS的弱方法。

校准+MLE/EM [L11](https://proceedings.mlr.press/v119/alexandari20a.html) 是重要强对照。较准的概率不只影响分类，也影响混合估计；但其成立依赖先验/类条件模型。V9须给它相同参考与CAL，不只跑未校准EM。

Kernel MPE [L12](https://proceedings.mlr.press/v48/ramaswamy16.html) 通常观察混合与一个已知成分，需要额外可识别条件。双参考P/Q问题给了更多信息；分开表述，不能假装方法得到相同输入。

RuLSIF [L13](https://arxiv.org/abs/1106.4729) 使用相对密度比；alpha=1/2可映射到T。Nyström平方回归与KDE分别代表直接回归与显式密度拟合，前者不是“不会学习的核方法”。RPC [L14](https://arxiv.org/abs/2103.11275) 与基础J重合，不作为重复训练出来的新算法行。

**V9新增公式的归属：**固定score三均值比值属于矩匹配；后验权重属于Bayes/重要性加权；有限置信集使用标准Hoeffding并合与线性反演。它们是方法设计依据，不是本包宣称的新定理。

## 3. 跨模态配准
CoMIR [L15,L16] 以InfoNCE学习稠密共享表示，随后进行配准。此前项目主要比较feature+shared-optimizer，不等于原生流水线。

MIND [L17](https://doi.org/10.1016/j.media.2012.05.008) 用局部自相似描述跨模态结构，是非神经/非CS的重要路线。2D实现与原三维协议不同，具体offset/尺度/优化器需要说明。

Information Theoretic Shape Matching [L18](https://pubmed.ncbi.nlm.nih.gov/26353150/) 已经用CS把点集表示成分布后求变换。它的输入是点集，不是现成的BF-SHG稠密配准系统。V9应保留传统CS在适用输入上的直接比较，同时对学习型能量做共享优化器表。

## 4. 开放对应与其限制
GRES/gRefCOCO [L19](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.html) 提供零/多目标场景；能消费绝对概率，但也需要实例去重、联合事件和同等校准。单调转换若不改变排序，在固定覆盖率指标上不能创造优势。

## 5. 下一轮前必须补核的材料
完整NCR/L2RM/ESC/ASL/ReCon正文与supp中的：损失方程、epoch/batch、噪声索引、host/ensemble、所有额外监督。SGR公共接口与evaluation函数逐项对照。最近可运行的方法若加入，先固定协议再读V9结果；无法取得实现则保留缺失和复现路线，不从参考表删除。
