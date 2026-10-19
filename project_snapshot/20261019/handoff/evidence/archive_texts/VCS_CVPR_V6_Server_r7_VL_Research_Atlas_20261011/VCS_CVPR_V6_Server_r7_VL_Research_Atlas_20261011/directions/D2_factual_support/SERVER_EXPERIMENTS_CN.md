# D2 服务器实验方案｜视觉事实支持与生成忠实性

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS07, DS08, DS09, DS10, DS11, DS12, DS18；全部定义见 `../../datasets/DATASET_ATLAS_CN.md`。

## A. 主任务：支持核验→完整回答改进

第一轮以图像I、问题q、声明a为输入，预测supported/contradicted；unknown单独保留。输出不仅是一个分数，还要在固定候选生成或答案修正流程中改变最终回答。先做caption/VQA，不同时做T2I扩散微调。

## B. 分布与模型

在明确的人类标注二元子任务上令P=P(I,q,a|supported)，Q=P(I,q,a|contradicted)，类先验训练平衡。对应T=tanh f；Logistic用sigmoid(2f)，两者在同模型上训练。部署支持概率按CAL先验/校准报告；neutral不能强行作为Q。

建立文本分支a_text(q,a)与视觉分支b_vis(I,q,a)。比较同一任务实验的最优平方增量，但实际使用时只将有限(b−a)²称作分歧代理；独立标签下同时计算真实风险改善。图像遮挡/替换仅作压力测试，不能未经语义复核认定它使支持标签翻转。

critic建议复用统一cross-modal宿主的joint representations，加同大小预测头或LoRA；二元子任务不使用任意自由生成概率当校准分。所有训练臂同参数预算、输入分辨率和训练标签。选定模型必须能输出完整VQA/caption；纯分类probe结果只算组件检查。

## C. 数据与构造

M-HalDetect按image和回答生成模型双重记录；第一轮使用官方train并从中划CAL，留出原图家族。可另设“某一回答生成器从未用于训练”的迁移评价。原标签若是span而非原子声明，按官方annotation单位训练，不随意自动拆分改变真值。

自建原子声明或图像互换需要人工复核的最小子集和一致性记录；在没有授权标注资源时不把它写为已完成数据。SNLI-VE可以补三类训练，迁移到二元supported/contradicted时明示过滤，neutral单独评价。POPE/AMBER/HallusionBench与旧RefCOCO的COCO交集先记录，必要时按未见图片子集及官方全表双报。

## D. 具体实验

| 单元 | 方法/操作 | 产物 |
|---|---|---|
| D2-00 | 标签schema审查、官方metrics fixture；统计label/source/length/image分布 | 明确known false、unknown和文本捷径 |
| D2-10 | 同架构VCS、Logistic、纯文本对照各3seeds；支持识别与校准 | Brier/NLL、ROC/PR、跨generator与数据集结果 |
| D2-20 | 固定同一caption/VQA生成器和候选缓存，每问题N={1,4}；VCS/JS/自置信/VQAScore重排 | 同候选/同生成成本下最终答案和coverage |
| D2-30 | 胜出机制的VCS与matchedJS/直接错误预测进入同一LoRA或偏好训练recipe | 3方法×3seeds完整回答模型，不仅奖励头分数 |
| D2-40 | 独立AMBER、POPE、HallusionBench与人工小审计；需要时另一个生成器 | 外部评价与失败类型，不由训练critic主评 |

D2-20的best-of-N是独立用途，但正文宣称训练改善时必须完成D2-30。完整生成器样本缓存冻结后不同ranker共用，不能VCS采四次JS采一次。标准SFT/原生成器是不可省基线；VCD按其原图/扰动图额外forward计成本，Woodpecker/SCI能复现则做相同test-time budget参考。

## E. 评价

组件主指标binary Brier/NLL与类别/来源分层；AUPRC随先验变化显式记录。最终主指标在固定回答覆盖/长度下的错误率与官方任务质量。AMBER的CHAIR必须与Cover一起看；POPE三个负例策略分别报；HallusionBench三个accuracy不合并掩盖pair失败。无法解析回答、拒答和未知输出都按冻结规则保留。

生成改进需至少一条不依赖被训练critic的证据：原始人类groundtruth、盲审或固定独立评价器。报告new-error/corrected-error/unchanged-correct以及内容遗漏，避免reward hacking。

## F. 期待与退出条件

H2a：视觉分支比纯文本更能泛化到未见回答风格；H2b：相同标注与模型下二次证据读数能支持更好的risk–coverage或纠错；H2c：这转化为完整回答改善。没有预测固定领先点数。

若纯文本和视觉相同，数据未迫使模型看图，先修任务/负例，不增加网络。若组件Brier好而生成不好，检查reward–decision接口，不以组件分数代替任务。若总拒答导致分数提升，按coverage匹配重判。若matchedJS同样好，作用可能是共同后验估计而非平方拟合；方法贡献要落在任务结构/估计使用，而不是撤掉JS。

T2I分支只有在支持预测跨域可靠后再启用：固定GenAI-Rank候选，先评价排序与事实覆盖；扩散LoRA需新协议、独立外部reward和安全/版权审查，不在本包自动启动。
