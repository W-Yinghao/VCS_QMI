# 指标约定与官方实现接入

所有“官方”最终以锁定代码为准；本文件是防止混淆的契约，不重新发明官方评分器。自定义辅助指标必须带`custom_`前缀。

| 任务 | 必报指标 | 关键约定 |
|---|---|---|
| GQA | balanced exact answer accuracy | 使用官方归一化；consistency等需对应程序/场景图。test-dev与val不同。 |
| TextVQA | 官方VQA soft accuracy | 10人参考的leave-one-out评分与EvalAIAnswerProcessor；不能用粗略全量count/3替换。 |
| SP-DocVQA | ANLS | normalized Levenshtein distance低于0.5才计1−distance，多个合法答案取官方max；空串/大小写/阈值边界按脚本。 |
| ChartQA | relaxed numeric accuracy; human/augmented分开 | 数值答案相对gold允许5%，非数值exact；零、百分号/单位边界用官方实现。 |
| MMBench/ScienceQA | versioned multiple-choice accuracy | 明确语言、circular evaluation、IMG子集、解析失败计错，不默删。 |
| POPE | accuracy/P/R/F1/yes ratio | random/popular/adversarial分表；正类约定和模型拒答处理pin脚本。 |
| HallusionBench | aAcc/fAcc/qAcc | question、figure、paired-question聚合均保留；不能用单个aAcc冒充三者平均；编辑家族重采样。 |
| AMBER生成 | CHAIR↓/Cover↑/Hal↓/Cog↓ | 原文定义对象提及与标注交集、coverage分母不同；CHAIR下降需同时保覆盖。 |
| AMBER判别 | all-question Acc + no-class P/R/F1 | 原文幻觉问题真值no作为P/R/F1的正类；AMBER分数用原式。 |
| COCO描述 | CHAIR_i/s + CIDEr/SPICE等任务指标 | 对象幻觉与关系/属性真实性分开；拒绝/短回答统计。 |
| 语义支持概率 | custom binary Brier; NLL; AUROC/AUPRC; reliability | 明确支持为正、负类真实含义、先验与校准集；ECE不是唯一指标。支持未知不计明确负类。 |
| 选择性预测 | custom risk–coverage / AURC | 置信排序和threshold在CAL冻结；错误风险分母为保留回答，拒答覆盖损失同时报告。 |
| 图文检索 | i2t/t2i Recall@1/5/10 | COCO1k五折与COCO5k分开；全部合法caption；Flickr不同候选池不混。 |
| ECCV Caption | mAP@R / R-Precision / Recall@K | R为该query发布的相关数；使用官方查询索引；未判定全集不转成额外真负例。 |
| CxC | graded semantic correlation + official expanded retrieval | 语义评级不是Bernoulli概率，不对0–5直接计算posterior MSE。 |
| SUGARCREPE++ | official triplet score | 两个正caption及一个hard negative保留；普通二选一不是相同指标。 |
| 条件分布 | common J / known finite S error / grouped MMD | 自定义指标标明观测空间、条件权重、fit/eval样本；自然S未知不写MSE。 |
| 生成质量 | FID/KID/CMMD/cFreD + precision/coverage | 同编码器、同样本数同预处理；KID/无偏MMD可能负，不将它裁零后只报低方差。 |
| 生成语义 | GenEval/T2I-CompBench(+)/ConceptMix | prompt版本、per-prompt图数、detector/VQA/LLM裁判全部pin。 |
| 人类偏好 | prompt-group Spearman/Kendall; pairwise accuracy | tie处理与样本/模型级聚合固定；不能混不同GenAI-Bench项目或释放版。 |
| DataComp | 官方38任务summary + 分项 | pool规模、architecture、examples-seen和配方匹配才称官方scale；内部toy另表。 |
| QVHighlights MR | mAP@[.50:.05:.95]; R@1@.5/.7 | 时间单位秒；全相关区间；官方max predictions/时长分桶；用其golden fixture。 |
| QVHighlights HD | mAP/Hit@1 | 2-sec clip saliency，官方“very good”等阈值配置固定；不自定义二值化后叫官方。 |
| Charades-STA | R@1/5@tIoU .3/.5/.7; mIoU辅助 | 明确特征、train/test分割、时间分辨率与是否弱监督。 |
| Ego4D NLQ | R@1/5@.3/.5 | 不与MQ/GoalStep指标混用，使用官方episodic-memory评估。 |

## 共同读数与归一化

平衡类下用y∈{0,1}、d=(T+1)/2时，signed square risk是binary Brier的4倍。报告必须写用哪种量；不能将数值直接混在同一列。

任务风险可保留模型每个样本的原始概率和标签，J则按真实P/Q单独加权。生成、压缩和数据选择中，一个全局J无法自动分解为各样本“价值”。推断用途需要另定义。

预算曲线至少同时给绝对任务分与相对full baseline保持率，避免用95%保持率误导为95%任务准确率。计算Pareto前固定成本范围；不同GPU/精度不合并成速度主张。

## 数值结果模板

`method_id,host_revision,dataset_revision,split_id,evaluator_revision,seed,metric_name,value,unit,n_groups,n_queries,fit_objective,readout_objective,cost_scope,status`。

本包无任何新任务结果。未运行行填状态，不填预测准确率。
