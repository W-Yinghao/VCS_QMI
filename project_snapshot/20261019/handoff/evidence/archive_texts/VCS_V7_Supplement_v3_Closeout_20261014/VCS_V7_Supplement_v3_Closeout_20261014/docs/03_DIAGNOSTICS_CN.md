# 补充分析：从元素估计到真正的选图效用

## 1. 先建立一张逐候选对齐表
每行包含 `prompt_group,prompt_id,image_id,generator_id,role,method,seed,E_gt,E_auto,O,H_frozen,parser_status,element_coverage,total_votes,element_vote_summary`。使用显式ID join；拒绝重复或缺失，不靠文件顺序reshape。旧模型不产生的字段填NA，而不是填0。

C01/C02只读取CAL/DEV；不读取GenAI评分。完整多模型的预测按原生输入与本模型parser重新导出即可。自动元素使用既有固定prompt-only parser，不用本轮结果修改。GT与auto的差异同时包含“元素清单改变”和“模型对新文本接口的泛化”，不能称为纯parser因果效应。

## 2. C01：人工元素参照与人工整体评分
对图像i定义 Hbar_i 为该图原始总体评分的均值。对主要有效元素集合A_i定义 Uann_i = mean_e(y_ie)。另报原生全元素、有效元素的样本与覆盖差异，不把bad decomposition当不满足。

在相同提示的候选组内比较 Uann、E_gt、E_auto、O、H_frozen 与Hbar：
- pooled SRCC/PLCC仅作背景；主分析是prompt-macro的within-prompt pairwise accuracy和Kendall tau-b；
- 先固定候选完整性，候选数>=2的提示都纳入；每个提示所有候选作为变长开发池；另报>=3和>=9的共同提示上的标准大小敏感性；
- 输出按各读数选中图像的Hbar、观测标签池内最大值与选择后悔值；多seed逐个选，不先集成；
- 选择Uann时使用了人工元素标签，所以这是**annotation-assisted诊断参照**，不是可部署基线，更不是噪声为零的性能上界；
- 元素和总体可能由相同标注者给出，且三/六次评分由重标注策略产生。相关性不能视为独立的语义真值验证。[P01,G04]

并列处理：与原主协议同样使用first-index列；敏感性列对全部最大分候选的人工分数求平均。禁止用人工H打破模型并列。人类相同评分的pair单列，不算谁排错。

### 如何阅读
Uann本身与整体排序不一致：聚合效用有错位；E_gt比Uann差很多：估计/视觉信息读取不足；E_auto比E_gt差很多：解析与输入覆盖有额外代价；O优于E：直接整体监督更贴近终点。各差值不是严格可加的误差分解，不画成总和恰好等于总体损失的瀑布图。

## 3. C02-a：同提示的尾部错误，不止平均Brier
固定一个提示、候选集和所有共同有效元素。记录：平均元素Brier、最高分图的元素Brier、模型第一名与第二名margin、第一名对Hbar最优候选的后悔值、最大正向预测误差（预测高而人工元素低）以及元素数。

按候选数N=1/3/9做开发池曲线，所有方法面对相同候选子集。N=3全组合只在开发诊断与既定敏感性里；不能替换官方三个固定分组。按FIT确定的元素数和prompt长度分位点分层；类别标签预先映射；不按哪类M2领先后定义“复杂”。

Brier低不保证best-of-N高：平均误差不控制有限池最坏候选，且E与H效用不同。报告两种分离，而不是把它们再总结成“缺少softmax”。

## 4. C02-b：解析可用性与身份检查
统计无问题/全部过滤、否定极性错误、未匹配跨度、重复词多跨度、关系绑定、空元素、过长文本、模型失败。空parser默认fallback仍按原V7规则，不改主分母。[G02]

X20协议记有“first64 tokens as pool_logits”，而输入上限128、FIT最大85。仅凭这句话不判bug：本地检查实际pool函数、同一element_id的可见子词和零覆盖情况，并报告覆盖。若原协议确为限制，保留；若实现与协议不同，先报告影响，不静默改变X21主行。[G05]

核方法无parser元素时，X21原协议写no score，而全模型V7写O fallback。两者不能被悄悄视为等价完整系统：主冻结结果照旧并报告覆盖；所有posterior方法在共同解析成功提示上的对比另列。新增“统一中性回退、按预定候选索引选图”的全分母敏感性，需要评分前登记且单独命名，不能把核方法失败提示删掉后写更强结果。

## 5. C02-c：标注不确定性与总体评分校验
只有原始离散votes可用于评分分布。三个评分快照不是三个独立图像。无rater_id时，不建立跨图标注者随机效应，也不假定第j个总体标签对应第j个元素标签。

可做“评分数组随机拆分”的噪声敏感性：在每图评分内按固定seed拆为两份，分析两份Hbar对候选排名的一致性。它是数组内重采样，不保证标注者独立、不提供真实oracle上限。3/6标注组分列，避免选择性重标注造成误读。不得通过筛掉分歧图使主结果更好。

## 6. 必交文件
`candidate_linkage.jsonl`、`within_prompt_metrics.csv`、`annotation_reference.csv`、`tail_error.csv`、`parser_coverage.csv`、`label_noise_sensitivity.csv`（无有效输入则说明NA）。每一张表列role、方法、seed、输入/模型hash、权重和有效prompt/component数。
