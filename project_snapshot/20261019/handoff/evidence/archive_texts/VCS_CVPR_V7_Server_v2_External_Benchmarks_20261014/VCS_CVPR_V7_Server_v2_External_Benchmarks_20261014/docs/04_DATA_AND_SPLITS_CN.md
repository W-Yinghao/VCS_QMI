# 数据接入、标签与划分

## 1. 数据地图
| 数据 | 官方来源与本轮角色 | 必須区分 |
|---|---|---|
| EvalMuse | 人工细粒度元素＋整体一致性；公开train训练和内部留出 | 40K是论文完整集合，不是本次确认已公开的训练量；官方test标签访问仍待核对 |
| GenAI-Bench | 外部整体一致性评价 | 1600提示的Bench与800提示Rank有重叠，不是两个独立选模集合 |
| GenAI-Rank | 主应用；DALLE_3 / SDXL_Base各9图/提示 | 两个生成器分列，不能合成18图后仍叫官方9图 |
| TIFA160 | 有人工评分的外部小样本对齐 | 不等于TIFA完整prompt/问题集合 |
| RichHF-18K | 外部对齐评分；有合法标注时做细粒度分析 | 对齐、审美、伪影是不同维度，不混成一个“正确率” |
| NegBench图像子任务 | 否定/反例诊断 | 固定具体子任务和官方规则，不选择我们表现好的语料 |
| Winoground | 固定组合关系诊断 | 官方text/image/group都报告，不当成可训练数据 |

来源与核验深度见[D01,D02,P04,P06,P11,P12]。除EvalMuse、GenAI两份卡片外，本轮未逐一materialize所有数据，不应把本表当成已下载完成。

## 2. EvalMuse原始字段
官方卡片当前可见train_list.json、test.json、prompt_t2i.json说明及分片images.zip。自动viewer因不同元素字典字段的转换发生错误，因此优先下载原始JSON与图片档案，不依赖自动Parquet是否成功。[D01]

至少保存：`prompt_id`, `prompt`, `img_path`, `type`, `total_score[]`, `element_score{element: votes[]}`, `promt_meaningless[]`, `split_confidence[]`, `attribute_confidence[]`, `fidelity_label[]`。字段名按实际源文件，不擅自修拼写破坏溯源。

官方卡片说明split_confidence=1表示分解可能有错，attribute_confidence=0表示标注者对属性判断不确定。它们不是同一极性。代码先平均这些字段并不能替代语义核对。建立字段字典和至少20条人工检查记录，覆盖0/1、缺失、多个投票、否定、重复词、长prompt。[D01,G06]

### 标签决策
- 有效二元投票只接受已确认的0/1编码。其他码值先保留为unknown，不能映射为0。
- 元素概率目标y取有效投票的均值；n_valid、n_positive和原始数组都保存。
- 无意义prompt不加入主要元素概率任务；完整应用如官方集合含该提示，按官方规则保留或分层报告，不能按某模型得分过滤。
- 分解被多数标注者明确判错的事件不作为“已定义真值”训练；保留排除计数与第二敏感性表。阈值/规则须在首次模型评价前锁定。
- 属性判断不确定仅能按照数据提供方字段作用范围处理；若该字段只在图片级，不能假造元素级不确定标签。
- 总体评分1–5与元素0–1分开。源码第0个token的总体分数不是新元素，不纳入事件风险。[G04]

## 3. 提示与图像的独立单位
先建立每个prompt_id的图片列表和生成器。使用规范化文本hash检查精确重复，图像hash检查重复/转码；语义近重复按预先定义的文本距离和人工盲审归组，不看任何模型表现。只依据输入内容构建连通组件。

外部评测提示名单可用于输入排重，但标签不得进入训练。将与终末GenAI-Bench/Rank、TIFA160、RichHF明显重复的训练组件排除或明确另列暴露集。原始prompt_id相同只是第一步；不同ID的同文改写也可能重复。

主划分提案：公开EvalMuse TRAIN的提示组件按固定seed=17071划为80% FIT、10% CAL、10% DEV。CAL再按组件分成CAL-SELECT与CAL-PROB各半。数量按实际组件分配，报告图像数、prompt数、元素数，不承诺恰好整比例。所有12个主训练单元共享划分。

**全量训练=全部FIT的有效数据，不等于拿走独立评价数据来凑“全部40K”。**主要结果完成后若再用FIT+CAL重训，需新方案并保证终末未被拿来选择配置。

公开的EvalMuse完整test若可获得，作为额外锁定测试；否则本地DEV写作“EvalMuse public-train prompt-disjoint holdout”，不与原论文正式test数值并排排名。

## 4. 发布权重的暴露
FGA已发布权重可能使用所有公开train。其对本地80/10/10留出的评价是暴露参照，不是独立训练验证。主要N0原生配方应从同一个已核对BLIP2初始权重重新训练，只使用FIT；VCS/LOG/L1也从同一个初始权重开始。

BLIP2模型配置load_pretrained与load_finetuned同时为True，必须展开有效URL、最终加载次序、state_dict覆盖范围。不能误从FGA成品权重初始化，再把“见过DEV”的信息当作新泛化。[G03]

外部模型预训练无法排除所有互联网暴露，分别写已知训练集、已知重叠、未知，不宣称绝对clean。

## 5. GenAI资源角色
Bench与Rank全都视作终末资源：不先在Bench挑聚合方式、再把重叠提示的Rank说成独立测试。允许基于不含人工分数的输入结构生成固定解析缓存、检查文件完整性；不得按候选可见质量手动修解析。

主GenAI-Rank：2个generator × 800个提示 × 9张，具体完整数以下载清单为准。官方脚本按`DALLE_3`、`SDXL_Base`区分。[G07] 原始候选顺序必须显式存`candidate_index`，不能依赖glob排序、文件系统顺序或者reshape假设。

## 6. 失败与缺失
数据下载缺图：评测前统一共同候选池，先说明缺失，再固定完整性要求；不得某方法失败就只删对它不利的样本。模型推理OOM/NaN：按固定策略重试；仍失败则整行报告不可用及覆盖率，不以少数成功样本的高分替代完整benchmark。

解析失败不是推理失败：按方法定义统一使用O fallback，并同时报告失败比例。GT元素读数和自动解析读数分表。原生FGA仍按官方输出，无需强迫使用我们的解析器。

## 7. 必须交付的manifest
`image_id,prompt_group,prompt_id,generator_id,source_dataset,role,image_sha256,prompt_sha256,raw_label_path,label_status,element_count,parser_version`。
另有元素manifest包含`element_id,raw_name,type,spans,expected_polarity,votes,y,n_valid,scope`。与JSONL预测通过显式ID join，拒绝重复、丢失、错位。


## 8. v2接续新数据入口
`data_fetch.py@aeb56b5…`已经登记 `DY-Evalab/EvalMuse` 和更精确的 `zhiqiulin/GenAI-Image-Ranking-800`，后者的human_rating.json单独封存。数据卡的HF split名称train不是本研究训练授权；GenAI-Rank仍是终末人工选图集。[G09,X14]

X21复用与完整训练完全相同的提示组件角色；提取共同特征不改变划分。DV归一化常数与概率校准不得用外部测试。发布FGA经过任务训练的特征不作为“所有估计器共同的未适配表示”；若单独分析须标明其训练暴露。
