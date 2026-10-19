# V7 完整研究与服务器实施全稿

版本V7；交接包v1；2026-10-13。单文件聚合版，原文件路径标于各节。


---

原文件：`README_CN.md`

# V7 · 从图文证据估计到生成图像选择
## 服务器交接包 v1 / 2026-10-13

**研究目标：同样的一组生成图像、同样的训练标注和评价预算，用更准确的细粒度满足概率，选出更符合文字要求的图像。**

这是新的论文/研究版本 **V7**，不是旧 V6 的服务器轮次 r7。旧 R5/R6/R7、r8 审查及其已经授权的运行继续保留；本包所有新工作使用 `V7-*` 编号。

## 先读什么
1. `V7_OVERVIEW_CN.pdf`：七页执行与研究总览。
2. `01_SERVER_AGENT_PROMPT_CN.md`：接入指令。
3. `docs/02_SERVER_PLAN_CN.md`：完整阶段安排。
4. `docs/03_METHOD_THEORY_CRITIC_CN.md`、`docs/05_BENCHMARK_METRICS_CN.md`：必须先统一的数学对象与评价。
5. `literature/REVIEW_CN.md`：任务先行工作；`literature/CODE_PROTOCOL_AUDIT_CN.md`：实际代码与口头方案之间的差异。

`INDEX.html` 与 `V7_FULL_PLAN_CN.html` 可以离线阅读；公式保留LaTeX源式，不依赖网络脚本。资料包不包含论文全文、第三方模型权重、数据、字体或新的自然图像实验结果。

## 本轮不是重新包装一个旧负结果
当前优先应用是 **EvalMuse上的细粒度图文满足估计 + GenAI-Rank固定候选池选择**。不再将“视觉token压缩一定有效”作为前提；r8的实现审查仍按已有授权完成。[G01]

## 三个独立终点
| 终点 | 问题 | 主要证据 |
|---|---|---|
| 估计 | 对某项文字要求，满足概率是否可靠？ | 独立投票标签上的Brier/平衡平方风险 |
| 泛化 | 离开训练提示与图像分布是否仍有效？ | 外部总体相关性、同提示内排序 |
| 应用 | 同一候选池中最终选得是否更好？ | 人工评价下的best-of-9均值、后悔值 |

**不得用更高的模型自身J替代外部选图收益，也不把更低Brier自动写成更好的总体人工评分。**

## 已落实到任务的新增核对
- FGA公开活跃ITM分支使用L1，不是Logistic；原论文有variance权重，但本次锁定代码的活跃路径未乘它。[G02]
- FGA训练数据读取有32-token截断，元素重叠会映射到同一token；第0位混入总体分数。[G04,G05]
- 官方GenAI-Rank的N=3是固定三个连续分组，不是从九张图任意挑三张；本包保留官方口径，另报全84子集敏感性。[G07]
- EvalMuse公开卡片没有确认可用测试标签；本包不虚构官方测试，也不把公开训练的派生留出称为官方测试。[D01]

## 交付顺序
数据与代码核对 → 原生方法复现 → 12个完整选定训练单元 → 概率与外部选图评价 → 有条件的样本效率/新生成器确认。预算以实测为准，不能以GPU空闲为理由自动扩实验。


---

原文件：`01_SERVER_AGENT_PROMPT_CN.md`

# 发给服务器Agent：V7交接

请接入本包，启动 **V7细粒度图文一致性与固定候选池选图** 的实施。先读取 README、主任务书、方法定义、数据划分、指标和源码核对。不要将V7理解为旧R7 token-pruning的另一个变体。

基线快照：`W-Yinghao/VCS_QMI@3e2fe59c4861abae46537574c4c59f9bedfe91c6`。接入时自行读取最新HEAD、已有V7工作与实际队列；报告本包截点之后的变更。旧R5/R6/R7、D1-FULL、E1/E2及r8审查按原授权继续，不静默取消、不重复提交、不覆盖已经见过官方测试的结果。

本轮核心：
- 从EvalMuse原始标注建立有明确定义的元素满足事件；保留投票、缺失和不确定标签。
- 以FGA-BLIP2的实际多模态训练路径为基础，分别运行原生代码基线、共享语义元素接口的L1、Logistic与VCS；后面三者共享网络/输入/辅助项。原生FGA不是BCE，不能省略L1对照。
- 主应用用GenAI-Rank的DALLE_3和SDXL_Base固定九图候选，比较独立人工评分；主对手至少有发布的FGA-BLIP2与原生强VQAScore。最终不是只给自己一个J表。
- 先完整复现所有预定seed，再读冻结的外部主终点；不根据seed0是否领先决定其余seed。

现在可执行：数据/代码审查、CPU单元验证、复现环境与真实资源计时。完整训练的总单元为4配置×3seed=12；在锁定数据可用性、有效配置、训练模块、成本和主要评价后，按现有owner权限提交。需要额外付费API、人工标注、扩散模型训练或新的测试许可时先明确请求，不能默认授权。

请特别回复 `literature/CODE_PROTOCOL_AUDIT_CN.md` 中的6项源码事实：接受、修正或反驳均可，给实际代码和样本。不得为了对齐论文表而静默修改原生基线。推荐分支为 `v7/t2i-alignment`，目录 `reports/V7/`、`outputs/V7/`，保留完整commit与hash。

第一轮交付：
1. `V7_00_INTAKE.md`：新HEAD、复用资源、队列、授权、数据与权重版本。
2. `V7_01_DATA_AUDIT.md`：原始字段、标签码、提示与图像排重、实际划分数量、所有过滤原因。
3. `V7_02_BASELINE_REPRO.md`：原生FGA/VQAScore和指标复现，不能用外部测试挑设置。
4. `V7_03_PROTOCOL_FROZEN.yaml`：填实模板中的机器/版本/样本数，列训练单元与时间实测。
5. 成对的原始预测文件、报告模板和未完成事项。交付应优先围绕科学结果，而不是仅列作业编号。

本包附的参考代码只覆盖损失、先验、分组指标与数据完整性，不是可直接提交GPU的完整训练runner；请实现实际视觉数据adapter并运行现有包测试及你新增的集成测试。不要报告未实际运行的测试或自然数据结果。


---

原文件：`docs/02_SERVER_PLAN_CN.md`

# V7 主任务书：细粒度图文证据估计与best-of-N选择

## 1. 科学问题与明确的应用
给定同一个提示生成的若干图像，评分器先判断提示中的哪些元素被满足，再选择一张交付用户。我们检验一条具体链条：**独立标签上估得更准 → 同提示内判断更好 → 人工评价下选得更好**。三个箭头都需要证据。[F01,P01,P02]

应用不是新发明。TIFA/DSG/PN-VQA已有分解式核验，FGA已有训练型整体与元素评分，VQAScore及GenAI-Rank已有黑盒图像重排。本轮不把任何一项重新认领。[P01–P05]

## 2. V7预期优势与证据位置
| 编号 | 待检验主张 | 验证 | 不成立时如何解释 |
|---|---|---|---|
| H1 | 相同模型和数据下，二次拟合降低元素概率风险 | VCS vs Logistic的独立Brier，原始与校准分开 | 不宣称VCS更准；保留模型或任务共有收益 |
| H2 | 细粒度读数改善实际图像选择 | GenAI-Rank best-of-9，独立人工评分 | 更低风险不等于选图更好；分析聚合与尾部错误 |
| H3 | 有限标注/新生成器下收益更稳健 | 固定提示子集/预先留出生成器家族 | 不把局部正结果扩成普遍鲁棒性 |
| H4 | 相同选图质量时评分开销更少 | 完整系统的质量–时延曲线 | 没有质量相当就不能宣称等质量加速 |

不填写预期百分点。实用容许差距可由使用成本事先定义；它不是理论预测或CVPR录用门槛。

## 3. 任务分解
### V7-00：接入与协议核对
核对当前HEAD、旧授权、FGA版本、数据许可、GPU与已有权重。建立版本清单与暴露清单。明确已见过的外部结果；不将已读公开论文结果称为新独立确认，但本项目未看过的逐样本标签仍可按冻结规则评价。

### V7-01：数据建设
从原始JSON而不是失败的自动Arrow视图读取EvalMuse。保留原始标注、分解置信度、元素类别与生成器来源。按提示group划分；同图、改写提示、同源提示连成组件后只进入一个角色。跨数据集先排重再锁定训练集合。[D01,G04–G06]

### V7-02：复现与实现检查
原生FGA发布权重与从相同BLIP2初始权重重新训练的FGA分开。发布权重若见过本地派生留出，不能用该留出上的成绩作公平训练对照。VQAScore原论文环境和现代相同宿主环境分开。[G08]

先检查基本推理和评价脚本；不以在终末GenAI-Rank上调到漂亮数值作为复现成功。EvalMuse source validation名称为test并不授权把test接到训练选模中。

### V7-03：主要完整训练
4个配置×3seed=12个选定训练单元：N0原生FGA活跃代码、M0共享元素接口L1、M1匹配Logistic、M2匹配VCS。N0用于任务配方复现；M0说明共享输入/聚合改变的作用；M1与M2才是严格的后验损失控制。

此前口头方案的9单元增加到12，是因为源码确认了“原生L1 + 特殊token标签与池化”这两项差异；不额外增加新的大模型。完整训练的含义是使用全部规定FIT数据、训练官方路线允许训练的多模态模块，随后运行完整评分与选图；不是只在冻结向量上拟合探针，也不等于必须解冻官方明确冻结的视觉主干。

### V7-04：冻结预测后完成主评价
EvalMuse派生DEV（官方test仅在可获取且获准时另列）、GenAI-Bench、GenAI-Rank、TIFA160、RichHF对齐维度。主表不得按哪种方法领先删行。任何新使用测试标签的训练选择都会令后续结果变为开发结果。

### V7-05：有条件扩展
先看全数据结果，再决定固定10%/25%提示子集、预先留出生成器或者相同宿主的新backbone。选择扩展理由须是辨别研究解释，不是换数据集寻找赢的一格。完整低样本确认若批准，为2目标×2子集×3seed=12个追加单元。

### V7-06：可复现展示与论文整合
从冻结终末评价中按预定规则选成功、平局、失败示例，展示相同提示/九候选/各评分器选择/人工评分与元素清单。示例不参与方法选择。演示必须允许查看所有候选，不能仅展示我们最好的图。

## 4. 暂不做
不重启CLIP大规模预训练、RefCOCO连续修补或六条VL方向并行。第一轮不训练扩散生成器、不做强化学习、不默认申请商业API或新人工标注。既有V6结果继续作为历史证据，token选择审查按r8完成。[G01]

## 5. 算力和完成条件
不预写自然数据GPU小时。服务器先报告：下载/解压存储、单个完整训练epoch、评分器每图/每元素/九图批的时间、峰值显存、解析开销与数据加载开销。保持global batch、有效步骤、AMP、梯度累计一致；不能只匹配epoch而让一方法接触更多图像。

已选定协议下全部seed完成后汇总。崩溃按同配置恢复；无法恢复的失败要列入分母及资源记录，不静默替换seed。方法变更要产生新配置ID，不覆盖旧结果。


---

原文件：`docs/03_METHOD_THEORY_CRITIC_CN.md`

# 方法、理论与critic实现

## 1. 原始理论保留
设M=(P+Q)/2，η=(p−q)/(p+q)：

\[
S=E_M\eta^2,\quad J(T)=E_PT-E_QT-\tfrac12E_PT^2-\tfrac12E_QT^2,
\quad S-J(T)=E_M(T-\eta)^2.
\]

\[
D_{CS}^{(M)}=2\log\frac{1+S}{1-S},\qquad T=\tanh(f),\quad d_M=\sigma(2f).
\]

训练使用有界S/J，不为图像选择强行使用敏感的对数变换。经验J可能为负；读数不能逐次称为认证下界。理论对一般P/Q成立；本轮监督事件不是自动意义下的QMI。[F01]

平衡RPC和J的目标相同，Logistic也可以拟合同一后验并读同一J。新应用不会消除该先行工作。[F02,T01,T02]

## 2. 本轮事件是什么
观测z=(I,p,e)：图像I、完整提示p、一个有明确文字支持的语义元素e。C=+1表示随机抽取的有效人工标注者认为该图满足这个元素；C=−1表示不满足。P和Q分别是这两个条件分布。标签描述的是规定人工判断任务，不宣称已经知道客观语义真值。

原始数据若只有投票均值y，0<y<1表示标注分歧，不应自动阈值成0/1。缺失、无法判断、无意义提示、错误分解是不同状态。每个事件保留定义及预期极性：否定要求对应的是“该对象不在图中”，不是把所有问句的正确答案设为yes。

基准自然采样D约定为：均匀取提示，再均匀取该提示的有效图像，再均匀取该图像的有效语义元素。令π=E_D[y]，在FIT上计算，测试不重新估计。只在存在两个类别且0<π<1时使用平衡化。该采样约定是V7设计，非EvalMuse官方指标定义。

## 3. 从原始标注得到平衡目标
令d=σ(z_logit)，T=2d−1=tanh(z_logit/2)。实现中统一使用 `z_logit` 为Bernoulli logit，避免把原生二分类logit差又误乘2。

\[
R_M(T)=E_D\left[\frac{y(1-T)^2}{2\pi}+\frac{(1-y)(1+T)^2}{2(1-\pi)}\right],\quad J=1-R_M.
\]

用于与辅助项相加的元素损失取R_M/4。这只是J目标的正比例变换，其“纯元素最优解”不变；与辅助项结合时尺度会影响权衡，因此必须记录系数，不声称任意相同λ都公平。

匹配Logistic：
\[
L_{log}=E_D\left[\frac{y\,softplus(-z)}{2\pi}+\frac{(1-y)\,softplus(z)}{2(1-\pi)}\right].
\]

VCS与Logistic具有相同的D、P/Q、π、输入、可训练模块和辅助项。M0-L1是对投票均值的自然L1任务对照；它不等于proper Bernoulli后验损失，不能给其原始分数强加相同总体概率解释。

不得把soft y直接转成bool，不得使用每batch波动的π，不得用soft label的最小值/最大值重新归一化。DDP/微批累计要按同一全局分母和同一数据采样权重求期望。

## 4. 读回自然概率与公共指标
在类别条件分布一致时：
\[
p_\pi=\frac{\pi(1+T)}{\pi(1+T)+(1-\pi)(1-T)}
=\sigma(z+\log\tfrac{\pi}{1-\pi}).
\]

先验来自FIT，不根据外部基准标签调整。该变换不是任意domain shift的修复。原始平衡概率、先验修正概率、独立CAL上的Platt/temperature校准概率分别保存，测试前固定主要输出。公共J以同一平衡P/Q读取，不能比较不同先验/类别取样下的native统计量。评价时用该评价样本的两类质量分别归一化条件均值，仅是公共J的统计分母；不把测试类比例拿来重新校准概率或改阈值。经验条件均值是比率估计，有限样本不声称逐次无偏或下界。

自然Brier（单个随机标注者）：
\[
E[(Y-p)^2\mid y]=y(1-p)^2+(1-y)p^2.
\]
它等于(p−y)^2+y(1−y)。只保存均值时可计算这项期望；但不知道原始标注人数时，不能虚构标注者置信区间。自然数据不报告“已知真实η的MSE”。

## 5. critic用什么输入
推荐共享实现复用BLIP2/Q-former图文交叉注意力，不引入r7中的PCA代理。视觉encoder按官方freeze_vit约定保留，Q-former、query tokens、ITM head及相关mask head按实际requires_grad表训练。[G02,G03]

每个语义元素有一个或多个已验证字符跨度。模型在完整提示上下文中产生token级两类logit；先在元素跨度内聚合logit差，再得到一个元素概率。默认对有效子词logit差取均值，每个元素只占一次权重，不按BPE长度重复计数。元素重叠时保留独立事件；完全相同观测却带冲突标签时记录为不可辨识的接口限制，不用平均标签掩盖。

对象/属性/关系必须带完整提示上下文；不能把“red”单独作为查询就声称正确绑定到哪一个对象。关系或否定无法在原提示中可靠锚定的元素列入覆盖率和错误分析。无法解析不等于图像不满足。

## 6. 整体与元素输出
- O：来自图文query的总体分数，按原生映射保留1–5范围；它不是概率。
- E：按同一提示的语义元素均匀平均先验修正概率，表示预期满足比例。
- Hα：预先登记的辅助混合 α(O−1)/4+(1−α)E，α只在CAL-SELECT选，不用GenAI标签。

**主estimator应用读数固定为E**；O与Hα作为次级归因与系统性能表。若只有O/Hα有效而E无效，结论是整体监督/混合系统有效，不是元素概率改善已被证明。

E不是“全部要求满足的概率”，也不是整体人类Likert评分的理论等价量。不假定元素独立，不默认用概率连乘。若后续拟合聚合器，需要相同监督的LOG/VCS控制与新配置ID。

## 7. 外部提示没有人工元素清单怎么办
主终末应用必须使用prompt-only生成的元素，不读取图片或人工H。固定本地可用的语言模型及解析模板，在看图像得分/标签之前缓存整个测试提示集的元素、跨度、极性、依赖关系；所有同架构方法共享同一文件。

原始EvalMuse人工元素用于有监督概率评价；外部自动元素用于完整部署评价。两种模式分开，不能在GenAI演示中人工挑选最有利元素。解析失败时，预先定义统一fallback为O并保留`decomposition_failed`，主应用分母不删除这些提示；报告纯E覆盖率及包含fallback的完整结果。

独立的固定prompt-only语义校验只在FIT/CAL开发；部署解析不得根据哪张候选好看或哪张有高评分回改。DSG等基线可以保留其原生解析，但另提供共同解析的组件分析。[P05]

## 8. 为什么低概率误差不等于必然选图更好
对于N个候选共享同一组元素，w_j≥0且Σw_j=1，U_i=Σw_jp_ij，估计值Û_i=Σw_jp̂_ij。令i*最大化U，î最大化Û：
\[
U_{i^*}-U_{\hat i}\le 2\max_i|\hat U_i-U_i|
\le 2\sqrt{N\epsilon_N},\quad
\epsilon_N=\frac1N\sum_i\sum_jw_j(\hat p_{ij}-p_{ij})^2.
\]
这是三角不等式与Cauchy–Schwarz的直接推论，**不是新CS定理**。它只针对预期满足数量，不针对主benchmark的人类整体评分；误差界还随候选N和最大错误敏感。更低平均误差不保证每个有限池更优，错误高分的尾部也要测。


---

原文件：`docs/04_DATA_AND_SPLITS_CN.md`

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


---

原文件：`docs/05_BENCHMARK_METRICS_CN.md`

# Benchmark、指标与统计规范

## 1. 三张主表
**表A：同条件的概率估计。** M0-L1、M1-Logistic、M2-VCS；同一基准下的核读数、常数先验。元素自然Brier、平衡风险、对数损失、类别宏平均准确率/判别、原始与CAL校准分开。

**表B：外部评价模型。** 发布FGA-BLIP2、VQAScore、PN-VQA、适用的DSG/TIFA、新评价器与我们的完整模型；同时列训练数据、宿主、参数规模、解析方式和时间。该表衡量系统表现，不解释成纯损失因果比较。

**表C：相同候选池的实际选图。** 两生成器分别报告GenAI-Rank；Random、oracle-by-observed-human-score、FGA、VQAScore、M0/M1/M2以及可执行的新方法。主读数是人类评分，不是模型自己的score。[P02,G07]

## 2. 元素风险
样本权重默认prompt→image→element宏平均。给每个prompt先算平均风险，再全prompt平均。另报官方元素micro平均用于复现，不能把两个分母都叫“Brier”。

自然Brier：`y*(1-p)^2+(1-y)*p^2`。概率对数损失：`-y log p -(1-y) log(1-p)`，只在计算对数时使用固定epsilon=1e-7；报告原始饱和比例，不能依此隐藏错误极端概率。

提供FIT常数先验、CAL-Platt常数/简单分数作为标定参照。ECE只为辅助，固定分桶边界和箱数；图示可靠性应包含样本量与不确定性。低ECE不证明信息量高，常数预测也可能校准。

元素accuracy：原始逐标注者标签可用时按标签计算；只有均值时将majority阈值规则和y=0.5排除/半信用明确固定。主要Brier保留分歧样本。宏AUROC遇到单一类别返回NA，不能记0或1后平均。

## 3. 整体相关性
- SRCC使用平均并列名次；常数数组返回NA。不得沿用旧r7两次argsort实现。
- PLCC使用原始分数线性相关，不用test拟合映射。若复现官方非线性映射规则，单独表述为该官方协议。
- Kendall用tau-b处理并列；报告有效数量。
- Pooled相关性、每生成器相关性、同提示pairwise accuracy分开。不同提示的固有难度可能提高pooled相关，却不能改善同提示选图。
- Pairwise中人类平局和模型平局的处理提前固定：主表对严格人类优劣对计算，模型平局给0.5；同时列被排除人类平局比例。

## 4. GenAI-Rank官方口径
每个prompt、每个generator恰有9候选，保持官方候选顺序。
- N=1：所有候选人工评分均值，等价于随机选1的期望。
- N=3官方：按[0,1,2]、[3,4,5]、[6,7,8]三个固定组，各选1，人工分数取三组均值。
- N=9：整组选择1。

官方script有`argmax`首个并列的行为。**官方复现列保持此规则**。V7稳健性列另外用所有最大分并列候选的人工分数期望，不通过human score解开平局。[G07]

前一轮建议“任意3张”的方案，在本包降为补充敏感性：枚举全部84个三图子集、所有方法使用同一集合；不能把该结果冠以原论文N=3。论文和展示的主线仍是标准N=9。

\[
M_N(m)=\frac1P\sum_p H(I_m^*(p),p),\quad
Regret_N(m)=\frac1P\sum_p[\max_{I\in C_p}H(I,p)-H(I_m^*,p)].
\]

H保留人工Likert量表原始均值；ordinal量表的平均是benchmark约定，不宣称比例尺度（不能写“质量提高两倍”）。oracle是有限候选+当前人工评分的参照，有标注噪声；没有多轮独立标注时不要称为真正最优图像。

人类胜率以两个方法所选图像的评分比较：win/tie/lose均给出；主数值M9不依赖任意胜负阈值。完整技能标签固定来自官方，不根据效果重新定义“困难”。

## 5. 三种seed不是三倍集成能力
每seed独立选图并评价，报告三seed均值和标准差。统计对比按同一prompt、同一generator、同一seed配对。不要先平均三个模型的score再选图，除非另写为3模型ensemble并计三倍模型开销。

## 6. 主要比较与区间
主后验比较：M2−M1的自然Brier（负为好），在EvalMuse派生DEV；原始prior修正输出为主，CAL-PROB校准输出为次。其他分解属于描述性分析。

主应用比较：M2-E对M1-E、发布FGA原生整体score、原生强VQAScore，DALLE_3与SDXL_Base分别评价N=9，共6个预设对比。报告效应量、prompt-cluster paired bootstrap 95%区间，家族检验Holm校正。相关prompt跨生成器同时抽样，不能当成1600个完全独立提示。

bootstrap以独立prompt为单位；多个元素、图像、子集不独立。三seed导致模型方差估计较粗，主结论同时列逐seed符号和大小；可做seed×prompt重采样敏感性，但不利用一种bootstrap取代另一种直到显著。

“区间包含0”不是等效证明。若主对比结果不好，仍完整保留，不自动转换到某一个positive skill或扩大等效边界。

## 7. 时间与成本
同GPU、相同精度、batch1和合理吞吐batch分别测；固定图像分辨率、prompt长度分层、warmup和重复数。记录图像解码、视觉forward、文本解析、每元素问答、整个9图池的评分与选择开销。解析可按提示缓存，但只对所有方法允许相同缓存条件。

原生公开大模型与共享BLIP2头分表；不能说“同条件”，除非输入和可用计算确实一致。已有固定候选池不包含本轮生成费用；若宣称端到端省钱，需额外计相同生成器相同N的真实生成费用，不能用0替代。

## 8. 测试使用
一次性冻结指模型、阈值、聚合、解析、指标、主要比较在读终末标签前确定；不是文件只能物理打开一次。允许复核已冻结预测的确定性evaluator，错误修复需版本记录。GenAI-Bench不能先被用来调E/O/混合，再把Rank当全新独立提示确认。


---

原文件：`docs/06_TRAINING_IMPLEMENTATION_CN.md`

# 完整训练与代码接入

## 1. 按源码而不是名字复现
已核对FGA来源commit `45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5`。
- 活跃ITM分支：overall L1 + 0.1 token L1 + 0.1 validity-mask L1；不是BCE。[G02]
- 论文Eq.(1)含exp(prompt variance)，当前活跃分支未乘该项。保留N0-Code行；Paper-var只在明确恢复并另记变体后运行，不能静默声称一样。
- YAML指定5epochs、初始lr1e-5、warmup100、wd0.05、训练batch14、image364、freeze_vit=True。这些是已读配置，不是实测可运行性保证。需展开model_type:coco对应初始模型配置。[G03]
- 原生dataset把mask/token label截成32，并将index0设为overall标签；这不适用于“每个语义元素独立概率”的直接解释。[G04]

## 2. 四配置训练矩阵
| ID | 输入/头 | 元素准则 | 总体/有效性辅助项 | 用途 |
|---|---|---|---|---|
| N0 | 原生FGA-code | 原生token L1 | 活跃原生代码 | 从相同FIT重训的任务复现 |
| M0 | 共享完整提示＋语义跨度池化 | abs(d−y)，自然采样 | 与M1/M2相同 | 检查输入/池化改变与L1任务拟合 |
| M1 | 同M0 | 平衡Logistic | 同M0/M2 | 后验损失主要对照 |
| M2 | 同M0 | (1−J)/4 | 同M0/M1 | VCS主方法 |

M0不被解释成严格proper后验基线。N0与M2有输入/权重/聚合差异，不能只归因于loss；M1/M2是对应的严格匹配比较。默认所有M共享自然overall L1（1–5尺度）、validity-mask L1，系数遵循源代码0.1并明确元素损失尺度；辅助项强度的对称敏感性只在CAL，不能按test结果改。

## 3. 初始化与可训练参数
发布FGA checkpoint只用于外部推理参考；主训练从同一个BLIP2 ITM/coco初始化，不从看过全部EvalMuse的FGA成品开始。保存每个tensor加载来源、unexpected/missing keys与requires_grad清单。

保留官方视觉冻结选择；训练Q-former、query tokens、ITM与新增/共享头。完整训练是完整多模态任务路径，并非强制解冻十亿参数视觉塔。共同参数初始化按seed完全一致；更换损失不能改变head大小、token长度、数据顺序和可见标签。

## 4. 输入与跨度
M组使用完整图像的确定性resize到364，无水平翻转/随机裁剪，以免左右、数量和边缘对象标签失效；N0原生处理器独立保留。该输入改动是公共接口设计，不计入VCS独有创新。首次冻结前报告原生处理器具体行为。

M组文本最大长度提案128：只用FIT输入长度分布检查覆盖；若尾部仍明显超出，先验证架构支持后固定更长长度。不要按DEV准确率选截断长度。图文模型超过位置表支持时禁止静默截断，需报错并使用预先固定分块/回退。

跨度使用tokenizer offset mapping/确定性的字符跨度转换；不要靠“同字符串token序列的第一个出现”解释所有实体。元素logit从其语义跨度平均后转换概率。重复词/多跨度/重叠/大小写/标点/否定均有测试。GT索引0总体score不作为元素，padding永不加入分母。

## 5. 从图片batch计算正确的期望
D的目标是prompt→image→element均匀。可使用uniform-image DataLoader并给每图权重N_images/(N_prompts*n_images_in_prompt)，再平均其有效元素；所有M组共享权重。π按FIT同一权重估计，不使用裸元素计数代替。

训练类平衡使用soft投票的两个分量权重，固定总体π。归一化分母不得随哪个标签在微批出现而改变。梯度累计/DDP应合并原始加权分子和全局分母；CPU测试对比大batch和分块精确一致（dropout关闭用于数值检查）。

## 6. 配方与选择规则
主epoch=5，global batch初始按N0源码14。资源允许不构成随意提高M2batch的理由。采用AMP，损失、logsumexp及校准计算至少fp32。warmup100steps、wd0.05，其余实际scheduler按锁定原生代码复现。

N0先跑原生lr1e-5；M0/M1/M2对称搜索lr={5e-6,1e-5,2e-5}，专用tuning seed=17，与正式seed={0,1,2}分开。每个搜索跑完整5epochs，所有epoch checkpoint按CAL-SELECT锁定规则选。

主选择规则：M1/M2按**自然Brier**最小选择，M0亦用同一风险选择以检验概率用途；同时保留各自native criterion选中的epoch作为次级诊断，不据test改选。N0按原生任务L1/官方可验证选择规则，若只有固定最后epoch则保持，单列差异。

主部署score=E，备用O与Hα分别登记，Hα的α∈{0,.25,.5,.75,1}仅用于次级表。CAL-SELECT比较总体score也要记录，不将后处理调参预算藏掉。CAL-PROB只做概率校准，不重选checkpoint。最终正式seed全部运行。

## 7. dropout和复现模式
训练期dropout按共同配置运行。所有评价和部署一律model.eval()；校准不更新模型参数。若新方案含冻结模块的可微优化，明确eval与requires_grad区别，不沿用旧r7冻结参数却未关闭dropout的做法。

保存初始、每epoch、选定权重的hash、RNG状态、数据顺序、effective steps与样本暴露。原始标签/输出未改写；NaN、缺图、跳batch按固定策略记录。

## 8. 冻结核路线的位置
核对照可以在一个共同、未针对M2单独训练的多模态表示上进行RFF/ridge/相对回归，并给Logistic相同特征，带宽和feature数从CAL选。报告N图片、pair数量、特征维数、调参次数与总时长。它是**估计子实验**，不替代完整M0/M1/M2训练；不给核方法一个早已受VCS监督、其他方法未见的表示。

## 9. 资源和执行器
本包不假设服务器有某种GPU或外部API。先实测一个epoch/推理池，再在既有owner批准资源内调度，保留每GPU型号。顺序为原生复现 → 9次专用tune → 12个正式fit → 完整主评价。调参单元计入成本，不能只报3seed的费用。

runner建议接口：`prepare-data / audit / train / export-predictions / evaluate / report`；每个命令仅凭manifest与frozen config运行。仓库可复用工具，但不要导入旧错误的Spearman或把旧R7输出目录混用。实际GPU训练代码由server实现，本包只提供CPU数学/指标参考。


---

原文件：`docs/07_ADVANTAGES_FAILURES_CN.md`

# 优势判定、负结果与有根据的调整

## 1. 哪种结果足以支持怎样的句子
| 结果组合 | 能写什么 | 不能写什么 |
|---|---|---|
| M2 Brier更低，选图也胜M1与公开强方法 | 在这些条件下，二次满足概率拟合改善实际图像选择 | 所有散度/任务都更优 |
| M2/M1接近，二者都胜N0/外部方法 | 共享的语义事件/训练接口有效 | VCS独有优势 |
| M2仅概率更好，人工选图不改善 | 独立标注任务的估计收益 | 理论已经带来更好的生成 |
| 仅O或混合Hα更好，E不改善 | 整体监督或聚合系统更有效 | 元素概率路线已被证实 |
| 只胜CLIPScore，不胜VQAScore/FGA | 基础可用性 | 当前领域领先 |
| 同质量更快 | 指定硬件/负载下的效率收益 | 忽略解析和图像编码的“免费”读数 |

整体研究目标是得到可重复的优势，不是保证一种损失一定获胜。失败不能只通过放宽阈值、删基线或挑一类提示解决。

## 2. 失败诊断顺序
### A. 所有方法接近常数或风险很差
先查raw标签码、正负先验、logit差、mask/token索引、padding、图片路径与训练权重。监督是否把overall当element？长提示是否被截断？这些是数据/实现问题，不能立刻改理论。

### B. VCS比Logistic差
先看相同观测的训练/校准风险、错误高置信样本、梯度范数与收敛；不因为有tanh就自动称“鲁棒”。采用同样数量的CAL学习率/辅助权重尝试，最多一轮有明确假说的修正；完整记录。不要重新开无限求解网格，之前项目已有JS→J无稳定收益的边界。

### C. Brier更好但best-of-9更差
区分同提示平均误差与最大错误候选、简单元素与决定整体评分的关系、E与人类H是否对齐。用CAL候选组分析，不打开GenAI标签调聚合。必要时增加**相同监督的聚合器或ranking auxiliary**，对LOG/VCS同时增加并重命名配置。它是新增系统设计，不再是“只换一个估计loss”。

### D. 原生FGA好，共享M组都差
优先检查源代码token接口到语义事件的变更、输入resize和长度、辅助项比例。恢复一项公共输入/池化对照，不一边修M2一边保留差的M1。若源原生token输出更适合任务，保留它作为主表示并收窄概率解释。

### E. 域内好，GenAI/RichHF差
排查生成器/提示分布、解析失败、上游权重暴露和后处理。明确这是跨提示还是跨生成器。只在预先定义的可用训练家族上做leave-generator-out；不拿外部测试做few-shot后还叫zero-shot。

### F. 解析是主要瓶颈
用同一固定CAL提示的人工元素作诊断上限，区分parser与critic；不能在终末每条提示人工补到“能赢”。第一修正选已有DSG式解析/更可靠的公共prompt-only生成器，所有相关方法共享。若只能用人工元素有效，应用只能称decomposition-conditioned，而非完整部署能力。

### G. 数据/测试不可获取
EvalMuse官方test没公开：使用明确标注的public-train派生DEV，外部Bench/Rank仍是主要证据。GenAI-Rank缺失：不暗改为另一个易任务；请求数据镜像/作者或声明主应用尚未完成。最多将本地EvalMuse多图prompt作为开发替代，不冒充官方Rank。

### H. 新方法代码未发布
DEPICT/DynEval优先文献定位，报告不可运行原因；可做标明“paper-based”的有限复现，但不能给出官方复现名义。不能因一个不可得方法阻塞已可运行的FGA、VQAScore和匹配LOG主表。

### I. 评分更强但开销高
报告准确率/人工质量与时延Pareto曲线，允许它是更准而不是更快的方法。仅当达到相同任务质量再谈节省预算。发布会展示不隐藏模型多次调用。

## 3. 有限扩展顺序
1. 已计划完整训练与冻结评价优先。
2. 最多一轮由诊断指向的公共修正；此前结果保留。
3. 若仍未优于强对照，明确定位为CS的应用证据或负结果，不自动改方向/场馆。
4. 样本效率、leave-generator-out、否定/关系困难集仅在事先规定协议下做；不能选择“仅此条件有效”却写普遍。

## 4. 哪些不是本轮批准
新付费人工标注、文本生成API、扩散生成训练、偏好RL、V7与旧token方法合并成大系统，都需单独预算与科学问题。不能因为“要像发布会”跳过独立人工评价。


---

原文件：`docs/08_V6_TO_V7_CN.md`

# V6 → V7：版本、证据与运行的关系

V7是用户明确允许的新研究/论文版本。旧“r7”是V6服务器探索轮次，本轮不得将`reports/R7`重命名为V7或覆盖。

## 延续
原始S/J、混合参考定义、理论假设与RPC定位延续；全部历史正负结果延续；r8对critic模式/软硬mask/Spearman的审查延续；已经批准的D1-FULL、E1/E2、官方评价按既有协议完成。

## 改变
新优先应用从token压缩转为**细粒度图文满足评价与固定候选选图**。不是将此前失败重新命名。该应用使用一般P/Q的语义判别；未经另外构造，不称自然图文QMI。

V7预期新增贡献是概率估计在既有视觉任务上的可检验收益，而不是初次提出元素核验、best-of-N、风险分数或二次目标。基础公式与平衡RPC重合必须继续说明。[F01,F02,T01]

## 不能覆盖的来源纠正
历史综合文档正文有已被附录4撤回的数学与场馆判断。只引用最新勘误，不复活“JS必然更好”“只有测量才能叫CS”“候选目标无学习”“超过固定百分点才能投会”等句子。[F02]

本次源码新发现也单独记录：FGA原生活跃路径L1/variance差异、32token标签混合与官方N3分组。新实现区别标为V7设计，不假称原论文已经如此。

## 当前仓库截点
`3e2fe59c4861abae46537574c4c59f9bedfe91c6`，2026-10-13 18:51:49 UTC。最新记录涉及ChartQA重试与phase-F前置完整性门；本包没有实时访问队列，也不据此推断那些任务现在仍未完成。

服务器接入时增量盘点新HEAD。若已有同类V7实现，复用并对照差异，不自动重建另一套相同数据流水线。所有新产物路径建议`reports/V7`与`outputs/V7`；标签和结果的读取日期继续记录。


---

原文件：`docs/09_PAPER_DEMO_CN.md`

# 论文与演示输出规范

## 1. 实验章节建议
1. 明确定义语义满足事件与两分布估计任务。
2. 与原生FGA、共享L1/LOG及核路线的概率风险和数据效率比较。
3. 与公开强方法的外部图文一致性评价。
4. GenAI-Rank标准候选池中的人工选图效果，作为主要应用图。
5. 对失败、标签分歧、解析、成本和新生成器的分析。

将Estimator、评分系统、生成器三层分开。我们第一轮训练的是评分器，未微调生成器；best-of-N改变最终交付，不改变底层生成分布本身。不能把“选图后人工平均分上升”写成生成器参数能力提高。

## 2. 主图和表
- 图1：固定提示+九张候选+不同评分器各自选择，显示人工评分和一个已确认的错误类型。
- 图2：N=1/3/9下人工选图分数，官方N3与补充84子集清楚分开。
- 图3：自然元素Brier和同提示选图结果，说明估计改善是否转化。
- 表1：完整12个训练单元的协议、风险、质量、成本。
- 表2：两生成器的主外部选图结果，3seed与公开方法，公开方法训练暴露说明。
- 补充：所有预先固定技能、解析失败、原生/校准输出、完整负结果。

不制作“VCS=稳定/可解释”的宣传雷达图，不用归一化到每方法最大值的指标掩盖差距。

## 3. 演示选择
终末评分冻结后，由prompt hash确定的分层抽样选择案例：对象/属性/数量/关系/否定及失败。至少包含我们输、平局和赢的案例。所有候选可展开，模型选择不得由人工重新替换。独立人工标签来自benchmark，模型自身输出与GT采用不同标记。

展示E元素清单时，对parser未识别的元素明确缺失；不得把未估计关系涂成“满足”。元素概率不是因果归因，不能据此画出“图像这些像素导致正确”的claim。

## 4. 允许的结果表述层级
**尚未运行：**“我们将检验……”；**只有域内风险结果：**“在规定标注任务下降低了风险……”；**完整外部收益：**“在指定候选池/训练预算下，改善独立人工选图评价……”；**只有共享系统效果：**“公共证据模型有效，损失差异未建立”。

模型标题和摘要只有在完整证据出现后填写“improves”“efficient”。不能从理论式直接预写普遍优势。CS-QMI原工作作为理论主线，语义判别应用使用一般CS形式，标题是否继续包含QMI由论文总内容决定。


---

原文件：`literature/REVIEW_CN.md`

# 文献综述：图文一致性评价、概率核验与选图
## 阅读结论
这个应用有成熟的任务与直接强对照，不能只证明“CS可以算分”。相关工作已经覆盖整体评分、细粒度分解、多维偏好和best-of-N。我们的研究空间是：在相同监督与输入下，用原始二次配对风险获得更好的有限样本概率估计，并证明这种改善转化为外部人类判断下的选择收益。是否存在这种收益由实验决定。

## 1. 问答式与语义分解式评价
| 方法 | 已有内容 | 数据/评价角色 | 本轮如何比较 |
|---|---|---|---|
| TIFA [P04] | 将prompt转成问答，用VQA回答检查图像 | TIFA160有人类对齐评价；完整TIFA还含更大prompt/questions集合 | 保留官方解析/VQA；用于外部相关性，不把所有question都当独立图像 |
| DSG [P05] | 原子问题的唯一性、覆盖与逻辑依赖 | DSG-1k检验语义覆盖；论文考察QG/VQA一致性 | 原生系统＋共同prompt解析的组件控制，不能声称首次依赖图 |
| VQAScore [P03,G08] | 利用VQA回答概率给图文整体打分 | 原生强模型及GenAI人类评价 | 必做；legacy与现代宿主分开，既比系统也做same-host对照 |
| PN-VQA [P01] | 同一要求的正反提问共同读取支持 | EvalMuse细粒度判断 | 共用解析的问答控制；计两次提问成本，不把原始yes概率直接当支持真值 |
| DEPICT [P10] | 图像与caption侧回答的一致性及整体分数结合 | 论文报告多基准/多宿主 | 最新直接相关预印本；官方代码未确认，不能报告想象的复现值 |

**对本方案的约束：**原子事实、否定处理、整体与局部结合都已有研究。新增critic须利用已有接口并接受相同解析的信息预算，而不是靠更好的prompt拆解独占输入。

## 2. 训练型整体与细粒度评分
**EvalMuse / FGA-BLIP2 [P01,G02–G06]** 是本轮最直接的训练基线。它提供总体人工分数和细粒度标签，模型以图文交叉注意力同时输出query与文本位置分数。V7借用其训练路径，不宣称新架构。实际代码和论文之间有需要披露的variance权重区别；本包有单独源码核对卡。

**RichHF / RAHF [P06]** 用更丰富的图像区域和文本不一致反馈，不只是一个总体分。其“用反馈改善生成”已经属于先例。本轮只先用对齐维度验证评分泛化；不把V7没有预测的区域热图当成完整RAHF任务。

**DynEval [P09]** 是2026年预印本中的紧凑模型完整微调路线，包含大规模生成样本和教师评估。它提醒我们，BLIP2时代的强基线不是所有现代评价能力的上限。其公开checkpoint与实际许可本轮未完成可运行核查，服务器要核验后加入，不能凭抽象介绍宣布它已复现。

**HPSv3 [P07]、VisionReward [P08]** 主要目标包含人类综合偏好，范围比“每项文字事实是否满足”更宽。它们可在同候选池中作为强系统参照，但概率指标只对有相同标签意义的输出使用。不能将偏好score直接套入Bernoulli Brier后称其不校准。

## 3. 从评价到选择
GenAI-Bench/Rank [P02,D02,G07] 已经将评分器用于同提示的多候选重排，并用独立人工评分衡量收益。本轮的正式应用直接接这个任务，而非自造只偏向VCS的新终点。

关键差别是**跨提示的评分相关性 vs 同提示的最佳图像选择**。一个模型可能知道“哪些prompt一般难”，从而具有不错的全局相关，但在每个prompt内仍选不好。V7同时报告二者，并用标准9图候选的人类平均分作主终点。

更多候选不总会提高选择质量：评分器尾部错判可能让错误图排第一。因此同时报告N=1/3/9、有限候选oracle和失误类型，不单独显示一次成功案例。

## 4. 概率估计与理论归属
严格proper损失有明确总体后验意义，但有限容量、优化、正则、输入表示和聚合都会影响实际结果。[T02] 不预期“Brier必胜Logistic”，也不把某个实验Logistic更准写成其统一最优性。

基础J与平衡RPC一致。[T01,F02] 本轮监督语义支持对应一般两分布CS；不是重新提出QMI、首次使用二次判别或首次产生校准概率。可写的新增能力必须来自相同条件下的估计改进与真实选择收益。若只有相同模型框架共有的改善，就按共有收益报告。

## 5. 辅助诊断与边界
NegBench [P11] 和Winoground [P12] 分别检验否定与组合关系。它们是预先固定的诊断，而不是后验发现哪一个子集最容易胜出后升级主任务。V7概率读数不能保证所有组合推理；没有相应训练支持时必须保留错误。

人类评价协议文献 [P13] 提醒：不能只展示自动分数，评价指令、提示抽样、评分者独立性和重复性需要具体记录。本轮优先使用已存在的人工标签，省去再次建立人类评测的成本，但不把旧标注当成无噪真值。

## 6. 实际必做清单
**训练归因：**N0源代码L1、M0共享L1、M1匹配Logistic、M2-VCS，12个主训练单元。

**主要公开系统：**发布FGA-BLIP2、原生强VQAScore、same-host VQAScore、PN-VQA；DSG/TIFA按官方可执行路径加入。HPSv3/VisionReward至少完成可用性检查并给出有信息量的系统参照。DynEval/DEPICT按代码/权重可获取程度加入，未运行者显式列出原因。

**估计子实验：**共同表示上的RFF/相对最小二乘，概率常数基线；不强制MINE等不同数值目标进入同一Brier表。原论文估计器章节的MINE/NWJ/InfoNCE/SMILE由已有项目证据承担，不用新VL任务替代。

## 7. 仍未证明的事
本轮没有验证任何新模型在自然图像上优于这些方法；没有完成所有新预印本的代码复现；没有建立一个保证优于Logistic的统计定理。V7是一个可执行、可被否定的应用研究方案。


---

原文件：`literature/CODE_PROTOCOL_AUDIT_CN.md`

# 已读源码：直接影响实验设计的六项发现

## C1. 原生FGA不是Logistic
`fga_blip2.py: forward(match_head='itm')`的活跃分支计算绝对误差。论文也把L_os、L_es、L_mask写成L1。V7必须保留N0/M0-L1，不能把“native FGA”与“matched BCE”当成同一行。[G02,P01]

## C2. variance weighting存在paper–code差异
论文有exp(σ(prompt))权重，下载到的活跃源码没有乘；读入了var，但含exp(var)的代码被注释且在此前return后。N0-Code与N0-Paper-var应独立命名。公开发布权重究竟采用哪条路径，本轮仅凭代码不能确认；若需要复现论文最优表，先查作者说明/具体checkpoint元数据，不擅自补乘再叫原生。[G02,G06]

## C3. 元素标签不是普通固定维向量
原始JSON每条样本的element_score是不同键的投票字典；自动viewer转换可能失败。`process_train.py`将投票均值化，`element2mask.py`把元素映射到token并对重叠位置平均，某元素找不到会过滤样本。不能先运行预处理再忘掉丢了多少数据。[D01,G05,G06]

## C4. 截断和index0会改变概率含义
dataset读取时截成32、padding为0，index0被置为总体评分归一化值。V7元素风险不得把该位当成人工二元元素；M组保留显式语义跨度和missing标记。N0原生复现照旧但说明覆盖率，不把原生32token结果与M组完整输入差异都归给损失。[G04]

## C5. 原生N=3不是全组合
GenAI脚本把9候选按3个连续分组，每组argmax再平均；N=9整组选一张。其并列score遵守argmax首个索引。V7官方复现保持候选顺序；全84个子集和并列期望作为单独敏感性，不能覆盖主值。[G07]

## C6. 模型/代码版本不是可交换的
t2v_metrics README明确原论文应使用legacy v3.0；现代模型环境另行维护。不能升级依赖后仍称完全复现原论文。各系统的输入模板、视觉分辨率、yes/no读取、token归一化和可训练模块均入manifest。[G08]

## 接入回复格式
每项：`fact_confirmed / fact_modified / not_applicable`；附本地commit、函数、实际有效配置、20条以内示例与修正决定。确认源码行为不等于它解释了全部性能；不预写“修了这一项VCS必胜”。


---

原文件：`literature/SOURCES_CN.md`

# 来源目录与核验范围


所有网络来源核验日期：2026-10-13。以下是原始论文/官方代码/卡片。不同来源包含不同层次的事实；未完成运行复现与全文核验的项明确说明。


原始理论全文已在对话提供；本包不复制他人全文或字体。


## [F01] 合作者原计划

**来源：** 用户提供：Variational Cauchy–Schwarz Quadratic Mutual Information — Research Plan, September 2026

**入口：** conversation:turn581file3

**本轮核验：** 全文；重点 §2–8、§10、§17

**注意：** 原始理论；不把原草稿全部创新表述当成已完成新颖性审计


## [F02] 历史审查的修订附录

**来源：** 用户提供：VCS_QMI 项目综合文档，附录4

**入口：** conversation:turn581file0#L326-L359

**本轮核验：** 已给出的修订附录

**注意：** 只继承已明确的归属与勘误；不把10月10日旧状态当成当前状态


## [G01] 本项目快照

**来源：** W-Yinghao/VCS_QMI @ 3e2fe59c4861abae46537574c4c59f9bedfe91c6

**入口：** https://github.com/W-Yinghao/VCS_QMI/tree/3e2fe59c4861abae46537574c4c59f9bedfe91c6

**本轮核验：** GitHub main API

**注意：** 截至2026-10-13 18:51:49 UTC；不是实时Slurm查询


## [P01] EvalMuse / FGA-BLIP2 / PN-VQA

**来源：** EvalMuse-40K, AAAI 2026；arXiv版本标题略不同

**入口：** https://ojs.aaai.org/index.php/AAAI/article/view/42458

**本轮核验：** 正式出版摘要；arXiv HTML方法和数据节

**注意：** arXiv: https://arxiv.org/html/2412.18150v2


## [D01] EvalMuse公开数据

**来源：** DY-Evalab/EvalMuse

**入口：** https://huggingface.co/datasets/DY-Evalab/EvalMuse

**本轮核验：** 官方卡片、字段预览

**注意：** 卡片有test标注稍后公开说明；以实际访问为准，不能假定可下载测试标签


## [G02] FGA核心代码

**来源：** lavis/models/blip2_models/fga_blip2.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/models/blip2_models/fga_blip2.py

**本轮核验：** 完整源码

**注意：** 活跃itm分支是L1；variance分支注释且在return后


## [G03] FGA训练配置

**来源：** lavis/projects/blip2/train/fga_blip2.yaml

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/projects/blip2/train/fga_blip2.yaml

**本轮核验：** 完整源码

**注意：** freeze_vit=True, 5 epochs, batch14, image364；有效运行配置仍须展开


## [G04] FGA数据读取

**来源：** align_datasets.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/lavis/datasets/datasets/align_datasets.py

**本轮核验：** 完整源码

**注意：** 32 token截断；第0位token_score是总体分数，不是语义元素标签


## [G05] FGA元素映射

**来源：** process/element2mask.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/process/element2mask.py

**本轮核验：** 完整源码

**注意：** 多个元素的重叠token标签被平均；失败样本会被过滤


## [G06] FGA标签预处理

**来源：** process/process_train.py

**入口：** https://github.com/DYEvaLab/EvalMuse/blob/45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5/process/process_train.py

**本轮核验：** 完整源码

**注意：** 原始投票转均值；按prompt算总体评分方差；保留原始投票


## [P02] GenAI-Bench / GenAI-Rank

**来源：** GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation, 2024

**入口：** https://arxiv.org/html/2406.13743

**本轮核验：** 全文相关数据、排名协议节

**注意：** 2024 SynData workshop记录；不写成CVPR主会论文


## [D02] GenAI数据入口

**来源：** BaiqiL/GenAI-Bench

**入口：** https://huggingface.co/datasets/BaiqiL/GenAI-Bench

**本轮核验：** 官方卡片

**注意：** Bench与Rank提示重叠，必须作为共同终末评测资源


## [G07] GenAI-Rank官方脚本

**来源：** genai_image_ranking.py

**入口：** https://github.com/linzhiqiu/t2v_metrics/blob/main/genai_image_ranking.py

**本轮核验：** 源码1–245行；blob 0e0fdb3e9550b1b1217c9feb78c46ed1095a7252

**注意：** DALLE_3 / SDXL_Base；N=3按固定三个连续分组；服务器锁定commit


## [G08] VQAScore工具版本

**来源：** t2v_metrics README

**入口：** https://github.com/linzhiqiu/t2v_metrics

**本轮核验：** 官方README

**注意：** 原论文复现用legacy v3.0；现代宿主用另一个已锁定环境


## [P03] VQAScore

**来源：** Evaluating Text-to-Visual Generation with Image-to-Text Generation, ECCV 2024

**入口：** https://arxiv.org/abs/2404.01291

**本轮核验：** 原论文与官方项目说明

**注意：** 必须比较原生公开强模型，不只比较CLIPScore


## [P04] TIFA

**来源：** TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering, ICCV 2023

**入口：** https://arxiv.org/abs/2303.11897

**本轮核验：** 原论文摘要与EvalMuse关联协议

**注意：** TIFA160是人工对齐子集，不等于4K prompt的TIFA v1.0


## [P05] DSG

**来源：** Davidsonian Scene Graph, ICLR 2024

**入口：** https://proceedings.iclr.cc/paper_files/paper/2024/hash/446905abfec9db8856f7f7465bbd5552-Abstract-Conference.html

**本轮核验：** 正式摘要

**注意：** 对象与属性依赖、唯一原子问题不是本文新发明


## [P06] RichHF

**来源：** Rich Human Feedback for Text-to-Image Generation, CVPR 2024

**入口：** https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.html

**本轮核验：** 正式摘要

**注意：** 有多种质量维度；只把对齐维度纳入本轮主要相关性


## [P07] HPSv3

**来源：** HPSv3: Towards Wide-Spectrum Human Preference Score, ICCV 2025

**入口：** https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html

**本轮核验：** 正式摘要

**注意：** 综合偏好，不是同定义的语义满足概率


## [P08] VisionReward

**来源：** VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning, AAAI 2026

**入口：** https://ojs.aaai.org/index.php/AAAI/article/view/38107

**本轮核验：** 正式摘要

**注意：** 多维偏好参考；记录实际输出维度与聚合方法


## [P09] DynEval

**来源：** DynEval: Holistic Evaluations of T2I Generative Models in the Wild, 2026 preprint

**入口：** https://arxiv.org/abs/2607.11199

**本轮核验：** 论文HTML与方法概述

**注意：** 模型/权重入口本轮未完成可执行复现核验；不得虚构官方数值


## [P10] DEPICT

**来源：** DEPICT: Scoring Text-to-Image Alignment by Answer Agreement, 2026 preprint

**入口：** https://arxiv.org/abs/2610.03617

**本轮核验：** 论文HTML

**注意：** 本轮未确认可用官方代码；论文复现必须独立命名


## [P11] NegBench

**来源：** Vision-Language Models Do Not Understand Negation

**入口：** https://arxiv.org/abs/2501.09425

**本轮核验：** 原始摘要与官方项目页

**注意：** https://negbench.github.io/；选取图像子任务，不能跨18种任务混成总准确率


## [P12] Winoground

**来源：** Winoground, CVPR 2022

**入口：** https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.html

**本轮核验：** 正式摘要

**注意：** 官方text/image/group三类判据，严格不等式


## [T01] RPC

**来源：** Self-supervised Representation Learning with Relative Predictive Coding, ICLR 2021

**入口：** https://arxiv.org/abs/2103.11275

**本轮核验：** 原文关系＋项目已核对附录

**注意：** 平衡RPC与J完全重合，不另造重复算法行


## [T02] 正规复合损失

**来源：** Composite Binary Losses, JMLR 2010

**入口：** https://jmlr.org/papers/v11/reid10a.html

**本轮核验：** 原始论文页面

**注意：** 用于区分总体proper性质、受限函数类和有限优化


## [P13] 人类评价协议

**来源：** Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation, CVPR 2023

**入口：** https://openaccess.thecvf.com/content/CVPR2023/html/Otani_Toward_Verifiable_and_Reproducible_Human_Evaluation_for_Text-to-Image_Generation_CVPR_2023_paper.html

**本轮核验：** 正式摘要

**注意：** 仅在新生成人工确认时使用；首轮优先已有人工标签


---

原文件：`reference/README_CN.md`

# CPU参考代码

包含数学目标、soft投票的损失、先验修正、并列名次、prompt级聚合、标准GenAI-Rank候选选择与manifest检查。**不包含已集成的FGA训练器、数据下载器、模型权重或GPU作业脚本。**

```bash
python -m pytest reference/tests -q
python reference/validate_manifest.py path/to/manifest.jsonl
python reference/score_rank.py --predictions frozen_prediction.jsonl \
  --labels sealed_human_labels.jsonl --output new_versioned_result.json \
  --allow-scoring
```

预测每行字段：prompt_id, generator_id, image_id, score。一次文件只放一个方法、一个训练seed和一个score口径，不混合。标签字段另有candidate_index(0..8)与human_score(1..5)。不能使用输入行顺序隐式配对。

`score_rank.py`默认保留官方N3分组与首索引平局。`--protocol all_subsets`、`--tie-rule mean`是补充列，不覆盖官方列。工具拒绝缺失候选/重复/非有限输出/覆盖旧结果。

`common_j`从评价标签形成两类条件均值，这仅用于公共统计量的分母。它不把测试类比例用来重新校准概率、修改阈值或选择模型。训练权重与部署prior仍仅来自FIT/CAL规定角色。

bootstrap函数的区间只覆盖提示抽样不确定性；近似双侧p不是精确随机化检验。多seed另外报告。实现集成后需与项目选定的正式统计函数作一致性测试，协议不可事后更换。

依赖：numpy、scipy、pytest；loss梯度测试另需torch。版本以VERIFICATION.json中的本地实测为参照；服务器应锁定兼容自己的视觉模型环境，不直接覆盖已有环境。
