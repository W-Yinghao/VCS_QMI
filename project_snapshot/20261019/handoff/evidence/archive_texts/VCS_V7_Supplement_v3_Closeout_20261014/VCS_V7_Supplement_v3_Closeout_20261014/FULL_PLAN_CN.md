# V7 补充与收尾 · 完整服务器方案

2026-10-14。来源事实、直接推导、新实验设计分别标明。



---

原文件：`README_CN.md`

# V7 补充与收尾包 v3
## 完成外部竞争评价，拆开估计、聚合与选择三个问题

**本包不更换 V7 方法，不重选已冻结的主输出，也不把它改写成新一轮成功故事。**它补齐当前尚缺的实验与分析，形成可以进入论文的完整结论。下一轮方法另见独立 V8 包。

建议先读 `01_SERVER_AGENT_PROMPT_CN.md` → `docs/02_TASK_PLAN_CN.md` → `docs/03_DIAGNOSTICS_CN.md` → `docs/04_EXTERNAL_BENCHMARK_CN.md`。`FULL_PLAN_CN.html` 是聚合离线阅读版；`V7_SUPPLEMENT_OVERVIEW_CN.pdf` 是概览。

| 本包必须完成 | 复用内容 | 新训练 |
|---|---|---|
| V7外部主家族、标准N=1/3/9选图与成本 | 已有完整模型和全部冻结预测 | 不默认重训 |
| 元素标签→总体评分的对应检查 | EvalMuse CAL/DEV原始投票、E/O/H预测 | 无 |
| 同提示排序、最大错误候选与解析分解 | 既有checkpoint重新导出必要字段 | 无 |
| 传统CS/密度比与非CS比较的完整收束 | X11/X21/PCA/N0特征敏感性 | 只补缺失/失败单元 |
| 二次读数、外部系统、归因三套结论 | 原v1/v2协议与结果 | 不以BCE胜负控制其他面板 |

已有记录表明完整M2和M1未分离；共同特征的KDE较弱而Nyström/RuLSIF并非都弱。这些是本轮起点，不是要通过删基线消除的问题。[G05,G07,G08]

本包包含本地数学/聚合参考实现，不包含完整GPU训练器、数据、权重或新的自然数据结果。引用键对应 `literature/SOURCES_CN.md`。

交付前增量与最新双分支快照见 `docs/00_DELIVERY_UPDATE_CN.md`；不替换旧冻结协议。


---

原文件：`01_SERVER_AGENT_PROMPT_CN.md`

# 发给服务器：接续 V7，不覆盖冻结方案

请在 `v7/t2i-alignment` 上接入本补充包。核对 main 与 V7 两个HEAD；本包依据 main `c7296d2`、V7 `891fb8a`。先盘点已有V7作业/输出，所有已完成项复用，已运行项不重复提交。

立即执行 `V7C-00..06` 中允许的接入、CAL/DEV诊断、外部预测与运行成本核对。V7模型、E主输出、O/H次级输出、checkpoint、parser和八个外部主对比均沿用已冻结协议。所有新分析写入 `reports/V7_CLOSEOUT/`，不覆盖 `reports/V7/` 原报告。

**权限：本请求授权补充设计与按现有权限开展工作，不自动授权读取封存GenAI人工标签，不授权付费API。**原owner要求的终末评分批准继续有效；将一份包含所有主家族预测hash、复现审查与精确命令的评分申请交owner。不能将“设计下一轮”解释为取消V7或将GenAI当校准集。

本轮报告先列公开任务对手，再列传统CS/非CS估计路线，最后用BCE说明归因。不能以M2没有胜过M1为理由不完成外部评价。

第一批交付：
1. `V7C_00_INTAKE.md`：两个新HEAD、已完成/运行/缺失矩阵、现有权限；
2. `V7C_01_LINKAGE_REPORT.md`：元素人工参照、模型E/O/H与总体人工评分之间的同提示比较；
3. `V7C_02_TAIL_PARSER_REPORT.md`：尾部错误、自动解析与人工元素、参考分布/先验和全部失败；
4. `V7C_03_EXTERNAL_READINESS.md`：B0/B1/B4T/B4D的完整预测、变体与成本；
5. `V7C_04_SCORING_REQUEST.md`：封存标签待批准事项，不能先评分后补签。

V8可在独立分支使用FIT/CAL开发，但不得回写V7主方案。先冻结V8版本/选择规则、记录其与V7终末标签读取的先后关系。不要等待V7某一行变正才开始下一轮设计，也不要让下一轮推迟当前完整主比较。

交付增量：先读 `docs/00_DELIVERY_UPDATE_CN.md`。最新已读main `2d38ea3`、V7 `4f90d6d`；TIFA问题缓存已产生，复用并保留33条空问题的原并列规则。接入仍须核对更晚HEAD。


---

原文件：`docs/00_DELIVERY_UPDATE_CN.md`

# 交付前增量核对：2026-10-14

## 双分支快照
本包的设计审查基准是 main `c7296d2`、V7 `891fb8a`。交付前通过GitHub比较接口再次检查：

| 分支 | 最新已读提交 | UTC时间 | 本次新增内容 |
|---|---|---|---|
| main | `2d38ea35e8e6e178f580e5bbb052fc37e05230c7` | 2026-10-14 19:26:19 | REV9/REV9-FULL将8个待评价占位作业打包为4个配对作业；报告称配置、shard、evaluator不变 |
| v7/t2i-alignment | `4f90d6dd68c3c8d0977b6eab30731abc03780c4f` | 2026-10-14 19:23:18 | TIFA本地问题生成的无标签统计 |

两个分支各新增一个提交；不据此推断实时队列或完整评测已完成。本包没有提交任何服务器作业。

## V7新增的具体事实与操作
TIFA公开本地生成器在800个GenAI-Rank提示上报告31个解析失败；平均生成8.04题、过滤后保留5.66题；33个提示没有保留问题，候选在该方法下全部并列。[G08]

这些是**不读取人工评分的实现统计**，不是TIFA选图效果。V7C-03应直接复用已生成文件，核对以下事项：
- 33个空问题提示继续沿用已冻结的分数/并列规则并保留在主分母；不得删掉再与完整M2比较。
- 记录31个解析失败与33个空问题提示的集合关系，不假设两个数字完全重合。
- 完成原生路由一致性和全部VQA预测检查；不存在“问题生成结束就等于外部基线完成”。
- V8复用外部baseline时，保留同一版本和失败规则；若后来合法修正baseline，必须另记版本并对新轮所有方法公平评价。

REV9作业包装只影响运行接续。本包不取消、不重复提交，也不将它改写成已得到正或负的干预结果。[G09]

## 来源
[G08] https://github.com/W-Yinghao/VCS_QMI/commit/4f90d6dd68c3c8d0977b6eab30731abc03780c4f
[G09] https://github.com/W-Yinghao/VCS_QMI/commit/2d38ea35e8e6e178f580e5bbb052fc37e05230c7


---

原文件：`docs/02_TASK_PLAN_CN.md`

# V7 补充实验任务书

## 1. 当前事实与尚未知道的事
已核对的分支协议规定：EvalMuse公开TRAIN为32,717图；FIT为26,177图，CAL_SELECT为1,653图，CAL_PROB为1,630图，DEV为3,257图。划分按文本/图像连通组件；DEV有307提示、304组件。所有总体评分保留3或6个有效原始值。[G04]

完整多模态M0/M1/M2和N0不等于X21固定特征实验。X21主要表征为未经过EvalMuse监督更新的768维BLIP-2/Q-former；PCA64/256及N0-s0表征是单列的敏感性，不能选择最有利维度替代主表。[G02,G05]

尚未有可引用的GenAI主家族结果。外部报告仍含早期queued与后续实施记录；服务器须以完整预测文件、有效样本数、日志和checksum判断完成，不靠标题/队列退出。[G06]

## 2. 任务列表
| ID | 要回答什么 | 数据/权限 | 操作与交付 |
|---|---|---|---|
| V7C-00 | 到底还缺什么 | 代码、manifest、既有结果 | 分支快照；12正式训练、X11/X21敏感性、外部预测逐格状态 |
| V7C-01 | 元素满足程度是否对应整体效用 | CAL/DEV原始标注 | 人工元素均值与总体分数；同提示选择/排序；不训练新聚合器 |
| V7C-02 | 错误出在元素预测、聚合还是parser | CAL/DEV已训练模型 | GT元素与固定自动元素双路径；E/O/H；覆盖率和尾部分析 |
| V7C-03 | 公开对照有没有被实现削弱 | label-free提示与开发数据 | 官方版本/本地适配一致性检查；部署成本与完整14,400预测 |
| V7C-04 | 实际选图是否有外部收益 | GenAI封存人工标签 | 先提交freeze+评分申请；获准后执行既定八个对比及原v1家族 |
| V7C-05 | 强核方法与非CS工具是否被公平呈现 | X11/X21全部原产物 | 统一目标、native/readout字段、尺度/容量/成本；只补缺行 |
| V7C-06 | 外部泛化边界是什么 | GenAI-Bench、TIFA160、RichHF | 先数据/重叠/标签权限清单；未批准的不读。与V8的留出计划协调 |
| V7C-07 | 哪些结论可进主文 | 全部冻结结果 | 一份完整阶段表、效应量、失败、方法适用范围和论文图 |

## 3. 复用与新增成本
默认 **0 个新的完整多模态训练单元**。C01/C05主要是CPU重读；C02只作已有完整模型的推理，不是拟合探针；C03/C04完成已经规定的外部方法。模型/数据损坏造成缺失时，以原配置补齐，保留失败日志。

不能为了“补齐”再次重训全部M2/M1。发现bug时分清：结果脚本bug只重算；推理bug重导出全部受影响方法；训练bug需单独列出影响、旧结果和重训预算。V8才是方法更改的地方。

## 4. 本轮结论问题
A. 外部完整系统：是否比FGA、VQAScore、TIFA、DSG在相同候选中选得更好？
B. 传统估计路线：相同表示下是否比KDE/RuLSIF/Nyström更准或更便宜？
C. 非CS：DV/NWJ等原生统计量与公共后验读数分别怎样？
D. 归因：VCS与BCE差距是否出现；没有差距不删A/B/C。

即使E不胜而O/H有效，也报告系统收益，但不能把它说成元素概率路线被证实。既有Hα不得因终末结果改α。[G02,G03]


---

原文件：`docs/03_DIAGNOSTICS_CN.md`

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


---

原文件：`docs/04_EXTERNAL_BENCHMARK_CN.md`

# 外部benchmark完成规范

## 1. 冻结主家族不变
依据2026-10-14的amendment v2：[G03]
- 主方法：M2-E，三个正式seed分别选图；
- 主对手：B0发布FGA-BLIP2原生overall、B1 VQAScore CLIP-FlanT5-XXL、B4T TIFA本地公开路由、B4D DSG本地LLM适配；
- 两生成器DALLE_3和SDXL_Base，N=9，共8个对比；
- M2-E对M1-E的两个对比是归因家族；旧v1六个对比完整保留；
- 原E/O/H模型、α、parser、校准器、候选顺序、样本范围不按新诊断更换。

## 2. 公共系统状态与补齐顺序
| 行 | 必做操作 | 不能怎样处理 |
|---|---|---|
| B0/B1 | 验证每生成器800×9记录、score方向、权重版本、原生预处理 | 不以开发表现差就降级模型 |
| B4T | 完成本地生成、过滤、VQA、全部空问题回退与成本 | 不能把官方本地路由伪称原论文GPT路由 |
| B4D | 本地LLM替代须明示；检查tuple/dependency/question结构和非法依赖处理 | 不能只检查JSON可解析就宣布复现等价 |
| B3 PN-VQA/B2现代VQA | 按已登记次级方案完成 | 不改变主家族来追更好数字 |
| B5 HPSv3/B6 VisionReward | 优先补至少一个可运行的现代评分系统；记录训练目标与全成本 | 偏好分数不直接当满足概率算Brier |
| B7/B8新预印本 | 核对公开代码/权重与许可；不可用写明证据 | 不编造官方复现结果，不因不可得无限阻塞四个可运行主对手 |

source fidelity：使用不含人工质量标签的固定输入检查本地DSG/TIFA与作者公开问题文件。结构相似不等于语义质量；在固定30–50条提示上输出盲审表供检查，不让人工按哪张图或哪方法赢改问题。API仍未获准；本包不引用旧粗略美元估算。[G06]

## 3. 终末评分的两个阶段
**先预测冻结。**主家族所有输入/预测完整，manifest包括代码、模型、parser、CAL选择与时间戳。任何主对手不可运行，评分前说明并按原权限修订家族；不能看到human分数后删行。

**再经原批准读标签。**复用 `scripts/v7/v7_score.py rank --allow-scoring` 的既定流程。不是本地参考脚本去绕过closed目录。一次冻结强调禁止结果驱动的重选择，不限制对固定预测的确定性纠错复核。

## 4. 指标
M_N=每提示所选图像的人工1–5均值；Regret_N=当前人工评分确定的池内最大均值减所选均值。N=1为随机期望；N=3按[1,2,3]/[4,5,6]/[7,8,9]三组各选一再平均；N=9整组选一。两个生成器不合成18候选。

SRCC/PLCC、prompt内pairwise、元素风险是独立面板，不能替代M9。人工Likert均值按benchmark约定使用，不称自然比例量表；“从3到4”不是质量提高33%。

8主对比按原prompt-cluster bootstrap 5,000、seed2718、跨生成器共享抽prompt、Holm。若有提示近重复连通组件，新增component-cluster敏感性，不静默替换原主统计。三模型种子不是三倍独立prompt；不先平均score变成ensemble。

## 5. 资源与失败
每图/每个9图池：图像解码、解析摊销、视觉forward、全部问题调用、聚合、峰值显存。native大型系统与同架构loss表分开。自然训练预算用真实日志，不拿CPU参考运行时间代表GPU成本。

同时报告选择一致率、win/tie/lose、所有冻结技能与失败案例。主比较未领先也完整发布；不再寻找一个小技能子集升级成“理论优势”。

## 6. 额外外部集与下一轮
GenAI-Bench中与Rank重叠的提示不能算独立验证；TIFA160/RichHF的下载/评分按原权限处理。可先登记候选的新外部资源和输入排重，不先读取标签。若owner希望给V8保留一个新确认源，明确逐提示标签暴露表，再决定是否现在扩V7。任何已经授权/运行的旧评价不擅自取消。


---

原文件：`docs/05_ESTIMATOR_AND_CLOSURE_CN.md`

# 估计器外部比较与收尾结论

## 1. 四个面板分开
A0 原生目标：经典CS-KDE读取固定参考散度；MINE/DV、NWJ、InfoNCE、SMILE读取各自定义。无自然真值不计算伪MSE。
A1 同目标S：KDE后验、RuLSIF、Nyström平方回归、神经VCS/BCE/DV/NWJ的明确定义桥接，共同P/Q与J。
A2 学习系统：完整Q-former微调N0/M0/M1/M2，不能与固定特征结果混作同等训练。
C 实际选择：完整评分器及具有合法单样本接口的估计器，按同一候选池与人类标注。

MMD/HSIC用于相应分布检验面板，不给它们捏造满足概率。RPC在平衡参数下等于J，不重复一行相同代码当外部竞争者；若用原生不同参数需另设对象和读数。[F01,F02,P10,P11]

## 2. X21补齐清单
复用主768维、PCA64/256、N0-s0表征的全部有效拟合；逐项检查共同特征hash、FIT标准化、seed、带宽、中心数、边界扩展、solve残差、clip比例、native与common选模、DV常数归一化和CAL-PROB角色。缺失单元先核对日志再按冻结配置恢复。[G05]

结论写成“相对何种密度/函数类、在何种维度和样本条件”，不写“学习型一定胜核方法”。核回归同样学习数据。若Nyström/RuLSIF更好，进入主表而不是附录隐藏。

概率主指标保留自然prior-corrected Brier。M0原生L1不保证学习平衡后验；既有统一先验修正列照协议保留，同时保留raw/calibrated并注明解释边界。不要因本轮发现哪个版本好而换原主输出。

## 3. 新分析能怎样进入主文
| 实际结果 | 支持 | 不支持 |
|---|---|---|
| V7外部强方法被超过，M1/M2近似 | 整套视觉证据系统有应用价值 | 二次拟合独占收益 |
| VCS更准且外部更好 | 指定设置中的估计与任务收益 | 所有域、所有P/Q均最好 |
| KDE差而Nyström/RuLSIF不差 | 密度估计与回归路线有重要差别 | 所有传统CS方法失败 |
| 人工元素平均已不能排序H | 原主效用连接有局限 | 任意逐候选分数都不能排序 |
| O/H好而E差 | 整体监督/混合系统有效 | 元素均值是充分的人工质量估计 |
| 外部全部不足 | 当前方法不能承担强应用主张 | 原始S/J数学被否定 |

## 4. 完成定义
完整报告必须有：八主外部对比+原v1家族、所有已规定完整训练seed、X21维度敏感性、真实全成本、解析覆盖、CAL/DEV效用连接、全部限制、下一轮独立配置。

R5不追加修补；D1按REV9既有授权收尾。V7尚未读出的终末标签不能被一个新“体检解释”提前判负。最终结论由冻结实验支持的范围决定，不用场馆和固定领先百分点裁决整个研究。


---

原文件：`docs/06_V7_V8_BOUNDARY_CN.md`

# 两个独立包怎样并行

V7收尾包保留已经训练、已经选出的模型。V8重新定义的是总体质量事件，必须新目录、新head、新配置、新训练记录。原S/J不变，旧模型不被覆盖。

**现在可以并行做**：V7外部预测、CAL/DEV分析；V8文献定位、原始FIT标签构造、数学测试、数据/代码适配以及按本地owner资源权限安排新训练。V8不必等待V7是否赢，但必须先写完整方法与选择规则，不能读GenAI逐样本错误再设计标签或聚合。

**终末暴露分三种**：
1. V8已完整预先登记且尚未看V7标签：按新计划评价可称预先规定的后续比较；它仍是同一benchmark，不能声称新增独立数据集。
2. V8设计/参数受已读V7 GenAI结果影响：该源只能称重复使用benchmark；必须另有未用于开发的确认资源。
3. 若新结果已看后又改V8：变化记为开发迭代，不能仍用第一次预注册名义。

建议维护 `LABEL_ACCESS_LEDGER.csv`，含数据集/提示集合hash、首次人工标签读取时间、读取者/目的、触发的模型改动。不要用文件权限或“只打开一次”的表面形式代替科学上的训练/评价隔离。

本包不授权新付费API、人工标注或生成器训练，不以推迟V7外部评价来为V8争取更好的故事。评分前依原owner流程确认；新模型的结果永不回写V7主表。


---

原文件：`literature/SOURCES_CN.md`

# 来源目录与使用范围
核对日期：2026-10-14。原始文献/作者页面与仓库事实分开。本包中新公式若注明“直接推导”，属于本轮分析，不冒充来源已证明的新定理。

## [F01] 合作者原始 CS-QMI 研究计划
来源：user-provided Research Plan, September 2026。
入口：conversation:turn599file3
核对范围：全文已在对话提供；本包使用§3–7、§10、§13。
解释边界：一般P/Q、S/J与gap；不复制论文全文。

## [F02] 历史审查附录4
来源：用户提供的2026-10-10勘误。
入口：conversation:turn599file0#L326-L359
核对范围：本次复核已提供内容。
解释边界：按最新勘误理解；不继承已撤回的不可能性和场馆断言。

## [G00] 当前主分支
来源：W-Yinghao/VCS_QMI。
入口：https://github.com/W-Yinghao/VCS_QMI/tree/c7296d2ebb15658fa93737803ff757c98f64ad4a
核对范围：GitHub分支API，2026-10-14。
解释边界：只读快照，不是实时队列。

## [G01] V7分支
来源：W-Yinghao/VCS_QMI。
入口：https://github.com/W-Yinghao/VCS_QMI/tree/891fb8a9948854b2ff4dc93e9212b0cf791b942e
核对范围：GitHub分支API，2026-10-14。
解释边界：分支与主分支分开锁定。

## [G02] V7冻结协议
来源：reports/V7/V7_03_PROTOCOL_FROZEN.yaml。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_03_PROTOCOL_FROZEN.yaml
核对范围：全文。
解释边界：12正式训练、E主读数、校准规则。

## [G03] V7外部主家族修订
来源：reports/V7/V7_03_AMENDMENT_V2_20261014.yaml。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_03_AMENDMENT_V2_20261014.yaml
核对范围：全文。
解释边界：8外部对比；BCE为另一个归因家族；封存标签权限。

## [G04] 实际数据审查
来源：reports/V7/V7_01_DATA_AUDIT.md。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_01_DATA_AUDIT.md
核对范围：全文。
解释边界：32717图、2689组件；3/6原始评分，不等于后验真值。

## [G05] 共同特征估计协议
来源：reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7X_20_COMMON_FEATURE_PROTOCOL.yaml
核对范围：全文。
解释边界：原生768维、PCA64/256、N0表征敏感性、核调参。

## [G06] 外部基线接入与状态
来源：reports/V7/V7X_30_EXTERNAL_BASELINES.md。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7X_30_EXTERNAL_BASELINES.md
核对范围：全文。
解释边界：本地TIFA/DSG/PN-VQA与API禁用；不把状态表当结果表。

## [G07] 全模型概率结果
来源：reports/V7/V7_TABLE_A_results/v7_table_a.json。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/891fb8a9948854b2ff4dc93e9212b0cf791b942e/reports/V7/V7_TABLE_A_results/v7_table_a.json
核对范围：本次读取开头90行；完整比较继承上一轮核对。
解释边界：不宣称本次重算全部训练结果。

## [G08] 新项目体检
来源：reports/PROJECT_MAP_20261014.md。
入口：https://github.com/W-Yinghao/VCS_QMI/blob/c7296d2ebb15658fa93737803ff757c98f64ad4a/reports/PROJECT_MAP_20261014.md
核对范围：本次重点核对理论定位与实验索引；长输出部分截断。
解释边界：事实与作者解释分开；不直接接受“逐点后验不可能排序”。

## [P01] EvalMuse-40K / FGA-BLIP2 / PN-VQA
来源：Han et al., arXiv:2412.18150v2;正式版AAAI2026。
入口：https://arxiv.org/html/2412.18150v2
核对范围：方法、数据及标注节。
解释边界：整体L1+元素监督已有；3–6次标注与重标注选择性。

## [P02] GenAI-Bench / GenAI-Rank
来源：Li et al., arXiv:2406.13743。
入口：https://arxiv.org/html/2406.13743
核对范围：数据和排名协议。
解释边界：固定候选的人类评价；不是新任务。

## [P03] VQAScore
来源：Lin et al., ECCV2024, arXiv:2404.01291。
入口：https://arxiv.org/abs/2404.01291
核对范围：论文摘要和项目在仓库中的协议。
解释边界：逐图概率分数可以完成best-of-N，不接受点式必败。

## [P04] NIMA: Neural Image Assessment
来源：Talebi and Milanfar, IEEE TIP 2018。
入口：https://doi.org/10.1109/TIP.2018.2831899
核对范围：公开方法段、EMD公式。
解释边界：评分分布、CDF损失和平均分预测已有；原任务不是图文对齐。

## [P05] Strictly Proper Scoring Rules, Prediction, and Estimation
来源：Gneiting and Raftery, JASA 2007。
入口：https://doi.org/10.1198/016214506000001437
核对范围：原始文章/作者技术报告摘要；标准RPS部分按定义推导。
解释边界：正规评分、RPS、Bayes风险；不声称新的概率评分原则。

## [P06] CORAL: Rank-consistent ordinal regression
来源：Cao, Mirjalili, Raschka, PRL 2020。
入口：https://arxiv.org/abs/1901.07884
核对范围：原始摘要、作者引用页。
解释边界：阈值分解、等级一致性已有；注意不同阈值先验。

## [P07] CORN: Rank-consistent ordinal regression based on conditional probabilities
来源：Shi, Cao, Raschka, arXiv:2111.08851。
入口：https://arxiv.org/abs/2111.08851
核对范围：原始摘要和作者实现入口。
解释边界：条件风险集与链式概率已有；不混为无条件阈值BCE。

## [P08] RankNet: Learning to rank using gradient descent
来源：Burges et al., ICML2005。
入口：https://doi.org/10.1145/1102351.1102363
核对范围：出版摘要。
解释边界：相同提示的配对排序强对照；不是概率校准基线。

## [P09] ListNet: From pairwise to listwise learning to rank
来源：Cao et al., ICML2007。
入口：https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/
核对范围：原作者研究页摘要。
解释边界：列表top-one分布已有；本包只是图文任务适配。

## [P10] Relative Predictive Coding
来源：Tsai et al., arXiv:2103.11275v3。
入口：https://arxiv.org/html/2103.11275v3
核对范围：基础目标与后验关系。
解释边界：平衡RPC与J同一；应用换成阈值不消除重合。

## [P11] RuLSIF: Relative Density-Ratio Estimation
来源：Yamada et al., NeurIPS2011。
入口：https://proceedings.neurips.cc/paper_files/paper/2011/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html
核对范围：原始论文摘要与项目已有同目标实现。
解释边界：强传统密度比路线，不只用弱KDE。

## [P12] Rich Human Feedback for Text-to-Image Generation
来源：Liang et al., CVPR2024。
入口：https://openaccess.thecvf.com/content/CVPR2024/html/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.html
核对范围：官方摘要与数据任务。
解释边界：外部对齐/丰富反馈；不强行将所有字段映射为1–5评级。

## [P13] HPSv3: Towards Wide-Spectrum Human Preference Score
来源：Ma et al., ICCV2025。
入口：https://arxiv.org/abs/2508.03789
核对范围：原始摘要。
解释边界：偏好与不确定性排序已有；审美/整体偏好不等于文字对齐。

## [P14] TIFA
来源：Hu et al., ICCV2023。
入口：https://arxiv.org/abs/2303.11897
核对范围：沿用V7已审官方接入；本次不重读全文。
解释边界：本地官方路由与API路由区分。

## [P15] Davidsonian Scene Graph (DSG)
来源：Cho et al., ICLR2024。
入口：https://proceedings.iclr.cc/paper_files/paper/2024/hash/446905abfec9db8856f7f7465bbd5552-Abstract-Conference.html
核对范围：沿用V7已审官方接入；本次不重读全文。
解释边界：依赖解析不是新方法；本地LLM替换单列。

## [D01] EvalMuse public data
来源：DY-Evalab/EvalMuse。
入口：https://huggingface.co/datasets/DY-Evalab/EvalMuse
核对范围：本次使用仓库数据审查，不重新下载标签。
解释边界：官方test的可评分性/许可按封存协议核验。

## [D02] GenAI-Rank public data
来源：zhiqiulin/GenAI-Image-Ranking-800。
入口：https://huggingface.co/datasets/zhiqiulin/GenAI-Image-Ranking-800
核对范围：仓库已固定图像/标签分离。
解释边界：800提示、两生成器各9图；本包未读人工标签。


## 交付增量来源
以下增量晚于上列设计快照。

## [G08] TIFA本地问题无标签统计
入口：https://github.com/W-Yinghao/VCS_QMI/commit/4f90d6dd68c3c8d0977b6eab30731abc03780c4f
核对范围：完整commit diff；31解析失败、33空问题提示。不是人工评价结果。

## [G09] REV9评价打包变更
入口：https://github.com/W-Yinghao/VCS_QMI/commit/2d38ea35e8e6e178f580e5bbb052fc37e05230c7
核对范围：完整commit diff；报告声明配置/shard/evaluator不变。不是实时队列。
