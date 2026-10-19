# V8.1 服务器完整方案 v1

已有模型诊断 → 有证据的有限修订。源码审查基线 a87b2d5，交付更新 580781f。



---

# V8.1 · 服务器诊断与条件修订包 v1

**任务：确认 V8 为什么在不同 utility 损失下得到相近选图质量，再决定是否值得做一项有限改动。**

本包不是 V9.1，也不是 V10。V8 原始 S/J、评分分布、正式模型与确认协议保留；V9-N2/G1 继续现有计划。本包第一轮新增完整训练数为 **0**。

## 快速入口

| 入口 | 作用 |
|---|---|
| [Agent 接入消息](01_SERVER_AGENT_PROMPT_CN.md) | 可直接发送给服务器 |
| [主任务书](docs/02_SERVER_PLAN_CN.md) | 任务范围、先后与交付 |
| [证据与源码审查](docs/03_EVIDENCE_AND_CODE_CN.md) | 当前结果、epoch 0、更正范围 |
| [诊断详细规范](docs/04_DIAGNOSTICS_CN.md) | 统一数据、指标、读法 |
| [统计与数学对象](docs/05_METRICS_AND_THEORY_CN.md) | 误差分解、并列、分布风险 |
| [条件实验](docs/06_CONDITIONAL_EXPERIMENTS_CN.md) | 有依据后只做一个小型完整应用改动 |
| [数据与并行](docs/07_EXECUTION_AND_EXPOSURE_CN.md) | 保护V8-05与V9，不重新打开许可问题 |
| [参考代码使用](reference/README_CN.md) | 只读适配、诊断、单元测试 |
| [来源](sources/SOURCE_INDEX_CN.md) | 原始链接、核对深度 |
| [完整离线全稿](FULL_PLAN_CN.html) | 集中阅读全部正文 |

## 基线

- main：`55d47c9fe73375230c86a7ec1b2e575bdaeb19d5`。
- V7：`2dc39b5f9892aba828ea6d857e40dfd8120cbc74`。
- V8：`a87b2d53380783db0932b144dddcd368ac4a7b2d`。
- V9：`1ce0e84012ff8c34524deeb42a967a80cb2dcaf4`。

**交付前更新：V8 已到 `580781f`，850提示确认结果已提交；原源码审查仍适用。详见[增量说明](docs/09_DELIVERY_UPDATE_CN.md)。**

这是已提交记录的截点，不是服务器队列快照。接入时如有更新，先写增量；已有诊断则复用，不重复发作业。[R1–R8]

## 授权边界

收到本包后可立即做：只读盘点、代码/日志核对、FIT/CAL/DEV诊断、缺失的开发侧推理、参考测试和报告更正。应遵守当前账户调度与资源上限。

不由本包自动授权：重新训练完整V8、扩学习率网格、修复排序臂的性能搜索、调用付费API、增加数据集、读取新的外部标签、修改V8-05或V9作业。条件训练提交前须返回方法、预算和对照表供负责人决定。

本包是实验指令和可运行的CPU参考工具；不包含真实模型权重、数据标签或已完成自然数据诊断。


---

# 给服务器 Agent：执行 V8.1 v1

请先读 `docs/09_DELIVERY_UPDATE_CN.md`：V8的850提示确认已在 `580781f` 提交，不再等待，也不重跑。然后执行 **V8.1：已有 utility 模型诊断与条件修订**。不是要求立刻提交新的完整训练，也不是重新寻找一个VCS超过BCE的配置。

## 现在执行

1. 核对 main/V8/V9 HEAD、48个原V8单元及24个正式导出；读取当前V8-05执行状态，但不要重复提交确认作业。将新提交和已有对应诊断写入 `V81_00_INTAKE.md`。
2. 按实际作业快照核对 `epoch=0`。本次读取的循环先完成训练、再CAL和保存，所以它是第一轮训练后的checkpoint，不是初始化。记录steps、images_seen、成功optimizer update与snapshot；只更正解释，不改旧数字。
3. 复用24个模型的CAL_SELECT/CAL_PROB/DEV预测，做任务表D02–D05：同提示选择一致性、排名、MSE分解、逐阈值/分布风险、训练曲线与选模对齐。统计对象与数据分母见docs/04–05。
4. 用FIT上由ID固定的审计子集做有限的输入依赖和梯度检查（D06）；不把错配输入的原标签继续当成真值。不能用这些人为干预的分数直接称“语义真值”或“因果归因”。
5. 主比较保留公开FGA、VQAScore、TIFA/DSG及已有核路线；BCE只是一个归因对照。原生分数非1–5量表者不计算未经校准的均值MSE。
6. 交 `V81_08_PHASE_REPORT.md`：结果分别标为已核事实、推导、诊断证据、待验证假设。判断是选模错位、适配过度、视觉区分不足、仅刻度差异、混合原因或证据不足。

## 分支、数据与资源

建议独立worktree `v8.1/utility-diagnostics`（若已有对应分支直接复用）；报告 `reports/V8_1_DIAGNOSTICS/`，输出独立 `outputs_v81/`。禁止覆盖原 `outputs_v8` 及V9；不取消/抢占已有V8-05、REV10、G1/N2任务。沿用负责人“现有数据可用”的决定，不再等待新的许可证明。

开发诊断只用公共EvalMuse TRAIN里既定FIT/CAL/DEV。850非Rank提示与TIFA160继续由原V8-05流程处理；V8.1不打开这些标签，也不以其结果选择改法。已读GenAI-Rank的报告可引用，若以后新增逐例分析需明确重复使用、另行批准；默认本包不读其标签。

## 暂不执行

不启动V10；不扩大八种损失网格；不因低沿LR或epoch 0就重新训练；不借D1的dropout问题推断V8也有同一bug；不只因VCS≈BCE就终止外部对照。

条件训练仅在D00–D06完成后提出。优先在同样完整评分路径中只改变适配范围：CS/RPS/MSE × 3种子；或在已有各epoch checkpoint充分时做独立标记的选模比较。两条不自动并行，不用确认标签选方案，预算批准前不提交。

首轮交付必须包括可追溯JSON/CSV与Markdown，不只是结论文字。运行本包测试后再接实际导出；它们验证的是参考公式和实现，不是自然数据效果。发现本包假设与实际代码不符时，先给证据再修改，不照单执行。


---

# 02 · 主任务书：从相近总分定位实际瓶颈

## 1. 本轮问题

V8已经把元素后验改为总体评分分布。完整训练和GenAI-Rank评价显示：U-CS在SDXL上优于部分外部方法，在DALLE_3上落后VQAScore；同条件有效训练配方没有检测到选图差异。[R2]

下一步不是再换一个任务，而是回答三个问题：

**A. 各方法是否选了同一张图？** 平均人评分相近可能来自相同决策，也可能来自不同成功/失败的抵消。

**B. 模型学到的是绝对评分刻度，还是同提示候选的质量差？** 总体MSE不自动等于选图效果。

**C. 优化、适配范围和视觉证据读取，哪个限制了结果？** 必须用同一数据、对应checkpoint和可解释干预区分，不能凭一个学习率边界就宣布原因。

本包将结果作为下一次决策依据，而非预定某一个原因必须成立。

## 2. 任务表

| ID | 工作 | 主要输入 | 输出 | 默认新训练 |
|---|---|---|---|---|
| D00 | 版本/单元/标签状态盘点 | 48个unit日志、24个正式导出、V8-05台账 | inventory.csv、intake.md | 0 |
| D01 | epoch语义、权重状态和源代码核对 | 实际job snapshot、log/best/resume元数据 | epoch_audit.csv、correction.md | 0 |
| D02 | 同提示决策与排序一致性 | CAL/DEV预测；公开方法已有预测 | agreement矩阵、逐prompt明细 | 0 |
| D03 | 绝对刻度与候选区分分解 | 同一候选池预测+投票均值 | MSE三项恒等核对、排序/选图 | 0 |
| D04 | 分布/阈值/不确定性读数 | q、u、mu、原始投票、FIT先验 | distribution.csv、变换与加权核对 | 0 |
| D05 | 训练曲线/选模/原生排序行为 | 每epoch日志、实际存在的checkpoint | curves.csv、selection_gap.md | 0 |
| D06 | 有限输入依赖与梯度诊断 | FIT-AUDIT、选定checkpoint、原损失 | input_checks.csv、gradient_checks.csv | 0 |
| D07 | 对外位置与成本保持 | 已冻结系统结果、标准计时负载 | benchmark_context.csv | 0 |
| D08 | 下一步裁决 | 全部诊断+局限 | phase_report.md、conditional_request.yaml | 0 |

D00/D01先做；D02–D05可并行。D06用同一批ID，不按错误个例手挑；补推理前给一次标准batch实测，规模见执行文档。D07只整理原外部结果与开发预测，不自动调用新的外部模型。

## 3. 结果的科学意义

| 现象 | 可以支持 | 不能推出 |
|---|---|---|
| 各臂Top-1几乎相同 | 当前损失变化未有效改变决策 | 所有点式后验不能排序 |
| 选择不同、均值相近 | 错误/收益有异质性 | 自动存在可部署的最佳融合 |
| 校准MSE改善、中心化误差无改善 | 主要修复刻度 | 估计器无用途或理论错误 |
| 原生rank loss下降而MSE不佳 | 排序与回归目标有分歧 | 排序臂已成功，或训练实现一定正确 |
| FIT改善、CAL退化 | 过拟合是候选解释 | 已证明预训练语义遗忘 |
| 错图仍同分、视觉梯度很弱 | 模型可能缺少视觉响应 | 图像没有有效证据或必须更换骨干 |

新诊断均是开发分析，不伪装成预注册确认。需要改变方案时给出一项最小可区分改动，保留原方法与强外部对照。

## 4. 与V8/V9的关系

V8-05确认继续按既定规则完成，V8.1不追加其比较、不修改其阈值或标签分母。V9-N2/G1与REV10按现有授权运行；本包不增加它们的权重或梯度任务，避免跨包蔓延。V8.1不等确认结果变正，先在开发侧完成诊断。

## 5. 三个完成条件

第一，能够从来源JSON重生成所有表格，并明确哪些checkpoint/预测不存在。

第二，每项原因都有支持与反对证据，区分相关性与干预；至少说明公开强方法相对我们的差距，而非只给CS/BCE表。

第三，提交“保持现状收尾／实施一项修复／申请一组有限新训练”的明确建议。没有足够依据时允许结束本轮，不以增加版本号替代科学问题。


**交付状态更新见 docs/09：V8-05确认已完成。上述协议保护不变；不再把等待该作业作为前置条件。**


---

# 03 · 证据、源码与需核对的解释

本节分三类。**已提交结果**复述来源，不是本包重新计算；**源码事实**以本次读取提交为准；**影响解释**尚需真实job snapshot和产物检验。

## 1. 已提交V8结果 [R2]

| U-CS对外差值：N9选中图像人工评分 | DALLE_3 | SDXL_Base |
|---|---:|---:|
| FGA released | −0.025，Holm未显著 | +0.039，Holm p=0.046 |
| VQAScore CLIP-FlanT5-XXL | −0.121，Holm显著 | −0.046，Holm未显著 |
| TIFA | +0.043，Holm未显著 | +0.130，Holm显著 |
| DSG本地适配 | +0.026，Holm未显著 | +0.147，Holm显著 |

这些是1–5量表的评分差，不是百分点。“未检测到差异”不能写为等效；RANK/MSE/CS不同风险尺度不能直接比较训练loss数值。

八臂×三种子已导出。五臂选到原LR网格下沿；这说明被测优化范围有限，不证明更低LR一定改善。U-RANK的CAL均值MSE约0.52–0.60，其他约0.40；排序臂在该回归选模规则下表现差，但应核对原生排序行为。[R1,R2]

## 2. 重要更正：epoch 0不是pre-update模型 [R3]

本次源码顺序：`for epoch ...` → 遍历完整训练batch → `predict`和CAL → 保存`best.pt`/`resume.pt`。因此，数组索引0代表第一轮完成后的模型。

原报告“epoch 0是带自然直方图bias的初始化head”与该循环不相符。提交一份证据对照：

| 字段 | 必须记录 |
|---|---|
| source_commit / snapshot_hash | 真实运行使用的源码，不只当前HEAD |
| epoch_index | 原始0-based值 |
| completed_epoch_count | 经日志核对后为epoch_index+1，不改旧字段 |
| planned_batches / attempted_steps | 原计划与日志步骤 |
| successful_updates | 若AMP历史没记录，填unknown，不以计数代替成功更新 |
| images_seen | 对应轮次是否完整，是否smoke或被截断 |
| resume_history | 从哪轮恢复、是否有数据/随机态缺失 |
| checkpoint_stage | INIT_PREUPDATE / AFTER_EPOCH / RECONSTRUCTED_INIT / UNKNOWN |

仅更正报告解释时，不改变数值、选中的epoch或主结论。若job snapshot实际不同，以快照为准，说明本包核对范围。

要比较真实初始状态，可用相同上游权重、seed、FIT直方图重新构造 **RECONSTRUCTED_INIT**。没有存下原初始化权重就不能冒称bitwise复现；未经训练的随机质量头不是强评分基线。

## 3. 保存策略限制 [R3]

源码持续覆盖`best.pt`与`resume.pt`；每epoch日志被保留，不能据此推定每epoch权重或逐图预测都在。先列实际文件：

- 存在某epoch checkpoint：可以重新推理开发角色，记录来源。
- 只有该epoch的聚合日志：只能做曲线分析，不能计算其逐promptTop-1。
- 只有最终best/resume：不启动整轮重训来“补历史”，缺项报告为not_saved。
- 没保存RNG的恢复路径：不声称精确配对重放已成立；未来条件实验必须验证恢复一致性。

## 4. 不把D1的问题搬到V8 [R3,E2]

V8训练时`model.train()`且冻结视觉骨干转eval；预测时`model.eval()`。D1曾出现的phase-B冻结critic/dropout问题不构成V8有同一错误的证据。

本轮检查的是实际freeze mask、模块mode和AMP成功更新。诊断梯度时允许eval-mode自动微分；不要对需要梯度的路径套整体`no_grad()`。推理重读则保持全部eval，且不写回任何训练状态。

## 5. 排序项与尺度 [R4]

U-RANK = 归一化均值MSE + pairwise BCE；U-LIST = 同一MSE + listwise CE，tau=1、lambda=1。无信息人评分pair在训练中可作为0.5 target，而评价通常排除同分pair。这是原设计，不直接判bug。

核对batch含多少有效subgroup/pair、点项和排序项的梯度量级/夹角、mu是否饱和、同一prompt是否跨组、归一化是否与R1一致。不能从“rank loss数值更大”推断梯度主导，也不因梯度不同就自动修改lambda。

## 6. 确认集与数据决定 [R6–R8]

V8-05已提交1042053，850非Rank提示×6生成器，原主指标是pooled Spearman，另有同提示pairwise；不是N9。其主模型是三种子mu均值，另报逐seed。这与V8-04的逐seed选图均值不同，不能混合成同一估计量。

main已记录负责人现有数据使用决定。本包沿用，不再建立申请/许可核查的实验门槛。标签隔离仍保持：数据可用与允许用确认标签调参是不同问题。


**交付状态更新见 docs/09：V8-05确认已完成。上述协议保护不变；不再把等待该作业作为前置条件。**


---

# 04 · 诊断实施规范

## D00：输入与键

默认使用原公共EvalMuse TRAIN拆出的 **CAL_SELECT、CAL_PROB、DEV**。FIT仅用于固定审计样本和实现检查。

唯一记录键：`dataset + role + method + seed + checkpoint_id + pool_id + image_id`。`pool_id`定义候选集合：EvalMuse按prompt；GenAI-Rank若仅做无标签一致性，按prompt×generator、每池9图；GenAI-Bench六生成器池只能由V8-05原流程评价。本包不把它们混为一种池。

每个pool另有`prompt_id`与`cluster_id`。优先沿用V7近重复连接成分作为重采样cluster；缺成分信息时用prompt并披露。相同pool在所有方法中必须候选ID一致、顺序一致；缺预测默认报错，不静默求交集。外部方法原协议的特殊缺失规则仅在其原表保留。

三种子逐个报告。**先平均分数再选图**与**先各选一图再平均质量**不是同一系统，必须另起method ID。主诊断是逐seed、再报告指标均值，不默认产生ensemble。

## D02：平均质量相近是否代表同一选择？

对24个正式模型的开发输出计算全部8臂同seed两两组合（28×3），形成矩阵；跨seed稳定性另列。公开FGA、VQAScore、TIFA/DSG若已有同角色预测，则加入相同表。传统KDE/RuLSIF/Nyström仅加入已有可比角色/特征条件，不额外再拟合。

每pool报告：

| 指标 | 定义 |
|---|---|
| exact Top-1 agreement | 相同的冻结候选顺序、相同的确切并列规则下是否选同一image_id |
| top-set overlap / Jaccard | 两模型所有最高分候选集合是否相交；避免恒定评分器在first-index下假装高度一致 |
| within-pool Spearman / Kendall tau-b | 平均名次；常数序列→null并记录原因，不填0或1 |
| first/second score gap | 原生量纲，跨方法不直接比较；可补within-pool IQR归一化，IQR=0则NA |
| different choice quality | 只在开发标签上：A选图与B选图的人均评分差；同时报全pool和不同选择pool |
| swap decomposition | 总体差 = 不同选择pool比例 × 这些pool的平均差（同选择时差为0）；核对代数残差 |

“最大互补收益”若计算，只能称在已固定候选字典上使用真标签选择的参照，不是可部署融合，不据此拟合确认集router。

## D03：误差分解与任务效用

主读数统一为prompt宏平均：同prompt所有有效图像等权，同prompt一票。另报image-micro和R1的natural-weight结果；不能选择分母较有利的列。

对有真实评分量纲的mu：e_i=mu_i−Hbar_i，分解`MSE = squared_mean_error + centered_error`。同时报告人均评分、模型分数的within-pool方差、同提示非并列pair accuracy、Top-1 regret、选择质量。pool仅一图时MSE仍有效，排序无信息。

**有标签居中后的mu只用于误差分解，不能导出为改进方法。**开发集群体拟合一个简单线性标定也不改变同pool排序，但属于新的次级分析；当前主结果不用其替换。

外部VQAScore概率或TIFA分数不是1–5人工评分；未在CAL独立学到固定映射时，均值MSE列应为NA。排序/选图比较照常。

## D04：分布质量与评分信息

V8 q为5档概率、u为4个尾概率、mu=sum(h*q_h)。先核对三者一致，原始votes为3或6票；每图权重不因票数多而增加。

报告随机标注者RPS、对投票频率的CDF平方误差、两者的常数差、histogram交叉熵、mu对标注均值的MSE，以及各阈值风险。使用R1 FIT先验，禁止从DEV/确认标签重新计算先验。

同时检查平均分接近是否掩盖不同q：分布L1、尾概率距离、预测评分方差。评分方差是对标注评分分布的预测，不是mu的置信区间，更不能直接称模型 epistemic uncertainty。

可使用固定规则按票数、均值区间、prompt长度及FIT定义的语义类别分层。样本分层只作描述；分歧投票图像不得从主评价删除。

## D05：训练到底发生了什么？

先复用所有24调参+24正式log，不重跑。保留每epoch的尝试更新数、图像数、训练原生项、CAL MSE/RPS/CE/pair accuracy及所选epoch。不同损失的数值只能各自随时间看，跨损失比较在共同CAL指标上进行。

FIT-AUDIT：按`sha256('v81-fit-audit-2718|'+prompt_id)`排序，取前128个prompt及其全部有效图片；超过2048图时保留完整prompt直到上限，不截断prompt内部。由输入ID冻结，不按loss或错误挑选。它是训练内的审计样本，不是泛化集。

有现成checkpoint时，使用相同eval模式重读FIT-AUDIT与CAL_SELECT；优先best与最终resume，再现成其他epoch。最多新增24正式模型best/resume各一遍开发推理；无必要时可复用已经存在的q/u/mu。真实初始化仅每seed重构一次，因为八臂初始结构共享；记为RECONSTRUCTED_INIT。

选模诊断分两级：

1. **日志级**：比较原MSE选epoch与CAL内pair accuracy最佳epoch。只是指出可能错位，不据此宣布外部提高。
2. **checkpoint级**：只有权重齐全时，才提出按预定CAL规则另选checkpoint并在DEV评价，标签为post-hoc development；不得回读V8-05给新选择打分。权重不齐全就停在第一级，不自动重训补轨迹。

## D06：critic是否使用了视觉信息？（有限实现检查）

使用FIT-AUDIT，固定下列干预：正常配对、图像在不同prompt间固定置换、相同prompt改变候选图像、可选全灰图。正常与置换路径同样eval，不改变预处理或文本截断。[R3]

对于错图/错文本，不沿用原图评分作正确标签。主要记录预测变化、分布距离、正常配对的人评分排序，以及对输入/中间视觉表示的梯度响应。固定置换仅是依赖检查，不是新的语义真值数据。

U-RANK/U-LIST另做同batch梯度检查：point/rank梯度各自范数、夹角、非有限状态、有效pair数；在head和实际可训练Q-former分别读数，不把head梯度替代整个网络。最多20个固定FIT batch，不更新正式模型；smoke若需要临时一步更新，必须使用独立拷贝且不导出正式checkpoint。

本任务能够指出“缺少响应”的证据，不能仅凭梯度小就断言模型没有视觉推理或解决方法一定是更大模型。

## D07：保持外部主表

结果至少同时显示原FGA、VQAScore、TIFA/DSG和V8各臂。原生宿主不同就在表头写出，不将强主干优势归给拟合损失；现有Table C核路线采用未微调表示，与完整微调网络的不同单列。无公开预测的角色填missing_external_export，不为凑表生成假数。

成本用标准工作负载计时：同设备、warm-up、固定batch/图像分辨率/提示长度，记录整图和候选池总时间及峰值显存。只重用可比历史计时；本包不让GPU空闲成为新增方法的理由。


---

# 05 · 数学对象与统计规范

## 1. 原理论不变，应用不偷换

原工具：

```text
M = (P + Q) / 2
eta = (p - q) / (p + q)
S = E_M[eta²]
J(T) = E_P[T] - E_Q[T] - 0.5 E_P[T²] - 0.5 E_Q[T²]
S - J(T) = E_M[(T - eta)²]
```

这是合作者的基础定义。[T1] 在V8中，每个P_k/Q_k对应人工评分是否达到阈值k，而不是自然图文joint/product；本包不把它重新命名为Shannon MI。精确gap是任意合适T的读数关系，包括BCE拟合后转换的T，不是平方拟合的独占证书。[T2]

V8自然尾概率u_k经过FIT先验pi_k转换：

```text
ell_k = logit(u_k) - logit(pi_k)
T_k   = tanh(ell_k / 2)
mu    = 1 + sum_k u_k  =  sum_{h=1..5} h q_h
```

S/J和经验的mu-MSE、RPS分别报告，不能因全部是平方项就混为同一个目标。

## 2. 本包直接推导：MSE分解

对一个固定候选池，非负权重v_i归一化为sum v_i=1，e_i=mu_i−Hbar_i：

```text
b = sum_i v_i e_i
MSE_pool = sum_i v_i e_i²
bias_part = b²
within_part = sum_i v_i (e_i - b)²
MSE_pool = bias_part + within_part
```

这是精确代数恒等式，不是新方法。普通prompt宏平均使用每池uniform v；若使用别的权重，分解三项必须完全一致。within_part也不是排名指标：正的线性尺度误差能增大它而不改变排序，所以必须并列报告pairwise与regret。

若真实尾概率为u*，则由Cauchy–Schwarz：

```text
(mu - mu*)² = (sum_k (u_k-u*_k))²
             <= 4 sum_k (u_k-u*_k)²
             = 16 mean_k (u_k-u*_k)² .
```

该界针对真实概率误差，不把有限标注投票均值当作已知u*。它也不是排名胜出的保证。

## 3. RPS三种对象

给定当前图像的投票频率y_k，预测尾概率u_k：

```text
random-rater RPS = mean_k [y_k(1-u_k)² + (1-y_k)u_k²]
CDF-fit squared  = mean_k [(u_k-y_k)²]
constant         = mean_k [y_k(1-y_k)]
random-rater RPS = CDF-fit squared + constant
```

constant在相同图像与相同分母下不影响方法比较，但影响绝对值。完整直方图CE = -sum_h hist_h log(q_h)，零概率仅为数值处理而截断，记录epsilon；投票计数3/6不改变每图自然权重。

## 4. 排序、并列与选择

- 相关性：Spearman用average ranks；常数序列或只有一个值返回null并标记。参考工具不产生传统大样本p值。[E1]
- 人类均值同分pair从pair accuracy分母排除；模型同分在有信息pair上记0.5。显式给出informative_pairs和零信息pool数。
- 选图主诊断用明确候选顺序。按模型原始全精度分数取argmax，严格相等才并列；不得把显示为三位小数相等当成真并列。敏感性可用固定阈值形成near-tie集合，但不替换原读数。
- regret = 当前候选池中最高标注均值 − 被选图标注均值。这里的最高者是有限标注参照，不是无噪声总体oracle。
- 阶段D02的pairwise agreement是两模型的顺序一致性，不是对人类顺序的准确率，表名不能混用。
- temperature作用在五类logit上能改变不同图的mu顺序；不要以为它总是单一score的单调变换。本包主诊断使用raw导出，已有校准列另标。

## 5. 统计单位和不确定性

主单位是prompt或已知近重复component；同prompt多个生成器/图片、同一图多个投票不是独立样本。逐seed报告；三种子不足以精确估计训练随机性，因此不把所有图片×种子当成巨大独立样本。

默认每个seed给完整level；方法配对差按共同cluster bootstrap（2000次、seed2718）估计区间。可以对每个cluster先平均三种子的指标再重采样，解释为在这组三种子固定条件下的prompt抽样区间；不等同重新训练的总体不确定性。

参考程序对可加的pool指标实现配对cluster bootstrap；不把“平均各prompt相关系数”称为全局Spearman。全局相关系数区间如需要，须在每次cluster重采样后重建成对行并重新排序，另写测试。

本轮区间属于开发诊断。默认不生成显著性标签或多个未经调整的p值，不以某个95%区间排除0自动批准新方法。既有V8-04/V8-05的正式家族与指标不被本包改变。

## 6. 预期与否定范围

我们不预期某个损失一定最好。假设H-A“只有刻度变化”、H-B“选模指标错位”、H-C“适配损害泛化”、H-D“输入/容量不足”分别收集支持与反对证据。可以多因素并存，也可以证据不足。

任何小修的收益首先归于被改变的组件；若CS与RPS/MSE同样受益，就不能说是CS专属。基础目标与RPC/相对密度比关系沿用已更正材料，不在诊断包中重新认领新颖性。[T2]


---

# 06 · 条件实验：一次只改变一个已定位因素

**本文件是待审批的下一步备选，不是收到包就提交训练。**每项均须完成D00–D06、写出触发证据和实测预算；负责人选择一条或选择收尾。

## C0：实现或解释修复

若仅发现epoch文字错误：更正报告、保留数值，不重训。

若发现冻结选模规则没有按原代码执行、损失公式/数据权重有实质错误：列出受影响unit和不受影响unit，制定最小修复重跑；标记为原实验修复，不称新方法。低沿学习率和小梯度本身不是实现bug。

## C1：适配范围干预（优先候选，但不是预定答案）

适用证据：相同eval负载下，早期学习后FIT继续改善、CAL候选区分退化；或有可核对的原模型/适配模型视觉响应差异。不能仅以外部VQAScore更强来推断遗忘。

### 条件完整实验

| 项目 | 规定 |
|---|---|
| 目标 | U-CS、U-RPS、U-MSE，覆盖CS、非CS分布风险、直接效用回归；BCE有原结果，不成为唯一对照 |
| 主干与输入 | 原FGA-BLIP2 COCO构造、364分辨率、全部既定FIT、同一5档head与FIT先验 |
| 唯一设计改变 | 原全Q-former微调 → 只训练最后2个Q-former transformer block、原ln_vision、query tokens与head；其余Q-former固定且eval |
| 参数清单 | 运行前导出名称/shape/数量；block路径从真实module解析，禁止只凭字符串猜索引 |
| 日程与选模 | 5 epochs、原组batch计划、同seed初始化、沿用每目标原选定LR、CAL_SELECT原MSE规则；不同时改排序选模、温度或损失 |
| formal seeds | 0、1、2全部完成，不按seed0筛选 |
| 新增单元 | 3目标×3种子 = 9个受限适配单元；原9个完整适配匹配结果可复用 |
| 局限 | 是固定recipe的适配范围干预，不是各适配方案充分调参后的最优性能比较 |

这不是只对冻结embedding跑探针：全部图像与文本通过完整多模态模型，所选跨模态block训练，最终导出完整评分系统。若要改为head-only或LoRA，是不同干预，需说明理由，不能同时替换。

只有历史原条件无法严格复用时，申请最多9个匹配原条件重放；理由不能是“需要一个更好看的baseline”。新的调参网格另行决策，不能隐含在这9个正式单元内。

评估首先在DEV作为新开发结果，报告对原全适配、公开FGA/VQAScore及适用核路线的差值与成本。原V8-05确认集不用于选改法，也不自动成为C1确认集。需要新的外部检验时，先锁定方案和用途。

停止条件：适配干预没有改善同提示候选区分或实际选图，则不再默认加入蒸馏、残差、全新骨干一起救该支线。

## C2：选模准则对齐（可以零新训练，但需要完整checkpoint）

适用证据：CAL原生排序持续改善，而MSE选模系统性挑了不同epoch；有实际保存的候选checkpoint。

固定两种规则：原CAL MSE，和CAL prompt宏平均pair accuracy（并列优先较低MSE、再早epoch）。不在GenAI标签上选规则。每臂各seed按同一规则处理；DEV报告新开发比较，不覆盖原表。

缺epoch checkpoint时，只保留日志级分析；禁止用新训练补出历史轨迹却标为原实验的重读。此时是否执行新的轨迹保存复验另行申请。

## C3：视觉输入/结构改动——本轮只出设计卡

若全部损失在相同细粒度条件失败，且输入诊断支持有效视觉信息未被读取，再提出一个具体输入/交互改变。当前模型已有Q-former cross-attention，“再加cross-attention”不是完整设计。须明确新增哪个观测、训练多少参数、哪些基线得到相同信息、评价何种视觉能力。

本包不规定换更大VLM，不新增数据，不将C3命名为V10。只有提出了超出损失替换的明确方法增量，才考虑下一版本。

## 统一申请格式

填写templates/CONDITIONAL_REQUEST.yaml：证据、反证、唯一改动、完整单元表、每GPU型号实测时间、teacher/导出/评价成本、原对照复用依据、数据暴露、失败后的停止规则。负责人批准前保持`approved: false`。


---

# 07 · 数据、执行与并行

## 1. 不触碰既有主结果

新报告建议`reports/V8_1_DIAGNOSTICS/`；新输出`/projects/EEG-foundation-model/yinghao/outputs_v81/`，实际可写路径由接入时确认。读取V8目录，但不修改best.pt、log、pred、confirm产物。

新分支建议`v8.1/utility-diagnostics`。若服务器已开始对应审查，复用其分支和编号。main/V8/V9的来源、协议和结果仍由原路径维护；更正报告采用明确附录/erratum，不覆盖原数值文件。

## 2. 标签用途表

| 数据 | V8.1 默认可做 | 默认不做 |
|---|---|---|
| EvalMuse FIT | 固定审计子集、原生损失/输入诊断 | 从FIT性能声称泛化 |
| CAL_SELECT | 原epoch/lr选择复核；新的选模假说只能另列 | 用新规则改原V8主模型 |
| CAL_PROB | 既有温度的次级核对 | 根据DEV重新拟合温度 |
| DEV | 本次开发诊断、原方法比较 | 伪称新的独立确认 |
| GenAI-Rank | 引用已发布聚合结果；需要时做明确的无标签预测一致性 | 默认不新增逐例标签读取或据此调方法 |
| GenAI-Bench 850非Rank | 等原V8-05固定报告，用于原V8结论 | V8.1的模型/阈值/结构选择；参考CLI拒绝此角色 |
| TIFA160 | 沿用V8-05已授权次级流程 | 额外新标签读取或改变用途 |
| V9/REG/SSL数据 | 不用 | 变成V8.1新训练集 |

即使确认报告在执行中到达，V8.1改动理由也应来自原开发诊断。记录设计卡时间；若团队已受确认具体错误启发，应如实标记暴露，不能自称完全独立。

## 3. 第一轮资源上限

新增完整训练0。CPU优先；现成预测优先；缺失推理上限为24正式模型的best与最终resume在固定FIT-AUDIT及CAL/DEV必要角色上的推理，每模型路径先去重。先在相同GPU类型测标准batch吞吐/峰值显存，填预算，不套用旧L40S/RTX时间作保证。

D06梯度检查最多20个固定FIT batch，先运行CS/RANK/LIST的实现检查；它不是训练网格。对任一临时模型的改动只发生在隔离输出，绝不写回原权重。

配额与nice遵循项目现有决定；本包不新设抢占优先权。确认评价和已批准V9/REV10作业不取消、不重排。没有权重的epoch不自动重训。

## 4. 时间与软件状态

来源截至2026-10-15的指定提交。记录服务器实际UTC、设备、torch/numpy/scipy版本、job snapshot和数据hash，不升级环境来迎合本包参考依赖。参考代码只需CPU numpy/scipy；pytest用于测试。

已有数据按负责人决定使用，保留来源事实，不新建许可阻塞项。本包不含第三方数据、模型权重或字体文件，也不授权再分发它们。[R8]

## 5. 交付结构

```text
reports/V8_1_DIAGNOSTICS/
  V81_00_INTAKE.md
  V81_01_EPOCH_AND_CODE_AUDIT.md
  V81_02_DECISION_AGREEMENT.md
  V81_03_SCALE_VS_SELECTION.md
  V81_04_DISTRIBUTION_AND_TRAINING.md
  V81_05_INPUT_GRADIENT_CHECKS.md
  V81_06_EXTERNAL_CONTEXT.md
  V81_08_PHASE_REPORT.md
  results/*.json, *.csv
```

机器结果字段保留NA原因、每列样本数、元数据、生成命令和引用文件hash。任何“所有方法一样”“VCS更好”“过拟合”“视觉不足”都要指明具体对象和统计范围。


**交付状态更新见 docs/09：V8-05确认已完成。上述协议保护不变；不再把等待该作业作为前置条件。**


---

# 08 · 最终报告怎么写，下一步怎么定

## 第一页必须回答

**模型实际训练了多少？** 先把epoch解释和恢复状态说清。**决策是否不同？** 给同提示agreement与人工选择效果。**主要改进余地在哪？** 分开刻度、排序、分布、适配和视觉输入。**与领域已有方法相比如何？** 不让CS/BCE差值替代外部benchmark。

## 建议主表

A：所有方法的当前公开结果和开发指标，分宿主/数据/读数/成本。原TIFA/DSG本地适配标记保留；不把N9质量差写成百分比。

B：同seed八臂的决策一致性矩阵，以及相对FGA/VQAScore的选择差异。固定候选顺序，列出tie比例。

C：均值MSE、prompt偏差项、中心化误差、pair accuracy、regret，严格同一分母。非评分量表模型的MSE空缺。

D：逐epoch共同评价曲线、原生loss与梯度检查、阶段0/初始化的解释。权重未保存时不要画虚构的逐epoch逐例曲线。

E：分布RPS/CE与原始投票参照，区别概率误差与标注分歧。原Table C核路线与完整模型有表示差别，单列而非当作纯估计器胜负。

## 决策矩阵

| 诊断 | 下一步 | 不做 |
|---|---|---|
| 单纯报告解释错误 | 更正并收尾 | 重跑48单元 |
| 明确实现错误 | 申请最小受影响修复 | 扩成新方法故事 |
| 学习早期后发生适配损失，有支持证据 | 申请C1唯一适配范围干预 | 同时换骨干/数据/损失/融合 |
| MSE与任务选模错位、权重齐全 | 申请C2独立选模分析 | 回到确认集挑epoch |
| 多损失共同缺细粒度输入能力 | 只出C3设计卡 | 立即启动V10 |
| 差异只有刻度，外部选择不改 | 保留概率层面结论，应用收尾 | 给校准改善换名成选图提升 |
| 证据不足或无可解释改动 | 完成本轮、保留未决点 | 因GPU空闲继续扫描 |

预期优势是待验证目标，不填写预测领先点数。V8.1可以得到“没有值得继续的新训练”这一有效结论。研究路线仍由外部收益、方法归因、相对先行工作三者共同决定。


---

# 09 · 交付前更新：V8确认报告已到达

本包初次源码审查截点为 `a87b2d5`。准备期间，V8更新到 **`580781f`，2026-10-15 18:04:37 UTC**。新提交已报告V8-05的850非Rank提示、6生成器、5100图像确认结果。本包只读取已发表的汇总和提交，不直接读取封存标签或重新评分。

## 已提交确认结果 [R9]

| 比较 | pooled Spearman差值 | 同prompt pair accuracy差值 |
|---|---:|---:|
| U-CS − FGA released | +0.003，未检测差异 | −0.001，未检测差异 |
| U-CS − VQAScore | −0.008，未检测差异 | +0.044，Holm p=0.0008 |
| U-CS − TIFA | +0.312，Holm p=0.0008 | +0.146 |
| U-CS − DSG | +0.193，Holm p=0.0008 | +0.143 |

原主要指标是pooled Spearman；次级pairwise的正结果单独报告，不能替换主指标得出“全面超过VQAScore”。相对BCE，CS的Spearman差−0.016被检测到，pairwise未检测差异。这不是“所有指标完全一样”。逐生成器描述性差异不能拿来选择新训练的生成器或改变本包超参数。

## 对执行的改变

- D00现在应登记 **确认已完成并提交报告**，不再等待或重复提交1042053。
- D07引用这份固定汇总；原主/次级家族、预测平均方式和未检测差异的表述保持。
- D01–D06的设计来自此前开发讨论，内容、数据角色和条件方案不因确认结果调整。V8.1不读取该5100图像的逐例人工标签，不选seed，也不删除表现较弱的seed1。
- 这些确认提示已经完成一次评价，不再称为V8.1的新独立测试。原文报告中的“epoch0”仍需按真实训练循环解释。
- 重新读取最新trainer的重点区段，其Git blob SHA仍为 `67f9dd5d6c4e5a9e1b02c8f14be5a55e20a9f83d`，本包epoch诊断的源码依据未变；实际job snapshot仍由服务器确认。
- main `55d47c9`、V7 `2dc39b5`、V9 `1ce0e84`在本次更新检查中未变化。

这是来源状态更新，不是新实验结果，也不授权对确认结果进行事后优化。


---

# 标准逐图行格式

参考CLI读取JSONL，所有行是明确提供的输入，不自动搜索服务器目录。

必需字段：`dataset, role, method, seed, checkpoint_id, prompt_id, pool_id, image_id, candidate_index, score`。

可选：`cluster_id`（默认prompt）、`score_kind`（rating_mean或native_score）、`q`（五概率）、`u`（四尾概率）、`human_hist`（五投票比例）、`human_mean`、`votes`、`source_sha256`。

- 同一dataset/role/method/seed/checkpoint/pool/image不得重复；同pool候选index唯一。
- `q`若存在则sum=1，非负，score_kind=rating_mean时必须与score一致；u若存在须与q一致。
- 标签可以是完整hist，或仅human_mean；只有mean的公开预测比较可以做MSE/排序，但不得制造五档标签。
- `score_kind=native_score`的外部方法可以做排序，不计算1–5 MSE。
- 任何label字段在label-free模式都拒绝；避免外部dummy均匀hist误入评价。
- labelled模式仅允许FIT_AUDIT/CAL_SELECT/CAL_PROB/DEV；其他角色拒绝。该限制是流程护栏，不能替代真正的数据隔离和权限控制。
- 三种子不要事先平均；ensemble须另命名，例如U-CS-ensemble，而不是覆盖seed0。

严格并列按浮点原值，不按显示精度；主选择按candidate_index升序后的首个最大分。模型缺候选默认停止并输出说明，不静默削掉该图。


---

# 参考代码：只读诊断，不替代冻结评价器

## 本地检查

在包根目录执行：

```bash
python -m pytest -q tests
python reference/make_demo.py --out /tmp/v81_demo_rows.jsonl
python reference/diagnose.py --rows /tmp/v81_demo_rows.jsonl \
  --label-mode development --out /tmp/v81_demo_result.json
```

示例是合成数据，不是V8实验结果。输出路径已存在会拒绝覆盖；请使用新的临时路径。`requirements.txt`是最低兼容范围，不要求升级已有环境。

## 接入真实V8开发导出

```bash
python reference/adapt_v8.py \
  --pred /ABS/V8-FIT-U-CS-s0/pred/images_DEV.jsonl \
  --public-train-manifest /ABS/outputs/V7/data/manifest_images.jsonl \
  --role DEV --method U-CS --seed 0 --checkpoint-id best_sha256 \
  --out /ABS/outputs_v81/normalized/U-CS-s0-DEV.jsonl
```

`/ABS/...`须替换成D00核对过的真实路径；该适配器根据已读取的`v8_labels.py`字段构建，不读取图像压缩包或任何外部标签。每个方法/seed生成一份后可顺序合并JSONL，再运行`diagnose.py`。不同数据/角色/候选池不会混合；方法间缺行会报错。

自然数据适配器尚未在真实服务器产物上运行，服务器需先用少量真实行验证字段。若manifest的component字段不是`component_id`，须显式适配并记录，不能猜测连接成分。

## 日志审查

```bash
python reference/audit_logs.py \
  --logs /ABS/unit0/log.json /ABS/unit1/log.json \
  --source /ABS/job_snapshot/scripts/v8/v8_train.py \
  --out /ABS/outputs_v81/epoch_audit.json
```

该工具只读JSON和Python文本，不加载`.pt`。调用行序只是静态证据；真实控制流、快照、成功optimizer更新数仍需人工/服务器核对。`steps`有时只是尝试步数，不能据此证明AMP每次更新成功。

## 原生外部方法

外部评分可以写入标准schema，使用`score_kind: native_score`。它们进入排名和选图比较，未经过独立标定时不进入人工评分MSE。V7/FGA原始接口不保证与V8完全相同，不能机械复用适配器。

本工具只有开发角色的带标签计算和纯无标签模式。无标签模式会拒绝hist/votes等字段，包括外部导出中的dummy标签。它不是安全沙箱；真实数据隔离仍由项目目录、角色和负责人决策保障。

## 尚未自动实现的工作

完整VLM推理、gradient hook、时延benchmark、真正的job snapshot追踪、全局相关系数的cluster bootstrap、每epoch缺失checkpoint恢复等由服务器在现有项目实现。不要将参考脚本通过解释成这些任务已经完成。


---

# 来源与核对范围

核对日：2026-10-15。源码事实、已提交结果、本包推导和待检验假设分开记录。仓库链接均固定到提交；Git blob SHA 不是内容的SHA-256。

## [R1] V8-03 冻结方案

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/reports/V8_UTILITY/V8_03_PROTOCOL_FROZEN.yaml)

已读取全文；原训练、模型选择、事件和指标的依据

## [R2] V8-04 正式结果及已附更正

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/reports/V8_UTILITY/V8_04_EVAL_REPORT_20261015.md)

已读取全文；数字来自报告，未在本地重新计算自然数据

## [R3] v8_train.py 训练循环与导出

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/scripts/v8/v8_train.py)

重点核对185–278行；训练先于epoch末CAL和best.pt保存；实际作业快照待服务器核对

## [R4] v8_core.py 损失及分组

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/scripts/v8/v8_core.py)

已读取；八种损失、五档概率、排序分组、批次归一化

## [R5] v8_labels.py 原始投票接口

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/scripts/v8/v8_labels.py)

已读取；FIT/CAL/DEV、overall_votes、prompt→image自然权重

## [R6] V8-05 确认集冻结协议

[原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/a87b2d53380783db0932b144dddcd368ac4a7b2d/reports/V8_UTILITY/V8_05_CONFIRMATION_PROTOCOL_FROZEN_20261015.yaml)

已读取主要协议、TIFA160与执行记录；850提示、5100图像；不等同Rank的9图池

## [R7] V8-05 提交记录

[原始来源](https://github.com/W-Yinghao/VCS_QMI/commit/a87b2d53380783db0932b144dddcd368ac4a7b2d)

1042053已提交；本次未读取到确认结果报告，非实时队列

## [R8] 负责人现有数据使用决定

[原始来源](https://github.com/W-Yinghao/VCS_QMI/commit/55d47c9fe73375230c86a7ec1b2e575bdaeb19d5)

沿用已有数据可用于项目的负责人决定；不重新设置许可讨论为阻塞项；不扩展到公开再分发

## [T1] 合作者 Variational CS-QMI Research Plan，September 2026

来源：本次对话提供的研究材料；未重新发布全文。

第4–7节：S/J、精确gap、分类解释；原文件已在对话完整提供，本包不附全文或改变公式

## [T2] 2026-10-10 综合稿，附录4勘误

来源：本次对话提供的研究材料；未重新发布全文。

优先采用附录4，不采用正文已撤回的数学论断；正文为历史状态，不覆盖10月15日结果

## [T3] 2026-09-28 补充意见 §3.7

来源：本次对话提供的研究材料；未重新发布全文。

保留MINE/NWJ/InfoNCE/SMILE等外部横向比较的组织原则；不把其中后被更正的专属性质照单接受

## [E1] SciPy rankdata 官方文档

[原始来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)

平均名次处理并列；本包独立实现以SciPy作测试参照，不要求升级服务器版本

## [E2] PyTorch Autograd mechanics 官方文档

[原始来源](https://docs.pytorch.org/docs/stable/notes/autograd.html)

eval模式和梯度控制是两回事；仅支持实现说明，不支持任何性能因果主张

## [E3] Reid & Williamson, Convexity of Proper Composite Binary Losses, AISTATS 2010

[原始来源](https://proceedings.mlr.press/v9/reid10a.html)

正规复合损失与link；仅核对出版页和摘要，不宣称全文逐式复现

## [R9] V8-05确认结果：交付前新提交

[提交与报告](https://github.com/W-Yinghao/VCS_QMI/commit/580781f47cbe2249765bc36af05221f307a2b42c)

已读取汇总；未读取原始确认标签。详情见交付增量。
