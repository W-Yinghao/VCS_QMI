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
