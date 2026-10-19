# V8 效用对齐 CS critic · 完整服务器方案

2026-10-14。来源事实、直接推导、新实验设计分别标明。



---

原文件：`README_CN.md`

# V8 候选方案 v1：面向选图效用的 CS critic
## 从“平均满足几个元素”改为“这张图会获得怎样的总体人工评分”

**应用不再更换；仍然是图文一致性评价与best-of-N选图。改变的是critic估计的事件与最终决策之间的连接。**本包是新一轮候选方法及完整实验计划，不替换V7，不宣称新算法已经有效。

主设计：以原始1–5人工评分构造四个阈值事件，使用同一VCS的S/J估计其后验，通过一个一致的评分分布直接计算预期人工分数。元素预测可以用于分析，但不再被当成整体人工质量的理论等价量。

**关键先行工作已经纳入**：NIMA、RPS、CORAL/CORN早已研究评分分布/累积分布/等级一致性；RankNet/ListNet提供直接排序对照；平衡RPC与基础J重合。因此，下一轮要证明的是一个具体的任务适配与实际外部收益，不能宣称首次用概率分布评分、首次阈值分解或首次二次泛函。

主线读法：`01_SERVER_AGENT_PROMPT_CN.md` → `docs/02_RESEARCH_DESIGN_CN.md` → `docs/03_MATHEMATICS_CN.md` → `docs/04_TRAINING_CN.md` → `docs/05_BENCHMARKS_CN.md`。`FULL_PLAN_CN.html` 为完整离线版。

| 主实验 | 规模 | 作用 |
|---|---:|---|
| 完整图文评分训练 | 8配方×3seed=24个正式单元 | VCS正面对比原生任务方法、直接回归、分布评分、pair/list排序 |
| 独立调参 | 8配方×3学习率=24个完整调参单元 | 同预算CAL选择，不只为VCS调参 |
| 传统核/密度比桥接 | 3条路线×4阈值，联合选定配置 | 不把核方法弱化成一个未经调参的KDE |
| 外部公开系统 | 复用V7不可变预测，新增方法另记 | 不只与自己的head比较 |

首先完成数据/单位测试/资源实测；8配方的全部正式seed都运行，不以seed0胜负筛选。没有真实GPU耗时预估，服务器提交前按既有权限填实预算。

交付前增量与最新双分支快照见 `docs/00_DELIVERY_UPDATE_CN.md`；不替换旧冻结协议。


---

原文件：`01_SERVER_AGENT_PROMPT_CN.md`

# 发给服务器：开始下一轮 V8 候选方案

请建立独立目录 `reports/V8_UTILITY/`、`outputs/V8_UTILITY/` 与建议分支 `v8/utility-aligned-cs`。先读取本包数学和先行工作，不把“ordinal-CS”当成已经证明的新发明。原始S/J/gap保持；新内容是阈值事件、自然概率一致性、任务用途与实际比较。

本轮从公开BLIP-2 COCO初始化完整图文评分路径，主实验不是只在冻结特征上训练小头。视觉主干是否冻结按共享FGA路线，Q-former与评分head按配方完整训练；所有方法共享数据、输入与参数容量。保留传统CS/密度比、非CS目标和强公开图文评分方法；BCE只作一个必要控制，不成为其他实验的准入条件。

现在可开始：核对原始FIT总体投票、构造四阈值标签、建立monotone head与全部对照、运行本地参考/集成测试、使用FIT/CAL计时与开发；将24正式+24调参单元预算提交到既有资源审批流程。GPU作业由服务器按owner权限提交，本包没有替用户提交。

V7已有E主读数、八个外部对比和封存标签不能改。V8不读取GenAI人类标签做开发；记录V8协议时间与V7终末读取时间。即使重复使用GenAI做最终比较，也必须按暴露情况命名，并安排一项未参与方法选择的确认资源。付费API/新人工标注/新生成器训练仍需单独批准。

第一批交付：`V8_00_INTAKE.md`、`V8_01_LABEL_HISTOGRAM_AUDIT.md`、`V8_02_METHOD_EQUIVALENCE.md`（尤其π=1/2时与RPS关系）、`V8_03_PROTOCOL_FROZEN.yaml`（完整主家族/数据/选择/预算）、真实smoke与集成测试结果。不要等V7某格获胜再启动设计，也不要拿V7终末错误做训练样本。

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

原文件：`docs/02_RESEARCH_DESIGN_CN.md`

# V8：科学问题、预期优势与方法范围

## 1. 由现有结果得出的设计动机
R5说明候选的条件结构会影响有限模型；它没有证明逐候选后验不能排序。D1暴露小critic分歧与宿主真实风险之间的代理距离。V7目前的元素Brier和同特征核比较没有建立普遍VCS优势；它尚需完成外部评价，并检查平均元素满足率E与总体人工评分H是否对应。[G07,G08；旧结果具体数值沿上一轮审查，非本包重算]

本轮不再在“一个未验证代理”之后叠另一个gate。直接让输出描述最终benchmark使用的总体评分分布。一个准确的逐图期望效用分数即可完成best-of-N，不必先假设需要集合网络。真正的难点是有限图文表示是否能估准、训练目标是否有效，以及新生成器上是否成立。[P02,P03]

## 2. 待检验的优势，外部比较优先
| 假说 | 对照 | 主证据 | 否定范围 |
|---|---|---|---|
| H-U：效用对齐改善同提示选图 | V7 E/O、原生FGA、VQAScore、TIFA/DSG、直接MSE | 相同候选的人工M9/后悔值 | 不改善则不能用它承担新应用优势 |
| H-D：评分分布比均值回归增加有用信息 | MSE/L1、类别CE、RPS/NIMA类目标 | 分布风险+外部选图/不同固定效用 | 只分布更好不算生成选择更好 |
| H-CS：同结构下CS拟合有指定优势 | threshold BCE、RPS、核/密度比 | RPS/Brier、MSE、训练成本与M9 | 不出现则收窄为共同目标/结构收益 |
| H-T：强传统估计器不能用低成本完全替代 | 调优RuLSIF/Nyström/KDE | 相同初始特征上的误差与实际选择 | 核更好就进入主表，不降带宽预算 |

不预写领先百分点，不以p<0.05或某个固定幅度决定CVPR路线。研究必须获得实际竞争力，但可以是较好质量、可靠性或明确的质量–成本取舍；不能只比弱随机/CLIPScore。

## 3. 主方法输出
输入整图I和完整提示p；不依赖自动元素拆解才能计算主分数。输出五档自然评分概率 q=(q1,…,q5)，从而得到：
- 主要选图分数 μ=Σh h qh；
- 诊断阈值 Pr(H≥k)，k=2,3,4,5；
- 次级固定效用例如Pr(H≥4)，仅单独预注册，不替代平均评分主终点。

“评分分布的离散方差”是对标注分歧的预测，不等于模型认识不确定性，也不能当成均值置信区间。不同seed的分歧另记。区分图文一致性与审美/综合偏好；主H使用原始alignment标签。

## 4. 保持最小必要改动
继续使用共享BLIP2/Q-former全图文交叉注意力。从相同未经过EvalMuse监督训练的初始权重开始。利用32 query表示的均值产生五档head；不额外加入大型judge或生成器。第一轮不加元素辅助任务，以隔离“整体目标”本身。若随后加元素辅助，所有主要分布目标共享，同样计入成本。

不是用多个整体评分head就有新理论。阈值分解、softmax单纯形和均值读数属于标准结构；新颖性需要在完整文献定位和实证里建立。结果也可能说明直接MSE或RPS已经足够。


---

原文件：`docs/03_MATHEMATICS_CN.md`

# 数学与可执行定义

标记：原S/J为[F01]；以下评分分解、误差界和参数化是本轮直接推导/标准方法实例，不是宣称新定理。相同读数可由其他损失拟合，RPC/RPS/CORAL/NIMA关系必须正面写明。[F02,P04–P11]

## 1. 原始观测和四个事件
自然采样D：均匀选提示、均匀选该提示有效图像、均匀选该图一个有效原始总体标注者。H∈{1,2,3,4,5}。不同图3/6评分均按每图归一，不将6票图赋双倍样本权重。

令Z=(I,p)。对k=2,…,5，Ck=1{H≥k}，yk(Z)=图中有效票里H≥k的比例，πk=E_D[yk]（只用FIT）。定义Pk=D(Z|Ck=1)、Qk=D(Z|Ck=0)。M_k=(Pk+Qk)/2；ηk=(pk−qk)/(pk+qk)。这里pk/qk是两类密度，勿与下面自然事件概率uk混淆。

如果某阈值πk为0或1，Pk/Qk不能同时定义：保留为常数阈值、单独报告，不用随意加epsilon制造二类。稀有但非零时保留FIT先验、报告有效正负质量；不根据测试比率校准。

## 2. 单个原始J保持不变
对每个k：
\[
J_k(T_k)=E_{P_k}T_k-E_{Q_k}T_k-\tfrac12E_{P_k}T_k^2-\tfrac12E_{Q_k}T_k^2,
\quad S_k-J_k=E_{M_k}(T_k-\eta_k)^2.
\]

以ak=1/4聚合训练；这是四项一般CS判别的加权和，不能称为一个新的单一QMI。

自然样本与soft投票下：
\[
L_{CS}=\sum_k a_k E_D\left[\frac{y_k(1-T_k)^2}{8\pi_k}+\frac{(1-y_k)(1+T_k)^2}{8(1-\pi_k)}\right]
=\tfrac14\sum_k a_k(1-J_k).
\]

它不是对投票均值取二值，不需要复制每名标注者为独立图像。源码中所有正负期望与大batch分母一致；微批累积不能改变权重。

## 3. 一致的自然分布，而不是强迫平衡critic单调
令五个logits经softmax得到qh，定义自然尾概率：
\[
u_k=\Pr(H\ge k\mid Z)=\sum_{h=k}^5q_h.
\]
（实现字段用 `tail_prob`，避免与参考分布ν混淆。）因此u2≥u3≥u4≥u5。

用FIT先验把自然概率映射到平衡critic：
\[
\ell_k=\operatorname{logit}(u_k)-\operatorname{logit}(\pi_k),\quad
T_k=\tanh(\ell_k/2)=\frac{u_k-\pi_k}{\pi_k+(1-2\pi_k)u_k}.
\]

训练稳定实现：`log_u=logsumexp(logits[k-1:])-logsumexp(all)`，`log_1mu=logsumexp(logits[:k-1])-logsumexp(all)`；ell=log_u−log_1mu−logitπ。避免极端softmax下先减1导致log(0)。输出概率仍保留原值；仅评分NLL使用预定epsilon并记录饱和。

πk各不相同，所以T_k不必随k单调。**约束的是自然尾概率，而不是四个不同先验下的T。**独立四个二元head可能相互矛盾；五档单纯形提供不依赖EVAL的结构一致性。CORAL/CORN已有解决有序一致性的不同方法，不认领首次。[P06,P07]

## 4. 与最终效用的精确连接
\[
\mu(Z)=E[H|Z]=1+\sum_{k=2}^5u_k(Z).
\]

更一般，对预先固定的数值效用v1,…,v5：E[v(H)|Z]=v1+Σk(vk−v{k−1})uk。主benchmark固定v_h=h，不用测试结果选效用。1–5是benchmark的数值约定，单调重编码后选图排序未必保持。

在假设 `E[H_i | prompt, entire candidate set] = E[H_i | prompt, I_i]` 下，argmax μ_i最大化期望人工分；若标注存在候选上下文效应，该假设须说明，不自动成立。Pr(H_i>H_j)也不等于μ_i−μ_j：最大化赢另一个样本的概率不必最大化期望分数。

有限误差：设δik=ûik−uik，r为按估计μ选图的真实期望损失，则
\[
r\le2\max_i|\hat\mu_i-\mu_i|\le4\sqrt{\sum_{i=1}^N\sum_{k=2}^5\delta_{ik}^2},\qquad 0\le r\le4.
\]
这是三角不等式/Cauchy–Schwarz的直接推论；并非候选数无关保证，平均误差较小不能保证每个有限池选对。

## 5. 与RPS/NIMA的关系必须先查明
定义归一RPS为(1/4)Σk E_D[(Ck−ûk)^2]。对soft投票，期望项为 `yk*(1-u)^2+(1-yk)*u^2`；若使用(yk−u)^2，两者差 `yk*(1-yk)` 与模型无关。

当所有πk=1/2时，T=2u−1，L_CS与上述归一RPS相同。一般先验时，映射为非线性有理式，平衡测度不同，**不是简单将RPS乘固定阈值权重**。完整概率分布下所有严格proper目标的总体最优仍是相同真实分布；有限容量/优化/样本下不预设VCS或BCE必胜。

NIMA用评分直方图与CDF距离，其常见r=2形式带平方根：sqrt(mean(CDFdiff²))。它与“平均平方CDF差”在按样本求均值时不是同一个训练目标。主U-RPS准确命名为平方CDF/RPS任务适配；要称NIMA-loss复现，运行独立U-EMD2配置并保留公式/归一常数。[P04,P05]

## 6. 先验恢复的误差传播不是免费的
自然概率pπ(T)=π(1+T)/(1+(2π−1)T)，导数为2π(1−π)/(1+(2π−1)T)²。其最大值Lπ=max(π,1−π)/(2min(π,1−π))。

又dD/dM≤2max(π,1−π)，故同一对类条件分布下，E_D[(pπ(T)−pπ(η))²]≤2max(π,1−π)Lπ² E_M[(T−η)²]。罕见阈值会使该界非常松；它不保证VCS改善自然风险。用FIT分布报告π和实测误差，不能以理论有界性忽略这个条件。

## 7. 不可声称的结论
不将概率直方图叫因果解释；不将总评分方差叫均值置信区间；不从一次标签得到真实后验MSE；不把四阈值gap直接称为最终人工regret的实测精确值；不把原始二次泛函、RPS或等级一致性重新命名为新发明。


---

原文件：`docs/04_TRAINING_CN.md`

# 完整训练与强对照矩阵

## 1. 相同表示与新模型起点
复用V7的数据身份与公开BLIP-2 COCO初始化，不能从见过本地DEV的发布FGA权重开始。保留FGA原生冻结ViT选择，训练Q-former/query tokens及新的评分head。主输入364完整图像、完整提示上限128；不翻转、不裁剪改变方向/数量事实。模型加载、参数量和每次sample exposure均记录。[G02]

主head：32个图像查询token的mean→5个logits，softmax为自然评分直方图。所有8臂使用相同5logit head和相同初始logits偏置（FIT总体直方图的log值，平滑仅为数值初始化并记录，不改标签）；每seed相同权重。MSE/L1/Rank/List从同一head计算μ，因此容量相同。

## 2. 八个完整配方
| ID | 目标 | 主输出 | 要回答的问题 |
|---|---|---|---|
| U-CS | 四阈值原J，FIT先验平衡，a=1/4 | μ=1+Σu | 原方法在效用对齐后的表现 |
| U-BCE | 同四阈值的平衡BCE，使用ell=logit(u)−logitπ | 同μ | 严格近邻损失控制 |
| U-RPS | 自然采样下平均平方CDF误差 | 同μ | 强非CS正规分布评分对照 |
| U-CE | −Σh empirical_hist_h log qh | 同μ | 标准评分分布最大似然 |
| U-MSE | ((μ−Hbar)/4)² | 同μ | 直接优化均值效用的强对照 |
| U-L1 | |μ−Hbar|/4 | 同μ | 任务常见L1与FGA目标风格 |
| U-RANK | U-MSE + 同提示RankNet-style BCE，λ=1 | 同μ | 外部排序机制，不假设必须CS |
| U-LIST | U-MSE + 同提示ListNet top-one CE，λ=1 | 同μ | 候选竞争的强任务对照 |

三seed={0,1,2}全部正式训练：24单元。每配方用独立tuning seed=17，lr={5e−6,1e−5,2e−5}，完整5epochs：24调参单元。所有运行都计成本。不能单独为U-CS加入辅助头、额外标签或更长训练。

## 3. group batch与监督公平性
冻结一种所有臂共用的grouped image batch：按seed/epoch洗牌提示，每提示图像洗牌切成≤7图的小组，将组打包到≤14图batch；全FIT每图每epoch恰出现一次（末批也处理），同seed所有臂读取相同batch-index manifest。单图提示仍训练均值/分布目标，不因排名目标没有pair而删除样本。

点式项使用自然目标的prompt→image权重（每提示质量一致），按有效样本权重精确累计。Rank/List只在同一子组≥2图时计算；组内平均后按prompt的子组数量归一，另记录pair/list实际曝光量。它们总有共同MSE项，单图组不会没有监督。任务基线同时改变损失形状是合法的，但不能称为“只换一项proper loss”。

RankNet任务适配：对所有无序图像对，目标为Hbar_i>Hbar_j的1/0，严格平局目标0.5；模型logit=(μ_i−μ_j)/τ，τ=1。基于观察到均值的pair并不是真实潜在偏好标签，保留标注噪声限制。ListNet：target softmax(Hbar/τ)、prediction softmax(μ/τ)，τ=1。温度敏感性另行申请，对两臂对称，不能看外部结果后搜索。

### 共同优化器与数值模式（本轮设计）
AdamW betas=(0.9,0.999)、weight_decay=0.05；100个优化步从1e−8 warmup至选定学习率，随后按实际总优化步cosine降至0。全部配方相同；AMP用于多模态forward，softplus/logsumexp/阈值风险与归一分母至少fp32。浮点类型、GradScaler和有效更新计数入配置。不为某个损失单独缩放梯度或改clip后仍称匹配。

ViT参数按共享约定冻结且保持eval；训练Q-former和head时启用同一dropout配置。所有预测/校准model.eval()。没有像旧D1那样单独冻结一个有dropout的critic再优化gate的步骤。

## 4. checkpoint与读数选择
统一主选择终点为CAL_SELECT的prompt-macro均值MSE（同量表、同样本），各epoch保存模型；平局优先早epoch，再低lr。原因是主要部署效用为期望评分，不以某一损失的native值选择其他方法。记录CAL同提示选择质量及native最优epoch仅作次级诊断，不根据DEV或GenAI重选。

主输出raw μ，不用混合α或V7的E后验平均。独立CAL_PROB仅做预登记的次级概率温度校准，主结果不以校准版取代。概率校准对分布head保持单纯形/顺序，不分别对四尾概率做可能交叉的任意Platt后还宣称一致性。

## 5. 数据规模与独立训练
5epochs按全部合格FIT图，不是小样本探针；训练量因无效overall标签的排除而改变须在第一步公布。使用相同有效FIT集合，不因bad element decomposition排除总体标签本身有效的图。两级质量标签有不同缺失规则，具体见数据章。

有限数据(10%/25%)、第二backbone、额外ranking辅助不是首轮自动网格。若完成主表后有明确问题，再冻结扩展；所有扩展必须有传统/非CS对应对照，不能只追加VCS。

## 6. 可选而非默认：有序结构对照与辅助项
U-EMD2：按NIMA公开r=2公式的平方根版本，3seed；CORAL/CORN：不同参数化与条件风险集，不能只替换名称。它们有新代码/结构预算，若要宣称“优于专门有序方法”则必须完成，否则只在Related Work定位。

CS+ranking只在概率好但同提示选择不足时作为下一次计划；同时给BCE/RPS相同辅助项、同λ选择机会。不要第一轮就把所有有利模块打包，导致无法判断哪一项有效。


---

原文件：`docs/05_BENCHMARKS_CN.md`

# Benchmark、指标与必须比较的工作

## 1. 三张主表
**A：完整系统**。V7不可变N0/M0/M1/M2的O/E、发布FGA、VQAScore、TIFA、DSG、可运行HPSv3/VisionReward与V8。所有外部公开方法保留原生输出；V7表格的旧结果不被V8覆盖。[G06]

**B：同数据完整训练**。8臂×3seed，比较自然评分分布、预期分数、同提示排名、实际选图与成本。U-BCE只是其中一行，不作为开展A/C的条件。

**C：传统CS与密度比路线**。相同初始多模态特征，KDE后验、RuLSIF、Nyström平方回归分别针对四阈值拟合；原生CS-KDE的全局数值另列。不把一个总体散度标量直接当成单图质量分。可复用V7 X21算法但标签、先验、feature pooling已变，必须新模型ID。

## 2. 传统估计器具体接入
统一表征用同一未监督更新BLIP2的32query mean，768维；FIT加权均值/sd统一。主768维，PCA64/256仅作共同敏感性，不挑最有利的一维当主结果。

每阈值自然y_k生成权重P∝w*y_k、Q∝w*(1−y_k)。RuLSIF α=1/2读相对比r∈[0,2]→T=r−1；Nyström拟合平衡标签η；KDE的log p−log q→T=tanh(ell/2)。所有原始/裁剪值和比例保留，再恢复自然尾概率。各阈值可能交叉：原始列保留，统一PAVA单调投影列次级报告；不能只给我们的网络一致性而隐藏核方法简单可行修正。

带宽倍数{.25,.5,1,2,4}、正则{1e−6,1e−4,1e−2,1}、中心{256,1024,4096}；一次有方向的边界扩展沿V7协议；真实矩阵时间计入成本。四阈值选择共同超参，以CAL_SELECT μ-MSE为主，native-CV为次级；不是每个阈值按DEV单独挑最强配置。3路线×4阈值=12个联合配置内的确定性拟合（网格搜索另计），非12个独立统计重复。

原生经典CS需明确参考测度、核宽与自身真值。MINE/NWJ/InfoNCE/SMILE已有独立估计器基准继续保留，不强迫其原生MI值变成1–5质量；若增加ratio桥接，需定义常数归一化和概率转换并与相同阈值对象比较。[F01,P10,P11]

## 3. 数据职责
| 数据 | 用途 | 规则 |
|---|---|---|
| EvalMuse FIT/CAL | 完整评分模型训练、独立调参 | 只用已有角色，不从DEV重新抽训练集 |
| EvalMuse DEV | 与V7相同开发参照 | 已被读过，称开发评价而非新独立确认 |
| GenAI-Rank | 800提示、两个生成器各9图的主要实用比较 | 遵守标签权限；V8相对V7读标签时间决定暴露表述 |
| GenAI-Bench | 整体对齐外部相关性；可登记与Rank无重叠部分 | 名单排重在看H之前；不能声称整个Bench独立 |
| TIFA160 | 外部小样本一致性 | 与训练/Rank重叠清点，不当大规模稳健结论 |
| RichHF对齐部分 | 额外来源的泛化 | 按原量表与官方alignment维度；不能把审美/伪影混成H |
| Winoground/NegBench | 可选语义诊断 | 不要求每个评分分布方法自然具备全部逻辑能力，不反选赢的子集 |

## 4. 主要指标
**应用主终点**：每生成器独立的GenAI-Rank M9人工评分。与V7相同N=1/3/9官方规则；regret用同一候选池观察人工最大值，胜/平/负完整报告。

**同提示主诊断**：在同一提示中的pairwise accuracy、Kendall tau-b、选择regret、分数margin。全局SRCC/PLCC与这些分开：prompt难度能抬高全局相关性而没有选图价值。

**分布指标**：每阈值自然Brier、归一RPS、直方图对数损失、平均评分MSE/MAE。vote-MSE与随机标注者风险分开；预测分布离散方差不是均值误差条。类别CE/RPS只有在原始标签分布有效时计算；若外部只有Hbar，只报均值/排序，不虚构直方图。

**稀有阈值/尾部**：Pr(H≥5)等事件的precision/recall或PR曲线仅辅助；每阈值π/有效票数/校准和clip比例；选中最高分图的误差与全部候选平均误差。平均RPS较低不保证best-of-9提高。

## 5. 预定比较家族
主外部家族：U-CS 对B0 FGA、B1 VQAScore、B4T TIFA、B4D DSG，两生成器8个对比，沿用prompt-cluster重采样5000与Holm。公共baseline预测版本不变可复用；旧标签已经公开读过则注明重复benchmark。

同条件家族：U-CS 对U-RPS、U-MSE、U-RANK、U-LIST，两生成器8对；U-CS对U-BCE的2对作为归因；U-L1/CE完整表不隐藏。主内部选模用CAL均值MSE，非终末上述比较。报告所有效应与逐seed结果，不只p值。

若要把“分布方法整体比标量回归强”作为主张，应预先登记其对应contrast，不在结果出来后挑U-RPS替U-CS宣布CS成功。反之，只要U-CS没胜BCE也不取消其他外部比较。

## 6. 成本
训练epoch、加载/解析、图像编码、query head、候选池重排都实测。V8不需要元素parser才能评分，但baseline可能需要；允许按prompt缓存且计完整代价。这是可能的系统效率区别，未实测前不称新estimator更快。不同GPU的时延不直接排名。


---

原文件：`docs/06_DATA_LABELS_AND_EXPOSURE_CN.md`

# 原始标签与新旧评价隔离

## 1. 数据已经支持什么
仓库审查确认 `total_score` 为1–5原始列表：25,071图有3票、7,646图有6票；不能将列表长度等同独立图像数。6票由重标注流程产生，不能假设是均匀抽取的第二套标注。[G04,P01]

对每图n有效票：hist_h=count(H=h)/n；tail_k=Σh≥k hist_h。检查Σhist=1、tail非增、1+Σtail=mean(votes)。本包参考代码在n=0、非整数、范围外值时拒绝构造；不把平均分四舍五入成人工等级。

数据采样目标为当前benchmark标注流程下的随机有效评分；不声称代表所有人类客观真值。没有rater_id时不能跨图按标注者bootstrap。双层prompt与seed统计为主，数组内vote重采样只为敏感性。

## 2. 原总体标签与元素过滤不能混同
bad_decomposition可能让元素标签不可用，但总体H仍有效；首轮U组不使用元素。因此先定义独立的overall_valid与meaningful规则，按原字段语义审核后所有U臂共用。保留原V7元素有效子集上的交集诊断，不能把子集改变收益都归于新criterion。

新训练仅用原FIT合格图，CAL_SELECT/PROB原角色不变；DEV已曝光，不能再称第一次测试。不存在“把全公开TRAIN重新划分一次就获得新clean test”。模型初始权重不含EvalMuse监督。

## 3. 全局H分布与提示偏差
对prompt/source/generator分别记录mean/var/3vs6票。图片分布有真实/合成prompt区分，但 `type=real` 指prompt来源，不代表图像是真实照片。原始generator身份不加入主评分器输入，防止凭生成器名预测质量。

不按prompt整体分数方差选择高区分度训练集；若使用原FGA的variance weighting，属于一个明确对照，不默认用于全部U。新训练权重仅prompt→image，目标是评价分布的一致采样。

## 4. 独立确认安排
V8开发开始时先记录GenAI主标签是否仍封存。若V7评测先完成，不延迟V7，也不修改V8方法来追其测试失败；相同GenAI结果命名为预定重复benchmark比较。

在首次V8结果前登记一个额外未用于开发的来源：优先是GenAI-Bench的Rank之外提示（实际交集由输入名单确认，不能假定850条恰好可用）；如果该源已被V7完整评过，则考虑未读的TIFA160或RichHF对齐资源。其部署格式不同，仍按官方指标而非伪造9图池。

没有新未读来源时，不许编造“独立确认已完成”：先完成已批准训练与重复benchmark，明确证据限制，向owner提交额外数据使用决定。下载原数据卡片/图像清单与读取人工评分是不同动作；现有权限照旧。


---

原文件：`docs/07_DECISION_AND_ALTERNATIVES_CN.md`

# 结果如何解释，何时改什么

## 1. 我们不把“点式分数”当成已确立根因
准确的E[H|I,p]可以对候选作最优期望效用排序。R5/D1/V7暴露的是有限模型、代理/效用连接与训练问题。V8先改变估计事件，不自动叠一个更重的set transformer。

## 2. 预先定义的读法
| 结果 | 决定 |
|---|---|
| U-CS超过强外部方法，且概率/效用误差与收益同向 | 继续做新生成器/数据效率确认；强调具体适用条件 |
| 所有整体目标都优于V7 E，但U-CS≈BCE/RPS | 任务效用对齐有效，不是CS拟合独占；保留这个认识，重新评估方法新增内容 |
| U-RPS/CE明显优于CS | 将CS与自然风险失配、阈值先验和有限容量作为检验对象，不削弱RPS |
| U-MSE最佳 | 分布估计未提供选图必要增量；除非在预定风险/尾部用途有收益，否则缩小ordinal路线 |
| Rank/List最佳 | 在当前数据训练下直接排序更有效；不宣布后验不可排序。是否加共同排序辅助作为独立下一实验 |
| 训练内好、外部差 | 区分prompt分布、生成器、rater标度和视觉表征；不在外部few-shot后继续称zero-shot |
| 核回归已足够好 | 公开强核结果；分析深度微调是否真正增加信息，而非预设高维核必坏 |
| 全部U不如公开FGA/VQA | 检查实现后结束当前简单head设计；不能仅因仍叫V8就继续无界调参 |

## 3. 两条有条件的备选，第一轮不并行铺开
**A. 元素到整体的受限聚合。**若人工元素能表达H，而V7平均聚合不足，可以将元素概率、类别和完整提示输入一个小聚合器；需与同样监督的BCE/RPS与直接overall模型比较。只能在FIT上用out-of-fold元素预测训练聚合器，避免FIT自预测过乐观。它是现有评分体系的任务设计，不是新散度。

**B. 明确的同提示pair证据。**若任务主要需要比较两张图，可以令观测=(I_i,I_j,p)，P/Q按人工偏好事件定义，使用原J；推理要付O(N²)或锚点比较成本。必须与RankNet、ListNet、HPS等比较，并明确偏好概率与期望分数不是同一个目标。不因为“set更强”就自动启动。

两条备选只能在前一轮定位证据支持后选一条，原U主表保留。不重新开token选择、RefCOCO连续修补或CLIP预训练。

## 4. 新颖性与论文承诺
本轮直接推导的tail-sum、误差传播与RPS关系可作为正确性与方法定位，不应列为核心首创。与RPC的泛函重合继续写明。若只有“换四个标签，再最小化已有loss”，方法新颖性仍可能不足；必须由相应估计/统计/视觉能力的具体新增结果支持论文，而不是靠V8命名解决。

当前计划的用途是得到可归因、可复现的答案；它不保证CS赢。外部竞争力、机制归因与理论创新是三项独立要求，任何一项不成立都要如实报告。


---

原文件：`literature/REVIEW_CN.md`

# 文献综述：评分分布、排序与图文一致性

## 1. 为什么下一轮必须把有序/分布方法纳入
NIMA已经不只回归MOS，而是预测人类评分直方图，使用CDF距离并从分布读取均值和方差；其评价是AVA/TID等图像质量/审美，不是本任务的文字alignment。我们借鉴其目标时应叫图文任务适配，不将其原生checkpoint的image-only输出当同条件对照。[P04]

RPS是有序离散结果的正规分布评分。预测四个H≥k事件后取均值，是标准的尾和表示。CORAL通过共享结构保证等级一致性；CORN使用条件风险集及链式概率，不能把任意四个BCE head冠以CORN名字。两者提供结构和标签设计的先例。[P05–P07]

**本轮结论：阈值分解/一致性/分布输出不构成首次创新。**CS的混合测度、有限拟合与用途要由实际比较体现；π=1/2与平方CDF的等价关系尤其必须公开。

## 2. 直接优化排序的工作
RankNet研究配对概率损失；ListNet将列表top-one分布作为目标。它们是Task-aligned强基线，不能只将BCE留在对照表里。我们用同一prompt的人工均值生成pair/list监督，是任务适配，并保留原始票的噪声与标度限制。[P08,P09]

这两项并不证明点式预测不能排序；VQAScore给每图单独评分，GenAI-Rank已经检验了其best-of-N能力。因此本轮不把“加竞争”写成唯一原因，而正面比较期望效用、评分分布和排名损失。[P02,P03]

## 3. 最直接的VL系统
FGA-BLIP2同时学习整体评分与元素标签；本轮已有N0/M0结果与源码差异需要保留。PN-VQA、TIFA、DSG采用不同分解/问答路线，它们是外部系统而不是我们的loss变体。GenAI-Rank固定池提供外部人工终点；不能用更高模型J证明选得更好。[P01–P03,P14,P15,G06]

HPSv3使用大规模偏好数据和不确定性排序，并通过选择改进图像；这意味着“预测不确定性帮助选图”也已有强先例。VisionReward等系统包含更宽的偏好维度，按原生输出作为应用对手，不能用错误的概率定义弱化它们。[P13,G06]

RichHF同时含图像区域/文字反馈与对齐信息，可以补充外部泛化。但其多个维度不能一并改写成1–5的alignment标签；官方任务和我们的变换需要分开。[P12]

## 4. CS与非CS估计器
RPC在平衡参数下与基础J精确重合；RuLSIF α=1/2对应相对密度比；KDE是另一类估计路线。Nyström是核函数类近似，不是一项无需学习的“弱方法”。本轮必须有调优核回归，并使用共同初始特征/相同事件，防止把表征提升误归于估计器。[P10,P11,G05]

原始研究要求的MINE/NWJ/InfoNCE/SMILE外部横向比较继续由对应受控依赖任务承担。对这些原生不同目标，必须分别报告原生真值或合法ratio桥接，不把所有输出都称CS精度。[F01]

## 5. 文献到实验的落实
| 先行工作 | 本轮具体落实 | 状态 |
|---|---|---|
| NIMA/平方CDF/RPS | U-RPS必做，U-EMD2可选准确复现公式 | 文献与定义核对；未训练 |
| CORAL/CORN | 解释一致性已有；需要结构优越主张时补其任务适配 | 原论文摘要/作者入口核对；非完整复现 |
| RankNet/ListNet | U-RANK/U-LIST必做，完整图文训练 | 目标定义已写；未训练 |
| FGA/VQAScore/TIFA/DSG | 同一候选池外部主对手；复用V7固定输出 | V7已登记，结果状态需server核实 |
| HPSv3/现代reward | 补一个可运行强系统，说明偏好范围与成本 | 可用性与许可按V7登记 |
| RPC/RuLSIF/KDE/Nyström | 先行归属+三类调优传统估计桥接 | 使用既有实现，V8新标签需新fit |

这是定向综述，不宣称搜索穷尽全部2026文献。进入论文前，应将候选方法与新出现的评分分布/VLM reward工作继续核对，不能仅凭标题相似判定无人做过或已被完全占据。


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
