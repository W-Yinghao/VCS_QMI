# VCS · VL 六方向研究与实施全稿

2026-10-11 · CVPR V6 / r7。各章节区分文献事实、项目来源与本轮研究建议。完整执行先读`01_SERVER_AGENT_PROMPT_CN.md`。



---

<!-- source: docs/02_EXECUTIVE_PLAN_CN.md -->


# 1. 这轮要作出的研究选择

**不再把VL等同于RefCOCO候选排序。**六个方向各有独立综述与实施方案：视觉证据压缩、事实支持、条件生成分布、多对多匹配、数据策展、视频时间支持。每条路线都规定统计对象、完整任务、外部基线、数据与评价方法。新方案不承诺VCS一定超过Logistic，也不把RPC已有的目标重新命名为新理论。

包内注册**67项论文/基础来源、35个数据或协议条目**。这些数量包含基础理论、数据论文和共用数据版本，不等于67项已复现方法或35套独立图片。核验深度逐项记录；方法页摘要、作者项目和完整正文分别标记。

| 优先级 | 方向 | 第一项实际决定 |
|---|---|---|
| 优先考察 | D1 问题条件证据压缩 | 是否能在最近信息论剪枝之外，形成风险驱动的完整压缩方法？ |
| 优先考察 | D2 事实支持与忠实性 | 是否有足够的人类标签，使视觉新增证据改善完整回答？ |
| 先作数学/数据核对 | D3 条件生成分布 | 能否定义合理reference并测出条件错配，而非生成器纹理？ |
| 可复用备选 | D4 多对多图文关系 | 正例不完整与歧义是否确实影响任务？ |
| 资源条件备选 | D5 数据选择与再描述 | 能否完成多个student的同预算训练？ |
| 可实施备选 | D6 视频时间支持 | 需要新的时间支持对象，还是仅换定位loss？ |

**建议首轮只完成D1/D2的接入与最近基线核对，再选择一条主线；D3可并行做CPU有限模型检查。**不自动启动六套大规模训练。

## 与已有项目衔接

仓库资料截点：`87a847e`，2026-10-11 18:59 UTC。它记录了r6数据/许可检查，而不是新数据已下载或新方法已有结果。服务器先核对当前HEAD；r5-QN、r5-X1及r6已批准作业不擅自取消、不重复提交。REG/GAIN是上一包的候选线，本包只补VL内部六方向，不改论文版本。




# 2. 理论怎样进入应用

保留合作者的统计构造：

$$
M=\tfrac12(P+Q),\quad \eta=\frac{p-q}{p+q},\quad S=E_M\eta^2,
$$
$$
J(T)=E_PT-E_QT-\tfrac12E_PT^2-\tfrac12E_QT^2,
\qquad S-J(T)=E_M(T-\eta)^2.
$$

原计划允许一般的P/Q；联合对边缘乘积才是QMI实例。D2的语义支持、D3的条件生成分布、D5的目标与候选池比较，不应全部统称原始QMI。

**原有结构与新增研究分开。**平衡RPC与基础J重合；RuLSIF与相对密度比有直接关系。Logistic拟合后转换成T，也可以用同一个J、gap和增量读数。因此，新用途的增量必须是明确的对象、可用估计流程或实际决策能力，而不是“现在BCE变成了平方”。[F01-F04；U01 §§4–8]

| 共同要求 | 实施方式 |
|---|---|
| 分清理论与实测 | Bayes恒等式不等于有限模型性能，预期优势单列假说 |
| 分开拟合与评价 | 按原始image/video/document/prompt分组；多pair不是更多独立样本 |
| 报告有限模型误差 | h=(C−a)²−(C−b)²；d=(b−a)²；二者残差同时记录 |
| 保留强控制 | 同架构Logistic、同目标核方法和实际任务强baseline |
| 使用完整任务终点 | 回答/生成/检索/重训student/定位；J提高不能替代 |
| 明确成本与监督 | selector、额外teacher、裁判、captioner和特征提取均计入 |

**原始经验读数不裁成好看结果。**负的J仍保存；小J不能认证分布相同。模型分歧不必等于信息增益。参考分布改变后，不直接比较原始散度大小。推导、基准数据和自然数据结果分表。

详细来源与用途见 `docs/03_THEORY_AND_CLAIM_MAP_CN.md`、共同评价规范及各方向综述。



# 3. D1 / D2：优先考察的完整VL方法

## D1：问题条件下的视觉证据压缩

**主问题：**删掉某些视觉token后，当前问题的回答风险增加多少？用这个估计决定保留位置与预算，而不只是使用attention大小。

**最近工作：**FastV、SparseVLM、FEATHER、PyramidDrop；必须特别核对IF-Prune（CVPR2026），以及DiffPrune、ET-Prune（2026年8月预印本）。后两者已经涉及信息削弱、证据预算，不能写成空白领域。[T01-T07]

**数据与指标：**GQA balanced accuracy；TextVQA官方soft accuracy；ChartQA relaxed accuracy并区分human/augmented；SP-DocVQA ANLS。统一图片分辨率、prompt、输出长度。成本分开vision、selector、prefill、decode，主表报告总时延和绝对任务分，不用token减少率冒充加速。

**服务器首轮：**复现一个完整宿主与一个官方剪枝baseline；验证有限风险残差；3种相同结构的选择器（VCS、Logistic→共同读数、直接风险回归）各3seeds；输出完整回答与4个固定预算曲线。9个训练单元是主线选定后的范围，不是本包已启动。

**预期优势：**任务条件风险更适合动态预算。反例控制是直接风险回归与最近信息论剪枝；若它们已解释全部收益，就不能把收益归给CS。

## D2：视觉事实支持与生成忠实性

**主问题：**区分语言上合理与图像真正支持的声明，再用支持证据改善完整回答。

**最近工作：**M-HalDetect、VCD、Woodpecker；SCI的反事实推理；TIFA/VQAScore用于生成图忠实性。[H01-H11]

**训练/评价：**M-HalDetect训练核验器；SNLI-VE可补3类语义结构但不能随意合并标签。POPE三设置、HallusionBench三种聚合、AMBER生成与判别分别保留；COCO caption上同时看CHAIR与内容覆盖。未知不作为反对证据，拒答不作为正确回答。

**服务器首轮：**先查人类标签、source-image分割与纯文本捷径；同架构VCS/Logistic/直接错误预测各3seeds；再在相同候选答案和调用预算下做完整重排/修正。只有证据有效时才进入匹配的LoRA/偏好优化，避免两个生成系统同时搭建。

**预期优势：**低标注、跨回答模型下更可靠的支持概率与更少错误，同时保持覆盖；不是笼统“更安全”或“天然校准”。



# 4. D3 / D4：分布与关系层面的VL用途

## D3：条件生成分布比较

**对象：**P(c,z)=π(c)P_ref(z|c)，Q(c,z)=π(c)P_model(z|c)。同条件权重下S是条件S的平均；CS非线性变换不能与平均交换。

**文献：**CMMD是CLIP空间的MMD，不等于conditional MMD；CFID/cFreD是必须面对的条件分布对照。GenEval、CompBench及++、ConceptMix、GenAI-Bench评价不同的语义/覆盖/人类偏好，不能混成一种真值。[G01-G10；H09-H10]

**服务器首轮：**13个受控条件分布（clean及条件错配、模式缺失、质量变化、混合污染），每条件8/32/128样本、16个条件组、3次抽样。只有VCS/Logistic/同目标核估计可进入同S误差表；MMD/cFreD比较检测与下游分辨，不拿其数值减S。

**自然用途：**参考分布有多个实例时才比较；GenAI-Bench固定release按prompt和generator留出；用额外样本检查选择生成配置的真实效果。单prompt的一张参考图不代表全部正确生成分布。

**预期优势：**有限样本下识别边缘不变的条件错配与模式覆盖缺失。风险是reference不足、来源纹理捷径、强方法已经解决问题。

## D4：多对多与不完整对应

**文献：**PCME、PCME++、ProLIP与FLAIR；LongProLIP是长文本相关扩展，但本轮核为ICLR2025不确定性workshop tiny paper，不写主会。[M01-M08]

**数据：**COCO/Flickr train用于学习；ECCV Caption的mAP@R/R-Precision是主评价；CxC提供扩展语义判断；CUB文字及SUGARCREPE++作为属性/改写补充。ECCV Caption只是测试池子集的扩展标注，不可拿其额外正例训练再报告相同测试。

**服务器首轮：**统一双塔、caption长度、输入和合法正对；VCS、matchedLogistic、multi-positivecontrastive、一个概率嵌入方法，各3seeds完整微调。原始模型始终保留，标签缺失压力只在训练侧进行。

**预期优势：**更好的多正例覆盖与概率风险，而非只改善一个R@1。概率模型的方差头、额外正例标签和损失变换分别做同条件控制；未标注不自动当真负例。



# 5. D5 / D6：数据与时间结构中的VL

## D5：数据选择与再描述

**主问题：**选一组覆盖目标关系的数据，而不是仅保留图文相似度最高的样本。

**文献：**DataComp固定训练预算；DFN/Sieve提供过滤方法；RecapDataComp研究重新描述；SDM（ICLR2026）已经做目标对齐数据过滤，MDM为2026预印本。[D01-D07]

**服务器首轮：**先审候选图片许可、URL可用性、重复图、目标数据暴露和caption来源。所有选择器使用同一个有限候选池、同样数量/曝光预算。完成随机、CLIP过滤、强现有过滤、Logistic同对象、VCS五条策略的完整student训练，各3seeds；要宣称胜过分布匹配，另补kernel/SDM的真实student行。

**评价：**按官方DataComp规模/配方运行才能称相应scale；否则写内部有限池实验。官方38项任务与分项、检索、长尾覆盖及过滤/再描述/student总成本全部保留。筛选AUROC或平均J不是最终成功。

**预期优势：**同预算下的目标适应与覆盖。密度比反演接近T=1会放大误差，必须报告clip/ESS；目标支持不在候选池中时，重加权不能创造它。

## D6：叙述视频的时间支持

**文献：**MIL-NCE处理弱对齐；Moment-DETR/QD-DETR/EaTR/CG-DETR、BM-DETR/Sim-DETR/SG-DETR提供强定位对照；HiREST与MomentSeeker扩展到层次或长视频。[V01-V11]

**数据：**QVHighlights为首轮：MR mAP@[.50:.05:.95]及R@1@.5/.7，HD mAP/Hit@1。第二项选Charades-STA或Ego4D NLQ，不同时铺开所有视频数据。QVHighlights官方README已说明Codalab关闭、test GT公开；这不是允许test调参。

**服务器首轮：**复现官方evaluator的golden fixture；同一CLIP/SlowFast特征上训练完整temporal模型，标准目标、matchedLogistic、VCS各3seeds；保留时间定位输出，不仅分类probe。特征缓存是官方比较协议的输入，结果不能声称包含原始视频编码的端到端速度。

**预期优势：**更可靠地表达多区间、偏移和背景支持。弱监督bag不意味着bag内每帧为真；若构造新的时间模型，Logistic必须获得相同结构与监督。



# 6. 服务器交付与实施规则

## 第一轮交付

| 文件/结果 | 最少包含什么 |
|---|---|
| 接入清单 | 当前HEAD、已有作业、数据/权重、代码环境和授权差异 |
| 近邻方法复现表 | 原文协议、代码版本、我方适配、实际运行状态、差异原因 |
| 数据审查表 | 原始独立单位、split、许可、下载状态、泄漏与test暴露 |
| 两条候选可行性 | 完整推理路径、一个外部baseline、数据样本核验、真实timing |
| 方向选择记录 | 统计对象、实际用途、与最近工作的差异、预算/风险 |
| 冻结实验协议 | 方法清单、CAL规则、primary endpoint、种子、数据指纹 |

**只选择一条主线与至多一条轻量数学验证线。**已有r5/r6工作正常接续；新方向不复制旧的近似平局搜索，不通过删除强对照制造预期优势。

## 必须报告的四种层次

原文结果、我方官方复现、我方改编结果、本方案假说，四者分列。各方向没有移植的runner、没有下载的数据和没有复现的checkpoint，明确标为未执行。补充大模型/teacher/captioner成本不隐藏。

原始脚本不由自制指标替代。TextVQA不是简单exactmatch，DocVQA不是VQAaccuracy，AMBER的判别正类方向与POPE需区分，ECCVCaption不是COCO5kR@1，QVHighlights不是任意阈值的单区间命中。

报告区间不等于证明两方法等效；反复读取DEV后取得的效应保持探索性。最终独立评价只在方案冻结后进行，测试集由数据公开变为可下载不改变其用途。

## 阅读导航

- 六份综述：`directions/D*/LITERATURE_REVIEW_CN.md`。
- 六份实施：`directions/D*/SERVER_EXPERIMENTS_CN.md`。
- 数据集和来源：`datasets/`、`literature/`；每条有原始URL与核验深度。
- 统一基线、指标与任务：`benchmarks/`、`machine/`、`templates/`。
- 数学/指标边界的CPU核对：`reference/`、`tests/`、`validation/`。

**本包不包含新的自然数据实验结果，不代替已发布方法的全文与许可，不打包第三方权重/数据。**它的用途是让服务器从明确的文献和评价协议出发，完成可以推翻也可以支持研究假说的实验。



---

<!-- source: docs/03_THEORY_AND_CLAIM_MAP_CN.md -->

# 共同理论对象与创新边界

## 1. 继承的工具，不重新命名为首次发明

对任意两个概率分布P,Q，M=(P+Q)/2，η=(p−q)/(p+q)，原计划[U01]定义：

\[
S(P,Q)=E_M\eta^2,\quad J(T)=E_PT-E_QT-\tfrac12E_PT^2-\tfrac12E_QT^2,
\quad S-J(T)=E_M(T-\eta)^2.
\]

混合参考CS为 \(2\log((1+S)/(1-S))\)。只有P=联合、Q=其边缘乘积时，本包才称QMI。训练或比较有界S，不必强制执行对数变换；S接近1时变换敏感。独立评价的经验J可负、可高于总体S；这是采样误差，不可裁剪后宣称精确下界。[U01 §4–10]

[F01]平衡RPC的目标和最优critic相同，平方gap是完成平方；[F02]相对密度比、[F03]正规损失也是先例。由logistic拟合D后令T=2D−1，同样可使用J、gap与平方增量。不能将共有后验、后验可校准或标准jackknife宣称为VCS独占创新。

## 2. 六个对象不是同一个互信息

| 路线 | P与Q的具体含义 | 主要用途 | 能否直接叫QMI |
|---|---|---|---|
| D1 | 同一二元任务的C=+1与C=−1观测分布 | 完整/压缩观测的任务证据变化 | 否，是任务两分布判别；多类Brier另标扩展 |
| D2 | 人工/可靠标注的支持与反驳声明 | 语义支持、核验与生成选择 | 否；支持未知不能混成反驳 |
| D3 | 同条件权重的参考与生成观测 | 条件分布差异 | 否；不是图文联合对随机配对 |
| D4 | 规定的语义匹配与非匹配分布，或明确联合对乘积 | 多对多关系与未知标签 | 依选择而定，必须逐臂标记 |
| D5 | 目标关系参考与来源/选中关系分布 | 数据覆盖、选择或重加权 | 通常否 |
| D6 | 时间片段支持/不支持查询的标注分布 | 支持场、区间定位 | 通常否；时间不确定性模型另定义 |

## 3. 从任务风险到观测增量：成立条件

先选择一个平衡的二元实验C∈{−1,+1}。B是完整观测，A=g(B)是嵌套观测；压缩mask、位置、随机数种子在需要时包含于观测。使用同一个M和同一个任务先验：

\[
\Delta=S_B-S_A=E_M(\eta_B-\eta_A)^2
=\inf_a E(C-a(A))^2-\inf_b E(C-b(B))^2.
\]

这是条件期望的标准L²投影结果，不是新互信息链式法则。若两个输入不嵌套、标签先验不一致、查询被更换或另做独立标准化，这个比较必须重新推导。

实际有限预测a,b则：
\[
(C-a)^2-(C-b)^2=(b-a)^2+2(C-b)(b-a).
\]

右侧第二项不能默认消失。D1/D2既要记录平方分歧，也要用独立标签检查真实风险改善。后验误差不明时，两者都不是精确“信息量”。用同一错误教师的预测作唯一真值会奖励保留错误。

多类别答案可以使用一热Y和 \(E\|p_B-p_A\|^2\) 的Brier投影结构；它是另一个任务风险量，不能直接把数值代入原始二元CS变换。自由文本token NLL也不能自动替代这个定理中的平方风险。

## 4. 条件生成比较

若P(c,z)=π(c)p(z|c)，Q(c,z)=π(c)q(z|c)，则
\[
S(P,Q)=\sum_c\pi(c) S(P_c,Q_c).
\]
同条件权重必须在抽样和估计中实现。若模型和参考的prompt比例不同，critic可只靠c判别。CS非线性变换不能与条件平均交换。

在自然图片上，有限组参考样本并不定义所有“合理图像”的总体。自定义生成器可以有精确有限分布，但需明确观测映射；标签可识别时才能把标签信道真值当成观测真值。一般只有DPI关系，不能编造自然CLIP特征的真互信息。

## 5. 相同信息预算与参考分布

D2改为语义支持时，(1+T)/2才是规定平衡类别问题中的概率；部署先验变动需要校准，输入分布变化不靠一个截距即可解决。D4/D5中由T反解r=(1+T)/(1−T)在T接近1时不稳定；“T有界”不等于重要性权重有界。记录clip、effective sample size和丢弃质量。

D5最小化低质量critic给出的J并不能认证分布匹配；独立更强critic、核检验和最终学生都要验证。D6中未标注片段不必然为负，叙述错位不是随机类噪声。

## 6. 可以争取的贡献与不能预先写出的贡献

本包提出的是候选用途：任务风险可辨的token预算；语言先验与视觉证据分开的核验；条件覆盖/忠实性共同控制的分布评价；漏标关系学习；目标覆盖数据策展；时间支持模型。它们需要方法、对照与独立任务结果才能成为贡献。

不能写：首次信息论剪枝、首次视觉反事实核验、首次条件生成评价、首次概率图文嵌入、首次目标分布过滤、首次叙述视频弱对齐。对应直接先例分别见[T05–T07]、[H01/H04]、[G02/G03]、[M01–M03]、[D05]、[V01]。

失去某一个领先条件只收窄对应主张，不自动关闭VL或某投稿场馆。相反，换方向也不能免除RPC及领域先例的归属审查。



---

<!-- source: docs/04_SHARED_EVALUATION_CN.md -->

# 共同评价、数据与资源规范

## 1. 三层对照必须同时有，不能互相替代

**同条件组件层。** 同输入、同可训练函数类、相同数据/PQ、相同初始化分配与调优机会，比较VCS平方拟合、平衡Logistic以及合理的核/密度比估计。同一个T最终共用J时，分别记录fit objective与readout objective。平衡RPC恒等配置作实现check，不重跑成不同方法。

**任务方法层。** 每条路线保留最接近的已发表方法；若复现需要不同的预训练特征或外部教师，不把它当纯损失比较。官方原生代码优先，改变网络/输入后标为适配版本。

**完整系统层。** 最终答案、生成文本、匹配列表、选中数据训练出的学生或视频区间必须被独立评价。内部J、critic功效和teacher一致性不能单独构成应用成功。

## 2. 基础样本而非pair是统计单位

COCO派生集按原image ID，文档按document/page family，视频按原video ID，图像编辑与改述按parent family；生成按prompt/concept family。先划分、再产生mask/caption/负例/渲染/生成样本。

固定FIT/CAL/DEV/FINAL manifest。CAL可以选超参数、checkpoint或校准器；DEV只用于开发期结果解释且必须记录曝光；FINAL只在预注册协议下读。新分割不能消除此前在同一基础图像上的使用。若eval-only数据是POPE、HallusionBench、AMBER、ECCV Caption或GenEval，不能反过来挖负例训练模型。

跨数据源建立`original_asset_id`、`family_id`、`source_domain`、hash/pHash可疑重复列表。没有可获得的预训练清单时标“pretraining exposure unknown”，不得声称完全无污染。

## 3. 选择预算、随机性和置信区间

选定路线的主对照全部运行预定seeds0/1/2，探索种子与确认种子记录分开，不见seed0落后就停止它而只给我们补种子。训练失败也列入表，说明是否因共同硬件/数据错误允许重试。

报告图片/视频/文档/提示级paired bootstrap与训练seed离散度两种不确定性，说明它们各自覆盖什么。样本内很多token/声明/frames不能当独立样本。临界显著结果在三个种子下不升级为普遍定理。

“CI包含0”只能写没检出差异；等效/非劣需先设实际可接受margin，不能事后调margin到通过。多指标与多子集保留完整表，不只突出显著一项。连续预算曲线使用固定横坐标和整曲线摘要，不事后挑一个最优token比率。

## 4. 成本

每个完整系统计入：teacher/评分器计算、vision encoder、selector、prefill、decode、后处理/反事实次数、数据筛选/caption生成及下游训练。训练时teacher成本与部署成本分开，摊销计算的假设明确。

同GPU型号、dtype、分辨率、输入长度、batch和生成长度下实测。报告batch1端到端中位/P95延迟、峰值显存、TTFT、decode速度；FLOPs仅辅助。不用“减少token50%”代替“加速50%”。D1若仍计算完整视觉主干，只能声称后续语言模型节省；D6使用缓存特征不能声称原视频端到端加速。

先测一个完整训练步和一个完整推理循环，再估算预算。小时数未测前留空，不沿用r4小双塔微调的1.3小时估计7B VLM。

## 5. 官方评估与自定义评估分开

锁定 evaluator repo、commit、配置、输入字段、样本清单和数值容差；先跑发布的golden prediction fixture。任何API裁判固定版本、prompt、temperature、缓存与失败重试。只生成平均分而没有原始预测不算完成。

若官方服务停运，使用其已发布本地脚本/groundtruth并记录；公开测试标签不允许用于训练。例如QVHighlights官方README现称Codalab关闭且test真值公开，不能继续要求向已关闭服务上传。[V02]

原始发布分数只用于配置一致的复现锚，不直接写进同一因果对照表。未能复现应查权重、输入/候选、数据子集、预处理与evaluator，不删除不利基线。

## 6. 资源和权限

代码LICENSE、模型LICENSE、数据annotation LICENSE、原始图像/视频权利及API条款分开记录。可下载不等于允许重新分发；本包不附第三方图像、模型、完整论文或字体。下载需账号时不规避；新外部服务或付费裁判按用户已有授权执行。

本包不自动取消R5/R6已批准任务，不自动提交六条线全部网格，不改论文，不打开未用最终测试。服务器依据本包科学优先级和真实既有授权执行范围明确的工作。



---

<!-- source: docs/05_STAGE_PLAN_AND_DECISIONS_CN.md -->

# 阶段安排、预期优势与路线决策

本文件是新的研究建议，不是六套已批准的训练指令。它不修改合作者的原始S/J，也不撤销已运行的r5/r6任务。协议编号使用R7-*，与仓库已有编号映射后再建立runner。

## 1. 首轮先解决三件事

**R7-00 接续。**读取当前HEAD及与87a847e的差异，记录旧线的完成/在跑/待评状态。不得由“GPU空闲”推出六条新线全部必要；已有常驻授权仍按原范围执行，发生范围冲突时记录并向负责人确认。官方保留集不因为数据公开就自动变成开发集。

**R7-10 文献与可执行性。**67项注册均给出核验深度，但不等于67篇逐公式审读或完成复现。服务器先通读拟选择路线中至少两篇最接近方法的实验章节、附录与实际代码，填准训练数据、宿主、评价脚本和计算范围。只有摘要/仓库的行必须补读全文；原实现未发布时记录缺口，不假装复现。

**R7-20 两个候选的完整链路。**建议优先D1与D2做数据/完整推理路径和直接竞争方法复现；D3可以并行做CPU有限条件分布核对。这不是三个完整研究网格。输出能够运行的最小pipeline、实际资源测量，以及决定一条主线的依据。

## 2. 方法优势必须写成可检验假说

| 方向 | 期待的优势（本方案假说） | 要观察的终点 | 最有力的替代解释/对照 |
|---|---|---|---|
| D1证据压缩 | 任务条件下的风险损失，比注意力/无条件重要性更适合预算分配 | 固定真实时延的答案质量；固定质量的总成本；预算校准 | IF-Prune/ET-Prune及同结构直接风险回归；额外teacher调用带来的收益 |
| D2事实支持 | 区分语言合理性与视觉支持，可减少跨模型回答错误而不降低有效覆盖 | 独立支持标签Brier/PR；生成CHAIR与Cover/任务质量；跨生成器 | 同架构Logistic/无图模型；VCD/现有核验；拒答造成的假改善 |
| D3条件分布 | 条件级学习型critic在有限样本下比固定全局统计更能分辨条件错配与覆盖不足 | 已知有限S误差；条件置乱/模式缺失检测；人评相关性 | 同目标核回归、grouped MMD、cFreD、Logistic→J；来源纹理捷径 |
| D4多对多 | 明确标注不完全与歧义后，关系证据能减少误排且表达风险 | ECCV mAP@R/R-Precision；CxC；已判定对Brier；原始检索不退化 | PCME++/ProLIP；同probabilistic架构换loss；额外正例监督 |
| D5数据策展 | 条件/目标分布匹配比仅选高相似度样本更能保留有用覆盖 | 完整student的同预算下游质量；分项和尾部覆盖；全部筛选成本 | DFN/Sieve/SDM及kernel匹配；更大captioner；数据重复/曝光差异 |
| D6时间支持 | 显式的支持分布和可核验后验可以处理多区间与错位，而不是把bag内全片段当正例 | 标准MR/HD及错位压力曲线；跨视频泛化；时间支持校准 | QD/CG/BM/Sim-DETR、MIL-NCE；更强视频特征和额外ASR监督 |

数字预测不凭空规定。区间/效应分析回答实际差异；性能保留门槛由任务价值事先设定。零差异检验不显著不等于等效。“未赢过JS”只限制该loss的独有优势，不自动否定所有CS读数用途；反之，换一种任务取得小幅提升也不自动解决RPC新颖性。

## 3. 三层比较，不混成一张排行榜

1. **相同信息和函数类的因果式控制。**VCS、Logistic→同J、适用的同目标核/相对回归；一致FIT/CAL、独立样本量、训练预算和选择机会。
2. **相同任务的强系统。**用官方实现的pruning、verifier、conditional metric、probabilisticretrieval、datafilter、temporalgrounder。模型/训练数据不同必须分栏，不能把系统差异归给散度。
3. **外部用途。**完整回答、生成质量、检索、重训student、时间定位等。不能用训练critic同一评分证明自己有效。

理论本身及其恒等式与RPC/正规评分规则的关系使用F01-F04。单独把现有二次loss放入宿主不命名新estimator；只有研究得到的新增对象、估计方法、决策能力或可重复认识才进入贡献。

## 4. 推荐的有限顺序

- **D1 / D2接入后选主线。**分别审清最近方法重叠、数据合法性和一次完整推理。主线候选不是按一个seed的准确率，而按可回答的问题、独立评价和可实施性决定。
- **D3优先解决reference。**允许无GPU的有限模型核对；没有合理条件reference时，不提前为自然图片定义oracle，更不启动大批生成。
- **D4 / D6作为备选。**一旦转入，完成任务系统；不把已有双塔或cached features的便利当作科学理由。
- **D5不默认启动。**需要真实的多个student训练成本；筛选信号图不足以证明数据选择有效。

主线确定后，将训练/选择集、primary endpoint、方法清单、种子与数据指纹冻结，完成全部预定单元。新增探索另编号，不覆盖历史选择规则。

## 5. 风险与转向

D1若与IF-Prune/ET-Prune的区别只剩损失换名，改进研究对象或转D2，而不是删最近工作。D2若下降仅来自短答/拒答，先修coverage评价；若支持标签不足，优先补标注核验而非使用POPE作训练集。D3若conditional MMD/cFreD已解决全部受控条件，保留事实，不专挑其失败切片。D4若多义标签不可识别，限定为已判定关系，禁止将未标注硬当negative。D5若预算不够完成student对照，延期而非只报过滤AUROC。D6若收益只是更强视频特征，增加同特征控制再决定。

这些分支用于诊断为何结果不理想，不用于保证最后一定产生正面故事。每条结论只覆盖实际实验与假设。



---

<!-- source: directions/D1_token_evidence/LITERATURE_REVIEW_CN.md -->

# D1 文献综述｜问题条件下的视觉证据压缩

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 领域是怎样发展的？

这条线研究长视觉序列的计算冗余，但“冗余”有三种不同定义：注意力较少、与当前问题不相关、删掉后任务风险几乎不变。三者不等价。我们希望研究第三种，不能仅凭换一种token重要性评分就认为完成了风险估计。

早期FastV[T01]利用模型内部注意力进行免训练裁减；SparseVLM[T02]引入文本raters、逐层预算和回收。FEATHER[T03]进一步说明：某些方法在通用QA均值上看似安全，却损害OCR/定位等视觉密集任务。这使最终比较必须同时看真正需要局部视觉信息的问题及整系统时延，而不是一张平均保持率表。

IF-Prune[T05]已经将小VLM和变分信息瓶颈用于问题条件化剪枝。2026年预印本DiffPrune[T06]又通过任务驱动可微节流训练选择器；ET-Prune[T07]明确使用“证据感知”和动态预算。因此，“信息论剪枝”“问题相关”“动态budget”都不是空白。最值得继续问的是：**能否预测压缩造成的实际任务风险，并在预算决策中比这些代理信号更有用？**

## 2. 核心文献与实验对象

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [T01] **An Image is Worth 1/2 Tokens After Layer 2: Plug-and-play Inference Acceleration for Large Vision-Language Models** — ECCV 2024 | FastV利用早期层后注意力裁减视觉token。 | LLaVA系; Qwen-VL; 视频任务；原任务表现; FLOPs; 推理延迟。 |
| [T02] **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference** — ICML 2025 | 文本raters、逐层稀疏与token回收；免训练。 | GQA; MMBench; MME; POPE; ScienceQA; SEED; TextVQA; MMVet；官方任务分数; 保持率; FLOPs; CUDA时间。 |
| [T03] **Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration** — ICCV 2025 | 分析视觉中心任务失效；RoPE-free注意力与两阶段裁减。 | 通用QA; TextVQA; 视觉定位等；任务质量与加速。 |
| [T04] **Conical Visual Concentration for Efficient Large Vision-Language Models (PyramidDrop)** — CVPR 2025 | 分阶段减少视觉token，训练/推理设定分别处理。 | 多项图像问答任务；任务准确率; 训练/推理成本。 |
| [T05] **IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models** — CVPR 2026 | 小VLM的变分信息瓶颈提供问题条件下的视觉裁减信号。 | 8项评测；官方README明确含ChartQA，Qwen2-VL/InternVL系；任务表现保持率; token/计算成本。 |
| [T06] **DiffPrune: Differentiable Information Throttling for Token Pruning in Vision-Language Models** — arXiv preprint, August 2026 | 可微信息节流训练token评分器；以任务损失学习裁减。 | LLaVA1.5-7B; LLaVA-NEXT7B; Qwen2.5-VL7B; GQA/TextVQA等10任务；固定预算任务质量; 实际效率。 |
| [T07] **ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs** — arXiv preprint, August 2026 | 问题条件证据评分、文本空间保护与动态预算下限。 | Qwen3-VL8B; InternVL3.5-8B; OCRBench-v2; MMBench1.1；OCR/QA; token数与时延。 |
| [F08] **Qwen2.5-VL Technical Report** — 2025 technical report / official model card | 动态视觉分辨率的开放VLM；作为D1/D2宿主。 | 文档/图表/OCR/通用视觉；各官方任务指标。 |

## 3. 应该重点读哪几篇，而不是全部平均用力？

**IF-Prune是方法定位的必读邻居。** 它使用VIB估计视觉token的信息信号，我们的方法若只是把KL改成CS，理论对象是否仍与“删掉后回答变差”对应，需要证明而不是命名。原摘要的“保留95%表现”是相对baseline，不是95%答案正确。

**DiffPrune是训练设计的直接邻居。** 它说明任务损失已经进入裁减训练。我们不能把“可微选择器+任务监督”据为己有。合理增量可能是：离线独立评价的风险损失、有限预测误差的校正，以及对跨问题预算的可靠分配。但这些都要比同监督量的直接风险回归更好，不能只赢注意力启发式。

**FEATHER决定评价不能太容易。** 文本细节和位置约束应作为预定任务，不是做完后再挑出来的特殊子集。TextVQA、ChartQA、SP-DocVQA都需要原生evaluator；图表数值容差和文档ANLS不能混成“VQA accuracy”。[B02–B04]

## 4. 基线分层

首轮必须有完整未剪枝模型、随机/均匀裁减、FastV、SparseVLM或FEATHER中的至少一条适配成功的强免训练方法；新增训练方法需要匹配的直接风险选择器和Logistic后验选择器。进入正文候选必须补IF-Prune，或明确给出无法复现的原因与同架构替代。T06/T07即便没有现成代码，也要纳入新颖性讨论，不必盲目重写大型未发布系统。

每种方法在同一host上的比较与原作者published host分开；层内attention读取可能改变FlashAttention执行路径，不能不计它的真实开销。更换更强7B宿主本身不算剪枝方法提升。

## 5. 数据与预期优势

GQA提供组合问答与结构化辅助标签，适合二元任务风险验证；TextVQA检验小文字；SP-DocVQA检验页面信息；ChartQA检验视觉加数值推理。MMBench/ScienceQA作为覆盖面补充，不能替代视觉密集任务。建议主宿主为固定revision的Qwen2.5-VL-7B，第二宿主选已经有强剪枝实现的LLaVA1.5-7B；模型规模不是“最新越大越好”。[F08,T01–T07]

**希望验证的优势：**同一token/时延预算下更少任务损失，或同一质量下使用更少计算；被预测的损失与实际损失在不同任务难度中相关且可校准。**不能预定：**CS必然比KL/BCE好，低J损失等于答案不变，或每种任务都能保留相同比例token。

## 6. 风险与转向

若直接风险回归与VCS没有区别，新增贡献可能来自选择器而非散度；若text-rich任务损害明显，优先定位token位置/分辨率处理和训练覆盖，不能改评测分母。若选择器成本吞掉节省，本方向应转为训练期证据分析，不能宣称部署加速。若无法获得超越IF-Prune/DiffPrune/ET-Prune的具体能力，不把它升为主贡献。

## 本方向主来源

- [T01] [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764)。代码/项目：https://github.com/pkunlp-icler/FastV
- [T02] [SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](https://proceedings.mlr.press/v267/zhang25s.html)。代码/项目：https://github.com/Gumpest/SparseVLMs
- [T03] [Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](https://arxiv.org/abs/2412.13180)。代码/项目：https://web.stanford.edu/~markendo/projects/feather
- [T04] [Conical Visual Concentration for Efficient Large Vision-Language Models (PyramidDrop)](https://arxiv.org/abs/2410.17247)。代码/项目：https://github.com/Cooperx521/PyramidDrop
- [T05] [IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.html)。代码/项目：https://github.com/snap-research/EVLM-IF-Prune
- [T06] [DiffPrune: Differentiable Information Throttling for Token Pruning in Vision-Language Models](https://arxiv.org/abs/2608.01985)。本轮未确认可直接运行的官方实现。
- [T07] [ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs](https://arxiv.org/abs/2608.01979)。本轮未确认可直接运行的官方实现。
- [F08] [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)。代码/项目：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct



---

<!-- source: directions/D1_token_evidence/SERVER_EXPERIMENTS_CN.md -->

# D1 服务器实验方案｜问题条件下的视觉证据压缩

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS01, DS02, DS03, DS04, DS05, DS06；全部定义见 `datasets/DATASET_ATLAS_CN.md`。

## A. 要完成的系统

输入图像I和问题q，产生保留的token集合K与budget，再由一个固定家族VLM生成答案。选择器不能访问真答案，不能先运行完整大VLM获得答案后才决定K。报告完整vision encoder、selector、prefill和decode总成本。

**首轮宿主建议：**Qwen2.5-VL-7B-Instruct固定revision、分辨率上下限与图片处理器；完整回答模型不是本地本轮已安装组件。可行性阶段验证一次完整推理后再计时，不能从r4双塔耗时推算。

## B. 理论和可实施近似

先在GQA的可验证二元问题定义C∈{−1,+1}，平衡任务类先验。完整观测B=(q,V)，压缩观测A=(q,V_K,K)。对每种固定mask机制拟合完整/压缩critic；使用原J等价的平方风险，对照相同函数类的Logistic。保存正负权重与每种mask的抽样概率。

有限模型给出b(B)、a(A)。每个验证样本同时保存d=(b−a)²和真实损失差h=(C−a)²−(C−b)²，另保存残差h−d。d不是可靠性的保证；h可负。教师Brier差很大时不得强行截断成正“信息损失”。

选择器z(q,V_cheap,K_budget)预测压缩损失；主选择规则是CAL上固定风险容许水平下的最小budget，或固定budget下选择K。预算动作先固定为保留{25%,50%,75%,100%}，另以12.5%作可选stress，不根据DEV赢家改变网格。mask学习可以用连续gate训练、hard topK部署，但记录soft/hard落差。

训练critic、生成策略监督与评价用grouped out-of-fold方式隔离。共有完整/压缩critic可共享主体，但mask只作为观测，不给它目标答案。第一阶段二元核对不冒充全任务结论；自由回答阶段采用标准answer NLL/LoRA训练，CS风险分支只是明确的辅助/选择信号，多类Brier扩展单列。

## C. 数据与角色

从官方GQA train按原image/scene graph划出FIT/CAL；形成二元风险校准样本及完整问答样本。示范性规模为10k个训练图片组（不是10k pairs），CAL与DEV图片不重叠；若资源只支持更小规模，应在首次比较前冻结且所有方法相同。每个训练问题固定3种mask试验，完整基线输出缓存一次；不做无限mask搜索。

TextVQA/ChartQA的官方train仅在适配分支获得明确同信息预算后加入；对应val/eval不进入选择器训练。SP-DocVQA作为跨文档验证或第二训练域，按文档分组。M-HalDetect/POPE等与COCO重叠要先审计。

## D. 实验顺序与运行单位

| 阶段 | 操作 | 必要输出 |
|---|---|---|
| D1-00 | 完整宿主与一个官方剪枝baseline复现；32个样本fixture只用于接口 | raw predictions/evaluator hash; token counts; 真实延迟 |
| D1-10 | 二元风险对象、嵌套mask、有限teacher残差验证 | h/d/残差和预算分层，独立CAL，不选“最好看”的指标 |
| D1-20 | 完整训练3种同架构选择器：VCS风险、Logistic后验→同平方读数、直接实际风险回归 | 每种3seeds，共9个训练单元；输出完整回答，不是只输出probe |
| D1-30 | 所有方法在固定4budget曲线上评估 | GQA/TextVQA/ChartQA质量与成本，IF-Prune等完整baseline |
| D1-40 | 冻结后跨宿主/文档确认 | 只扩展最终方案及主要强对照，不自动重跑所有组合 |

相同mask试验、teacher调用数、训练steps与优化选择机会。随机/均匀/FastV类无需训练，仍有完整推理成本行。9个训练单元是proposal，先实测资源并与授权核对，不在本包执行。

## E. 指标与消融

主终点：固定平均真实时延下的官方任务质量，及固定质量容许差距下的时延。次终点：TTFT、每token解码、峰值显存、视觉token数、预算分布；绝对准确率和相对保持率一起报。

消融只选3类：移除问题条件；用平方分歧d替代真实风险h的校正；固定预算替代动态预算。每项固定同训练信息和候选mask，不把额外teacher supervision混进方法差异。完整模型本身错误时，分别报告压缩保留错误与纠正错误的比例。

## F. 何时转向

若有限h−d大且不可预测，先用实际风险策略作为实用参考，不能靠“平方gap”声称已准确估计信息。若同架构直接风险回归占优，研究贡献需转为对后验增量局限的分析，或者停止这一应用扩展。若预算网格完全没有质量保持空间，检查模型的token接口/位置信息和baseline，不只继续调critic。

判定标准是完整quality–cost曲线和独立验证，不是指定必须领先若干点。最接近工作T05/T06/T07未充分复现/定位前，不写“新的信息论剪枝范式”。



---

<!-- source: directions/D2_factual_support/LITERATURE_REVIEW_CN.md -->

# D2 文献综述｜视觉事实支持与生成忠实性

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 三种研究问题不能混为“幻觉”

图文相似性衡量话题相关；语义支持判断图像是否支持一个声明；生成忠实性评价完整答案有没有错误且有没有回答问题。我们希望研究后两者，不能把原先joint-vs-product的配对概率直接改名为真实性。

CHAIR[H08]和POPE[H05]使对象幻觉可量化，但不覆盖所有关系与属性。HallusionBench[H06]用图像/问题配对揭示语言先验与视觉错觉；AMBER[H07]同时看生成与判别，特别要求保留信息覆盖。这些工作共同说明：总说“不确定”或生成极短答案不能算成功。

## 2. 核心文献

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [H01] **Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding** — CVPR 2024 | 原图和扰动图条件的输出对比，免训练减轻幻觉。 | POPE等对象幻觉与通用理解；幻觉指标; QA质量; 额外forward成本。 |
| [H02] **Detecting and Preventing Hallucinations in Large Vision Language Models** — AAAI 2024 | M-HalDetect细粒度标注；奖励/偏好优化与best-of-N。 | M-HalDetect; 多源模型回答；声明/片段正确性; 人评; 生成质量。 |
| [H03] **Woodpecker: Hallucination Correction for Multimodal Large Language Models** — 2023 arXiv; venue not re-verified here | 结构化核查与多步骤生成纠错。 | 图文问答/幻觉评测；纠错效果; 信息保持; 推理成本。 |
| [H04] **Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework** — CVPR 2026 | SCI多轮视觉/文本反事实；DRBench动态压力测试。 | DRBench及VLM鲁棒性任务；正确性与测试时计算。 |
| [H05] **Evaluating Object Hallucination in Large Vision-Language Models** — EMNLP 2023 | POPE对象存在性问答，random/popular/adversarial负例。 | COCO来源图像等发布子集；Accuracy; Precision; Recall; F1; Yes ratio。 |
| [H06] **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models** — CVPR 2024 | 成对控制图像/问题诊断语言幻觉与视觉错觉。 | HallusionBench；aAcc; fAcc; qAcc; 分组诊断。 |
| [H07] **AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation** — 2023 arXiv; venue not asserted | 生成和判别两种幻觉评价，不依赖生成式LLM裁判。 | AMBER；CHAIR; Cover; Hal; Cog; Acc/P/R/F1。 |
| [H08] **Object Hallucination in Image Captioning** — EMNLP 2018 | CHAIR用对象提及检测图像描述幻觉。 | COCO captions；CHAIR_i; CHAIR_s; caption质量。 |
| [H09] **TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering** — ICCV 2023 | 把提示词拆成问答，检查生成图是否满足要求。 | TIFA评测集; 多个T2I模型；VQA问答一致性; 人类相关性。 |
| [H10] **Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)** — ECCV 2024 | VQA模型对“是否显示该文字”的Yes概率作为配对评价。 | 多个图文/生成对齐benchmarks；人类偏好相关性/成对排序。 |
| [H11] **Visual Entailment: A Novel Task for Fine-Grained Image Understanding** — 2019 arXiv / SNLI-VE dataset | 图像前提与文本声明的entailment/neutral/contradiction。 | SNLI-VE (Flickr30k + SNLI)；3类accuracy; 本方案另加macro-F1/校准。 |

## 3. 方法线索与真正的竞争者

VCD[H01]利用原图/扰动图的输出差异，说明“视觉对比”不是新点。SCI[H04]进一步使用多轮视觉和文本反事实，意味着“自我核验”也不是空白。它们的多次推理需按同预算比较，而不能让我们的verifier多看几张图、多生成几个答案后与单次baseline比。

M-HalDetect[H02]提供细粒度人类标注并将它用于检测、奖励与偏好优化，是最直接的训练数据/方法参照。我们更有价值的提问是：**把语言本身可预测的支持倾向与视觉新增证据分开，会不会使未见模型回答的核验更准确，并转化为真实的生成改善？**

TIFA[H09]、VQAScore[H10]已把VQA概率用于生成图忠实性评价。采用VCS给候选图打分不是足够的新方法；需要在独立人类标签上证明不同于单张CLIP/VQA分的能力，并在best-of-N或有限生成适配中体现。D2首轮优先caption/VQA核验，T2I奖励保留为第二应用，避免两个生成系统同时搭建。

## 4. 数据应怎样选择？

训练优先M-HalDetect，必要时用SNLI-VE[H11]补充supported/neutral/contradicted语义结构。二者任务标签并不完全相同，不能直接拼接为一组二值数据。POPE、AMBER、HallusionBench只用于独立评价，不提供方便的训练负例。COCO生成caption上用CHAIR配合内容质量，而不是只看CHAIR下降。

AMBER判别P/R/F1以“真值no的幻觉问题”为正类，这是原文定义；POPE的yes/no统计不能原样搬过来。HallusionBench必须保留成对聚合fAcc/qAcc，不能把aAcc当整个benchmark。[H05–H08]

若走T2I评分，则使用BaiqiL版GenAI-Bench/Rank[G08]并留出prompt和generator。人类喜欢某张图不等于图像支持每个声明；偏好、细粒度事实与审美分表。

## 5. 创新与预期优势边界

候选新增点是：有限标注下，估计视觉相对纯文本的证据增益，并将可靠的支持分数用于纠错/生成，而不是简单的二元平方损失。原J保证的后验与平方风险结构可以帮助分析，但不保证生成偏好优化稳定或泛化。

期待的优势是较低的支持概率风险、更强的跨回答生成器泛化，以及在相同覆盖/计算预算下更少错误。必须保留同架构Logistic、直接错误预测、无图/错图控制、VCD与一项现有纠错/奖励方法。若收益消失于这些匹配控制，只能报告额外监督或系统结构收益。

## 6. 实际可行性

D2比D1少一些token内核兼容性风险，且存在人类错误标注；但语义unknown、文本偏差和生成评估很容易污染结论。首先审样本和标签，再训练。最危险的假成功是verifier学到回答风格、critic给自己的输出打高分、或者通过拒答降低幻觉。三个都需要独立外部终点才能排除。

## 本方向主来源

- [H01] [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)。代码/项目：https://github.com/DAMO-NLP-SG/VCD
- [H02] [Detecting and Preventing Hallucinations in Large Vision Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/29771)。代码/项目：https://github.com/hendryx-scale/mhal-detect
- [H03] [Woodpecker: Hallucination Correction for Multimodal Large Language Models](https://arxiv.org/abs/2310.16045)。代码/项目：https://github.com/BradyFU/Woodpecker
- [H04] [Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Scaling_Test-Time_Robustness_of_Vision-Language_Models_via_Self-Critical_Inference_Framework_CVPR_2026_paper.html)。本轮未确认可直接运行的官方实现。
- [H05] [Evaluating Object Hallucination in Large Vision-Language Models](https://aclanthology.org/2023.emnlp-main.20/)。代码/项目：https://github.com/AoiDragon/POPE
- [H06] [HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models](https://arxiv.org/abs/2310.14566)。代码/项目：https://github.com/tianyi-lab/HallusionBench
- [H07] [AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation](https://arxiv.org/abs/2311.07397)。代码/项目：https://github.com/junyangwang0410/AMBER
- [H08] [Object Hallucination in Image Captioning](https://aclanthology.org/D18-1437/)。代码/项目：https://github.com/LisaAnne/Hallucination
- [H09] [TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_TIFA_Accurate_and_Interpretable_Text-to-Image_Faithfulness_Evaluation_with_Question_Answering_ICCV_2023_paper.html)。代码/项目：https://github.com/Yushi-Hu/tifa
- [H10] [Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)](https://arxiv.org/abs/2404.01291)。代码/项目：https://github.com/linzhiqiu/t2v_metrics
- [H11] [Visual Entailment: A Novel Task for Fine-Grained Image Understanding](https://arxiv.org/abs/1901.06706)。代码/项目：https://github.com/necla-ml/SNLI-VE



---

<!-- source: directions/D2_factual_support/SERVER_EXPERIMENTS_CN.md -->

# D2 服务器实验方案｜视觉事实支持与生成忠实性

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS07, DS08, DS09, DS10, DS11, DS12, DS18；全部定义见 `datasets/DATASET_ATLAS_CN.md`。

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



---

<!-- source: directions/D3_conditional_generation/LITERATURE_REVIEW_CN.md -->

# D3 文献综述｜条件生成分布比较

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 这不是另一个CLIPScore

单张生成图与prompt是否相符、整组图是否逼真、合理结果是否覆盖充分，是三个不同问题。FID/CMMD主要比较边缘图像分布；TIFA/VQAScore/GenEval主要看prompt是否被满足；条件分布评价则希望区分“图像都很好但条件配错”与“条件正确但模式缺失”。

本轮特别核对到CFID[G02]和cFreD[G03]，后者已是WACV2026工作。**条件生成分布比较不是未被占据的空白。** 因而我们的潜在增量只能是有限每条件样本、高维/多模态非高斯关系或更可靠的分辨能力，不能简单声称首次把文本加到分布指标。

## 2. 核心工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [G01] **Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)** — CVPR 2024 | CLIP空间核MMD，不再使用FID的高斯假设。 | 多个图像生成模型/受控扰动；分布距离; 人评相关性; 样本效率。 |
| [G02] **Conditional Frechet Inception Distance** — 2021/2022 arXiv revision | 条件Wasserstein/FID的闭式高斯形式。 | 条件生成；CFID与FID/MSE对比。 |
| [G03] **Evaluating Text-to-Image and Text-to-Video Synthesis with a Conditional Fréchet Distance (cFreD)** — WACV 2026, author publication page verified | 联合视觉质量与条件文本的一致性，条件Fréchet距离。 | HPDv2; GenAI-Bench; PartiPrompts; COCO；人类相关性与模型排序。 |
| [G04] **GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment** — NeurIPS 2023 Datasets and Benchmarks | 基于对象检测的可组合T2I要求检查。 | GenEval prompts；单物体/多物体/数量/颜色/位置/颜色绑定及平均。 |
| [G05] **T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation** — NeurIPS 2023 | 属性绑定、对象关系及复杂组合的分项评价。 | 6000 prompts / 6 categories (original)；绑定/关系各官方评价器。 |
| [G06] **T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation** — TPAMI 2025 | 扩展组合类别和更细粒度诊断。 | 8000 prompts / 8 categories (++ release)；版本对应的官方子类分数。 |
| [G07] **ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty** — NeurIPS 2024 Datasets and Benchmarks | 按概念数量控制生成要求难度，自动问答并有人类验证。 | ConceptMix；各难度满足概念比例与全部满足率。 |
| [G08] **GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation** — 2024 arXiv / CVPR 2024 SynData workshop, not claimed main track | 复杂提示、图/视频生成的人类对齐评分；GenAI-Rank候选比较。 | GenAI-Bench / GenAI-Rank；人类相关性; 成对偏好; best-of-N。 |
| [G09] **Demystifying MMD GANs** — ICLR 2018 | KID的无偏MMD评价背景。 | 生成图像；KID; MMD。 |
| [G10] **Reliable Fidelity and Diversity Metrics for Generative Models** — ICML 2020 | Density/Coverage把保真和覆盖分开。 | 生成模型；Density; Coverage; precision/recall相关比较。 |
| [H10] **Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)** — ECCV 2024 | VQA模型对“是否显示该文字”的Yes概率作为配对评价。 | 多个图文/生成对齐benchmarks；人类偏好相关性/成对排序。 |

## 3. 三类基线各自负责什么？

CMMD[G01]使用CLIP表示的核MMD；它的“C”不是conditional。我们应给它同样的特征和每条件样本，同时增加强的条件/分组MMD控制，不能只跟无条件FID比来制造容易的条件错配优势。CFID/cFreD是直接比较对象；KID和Density/Coverage[G09/G10]分别提供采样与模式覆盖参照。

GenEval、T2I-CompBench与ConceptMix[G04–G07]不是“真实生成分布”，而是提示语义要求的评价器。它们提供外部任务检查，不提供自然S真值。GenAI-Bench[G08]和HPDv2的人评可以评价指标是否有用，但人类审美、语义忠实与多样性需要分项，不指望一个标量解释全部。

## 4. 最重要的数据难题：参考分布究竟是什么？

给每个自然prompt一张真实图，并不能定义该prompt下所有合理图像的总体。我们建议先用可控制的场景/渲染条件建立分布对象，再在有多实例的语义条件组上验证；自然细描述只有单例时，改称有限参考条件兼容性，不宣称准确估计未知总体。

同一prompt中的真实/生成图片如果有renderer、水印或来源风格差别，critic可能仅识别来源。跨生成器、风格控制和同边缘条件置乱可以检查这个问题。固定共同视觉特征的指标检验的是该特征观测中的分布，不等于原像素总体。

## 5. 可以争取什么优势？

希望VCS的学习型critic在有限样本下比固定核特征更善于捕获非线性条件差异，并通过共同读数说明误差；目标是识别受控条件破坏和模式缺失、可靠排序生成分布，而不是高J就好。拟合方式匹配Logistic后读取同J，若同样好则承认学习型函数类收益。

相比D1/D2，这条对estimator依赖最直接，完整“应用”可以是一套从样本到可靠评价的系统，不强制训练大扩散模型。但至少要有已知真值/可验证方向的受控结果与真实外部用途，不能只用一个对真实图像无oracle的J表自证。

## 6. 风险

小样本条件critic可能记住prompt或样本身份；最大化估计的有限经验差异产生偏高读数；强依赖又可能饱和。只要求conditional semantic discrepancy可检测，而没有比CFID/条件MMD更好的精度、成本或适用条件，则不够形成新评估方法。生成模型微调属于第二阶段，别让“又训练一个generator”掩盖指标可靠性问题。

## 本方向主来源

- [G01] [Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603)。代码/项目：https://github.com/google-research/google-research/tree/master/cmmd
- [G02] [Conditional Frechet Inception Distance](https://arxiv.org/abs/2103.11521)。代码/项目：https://github.com/Michael-Soloveitchik/CFID
- [G03] [Evaluating Text-to-Image and Text-to-Video Synthesis with a Conditional Fréchet Distance (cFreD)](https://arxiv.org/abs/2503.21721)。代码/项目：https://vislang.ai/publications/koo2026evaluating
- [G04] [GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)。代码/项目：https://github.com/djghosh13/geneval
- [G05] [T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350v2)。代码/项目：https://github.com/Karine-Huang/T2I-CompBench
- [G06] [T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation](https://doi.org/10.1109/TPAMI.2025.3531907)。代码/项目：https://github.com/Karine-Huang/T2I-CompBench
- [G07] [ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty](https://arxiv.org/abs/2408.14339)。代码/项目：https://princetonvisualai.github.io/conceptmix/
- [G08] [GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation](https://arxiv.org/abs/2406.13743)。代码/项目：https://huggingface.co/datasets/BaiqiL/GenAI-Bench
- [G09] [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)。本轮未确认可直接运行的官方实现。
- [G10] [Reliable Fidelity and Diversity Metrics for Generative Models](https://proceedings.mlr.press/v119/naeem20a.html)。本轮未确认可直接运行的官方实现。
- [H10] [Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)](https://arxiv.org/abs/2404.01291)。代码/项目：https://github.com/linzhiqiu/t2v_metrics



---

<!-- source: directions/D3_conditional_generation/SERVER_EXPERIMENTS_CN.md -->

# D3 服务器实验方案｜条件生成分布比较

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS13, DS12, DS14, DS15, DS16, DS17, DS18, DS19, DS20；全部定义见 `datasets/DATASET_ATLAS_CN.md`。

## A. 统计对象

固定条件权重π(c)，P(c,z)=π(c)P_ref(z|c)，Q_m(c,z)=π(c)P_model_m(z|c)。主读数S；CS变换按全局S单独提供，不能对每条件CS求均值后冒充同一值。共同观测编码器固定，不为VCS偷偷提供更强文本条件特征。

首次原型先在低维latent条件和已知观测映射上做精确truth，再渲染为CLEVR型场景。若渲染/特征变换不单射，先计算pushforward分布或放弃oracle标签；不要把scene label的S直接当CLIP特征真值。自然数据只报held-out common J与不确定性，不报未知S的MSE。

## B. 可控实验设计

条件c取对象/属性/关系组合，图像中的未指定位置、材质、背景视为合理变化。固定生成P_ref和四类Q：条件误配（保持总图像边缘不变）、类内模式缺失、视觉噪声/质量下降、混合污染。各类型强度3档，加clean reference，共13个规定分布。采用相同条件频率，生成/采样seed在角色间独立。

每条件FIT样本数{8,32,128}；条件组数先固定16，再以更多组测试可迁移性；3次独立抽样。主要6读数：VCS、matchedLogistic→J、same-target核relative fit、grouped RBF MMD、CFID/cFreD适用实现、无条件CMMD负控制。只有VCS、Logistic→J与同目标核回归进入posterior/S误差表；MMD、CFID/cFreD、CMMD只比较各自的分辨、稳定性与外部用途，绝不拿它们的原始值减去S。正确信息保持的null和恒定估计器同时列出，不能靠恒零的“低方差”获胜。

在多个Q_m比较时，每个critic都用相同FIT/SELECT/EVAL预算；不能用同一evaluation pool选网络/带宽。可增加共享条件critic，但“跨条件泛化”必须在未用于训练的condition/template上另测。

## C. 自然数据评价

第一部分选择已存在多真实实例的语义条件组，明确reference究竟是这批受限组，不是所有自然图像。COCO对象/属性group需要剔除歧义、共现偏差并记录；分组条件本身不赋予真实密度。不同生成器用固定版本/seed/prompt，先用已有可再分发样本，生成新图前估算预算与许可。

第二部分用GenAI-Bench/Rank固定release的人类判断，按prompt-group和generator-family做留出。评价模型级和prompt组级相关性；同一prompt中样本不足以拟合条件分布时聚合到预定组，不为得到显著性事后重分组。第三部分用GenEval/CompBench版本和ConceptMix检验语义缺陷排序，不把这些分数当S数值真值。

## D. 主表与指标

| 表 | 必报 |
|---|---|
| 受控同目标估计 | posterior MSE、S bias/RMSE、评价方差、null错误与样本量；fit/eval成本 |
| 缺陷分辨 | 各类型强度的排序/分辨概率；marginal不变但条件破坏的专门对照 |
| 多样性与忠实性 | GenEval/CompBench分项、Density/Coverage、固定人评相关性；不能只报整体好坏 |
| 自然分布评价 | 样本数曲线、跨generator/template一致性、独立bootstrap；不写oracle MSE |

输入尺度/特征/样本数必须匹配；核方法的带宽/维数在相同SELECT预算内选。无偏MMD负值照实保留，confidence不是自制分数。采用prompt群组重采样；生成次数是独立图像而不是一个样本的多裁剪。

## E. 完整应用与候选新增点

完整pipeline应能接收固定reference和新generator样本，输出标量、每条件诊断、有效样本量、fit/eval分离日志与不确定性。预注册一个外部用途：在受控质量/覆盖变化的生成器集合中正确识别退化，或根据独立评价选择生成配置；选择后再用新prompt评价，不能用同一人评挑模型再报相关性。

可选后续是条件生成适配，但minimizing a lower bound J会受critic不足影响；必须交替重拟合和独立外部评价，且不声称小J认证分布匹配。

## F. 失败后的处理

若CFID/groupedMMD已经稳定解决，研究应收束到VCS在哪些非高斯或高维条件有可复现增量，而不是撤强对照。若critic靠来源风格胜出，增加同风格/同生成器条件置乱对照，并固定与文本相关的观测，而非继续扩大网络。若自然reference不足，限定为有限参考评价或暂停自然主张；不能依靠“总会有真分布”继续跑。

首轮不是生成十万张图：先做13×3样本档的受控估计和一个固定人评release的资源核验。自然生成量由第一个完整timing决定，所有新生成/付费人评需按实际授权。



---

<!-- source: directions/D4_many_to_many/LITERATURE_REVIEW_CN.md -->

# D4 文献综述｜多对多、不完整标注与概率图文关系

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 任务本身是多义关系，而不只是错误配对

同一图可由不同文字正确描述，同一短句也可适用于多图。原始COCO给每图几条caption，不代表其他图文组合都错。CxC[M05]与ECCV Caption[M06]正是为检索漏标补充语义判断。它们表明传统Recall@1可能奖励寻找唯一标注出处，而不完整反映语义关系。

概率嵌入PCME[M01]、PCME++[M02]已经显式处理多义、不确定性及假负例；ProLIP[M03]把这条线扩到大规模预训练。因此，VCS的有界critic本身不构成“新概率模型”。需要把可训练分布表达、缺失标签假设和实际多正例决策一起定义。

## 2. 相关工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [F05] **Learning Transferable Visual Models From Natural Language Supervision** — ICML 2021 | 图文对比双编码器与零样本迁移。 | 大规模图文预训练; 多个分类和检索集；零样本准确率; 双向Recall@K。 |
| [F06] **Sigmoid Loss for Language Image Pre-Training** — ICCV 2023 | 逐对sigmoid替代批softmax；关注正负比例与配对预算。 | 图文预训练; 零样本分类与检索；准确率; 检索; batch/配对效率。 |
| [F07] **SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features** — 2025 technical report | 多目标和数据策略增强图文表示；作为现代预训练参考。 | 多语言图文; 定位及稠密特征任务；分类/检索/定位。 |
| [M01] **Probabilistic Embeddings for Cross-Modal Retrieval** — CVPR 2021 | 高斯概率嵌入与不确定性，多对多检索。 | COCO Caption; CUB Captions；Recall@K; PMRP/类相关检索。 |
| [M02] **Improved Probabilistic Image-Text Representations (PCME++)** — ICLR 2024 | 闭式概率距离与伪正例、混合训练缓解假负例。 | COCO Caption; CxC; ECCV Caption；多正例检索; 噪声/不确定性。 |
| [M03] **ProLIP: Probabilistic Language-Image Pre-Training** — ICLR 2025 | 大规模概率预训练与语义包含/不确定性。 | DataComp; 零样本分类; 检索; 不确定性任务；检索/分类/包含和不确定性。 |
| [M04] **LongProLIP: A Probabilistic Vision-Language Model with Long Context Text** — ICLR 2025 uncertainty workshop tiny paper / arXiv, not main | 概率模型拓展长文本。 | Urban1k; DataComp任务等；长描述检索; 分类保持。 |
| [M05] **Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO** — EACL 2021 | 补充图图/文文/图文的人类语义相似度。 | CxC on COCO；语义相关性; retrieval。 |
| [M06] **ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO** — ECCV 2022 | 修复COCO检索漏标正例，引入mAP@R。 | ECCV Caption / COCO Karpathy test subset；mAP@R; R-Precision; Recall@K。 |
| [M07] **FLAIR: VLM with Fine-Grained Language-Informed Image Representations** — CVPR 2025 | 细描述与文本条件视觉池化支持局部关系。 | 细粒度/长描述检索与区域任务；检索/局部对应。 |
| [M08] **SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations** — 2024 arXiv / verify venue before manuscript | 同义正描述与困难负描述，分离句式和语义。 | SugarCrepe++；官方双正例组合指标。 |

## 3. 三个最重要的比较

PCME++是自然训练对照，因为它不仅替换距离，还使用伪正例/混合策略。若VCS有额外伪标签、PCME++没有，应分开同标签预算与完整官方recipe两张表。

ProLIP提供强预训练不确定性参考，其额外训练数据、包含目标和概率表示不能免费算成“BCE对照”。LongProLIP是workshop tiny paper，作用是提醒长文本context本身会改变检索与零样本取舍，不能让VCS输入256tokens、基线截断64后归因于散度。

ECCV Caption是主要外部评价，不是训练负例矿。其查询子集和人工judge范围有限；官方mAP@R结果不等于全集所有pair的概率准确率。CxC的0–5语义相似度也不能除以5后就当groundtruth Bernoulli后验。[M05,M06]

## 4. 两条不同应用选择

**A. 有标注或类相关的多对多关系。** 在CUB class-relevant协议中可以明确多个相关目标，作为受控任务，但需称“同类检索”，不称每条caption真实支持每张图。自然数据若有充分judged positives/negatives，再评价语义概率。

**B. 不完整自然图文训练。** 未观察到的关系不是负类。可以维持joint-vs-product机制、明确它估计的是采样关系；也可以假设正未标记模型并估计标注倾向，但需要额外的可识别条件和PU先例，不能直接以“软标签”修好。第一轮不强制增加新的PU理论。

## 5. 可能优势和最大风险

期待在同信息量下保留更多正确匹配、对歧义查询更合理地表达不确定性，而不仅R@1高一点。真正需要同时比较mAP@R、R-Precision、官方R@K、已judge子集风险、检索集合大小与准确率。

若只有一个标量T，uncertainty应称“匹配证据不确定”，不能称数据模态的概率分布。若使用K个原型或Gaussian embeddings，需要同样给Logistic/contrastive基线，以免把架构收益归给VCS。CUB positive定义、COCO漏标、CxC语义连续标签三个问题也不能混成一个目标。

这条最容易接续现有双塔，但也最容易退回“新配方vs旧检索”。所以本包把它列为自然多正例标签足够时的备选，而不是默认重新训练CLIP。

## 本方向主来源

- [F05] [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)。代码/项目：https://github.com/openai/CLIP
- [F06] [Sigmoid Loss for Language Image Pre-Training](https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html)。代码/项目：https://github.com/google-research/big_vision
- [F07] [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)。代码/项目：https://huggingface.co/google
- [M01] [Probabilistic Embeddings for Cross-Modal Retrieval](https://arxiv.org/abs/2101.05068)。代码/项目：https://github.com/naver-ai/pcme
- [M02] [Improved Probabilistic Image-Text Representations (PCME++)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ad9d6ab10446114cf5482d5e1f971a84-Abstract-Conference.html)。代码/项目：https://github.com/naver-ai/pcmepp
- [M03] [ProLIP: Probabilistic Language-Image Pre-Training](https://arxiv.org/abs/2410.18857)。代码/项目：https://github.com/naver-ai/prolip
- [M04] [LongProLIP: A Probabilistic Vision-Language Model with Long Context Text](https://arxiv.org/abs/2503.08048)。本轮未确认可直接运行的官方实现。
- [M05] [Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO](https://aclanthology.org/2021.eacl-main.249/)。代码/项目：https://github.com/google-research-datasets/Crisscrossed-Captions
- [M06] [ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO](https://arxiv.org/abs/2204.03359)。代码/项目：https://github.com/naver-ai/eccv-caption
- [M07] [FLAIR: VLM with Fine-Grained Language-Informed Image Representations](https://arxiv.org/abs/2412.03561)。代码/项目：https://github.com/ExplainableML/flair
- [M08] [SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations](https://arxiv.org/abs/2406.11171)。本轮未确认可直接运行的官方实现。



---

<!-- source: directions/D4_many_to_many/SERVER_EXPERIMENTS_CN.md -->

# D4 服务器实验方案｜多对多、不完整标注与概率图文关系

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS12, DS21, DS22, DS23, DS24, DS25；全部定义见 `datasets/DATASET_ATLAS_CN.md`。

## A. 首先选关系，不先选损失

主候选是“多个合法语义对应下的检索与集合预测”。不再只在同一图的2–4个框中选择。以原图像为split单位；基础双塔先固定CLIP B/16或当前已存SigLIP2 B/16的一个，不能每格挑最有利checkpoint。

首轮分两张表：受控CUB类相关多正例；自然COCO/Flickr训练→ECCV Caption/CxC评价。CUB的真关系是class relevance；自然评价的真关系仅限官方judged universe。若自然训练没有已判断负例，必须列出这一缺口，不能使用eval标注来补。

## B. 模型和损失设计

使用共同多原型或Gaussian embedding宿主时，固定它的容量、采样次数和所有辅助正则。VCS定义为已标注/按协议构造的P/Q上的原J；Logistic在同P/Q同函数类拟合。InfoNCE或multi-positive softmax按同一合法正例集合训练。若选择机制Q=乘积，其T读取机制后验；需要语义支持概率时另做带人工标签的CAL映射。

未知pair不要赋硬负标签。第一轮可比较原始训练标注、受控删除一部分已知关系的标注变稀，及保留的已judge子集；所有方法看到相同删除mask。对COCO未知关系仅做敏感性分析，不声称有真实删标oracle。语义伪正例教师及阈值对所有方法共享；无教师和有教师分开。

## C. 最小完整训练

| 单元 | 内容 |
|---|---|
| D4-00 | 重现原双塔、PCME++的检索评估输入和ECCV evaluator fixture；清理COCO/Flickr/CUB角色 |
| D4-10 | 同宿主VCS、Logistic、multi-positive softmax、PCME++共4方法×3seeds，完整双塔适配 |
| D4-20 | 固定删标比例{0,.3,.6}在已知关系数据上检验；先共享训练budget，不额外每档多调参 |
| D4-30 | 未用于选择的ECCV Caption/CxC/跨Flickr与SUGARCREPE++语义读数；ProLIP作为完整系统参考 |

D4-10是12个训练单元提案；D4-20为条件扩展，不自动把12变36。训练都从相同原始权重开始；不能一边给VCS已有task-finetuned权重一边让基线从pretrain开始。

## D. 评价

主表两方向mAP@R与R-Precision，COCO5k/Flickr R@1/5/10同时保留。COCO1k五折另列，不与5k比较。CUB按PCME的class-heldout与类相关定义，不能和caption semantic truth混排。CxC相关性在公布judgements上算；支持概率/校准只在真正人工二值judge子集算。

新增集合预测用CAL选择阈值或coverage预算，比较相同平均返回条目数下的precision/recall。所有方法都可temperature/Platt校准；T有界不等于天然概率更好。按原图群组bootstrap，重复caption不计独立图像。

## E. 我们希望什么、什么不算成功

希望缺失关系下的多正例召回、概率风险或集合覆盖更好，且标准检索没有不合理损失。仅在原始单正例R@1上平局，不是新增能力；仅uncertainty与caption长短相关也不是正确的多义表达。

若所有收益来自prototype数、伪标签或更长context，则报告共有结构而非平方损失优势。若不完整标注导致无法评价语义真值，保留标准检索与annotation-sensitivity，暂停“准确估计真实匹配概率”的主张。需要新标注时明确人工成本/许可，不偷偷使用评价集训练。

下一步可转到D2的有真实支持标签核验，或D5的数据覆盖用途；不继续加内部余弦head直到某一列赢。



---

<!-- source: directions/D5_data_curation/LITERATURE_REVIEW_CN.md -->

# D5 文献综述｜图文数据选择与再描述

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 这条线为什么需要完整学生训练？

数据过滤的目标不是“过滤分数好看”，而是用它选出的数据训练出更好的模型。DFN[D02]明确表明，筛选网络自己的识别能力与它构造训练集的能力可以不同。DataComp[D01]因此固定候选池、模型、训练预算和多任务评价，避免用数据规模/训练时长变化掩盖策略作用。

Sieve[D03]利用captioner提供另一个语义信号；Recap-DataComp[D04]将原文与重写描述混合；BLIP[D07]更早已经用captioner/filter清洗训练。新的VCS系统不能仅把“检测错配”作为评价终点，更不能只挑高分简单图文对然后假定下游必好。

## 2. 相关工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [D01] **DataComp: In Search of the Next Generation of Multimodal Datasets** — NeurIPS 2023 Datasets and Benchmarks | 固定池、模型及计算的图文数据策展基准。 | DataComp four scales; 38 downstream tasks；38任务平均及ImageNet/检索。 |
| [D02] **Data Filtering Networks** — ICLR 2024 | 筛选网络的效用不等于其自身下游准确率。 | DFN pools; 多项零样本评价；由筛选集训练的模型质量。 |
| [D03] **Sieve: Multimodal Dataset Pruning Using Image Captioning Models** — CVPR 2024 | 生成描述与原文字的语义一致性筛选，结合CLIP。 | DataComp medium/large; 38 tasks；下游质量; 筛选与caption成本。 |
| [D04] **What If We Recaption Billions of Web Images with LLaMA-3?** — ICML 2025 | 大规模再描述，原文与新caption混合用于CLIP及生成训练。 | Recap-DataComp-1B; 多项图文检索和生成；检索; 生成忠实性。 |
| [D05] **Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders** — ICLR 2026 | 稀疏自编码器的目标分布匹配用于图文过滤(SDM)。 | DataComp medium; 38 tasks；目标与整体迁移性能。 |
| [D06] **Multimodal Distribution Matching for Vision-Language Dataset Distillation** — arXiv preprint, May 2026 | 多模态分布匹配用于合成数据蒸馏。 | 图文数据蒸馏；由合成数据训练的下游效果。 |
| [D07] **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** — ICML 2022 | captioner/filter组合清理和补充图文预训练。 | 图文预训练/检索/生成任务；检索与生成质量。 |

## 3. 本轮检索对创新定位的直接改变

ICLR2026的SDM[D05]已经研究目标分布对齐的过滤。因此我们提出“让选出数据接近目标分布”并非新颖性本身。2026预印本MDM[D06]还研究多模态分布匹配的数据蒸馏，但它生成合成数据，不等同于从固定池过滤；不应为了凑强baseline把两个预算体系混合。

可以研究的具体增量是：**目标关系分布未知且样本有限时，如何用学习型CS估计筛选/加权一个关系覆盖更合适的子集，并用下游模型验证这份估计确实有用？** 需要与同目标核MMD/相对回归、Logistic ratio、SDM可用实现和普通CLIP/Sieve过滤比较。

## 4. 数据与评估

首个正式基准推荐DataComp-CLIP Small的完整官方训练协议，而非自动启动medium。其独立图片数、保留比例、examples seen和模型固定；同样数量的小子集反复采样可能与更大unique set不同，二者都要记录。38任务平均和细分任务完整报告，不能只报一个目标域。[D01]

若当前磁盘/许可不允许DataComp，使用CC3M固定可访问子集进行内部验证，但明确它不是官方DataComp结果。Recap子集要保留相同图片的新旧caption，才能隔离“文本变化”与“选图变化”。Caption生成成本与下游训练成本都要计入。[D03,D04]

## 5. 理论风险

从VCS输出反解重要性比r=(1+T)/(1−T)可能极大，所以有界critic并不自动给出稳定数据权重。目标分布在来源池外的部分，也无法通过过滤创造出来。最小化一个拟合不足的critic的J会造成伪匹配，需要独立refit、替代统计量和学生验证。

D5因此不是一条“便宜的备用”。过滤计算便宜，但每个策略要训练学生才能判断；反而可能比D2或D6贵。建议只在固定数据池和学生预算能得到保障时启用。

## 本方向主来源

- [D01] [DataComp: In Search of the Next Generation of Multimodal Datasets](https://arxiv.org/abs/2304.14108)。代码/项目：https://www.datacomp.ai/dcclip/
- [D02] [Data Filtering Networks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html)。代码/项目：https://github.com/apple/ml-dfn
- [D03] [Sieve: Multimodal Dataset Pruning Using Image Captioning Models](https://openaccess.thecvf.com/content/CVPR2024/html/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.html)。代码/项目：https://github.com/facebookresearch/SIEVE
- [D04] [What If We Recaption Billions of Web Images with LLaMA-3?](https://proceedings.mlr.press/v267/li25ch.html)。代码/项目：https://github.com/UCSC-VLAA/Recap-DataComp-1B
- [D05] [Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8291acb610ddc260d554da5a3d273ae9-Abstract-Conference.html)。本轮未确认可直接运行的官方实现。
- [D06] [Multimodal Distribution Matching for Vision-Language Dataset Distillation](https://arxiv.org/abs/2605.23482)。本轮未确认可直接运行的官方实现。
- [D07] [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://proceedings.mlr.press/v162/li22n.html)。代码/项目：https://github.com/salesforce/BLIP



---

<!-- source: directions/D5_data_curation/SERVER_EXPERIMENTS_CN.md -->

# D5 服务器实验方案｜图文数据选择与再描述

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS26, DS27, DS28, DS12, DS21；全部定义见 `datasets/DATASET_ATLAS_CN.md`。

## A. 两个可分开的任务

**Selection主任务：**从固定pool选K个图文样本，按同一个标准CLIP/SigLIP训练recipe训练学生，比较下游效果。学生本身用标准任务损失，以免VCS筛选与VCS训练两个因素混在一起。

**Recaption条件扩展：**对同一组选中图片，在原始caption、已有recap、按规则选择/混合caption之间比较。先固定图像，再比较文字，不能同时改样本数量/长度/教师。

## B. 统计对象与原型算法

目标reference A是从FIT允许的目标关系样本中得到，不能由官方测试标签构造。来源pool B和选中子集B_w在共同joint representation z=(image,text,relation features)上比较S(P_A,P_Bw)。主critic使用VCS，matchedLogistic读同J，核相对回归/groupedMMD为对照。

可行的第一实现有两种但首轮只选一种：
1. **ratio reweighting**：拟合target-vs-source后验，转换ratio，按CAL决定clip并归一化采样；记录ESS/maxweight与source coverage。
2. **固定字典子集选择**：在预定语义/质量簇上求权重，反复用独立样本估计分布差异；终点仍是学生质量。

二者都是已有分布匹配/重加权思路的CS实例，不自动是新算法。若要形成贡献，需要有限target样本下更可靠的覆盖/成本表现。高T筛选“最好配对”另作质量baseline，不与target-vs-source ratio混同。

## C. 数据与训练

固定共同可访问pool及安全/许可过滤，记录URL失败、重复图、哈希和下载时间。所有策略面对相同pool。保留比例首轮固定一个实用K；第二比例仅在主机制成立后追加。DataComp正式行采用官方small architecture/exposure，内部scale另名。

同样的unique images、examples seen、augmentation、batch和优化steps；若算法重复采样更多，要记录实际重复分布和有效数据量。目标reference不得用下游test label挑选。目标组外性能也要看，避免过拟合reference。

## D. 主baseline和运行单位

原池random、CLIP相似度过滤、Sieve、可运行的DFN/SDM、核目标匹配、Logistic ratio、VCS selection。先用公开/缓存分数尽量复用生成成本，但所有额外教师标注单列。

资源充足前的最小正式阶段为5策略：random、CLIP、一个已验证强过滤器、matchedLogistic、VCS，各3个学生seed=15次完整学生训练。核和SDM可先在同一子集selection层完成；投稿主张优于这些方法时必须加入其学生训练，而不是只有过滤分数。

Recaption采用2×2固定设计：图像选择baseline/VCS × caption原始/相同recap教师，学生recipe相同；属于追加资源项目，不能在第一轮15次之外自动启动。

## E. 指标与预期

主终点DataComp官方38任务summary；另列ImageNet、retrieval、target-type任务和非目标域。筛选阶段报告分布读数/coverage/ESS，不作为学生质量的替代。总成本包括feature extraction、teacher caption、filter fit、selection及student训练，分别报告一次成本和摊销假设。

H5：相同保留数和训练exposure，VCS选择可能在目标relation稀缺时更有效覆盖，并保持总体迁移。不是理论保证，也不预设抗噪/尾部优势。识别原因需要匹配Logistic、核和强过滤器。

## F. 失败处理

若权重饱和/ESS塌陷，先查support与clip，而不是把clip后的分数称无偏。若只改善target小集合、整体下降，报告取舍并考虑混合uniform quota；quota为新策略，要给对照相同机会。若teacher成本高于学生节省，改用已有recap或冻结策略，不宣称高效。若没有足够数据/训练预算，D5保持综述与方案状态，不拿10k图片toy替代完整DataComp证据。



---

<!-- source: directions/D6_temporal_support/LITERATURE_REVIEW_CN.md -->

# D6 文献综述｜叙述视频的时间支持与定位

检索截止2026-10-11。事实来源按[Pxx]编号可在文末直接打开；“本方案”均为新设计，不是原论文结果。核验深度见总注册表。

## 1. 从弱叙述对齐到查询条件时间支持

视频中的一句话可能早于动作、晚于动作、概括几个步骤，或者与当前画面无关。MIL-NCE[V01]早已用多实例正袋处理叙述错位；因此“允许时间不对齐”不是一个新点。

QVHighlights[V02]将自然语言查询、多个相关时间段和片段显著性统一起来；QD-DETR[V03]、CG-DETR[V05]强调查询对视频表示的选择性；EaTR[V04]利用事件结构，BM-DETR[V06]处理背景。近年的Sim-DETR[V07]、SG-DETR[V08]继续改进完整定位与显著性系统。这里的机会不是把RefCOCO的框换成视频帧继续排Top-1，而是**学习随时间变化、能支持多段输出和不同持续时长的语言证据**。

## 2. 相关工作

| 工作与状态 | 原文方法 | 原文数据/评价 |
|---|---|---|
| [V01] **End-to-End Learning of Visual Representations from Uncurated Instructional Videos (MIL-NCE)** — CVPR 2020 | 多实例NCE处理叙述视频固有时间错配。 | HowTo100M train; YouCook2/MSR-VTT/CrossTask/COIN等；检索; 动作分类/定位/分割。 |
| [V02] **QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries** — NeurIPS 2021 | 查询条件多时间段定位与显著性，Moment-DETR端到端头。 | QVHighlights；MR mAP; R@1; HD mAP/Hit@1。 |
| [V03] **Query-Dependent Video Representation for Moment Retrieval and Highlight Detection (QD-DETR)** — CVPR 2023 | 查询依赖表示与负查询训练。 | QVHighlights; TVSum; Charades-STA；moment retrieval/highlight官方指标。 |
| [V04] **Knowing Where to Focus: Event-aware Transformer for Video Grounding (EaTR)** — ICCV 2023 | 事件结构引导视频定位。 | 视频语言时间定位集；R@1 tIoU; mAP等官方任务指标。 |
| [V05] **Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding (CG-DETR)** — 2023/2024 arXiv revision; venue not re-verified | dummy tokens与跨模态相关性调节查询参与程度。 | QVHighlights; temporal grounding benchmarks；MR/HD指标。 |
| [V06] **Background-Aware Moment Detection for Video Moment Retrieval (BM-DETR)** — WACV 2025 | 显式处理相关片段与背景。 | QVHighlights等时间定位集；moment retrieval指标。 |
| [V07] **Sim-DETR: Unlock DETR for Temporal Sentence Grounding** — ICCV 2025 | 处理查询竞争与全局/局部定位。 | temporal sentence grounding benchmarks；官方定位指标。 |
| [V08] **Saliency-Guided DETR for Moment Retrieval and Highlight Detection (SG-DETR)** — WACV 2026 | 显著性引导MR/HD，使用更强视频预训练信息。 | QVHighlights及时间定位；MR/HD。 |
| [V09] **Hierarchical Video-Moment Retrieval and Step-Captioning (HiREST)** — CVPR 2023 | 长视频中的层级片段和步骤描述。 | HiREST；视频/片段retrieval; step captioning。 |
| [V10] **MomentSeeker: A Benchmark for Long-Video Moment Retrieval** — NeurIPS 2025 Datasets and Benchmarks | 更长视频和多种查询形态。 | MomentSeeker；长视频定位/检索。 |
| [V11] **NaQ: Leveraging Narrations as Queries to Supervise Episodic Memory** — CVPR 2023 | 将叙述转为Ego4D NLQ训练监督。 | Ego4D NLQ；R@1/5 at tIoU 0.3/0.5。 |

## 3. 哪个benchmark最合适？

首选QVHighlights，因为它同一视频/查询下有多相关moments与saliency，可以同时评价“是否支持”和“在哪里支持”。官方提供预提取特征与完整Moment-DETR训练/eval，适合在可控资源内比较。缓存特征训练完整定位头是该领域标准协议，但不是更新全部视频encoder，更不是实时原视频加速。[V02]

Charades-STA作为第二数据集，检验场景与表达变化；固定train/test及feature来源。ActivityNet Captions可扩展更长活动；HiREST有层级步骤；Ego4D NLQ的第一人称长视频和MomentSeeker长查询适合最后验证，不宜第一批全部下载。[V04,V09–V11]

## 4. 我们的方法可能做什么？

选择被标注为支持/不支持查询的片段分布P/Q，学习T并经过明确先验校准生成时间支持场。由该支持场指导片段预算、融合和区间头，同时保留标准span边界/匹配损失。原始S/J可以分析证据拟合，实际终点则是MR/HD和置信度，而不是只最大化时间文本依赖。

最接近的控制是同宿主Logistic saliency和标准QD-DETR/CG-DETR，而不是只比随机负例InfoNCE。若增加相邻时间的平滑或持续时间先验，每条方法得到同样后处理机会；这些结构也是经典时间建模，不归给CS散度。

## 5. 预期价值与限制

希望在多段支持、长背景以及规定时间抖动条件下，得到更准确的support风险和完整定位；或用较少片段取得相似定位质量。先检验噪声模型是否对应真实标注，不将人工时间抖动的成功推广为自然叙述鲁棒性。

QVHighlights官方当前已写明Codalab关闭、test真值发布；这只是评价方式变化，不允许训练用test。MR mAP和HD mAP不是同一个指标，不能用更容易的saliency峰值替代边界定位。

主要风险是未标注片段被误认为负、长视频训练靠位置先验、特征预训练/提取器不匹配，以及又退回只看最高帧分数。需要完整任务头和独立的时间支持校准来避免这些问题。

## 本方向主来源

- [V01] [End-to-End Learning of Visual Representations from Uncurated Instructional Videos (MIL-NCE)](https://arxiv.org/abs/1912.06430)。代码/项目：https://www.di.ens.fr/willow/research/mil-nce/
- [V02] [QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries](https://arxiv.org/abs/2107.09609)。代码/项目：https://github.com/jayleicn/moment_detr
- [V03] [Query-Dependent Video Representation for Moment Retrieval and Highlight Detection (QD-DETR)](https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Query-Dependent_Video_Representation_for_Moment_Retrieval_and_Highlight_Detection_CVPR_2023_paper.html)。代码/项目：https://github.com/wjun0830/QD-DETR
- [V04] [Knowing Where to Focus: Event-aware Transformer for Video Grounding (EaTR)](https://openaccess.thecvf.com/content/ICCV2023/html/Jang_Knowing_Where_to_Focus_Event-aware_Transformer_for_Video_Grounding_ICCV_2023_paper.html)。代码/项目：https://github.com/jinhyunj/EaTR
- [V05] [Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding (CG-DETR)](https://arxiv.org/abs/2311.08835)。代码/项目：https://github.com/wjun0830/CGDETR
- [V06] [Background-Aware Moment Detection for Video Moment Retrieval (BM-DETR)](https://openaccess.thecvf.com/content/WACV2025/html/Jung_Background-Aware_Moment_Detection_for_Video_Moment_Retrieval_WACV_2025_paper.html)。代码/项目：https://github.com/minjoong507/BM-DETR
- [V07] [Sim-DETR: Unlock DETR for Temporal Sentence Grounding](https://openaccess.thecvf.com/content/ICCV2025/html/Tang_Sim-DETR_Unlock_DETR_for_Temporal_Sentence_Grounding_ICCV_2025_paper.html)。本轮未确认可直接运行的官方实现。
- [V08] [Saliency-Guided DETR for Moment Retrieval and Highlight Detection (SG-DETR)](https://openaccess.thecvf.com/content/WACV2026/html/Gordeev_Saliency-Guided_DETR_for_Moment_Retrieval_and_Highlight_Detection_WACV_2026_paper.html)。本轮未确认可直接运行的官方实现。
- [V09] [Hierarchical Video-Moment Retrieval and Step-Captioning (HiREST)](https://openaccess.thecvf.com/content/CVPR2023/html/Zala_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_CVPR_2023_paper.html)。本轮未确认可直接运行的官方实现。
- [V10] [MomentSeeker: A Benchmark for Long-Video Moment Retrieval](https://proceedings.neurips.cc/paper_files/paper/2025/hash/281e0b9142763f2b6c944fedb8550ba9-Abstract-Datasets_and_Benchmarks_Track.html)。本轮未确认可直接运行的官方实现。
- [V11] [NaQ: Leveraging Narrations as Queries to Supervise Episodic Memory](https://arxiv.org/abs/2301.00746)。代码/项目：https://vision.cs.utexas.edu/projects/naq/



---

<!-- source: directions/D6_temporal_support/SERVER_EXPERIMENTS_CN.md -->

# D6 服务器实验方案｜叙述视频的时间支持与定位

**性质：候选方案，尚无本方案自然数据结果。** 先读本目录综述和共同理论/评价规范。数据条目：DS29, DS30, DS31, DS32, DS33, DS34, DS35；全部定义见 `datasets/DATASET_ATLAS_CN.md`。

## A. 完整任务与宿主

采用Moment-DETR或已能严格复现的QD-DETR作为同架构host，输入官方同版SlowFast/CLIP等features与text tokens，输出若干[start,end,score]及clip saliency。第一轮训练完整融合/定位模型，不只是冻结features上的一个小判别probe。若研究终点是计算节省，最后另做raw-video end-to-end计时与适配，不能拿缓存特征耗时替代。

## B. 两种监督情形分开

**强监督主线：**QVHighlights query的发布相关区间定义support，在其标注协议下建立P(+支持clip|query/video)与Q(已定义背景clip|query/video)。发布标签只代表这个annotation task，不保证所有背景在真实世界绝无关联；边界邻域可标uncertain并在所有方法中统一处理。

**弱监督扩展：**HowTo100M或ASR只能提供正袋，不能将每个时间点当确定标签。必须使用MIL-NCE类袋目标或明示潜在变量模型；不在第一轮强监督recipe里混入弱袋再叫同P/Q。

VCS/Logistic support head同函数类，T在平衡实验上拟合，部署时映射到实际片段support先验再校准。主任务span L1/GIoU/Hungarian或host原损失保留；只有support学习组件替换。新增loss权重在相同CAL预算内选择，不让ours特调而host只用旧默认。

## C. 首轮实验清单

| 单元 | 要做什么 |
|---|---|
| D6-00 | 官方Moment-DETR/QD-DETR training和evaluation fixture；视频切片parent ID、帧率/特征步长/秒单位检查 |
| D6-10 | 同一host标准support loss、matchedLogistic、VCS、VCS+预定义时间结构控制，共4方法×3seeds完整训练 |
| D6-20 | 公共CAL概率校准/时间支持评价，外加MR/HD；时间结构也给Logistic同机会以判断归因 |
| D6-30 | Charades-STA第二数据集，同主要目标3seeds；不自动复制所有ablation |
| D6-40 | 明确弱监督时间噪声下的MIL-NCE对照，或长视频域验证二选一 |

D6-10第四臂只有预算允许且host存在时间结构时运行；若新增结构，必须同时跑JS+同结构，不保持不公平的四臂计数。基础3×3=9为最小，结构成对新增6。自然benchmark结果必须完整长度/多区间，不能只评oracle候选片段。

## D. Benchmark与metrics

QVHighlights：MR mAP@[.5:.05:.95]及R@1@.5/.7；HD mAP/Hit@1按官方saliency阈值；短/中/长事件分层。输入输出使用官方JSONL(qid,vid,pred_relevant_windows,pred_saliency_scores)，先跑sample fixture数值校验。test已公开但保持封存。

Charades-STA：R@1/5 at .3/.5/.7和mIoU，报告原feature类与是否额外预训练。Ego4D仅扩展时使用NLQ R@1/5@.3/.5，不混MQ。自定义support指标包括Brier、frame/clip PR、区间覆盖/长度和calibrated set coverage，均单列custom，不能替代官方MR/HD。

干预分析：固定视频和query，在FIT-only构造时间偏移或背景扩展，DEV用于预定stress。视频反转未必语义不变；无依据不能把它当正对增强。参考负片段的时间距离分布、持续长度和sampler在三方法间相同。

## E. 预期与转向

希望共享平方读数帮助更准确地表达多段支持并改善完整定位/片段选择；无需所有R@1都第一，但必须有实际的risk–localization或cost–quality收益。若改善仅在自制抖动数据，不能宣称自然噪声鲁棒。

若所有收益来自额外时间平滑/更强feature，重新归因；若VCS support校准改善但MR不变，检验它是否服务多区间/选择性定位，不凭内部J自动命名新能力。若问题仍只是softmax竞争，停止在同一视频上继续换参考常数，考虑D2支持核验的完整用途。



---

<!-- source: benchmarks/LITERATURE_PROTOCOL_CARDS_CN.md -->

# 最近相关工作的实验协议卡

本表只写本轮原始来源明确支持的内容，供服务器建立复现清单。没有核实的训练数量、版本和超参数不补猜。更完整的论文摘要、数据与脚本链接见`literature/papers.json`、各方向综述和数据图谱。原论文结果不能直接当成我们在新宿主上的baseline结果。

## D1：剪枝

**T02 SparseVLM（ICML2025）。**研究文本相关token驱动的视觉稀疏化；主结果是完整VLM问答/理解与效率，而不是每token标签预测。移植时要对齐所用宿主、剪枝发生位置、token保留层次与推理backend。数据/评价列表以其官方实现和补充为完整清单；本包选择GQA/TextVQA/ChartQA/DocVQA是新方案，不表示每篇baseline原文都跑了相同四项。

**T05 IF-Prune（CVPR2026）。**本轮核对作者页面、CVF标题摘要及官方仓库；仓库提供Qwen2-VL和InternVL训练/测试入口、ChartQA示例和lmms-eval集成。不同宿主使用不同transformers版本。全文不可访问的实验细节须服务器补读；不能因为repo里有一个脚本就断言某个全部benchmark已在原文验证。其VIB与小模型指导是D1必须逐项区分的先行结构。

**T06 DiffPrune（2026预印本）。**原文列LLaVA-1.5、LLaVA-NEXT与Qwen2.5-VL，并在GQA、MMBench/中文、MME、POPE、ScienceQA-IMG、VQAv2、TextVQA、SEED、VizWiz等任务评价。其训练弱化信息、部署hardtopK；所以soft训练到hard部署与额外模块成本都需单列。不能将其论文中的LLM-prefill倍数改称端到端倍数。

**T07 ET-Prune（2026预印本）。**研究文本密集视觉中的动态证据预算。它与“我们第一次按证据确定budget”直接重合；具体训练/测试集以全文实验与released-code revision为准，不能把RelatedWork里列举的DocVQA自动算成已完成主实验。它应先完成方法级比较，再决定D1何处有新增空间。

## D2：事实与幻觉

**H01 VCD。**原文采用LLaVA-1.5、InstructBLIP、Qwen-VL；POPE表包含COCO、A-OKVQA、GQA来源及random/popular/adversarial设置；另评价MME与LLaVA-Bench。无需训练但增加原图/扰动图两路计算。新宿主比较要同步temperature、decode策略和最大输出长度，不能拿其原文分数填我方Qwen2.5-VL表。

**H02 M-HalDetect。**人类细粒度标注用于检测/奖励与FDPO；原文best-of-n测试包含n=16、64，并研究InstructBLIP奖励对LLaVA/mPLUG-OWL的泛化。我们的首轮n=1/4是节约资源的新协议，不能说复现原64采样结果；每个baseline共享候选回答才能隔离重排序器。原文对analysis类的处理本身有消融，服务器须保留标签语义。

**H06/H07 HallusionBench、AMBER。**前者保留question/figure/pair家族聚合，后者分生成与判别两个任务。它们不是训练错误标签的默认来源；AMBER生成coverage与幻觉同报，判别P/R/F1的正类采用原定义。原图、编辑图及配对问题必须同组留出。

## D3：生成分布

**G01 CMMD。**原文方法是固定CLIP表示上的MMD；它既不是conditionalMMD，也不是以每张图语义真假训练的分类器。比较时所有分布统计使用相同样本数/featurepreprocess，单独报告自己学习critic所需的FIT样本。

**G03 cFreD。**正文及附录使用PartiPrompts、HPDv2与COCO提示，对多种生成器和图文特征组合评价，并检查人类判断关联。D3不能只对比FID和CLIPScore而省略它。原文的feature选择研究不表示我们可以在最终人评集选择最有利的feature；我方选择须留出。

**G04-G08 语义/偏好基准。**GenEval、CompBench原版与++、ConceptMix解决不同组合要求；BaiqiL的GenAI-Bench/Rank有发布版本差异，不能混用同名GenAI-Arena。它们评价的是语义或偏好，不提供自然条件分布的S真值。条件频率、每prompt生成数和judge版本在比较前固定。

## D4：多对多关系

**M01/M02/M03 PCME、PCME++、ProLIP。**它们已经研究概率嵌入与不确定关系。任务控制不仅换loss，还要给Logistic/VCS一致的概率头、采样次数和标签图。将官方大规模预训练checkpoint作为外部参考，与我们小样本完整微调的控制分表，避免归因错误。

**M06 ECCV Caption。**官方工具统一评价COCO1k、5k、CxC、ECCVRecall、R-Precision和mAP@R。额外标注覆盖Karpathy测试池的1,332个imagequery和1,261个captionquery子集，而不是每个query都有全部潜在真对应。mAP@R必须调用其发布的query列表。官方README示例API在数据图谱中给出。

**M07 FLAIR。**文本条件化池化是方法的一部分。只读取一个全局vector再算cosine属于削弱版，不代表论文的局部/细粒度能力；DCI/DOCCI扩展若采用它，先锁其原生score路径。

## D5：数据选择

**D01 DataComp。**原文固定四档pool与训练预算，通过从所选数据训练CLIP，再评价38个分类/检索任务判断数据质量。小档pool是12.8M，表中的samples-seen与选择后unique样本数是不同列。我们缩小成有限本地池时必须称customscale，不能套small标签。

**D02/D03/D04 DFN、Sieve、RecapDataComp。**分别引入过滤网络、caption模型或重新生成描述。filter/captioner的监督、训练来源与推理开销是系统的一部分。仅比较CLIPcosine而跳过这些已有信号，会让VCS的额外model调用看起来像损失优势。

**D05 SDM（ICLR2026）。**官方出版页明确目标对齐数据过滤及稀疏特征匹配；它是“选择与目标分布更接近的数据”的直接先行。服务器拟走D5时必须补读全文和实现，并给出同目标可执行对照；摘要核对不等于已经复现。

## D6：时间支持

**V02 Moment-DETR / QVHighlights。**官方仓库提供约8GB的CLIP+SlowFast缓存特征、200epochs训练入口与独立evaluator。这个时间只属于原作者的硬件记录，不作为我们的预算。注释是CCBY-NC-SA4.0，代码MIT；原始视频另有来源约束。先运行golden predictionfixture。

**V03/V05/V06 QD-DETR、CG-DETR、BM-DETR。**任务中已经存在query-dependent、相关性校准和background建模。新增VCS支持模块必须在共同feature与模型上与这些机制比较；不以额外ASR/更强feature来解释loss贡献。

**V07/V08 Sim-DETR、SG-DETR。**作为较新的强定位候选，服务器需补读原实验配置和数据预训练清单，至少选择一项与拟定任务真正可比的baseline。不可将只读摘要的SOTA数字直接写入共同协议表。

**官方服务状态。**QVHighlights standalone_eval README已说明Codalab停止、test标注发布；文档下方旧提交说明仍在，但不应建议使用已经停止的服务。公开test继续只作冻结后的最后评价，不能用于阈值/epoch选择。

## 接入完成的定义

每项最近方法只有在`baseline_reproduction.csv`具备代码revision、数据版本、宿主、preprocessing、evaluator和实际输出后，才标为“本地复现完成”。论文页/项目页存在只代表可查来源，不代表现有server已装好依赖或权重。



---

<!-- source: benchmarks/METRIC_SPECIFICATIONS_CN.md -->

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



---

<!-- source: benchmarks/BASELINE_MATRIX_CN.md -->

# 必做对照与公开系统

每条实物方法/预算见同目录`baseline_matrix.csv`。下表是运行类别，不表示本包已复现。

|方向|方法|类别|实现模式|对照要求|
|---|---|---|---|---|
|D1|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D1|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D1|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D2|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D2|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D2|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D3|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D3|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D3|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D4|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D4|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D4|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D5|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D5|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D5|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D6|VCS,原J|core|matched_training|不是首次二次泛函；有限函数类/优化/数据角色明确|
|D6|Matched Logistic → same J|core|matched_training|相同输入/结构/监督/选择预算；Logistic也能读取原J|
|D6|Same-target kernel/relative regression|estimator_comparison|matched_readout|带宽、核维数与独立样本量/配对计算量分开|
|D1|Full unpruned host|must_or_direct_prior|inference|完整计算与准确率基线|
|D1|Random / uniform token selection|must_or_direct_prior|inference|相同token数；相同生成参数|
|D1|FastV|must_or_direct_prior|official_inference|位置/attention backend及速度范围一致|
|D1|SparseVLM|must_or_direct_prior|official_inference|文本条件与剪枝层pin|
|D1|FEATHER|must_or_direct_prior|official_inference|报告视觉依赖任务，保留原生策略|
|D1|IF-Prune|must_or_direct_prior|official_training_inference|小模型/teacher额外开销及专有训练数据单列|
|D1|DiffPrune / ET-Prune|must_or_direct_prior|implementation_audit|2026预印本，代码未核实时不得假装复现|
|D1|Direct realized-risk selector|must_or_direct_prior|matched_training|同teacher输出同训练信息；强机制控制|
|D2|Base VLM|core_or_task_anchor|inference|相同prompt/maxnewtokens|
|D2|VCD|core_or_task_anchor|official_inference|增加forward的成本不能忽略|
|D2|M-HalDetect detector/reward|core_or_task_anchor|official_training|同人类标注预算|
|D2|Woodpecker|core_or_task_anchor|official_pipeline|检测/OCR/语言模型调用全部记录|
|D2|SCI|core_or_task_anchor|audit_optional|动态counterfactual题集与静态任务分表|
|D2|VQAScore|core_or_task_anchor|official_inference|用于图文支持/生成图时，不能冒充所有原子事实监督|
|D2|Direct error classifier; text-only classifier|core_or_task_anchor|matched_training|同架构extra监督分开，检查文本捷径|
|D3|Conditional/grouped RBF MMD|core|official_or_declared_adaptation|条件π和feature共同；与S数值不同|
|D3|CMMD|core|official_or_declared_adaptation|无条件CLIP-MMD；条件置乱时是负控制而非弱化主baseline|
|D3|CFID / cFreD|core|official_or_declared_adaptation|最直接条件分布方法；同reference/features|
|D3|FID / KID|core|official_or_declared_adaptation|样本数与编码器预处理一致|
|D3|Density / Coverage|core|official_or_declared_adaptation|多样性与忠实性分离|
|D3|TIFA / VQAScore|core|official_or_declared_adaptation|单图语义工具，不能顶替分布比较|
|D4|Raw CLIP / SigLIP2|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|未适配能力|
|D4|Multi-positive InfoNCE / sigmoid|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|与VCS同标签图、同负例信息|
|D4|PCME++|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|同probabilistichead与train监督|
|D4|ProLIP|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|公开checkpoint锚点与同架构loss控制分表|
|D4|FLAIR|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|保留text-conditionedpooling|
|D4|LongProLIP|core_or_task_anchor|complete_dual_tower_or_official_checkpoint|可选workshop长文本方法，不混主会|
|D5|Random subset|core|selector_plus_full_student|全部策略使用同一pool与exposure|
|D5|CLIP score filtering|core|selector_plus_full_student|图片质量与匹配混合效应要记录|
|D5|DFN|core|selector_plus_full_student|独立filter训练数据与权重|
|D5|Sieve|core|selector_plus_full_student|captioner成本与许可|
|D5|SDM|core|selector_plus_full_student|目标数据对齐最直接先行，不能跳过|
|D5|Recaptioning control|core|selector_plus_full_student|相同captioner相同图片；区分选样与换caption|
|D6|Moment-DETR|core_or_optional|full_temporal_model|同CLIP/SlowFastfeature与官方loss|
|D6|QD-DETR|core_or_optional|full_temporal_model|query结构与监督完整|
|D6|CG-DETR|core_or_optional|full_temporal_model|时间相关/无关校准能力|
|D6|BM-DETR|core_or_optional|full_temporal_model|background-aware直接比较|
|D6|EaTR / Sim-DETR / SG-DETR|core_or_optional|full_temporal_model|至少选择一项新强法，SG额外数据/模型分栏|
|D6|MIL-NCE / NaQ|core_or_optional|full_temporal_model|只在弱叙述/ASR监督路线是直接训练对照|



---

<!-- source: datasets/DATASET_ATLAS_CN.md -->

# 数据集与评价接入图谱

本表列出35个数据/协议条目，不等于35套独立图像来源。很多COCO、Flickr派生集共享原图。入口可查不等于数据已下载或所有许可证已接受。

执行前必须提交原始ID去重、许可、内容hash、可访问样本数、官方评估器commit和预训练暴露记录。涉及账号、付费裁判、受限下载，按现有授权处理，不代用户接受条款。

## DS01 — GQA (balanced / original release)
**用途：** D1;D2；train+CAL+official eval。
**独立单位与划分：** original image / scene graph。官方balanced train内按image分FIT/CAL；test-dev按预注册一次读；不得将同图问题跨角色。
**指标：** official answer accuracy; consistency/validity/plausibility等另列。
**获取：** https://cs.stanford.edu/people/dorarad/gqa/index.html
**评价入口：** https://cs.stanford.edu/people/dorarad/gqa/evaluate.html
**风险／权限：** VG预训练region features可能见过部分GQA val；按官方test-dev建议核对；yes/no子集为自定义辅助任务不冒充全GQA。
**来源：** B01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS02 — TextVQA
**用途：** D1;D2；train/CAL and held-out eval。
**独立单位与划分：** original image。官方train内图像分组；val/test沿官方；图片与TextCaps等共享源排重。
**指标：** 官方10人答案VQA soft accuracy。
**获取：** https://textvqa.org/
**评价入口：** https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py
**风险／权限：** 区分image-only与提供OCR文字/框；图像许可与标注许可分别核验；答案处理器pin commit。
**来源：** B02;T02;T03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS03 — SP-DocVQA (single-page, 2020/2021)
**用途：** D1；train/CAL and eval。
**独立单位与划分：** source document; page family。官方train/val/test；内部按文档/模板源group，不只问答ID。
**指标：** ANLS; exact match辅助。
**获取：** https://www.docvqa.org/datasets/docvqa
**评价入口：** https://rrc.cvc.uab.es/
**风险／权限：** 下载/提交可能要求注册；本包不接受条款/不上传结果；不能混为DocVQA2026。
**来源：** B03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS04 — ChartQA
**用途：** D1；train/CAL/eval。
**独立单位与划分：** chart image; source chart/template。官方human/augmented分别记录；同图两类问题同角色。
**指标：** numeric relaxed accuracy within5%; nonnumeric exact; human/augmented分开及官方平均。
**获取：** https://github.com/vis-nlp/ChartQA
**评价入口：** https://github.com/vis-nlp/ChartQA
**风险／权限：** gold table/OCR作为额外输入单列；数值零、百分号、列表/单位以锁定官方代码为准。
**来源：** B04;T05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS05 — MMBench (pin language/version)
**用途：** D1；evaluation only in first packet。
**独立单位与划分：** image/question family。固定en或cn及v1.1等版本，不混多次排行榜。
**指标：** 官方MCQ/circular evaluation版本。
**获取：** https://github.com/open-compass/MMBench
**评价入口：** https://github.com/open-compass/VLMEvalKit
**风险／权限：** 答案解析可能需judge/API；先固定解析器；不能仅换prompt获取最好分。
**来源：** T02;T07（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS06 — ScienceQA image subset
**用途：** D1；evaluation secondary。
**独立单位与划分：** question/image。明确IMG子集；不以全ScienceQA分母报告IMG成绩。
**指标：** multiple-choice accuracy。
**获取：** https://scienceqa.github.io/
**评价入口：** https://github.com/lupantech/ScienceQA
**风险／权限：** 有无解释/CoT/文本背景相同，图像缺省样本不能混入视觉压缩效果。
**来源：** T02;T06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS07 — M-HalDetect
**用途：** D2；verifier/reward train and grouped CAL; holdout transfer。
**独立单位与划分：** original image + response generator。沿官方划分；生成模型留出另做；同图回答/片段不跨角色。
**指标：** 官方span/response判别; supported-vs-contradicted Brier/AUPRC自定义补充。
**获取：** https://github.com/hendryx-scale/mhal-detect
**评价入口：** https://github.com/hendryx-scale/mhal-detect
**风险／权限：** 标签可能含分析/不可验证；先审schema，不把所有非支持都当错误；冻结judge，图片COCO重叠审查。
**来源：** H02（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS08 — SNLI-VE
**用途：** D2；optional supervised entailment train/eval。
**独立单位与划分：** Flickr original image。图像级官方train/val/test；Flickr检索同源排重。
**指标：** 3-way accuracy; macro-F1/Brier supplementary。
**获取：** https://github.com/necla-ml/SNLI-VE
**评价入口：** https://github.com/necla-ml/SNLI-VE
**风险／权限：** entailment/neutral/contradiction分开；纯文本可解偏差控制；neutral不是明示false。
**来源：** H11（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS09 — POPE
**用途：** D2;D1；evaluation only。
**独立单位与划分：** original COCO image。random/popular/adversarial三配置分别锁定；不用于训练verifier。
**指标：** accuracy; precision; recall; F1; yes ratio。
**获取：** https://github.com/AoiDragon/POPE
**评价入口：** https://github.com/AoiDragon/POPE
**风险／权限：** 对象存在性范围有限；所有方法相同生成/解析规则；子集图片可能已在其他任务出现。
**来源：** H05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS10 — HallusionBench
**用途：** D2；evaluation only。
**独立单位与划分：** original/edited-image family and question-pair family。完整官方split及VD/VS；不将image-only镜像951样本自动等同原1129问答。
**指标：** aAcc; fAcc; qAcc分别报告。
**获取：** https://github.com/tianyi-lab/HallusionBench
**评价入口：** https://github.com/tianyi-lab/HallusionBench
**风险／权限：** 官方自动评价和自定义strict yes/no分开；API调用需已授权；配对不能打散bootstrap。
**来源：** H06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS11 — AMBER
**用途：** D2；evaluation only。
**独立单位与划分：** image。generative与discriminative及existence/attribute/relation分项。
**指标：** CHAIR/Cover/Hal/Cog; no-class P/R/F1; overall Acc。
**获取：** https://github.com/junyangwang0410/AMBER
**评价入口：** https://github.com/junyangwang0410/AMBER
**风险／权限：** 判别P/R/F1正类是幻觉问题(真值no)，不能反用POPE模板；规则库/spaCy版本pin。
**来源：** H07（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS12 — COCO captions / Karpathy split
**用途：** D2;D3;D4；train/secondary eval; D3 grouped reference only after audit。
**独立单位与划分：** COCO original image。Karpathy train/val/test与COCO2014/2017编号对应；跨RefCOCO/POPE/GQA关系图审查。
**指标：** D4 bidirectional R@1/5/10,1k5fold与5k分开; D2 CHAIR/CIDEr。
**获取：** https://cocodataset.org/
**评价入口：** https://github.com/karpathy/neuraltalk2
**风险／权限：** 五条caption不是五个独立图像；未标注对未知；不把单图caption当完整条件生成真值。
**来源：** F05;M05;M06;H08（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS13 — CLEVR / controlled rendered extension
**用途：** D1;D3；controlled checks; not substitute for natural task。
**独立单位与划分：** scene latent/template family。先生成并冻结基础scene split，再渲染/改述/污染；所有同scene变体同组。
**指标：** exact answer; finite S truth; condition/mode error。
**获取：** https://cs.stanford.edu/people/jcjohns/clevr/
**评价入口：** https://github.com/facebookresearch/clevr-dataset-gen
**风险／权限：** 自定义信道须标custom；只有观察映射已知时才能称observed S oracle；renderer版本/许可证核对。
**来源：** B05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS14 — GenEval
**用途：** D3;D2T2I；generation evaluation only。
**独立单位与划分：** prompt template / concept family。官方prompt list与原生默认采样数固定；自定义难度训练不重用eval prompts。
**指标：** six official subcategory scores and overall。
**获取：** https://github.com/djghosh13/geneval
**评价入口：** https://github.com/djghosh13/geneval
**风险／权限：** detector version阈值pin；语义分不是分布距离真值；有无best-of-N不同预算单列。
**来源：** G04（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS15 — T2I-CompBench original
**用途：** D3；evaluation only。
**独立单位与划分：** prompt/family。original6000/6cat；不要与++混用。
**指标：** 原版attribute/relation/compositional evaluator outputs。
**获取：** https://github.com/Karine-Huang/T2I-CompBench
**评价入口：** https://github.com/Karine-Huang/T2I-CompBench
**风险／权限：** 多种评估器含LLM/VQA；记录缓存题目、权重与调用成本。
**来源：** G05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS16 — T2I-CompBench++
**用途：** D3；extension evaluation。
**独立单位与划分：** prompt/family。++8000/8cat固定release。
**指标：** ++官方分项与聚合。
**获取：** https://github.com/Karine-Huang/T2I-CompBench
**评价入口：** https://github.com/Karine-Huang/T2I-CompBench
**风险／权限：** 必须独立version字段；原版eval不能沿用++名称。
**来源：** G06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS17 — ConceptMix
**用途：** D3；evaluation/controlled difficulty。
**独立单位与划分：** prompt/concept tuple。固定概念k、缓存prompts/questions；template families heldout。
**指标：** per-concept and all-constraints success。
**获取：** https://princetonvisualai.github.io/conceptmix/
**评价入口：** https://princetonvisualai.github.io/conceptmix/
**风险／权限：** 生成题目和裁判可能用付费LLM；不得静默自动调用；检查human subset。
**来源：** G07（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS18 — GenAI-Bench / BaiqiL release
**用途：** D2T2I;D3；human evaluation; no verifier training on holdout。
**独立单位与划分：** prompt and generator family。固定HF版本; train/eval by prompt、generator，旧版/新版标注数不混。
**指标：** human alignment correlation; pairwise accuracy; GenAI-Rank bestof。
**获取：** https://huggingface.co/datasets/BaiqiL/GenAI-Bench
**评价入口：** https://github.com/linzhiqiu/t2v_metrics
**风险／权限：** 与GenAI Arena同名数据分开；多版本标注量变化，按本地清单统计，不写固定通用80k。
**来源：** G08;H10（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS19 — HPDv2 / HPSv2
**用途：** D3；external human preference evaluation。
**独立单位与划分：** prompt/grouped generation。使用官方heldout，不作为拟合critic训练数据。
**指标：** human pair preference accuracy/correlation。
**获取：** https://github.com/tgxs002/HPSv2
**评价入口：** https://github.com/tgxs002/HPSv2
**风险／权限：** 偏好不等于支持事实；固定judge/model，授权与素材许可核验。
**来源：** G03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS20 — PartiPrompts
**用途：** D3；prompt stress evaluation。
**独立单位与划分：** prompt/concept family。发布prompt分组固定；没有真实图集oracle。
**指标：** 官方/预注册human or semantic evaluators。
**获取：** https://github.com/google-research/parti
**评价入口：** https://github.com/google-research/parti
**风险／权限：** prompt-only不能估计未知真实条件图像分布。
**来源：** G03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS21 — Flickr30k captions
**用途：** D4；training/transfer evaluation。
**独立单位与划分：** original image。官方/Karpathy image split标明；SNLI-VE/FlickrEntities共享排重。
**指标：** i2t/t2i R@1/5/10。
**获取：** https://shannon.cs.illinois.edu/DenotationGraph/
**评价入口：** https://github.com/naver-ai/pcmepp
**风险／权限：** 图像下载许可可能受限；五描述不是全正例图；不得复用已消费DEV当独立确认。
**来源：** M01;M02（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS22 — CUB-200-2011 + Reed captions
**用途：** D4；controlled class-heldout retrieval。
**独立单位与划分：** bird image and class。PCME150train/50heldout class protocol复核；caption/image共同分组。
**指标：** class-relevant retrieval / PMRP per code; query-image matching分开。
**获取：** https://github.com/naver-ai/pcme
**评价入口：** https://github.com/naver-ai/pcme
**风险／权限：** 同类正例协议是类相关而非逐声明视觉真值；文本数据单独许可。
**来源：** M01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS23 — CxC / Crisscrossed Captions
**用途：** D4；evaluation only。
**独立单位与划分：** COCO image and paraphrase family。官方COCO eval索引；稀疏人工judgment范围。
**指标：** semantic score correlation; expanded retrieval。
**获取：** https://github.com/google-research-datasets/Crisscrossed-Captions
**评价入口：** https://github.com/naver-ai/eccv-caption
**风险／权限：** 267095标注对非独立样本；0–5相似度非匹配后验，未judge不能自己当负。
**来源：** M05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS24 — ECCV Caption
**用途：** D4；evaluation only。
**独立单位与划分：** COCO image/caption query family。Karpathy test子集1332query images/1261query captions按官方索引。
**指标：** eccv_map_at_r; eccv_rprecision; recalls。
**获取：** https://github.com/naver-ai/eccv-caption
**评价入口：** https://github.com/naver-ai/eccv-caption
**风险／权限：** mAP@R的R是已发布合法相关数；保留官方评价宇宙，另对fully judged子集评风险。
**来源：** M06（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS25 — SUGARCREPE++
**用途：** D4;D2；evaluation only。
**独立单位与划分：** COCO original image/triplet。官方两正一负triplet完整保留；不加入训练伪正例。
**指标：** 官方triplet语义/词汇一致性指标。
**获取：** https://arxiv.org/abs/2406.11171
**评价入口：** https://arxiv.org/abs/2406.11171
**风险／权限：** 服务器从原论文核到author dataset后冻结hash；先不猜数据仓库。
**来源：** M08（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS26 — DataComp-CLIP Small
**用途：** D5；complete downstream-training benchmark。
**独立单位与划分：** source image; domain/duplicate cluster。官方pool+12.8M training exposure设置核验；内部小样本另名。
**指标：** 38 downstream任务官方summary; ImageNet/检索分项。
**获取：** https://www.datacomp.ai/dcclip/getting_started.html
**评价入口：** https://github.com/mlfoundations/datacomp
**风险／权限：** URL失效与安全过滤所有臂同处理；需磁盘/下载授权，代码许可不等于图像许可；不可默下完整medium。
**来源：** D01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS27 — Conceptual Captions 3M
**用途：** D5；bounded fallback training, not DataComp。
**独立单位与划分：** source image/domain。先共同抓取固定pool，按域/图片hash分target-reference与源。
**指标：** fixed student heldout classification/retrieval。
**获取：** https://ai.google.com/research/ConceptualCaptions/
**评价入口：** https://github.com/mlfoundations/open_clip
**风险／权限：** 下载失效清单、许可审核；不是官方DataComp-small。
**来源：** D01;D03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS28 — Recap-DataComp-1B matched caption subset
**用途：** D5；recaptioning controlled experiment。
**独立单位与划分：** same image across all captions。同图original/recap不能跨角色；只用预先允许的子集。
**指标：** student quality at same images/exposure; factuality audit。
**获取：** https://github.com/UCSC-VLAA/Recap-DataComp-1B
**评价入口：** https://github.com/UCSC-VLAA/Recap-DataComp-1B
**风险／权限：** 不要下载1B全量；synthetic caption teacher污染/成本记账；caption长度独立控制。
**来源：** D04（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS29 — QVHighlights
**用途：** D6；train/CAL/full task eval。
**独立单位与划分：** parent YouTube video incl sliced IDs。固定官方train/val/test；内部CAL按parent vid；2sec clip索引和duration锁定。
**指标：** MR mAP(0.5:0.05:0.95), R@1(0.5/0.7); HD mAP/Hit@1。
**获取：** https://github.com/jayleicn/moment_detr
**评价入口：** https://github.com/jayleicn/moment_detr/tree/main/standalone_eval
**风险／权限：** annotations CC BY-NC-SA4; code MIT; 当前官方称Codalab关闭、test GT公开，公开不等于训练许可。
**来源：** V02;V03（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS30 — Charades-STA
**用途：** D6；secondary full temporal grounding。
**独立单位与划分：** original video / recording family。官方STA train/test；没有官方CAL则只从train按视频切。
**指标：** R@1/5 at tIoU0.3/0.5/0.7 and mIoU where official。
**获取：** https://github.com/jiyanggao/TALL
**评价入口：** https://github.com/jiyanggao/TALL
**风险／权限：** 使用STA而非动作分类annotation；缓存C3D/I3D/CLIP features不同不可混排。
**来源：** V03;V04（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS31 — ActivityNet Captions
**用途：** D6；optional broader videos。
**独立单位与划分：** original video。train/val1/val2 protocol明写；不把val2随口称官方test。
**指标：** temporal R@K; dense caption指标分别。
**获取：** https://cs.stanford.edu/people/ranjaykrishna/densevid/
**评价入口：** https://github.com/ranjaykrishna/densevid_eval
**风险／权限：** YouTube URL attrition；单/多描述、弱/强监督分表；基础视频排重。
**来源：** V04;V05（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS32 — HiREST
**用途：** D6；hierarchical extension。
**独立单位与划分：** original video and step family。官方split按video；不将同视频各step打散。
**指标：** video retrieval; moment retrieval; step captioning。
**获取：** https://openaccess.thecvf.com/content/CVPR2023/html/Zala_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_CVPR_2023_paper.html
**评价入口：** 主论文/项目中核验后再执行，当前未给出已验证接口。
**风险／权限：** 官方数据/评测入口从paper核验；当前摘要级，不自动执行。
**来源：** V09（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS33 — Ego4D NLQ
**用途：** D6；long-video extension。
**独立单位与划分：** original recording and participant where available。官方NLQ train/val/test与license流程；不混MQ/GoalStep。
**指标：** R@1/5 at tIoU0.3/0.5。
**获取：** https://ego4d-data.org/
**评价入口：** https://github.com/EGO4D/episodic-memory
**风险／权限：** 需数据许可/下载凭证；不能以冻结特征免去原视频成本声称部署加速。
**来源：** V11（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS34 — HowTo100M
**用途：** D6；optional weak pretraining only。
**独立单位与划分：** parent video/creator。叙述配对与下游数据去重；只对预先授权子集。
**指标：** zero-shot retrieval / downstream training。
**获取：** https://www.di.ens.fr/willow/research/howto100m/
**评价入口：** https://www.di.ens.fr/willow/research/mil-nce/
**风险／权限：** 网页许可和视频授权分开；没有标注不等于随机负例都真实错；不列首批全量。
**来源：** V01（见 ../literature/SOURCE_CATALOG_CN.md）。

## DS35 — MomentSeeker
**用途：** D6；recent long-video external evaluation。
**独立单位与划分：** long original video。官方release/split待服务器接入核验。
**指标：** official long-video retrieval metrics。
**获取：** https://proceedings.neurips.cc/paper_files/paper/2025/hash/281e0b9142763f2b6c944fedb8550ba9-Abstract-Datasets_and_Benchmarks_Track.html
**评价入口：** 主论文/项目中核验后再执行，当前未给出已验证接口。
**风险／权限：** 本輪主源摘要已核，数据与完整evaluator未下载；不能虚构参数/运行成本。
**来源：** V10（见 ../literature/SOURCE_CATALOG_CN.md）。



---

<!-- source: literature/VERIFICATION_AND_SEARCH_CN.md -->

# 检索方法、覆盖与待核验项

检索日期：2026-10-11。问题导向的选择性综述，不宣称穷尽2024–2026全部VL论文。围绕六组关键词查原始会议页、arXiv、作者项目/代码和数据/evaluator。先查最新直接邻居，再追溯基础工作；不按“是否支持VCS”筛文献。

## 本轮检索范围

| 方向 | 核心检索词 | 重点出处 |
|---|---|---|
| D1 | visual token pruning; information bottleneck; evidence-aware; dynamic budget; DiffPrune | CVPR/ICCV/PMLR与arXiv/official repos |
| D2 | visual factual support; hallucination; contrastive decoding; visual entailment; VQAScore | CVF/ACL/AAAI/arXiv，官方hallucination evaluator |
| D3 | conditional generative evaluation; cFreD/CFID; CMMD; text-image faithfulness/coverage | CVF/PMLR/arXiv，生成评估项目 |
| D4 | probabilistic embeddings; missing positives; PCME/ProLIP; CxC/ECCV Caption | CVF/ICLR/ACL/official repos |
| D5 | DataComp; data filtering network; recaption; submodular distribution matching | DataComp官方、ICLR/PMLR/CVF |
| D6 | temporal support; narrated video; moment retrieval; QVHighlights | CVF/NeurIPS/作者代码与官方metrics |

## 必须影响设计的近期先例

T05 IF-Prune已经是信息瓶颈引导剪枝，T06 DiffPrune是任务驱动可微信息节流，T07 ET-Prune是证据感知动态预算。D1不能以这些词本身宣称首次。T06/T07按2026预印本处理，不虚构会议录用或可运行代码。

H04 SCI已有视觉/文本反事实迭代；G03 cFreD已有条件生成质量评价；D05 SDM已有目标分布过滤。它们需要实质对照，不应被从“最接近工作”表中省略。

M04 LongProLIP在arXiv备注中是ICLR2025不确定性workshop tiny paper，不是ICLR主会。D02 DFN确认是ICLR2024，D04再描述确认是ICML2025。

## 核验深度与限制

`papers.csv`逐项注明摘要、全文或官方README。能核对方法和任务，不等于已下载模型、重建训练或核实每张结果表。若只读到摘要，就不臆造全部数据集/训练预算/超参数；服务器须在进入该基线时补全文和evaluator。

IF-Prune的CVF全文本轮访问受限；论文摘要、作者页及官方仓库已核，README明确ChartQA/Qwen/InternVL接口，但“八项评测”完整清单尚需补。SCI、Sim-DETR、SG-DETR、MomentSeeker的全文/部署细节未逐项复现。DiffPrune/ET-Prune是可读预印本，代码可用性未保证。当前部分源码链接来自论文公开入口，尚未安装；不存在的/迁移的链接按原作者路线恢复，不猜镜像。

本包没有将发表论文的SOTA数值复制成服务器预期结果。只有相同host、split、输入和evaluator时才能建立reproduction anchor；其余先以原方法定义作为对照，实际跑出本地结果。

## 数据与评估的版本陷阱

- SP-DocVQA与DocVQA2026不同。
- T2I-CompBench原版6000/6类与++8000/8类不同。
- BaiqiL/GenAI-Bench与GenAI Arena命名不同团队；固定项目身份。
- HallusionBench full版与镜像image-only样本数不同。
- QVHighlights官方当前说明Codalab已关闭并公开test真值；保持最终评价封存，不向旧服务误提交。
- PCME的CUB类相关性不是声明真值；CxC分级相似度不是概率；ECCV Caption标注域不完整。

## 论文引用

本包BibTeX只包含已核对的最小metadata，不猜作者拼写、页码和DOI；成稿从出版社BibTeX补齐。第三方论文全文和代码未打包分发，避免无必要复制与许可证问题。



---

<!-- source: literature/SOURCE_CATALOG_CN.md -->

# 文献与来源注册表

检索截止：2026-10-11。本表是按六个研究问题选择的原始文献，不是所有VL论文的系统综述。`verification_depth`描述本轮实际读到的层次；所有“复现”均指计划，未在本地运行第三方神经模型。论文数据/方法记录与后面的自拟实验分开。

## F01 — Self-supervised Representation Learning with Relative Predictive Coding
- 发表状态：ICLR 2021
- 主来源：https://arxiv.org/abs/2103.11275
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：相对密度比二次目标；平衡参数与当前J相同。
- 原文的数据／评估：CIFAR; STL-10; ImageNet; LibriSpeech；表示学习与依赖估计。
- 核验层次：primary_full_text。不是另一个不同的平衡VCS基线。

## F02 — Relative Density-Ratio Estimation for Robust Distribution Comparison
- 发表状态：NeurIPS 2011 / arXiv revision
- 主来源：https://arxiv.org/abs/1106.4729
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：混合分母的相对密度比与最小二乘估计。
- 原文的数据／评估：受控分布及分布比较应用；相对Pearson误差/应用性能。
- 核验层次：primary_full_text。

## F03 — Composite Binary Losses
- 发表状态：JMLR 2010
- 主来源：https://www.jmlr.org/papers/v11/reid10a.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：正规复合损失与概率估计；有限类下不存在无条件统一优劣。
- 原文的数据／评估：理论；条件风险与后验拟合。
- 核验层次：primary_full_text。

## F04 — f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization
- 发表状态：NeurIPS 2016
- 主来源：https://arxiv.org/abs/1606.00709
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：一般f散度的变分估计与生成学习。
- 原文的数据／评估：生成建模；散度与生成样本评价。
- 核验层次：primary_abstract。本轮用于一般先行工作，不移植其数值。

## F05 — Learning Transferable Visual Models From Natural Language Supervision
- 发表状态：ICML 2021
- 主来源：https://proceedings.mlr.press/v139/radford21a.html
- 官方代码／项目入口：https://github.com/openai/CLIP
- 原文方法：图文对比双编码器与零样本迁移。
- 原文的数据／评估：大规模图文预训练; 多个分类和检索集；零样本准确率; 双向Recall@K。
- 核验层次：primary_paper_or_proceedings。

## F06 — Sigmoid Loss for Language Image Pre-Training
- 发表状态：ICCV 2023
- 主来源：https://openaccess.thecvf.com/content/ICCV2023/html/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.html
- 官方代码／项目入口：https://github.com/google-research/big_vision
- 原文方法：逐对sigmoid替代批softmax；关注正负比例与配对预算。
- 原文的数据／评估：图文预训练; 零样本分类与检索；准确率; 检索; batch/配对效率。
- 核验层次：primary_paper_or_proceedings。

## F07 — SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
- 发表状态：2025 technical report
- 主来源：https://arxiv.org/abs/2502.14786
- 官方代码／项目入口：https://huggingface.co/google
- 原文方法：多目标和数据策略增强图文表示；作为现代预训练参考。
- 原文的数据／评估：多语言图文; 定位及稠密特征任务；分类/检索/定位。
- 核验层次：primary_abstract。不能把整个预训练模型优势解释成单一sigmoid损失优势。

## F08 — Qwen2.5-VL Technical Report
- 发表状态：2025 technical report / official model card
- 主来源：https://arxiv.org/abs/2502.13923
- 官方代码／项目入口：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- 原文方法：动态视觉分辨率的开放VLM；作为D1/D2宿主。
- 原文的数据／评估：文档/图表/OCR/通用视觉；各官方任务指标。
- 核验层次：official_model_card。固定7B checkpoint版本、max/min pixels、token数与dtype；不是调用在线服务。

## T01 — An Image is Worth 1/2 Tokens After Layer 2: Plug-and-play Inference Acceleration for Large Vision-Language Models
- 发表状态：ECCV 2024
- 主来源：https://arxiv.org/abs/2403.06764
- 官方代码／项目入口：https://github.com/pkunlp-icler/FastV
- 原文方法：FastV利用早期层后注意力裁减视觉token。
- 原文的数据／评估：LLaVA系; Qwen-VL; 视频任务；原任务表现; FLOPs; 推理延迟。
- 核验层次：primary_full_text。

## T02 — SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
- 发表状态：ICML 2025
- 主来源：https://proceedings.mlr.press/v267/zhang25s.html
- 官方代码／项目入口：https://github.com/Gumpest/SparseVLMs
- 原文方法：文本raters、逐层稀疏与token回收；免训练。
- 原文的数据／评估：GQA; MMBench; MME; POPE; ScienceQA; SEED; TextVQA; MMVet；官方任务分数; 保持率; FLOPs; CUDA时间。
- 核验层次：primary_full_text。保持原表现97%不等于准确率97%。

## T03 — Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration
- 发表状态：ICCV 2025
- 主来源：https://arxiv.org/abs/2412.13180
- 官方代码／项目入口：https://web.stanford.edu/~markendo/projects/feather
- 原文方法：分析视觉中心任务失效；RoPE-free注意力与两阶段裁减。
- 原文的数据／评估：通用QA; TextVQA; 视觉定位等；任务质量与加速。
- 核验层次：primary_full_text。均值QA不足以说明视觉证据保持。

## T04 — Conical Visual Concentration for Efficient Large Vision-Language Models (PyramidDrop)
- 发表状态：CVPR 2025
- 主来源：https://arxiv.org/abs/2410.17247
- 官方代码／项目入口：https://github.com/Cooperx521/PyramidDrop
- 原文方法：分阶段减少视觉token，训练/推理设定分别处理。
- 原文的数据／评估：多项图像问答任务；任务准确率; 训练/推理成本。
- 核验层次：primary_abstract。arXiv原标题PyramidDrop与最终题名区分。

## T05 — IF-Prune: Information-Flow Guided Token Pruning for Efficient Vision-Language Models
- 发表状态：CVPR 2026
- 主来源：https://openaccess.thecvf.com/content/CVPR2026/html/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.html
- 官方代码／项目入口：https://github.com/snap-research/EVLM-IF-Prune
- 原文方法：小VLM的变分信息瓶颈提供问题条件下的视觉裁减信号。
- 原文的数据／评估：8项评测；官方README明确含ChartQA，Qwen2-VL/InternVL系；任务表现保持率; token/计算成本。
- 核验层次：primary_abstract_and_official_repo。CVF全文本轮访问受限，八集完整名称不能据其他论文补齐；官方README BibTeX仍有占位符，不采用。

## T06 — DiffPrune: Differentiable Information Throttling for Token Pruning in Vision-Language Models
- 发表状态：arXiv preprint, August 2026
- 主来源：https://arxiv.org/abs/2608.01985
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：可微信息节流训练token评分器；以任务损失学习裁减。
- 原文的数据／评估：LLaVA1.5-7B; LLaVA-NEXT7B; Qwen2.5-VL7B; GQA/TextVQA等10任务；固定预算任务质量; 实际效率。
- 核验层次：primary_full_text。所读版本未核实可用官方代码，不能列为已可直接复跑。

## T07 — ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs
- 发表状态：arXiv preprint, August 2026
- 主来源：https://arxiv.org/abs/2608.01979
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：问题条件证据评分、文本空间保护与动态预算下限。
- 原文的数据／评估：Qwen3-VL8B; InternVL3.5-8B; OCRBench-v2; MMBench1.1；OCR/QA; token数与时延。
- 核验层次：primary_full_text。属于直接概念邻居；不宣称首次evidence-aware动态预算。

## H01 — Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
- 发表状态：CVPR 2024
- 主来源：https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html
- 官方代码／项目入口：https://github.com/DAMO-NLP-SG/VCD
- 原文方法：原图和扰动图条件的输出对比，免训练减轻幻觉。
- 原文的数据／评估：POPE等对象幻觉与通用理解；幻觉指标; QA质量; 额外forward成本。
- 核验层次：primary_paper_or_proceedings。

## H02 — Detecting and Preventing Hallucinations in Large Vision Language Models
- 发表状态：AAAI 2024
- 主来源：https://ojs.aaai.org/index.php/AAAI/article/view/29771
- 官方代码／项目入口：https://github.com/hendryx-scale/mhal-detect
- 原文方法：M-HalDetect细粒度标注；奖励/偏好优化与best-of-N。
- 原文的数据／评估：M-HalDetect; 多源模型回答；声明/片段正确性; 人评; 生成质量。
- 核验层次：primary_full_text。

## H03 — Woodpecker: Hallucination Correction for Multimodal Large Language Models
- 发表状态：2023 arXiv; venue not re-verified here
- 主来源：https://arxiv.org/abs/2310.16045
- 官方代码／项目入口：https://github.com/BradyFU/Woodpecker
- 原文方法：结构化核查与多步骤生成纠错。
- 原文的数据／评估：图文问答/幻觉评测；纠错效果; 信息保持; 推理成本。
- 核验层次：primary_abstract。不要仅凭仓库标记推定发表场馆。

## H04 — Scaling Test-Time Robustness of Vision-Language Models via Self-Critical Inference Framework
- 发表状态：CVPR 2026
- 主来源：https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Scaling_Test-Time_Robustness_of_Vision-Language_Models_via_Self-Critical_Inference_Framework_CVPR_2026_paper.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：SCI多轮视觉/文本反事实；DRBench动态压力测试。
- 原文的数据／评估：DRBench及VLM鲁棒性任务；正确性与测试时计算。
- 核验层次：primary_abstract。全文/代码未完整读到；动态模型专属题池不能无说明横向汇总。

## H05 — Evaluating Object Hallucination in Large Vision-Language Models
- 发表状态：EMNLP 2023
- 主来源：https://aclanthology.org/2023.emnlp-main.20/
- 官方代码／项目入口：https://github.com/AoiDragon/POPE
- 原文方法：POPE对象存在性问答，random/popular/adversarial负例。
- 原文的数据／评估：COCO来源图像等发布子集；Accuracy; Precision; Recall; F1; Yes ratio。
- 核验层次：primary_paper_or_proceedings。

## H06 — HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models
- 发表状态：CVPR 2024
- 主来源：https://arxiv.org/abs/2310.14566
- 官方代码／项目入口：https://github.com/tianyi-lab/HallusionBench
- 原文方法：成对控制图像/问题诊断语言幻觉与视觉错觉。
- 原文的数据／评估：HallusionBench；aAcc; fAcc; qAcc; 分组诊断。
- 核验层次：primary_full_text。图像编辑家族是独立分组单位；原始full版和image-only子集不能混。

## H07 — AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation
- 发表状态：2023 arXiv; venue not asserted
- 主来源：https://arxiv.org/abs/2311.07397
- 官方代码／项目入口：https://github.com/junyangwang0410/AMBER
- 原文方法：生成和判别两种幻觉评价，不依赖生成式LLM裁判。
- 原文的数据／评估：AMBER；CHAIR; Cover; Hal; Cog; Acc/P/R/F1。
- 核验层次：primary_full_text。判别P/R/F1以幻觉问题(真值no)为正类，区别POPE约定。

## H08 — Object Hallucination in Image Captioning
- 发表状态：EMNLP 2018
- 主来源：https://aclanthology.org/D18-1437/
- 官方代码／项目入口：https://github.com/LisaAnne/Hallucination
- 原文方法：CHAIR用对象提及检测图像描述幻觉。
- 原文的数据／评估：COCO captions；CHAIR_i; CHAIR_s; caption质量。
- 核验层次：primary_abstract。CHAIR只覆盖对象，不是完整事实性。

## H09 — TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering
- 发表状态：ICCV 2023
- 主来源：https://openaccess.thecvf.com/content/ICCV2023/html/Hu_TIFA_Accurate_and_Interpretable_Text-to-Image_Faithfulness_Evaluation_with_Question_Answering_ICCV_2023_paper.html
- 官方代码／项目入口：https://github.com/Yushi-Hu/tifa
- 原文方法：把提示词拆成问答，检查生成图是否满足要求。
- 原文的数据／评估：TIFA评测集; 多个T2I模型；VQA问答一致性; 人类相关性。
- 核验层次：primary_paper_or_proceedings。

## H10 — Evaluating Text-to-Visual Generation with Image-to-Text Generation (VQAScore)
- 发表状态：ECCV 2024
- 主来源：https://arxiv.org/abs/2404.01291
- 官方代码／项目入口：https://github.com/linzhiqiu/t2v_metrics
- 原文方法：VQA模型对“是否显示该文字”的Yes概率作为配对评价。
- 原文的数据／评估：多个图文/生成对齐benchmarks；人类偏好相关性/成对排序。
- 核验层次：primary_full_text。

## H11 — Visual Entailment: A Novel Task for Fine-Grained Image Understanding
- 发表状态：2019 arXiv / SNLI-VE dataset
- 主来源：https://arxiv.org/abs/1901.06706
- 官方代码／项目入口：https://github.com/necla-ml/SNLI-VE
- 原文方法：图像前提与文本声明的entailment/neutral/contradiction。
- 原文的数据／评估：SNLI-VE (Flickr30k + SNLI)；3类accuracy; 本方案另加macro-F1/校准。
- 核验层次：primary_abstract。neutral不能当成明确contradiction；检查文本偏差。

## G01 — Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)
- 发表状态：CVPR 2024
- 主来源：https://arxiv.org/abs/2401.09603
- 官方代码／项目入口：https://github.com/google-research/google-research/tree/master/cmmd
- 原文方法：CLIP空间核MMD，不再使用FID的高斯假设。
- 原文的数据／评估：多个图像生成模型/受控扰动；分布距离; 人评相关性; 样本效率。
- 核验层次：primary_full_text。CMMD中的C是CLIP；不等于conditional MMD。

## G02 — Conditional Frechet Inception Distance
- 发表状态：2021/2022 arXiv revision
- 主来源：https://arxiv.org/abs/2103.11521
- 官方代码／项目入口：https://github.com/Michael-Soloveitchik/CFID
- 原文方法：条件Wasserstein/FID的闭式高斯形式。
- 原文的数据／评估：条件生成；CFID与FID/MSE对比。
- 核验层次：primary_abstract。

## G03 — Evaluating Text-to-Image and Text-to-Video Synthesis with a Conditional Fréchet Distance (cFreD)
- 发表状态：WACV 2026, author publication page verified
- 主来源：https://arxiv.org/abs/2503.21721
- 官方代码／项目入口：https://vislang.ai/publications/koo2026evaluating
- 原文方法：联合视觉质量与条件文本的一致性，条件Fréchet距离。
- 原文的数据／评估：HPDv2; GenAI-Bench; PartiPrompts; COCO；人类相关性与模型排序。
- 核验层次：primary_full_text。比无条件FID/CMMD更直接的概念对照；版本标题有增添video。

## G04 — GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment
- 发表状态：NeurIPS 2023 Datasets and Benchmarks
- 主来源：https://arxiv.org/abs/2310.11513
- 官方代码／项目入口：https://github.com/djghosh13/geneval
- 原文方法：基于对象检测的可组合T2I要求检查。
- 原文的数据／评估：GenEval prompts；单物体/多物体/数量/颜色/位置/颜色绑定及平均。
- 核验层次：primary_abstract_and_official_repo。检测器错误另审计；并非目标生成分布真值。

## G05 — T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation
- 发表状态：NeurIPS 2023
- 主来源：https://arxiv.org/abs/2307.06350v2
- 官方代码／项目入口：https://github.com/Karine-Huang/T2I-CompBench
- 原文方法：属性绑定、对象关系及复杂组合的分项评价。
- 原文的数据／评估：6000 prompts / 6 categories (original)；绑定/关系各官方评价器。
- 核验层次：primary_abstract。只用original版本时不能引用++的8000/8。

## G06 — T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation
- 发表状态：TPAMI 2025
- 主来源：https://doi.org/10.1109/TPAMI.2025.3531907
- 官方代码／项目入口：https://github.com/Karine-Huang/T2I-CompBench
- 原文方法：扩展组合类别和更细粒度诊断。
- 原文的数据／评估：8000 prompts / 8 categories (++ release)；版本对应的官方子类分数。
- 核验层次：primary_abstract。固定version/commit；与原版不得混排。

## G07 — ConceptMix: A Compositional Image Generation Benchmark with Controllable Difficulty
- 发表状态：NeurIPS 2024 Datasets and Benchmarks
- 主来源：https://arxiv.org/abs/2408.14339
- 官方代码／项目入口：https://princetonvisualai.github.io/conceptmix/
- 原文方法：按概念数量控制生成要求难度，自动问答并有人类验证。
- 原文的数据／评估：ConceptMix；各难度满足概念比例与全部满足率。
- 核验层次：primary_abstract。涉及LLM题目/裁判版本与付费调用；缓存优先。

## G08 — GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation
- 发表状态：2024 arXiv / CVPR 2024 SynData workshop, not claimed main track
- 主来源：https://arxiv.org/abs/2406.13743
- 官方代码／项目入口：https://huggingface.co/datasets/BaiqiL/GenAI-Bench
- 原文方法：复杂提示、图/视频生成的人类对齐评分；GenAI-Rank候选比较。
- 原文的数据／评估：GenAI-Bench / GenAI-Rank；人类相关性; 成对偏好; best-of-N。
- 核验层次：primary_full_text。以BaiqiL/GenAI-Bench为身份；不与GenAI Arena同名数据合并，固定release。

## G09 — Demystifying MMD GANs
- 发表状态：ICLR 2018
- 主来源：https://arxiv.org/abs/1801.01401
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：KID的无偏MMD评价背景。
- 原文的数据／评估：生成图像；KID; MMD。
- 核验层次：primary_abstract。作为公式先行工作，数值由共享评价实现复跑。

## G10 — Reliable Fidelity and Diversity Metrics for Generative Models
- 发表状态：ICML 2020
- 主来源：https://proceedings.mlr.press/v119/naeem20a.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：Density/Coverage把保真和覆盖分开。
- 原文的数据／评估：生成模型；Density; Coverage; precision/recall相关比较。
- 核验层次：primary_abstract。和条件忠实性分开，不用单个S宣称区分所有质量因素。

## M01 — Probabilistic Embeddings for Cross-Modal Retrieval
- 发表状态：CVPR 2021
- 主来源：https://arxiv.org/abs/2101.05068
- 官方代码／项目入口：https://github.com/naver-ai/pcme
- 原文方法：高斯概率嵌入与不确定性，多对多检索。
- 原文的数据／评估：COCO Caption; CUB Captions；Recall@K; PMRP/类相关检索。
- 核验层次：primary_abstract_and_official_repo。CUB同类相关性不等于每条caption对每张图都真实。

## M02 — Improved Probabilistic Image-Text Representations (PCME++)
- 发表状态：ICLR 2024
- 主来源：https://proceedings.iclr.cc/paper_files/paper/2024/hash/ad9d6ab10446114cf5482d5e1f971a84-Abstract-Conference.html
- 官方代码／项目入口：https://github.com/naver-ai/pcmepp
- 原文方法：闭式概率距离与伪正例、混合训练缓解假负例。
- 原文的数据／评估：COCO Caption; CxC; ECCV Caption；多正例检索; 噪声/不确定性。
- 核验层次：primary_abstract_and_official_repo。

## M03 — ProLIP: Probabilistic Language-Image Pre-Training
- 发表状态：ICLR 2025
- 主来源：https://arxiv.org/abs/2410.18857
- 官方代码／项目入口：https://github.com/naver-ai/prolip
- 原文方法：大规模概率预训练与语义包含/不确定性。
- 原文的数据／评估：DataComp; 零样本分类; 检索; 不确定性任务；检索/分类/包含和不确定性。
- 核验层次：primary_full_text。预训练数据规模不同需系统参考表与同条件损失表分开。

## M04 — LongProLIP: A Probabilistic Vision-Language Model with Long Context Text
- 发表状态：ICLR 2025 uncertainty workshop tiny paper / arXiv, not main
- 主来源：https://arxiv.org/abs/2503.08048
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：概率模型拓展长文本。
- 原文的数据／评估：Urban1k; DataComp任务等；长描述检索; 分类保持。
- 核验层次：primary_abstract。标题/版本以注册URL为准；不列未核实场馆。

## M05 — Crisscrossed Captions: Extended Intramodal and Intermodal Semantic Similarity Judgments for MS-COCO
- 发表状态：EACL 2021
- 主来源：https://aclanthology.org/2021.eacl-main.249/
- 官方代码／项目入口：https://github.com/google-research-datasets/Crisscrossed-Captions
- 原文方法：补充图图/文文/图文的人类语义相似度。
- 原文的数据／评估：CxC on COCO；语义相关性; retrieval。
- 核验层次：primary_abstract_and_official_repo。0–5语义分不直接当后验概率；267095是标注对，不是独立图像。

## M06 — ECCV Caption: Correcting False Negatives by Collecting Machine-and-Human-verified Image-Caption Associations for MS-COCO
- 发表状态：ECCV 2022
- 主来源：https://arxiv.org/abs/2204.03359
- 官方代码／项目入口：https://github.com/naver-ai/eccv-caption
- 原文方法：修复COCO检索漏标正例，引入mAP@R。
- 原文的数据／评估：ECCV Caption / COCO Karpathy test subset；mAP@R; R-Precision; Recall@K。
- 核验层次：official_repo_full。只标注部分查询；公布的judge universe与未判定对严格区分。

## M07 — FLAIR: VLM with Fine-Grained Language-Informed Image Representations
- 发表状态：CVPR 2025
- 主来源：https://arxiv.org/abs/2412.03561
- 官方代码／项目入口：https://github.com/ExplainableML/flair
- 原文方法：细描述与文本条件视觉池化支持局部关系。
- 原文的数据／评估：细粒度/长描述检索与区域任务；检索/局部对应。
- 核验层次：primary_abstract_and_official_repo。代码原生文本条件池化不能被换成弱全局cosine。

## M08 — SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations
- 发表状态：2024 arXiv / verify venue before manuscript
- 主来源：https://arxiv.org/abs/2406.11171
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：同义正描述与困难负描述，分离句式和语义。
- 原文的数据／评估：SugarCrepe++；官方双正例组合指标。
- 核验层次：primary_abstract。身份URL需服务器按原作者页面复核后下载，非核心训练数据。

## D01 — DataComp: In Search of the Next Generation of Multimodal Datasets
- 发表状态：NeurIPS 2023 Datasets and Benchmarks
- 主来源：https://arxiv.org/abs/2304.14108
- 官方代码／项目入口：https://www.datacomp.ai/dcclip/
- 原文方法：固定池、模型及计算的图文数据策展基准。
- 原文的数据／评估：DataComp four scales; 38 downstream tasks；38任务平均及ImageNet/检索。
- 核验层次：official_site_and_repo。tiny内部样本不能称官方DataComp-small。

## D02 — Data Filtering Networks
- 发表状态：ICLR 2024
- 主来源：https://proceedings.iclr.cc/paper_files/paper/2024/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html
- 官方代码／项目入口：https://github.com/apple/ml-dfn
- 原文方法：筛选网络的效用不等于其自身下游准确率。
- 原文的数据／评估：DFN pools; 多项零样本评价；由筛选集训练的模型质量。
- 核验层次：primary_abstract。ICLR2024，不是把2023 arXiv当NeurIPS2023。

## D03 — Sieve: Multimodal Dataset Pruning Using Image Captioning Models
- 发表状态：CVPR 2024
- 主来源：https://openaccess.thecvf.com/content/CVPR2024/html/Mahmoud_Sieve_Multimodal_Dataset_Pruning_using_Image_Captioning_Models_CVPR_2024_paper.html
- 官方代码／项目入口：https://github.com/facebookresearch/SIEVE
- 原文方法：生成描述与原文字的语义一致性筛选，结合CLIP。
- 原文的数据／评估：DataComp medium/large; 38 tasks；下游质量; 筛选与caption成本。
- 核验层次：primary_abstract_and_official_repo。

## D04 — What If We Recaption Billions of Web Images with LLaMA-3?
- 发表状态：ICML 2025
- 主来源：https://proceedings.mlr.press/v267/li25ch.html
- 官方代码／项目入口：https://github.com/UCSC-VLAA/Recap-DataComp-1B
- 原文方法：大规模再描述，原文与新caption混合用于CLIP及生成训练。
- 原文的数据／评估：Recap-DataComp-1B; 多项图文检索和生成；检索; 生成忠实性。
- 核验层次：primary_paper_or_proceedings。同一图片的新旧caption不能跨划分。

## D05 — Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders
- 发表状态：ICLR 2026
- 主来源：https://proceedings.iclr.cc/paper_files/paper/2026/hash/8291acb610ddc260d554da5a3d273ae9-Abstract-Conference.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：稀疏自编码器的目标分布匹配用于图文过滤(SDM)。
- 原文的数据／评估：DataComp medium; 38 tasks；目标与整体迁移性能。
- 核验层次：primary_abstract。目标分布匹配已被研究；不是VCS首次。

## D06 — Multimodal Distribution Matching for Vision-Language Dataset Distillation
- 发表状态：arXiv preprint, May 2026
- 主来源：https://arxiv.org/abs/2605.23482
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：多模态分布匹配用于合成数据蒸馏。
- 原文的数据／评估：图文数据蒸馏；由合成数据训练的下游效果。
- 核验层次：primary_abstract。数据蒸馏不等于本方案从固定池过滤；相关而非首批必跑。

## D07 — BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
- 发表状态：ICML 2022
- 主来源：https://proceedings.mlr.press/v162/li22n.html
- 官方代码／项目入口：https://github.com/salesforce/BLIP
- 原文方法：captioner/filter组合清理和补充图文预训练。
- 原文的数据／评估：图文预训练/检索/生成任务；检索与生成质量。
- 核验层次：primary_abstract。再描述/过滤先行工作。

## V01 — End-to-End Learning of Visual Representations from Uncurated Instructional Videos (MIL-NCE)
- 发表状态：CVPR 2020
- 主来源：https://arxiv.org/abs/1912.06430
- 官方代码／项目入口：https://www.di.ens.fr/willow/research/mil-nce/
- 原文方法：多实例NCE处理叙述视频固有时间错配。
- 原文的数据／评估：HowTo100M train; YouCook2/MSR-VTT/CrossTask/COIN等；检索; 动作分类/定位/分割。
- 核验层次：primary_abstract_and_author_site。

## V02 — QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries
- 发表状态：NeurIPS 2021
- 主来源：https://arxiv.org/abs/2107.09609
- 官方代码／项目入口：https://github.com/jayleicn/moment_detr
- 原文方法：查询条件多时间段定位与显著性，Moment-DETR端到端头。
- 原文的数据／评估：QVHighlights；MR mAP; R@1; HD mAP/Hit@1。
- 核验层次：official_repo_full。2026-10-11官方README称Codalab关闭并发布test真值；仍保持一次性评价边界。

## V03 — Query-Dependent Video Representation for Moment Retrieval and Highlight Detection (QD-DETR)
- 发表状态：CVPR 2023
- 主来源：https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Query-Dependent_Video_Representation_for_Moment_Retrieval_and_Highlight_Detection_CVPR_2023_paper.html
- 官方代码／项目入口：https://github.com/wjun0830/QD-DETR
- 原文方法：查询依赖表示与负查询训练。
- 原文的数据／评估：QVHighlights; TVSum; Charades-STA；moment retrieval/highlight官方指标。
- 核验层次：primary_paper_or_proceedings。

## V04 — Knowing Where to Focus: Event-aware Transformer for Video Grounding (EaTR)
- 发表状态：ICCV 2023
- 主来源：https://openaccess.thecvf.com/content/ICCV2023/html/Jang_Knowing_Where_to_Focus_Event-aware_Transformer_for_Video_Grounding_ICCV_2023_paper.html
- 官方代码／项目入口：https://github.com/jinhyunj/EaTR
- 原文方法：事件结构引导视频定位。
- 原文的数据／评估：视频语言时间定位集；R@1 tIoU; mAP等官方任务指标。
- 核验层次：primary_paper_or_proceedings。

## V05 — Correlation-Guided Query-Dependency Calibration for Video Temporal Grounding (CG-DETR)
- 发表状态：2023/2024 arXiv revision; venue not re-verified
- 主来源：https://arxiv.org/abs/2311.08835
- 官方代码／项目入口：https://github.com/wjun0830/CGDETR
- 原文方法：dummy tokens与跨模态相关性调节查询参与程度。
- 原文的数据／评估：QVHighlights; temporal grounding benchmarks；MR/HD指标。
- 核验层次：primary_abstract。不要将confidence calibration与它的query-dependency calibration混同。

## V06 — Background-Aware Moment Detection for Video Moment Retrieval (BM-DETR)
- 发表状态：WACV 2025
- 主来源：https://openaccess.thecvf.com/content/WACV2025/html/Jung_Background-Aware_Moment_Detection_for_Video_Moment_Retrieval_WACV_2025_paper.html
- 官方代码／项目入口：https://github.com/minjoong507/BM-DETR
- 原文方法：显式处理相关片段与背景。
- 原文的数据／评估：QVHighlights等时间定位集；moment retrieval指标。
- 核验层次：primary_paper_or_proceedings。

## V07 — Sim-DETR: Unlock DETR for Temporal Sentence Grounding
- 发表状态：ICCV 2025
- 主来源：https://openaccess.thecvf.com/content/ICCV2025/html/Tang_Sim-DETR_Unlock_DETR_for_Temporal_Sentence_Grounding_ICCV_2025_paper.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：处理查询竞争与全局/局部定位。
- 原文的数据／评估：temporal sentence grounding benchmarks；官方定位指标。
- 核验层次：primary_abstract。完整数据/权重兼容性先由服务器核实。

## V08 — Saliency-Guided DETR for Moment Retrieval and Highlight Detection (SG-DETR)
- 发表状态：WACV 2026
- 主来源：https://openaccess.thecvf.com/content/WACV2026/html/Gordeev_Saliency-Guided_DETR_for_Moment_Retrieval_and_Highlight_Detection_WACV_2026_paper.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：显著性引导MR/HD，使用更强视频预训练信息。
- 原文的数据／评估：QVHighlights及时间定位；MR/HD。
- 核验层次：primary_abstract。额外InterVid等预训练/特征单列，不能当纯损失比较。

## V09 — Hierarchical Video-Moment Retrieval and Step-Captioning (HiREST)
- 发表状态：CVPR 2023
- 主来源：https://openaccess.thecvf.com/content/CVPR2023/html/Zala_Hierarchical_Video-Moment_Retrieval_and_Step-Captioning_CVPR_2023_paper.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：长视频中的层级片段和步骤描述。
- 原文的数据／评估：HiREST；视频/片段retrieval; step captioning。
- 核验层次：primary_abstract。captioning与定位不可混成单一分数。

## V10 — MomentSeeker: A Benchmark for Long-Video Moment Retrieval
- 发表状态：NeurIPS 2025 Datasets and Benchmarks
- 主来源：https://proceedings.neurips.cc/paper_files/paper/2025/hash/281e0b9142763f2b6c944fedb8550ba9-Abstract-Datasets_and_Benchmarks_Track.html
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：更长视频和多种查询形态。
- 原文的数据／评估：MomentSeeker；长视频定位/检索。
- 核验层次：primary_abstract。先核数据可得性与实际特征成本；不列入首批全面训练。

## V11 — NaQ: Leveraging Narrations as Queries to Supervise Episodic Memory
- 发表状态：CVPR 2023
- 主来源：https://arxiv.org/abs/2301.00746
- 官方代码／项目入口：https://vision.cs.utexas.edu/projects/naq/
- 原文方法：将叙述转为Ego4D NLQ训练监督。
- 原文的数据／评估：Ego4D NLQ；R@1/5 at tIoU 0.3/0.5。
- 核验层次：primary_abstract。扩展训练监督量必须同信息预算控制。

## B01 — GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering
- 发表状态：CVPR 2019
- 主来源：https://cs.stanford.edu/people/dorarad/gqa/index.html
- 官方代码／项目入口：https://cs.stanford.edu/people/dorarad/gqa/evaluate.html
- 原文方法：场景图与可组合问题。
- 原文的数据／评估：GQA；accuracy; consistency; validity; plausibility; grounding。
- 核验层次：official_site_full。VG特征与原val重叠警告遵照官方说明。

## B02 — Towards VQA Models That Can Read
- 发表状态：CVPR 2019
- 主来源：https://textvqa.org/
- 官方代码／项目入口：https://github.com/facebookresearch/mmf
- 原文方法：自然图像OCR问答；多个人类答案。
- 原文的数据／评估：TextVQA；官方VQA软准确率。
- 核验层次：official_site。具体归一化用官方/M4C evaluator，不手写简化count/3。

## B03 — DocVQA: A Dataset for VQA on Document Images
- 发表状态：WACV 2021 / SP-DocVQA challenge 2020
- 主来源：https://www.docvqa.org/datasets/docvqa
- 官方代码／项目入口：本轮未核实可运行官方代码；不要猜仓库地址。
- 原文方法：文档图像上的问答。
- 原文的数据／评估：SP-DocVQA；ANLS; 原任务协议。
- 核验层次：official_site_full。不等同新DocVQA2026；获取受官方注册流程约束。

## B04 — ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning
- 发表状态：Findings of ACL 2022
- 主来源：https://aclanthology.org/2022.findings-acl.177/
- 官方代码／项目入口：https://github.com/vis-nlp/ChartQA
- 原文方法：human和augmented图表问题。
- 原文的数据／评估：ChartQA; FigureQA; PlotQA; DVQA；数值5%容差、非数值精确匹配。
- 核验层次：primary_full_text。

## B05 — CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning
- 发表状态：CVPR 2017
- 主来源：https://cs.stanford.edu/people/jcjohns/clevr/
- 官方代码／项目入口：https://github.com/facebookresearch/clevr-dataset-gen
- 原文方法：结构化场景与问题程序，适合规定有限信道。
- 原文的数据／评估：CLEVR；exact answer accuracy; 本方案有限分布真值。
- 核验层次：official_project。标签信道真值不自动是任意冻结视觉特征的密度比真值。



---

<!-- source: sources/USER_SOURCES_CN.md -->

# 用户材料与仓库截点

U01：用户提供的《Variational Cauchy–Schwarz Quadratic Mutual Information — A Stable Alternative to MINE》，2026年9月研究计划。重点§2–8、§10–13、§17。文件副本SHA256：`0f87809ebaad8fb64cfbec9e442b23104150f73294401a2177a5b9e47de884da`。

U02：2026-09-28《对 VCS_QMI_Estimator_Research_and_Server_v1 的补充意见》，重点§2.6与§3.3。其“只有平方拟合才可用增量”等说法须结合后续修订：恒等式属于目标/读数，可使用Logistic拟合后转换的T。

U03：2026-10-10项目审查及附录4。本文只继承更正后的事实/边界：RPC目标重合、不同损失仍能改变编码器、经验J非逐次下界、先验与语义事件区分、标签信道观测条件。旧正文尚存冲突句，不作为本包算法事实。

本轮只读GitHub检查：`W-Yinghao/VCS_QMI` main = `87a847e7a9138e5b52ff19546b62e188d07e315d`，提交日期2026-10-11 18:59:12 UTC。该提交是r6数据/许可/基线审查，未下载新数据；接入时应重新核对实际HEAD，不据本包断言服务器任务仍处同一状态。

URL：https://github.com/W-Yinghao/VCS_QMI/commit/87a847e7a9138e5b52ff19546b62e188d07e315d

本包不是对67篇论文全部结果表的逐条复算；其核验深度见SOURCE_CATALOG_CN.md。没有从已被修订的旧审查复制录用门槛、未经证实的普遍优势或不成立定理。
