# V9 · 可靠跨模态对应：量化、加权与有限配准验证
**版本：V9 / server v1 · 2026-10-16。与 V8 并行，不替代 V8。**

主问题：面对不完美的跨模态对应，能否先正确量化其可靠度，再让这个数值改善完整视觉任务，而不是仅改变一个不会影响 argmax 的评分刻度？

主线 **NR（Noisy Correspondence Reliability）**：可信参考 → 批次混合比例 → 样本成分后验 → 完整图文检索训练。辅线 **REG**：复用已有跨模态配准，检验对应监督质量与训练样本量。第三方向“零/多目标对应”只保留问题卡。

## 阅读顺序
1. `01_SERVER_AGENT_PROMPT_CN.md`：接入与并行约束。
2. `docs/02_MASTER_PLAN_CN.md`：任务依赖与本轮范围。
3. `docs/03_THEORY_CONTRACT_CN.md`：原理论、直接推导、假设和反例。
4. `docs/04_NR_QUANTIFICATION_CN.md`、`05_NR_FULL_TRAINING_CN.md`：主线实施。
5. `docs/06_REG_EXTENSION_CN.md`：配准的有限扩展。
6. `docs/07_DATA_AND_BENCHMARKS_CN.md`、`08_STATISTICS_CN.md`、`09_PARALLEL_EXECUTION_CN.md`。
7. `literature/REVIEW_CN.md`：相关工作与核验深度。

完整合并文本见 `FULL_PLAN_CN.md` / `FULL_PLAN_CN.html`；七页左右的执行概览见 `V9_OVERVIEW_CN.pdf`。`INDEX.html` 是离线入口。

## 本包完成与未完成的部分
这是可执行规范、参考数学实现与接入材料，不是已经接入服务器的数据管线。本地测试核对有限分布恒等式、比例区间、权重与分组规则；不包含自然图像训练、原方法复现、服务器作业或新实验结果。自然数据 adapter 和训练脚本由 server 依规范完成并记录差异。

```bash
python -m pytest -q tests
python scripts/demo_finite.py --out validation/demo_finite.json
python scripts/audit_mixture.py examples/independent_scores.json --out validation/example_audit.json
```

无需把第三方数据、模型权重、字体或完整论文打包进仓库。保留来源、版本、文件哈希与许可记录。


---

# 发给 server agent：V9 v1，与 V8 并行

请接入本包，研究版本为 **V9**。V8 继续按自己的冻结协议训练与评价；V9 不等待 V8 结果转正，也不读取 V8/GenAI 封存标签来调 V9。

**本轮主任务不是重新跑 J-only adapter。** 主线是利用有界 CS critic 估计规定配对机制的比例与样本可信度，再将固定权重用于完整图文任务；辅线在既有 REG 上只增加“训练对应质量 × 训练样本量”。

首先核对 HEAD 与运行环境。本包检查截点：main=`2d171a6`，v7=`2dc39b5`，v8=`a59c833`。新提交优先读取，逐项写增量，不覆盖已有冻结结果。建议独立 worktree/branch `v9/reliable-correspondence`，输出与缓存写到本机配置的V9目录；不能写V8目录。新分支由服务器在实际仓库建立，本包没有远程写操作。

现在就执行：
- V9-00 接入盘点、单元测试、数据/许可/基线可运行性检查；报告可信参考从哪里来、占多少标注预算。
- V9-N1 先跑有限支持与已有图文特征的比例/概率实验，包含raw score、BBSE、MLE/EM、KDE、RuLSIF、Nyström；BCE仅是其中的归因臂。
- V9-N2 准备Flickr30K和COCO的完整SGR检索训练；先复现NCR/L2RM的原生评价，再执行同等可信锚点预算的主比较。不是冻结探针终结。
- V9-G1 准备BF-SHG的有限配准扩展；若EPFL使用依据未解决，不自动开启该数据集新训练。

不要等待“VCS先打赢BCE”才做外部比较；不得按seed0是否领先决定是否跑后两种子。技术失败只暂停受影响单元。大任务启动前填清预算与依赖，以实际测得时间排队；V8有独立保留份额，V9不得取消或挤出其在跑任务。无既有算力授权覆盖的部分提交负责人确认，不猜测授权。

请特别落实：
1. `P/Q` 是配对生成机制，不自动是语义正确/错误；训练损失读取不了注入标签或原始正确映射。
2. 需要可信P/Q参考的方案，所有比较方法获得相同锚点与标签预算；它是 **trust-assisted** 设置，不冒充原生无锚点NCR榜单。
3. 比例公式对任意固定、可分辨的score成立，因此没有“J更高必然比例更准”；弱分离返回宽区间，不输出虚假精度。
4. 特征后验的精确重加权不能无条件外推到读取原图/全特征的student；普通in-batch损失也不自动无偏。严格风险核对与完整原生学习分表。
5. 修复配对、在线更新teacher、开放目标定位、额外大规模预训练均不是首批默认作业。

首轮请交 `V9_00_INTAKE.md`、`V9_01_SOURCE_AND_BUDGET.md`、`V9_02_PROTOCOL_FILLED.yaml`、`V9_03_BASELINE_FIDELITY.md` 与已完成的N1结果。随后按已批准资源完成全部预定正式单元。报告把[已知证据]、[本轮结果]、[解释]、[未验证假设]分开。


---

# 02 · 科学目标、范围与任务清单

## 1. 一句话方向
用规定的可信与错误配对参考，把不完美对应的**批次比例与样本可信度**估计出来，并通过完整检索学习与有限配准验证，检验数值是否有实际价值。

这不是已证明优于全部方法的新算法。V9要验证三段链条：

`配对机制可辨认 → 比例/后验估计足够准确 → 完整任务得到可重复的收益`。

## 2. 继承的证据，而非成功保证
| 已有材料 | 可继承内容 | 本轮不能直接推出来的结论 |
|---|---|---|
| P157 [R01](https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/P157_SEP2_REPORT_20261015.md) | 指定受限critic、合成污染后验下，平方拟合有重复优势 | 自然干净后验恢复、端到端VL鲁棒性 |
| P159 [R02](https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/P159_SEP3_NATURAL_REPORT_20261015.md) | 污染拟合后干净公共J的改善；cos-2排名完全相同 | 校准自动提高Top-1；所有核方法较弱 |
| REG [R03](https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/R6/R6_REG_P3_REPORT_20261012.md) | 已测跨模态配准中学习型关系能量有效 | VCS独占收益；原生CoMIR全面被超过 |
| 旧R线 [R05，附录4及旧负结果] | J-only adapter及错配检测并未支持普遍优势 | 不能仅换名称重跑并声称新的理论应用 |

V9保留原S/J、总体精确gap；新的比例/风险接口和反例是本包的直接推导。有关旧材料的数学误述，仅引用已更正的附录4，不恢复旧预测数字和场馆门槛。

## 3. 优先级与依赖
| ID | 内容 | 前置 | 本轮状态 |
|---|---|---|---|
| V9-00 | 双线隔离、来源、数据/预算、CPU核对 | 无 | 立即 |
| V9-N1a | 有限支持：已知混合、弱分离、先验改变、负控制 | 00 | 立即 |
| V9-N1b | 图文特征：比例、可信度、统计条件、外部量化基线 | 00/数据到位 | 主线首批 |
| V9-N2-F | Flickr30K完整trust-assisted检索主比较 | N1定义正确、基线复现、预算 | 主线完整训练第一块 |
| V9-N2-C | COCO完整主比较 | 同上；不以Flickr胜负为条件 | 主线第二块 |
| V9-N3 | 5%锚点/自然CC152K/重匹配扩展 | 主实验后独立决定 | 条件实验 |
| V9-G1 | BF-SHG两种训练规模、干净/错配训练配准 | 00/既有REG adapter | 小型并行辅线 |
| V9-G2 | 第二模态、局部缺失测试、原生系统 | 权限/预算另行完成 | 条件实验 |
| V9-Z | 零目标/多目标对应 | 只准备文献与指标 | 不自动训练 |
| V9-END | 三类证据：统计、外部任务、归因 | 各结果完整 | 收尾 |

**并行不等于所有网格同时提交。** NR为V9主计算线；REG先复用缓存/模型，限制一组新的科学变量。V8结果不改变N1/N2的主指标或选模。

## 4. 期待的优势与正式评价
H1：在相同参考信息和预算下，批次比例误差/置信区间效率优于BBSE或MLE/EM等强方法。H2：可信度数值带来比同排序过滤更好的加权学习。H3：完整检索超过NCR/L2RM等同信息对照或在相同准确率下用更少可信标注。H4：配准在训练对应不准或样本减少时改善成功率/成本。

四条均待验证。H1不能单独支撑“鲁棒检索”；H3即使成立也需区分teacher、数据与VCS目标的贡献。没有承诺固定百分点，没有以单项p值作为整个项目生死门。

## 5. 完整实验规模
N2每数据集：8种已定义方法 × 2种主噪声率(0,0.4) × 3个种子 = **48个正式训练单元**；Flickr30K与COCO合计96，分块执行。8种方法为Unweighted、NCR+A、L2RM+A、VCS-Weight、BCE-Weight、RuLSIF-Weight、Nystrom-Weight、RankMass。所有方法都有相同可信锚点预算，A表示anchor-assisted适配。

这不是现在就提交96作业。先量测一epoch、确定host/数据/评价兼容并填预算。可以合并短作业；不得因某种子输而省略同一正式块其余种子。若预算只允许Flickr块，明确称单数据集结果，COCO标记未完成。

REG-G1：4种可训练方法(VCS/BCE/InfoNCE/CoMIR) × 2种训练规模 × 2种训练对应质量 × 3种子 = **48个训练单元**，其中完全一致的既有全量干净单元可复用，须列明实际新增数。无训练的MI/NMI/MIND/经典CS等只拟合其需要的CAL参数。

外部论文复现、teacher、critic调参、核求解、所有失败/重试成本单列，96/48不是全部计算量。参考代码不是GPU性能承诺。


---

# 03 · 理论与实施对象：保持S/J，补齐使用条件

## A. 原理论 [R05](local-source://conversation/Variational-CS-QMI-and-supplement)
设P,Q为两个概率分布，M=(P+Q)/2，eta=(p-q)/(p+q)。

\[
S=E_M\eta^2,\qquad J(T)=E_PT-E_QT-\tfrac{1}{2}E_PT^2-\tfrac{1}{2}E_QT^2,\quad S-J(T)=E_M(T-\eta)^2.
\]

这是总体恒等式，不是每次经验读数的下界。基础J与平衡RPC重合；密度比回归、正规评分及CS参考联系分别归属，不认领新散度。[L13,L14]

V9的P/Q可以是“原始标注配对/规定随机重配”；只有严格联合对乘积构造才称QMI。拒绝同图donor的Q是条件化抽样分布，不等同于无条件乘积Q。

## B. 本包直接推导1：任意固定读数的比例估计
令R_rho=(1-rho)P+rho Q。对固定且a!=b的T，a=E_PT,b=E_QT,c=E_RT，则
\[
\rho=(a-c)/(a-b).
\]
不需要T已达到eta，也不需要概率校准。对T=eta，E_P eta=S、E_Q eta=-S，所以a-b=2S。

**关键实施含义：**
- 此公式本质属于矩匹配/label shift思路，BBSE是强对照，不是新定理。[L10](https://proceedings.mlr.press/v80/lipton18a.html)
- T换成uT+v(u!=0)，同样本的比例点估计严格不变。线性尺度优化不能单独创造比例估计优势。
- 单调但非线性变换可能改变矩估计效率；更小后验MSE或更高J不保证该比值方差最小。
- a-b接近0时问题弱识别。报告原始比值、投影到[0,1]的显示值与识别状态；不能仅裁剪后声称准确。
- 已知P/Q参考和一份混合不是单成分PU问题。只有混合样本、没有参考或其他可识别条件时，不能无条件识别rho。[L12](https://proceedings.mlr.press/v48/ramaswamy16.html)

若独立估计三个均值，delta近似方差为
\[
\operatorname{Var}(\hat\rho)\approx[\operatorname{Var}(\hat c)+(1-\rho)^2\operatorname{Var}(\hat a)+\rho^2\operatorname{Var}(\hat b)]/(a-b)^2.
\]
这是诊断近似，不用于弱分离的主置信区间。

## C. 本包直接推导2：弱分离时用集合反演
对互相独立、每个值在[-1,1]内的P/Q/R评价簇均值，各有n_P,n_Q,n_R个独立簇。令
`e_j=sqrt(2 log(6/alpha)/n_j)`。
用并合界至少1-alpha概率同时覆盖三个均值。比例置信集为
\[
\mathcal{C}=\{r\in[0,1]: |\hat c-(1-r)\hat a-r\hat b|\le e_R+(1-r)e_P+r e_Q\}.
\]
这是两个线性不等式的交集；不需要除以小的估计分母。输出可能是宽区间、[0,1]或空集。空集表示该单一矩在规定误差带内不相容，不自动定位原因；非空也不证明没有域移位。

**仅当相应独立簇/有界/混合假设成立才使用该有限样本解释。** 多caption、交叉配对共享图像时，pair数不是独立n。自然数据普通bootstrap另报告为重采样近似；refit bootstrap需在每次重复重做被抽样的拟合/选择步骤。

参考模块的`rho_confidence_set`只实现上述数学。调用者必须提供独立性检查，不能仅传一个`True`替代数据证明。

## D. 本包直接推导3：正确的成分后验
如果T逼近的是**干净参考P对Q**的eta，则可信成分后验为
\[
w_\rho(z)=\frac{(1-\rho)(1+T(z))}{(1-\rho)(1+T(z))+\rho(1-T(z))}.
\]
直接输出sigma(logit(1-rho)+2f)，T=tanh f，避免先除出巨大p/q。rho=0/1的合法支持点单独处理；数值裁剪率要记录。若T在R对Q上训练，公式不能直接套用：必须单独定义其密度比并检查识别/反演误差。

w是“该配对来自规定P成分的概率”，不自动是语义满足概率。随机重配中偶然正确的caption仍来自Q；语义标签另设，不能暗中更换真值。

## E. 重要补充：权重并不无条件恢复任意student的风险
若W是全部观测，C表示来自P，真实w(W)=E[C|W]，对固定可积loss ell(W)：
\[
E_R[w(W)\ell(W)]=(1-\rho)E_P\ell(W).
\]
若额外负样本U与(C,W)独立，固定分布，则同式对ell(W,U)也成立。

但实际critic常只读Z=phi(W)。即使准确学到w_Z=E[C|Z]，对读取更多信息的student也有
\[
E_R[w_Z\ell(W)]-(1-\rho)E_P\ell(W)
=-E_R\operatorname{Cov}(C,\ell(W)\mid Z).
\]
再有估计误差时，另加`E[(w_hat-w_Z)ell]`。如果|ell|<=L，该额外项绝对值不超过`L sqrt(E(w_hat-w_Z)^2)`；**协方差项不会自动消失**。

这说明“teacher特征后验准确 → 原图student无偏学习”不是定理。只有loss对Z可测，或相应充分性条件成立时才可省略该项。参考测试包括Z为常数而C与loss相关的反例。

同样，SGR/CLIP的常规in-batch hard negatives、不同权重改变的采样、阈值删对、EMA重拟合，都可能改变目标。完整任务仍可实证有效，但不能把单样本恒等式作为整个批次损失的精确保证。

## F. 固定权重的优化规范
首轮critic冻结、权重detach；student不通过权重反向更新critic。训练集尺度常数在启动前固定，不按当前minibatch的sum(w)改变有效目标。若为了稳定采用自归一化权重，它是有限样本有偏的ratio estimator，应命名并对所有方法共享。

不要直接乘原生softmax logit：那会改变比较对象。对匹配loss的每个anchor行/列损失乘对应w_i，具体改动写diff。原始权重、加权后总质量、有效样本量(ESS)均记录。

## G. REG中的目标
固定T时优化J(T;P_phi,Q_phi)是一个学习到的能量，不等于每个phi的真实S。监督错配、测试局部缺失分别改变不同分布。成功率、优化预算与真实几何误差才是应用终点。

## H. 新颖性与证据
本包不把矩匹配、BBSE、EM、CS点集配准、L2投影、重要性加权重新称为新发明。候选新增内容是：在明确参考条件下完成可量化的对应可靠性与学习接口，并在强外部方法上验证收益。最终新颖性是否充分，需要实际方法和结果支持。


---

# 04 · NR-N1：比例与可信度估计

## 1. 三个场景分表
**N1-Exact（主数学验证）：**有限支持P/Q已知，R由iid混合生成；真实eta/w/rho可计算。检查比例、概率、风险及置信集。

**N1-Image（主自然表示验证）：**可信原始图文参考P、明确抽样的Q、未知混合比例R；按图像隔离拟合、校准、参考矩与批次。注入标签只给离线评分器。这里可评价生成机制标签Brier，不把原始caption保证为语义完美。

**N1-Stress（模型假设压力）：**只改变域不改变rho；相似物体hard mismatch；参考锚点有误；重复caption或新类别导致类条件改变。此处不声称混合置信区间必然覆盖。目标是揭示不可识别/错误解释，不隐藏失败。

## 2. 默认输入与critic
优先复用P49的COCO image-disjoint缓存作探索，记录其已被观察，不能将其称新独立确认。Flickr30K的正式角色重新锁定。第一次主量化使用共同teacher的**单一配对相似度s**，以隔离概率刻度，不借更强输入战胜标量基线。

VCS：T=tanh(a s+b)，a=softplus(a_raw)，避免score方向翻转。匹配BCE用同一自然logit 2(a s+b)，不是两种不同函数类。

确定性共同特征敏感性：加入teacher共同池化图文向量、低秩双线性(rank=16)和小MLP；所有方法得到同样输入。主效果只发生在一个受限类时，保留这个范围。不得在看到N1评价后再只留下该类。

## 3. 拟合与选择
- P/Q各1/2权重；源参考不是U噪声训练池；所有标准化仅FIT。
- 3个训练种子{901,902,903}，独立调参种子917；标量类用确定性多启动求解检查加同预算Adam敏感性。
- 学习率/正则/epoch在REF_SELECT上选择，主规则共同J；另保留各方法原生准则的敏感性。不能依据目标批次真实rho挑model。
- 原生BCE、温度/Platt/isotonic校准的BCE分开；所有校准使用相同reference budget。
- 核拟合记录带宽/中心/正则和边界。1D标量主面板不强迫KDE去做不必要的高维密度估计；共同向量敏感性允许PCA与带宽选择。

## 4. 必做量化方法
| 方法 | 使用信息/功能 |
|---|---|
| Raw-score moment | 未训练的共同score，rho=(a-c)/(a-b)；排除只是刻度改变 |
| VCS moment | VCS的固定score接同一矩式 |
| BCE moment | matched score；Logistic不是唯一外部基线 |
| BBSE hard/soft | 同一reference信息、混淆/均值矩阵；检查可逆性 |
| Bias-corrected calibration + MLE/EM | 按label-shift似然估计rho；强概率基线 |
| KDE posterior + moment/EM | 相同s或共同特征；标签与调参预算匹配 |
| RuLSIF(alpha=1/2) + moment/EM | 传统相对密度比；T=r_rel-1，裁剪与原始值均存 |
| Nyström squared regression + moment/EM | 调好核回归，不把RFF名当独立理论方法 |
| Kernel MPE（条件列） | 仅P+R输入和irreducibility条件与双参考不同，单列信息预算 |

先估计rho，再用同一rule计算可信度w；双参考方法可以复用相同rho估计接口以隔离后验拟合。真rho给权重的版本只能是oracle-rho诊断，不能进入部署主表。

## 5. 固定初始网格
主比例{0,.1,.3,.5}，独立批次大小{128,512,2048}。N1-Exact每格200个独立重复，检查区间覆盖；N1-Image先30个随机批次重复/模型种子，明确共享数据会导致重复间相关，不把30当30个新数据集。

画三个图：rho真实值与估计；相同覆盖下区间宽度；概率Brier与ESS。主比较按两个自然来源与4种比例平均RMSE分source报告，不按有利rho挑结果。

## 6. 主指标与失败处理
rho_raw、rho_clipped、absolute_error、RMSE、interval_width、coverage、empty/full interval率、有效P/Q/R簇数；样本成分Brier/logloss、AUROC、分桶校准；读数J作为辅指标。

弱分离时返回[0,1]或明确无法精确量化；数值分母检查不是成功标准。若纯域移位造成高“错配率”，报告模型假设失效，不能以更窄区间补救。增加多个固定score一致性检查可发现部分移位，但不能认证没有移位。

## 7. N1如何进入N2
N1不是“VCS赢BCE才能进入N2”的门。它要求定义/实现正确、权重可计算且无标签泄漏；即使N1方法接近，完整外部比较仍按预算执行。但如果部署接口在当前信息下不可识别，则先修改任务条件，而不是训练一个必然依赖oracle的系统。


---

# 05 · NR-N2：完整噪声图文学习，而不是只做读数器

## 1. 主任务与两个公平性层面
本轮主任务是 **少量可信锚点辅助的噪声图文检索（trust-assisted noisy correspondence learning）**。可信锚点是一项额外信息，不能隐藏。

A表：NCR/L2RM等官方路线、原特征/索引/原生评价，先复现与报告偏差。它不使用新增干净锚点，不能与B表直接宣称同条件SOTA。

B表：所有方法看到相同的可信锚点、相同初始化、相同噪声配对与验证数据；原方法有需要则加入同样anchor-supervised项，标明NCR+A/L2RM+A。这是V9的主要因果与应用表。

C表：固定同一teacher/输入/学习宿主，只替换可靠度估计与使用规则，区分概率读数、rank-only与核方法。

## 2. 数据与可信锚点
Flickr30K先行，COCO第二块；数据保持公开检索train/val/test分区。主协议按原训练图像的10%抽取可信锚点A，90%为未知U；抽取在噪声前，固定seed和图像ID，所有caption随图像分组。

A内按图像再分40% teacher-warmup、30% critic-fit、15% critic-select、15% reference-moments。比例按整数分配后的实际数落盘；每个组不少于可运行最低样本数，否则报数据不足，不移用test。所有角色均可在冻结权重后并入最终student训练，anchor权重为1；这样不会引入测试标签，但性能评价只在官方独立任务集。

teacher：按共同SGR路径，仅在A的teacher-warmup上训练5个完整epoch，冻结其图像投影、文本编码、相似性模块后提取共同score。成本、额外干净监督与所有基线共享。teacher不是VCS先训出来的专用特征；critic-fit等角色不参与teacher选择。

U按固定生成协议注入0或40%**配对机制错配**。为维持A干净且donor只来自U，这一映射不一定等于原论文的全TRAIN索引；故B表明确为trust-assisted构造，原生A表仍读官方索引。所有B表方法使用同一U映射。

生成器保留design_rho、realized_rho、原配对映射、注入位于仅评价侧；critic/student loader只读观测后的pair。真rho不能流进权重函数。实际每个图片组是否错配、caption比例是否同一都落manifest。

## 3. 固定teacher与权重流程
1. 在A-critic-fit上构造P与指定Q，训练/拟合N1各后验方法；Q donor按角色隔离，拒绝同图时明确Q不是无条件乘积。
2. 用A-critic-select作选模；固定模型后用A-reference-moments估计P/Q矩。
3. 在U的**无标签观测**上估计rho。记录这是transductive training-pool估计，不是用EVAL调参；对另一个独立批次报告区间。
4. 对每个U pair算w，写`pair_id,w,score,rho_hat,fit_hash,reference_hash`，锁定后训练student；不在线更新。
5. N1固定的主critic是cos-2；低秩/MLP只有预登记敏感性，不依据任务DEV再挑主函数类。

交叉拟合不是免费证明：首轮采用独立A/U；若以后critic改为读U标签或拟合U，则重新安排out-of-fold，禁止原地切到自训练。

## 4. 完整学习宿主
首选NCR/L2RM共同上游的 **SGRAF中的单模型SGR** 路径。使用原论文公开region特征和词表，训练图像投影、文本编码、相似性推理模块等全部原生可训练部分。

这是该benchmark的完整matching model训练，不是冻结向量上的线性探针；同时不得称为“raw-image视觉骨干端到端训练”。SGR与SGRAF双模型ensemble分别标注。未来CLIP完整微调是另一个现代宿主扩展，不作为本包默认大实验。

server必须读取实际上游配置后填`HOST_LOCK.yaml`：具体commit、feature dims、模块、batch、epoch、优化器、margin、hard-negative规则、warmup、checkpoint选择。不知道的部分标为待核，而不是沿用一个看似合适的猜测。基线能够完整复现后锁定共同recipe，不按VCS的DEV结果优化。

## 5. 权重怎样进入loss
对每个观测正对i，原宿主产生anchor条件下的双向损失ell_i（先得到逐样本row/column项，再聚合）。首轮加权版本：

对合并的 A 与 U，每个 A 样本令 v_i=1，每个 U 样本令 v_i=w_i。主训练目标固定为：

`L = (sum_{i in A} ell_i + sum_{i in U} w_i ell_i) / (n_A + sum_{i in U} w_i)`。

分母在训练前对整个训练池计算。minibatch 用 `mean(v_i*ell_i) / mean_v_full_pool` 估计这一固定目标；所有主方法的anchor角色和自然抽样权重一致。另存U独立的未归一化风险诊断，不将A/U分别归一化后任意相加。

这里mean_w在训练前由全部U冻结，不随minibatch改变；同一recipe所有权重法共用定义。记录这是一种有限样本自归一化加权，而非严格无偏估计。保留未归一化loss的诊断。

w固定且detach，不乘logit，不将低w样本自动从负库删除。原生in-batch hard-negative的潜在错误负例仍存在；加权只能作为**经验方法**，其理论保证不能直接引用独立负库恒等式。

另做一个小但完整的严格接口对照：固定teacher表示，固定独立负库U_neg，逐对loss仅依赖critic可用表示时，核对oracle/posterior权重恢复的风险。它是机制验证，不替代SGR终点。

## 6. 八个正式方法
| 编号 | 方法 | 信息与角色 |
|---|---|---|
| N0 | Unweighted+A | 同锚点teacher/初始化和全部数据，不加可靠度控制 |
| N1 | NCR+A | NCR原纠错机制，同可信锚点；适配写清diff |
| N2 | L2RM+A | 原OT重匹配能力，同锚点；不削成仅过滤 |
| N3 | VCS-Weight | 本包的固定critic → rho → w |
| N4 | BCE-Weight | 同函数类、同teacher、同参考、同rho策略，归因 |
| N5 | RuLSIF-Weight | 传统相对密度比，映射到相同w |
| N6 | Nystrom-Weight | 强核回归，同w接口 |
| N7 | RankMass | 共同原始score排序，保留与N3同等总权重质量的top部分；末端可分数权重 |

RankMass只读取N3训练侧总质量，不读其逐样本排序/测试值；是配对的“数值是否超越同等保留量”诊断，不宣称独立最佳过滤器。另报告自身CAL选择保留率的普通过滤baseline（不额外加入完整主家族）。

ESC/ASL/ReCon是重要外部方法，必须完成论文/代码可用性报告。ReCon本轮官方README仍写coming soon，ESC/ASL代码未核实，不伪造已可运行状态。若实现可用，在结果前追加独立块；若只能按论文复现，清楚命名，报告source fidelity。未补齐时不能声称覆盖所有最新方法或SOTA。

## 7. 调参、种子与单元
正式种子901/902/903；调参种子917不进入正式均值。每个dataset/noise/method完整epoch数据训练，所有种子都运行。各方法获得同样的3个预先固定学习率倍率{0.5,1,2}；原生额外超参数按官方默认，需调时提前写对称可比的预算。

先按Flickr的官方val rSum选任务checkpoint，COCO亦使用同名原生val设置；不可读取test选择。对rho估计器、critic的选择仍只用reference角色。把混合估计与检索选择两层分开。

96正式单元 + 最多96个单种子三点调参单元（8方法×2数据×2噪声×3倍率）是完整规划上限，teacher与原生复现额外计费；执行前实测预算并分块确认。不得把上限当即时提交指令。现有吻合产物可复用并从账单扣除，但必须有相同数据/状态hash。

## 8. 正式评价
每方向R@1/5/10、rSum=六项之和，另给mR=rSum/6，指标单位统一为百分点。COCO 5K gallery为主、5×1K单列；Flickr官方1K为主，不混跨gallery数字。多caption的image-to-text只需任一真实caption命中；text-to-image按真实image ID。

不删除难例/错配误判。每一方法面对同一test candidate全集；两阶段方法/ensemble计全部成本。报告noise=0与0.4的绝对性能和退化，不只画退化百分比。

## 9. 首轮真正要证明什么
主要系统家族：N3 vs NCR+A/L2RM+A，每数据集在rho=.4的rSum，共4个预定主对比；另分传统方法N3 vs RuLSIF/Nystrom共4个；N3 vs N4为归因家族。清洁性能保持和RankMass差异完整报告，不以宽松“非劣效”门槛重新命名失败。

优势可以是任务准确率，或在明确相同质量下减少可信标注/实际成本。仅J/Brier变好而检索无改进，只支持估计结果。下一轮修复/重配对不是默认；先判定权重是否有效以及失败在哪一层。


---

# 06 · REG：有限训练数据 × 对应监督质量

## 1. 本轮定位
REG是V9的低优先级辅线，复用R6已有数据、模型接口、CAL规则与几何评价，不重新启动一个3D/非刚体大项目。[R03,L15-L18]

主问题：在更少的对齐训练图像、或训练中的正对应不完全可靠时，VCS训练的局部关系能量能否比传统统计能量及成熟跨模态方法恢复更多正确变换？不是先假定“有界必然耐噪”。

## 2. 数据和角色
主集BF–SHG，保持原物理切片/样本分组，patch不能跨训练/评价泄漏；原R6已看过DEV，新增分析在其上是开发验证。要新的确认结果，必须锁未参与设计的独立物理样本，否则如实标记重复benchmark。

EPFL作为第二模态数据的条件扩展。其33个旧排除与像素对齐精度限制保留；没有使用依据时不新跑数据。不因内部获准旧实验就宣称得到公开再分发许可。

## 3. 首批二维设计
训练样本量：原FIT的25%与100%，以独立物理图像/切片分层随机抽样，三个seed固定，不以patch数替代样本量。

训练对应：0与20%正配对中的位置错配。错配位置从同一训练图像的规定空间分布选取，排除小于两个patch半径的邻近位置；这个空间条件Q不等同无条件像素乘积。所有可训练方法使用相同错误对应，不把真标签交给VCS去屏蔽。

初始只测训练错配，测试仍用原固定刚体扰动。**测试局部缺失是另一个轴**，不得将这两个混成一个“噪声”曲线。

4可训练方法(VCS、matched BCE、InfoNCE局部critic、CoMIR)×2样本量×2监督状态×3seed=48。全量干净已完成且完全一致的单元复用。MI/NMI、MIND、经典CS无需假装按训练种子重复拟合；核critic需要的训练/校准照实计费。

## 4. 两张方法表
A：所有能量用同一个刚体优化器、同一初始变换、相同可见区域和评价点。这隔离能量与共同搜索的效果。
B：MI的合理多分辨/多起点原生实现、CoMIR原生配准流程、可用MIND系统。按实际搜索成本报告。不能只因为共享优化器更适合宽峰，就声称全面胜过另一方法。

经典CS：可在共同局部特征的pair分布上实现KDE-CS/相对核读数；点集CS原论文是另一观察对象，需要明确关键点抽取与适配，不能将其名字直接挂在稠密图像能量上。

## 5. 主指标
主指标原R6定义的角点误差阈值成功率；同时median/p90角点误差、旋转/平移误差、完整失败率、每次注册时间、能量评估次数。BF–SHG原主阈值42 native px、严格9 px；EPFL10.5/2.25 px不同尺度，分表。

主交互：同方法从0→20%训练错配的绝对变化，以及VCS相对MI/CoMIR/核critic的差值；不把所有重复刚体扰动当独立切片。主家族先锁两个对照MI与CoMIR×两样本量=4，其他完整报告。

## 6. 能量诊断
固定预抽样窗口，画旋转与平移剖面、真值处与估计处能量。能量错误和搜索错误分开。固定T的J是恢复的能量，不声称每个位置已估出真实S。变换后越界像素、可见位置数量、padding必须在所有方法一致；给模型识别padding来获胜不是对齐能力。

## 7. 第二阶段（非首批默认）
只在单独协议下加入50%/75%空间重叠或局部遮挡；固定像素支持/权重，避免重叠面积变小虚增统计量；报告是否给了oracle可见性mask。还可在等质量下研究更少对应监督。不能同时增加新backbone、大变形和新数据，使收益无法解释。


---

# 07 · 数据、基线和评价接口

## 1. 数据矩阵
| 数据 | 首轮用途 | 独立单位/划分 | 正式评价/限制 |
|---|---|---|---|
| COCO P49缓存 | N1探索与代码接续 | 已有按image隔离FIT/CAL/EVAL | 已反复观察，不能当新确认；不是原始训练模型 |
| Flickr30K / NCR数据版本 | N2第一块完整检索 | image ID，全部5 captions同角色 | 官方1K test gallery；全部数据/词表/噪声hash |
| COCO / NCR数据版本 | N2第二块完整检索 | image ID；2014/2017映射要核查 | 5K主表、5×1K另列；不把其混同 |
| CC152K | 自然噪声条件扩展 | 公开train/val/test，URL/图像实存核验 | 无天然噪声比例真值；不能按名给出rho accuracy |
| BF–SHG | REG首批 | 物理切片/样本先分，再切patch | 共用R6真值/单位；被看过的DEV只作开发 |
| EPFL RGB–NIR | REG次轮 | 场景原图/窗口映射 | 许可待核；旧33排除及2px分辨记录保留 |
| gRefCOCO | 开放对应备选 | 原image划分、zero/single/multi | 只准备评测，首批不训练 |

## 2. 从文献到运行的要求
NCR官方README [L02](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) 指明COCO `split=testall`，`fold5=false`为5K、`true`为五个1K。L2RM [L04](https://github.com/hhc1997/L2RM) 指向同一NCR特征/词表与0.2/0.4/0.6/0.8错配索引。**这两个读取事实并不意味着本地文件已存在。**

先生成`DATA_AUDIT.csv`：source URL、local path、sha256、原许可/用途、image/caption数量、去重数、各role、pretraining exposure、坏文件处理。官方split数目以下载实物核验，不凭论文标题猜。

官方特征版和现代CLIP版是不同系统条件。只能在相同条件表里比较直接胜负；额外宿主/外部预训练/ensemble单独列。

## 3. Benchmark三层
**估计层：**有限真值、已定义自然混合；VCS、rawscore、BBSE、MLE/EM、KDE、RuLSIF、Nyström。MPE单成分方法标信息条件。

**应用层：**NCR、L2RM必做；ESC/ASL/ReCon逐项准备；无代码时做源码/论文复现报告，不编结果。混合比例及独立可信集都给所有方法相同使用机会。

**归因层：**matched BCE、RankMass、真rho权重、真注入标签权重（只作带额外信息的diagnostic）。真标签训练不是可部署基线，也不一定是理论性能上界。

## 4. 当前代码可用性与风险
- NCR、L2RM：本轮查到公开实现及README；完整运行/数据下载尚未做。
- ReCon：官方仓库README仍称“代码即将发布”。不能在任务清单中写“直接运行官方实现”。
- ESC、ASL：论文入口确认，本轮未核实执行仓库；负责agent应先获取正文与补充实现说明。
- RRSITR、ConeSep等新领域工作：相关性已经检索，但任务不同，不为增加方法名强塞同一表。
- CoMIR：已有R6接入；原生注册与feature+shared-optimizer必须分开。
- MIND：原论文主要3D，若用2D适配，说明descriptor/offset/优化器，不能直接引用原论文数值。

## 5. 防止“同一来源导致虚假独立”
图片重复、COCO跨版本重复、同caption多个形式、相同切片多个patch按源ID聚合。预训练见过公开数据与本项目标签泄漏是不同问题：公开已知训练来源披露，未知不等于无暴露。

参考集与被量化批次共享同一图像，会低估rho的不确定性。任何跨角色配对都同时检查image-id和caption-donor-id。多次批次抽样不创造新的训练模型或新数据集。


---

# 08 · 指标、统计与结果解释

## 1. N1主指标
rho MAE/RMSE，95%区间coverage/平均宽度/full-set/empty-set比例，成分后验Brier与log loss；配对机制分类AUROC仅补充。原生J单列，先解释对应P/Q。

比较至少包含各方法原始与校准后概率；ECE只作诊断，不作为唯一证据。相同排序的两种概率可以有不同Brier，但不能在固定保留率的rank-only表中伪造差异。

区间反演条件见03。普通ratio bootstrap可能在弱分离下不可靠；保存raw样本均值和间隔，不只存最后裁剪的rho。空区间比例是压力测试输出的一部分，不删除。

## 2. 检索
I2T R@K：top-K中有任一同image的官方caption；T2I R@K：真实image进入top-K。K={1,5,10}；rSum六项相加，mR=rSum/6。打分并列按固定ID排序；额外报告随机并列敏感性。所有baseline用同gallery和ground-truth映射。

主系统比较在rho=.4：VCS vs NCR+A/L2RM+A × 两数据，共4项，Holm校正。传统路线VCS vs RuLSIF/Nystrom为独立4项家族；VCS vs BCE为2项归因家族。报告原始差值、每seed数值和置信区间，不只写“通过/未通过”。

测试图像为配对bootstrap单位，所有caption跟随；gallery固定，重新采样query图像并不构成新gallery。训练种子的离散程度另列，不能将数万caption当作训练重复。3seed时seed CI不稳定，应给全部seed而非仅小p值。

## 3. REG
以物理图像/切片为bootstrap单位，内部patch与10个扰动全部同组；配对同一随机变换。误差在像素坐标中精确反算；native与resize尺度不同必须转换。失败保留并报告，不能只算成功项median掩盖尾部。

## 4. 无数值预言，区分成功的层级
- 统计优：概率/比例估计更准，只在对应样本条件成立。
- 决策优：数值进入权重/注册后，最终任务优于外部方法。
- 方法归因：匹配critic/teacher/信息/成本后还能解释来源。

统计优而任务平局可保留为估计贡献；系统优而BCE相近不能说VCS独占；单一核进入同水平不等于研究无效。不存在本包预先给定的“必须赢X点否则CVPR关闭”。

## 5. 额外监督与成本
A的确认/清理本来就需要人工或可信标注，报告anchor_fraction及每方法可见信息。计算包括teacher、模型提取、核调参、critic训练、rho重估、student全部训练及失败。相同准确率下更便宜的声明需要质量匹配规则在CAL确定。

REG与VLM不同机器的墙钟不能直接相除得加速倍数。采用固定机型/负载的profiling；资源不可用时报告次数与待补计时，不编GPU小时。


---

# 09 · 与V8并行的执行契约

## 1. 不串线
| 项目 | V8 | V9 |
|---|---|---|
| 科学问题 | 总体评分分布与选图 | 可靠对应估计、加权学习、有限配准 |
| 输出 | V8既有目录 | 新V9命名空间 |
| 主评价 | GenAI-Rank与保留确认集 | Flickr/COCO retrieval、独立量化、REG |
| 模型选择 | 原V8 frozen规则 | 本包reference角色与任务CAL |
| 共享 | 原始只读图像/固定特征/工具版本可共享 | 哈希与角色审查后只读挂载 |

不修改V8的labels seal、校准规则、checkpoint或正在运行的脚本。V9不得依据V8外部失败例选择自己要报告的子集。

## 2. 作业设计
新worktree建议`v9/reliable-correspondence`。先读该分支是否已存在；不要猜本地路径。由`.env.local`或YAML输入repo_root/data_root/output_root；示例中不给不可验证的服务器绝对路径。

CPU参考测试、有限支持实验、数据查验先行。GPU任务按实验块打包，支持恢复，保存RNG与optimizer状态；恢复后先与未中断小运行核对。V8已经在跑的任务不取消；并发份额按既有授权与实测可用资源安排，未授权大块由owner确认。

每个任务开始锁code hash/config/data/teacher/weights/noise-map/evaluator hash。结果文件和解释分开提交。技术失败重试不能悄悄换loss/数据/学习率；修改注明新version和受影响单元。

## 3. 第一轮交付
V9_00_INTAKE.md：当前heads、环境、V8隔离、已有数据可复用项。
V9_01_SOURCE_AND_BUDGET.md：基线代码/许可、训练/验证独立性、one-epoch时间。
V9_02_PROTOCOL_FILLED.yaml：host真实字段、科学主指标、methods、seeds、calibration roles。
V9_03_BASELINE_FIDELITY.md：NCR/L2RM官方评价与本地差异；ESC/ASL/ReCon可执行性。
V9_N1_RESULTS/：有限支持+自然混合结果，不含测试注入标签泄漏。

定义正确和资源到位后，执行N2-F完整块，不以VCS胜BCE为启动条件。NR与REG的主结论分开。

## 4. 不在本包自动授权范围
付费API、大规模人工标注、重开已关闭SSL/R线、读取V8未开放评价、删除历史产物、上传第三方数据/权重、CC3M大预训练、无权限数据下载、重匹配OT新机制与GREC完整训练。需要时形成单独增量，不能从“V9并行”推断全部都可做。


---

# 10 · 结果如何决定下一步

## 本轮最可能遇到的五种情况
| 结果 | 应当做什么 | 不应做什么 |
|---|---|---|
| VCS与rawscore/BBSE比例相同 | 检查仿射不变性与方差；将优势范围转到有实际概率用途的部分 | 换成另一个命名指标制造差异 |
| rho准、w准、完整检索无增益 | 看student信息充分性、负库污染与权重ESS；报告决策连接失败 | 继续宣称“无偏恢复全部对比损失” |
| 只在额外干净锚点下好 | 报anchor-assisted任务，评估标注效率 | 与无锚点论文数值直接称SOTA |
| 只在随机错配好、hard mismatch/域变差 | 明确Q机制范围，开发新Q需要新协议 | 把所有语义错误都当独立负样本 |
| 核回归或NCR/L2RM最好 | 保留强baseline；研究实际成本/特殊条件是否有价值 | 缩小核容量、删掉最强方法 |

若REG在训练监督变差时仍同样接近BCE/InfoNCE，却比原生MI或CoMIR系统有改善，归因于学习型关系方法的实际用途；不用强行说二次损失独有。

## 条件性的下一轮
NR-N3a：锚点10%→5%或1%的标注效率，所有方法相同预算；在第一轮协议固定后追加，不按有利anchor数量挑点。
NR-N3b：自然CC152K。若没有成分参考与人工错误标签，只报告检索终点与诊断，不发布“真实噪声率RMSE”。
NR-N3c：重匹配。先将VCS可靠度接入同一L2RM重匹配模块，并给核/BCE同样接口；修复映射不能从原始真值恢复。只有加权接口已分析清楚才启动。
REG-G2：独立对应机制与第二模态/局部缺失，按06的限制。

## 收尾应该留下什么
四张正式表：量化指标；同锚点完整检索；外部原生系统及成本；REG完整几何性能。另用一张归因表记录loss/容量/概率使用方式。所有历史反证和负结果可追溯，不用“只允许某一句话”替代对不确定性的解释。


---

# 11 · 备选：零目标/多目标的开放式对应（不自动运行）

研究问题：候选是否满足表达，而不只是哪个最高。数据入口gRefCOCO/GRES；补ReLA、GSVA、CoHD和近期存在性/数量方法的正式协议后才可立项。

若p_r是真实语义满足概率，固定候选实例定义下sum p_r是满足候选数量期望。但重叠框可能重复计数；P(no target)不能无条件写成乘积(1-p_r)。图内QMI的T>=0只表示相对参考先验的证据，不是语义概率过半。

必要比较：语义BCE/经独立CAL校准的softmax、原生公开GREC/GRES方法、核后验与VCS。指标用该benchmark正式的no-target / target、cIoU/gIoU或精确集合判别，并将概率风险另列。不给VCS原生0阈值特权。

本包只要求准备数据/标签/指标卡，不重开R5训练。只有NR或REG的结果提示明确的新能力、且owner选择此方向时，才形成下一版完整实验。


---

# 12 · 交付复查与源材料使用范围

## 仓库复查
交付前再次读取三个分支，main=2d171a64b90f99b31fda44a6a844f25904c542b0；v7=2dc39b5f9892aba828ea6d857e40dfd8120cbc74；v8=a59c8333e3f272829afd93dcee72d162f98f4285，与开始核查一致。

V8最新提交涉及histogram log loss和温度目标的per-image聚合修正，提交说明写dry-run有6/24单位在场。没有读取V8封存评价，不据此预测V8实验结论。本包不修改其脚本或重新提交作业。

## 合作者材料与旧意见的范围
本次用户给出了《Variational CS-QMI – Research Plan》（September 2026）全文文本；本包采用其中第4–7节的S/J/eta/风险关系、第10节的稳定性限定、第13节的先行工作要求。第17节要求依赖估计在应用中居于核心；本包不改写其基础定义。

《对 VCS_QMI_Estimator_Research_and_Server_v1 的补充意见》（2026-09-28）第1.1节记录旧R线J-only adapter和错配检测负结果。本包将其作为避免重跑旧方案的设计约束，不因新的P157/P159发现而删除。

《VCS-QMI项目：进展、问题、文献组织模式与下一步》（2026-10-10）的早期正文含已更正判断。这里只采用附录4中关于目标/读数、先验、阈值、误差传播、RPC归属与公平性的修正；不恢复正文的“无某性能即关闭场馆”、数字预言或“点式分数不可能排序”等论断。

## 哪些是V9新设计
可信锚点10%及分区、双参考比例/可信度接口、按弱分离反演置信集、信息充分性协方差限制、完整检索主比较与REG的样本量×训练错配网格，均是将上一轮分析实施化的本包建议，不是合作者原稿已经声明的新定理或已经批准运行的旧实验。

## 可用性限制
ReCon官方README本轮仍称代码即将发布；NCR/L2RM只确认公开实现入口，不等于在本机完成复现。旧PDF/论文/数据/模型未被重分发；来源目录保留准确入口与核验深度。服务器需要按照每项真实许可取得使用依据。


---

# 文献综述 · V9所需的三组外部比较

**核验日期：2026-10-16。** 本综述区分原论文/官方README、仓库实证与本包设计。完整入口和核验深度在sources.csv/json；不声称已复现或已逐式读完全部论文。没有将第三方论文全文打包。

## 1. 噪声图文对应：从判别到纠错/修复
| 工作 | 其方法与原任务 | V9应怎样比较 |
|---|---|---|
| NCR, NeurIPS 2021 [L01,L02] | 依据训练中的记忆现象分离干净/错误，再相互监督修正对应 | 原生checkpoint和完整流程复现；A表原协议，B表相同可信锚点 |
| L2RM, CVPR 2024 [L03,L04] | 用OT/partial OT寻找可用重匹配，不只下调错误对 | 保留其修复能力与额外计算；不削成弱过滤对照 |
| ESC, CVPR 2024 [L05](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Robust_Noisy_Correspondence_Learning_with_Equivariant_Similarity_Consistency_CVPR_2024_paper.html) | 比较两模态在anchor附近的语义变化一致性 | 关系性可靠度的外部对照；代码/实现待核，不编现成结果 |
| ASL, AAAI 2025 [L06](https://ojs.aaai.org/index.php/AAAI/article/view/35439) | 正负对非对称学习并对正对自步加权 | 与单纯概率加权直接相关；正式名称是Asymmetric Similarity Learning |
| ReCon, CVPR 2025 [L07,L08] | 模态内外关系一致性帮助区分真假对应 | 不能只对比2021方法；但本次官方README称代码即将发布，需实际复现计划 |
| RRSITR, CVPR 2026 [L09](https://openaccess.thecvf.com/content/CVPR2026/html/Song_Robust_Remote_Sensing_Image-Text_Retrieval_with_Noisy_Correspondence_CVPR_2026_paper.html) | 遥感图文中的清洁/模糊/噪声分组与可靠度调节 | 说明此领域不是空白；不强行混进普通COCO原生表 |

这些工作把任务性能作为终点。V9的区别候选是额外提供明确定义的批次量化和可校验概率接口，但这还需要实验证明有价值。它不是首次做可靠度加权或噪声检索。

## 2. 比例估计、密度比与校准
BBSE [L10](https://proceedings.mlr.press/v80/lipton18a.html) 研究先验改变、类条件不变；预测器即使不完美或未校准，只要对应矩阵可辨认也能估比例。因此rawscore/BBSE不是应当输给VCS的弱方法。

校准+MLE/EM [L11](https://proceedings.mlr.press/v119/alexandari20a.html) 是重要强对照。较准的概率不只影响分类，也影响混合估计；但其成立依赖先验/类条件模型。V9须给它相同参考与CAL，不只跑未校准EM。

Kernel MPE [L12](https://proceedings.mlr.press/v48/ramaswamy16.html) 通常观察混合与一个已知成分，需要额外可识别条件。双参考P/Q问题给了更多信息；分开表述，不能假装方法得到相同输入。

RuLSIF [L13](https://arxiv.org/abs/1106.4729) 使用相对密度比；alpha=1/2可映射到T。Nyström平方回归与KDE分别代表直接回归与显式密度拟合，前者不是“不会学习的核方法”。RPC [L14](https://arxiv.org/abs/2103.11275) 与基础J重合，不作为重复训练出来的新算法行。

**V9新增公式的归属：**固定score三均值比值属于矩匹配；后验权重属于Bayes/重要性加权；有限置信集使用标准Hoeffding并合与线性反演。它们是方法设计依据，不是本包宣称的新定理。

## 3. 跨模态配准
CoMIR [L15,L16] 以InfoNCE学习稠密共享表示，随后进行配准。此前项目主要比较feature+shared-optimizer，不等于原生流水线。

MIND [L17](https://doi.org/10.1016/j.media.2012.05.008) 用局部自相似描述跨模态结构，是非神经/非CS的重要路线。2D实现与原三维协议不同，具体offset/尺度/优化器需要说明。

Information Theoretic Shape Matching [L18](https://pubmed.ncbi.nlm.nih.gov/26353150/) 已经用CS把点集表示成分布后求变换。它的输入是点集，不是现成的BF-SHG稠密配准系统。V9应保留传统CS在适用输入上的直接比较，同时对学习型能量做共享优化器表。

## 4. 开放对应与其限制
GRES/gRefCOCO [L19](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.html) 提供零/多目标场景；能消费绝对概率，但也需要实例去重、联合事件和同等校准。单调转换若不改变排序，在固定覆盖率指标上不能创造优势。

## 5. 下一轮前必须补核的材料
完整NCR/L2RM/ESC/ASL/ReCon正文与supp中的：损失方程、epoch/batch、噪声索引、host/ensemble、所有额外监督。SGR公共接口与evaluation函数逐项对照。最近可运行的方法若加入，先固定协议再读V9结果；无法取得实现则保留缺失和复现路线，不从参考表删除。


---

# 来源目录

每项的核验深度限制其可支持的主张；未核实的运行细节交server补齐。

### R01 · P157 SEP-2：受限 critic 在污染后后验上的比较
- 来源：https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/P157_SEP2_REPORT_20261015.md
- 类型：repository；核验：本轮上下文含报告；此前已读取
- 范围：合成数据，评价污染后后验；不是干净恢复或真实学习结论。

### R02 · P159 SEP-3：冻结自然表示上的比较
- 来源：https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/P159_SEP3_NATURAL_REPORT_20261015.md
- 类型：repository；核验：本轮重新读取全文
- 范围：FIT-ONLY；32/32污染条件公共J较高；cos-2排序相同；quad-3有学习率边界。

### R03 · REG-P3：EPFL RGB–NIR 配准
- 来源：https://github.com/W-Yinghao/VCS_QMI/blob/2d171a64b90f99b31fda44a6a844f25904c542b0/reports/R6/R6_REG_P3_REPORT_20261012.md
- 类型：repository；核验：上下文报告完整
- 范围：共享优化器、2D刚体；不是原生CoMIR系统；许可仍需确认。

### R04 · V8 最新执行提交
- 来源：https://github.com/W-Yinghao/VCS_QMI/commit/a59c8333e3f272829afd93dcee72d162f98f4285
- 类型：repository；核验：本轮读取提交diff
- 范围：v8_eval聚合修正；提交写6/24单元在场，不能当实时进度。

### R05 · 合作者原计划及10月10日综合意见附录4
- 来源：local-source://conversation/Variational-CS-QMI-and-supplement
- 类型：user-source；核验：当前对话给出完整文本
- 范围：保持S/J；旧正文冲突以附录4为准；本包只转述所引用部分。

### L01 · NCR: Learning with Noisy Correspondence for Cross-modal Matching, NeurIPS 2021
- 来源：https://papers.neurips.cc/paper_files/paper/2021/hash/f5e62af885293cf4d511ceef31e61c80-Abstract.html
- 类型：paper；核验：官方摘要+代码README
- 范围：记忆效应、co-teaching、对应关系修正；原数据与训练协议参照。

### L02 · NCR 官方代码
- 来源：https://github.com/XLearning-SCU/2021-NeurIPS-NCR
- 类型：code；核验：检索到官方README
- 范围：COCO testall/fold5需区分；README与许可证展示有差异，以实际LICENSE核对。

### L03 · L2RM: Learning to Rematch Mismatched Pairs for Robust Cross-Modal Retrieval, CVPR 2024
- 来源：https://openaccess.thecvf.com/content/CVPR2024/html/Han_Learning_to_Rematch_Mismatched_Pairs_for_Robust_Cross-Modal_Retrieval_CVPR_2024_paper.html
- 类型：paper；核验：CVF摘要+arXiv入口+官方README
- 范围：OT/partial OT重匹配；不是仅删除低分对。

### L04 · L2RM 官方代码与数据入口
- 来源：https://github.com/hhc1997/L2RM
- 类型：code；核验：本轮GitHub读取README
- 范围：README明确NCR特征数据与0.2/0.4/0.6/0.8索引；CC152脚本命名须检查目录。

### L05 · ESC: Robust Noisy Correspondence Learning with Equivariant Similarity Consistency, CVPR 2024
- 来源：https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Robust_Noisy_Correspondence_Learning_with_Equivariant_Similarity_Consistency_CVPR_2024_paper.html
- 类型：paper；核验：CVF搜索返回摘要；直接open一度403
- 范围：语义变化一致性；本轮没有核实官方可执行代码，服务器需核实/按论文复现。

### L06 · ASL: Noisy Correspondence Rectification via Asymmetric Similarity Learning, AAAI 2025
- 来源：https://ojs.aaai.org/index.php/AAAI/article/view/35439
- 类型：paper；核验：出版方摘要与元数据
- 范围：正式全称是Asymmetric Similarity Learning，不是Asymmetric Self-paced Learning。

### L07 · ReCon, CVPR 2025
- 来源：https://arxiv.org/abs/2502.19962
- 类型：paper；核验：arXiv摘要+作者机构正式出版记录
- 范围：关系一致性；作者代码README仍说coming soon，不能算已有可运行基线。

### L08 · ReCon 官方仓库
- 来源：https://github.com/qxzha/ReCon
- 类型：code；核验：本轮GitHub读取README；blob 89e0140b65ec555d706424fda42562e2744ac6c8
- 范围：本轮读取到的README称代码即将发布；不是已完成复现。

### L09 · RRSITR: Robust Remote Sensing Image-Text Retrieval with Noisy Correspondence, CVPR 2026
- 来源：https://openaccess.thecvf.com/content/CVPR2026/html/Song_Robust_Remote_Sensing_Image-Text_Retrieval_with_Noisy_Correspondence_CVPR_2026_paper.html
- 类型：paper；核验：CVF摘要
- 范围：遥感域噪声分组和可靠度加权；不因换遥感而获得新颖性。

### L10 · BBSE: Detecting and Correcting for Label Shift with Black Box Predictors, ICML 2018
- 来源：https://proceedings.mlr.press/v80/lipton18a.html
- 类型：paper；核验：PMLR摘要与方法定位
- 范围：先验变化、类条件不变；混淆矩阵可逆；校准不是一致性的必要条件。

### L11 · Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation, ICML 2020
- 来源：https://proceedings.mlr.press/v119/alexandari20a.html
- 类型：paper；核验：PMLR摘要与实现入口
- 范围：校准+MLE/EM必须作为强先验估计基线；不是仅比未校准softmax。

### L12 · Mixture Proportion Estimation via Kernel Embeddings of Distributions, ICML 2016
- 来源：https://proceedings.mlr.press/v48/ramaswamy16.html
- 类型：paper；核验：PMLR摘要
- 范围：通常给混合样本+一个成分；与双参考P/Q信息条件不同，需分表。

### L13 · Relative Density-Ratio Estimation for Robust Distribution Comparison, NeurIPS 2011
- 来源：https://arxiv.org/abs/1106.4729
- 类型：paper；核验：此前项目定位；本包保留原始论文入口，服务器补全文
- 范围：RuLSIF alpha=1/2与当前T的变换；主要传统基线。

### L14 · Relative Predictive Coding, ICLR 2021
- 来源：https://arxiv.org/abs/2103.11275
- 类型：paper；核验：当前提供材料附录4明确对应
- 范围：平衡RPC与基础J精确重合；不重复造独立算法行。

### L15 · CoMIR: Contrastive Multimodal Image Representation for Registration, NeurIPS 2020
- 来源：https://arxiv.org/abs/2006.06325
- 类型：paper；核验：本轮arXiv官方摘要
- 范围：InfoNCE、两模态稠密表示、RGB-NIR及BF-SHG。

### L16 · CoMIR 官方代码
- 来源：https://github.com/MIDA-group/CoMIR
- 类型：code；核验：前轮已读取；本轮不宣称重新核对运行代码
- 范围：原生配准流程和特征+共享优化器两表分开。

### L17 · MIND: Modality Independent Neighbourhood Descriptor, Medical Image Analysis 2012
- 来源：https://doi.org/10.1016/j.media.2012.05.008
- 类型：paper；核验：作者机构库+PubMed元数据
- 范围：基于局部自相似的模态独立描述子；2D适配须记录。

### L18 · Information Theoretic Shape Matching, TPAMI 2014
- 来源：https://pubmed.ncbi.nlm.nih.gov/26353150/
- 类型：paper；核验：本轮PubMed摘要
- 范围：包含CS点集分布匹配；不能冒称现成的稠密BF-SHG算法。

### L19 · GRES / gRefCOCO, CVPR 2023
- 来源：https://openaccess.thecvf.com/content/CVPR2023/html/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.html
- 类型：paper；核验：上轮分析所用原文；本轮仅保留备选入口
- 范围：无目标与多目标；第三方向不自动训练。

### L20 · SGRAF: Similarity Reasoning and Filtration for Image-Text Matching, AAAI 2021
- 来源：https://github.com/Paranioar/SGRAF
- 类型：code；核验：NCR/L2RM官方README共同上游
- 范围：服务器必须锁SGR单模型结构，不能把SGR与SGRAF双模型分数混同。

### L21 · CS-Aligner: distribution-level vision-language alignment
- 来源：https://arxiv.org/abs/2502.17028
- 类型：paper；核验：前轮资料来源；本轮未重读全文
- 范围：相关工作/条件适配，不是现成噪声后验评分器。
