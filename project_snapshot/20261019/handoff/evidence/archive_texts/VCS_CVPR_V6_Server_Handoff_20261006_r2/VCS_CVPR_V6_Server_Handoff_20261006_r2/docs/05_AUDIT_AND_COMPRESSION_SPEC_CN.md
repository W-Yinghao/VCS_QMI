# 05｜冻结模型条件审计与真实压缩：实施规范

**定位：用原 estimator 测量 SSL 表示保留了什么。先检查测量程序，再解释最终 SSL 模型；不是独立审计平台，也不宣称内生可解释性。**

**r2 新分工：**Table 3 固定 encoder、变 estimator；Table 4 固定测量流程、变 encoder。数学对象和原实现规则不变。  
`[V6]` 为稿件与补充材料已有规定；`[SNAPSHOT]` 为同一提交的 P116/P118 实施信息；`[PROPOSED]` 为完成缺口所需的新协议建议。已有 frozen protocol 不被新建议追溯改写。

## A. fixture 是最小比较单位

### A1. 必须区分的两个旧 fixture [V6 / SNAPSHOT]

**F-SOLVER-P116**：同一外部 SimCLR的固定 feature fixture，用于同函数类solver对照。P116每重复重新抽 N|Y，FIT/EVAL/POOL 分离；FIT再共享80/20内部split。旧代码写 `phi=[z(2n-1),1]`，这里 z 是原fixture的输入变量名；不能只凭符号就认定是论文 normalized projector z，必须核对feature manifest与实际提取位置。

**F-PAIRED-P109/P118**：20k base images，FIT10k / VAL2k / EVAL8k；同一图像 clean/modified 都留在同一split。clean-h classifier和成对响应，以及后续P118的多seed/嵌套实验使用此谱系。它与P116不是同一个实验池。

两者分别标注 `fixture_id`、`encoder_checkpoint_hash`、`representation_site`、`base_id_hash`、`split_hash`、`transform_hash`、`assignment_hash`、`head_hash`。若原记录缺这些字段先恢复来源，不能猜。

### A2. 修订稿已对齐的表示分析 [V6-R]

Table 4 属于 `F-PAIRED-P109/P118` 谱系，不是新造的实验池。已有最终VCS/SimCLR、colour.1、seeds1/2读数先复用；局部来源见修订稿 `evidence/retention_source_rows.json`。

一行内的检测和响应应对应同一encoder seed、属性操作/强度及head谱系。检验 n=2000、100重复，响应8000对图像；只要求它们的底层协议对齐，不谎称每次检测都用了全部8000张，也不要求不同encoder权重hash相等。

记录 `comparison_axis=fixed_measurement_vary_encoder`、`measurement_protocol_id`、`matched_condition_id`、`paired_response_fixture_id`。固定的是测量程序和预算，不是同一份critic权重。颜色null H/O范围来自已报来源，不能仅复制汇总范围当作新模型的null结果。

### A3. 补缺时的新联合 fixture [PROPOSED]

已有 Table 3 与 Table 4 保持两套实验，不要求为统一它们而重跑。为后续缺失行共享底层实验，可在P109/P118协议上增加缺失的测量算法；或者在原P116 fixture可完整恢复成对输入时补齐响应。**选择哪条路径在跑新EVAL前固定**，以数据与缓存可复用性为依据，不按结果好坏选择。

确有新定义的fixture才分配新ID，不继承旧拒绝率或response数值；纯表格重排沿用旧fixture与来源。优先复用已有encoder与clean/modified feature caches，不重新预训练。审计数据是否曾用于SSL预训练要单独记录；“测量器held-out”不自动意味着“从未被encoder训练见过”。

## B. 概率对象与拆分 [V6]

对属性 N、类别 Y、固定表示 H：

\[
P_c=P_YP_{H,N|Y},\qquad Q_c=P_YP_{H|Y}P_{N|Y}.
\]

balanced pair label C 与下游 class Y 是两种不同标签。N是受控属性变量，不是样本量；记录中样本量一律用 `n_base_images` / `n_eval`。

P109的binary assignment沿用 `pi_y=P(N=1|Y=y)=0.5±0.3`、class parity的既定规则。exact-Q时，固定(H,Y)并对属性n枚举：

\[
\widehat{\mathbb E}_{Q_c}a_-(T)
=\frac1m\sum_i\{\pi_{y_i}a_-(T(H_i,1,Y_i))
+(1-\pi_{y_i})a_-(T(H_i,0,Y_i))\}.
\]

枚举改变的是**critic收到的属性坐标**；不是再按新n重选对应的modified H。如果这样重选，会重新建立本应打断的关系。sampled-Q沿用同类独立池；抽样负例不能跨类混成无条件product。

所有版本按base image分组拆分。FIT用于critic或classifer拟合，VAL用于超参/停止，EVAL仅用于最终读数。PROBE/PCA等额外角色只有在原协议已定义或新协议显式划分后才存在，不能从EVAL学习表示映射。共享preprocessing、标准化、PCA和任务头均记录拟合角色和样本ID。

## C. 两种比较轴和各自终点 [V6-R]

- `fixed_encoder_vary_estimator`：同一encoder/head/数据，比较VCS、JS、same-target kernel、条件HSIC；解释的是测量器性能。
- `fixed_measurement_vary_encoder`：同一测量协议、预定属性条件和读出位置，比较最终VCS与外部SSL encoder；解释的是模型性质。若以JS复核这一轴，另出一张同流程表，不能逐encoder挑效果最有利的测量器。

`encoder_method`、`encoder_paper_name` 与 `estimator_method` 必须分列。`encoder_paper_name=VCS` 不决定 `estimator_method=VCS`；`estimator_method=VCS` 也不意味着被测encoder是VCS。内部run_id用于追溯，主表只有一个最终VCS名称。

### 测量器实现与读数

| 算法 | 实施要求 | 允许比较的量 |
|---|---|---|
| VCS | 原J；主solver沿用 `ridge_tanh_calibrated`，或该fixture已冻结的neural critic | 同目标J；known-truth才有posterior MSE；检测；成本 |
| 匹配JS | 同输入、同函数类、共同split；原posterior坐标转换 | 映射后的共同J与native JS统计分别记录；检测；成本 |
| Same-target kernel | 复用同S目标的RFF/Nyström/RLS或S-KDE已有实现；先冻结具体实现和带宽选择 | 同目标J/known-truth误差；检测；成本 |
| 条件HSIC | 复用已验证的class-conditional统计与带宽选择；保留native尺度 | 水平、拒绝率/检测、成本；不与S做原始误差比较 |

### C1. 同函数类solver对照 [SNAPSHOT]

P116原ridge正则网格为 `{1e-3,1e-2,1e-1,1}*sc`，`sc=mean diag(second-moment matrix)`；tanh calibration c 为25点 `logspace(-1,1.5)`。A1计入4个linear solves与100次VAL评价。原JS按同一split的L-BFGS与lambda选择执行，预算50/200/800 closures。主表不能只有长预算800版本，至少保留短预算50以及实际收敛记录。

VCS ridge步骤解的是raw linear J的penalized问题，之后tanh和scale选择才得到bounded输出；没有“整个非线性bounded J闭式解”。JS若用u作为原logit，common T为tanh(u/2)，不是tanh(u)。必须把这个转换写入metadata与单测。

保留原phi函数类，不能为一个算法单独增加Y-aware交互、hidden layer、不同normalization或更多validation访问。若增加一个Y-aware／非线性共同函数类，应作为另一个预注册比较块，原P116数字不变。

### C2. 核与HSIC补行 [PROPOSED]

首先在仓库已有方法中选出可复现的same-target kernel与条件HSIC入口，并记录native statistic、feature map、bandwidth grid、fitting role、复杂度及资源。不得把classical fixed-reference CS native值直接放进S的MSE列。

若现有条件HSIC接口或null校准不可恢复，报告缺失并先实现/验证该统计；不能用无条件HSIC冒名填上。V6未指定新核网格、新HSIC参数或新增encoder，因此这些字段在protocol冻结前必须由库存与实现核对确定，而不是本包声称已定。

## D. 属性操作和null

### D1. 不从论文简写重造图像变换

[V6] 已报条件包括colour.2、blur.5，后续visual confirmation包括colour.1、blur.25。这里的数值单位按原renderer/transform代码解释。**colour shift不等于训练用color jitter；blur强度不自动等于某库的kernel size。** 接入时保存操作顺序、输入范围、归一化前后的位置、kernel边界规则、随机种子与版本。

[PROPOSED] 首批补齐旧P116三格的外部方法，再在联合fixture复用已有的弱／中等条件。不要按EVAL的power动态挑“最能显出VCS优势”的强度。大blur损伤class content时报告为content-damaging intervention，不自动叫shortcut。

### D2. null机制需要实际代码支持

[SNAPSHOT] 原P118包含 `null_label_only`、`null_all_planted`；原P116有相应null检查。其机制必须从生成代码、参数、feature cache确认；名字本身不够。新dataset若没有相同机制，先写明对应关系和 `H ⟂ N | Y` 为什么成立。不能随手置零输入得到一个容易通过但不同于原假设的null。

[PROPOSED] 预定义null、alternative和方法／layer family后再看EVAL。原200重复null的旧flag规则按原协议保留；新条件的重复数／校准规则单独冻结，不能看到flag后改阈值。

## E. 检验与不确定性

[V6 / SNAPSHOT] 同一重复对所有方法共享FIT/VAL/EVAL、N draw和within-class permutations。原T1/P118采用200 permutations、nominal alpha=.05；已报power100重复、null200重复。新增正式预算可沿用，但必须披露有限重复精度。

对于fitted critic与独立EVAL，固定H、Y和critic，置换同类内的N；不在每次置换后重新从(H_clean,H_modified)选择H。对第b次permutation记录同一统计量的null值。实现常用Monte Carlo p值为

\[
p=(1+\#\{b:S_b\ge S_{obs}\})/(B_{perm}+1).
\]

若现有协议使用不同可复核的精确规则，先复现它并标明差异，不追溯改变历史读数。若算法在EVAL上拟合／选择，不能再当fixed-statistic置换处理；必须拆分或在相应null流程中重复完整选择。本包默认前者。

[V6] 同一方法的预定义layer family共享permutations，用max-T处理any-layer。原family为pooled layer3、h、z、logits；不能跑完再挑layer。跨方法择优是另一个选择问题，不由layer max-T自动解决。

[PROPOSED] 每种方法分开报告native statistic与common J的检测结果，不能混合挑最大power。报告逐重复reject/p值和拒绝率的区间；方法差异使用共享重复的paired差异。由不同encoder seed得到的结果先逐seed报告，不能把100抽样重复当成100个独立训练seed。

null calibration不合格的格子保留原数值和flag；其alternative可以显示但不能据此支持“更强的有效检验”。总体均值不得藏掉该问题。未检出写成“在该n、critic类和预算下未检出”。

## F. 配对预测响应 [V6]

冻结同一个encoder和clean-FIT训练的task head。对相同base image的clean/modified版本计算

\[
\Delta Acc=\operatorname{mean}(1[\hat y_1=y]-1[\hat y_0=y]),
\quad \Delta p_y=\operatorname{mean}(p_1(y)-p_0(y)),
\quad Flip=\operatorname{mean}1[\hat y_1\ne\hat y_0].
\]

报告margin变化及原协议中已定义的其他响应指标。accuracy差用percentage points，probability差保留0–1单位，flip注明百分数。按base image paired bootstrap得到不确定性；重采样时clean/modified不能分开。class distribution或weights改变则另立fixture。

把检测与响应并排展示，但不把S或J当成响应大小。大信息相关性和小概率改变可并存；它们不是互相矛盾的实现bug。对不同fixture仅作各自例子描述，不计算跨fixture的相关系数或组合效应。

## G. 真实视觉压缩

### G1. 验证实际确定性映射 [V6]

从同一个realized input得到 `h→r→z`，其中r是实际projector输出；或 `h→logits` 是固定task head。冻结BatchNorm并设eval模式；关闭不受控dropout；若图像操作本身随机，固定同一次实现再提取所有sites。图像pipeline、映射参数及normalization保持一致。

source=pooled layer3、target=h通常不能直接宣称nested，因为池化可丢弃后续网络所需空间结构。不同encoder之间也不是压缩链。新h→r映射需要代码重算r与缓存r的误差记录（含dtype与预设容差），不能仅按层名判断。

条件实验的完整观测是 `(Y,H,N)`，压缩后为 `(Y,g(H),N)`；Y、N和同一个P/Q实验一起保留。不能在两端重抽不同dataset、不同N|Y或不同class prior后仍调用Proposition3。

### G2. 拟合与读数

[SNAPSHOT] 复用 `src/vcs_measure/nested.py` 与P118 visual实现；coarse critic先拟合，rich critic包含lift后的coarse预测作为候选／初始化，再拟合residual。保留independent对照、exact和sampled Q的清楚标识；主比较选一个固定readout，不按结果切换。

每个映射、条件、seed和重复记录：

\[
\widehat J_B,\ \widehat J_A,\ \widehat\Delta_J=\widehat J_B-\widehat J_A,
\quad \widehat D_T=\widehat{\mathbb E}_M(T_B-T_A)^2,
\quad \widehat\Delta_J-\widehat D_T.
\]

D_T作为平方均值天然非负；Delta_J保留符号。使用共同base-image EVAL和pair构造作paired uncertainty，记录重复fit spread、VAL/EVAL差和optimizer状态。known-truth链另报posterior MSE与实际decrement error；图像端这些oracle字段必须null。

不能把 `Delta_J-D_T≈0`、拟合中forced orthogonality、或差值望远镜相消当成估计准确的证据。恒等相消可作代码检查，但不是独立验证。constant-zero控制在真值实验中有用途：残差零同时真实正decrement估计失败，需保留。

[PROPOSED] 图像端先对一个已注册条件作有限的fit-budget／独立拟合敏感性检查；不追加任意大critic直到排序符合预期。若signed Delta和D_T差异明显，报告estimation-sensitive，保留较小可靠主张；不要通过clipping、isotonic修正或额外response noise造出单调链。

## H. 成本与必交产物

fit时间包含lambda/scale selection、校准和continuation；另列permutation、feature extraction、head fit、peak memory、CPU threads、GPU型号与实际时钟计时范围。总耗时与可摊销的一次性缓存成本分开。长solver与短solver必须对称可见；closed-form内部solve次数与iterative closure次数不是同一种预算单位。

每个结果格必须关联fixture manifest、method/statistic定义、config hash、split/seed、status、null flag、选择规则、时间计量范围和逐重复文件。产物格式见 `templates/RESULT_RECORD_TEMPLATE.json`。本包不提供未经仓库核对的GPU runner或batch命令。


## I. r2 表格接纳条件 [PROPOSED：实现修订稿所需记录]

Table 4 不要求所有行的效果同方向；要求每行的检测与响应对齐、样本量和单位清楚、null依据可查。原四行先核对并复用，后来增加模型或dataset的行单独记录protocol版本。

`retention_source_rows.json` 是已报数值来源，不是新运行的目标或验收分数。现场若发现raw结果不同，记录冲突，不改数据凑原数值。未完成的VICReg/logistic/C100行留在计划表，不进入论文结果表。

同一模型跨层非拒绝不等于信息不存在；不以较低属性依赖认定表示更好。已知压缩真值检查继续在estimator实验中，不因本次章节调整改变测试或理论。
