# 02｜论文与方法合同

来源：V6 正文 §1–5、补充 A–F、中文定位说明 §1–4。`[PROPOSED]` 表示本包新增的实施组织，不是 V6 已完成结果。

## 1. 论文对象与职责 [V6]

论文题目：**Variational Cauchy–Schwarz Quadratic Mutual Information for Visual Representation Learning**。

主对象是 mixture-reference CS estimator；不是 Shannon MI estimator，不是新的三角散度，不是首次二次判别损失。RPC 的平衡特例与原 J 相同；equal-mixture relative LS 可经变量平移与 J 对应。不得删掉这些关系以制造新颖性。

| 模块 | 证据要说明什么 | 不能拿来替代什么 |
|---|---|---|
| 固定分布 estimator | 在明确目标上测量误差、样本与计算行为 | 下游 accuracy |
| 端到端 SSL | 目标及评分几何能否训练有用表示 | 固定分布测量精度 |
| 固定模型的测量器检查 | 同一模型上比较不同 estimator 的使用表现 | 不同 encoder 的性质比较 |
| 固定测量流程的 SSL 表示分析 | 解释最终 VCS 与外部 SSL 模型保留的指定关系及预测响应 | 因果 attribution 或“表示普遍更可解释” |
| 压缩测量 | 实际确定性观测映射下依赖变化如何估计 | 不同模型间随意相减或 probe accuracy 当信息量 |

## 2. 核心数学保持不变 [V6]

对同一空间上的概率测度 P、Q，定义

\[
M=(P+Q)/2,\qquad \eta=(dP/dM-dQ/dM)/2,\qquad S=\mathbb E_M\eta^2.
\]

\[
J(T)=\mathbb E_P[T-T^2/2]+\mathbb E_Q[-T-T^2/2],\qquad |T|\le1.
\]

\[
S=\sup_{|T|\le1}J(T),\qquad S-J(T)=\mathbb E_M(T-\eta)^2,
\qquad D_{CS}^{(M)}=2\log\frac{1+S}{1-S}.
\]

P 为 joint、Q 为 product 时，`T*=tanh(PMI/2)`。训练／估计主读数是原始 J，不把 log transform 放回训练去改变目标。有限样本 J 可为负；不因其目标 S 非负就裁剪主结果。S 接近 1 时变换敏感，不声称变换后的量也有均匀低方差。

合作者原构造是 Theorem 1 的依据。V6 Proposition 2、3 是在此基础上的后续推导；保留其独立评价、确定性观测关系与同一底层实验等条件，不冒称 SSL 收敛定理。

### 独立评价的边界

当 fitted critic 与两组互相独立的 EVAL 池独立时，令 `eps_T=E_M(T-eta)^2`：

\[
\mathbb E[(\widehat J-S)^2\mid T]
=\varepsilon_T^2+\operatorname{Var}_P(a_+(T))/n+\operatorname{Var}_Q(a_-(T))/m.
\]

不能把共享 image 的全配对数填成这里的 n、m。训练态 BatchNorm 和共享负对都需要独立说明；本命题不直接给出它们的独立样本误差。

### 压缩的边界

在同一个 balanced experiment 中，A=g(B) 才有

\[
S_B-S_A=\mathbb E(\eta_B-\eta_A)^2,
\quad \left|\sqrt{D_T}-\sqrt{S_B-S_A}\right|
\le\sqrt{\varepsilon_B}+\sqrt{\varepsilon_A}.
\]

若有限 critic 分别拟合，则 `J_B-J_A=(S_B-S_A)-eps_B+eps_A`。不要求观测到的差为正；无 oracle 的图像数据不填未知 eps，不把 D_T 自动称为真实 decrement。

## 3. 当前完整 SSL 实现 [V6]

对 B 张图像，各抽取 V 个独立同分布增强。共享 encoder、projector：

\[
h=e_\theta(t(x)),\quad r=g_\psi(h),\quad z=r/\max(\|r\|_2,10^{-8}),
\quad T_{a,\kappa}=\tanh[a(z^\top z'-\kappa)].
\]

当前主配置是 `(a,kappa)=(2,0.5)`，即 `tanh(2s-1)`。a、kappa 在一次训练中固定；数据集间的选择按原规则披露。自由学习 a、kappa 是内部消融，不是主方法。

正对：同一 image 的不同 view 的**有向**配对。负对：不同 image、任意 view（包括相同 view 索引）的有向配对。排除负对中所有同 image 组合，而不仅排除矩阵对角线。

\[
|\mathcal P|=VB(V-1),\quad |\mathcal Q|=V^2B(B-1),
\]

\[
\mathcal L=\frac1{2|\mathcal P|}\sum_{p}(1-T_p)^2
+\frac1{2|\mathcal Q|}\sum_q(1+T_q)^2=1-\widehat J.
\]

正负两项分别 mean；两个端点完整求导；每 step 一次共享 forward／optimizer update。没有 negative-side stop-gradient，没有 momentum teacher，没有 queue。blockwise matrix 是内存实现，不改变 pair 集合、分母或梯度。

### 匹配 logistic 不混淆 logit 约定

SSL 写 `f=a(s-kappa)`，使用

\[
L_{JS}=\operatorname{mean}_P\mathrm{softplus}(-2f)
+\operatorname{mean}_Q\mathrm{softplus}(2f),\quad T=\tanh f.
\]

测量端旧 P116 可能将 logistic logit 记作 `u=2f`，用 softplus(±u) 和 `T=tanh(u/2)`。这是等价记号，不要因变量名相同而丢掉因子 2。共享 scorer 比较与各自选 scorer 比较必须是两张协议／两组明确标注的结果。

### CIFAR 已报配置，不推断缺失值

V6 supplement E 指定 CIFAR ResNet-18（3×3 stride1，无初始 maxpool）、projector 512→512→128，中间 BN/ReLU，无末层 BN；FP32；AdamW，矩阵 decay 1e-4，bias/normalization 无 decay，betas=(.9,.999)，eps=1e-8；10 epoch warmup，cosine 至峰值 1%。已报主训练为四视图、800 epochs、45k/5k validation 协议。

标准增强：crop [0.2,1]，ratio [.75,4/3]，flip .5，jitter 概率 .8／强度 .4,.4,.4,.1，grayscale .2，mean/std 均 .5。强增强将 crop 下限改 .08、jitter 改 .8,.8,.8,.2。具体 train batch、SimCLR temperature、drop_last 等以原配置为准；未在 V6 精确给出的字段不能靠此摘要补造。

SimCLR 四视图参考是六个无序 view-pair 的 two-view NT-Xent 平均，不是这里的 all-token balanced logistic。预算报告区分 image presentations、encoded views、optimizer steps 和总 pair evaluations。

标准 raw-h linear probe：100 epochs SGD、lr .1、momentum .9、batch256、无 weight decay、cosine 至峰值 .1%，取 final epoch；kNN 用归一化 h、k200、temperature .1、45k bank。P141 标准化分层 probe 是另一个 readout，不替换主表。

## 4. 测量端不是训练 scorer 的强制复用 [V6]

固定 encoder 和任务头；另拟合 measuring critic。VCS 既可测 VCS encoder，也可测外部 SimCLR。属性 N、类别 Y、表示 H 定义

\[
P_c=P_YP_{H,N|Y},\qquad Q_c=P_YP_{H|Y}P_{N|Y}.
\]

这是论文定义的条件 joint-vs-product discrepancy；不将其直接等同于另一篇 conditional-CS 的不同定义。真实数据报告估计 J、检测、成本；只有有 oracle 的受控实验报告 posterior MSE。

P116 的 ridge→VAL calibration→tanh 是原二次估计器的一种有限函数类实现；不是原始 bounded nonlinear optimization 的闭式精确解。必须比较短预算／收敛充分的 JS，不能只选 800-closure JS 来宣称 30× 速度优势。

### 两种比较轴 [V6-R]

Table 3 固定被测 encoder，改变测量器。Table 4 固定测量协议，改变被测 encoder；后者正式包含最终 VCS 与 SimCLR，不再仅作为可选外部案例。固定协议指同一算法、输入站点定义、选择预算和统计规则，不是固定一份 critic 权重用于所有 encoder。

两个表都在模型冻结后运行，但各自支持不同结论。Table 4 不使用内部版本的差异证明 estimator 优越，也不以属性更少/更多作为普遍胜负标准。新增的附录正交不变性说明不改变原目标或训练代码。

## 5. 本包不启动的新主线 [V6]

空间随机掩膜、RISE/RELAX/VOC、额外响应噪声、生成式反事实、依赖分数辅助选模均不属于这轮必须完成任务。已有内部配方、门控诊断、逐层 probe 和负结果可整理到 supplement，但不据此自动启动新 sweep。新的研究方向应单独说明，不通过“补一张图”混入本包。
