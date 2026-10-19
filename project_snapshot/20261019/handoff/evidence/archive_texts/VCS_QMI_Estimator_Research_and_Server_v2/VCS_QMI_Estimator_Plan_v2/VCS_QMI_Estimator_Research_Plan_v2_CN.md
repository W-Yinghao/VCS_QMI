---
title: "VCS-QMI：估计器驱动的视觉表征学习"
subtitle: "研究计划 v2｜CS 估计路线比较、完整 SSL 与二次结构分析"
author: "研究讨论文档"
date: "2026 年 9 月 28 日"
---

# 1. 论文要解决的问题

**本文的贡献主体是合作者的变分 CS-QMI 估计器。完整视觉 SSL 是主要应用，实验首先解释它相对已有 CS 估计路线改善了什么，再考察这些特点如何进入视觉学习。**

主线是：

> 混合参考测度与二次变分表示 → 有限样本下的估计质量和计算特性 → 完整视觉表征学习 → 同一估计器的依赖分析。

三项主要研究问题如下。

| 问题 | 直接比较 | 主要结果 |
|---|---|---|
| 估计质量 | 神经 VCS 与同目标的核估计、相对密度比回归 | 误差随独立样本量、维度和依赖强度的变化 |
| 学习价值 | 相同视觉框架中的神经／核 CS 估计路线 | 完整预训练后的冻结表示质量、时间与显存 |
| 二次结构 | 原始回归目标的求解与误差关系 | 可计算的组合优化、可分析的依赖增量 |

SimCLR、VICReg 提供视觉表征水平的参照；MINE、NWJ、InfoNCE、SMILE、JS 提供神经依赖学习的参照。它们不替代直接的 CS 对照。残差组合和观测噪声属于可选的估计器分析与改进，不作为完成 SSL 的前提。图文模型只用于读取已完成实验的诊断，不新增 CLIP 项目。

**本版依据。** 合作者原计划 [P1]、研究／服务器文档 v1 [V1]、用户补充 [S1] 是主依据。补充记录的是 `af7172c` 时点；本轮另读取了 `d3b4d5f`（2026-09-28 16:14:46 UTC）中的 P84 报告 [R84]。下文把这项新增仓库证据单独列出，不把它写成补充原文的结论。本文没有启动新训练。

# 2. 补充意见如何改变研究安排

## 2.1 P72 使“配对噪声优势”退出主假设

补充记录：P72 在冻结图文双塔加 adapter 的主题配对设置中，用原生分数识别注入的错配。平衡 JS 在四个单元中均最高。[S1, §1.1]

| 偏移 | 错配比例 | VCS AUROC | InfoNCE | Logistic | 平衡 JS |
|---|---:|---:|---:|---:|---:|
| animal | 0.2 | 0.880 | 0.885 | 0.879 | 0.896 |
| animal | 0.4 | 0.856 | 0.862 | 0.843 | 0.887 |
| indoor→outdoor | 0.2 | 0.822 | 0.816 | 0.812 | 0.842 |
| indoor→outdoor | 0.4 | 0.790 | 0.797 | 0.756 | 0.833 |

这组结果否定了该设置下的 VCS 错配识别优势。R2 的主题配对训练与自身 caption 检索并不一致，因此其检索曲线不用于建立噪声鲁棒性结论。两项结论都进入本文的实验依据，不再把“VCS 更耐错配”作为残差或噪声方案的前提。

值得保留的工作是读取现有 P72 adapter，解释为什么 VCS 的表示漂移较大而 JS 变化较小。该诊断可以复用现有模型和数据，不需要重训图文系统。

## 2.2 完整 SSL 的直接证据仍需补齐

已记录的官方 CIFAR-10 测试中，四视图、800 epochs 的 VCS 为 **86.65±0.26%**，调优 SimCLR 为 **88.13±0.07%**。VCS 强增强单种子为 **88.26%**。补充指出，P41 的 SimCLR 尚未获得同一强增强设置。[V1, §2; S1, §§1.4, 3.6]

因此，下一轮最有信息量的 SSL 新实验是**同一协议中的“方法×增强强度”对照，以及神经／核 CS 路线的完整训练比较**。继续延长单个 VCS 配置的训练，并不能回答这两个问题。

## 2.3 新增仓库核对：P84 已完成部分 v1 诊断

P84 在十个冻结 checkpoint 上评估了重拟合与组合；评估单位为独立的图像配对块。[R84]

| VCS 四视图 checkpoint | 训练 critic 的 J | 收敛诊断的同类 cosine J | 已测试组合 J | 已测试残差 J |
|---|---:|---:|---:|---:|
| epoch 100 | 0.948 | 0.973 | 0.969 | 0.970 |
| epoch 400 | 0.981 | 0.991 | 0.982 | 0.983 |
| epoch 800 | 0.986 | 0.993 | 0.986 | 0.987 |

同类 critic 的重拟合空间约为 0.025、0.010、0.007；已有字典组合未显示可测收益，残差增益约 0.001–0.002，JS 重拟合相近或略高。收敛的 cosine 只作为诊断，没有进入当时的字典。

本版据此调整：不重复 P84，不把组合写成已经有效的新方法。后续最多先补“收敛成员＋现有成员”的低成本比较，再根据分歧和独立误差决定是否扩展。较高的冻结 J 也不直接成为在线更新 critic 的理由：P84 的陡峭 critic 可能进一步压缩学习信号。

# 3. 保持合作者的估计器定义

在平衡的联合／乘积二元实验中，

$$
P=P_{XY},\quad Q=P_XP_Y,\quad M=\tfrac12(P+Q),
$$

$$
\eta=\frac{p-q}{p+q},\quad S=\mathbb E_M\eta^2,\quad T=\tanh f.
$$

原始目标是

$$
J(T)=\mathbb E_PT-\mathbb E_QT
-\tfrac12\mathbb E_PT^2-\tfrac12\mathbb E_QT^2.
$$

若正负类别分别记为 $C=+1,-1$，则

$$
J(T)=1-\mathbb E(C-T)^2,\qquad
S-J(T)=\mathbb E_M(T-\eta)^2.
$$

CS 解释为

$$
D_{\rm CS}^{(M)}=2\log\frac{1+S}{1-S}.
$$

训练与主要数值分析使用原始二次目标，不以对数变换替代。[P1, §§3–7, 9–10]

**同目标与跨目标比较分开。** 经典固定参考测度 CS 和这里的混合参考 CS 不等于同一个量。前者作为现有 CS 学习方法参加应用比较；后者与同目标的核／回归估计比较数值误差。参考测度的定义一次说明清楚，后续结果按这两种比较组织。

# 4. 直接的 CS 估计路线比较

## 4.1 三类核心参照

| 参照 | 目标与作用 | 本文采用的比较方式 |
|---|---|---|
| 经典 KDE／核统计 CS-QMI | 固定参考测度下的 CS，沿用已有信息势计算 | 原生目标与原生梯度；完整 SSL 对照 |
| 同目标核 plug-in | 密度模型给出 $\widehat\eta=(\widehat p-\widehat q)/(\widehat p+\widehat q)$ | 与神经 VCS 比较相同 S 的误差 |
| 同目标核回归／RuLSIF 相关形式 | 在核特征类中拟合配对后验或相对密度比 | 分离函数类、求解和神经参数化的作用 |

经典核 CS-QMI 的方法背景引用 CS-IB [L1]；它的回归任务结构不整体移植到 SSL。相对密度比参照引用 RuLSIF [L2]。不能给自己增加的核 plug-in 形式套用已有论文名称。

同目标的核 plug-in 在独立 query 上同时报告两种读数：

$$
\widehat S_{\rm plug}=\mathbb E_{M_{\rm eval}}\widehat\eta^2,
\qquad
\widehat J_{\rm kernel}=\widehat J_{\rm eval}(\widehat\eta).
$$

前者是 plug-in 读数；后者把所有后验估计放进同一个二次风险评价。二者在有限样本与有限拟合下不相等。

对 RuLSIF，取 $g=1+T$，有

$$
\tfrac12\mathbb E_Mg^2-\mathbb E_Pg
=-\tfrac12-\tfrac12J(T).
$$

这一恒等式用于数值一致性核对。必须匹配函数类、常数特征和正则；raw ridge 与其 tanh 包裹形式分开报告。tanh 后的优化不能称为闭式 ridge 解。

## 4.2 神经参照保持完整

继续 P69 已规划的 MINE/DV、NWJ、InfoNCE、SMILE、JS 和 VCS，不删掉 MINE 行。[S1, §3.7] 参数化公平性有两层：共有相近的隐藏特征能力，同时保留各方法定义要求的输出变换。

RPC 的平衡参数特例与当前二次 J 相同，作为目标／实现一致性参照保留；调参后的 RPC 用于完整方法比较 [L13]。不把同一个目标的两种记法作为应当产生系统差异的算法。

平衡 JS 与 VCS 都可表示同一后验

$$
q=\sigma(2f)=\frac{1+\tanh f}{2}.
$$

因此，绝对后验解释属于平衡正规评分规则的共同结构。VCS 的重点应放在二次风险、估计误差和求解特性，而非“只有 VCS 能产生有意义的概率”。[S1, §1.1; L3]

# 5. 估计器实验：精度、样本、维度和成本

## 5.1 优先复用 P69 的生成器和 oracle

沿用相关 Gaussian、可逆非线性变换与非高斯混合，增加后验 MSE、梯度误差和实际门控。P69 的正式运行、P81/P82 等后续探针及 P83/P84 的文件由服务器先核对；已有数据与已完成结果直接复用，不再建立平行生成器。[S1, §§1.3, 5; R84]

在原实验上新增一个清楚的维度压力测试：

$$
X=(X_s,N_x),\quad Y=(Y_s,N_y),
$$

其中 $N_x,N_y$ 相互独立且与信号独立。联合与乘积密度中的噪声因子抵消，故

$$
S(X;Y)=S(X_s;Y_s).
$$

真实依赖不变，观测维度增加。这个设计直接检验估计器能否处理无关维度，不混入依赖强度的变化。它是本版增加的受控实验设计，而非补充文件中已经执行的实验。

## 5.2 独立样本数与 batch 分开

固定独立数据池而改变 batch，测量优化组织；改变数据池大小而固定 batch，测量样本效率。新鲜数据流上的阶梯实验属于追踪能力，不等价于固定有限样本估计。

每种方法都得到合理的选择预算。核方法选择带宽／ridge，神经方法选择学习率／早停；同时报告相同预算和各自合理配置下的表现。

## 5.3 主要指标

| 评价层 | 指标 |
|---|---|
| 同目标精度 | S 的偏差、RMSE、独立重复的标准差 |
| 后验拟合 | $\mathbb E_M(T-\eta)^2$，VCS 和匹配 JS 直接可比 |
| 强依赖 | 后验误差除以 $1-S$；只在分母可分辨时报告 |
| 跨目标分辨 | 相邻依赖条件的排序概率／分辨 AUC，连同原生目标值 |
| 梯度 | 固定 critic 时，对可知真值通道参数的导数误差 |
| 成本 | 拟合与评价时间、峰值内存、独立样本数与实际 pair 数 |

$J$ 的无量纲范围为 $[-3,1]$；不使用“5 nats 跳变”衡量 VCS 稳定性。InfoNCE 的有限候选目标与真实 Shannon MI 分开报告；MINE 的加性 gauge 未处理前，不把其 logit 直接变成可比后验。闭式和神经方法都要计入选择与拟合成本。

Gaussian 上的坐标立方变换保留 mixture-S 的目标；经典 L²-CS 的参考密度可能不再平方可积，不能直接搬用 Gaussian 的经典 CS 真值。经典 CS 数值误差先在满足平方可积条件的设置中评价。

# 6. 完整视觉 SSL：必需的主应用

**完整 SSL 不等待残差、观测噪声或解释性方案成功。** 当前 VCS 配方继续作为主方法，直接 CS 对照补入完整训练；P75 标准框架移植按已有批准和实际状态接续。

## 6.1 两张表回答两种问题

**表 A：同目标估计方式。** 神经 VCS 与核特征–tanh VCS 使用同一个 $J$、相同配对、相同编码器输入和梯度路径。该表主要检验函数近似与估计方式。

**表 B：完整 CS 学习方法。** 经典核 CS-QMI 使用其原生可微核统计目标；神经 VCS 使用当前学习方案。编码器、增强、数据、评估和预算对应，方法内部必要设计明确列出。这个比较包括目标定义与计算方式，不单独归因为 estimator 替换。

SimCLR／VICReg 给出视觉参照；JS 是机制对照。已有结果可在协议一致时复用。最终采用完整预训练、多种子、冻结 backbone 线性评估，并加入此前未用于该轮开发的视觉数据集验证。

## 6.2 最有优先级的新训练：方法×增强强度

沿用补充 S4，比较 VCS、SimCLR 各自的标准增强与强增强，三种子，四视图与 800-epoch 的既有框架保持一致。[S1, §3.6] 核心量是

$$
\Delta_{\rm int}=
(Acc_{V,s}-Acc_{V,b})-(Acc_{C,s}-Acc_{C,b}).
$$

其中 $s,b$ 分别为强、标准增强。报告逐种子差值、均值与不确定性，连同 J、实际门控及梯度。若两种方法同样受益，结果归因于共同增强效果；若 VCS 获得额外收益，再检验它是否与二次损失的学习状态一致。

补充用 $1-S$ 将 0.974→0.877 换算成约 4.7 倍门控变化。这是 oracle 层面的机制提示。训练中观察到的是 J，实际平均门控应直接计算 $1-\mathbb E_MT^2$，不能用 $1-J$ 代替。

## 6.3 外部确认与完整性

CIFAR-10 保留完整历史，CIFAR-100 或 ImageNet-100 用作后续独立视觉确认。具体完整日程与规模在读取 P75 实际配置后固定，不为了最优分数重复展开大网格。官方 CIFAR test 已被读取；S2 的既有授权只覆盖相应协议，不能自动扩展成所有新分支的测试访问权限。[S1, §1.3]

# 7. 机制诊断：从已完成模型中回答具体问题

## 7.1 先解释 P72 的 adapter 漂移

在 $m=0,0.4$ 等已有 checkpoint 上，按干净／注入配对拆分分数、残差、$1-T^2$、输入梯度和有效参数更新。先统一后验 logit 尺度，再分别检查选择的学习率、更新次数以及 Adam 预条件后的更新。[S1, §§1.1, 3.5]

原始局部门控关系为

$$
\partial_f\ell_V=(T-C)(1-T^2),\qquad
\partial_f\ell_{\rm JS}=T-C.
$$

这是相同 f、相同 pair 的局部关系。两个训练后的系统处在不同 f、表示和优化器状态，不能仅据这个比值推断谁累计漂移更小。P72 诊断的目的就是确定这些状态差异。

## 7.2 同类负配对诊断

按补充建议，在冻结 checkpoint 的相同负配对上，区分语义同类与异类，测量梯度份额。[S1, §3.4] 标签只用于诊断，不参与负样本选择、损失权重或训练。

同类负配对不必然是 tanh 已饱和的 pair。因此先报告两组的相似度与置信分布，再比较梯度。SimCLR 的负梯度由候选 softmax 权重决定，也有易负例衰减；实际对照应保留其完整分母。主结果是每 anchor 的绝对贡献与归一化份额，而非未经尺度处理的总梯度。

这项分析检验特定学习状态下 VCS 是否减弱语义冲突。它借鉴假负例文献的研究问题 [L8, L9]，但不引入它们的假负例删除或去偏目标。

# 8. 二次结构的两项辅助研究

## 8.1 组合与残差：利用已有求解结构，不另造主方法

固定有界字典，$T_w=\sum_jw_jT_j$，$w$ 在 simplex 中。则

$$
J(T_w)=\sum_jw_jJ(T_j)+
D(w),\quad
D(w)=\mathbb E_M\!\left[\sum_jw_jT_j^2-T_w^2\right].
$$

因此 $J(T_w)-\max_jJ(T_j)\le D(w)$。补充建议先计算分歧再投入更多组合训练，本版采纳。[S1, §2.3]

若要筛查整个固定字典，需要计算 $\max_wD(w)$ 或使用对全部权重有效的上界，不能只拿任意一组权重的 D 当作所有组合的上界。TUNE 上的上界描述该经验分布，不是独立 EVAL 的总体保证。

字典可加入 P48 的 tanh-wrapped ridge 和已验证收敛的 cosine。raw ridge 保留为非有界诊断参照，不直接加入要求有界的组合。P84 已测组合没有增益，先复用其输出，不重跑同一字典。

对残差候选，固定 $T_0,\lambda$，令 $T_\lambda=(1-\lambda)T_0+\lambda U$，有

$$
\min_U\mathbb E(C-T_\lambda)^2
\equiv
\min_U\mathbb E(r-U)^2,
\quad r=\frac{C-(1-\lambda)T_0}{\lambda}.
$$

当 $\lambda=1/2$，$r\in[-3,3]$。它与平方损失 boosting、stacking 相关 [L4–L6]；保留原 J 直接求导即可。可利用的二次结构是固定输出组合的精确 QP 与线段步长，不宣称发明通用集成或 boosting。JS 的概率混合权重也可做凸优化，只是不具有同一个二次求解式。

## 8.2 依赖增量：突出 L² 误差几何

在同一个二元实验内，$A$ 是 $B$ 的确定性压缩，$T_A^*=E[C|A]$、$T_B^*=E[C|B]$。则

$$
\Delta=S_B-S_A=\mathbb E_M(T_B^*-T_A^*)^2.
$$

对有限 critic，令 $D_T=\mathbb E_M(T_B-T_A)^2$、$\mathcal E_j=\mathbb E_M(T_j-T_j^*)^2$，有

$$
\left|\sqrt{D_T}-\sqrt\Delta\right|
\le \sqrt{\mathcal E_B}+\sqrt{\mathcal E_A}.
$$

**值得强调的是平方风险的对称欧氏形式和直接的范数误差传递。** 这属于二次评分规则的性质，并非 VCS 独占。对数损失也有对应增量：$E_M\,KL(\mathrm{Ber}(q_B^*)\|\mathrm{Ber}(q_A^*))$；它不具有这里同一平方范数表达。不能把“KL 不满足三角不等式”扩大成“JS 无法作增量分析或误差分析”。[S1, §2.6; L3, L7]

该分析可用于表示→logits 的 nuisance 依赖保留。沿用已有条件任务，并等待 T1 的公平对照帮助决定拟合配置；检验功效追平本身不决定哪个 critic 更适合幅度估计。

**真实数据的自检必须有信息量。** 任意标量 $\widehat J_l$ 的相邻差求和都等于端点差，这只是代数恒等式。有效的自检是平方增量的可加性与正交性，或者

$$
(J_B-J_A)-D_T
=2\mathbb E_M[(C-T_B)(T_B-T_A)].
$$

该残差在 oracle 处为零。对嵌套的多层观测，还应测量不相邻 critic 差分的交叉内积。统一按基础图像做成对 bootstrap；区间包含零是相容性证据，不是完整正确性证明。各层单独池化后的特征未必嵌套。

# 9. 观测尺度：修正标度，保留为辅助实验

## 9.1 Gaussian 只作精确性检查

单位方差高斯对两端分别加入每坐标标准差 $\sigma$ 的独立噪声后，重新按固定总方差标准化的相关性为

$$
\rho'=\frac{\rho}{1+\sigma^2}.
$$

所以其 $S_\sigma$ 是原高斯族在 $\rho'$ 的值。采纳补充的处理：这用于检查代码与 oracle，不另算一项方法性能实验。[S1, §2.4]

## 9.2 冻结表示使用拟合侧的目标分数标定

对单位向量 $z\in\mathbb R^d$，设

$$
u=z+\frac{\tau}{\sqrt d}\epsilon_1,\quad
v=w+\frac{\tau}{\sqrt d}\epsilon_2.
$$

两端噪声独立。给定两单位向量，加噪后未归一化点积的精确扰动方差为

$$
\operatorname{Var}(u^Tv-z^Tw)=\frac{2\tau^2+\tau^4}{d}.
$$

补充正文给出的 $\tau\sqrt{2/d}$ 是一阶近似；本版补上噪声×噪声项。其主要判断保留：v1 的固定小网格不足以覆盖中等依赖区间。[S1, §§2.4–2.5]

采用补充提出的目标刻度 $\{0.95,0.85,0.70,0.50\}$，但在真实表示上把它明确命名为 **FIT-CAL 上某个固定校准 critic 的 J 刻度**，而非已知的真实 S。只在拟合侧确定噪声，冻结后用新的评估样本报告结果。目标不可达时如实记录，不依赖 EVAL 延长搜索。

同时比较各向同性与 FIT-PCA 固定子空间的噪声。PCA 的秩和基底只用拟合侧选择；整个尺度扫描固定。噪声后不再隐式 L2 归一化。

Goldfeld 等的平滑经验测度工作 [L10] 作为背景引用。其针对平滑经验测度和熵等对象的速率，不能直接转写成本神经 VCS 估计器的参数速率定理。

# 10. 论文应形成的实验结构

| 部分 | 主要图表 | 作用 |
|---|---|---|
| CS 估计路线 | 同目标误差与维度／样本／成本曲线 | 直接显示 estimator 的特点 |
| 神经估计参照 | MINE/NWJ/InfoNCE/SMILE/JS 的对应评价 | 保留合作者提出的稳定性问题 |
| 完整视觉 SSL | 从零训练、多种子、冻结表示与 CS 对照 | 主要视觉应用 |
| 机制 | 方法×增强交互、同类负配对贡献、P72 诊断 | 解释具体学习状态 |
| 二次结构分析 | 条件增量与可计算求解 | 有证据时作为分析小节或附录 |

论文首先报告哪些维度、样本和预算条件下得到改善，再用 SSL 说明应用价值。没有证据的噪声优势、标定优势、组合优势不进入贡献陈述。第二任务不设为硬指标，解释性服务于同一个 estimator。

# 11. 对补充六部分的处理索引

| 补充位置 | 本版处理 |
|---|---|
| §1 新结果与分支 | 纳入 P72；复用 P69/P73/P75；另外标注本轮核对的 P84 |
| §§2.2–2.3 残差与分歧 | 采纳平方回归和组合上界；说明上界的固定权重／整个字典区别 |
| §§2.4–2.5 噪声标度 | 采纳 FIT 标定；完整点积方差；真实 S 与校准 J 分开 |
| §2.6 二次结构 | 强调 L² 增量误差几何；归属二次评分规则，不称 VCS 独有 |
| §3.1 求解路线 | 加入收敛 cosine 和 tanh-ridge；P84 已有负结果不重复运行 |
| §§3.2、3.6 去饱和 | 优先强增强公平对照；门控直接测量，不以 1−J 代替 |
| §3.3 望远镜自检 | 改为平方增量／正交性／残差自检，排除纯代数恒等式 |
| §§3.4–3.5 梯度诊断 | 采纳现有模型诊断；同类配对饱和与梯度份额均作为待测 |
| §3.7 横向神经参照 | 保留全部 P69 estimator，不缩成只比 VCS/JS |
| §§4–5 降级与衔接 | 已知错配率校正留作备选；不重开 CLIP；不覆盖已批准实验 |
| §6 文献 | 纳入 stacking、boosting、proper scoring、平滑与假负例文献，并限定归属 |

# 12. 来源

[P1] 合作者，《Variational CS-QMI – Research Plan》，2026-09，用户提供的原始 PDF，特别是 §§3–10、12–13。

[V1] `VCS_QMI_Estimator_Research_Plan_v1_CN.md` 与 `VCS_QMI_Estimator_Server_Agent_Spec_v1.md`，上一版工作文档。

[S1] 用户补充：`VCS_QMI_Estimator_Package_v1_Supplement.md`，2026-09-28；包内保存原件。P72、P73、P69/P75 的补充时点描述均来自该文件，不能替代服务器实时状态。

[R84] [P84 冻结表示诊断报告](https://github.com/W-Yinghao/VCS_QMI/blob/d3b4d5ffb1539733314c83ab459535202e94c295/reports/P84_FROZEN_DIAG_REPORT_20260928.md)，本轮另行读取。

[L1] Yu et al. [Cauchy–Schwarz Divergence Information Bottleneck for Regression](https://arxiv.org/abs/2404.17951). ICLR, 2024.

[L2] Yamada et al. [Relative Density-Ratio Estimation for Robust Distribution Comparison](https://arxiv.org/abs/1106.4729). NeurIPS, 2011；[作者的 RuLSIF 页面](https://riken-yamada.github.io/RuLSIF)。

[L3] Gneiting and Raftery. [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). JASA, 2007.

[L4] Breiman. [Stacked Regressions](https://doi.org/10.1023/A:1018046112532). Machine Learning, 1996；Wolpert. [Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1). Neural Networks, 1992.

[L5] Friedman. [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). Annals of Statistics, 2001.

[L6] Bühlmann and Yu. [Boosting with the L2 Loss: Regression and Classification](https://stat.ethz.ch/Manuscripts/buhlmann/boosting.rev5.pdf). JASA 98:324–339, 2003（链接为作者稿）。

[L7] Reid and Williamson. [Information, Divergence and Risk for Binary Experiments](https://jmlr.org/papers/v12/reid11a.html). JMLR, 2011.

[L8] Chuang et al. [Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224). NeurIPS, 2020.

[L9] Huynh et al. [Boosting Contrastive Self-Supervised Learning with False Negative Cancellation](https://openaccess.thecvf.com/content/WACV2022/html/Huynh_Boosting_Contrastive_Self-Supervised_Learning_With_False_Negative_Cancellation_WACV_2022_paper.html). WACV, 2022.

[L10] Goldfeld et al. [Convergence of Smoothed Empirical Measures with Applications to Entropy Estimation](https://dspace.mit.edu/entities/publication/77790096-c0c0-4593-99af-fe09d22d320f). IEEE TIT 66(7):4368–4391, 2020.

[L11] Wang and Isola. [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://proceedings.mlr.press/v119/wang20k.html). ICML, 2020.

[L12] Song and Ermon. [Understanding the Limitations of Variational Mutual Information Estimators](https://arxiv.org/abs/1910.06222). ICLR, 2020.

[L13] Tsai et al. [Self-supervised Representation Learning with Relative Predictive Coding](https://arxiv.org/abs/2103.11275). ICLR, 2021.

**本版的本地核对。** `support/checks_v2.py` 核对了风险／变分 gap、残差伪目标与线段增益、字典分歧上界、嵌套增量及有限残差、log-loss 的 Bregman 对应和独立噪声因子的抵消。精确代数误差不超过 $1.4\times10^{-16}$，有限差分梯度误差约 $6.0\times10^{-11}$。固定单位向量的噪声方差 Monte Carlo 相对误差约0.19%。这是公式与标度的本地检查，不包含神经训练；没有复现补充未附带的完整 `noise_scale_check.py`。
