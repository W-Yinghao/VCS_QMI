---
title: "VCS-QMI：以估计器为中心的优化计划"
subtitle: "后验拟合、观测分辨率与依赖增量"
author: "研究分析稿 · v1"
date: "2026 年 9 月 28 日"
---

# 1. 研究目标与下一轮主线

这篇文章的中心是合作者提出的 **variational CS-QMI estimator**。下一轮的任务，是提高它在有限数据与有限计算下的估计质量，并说明这种改进如何帮助表示学习和依赖分析。

建议围绕一条连续的研究主线展开：

> **准确拟合配对后验 → 在强依赖区间保持有效分辨率 → 用同一估计器学习表示、测量表示中保留的信息。**

具体推进三个方向：

| 方向 | 优化的对象 | 首先验证什么 | 应用落点 |
|---|---|---|---|
| 有界残差校正 | critic 对最优后验的逼近 | 是否减少独立评估上的回归误差 | SSL 的训练信号；依赖测量精度 |
| 观测尺度下的依赖估计 | 强依赖区间的可分辨性 | 不同观测误差下，哪些依赖仍然可识别 | 有效的配对学习；信息保留曲线 |
| 依赖增量分析 | 两个嵌套信息集之间的差值 | 增量估计是否准确、可重复 | 表示与 logits 的 nuisance 信息比较 |

三条路线共享原来的混合参考构造和二次目标。第一条改变 critic 的拟合方式；第二条明确定义扰动后的被观测变量；第三条在同一个二元实验中比较两个信息集。[P1; A1]

**下一轮的主结果应当是：某项 estimator 改进减少了一类可测量误差，并进一步改善学习或测量。** SSL 准确率、检验功效和可解释性结果都围绕这条联系组织。

本文依据上一轮已核对的仓库快照 `7b7402c05161c33d77a4301f6efc27bb55420e6a` 整理，不代表本轮重新检查了 GitHub HEAD。仓库结果、由定义推导的结论、拟议方法在下文分别说明。本文是研究计划，不是新实验结果报告。

# 2. 现有证据告诉了我们什么

## 2.1 SSL 已经能学到有效表示，优势仍需由 estimator 改进建立

P68 已完成官方 CIFAR-10 测试。以下是三种子冻结 backbone 线性分类准确率，预算名称沿用仓库报告；它们表示该报告中的计算分组，并非实测时间或总 FLOPs 的严格等值。[R1]

| 预算 | VCS | 调优后的 SimCLR | 调优后的 VICReg |
|---|---:|---:|---:|
| 1× | 82.98 ± 0.45 | 86.94 ± 0.12 | 87.00 ± 0.25 |
| 2× | 84.52 ± 0.18 | 87.99 ± 0.07 | 87.11 ± 0.31 |
| 4× | 85.87 ± 0.07 | 88.21 ± 0.10 | 86.91 ± 0.15 |
| 8× | 86.65 ± 0.26 | 88.13 ± 0.07 | 86.87 ± 0.06 |

当前 VCS 主配置采用余弦–tanh critic、负配对单侧 stop-gradient、四视图和八个负配对。它证明原工具可以支持有用的视觉表示；表格没有显示相对于调优 SimCLR 的准确率优势。[R1; R3]

一条重要线索来自强增强：四视图、800 epochs 的强增强单种子测试结果为 **88.26%**；技术说明记录其 held-out $J$ 约为 **0.877**，常规长训练约为 **0.974**。较早的强增强试验并未普遍改善分类，因此值得研究的是增强、训练时长和估计饱和的交互。[R1; R3]

## 2.2 最终 critic 还有拟合空间，但改善幅度有限

冻结表示后的重新拟合，在最终 VCS 配置上将独立配对分数从约 0.971 提高到 0.978；在较早的 negative-detach 配置上，从 0.927 提高到 0.952。[R5]

这两项结果提示，应分别研究：

- **拟合瓶颈**：训练过程中的 critic 是否充分追踪了当前配对后验？
- **分辨率瓶颈**：配对已经接近完全可分时，更准确的估计还能给 encoder 提供多少有用信息？

最终 checkpoint 上的小幅重拟合增益不能回答训练早期是否存在较大追踪误差。下一轮应加入早、中、晚 checkpoint 的固定表示分析。

## 2.3 第二应用应继续围绕测量本身

仓库的第二应用汇总显示：闭式线性类拟合具有低成本测量价值；配准和小 batch 场景尚未显示 VCS 特有的收益；条件 nuisance 检测存在局部积极结果，并需要更强的 JS、核方法和连续分数对照。部分结果还受到强依赖饱和的影响。[R4]

因此，下一步将解释性写成明确的估计问题：**估计表示与属性的依赖，以及信息压缩后的依赖增量。** 任务只是让估计器的精度、分辨率和成本具有实际含义。

# 3. 统一 estimator 的学习对象和评价对象

## 3.1 原始定义保持不变

设 $P$ 为真实联合分布，$Q$ 为边缘乘积分布，

$$
M=\frac12(P+Q).
$$

定义平衡二元实验：$C=+1$ 时观测来自 $P$，$C=-1$ 时来自 $Q$。对于完整配对观测 $W$，

$$
\eta(w)=\mathbb E[C\mid W=w]
=\frac{p(w)-q(w)}{p(w)+q(w)},
\qquad
S=\mathbb E_M[\eta^2].
$$

原始变分目标为

$$
J(T)=\mathbb E_PT-\mathbb E_QT
-\frac12\mathbb E_PT^2-\frac12\mathbb E_QT^2.
$$

它满足

$$
J(T)=1-\mathbb E[(C-T)^2],
\qquad
S-J(T)=\mathbb E_M[(T-\eta)^2].
$$

基础 critic 采用 $T_\phi=\tanh f_\phi$，总体最优函数为 $\tanh(\mathrm{PMI}/2)$。训练和主要报告量是 $S$ 对应的二次目标；CS 对数变换仅承担解释或单独报告的作用。[P1, §§3–7, 9–10]

## 3.2 下一轮首先评价后验逼近

精确 gap 把问题变成一个直接的回归任务：**用有限样本尽可能准确地拟合 $\eta$。**

| 评价层 | 指标 | 用途 |
|---|---|---|
| 目标数值 | $\widehat J-S$、RMSE、偏差、重复实验波动 | 判断估计数值是否准确 |
| 后验拟合 | $\mathcal E=\mathbb E_M[(T-\eta)^2]$ | 直接测量 critic 的逼近误差 |
| 强依赖分辨率 | $\mathcal E/(1-S)$、相邻条件的排序概率 | 判断近饱和区间还剩多少可用分辨率 |
| 训练信号 | $\partial_\rho J$ 与 $\partial_\rho S$ 的误差 | 检查 estimator 能否给出正确优化方向 |
| 资源代价 | 独立样本数、拟合时间、评估时间、显存 | 建立误差与成本之间的关系 |

相对 Bayes risk 指标只在 $S<1$、$1-S$ 有足够精度时报告。例如 $S=0.99$、$J=0.97$，两者比值接近 1，但额外回归误差已经是最优回归风险的两倍。

对固定、独立评估得到的 critic，还可以写成

$$
\widehat J(\widehat T)-S
=\bigl[\widehat J(\widehat T)-J(\widehat T)\bigr]
-\mathcal E(\widehat T).
$$

因此，采样波动与 critic 逼近误差要分别分析。真实表示没有 $\eta$ 真值，独立 refit 的增益只提供欠拟合证据。

# 4. 一条核心机制：tanh 二次回归的梯度门控

## 4.1 与同后验 logistic 的直接对照

令 $T=\tanh f$。将局部二次损失和 logistic 损失的初始梯度尺度对齐：

$$
\ell_V=\frac12(C-T)^2,
\qquad
\ell_{\mathrm{JS}}=-\log\sigma(2Cf).
$$

直接求导：

$$
\frac{\partial\ell_V}{\partial f}
=(T-C)(1-T^2),
\qquad
\frac{\partial\ell_{\mathrm{JS}}}{\partial f}=T-C.
$$

两种损失都对应同一个平衡后验，VCS 多出 $1-T^2$ 因子。关于损失与连接函数共同决定优化行为的背景，见 [L1]；上式由当前定义直接推出。

| $|T|$ | 门控因子 $1-T^2$ |
|---:|---:|
| 0 | 1 |
| 0.9 | 0.19 |
| 0.99 | 0.0199 |

该因子既可能减弱高置信错误配对的影响，也可能减慢对高置信错误的纠正。这是待通过实验区分的取舍。

上述 $\ell_V$ 用于推导局部梯度。实际 VCS 训练仍使用原始 $-J$；服务器文档明确给出整体风险和 JS 控制的倍数，避免通过隐含 loss scaling 造成学习率不公平。

## 4.2 强依赖与平均门控的关系

在总体最优 critic 处，

$$
\mathbb E_M[1-(T^*)^2]=1-S.
$$

当 $S$ 接近 1 时，平均门控因子也接近零。完整参数梯度还取决于回归残差与网络 Jacobian，因此要测量实际的配对梯度，不能只看 $J$ 或饱和比例。

建议在固定早、中、晚 checkpoint 上，记录正负配对的 $T$、残差、$1-T^2$、logit 梯度和输入表示梯度。加入可控错配后，比较干净与错配样本的贡献。

**需要验证的假设：VCS 对高置信配对的抑制，是否同时解释某些噪声条件下的耐受性与干净 SSL 中的学习效率损失？**

当前 negative detach 也改变 encoder 更新。相同梯度路径的 JS 对照必须保留，才能分离损失形式和更新方式。[R6]

# 5. 路线一：有界残差校正

## 5.1 两个 critic 之间的精确更新

设已有 critic 为 $T_0$，新候选为 $U$，两者都取值于 $[-1,1]$。定义

$$
T_\lambda=(1-\lambda)T_0+\lambda U,
\qquad 0\leq\lambda\leq1.
$$

令

$$
A=\mathbb E[(C-T_0)(U-T_0)],
\qquad
B=\mathbb E[(U-T_0)^2].
$$

展开原二次风险可得

$$
J(T_\lambda)-J(T_0)=2\lambda A-\lambda^2B.
$$

当 $B>0$，给定两个 critic 的最优线段步长为

$$
\lambda^*=\Pi_{[0,1]}(A/B).
$$

当 $B=0$，保留 $T_0$。这个投影约束作用在组合系数上，不是对 pair 或 $J$ 的裁剪。

方法的工作方式很直接：**寻找能补足当前后验回归误差的有界候选，再按二次目标确定加入多少。**

## 5.2 小型 critic 字典

对固定的候选 $T_1,\ldots,T_m$，令

$$
T_w=\sum_{j=1}^m w_jT_j,
\qquad w_j\geq0,\quad\sum_jw_j=1.
$$

则

$$
J(T_w)=\sum_jw_jJ(T_j)
+\mathbb E_M\left[\sum_jw_jT_j^2-T_w^2\right].
$$

最后一项是非负的分歧项。固定候选输出后，权重拟合成为 simplex 上的凸二次问题；拟合目标不低于任何单个候选，独立评估收益则需要验证。

每个基础 critic 保持 tanh 输出，凸组合也保持有界。只要组合值在开区间内，它在数学上可写作 $\tanh(\operatorname{atanh}T_w)$；实现直接计算加权和，不进行这个不必要的数值逆变换。

这一组合是本轮提出的 critic 参数化与求解方案，不是原计划已经实现的内容。首轮使用三至五个小型候选：余弦或点积、低秩交互、一个小型联合 MLP。候选与权重采用不同数据拟合，最终在独立 EVAL 上评价。

## 5.3 先分离两个问题

**固定字典组合**回答：多个现有 critic 的误差是否互补？

**残差候选更新**回答：按已有 critic 的缺口拟合新函数，是否比独立增加容量更有效？

第二类实现可以保持原目标：固定 $T_0$，用一个预先给定的混合系数训练 $U$，直接最大化 $J((1-\lambda_0)T_0+\lambda_0U)$；随后在独立数据上拟合最终步长。这里的训练安排是服务器文档新增的实施默认。

比较时包含同参数量的大 MLP、多随机初始化的候选选择，以及同预算的 JS 组合，区分容量、集成和二次求解各自的作用。共同饱和和相关误差会限制组合收益，这也是需要测量的结果。

# 6. 路线二：观测尺度下的依赖估计

## 6.1 为什么要研究观测分辨率

考虑精确离散例子：$X=Y$，且 $X$ 在 $n$ 个状态上均匀分布。原定义给出

$$
S=\frac{n-1}{n+1}.
$$

$n=100$ 时 $S\approx0.9802$，$n=1000$ 时 $S\approx0.9980$。实例身份本身就能产生很大的依赖值；这个例子揭示目标的饱和结构，不是对真实 CIFAR 编码机制的直接证明。

## 6.2 明确改变观测变量，保留 estimator

对尺度固定的表示定义

$$
\widetilde Z_1=Z_1+\sigma\epsilon_1,
\qquad
\widetilde Z_2=Z_2+\sigma\epsilon_2,
\qquad
\epsilon_1\perp\epsilon_2.
$$

继续使用原 $J$，估计

$$
S_\sigma=S(P_{\widetilde Z_1\widetilde Z_2},
P_{\widetilde Z_1}P_{\widetilde Z_2}).
$$

高斯加性观测在较大方差下可以由较小方差继续加入独立噪声得到。因此，对未再归一化的完整观测变量，数据处理性质给出 $S_\sigma$ 随噪声方差单调下降。[P1, §8]

每个尺度定义自己的目标。受限 critic 的估计曲线可能因函数类和拟合误差而不单调；不能通过事后单调化隐藏这种差异。

## 6.3 研究问题与实验顺序

首先固定编码器，使用少数预设尺度，测量依赖曲线、后验拟合误差和实际门控。记录类别、实例与 nuisance 相关信息的保留情况。

若在某个尺度下，易分的实例配对不再迅速饱和，同时有用关系仍然保留，再将该设置带入有限 SSL 训练。VCS 与 JS/InfoNCE 共用相同观测噪声、增强、梯度路径和预算。

该路线的潜在价值是：**测量和学习在指定观测误差下仍然存在的关系。** 当前没有新的真实训练结果支持其性能优势。

服务器初始默认将噪声用“总 RMS 幅度”表示，以便跨维度解释；它与上式每坐标标准差之间的换算单独写明。噪声之后不再隐式 L2 归一化，避免更换测量对象。

# 7. 路线三：用依赖增量组织解释性

## 7.1 同一个二元实验中的嵌套信息集

设 $A$ 是 $B$ 的确定性压缩，并保持同一个平衡 $P/Q$ 实验。定义

$$
T_A^*=\mathbb E[C\mid A],
\qquad
T_B^*=\mathbb E[C\mid B].
$$

由条件期望正交性，

$$
\boxed{S_B-S_A=\mathbb E_M[(T_B^*-T_A^*)^2].}
$$

这里 $S_A$、$S_B$ 分别是原 $P/Q$ 经对应观测映射后的二次依赖。它把“增加可观测信息后多识别了多少配对关系”写成精确回归差值。二元风险与散度的背景见 [L2]。

## 7.2 具体任务：表示与 logits 中的 nuisance 信息

设 $H$ 为冻结表示，$N$ 为颜色、模糊或其他属性，$Y$ 为类别。定义

$$
P=P_Y P_{H,N\mid Y},
\qquad
Q=P_Y P_{H\mid Y}P_{N\mid Y}.
$$

比较

$$
B=(H,N,Y),\qquad A=(F(H),N,Y),
$$

其中 $F$ 是已固定、确定性的分类头，输出 logits。则

$$
\Delta=S(H;N\mid Y)-S(F(H);N\mid Y)
=\mathbb E_M[(T_B^*-T_A^*)^2].
$$

这里条件 $S$ 指上述条件联合与条件乘积分布的二次依赖。$\Delta$ 衡量表示中保留、在 logits 中被压缩掉的 nuisance 依赖。

同步报告可控 nuisance 改变后的 logits 和准确率变化，检验测量与实际决策行为的联系。依赖表示“观测中含有属性信息”；预测依赖该属性需要额外的干预证据。[R4]

## 7.3 estimator 的误差直接影响解释结果

对于有限 critic，

$$
J_B-J_A=(S_B-S_A)-\mathcal E_B+\mathcal E_A,
\quad
\mathcal E_j=\mathbb E[(T_j-T_j^*)^2].
$$

另一个候选估计量是

$$
D_T=\mathbb E_M[(T_B-T_A)^2].
$$

它满足

$$
\left|\sqrt{D_T}-\sqrt{\Delta}\right|
\leq\sqrt{\mathcal E_B}+\sqrt{\mathcal E_A}.
$$

因此需要同时报告差值与平方差，并先在可知真值的嵌套分布上验证。平方差天然非负，但正值也可能来自两个 critic 的拟合差异。

将路线一用于两个信息集的拟合，能够建立一条直接的方法联系：**更好的后验逼近，是否带来更准确、可重复的依赖增量？**

# 8. 下一轮实验怎样围绕 estimator 收束

## 8.1 修改 E/R/T/S 四条线

| 现有方向 | 下一轮的主要问题 | 主要交付 |
|---|---|---|
| E：受控估计 | 原 VCS、残差 VCS 和 JS 的后验误差、目标误差、梯度误差如何随样本和预算变化？ | estimator 精度与成本曲线 |
| R：污染和错配 | 门控如何改变错配影响？是在估计受污染目标，还是在恢复清洁目标？ | 污染真值误差与清洁任务结果 |
| T：条件检验 | 相同 critic 能力下是否改进检验与增量测量？ | 功效、水平、增量误差与重复性 |
| S：SSL | 已验证的 estimator 改进能否转化为表示学习收益？ | 有限、可归因的 SSL 对照 |

E 为第一优先级。初期不把所有数据集、所有 critic 和所有参数做成全排列。每一项扩展都应回应前一轮识别出的误差类型。

## 8.2 指标需要匹配目标

$J$ 是无量纲量。有界 critic 下，任意经验值在 $[-3,1]$ 内，使用“超过 5 nats 的跳变”无法衡量它的稳定性。应报告原目标误差、分位数、参数梯度与分辨能力。

JS 的最优散度有界，不意味着任意 critic 的经验变分值双侧有界。以原始 log-density-ratio 参数 $\ell$ 书写，

$$
J_{\mathrm{JS}}(\ell)=\log2+
\frac12\mathbb E_P\log\sigma(\ell)
+\frac12\mathbb E_Q\log\sigma(-\ell).
$$

其上确界为 JS，有限 critic 的值可以很负。匹配后验时取 $\ell=2f$，与 $T=\tanh f$ 对齐。

不同目标之间先比较共同后验误差、相邻条件的分辨概率、检验功效和成本；各自目标误差分别列出。InfoNCE 的原生分数未经校准时不是平衡后验，不直接放入后验 MSE 表。

## 8.3 公平对照与后续验证

RuLSIF 在 $\alpha=1/2$ 时满足

$$
r_{1/2}=p/M=1+T^*.
$$

对任意 $g=1+T$，

$$
\frac12\mathbb E_M[g^2]-\mathbb E_Pg
=-\frac12-\frac12J(T).
$$

这是同一函数类下的仿射等价关系。[L3] 数值一致性检查需要匹配常数项、函数边界和正则化。ridge 解再套 tanh 应作为另一个拟合过程评价。

官方 CIFAR 测试已经完成。新方法继续只在开发划分选择；后续确认需要清楚区分已有测试与新一轮研究，并增加此前未参与开发的数据或其他独立确认。[R1]

# 9. 次级可能性：已知错配率下恢复清洁目标

若

$$
P_\varepsilon=(1-\varepsilon)P+\varepsilon Q,
$$

且边缘不变，$\varepsilon$ 已知、$0\leq\varepsilon<1$，则

$$
T_\varepsilon^*=
\frac{(1-\varepsilon)T^*}{1-\varepsilon T^*},
\qquad
T^*=\frac{T_\varepsilon^*}{1-\varepsilon+\varepsilon T_\varepsilon^*}.
$$

也可把清洁目标写成

$$
J(T;P,Q)=\frac{
\mathbb E_{P_\varepsilon}[T-\tfrac12T^2]
-\mathbb E_Q[T+\tfrac12(1-2\varepsilon)T^2]
}{1-\varepsilon}.
$$

该方向需要额外的噪声率信息，并会放大误差与方差。当前只保留为后续研究备选，不放入首轮执行范围。

# 10. 独立核对的范围

上一轮附件中的 `independent_checks.py` 和 JSON 记录包含：变分 gap、凸组合、残差步长、RuLSIF 仿射关系、嵌套增量的离散精确求和，以及局部梯度的有限差分核对。[A1]

记录中五项恒等式的绝对误差均低于 $3\times10^{-16}$；二次和 logistic 的有限差分误差低于 $4\times10^{-11}$。以下 Gaussian 数值同样来自该合成核对，每个分布使用 300,000 个样本：

| 设置相关性的 Shannon MI（nats） | oracle $S$ 的 MC 估计 | MC 标准误 |
|---:|---:|---:|
| 2 | 0.56725 | 0.00102 |
| 4 | 0.80218 | 0.00080 |
| 6 | 0.91203 | 0.00056 |
| 8 | 0.96280 | 0.00038 |
| 10 | 0.98525 | 0.00024 |

Shannon MI 在此仅用于设定相关性，$S$ 仍是目标。这组结果展示近饱和区间的压缩，不是训练出的 neural estimator 比较。

本次交付另附一个小型 NumPy 参考模块和检查，验证权重求解、残差系数、风险归一化和已知错配恒等式。验证范围记录在 `support/validation_report.json` 中，不包含真实神经训练或服务器作业。

# 11. 建议的推进顺序

**第一步：建立共同误差评价。** 用少数合成条件核对 oracle、后验误差和梯度定义，并测量已有 checkpoint 的早、中、晚 refit 增益。

**第二步：检验有界组合与残差校正。** 在固定数据和表示上证明 estimator 的误差与成本是否改善；保留 JS 和容量匹配对照。

**第三步：测量观测尺度曲线。** 区分真实目标变小、critic 更容易拟合和有用关系被保留这三件事。

**第四步：应用同一有效改进。** 进行有限 SSL 对照和条件依赖增量测量。由 estimator 的效果决定应用扩展，而不同时建设多个独立任务。

论文最终应呈现的是：**这个 estimator 估计什么、怎样拟合得更好、在哪些条件下有优势，以及这个优势怎样进入学习和解释。**

# 来源与阅读依据

**[P1] 合作者原计划。** 用户提供的《Variational CS-QMI – Research Plan》，2026 年 9 月，特别是 §§3–7、9–10。本文保留其中的 $M$、$S$、$J$、tanh critic 和回归 gap。

**[A1] 上一轮分析附件。** 本文是该分析的整理版；服务器的具体实施默认另行标注。原分析及随附的数值检查文件名如下：

```text
VCS_QMI_Estimator_Optimization_Review_20260928.md
independent_checks.py / independent_checks.json
```

以下仓库来源均固定到 `7b7402c05161c33d77a4301f6efc27bb55420e6a`：

- **[R1]** [P68 官方测试结果](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md)。
- **[R2]** [下一轮 E/R/T/S 执行草案](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md)。
- **[R3]** [SSL 技术说明](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/SSL_TECHNICAL_NOTE_20260926.md)。
- **[R4]** [第二应用最终汇总](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/SECOND_APP_FINAL_SUMMARY_20260928.md)。
- **[R5]** [冻结表示 critic 拟合 P48](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/P48_PRECHECK_B1_REPORT_20260927.md)。
- **[R6]** [训练目标与梯度路径](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/src/vcs_ssl/objectives.py)。

以下文献沿用上一轮分析的引用，本轮未新增文献检索：

- **[L1]** Reid and Williamson. *Composite Binary Losses*. JMLR, 2010.
- **[L2]** Reid and Williamson. *Information, Divergence and Risk for Binary Experiments*. JMLR, 2011.
- **[L3]** Yamada et al. *Relative Density-Ratio Estimation for Robust Distribution Comparison*. NeurIPS, 2011.
- **[L4]** Song and Ermon. *Understanding the Limitations of Variational Mutual Information Estimators*. ICLR, 2020.
- **[L5]** Tschannen et al. *On Mutual Information Maximization for Representation Learning*. ICLR, 2020.
