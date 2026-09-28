# VCS-QMI：围绕 estimator 的下一轮优化

日期：2026-09-28。仓库核对版本：`7b7402c05161c33d77a4301f6efc27bb55420e6a`。

本文区分已完成的仓库实验、由原定义推导的结论、下一轮研究建议。没有启动服务器训练，也没有修改仓库。独立数值核对仅涉及本文列出的合成分布和恒等式。

## 1. 当前建议

保留合作者的混合参考测度、二次变分目标和有界 critic。下一轮集中改进三个对象：critic 对最优后验的逼近、强依赖区间的观测分辨率、依赖增量的估计。先在可知真值的实验和冻结表示上验证，再把同一改动用于 SSL 和表示分析。

优先顺序是：**估计误差与梯度诊断 → 有界残差校正 → 观测尺度实验 → 同一 estimator 的 SSL 与条件依赖分析**。标准训练框架迁移仍有外部验证价值，但不应承担发现 estimator 特点的任务。

## 2. GitHub 最新证据

P68 已完成官方测试集评估。以下为三种子 linear accuracy，预算标签沿用仓库定义：

| 预算 | VCS | 调优后的 SimCLR | 调优后的 VICReg |
|---|---:|---:|---:|
| 1× | 82.98 ± 0.45 | 86.94 ± 0.12 | 87.00 ± 0.25 |
| 2× | 84.52 ± 0.18 | 87.99 ± 0.07 | 87.11 ± 0.31 |
| 4× | 85.87 ± 0.07 | 88.21 ± 0.10 | 86.91 ± 0.15 |
| 8× | 86.65 ± 0.26 | 88.13 ± 0.07 | 86.87 ± 0.06 |

四视图、800 epochs 的强增强单种子为 88.26%；这是一条探索结果，不是新的三种子基准。技术说明记录其 held-out J 约为 0.877，常规长训练约为 0.974。较早的强增强实验并未普遍改善性能，因此应检验“增强、训练时长、估计饱和”的交互，而不能只用一个点归因。

P48 在冻结表示上重新拟合 critic：最终配置的 J 从约 0.971 提升到 0.978；中间的 negative-detach 配置从 0.927 提升到 0.952。这说明最终配置仍有有限的拟合空间，但最大的 SSL 差距未必由最终 critic 的欠拟合解释。训练早期是否存在更大的追踪误差，目前仍需检查。

第二应用的最终汇总表明：闭式求解提供了有效的低成本测量方案；一般配准任务没有显示优势；小 batch 没有显示 VCS 特有优势；条件 nuisance 检测有局部积极结果，同时存在饱和与比较对象不足的问题。下一轮应直接改进测量本身。

## 3. 统一的估计对象

令 P 为联合分布，Q 为边缘乘积，M=(P+Q)/2。在平衡配对判别中，C=+1 表示来自 P，C=-1 表示来自 Q。

\[
\eta(w)=E[C\mid W=w]=\frac{p(w)-q(w)}{p(w)+q(w)},\qquad
S=E_M[\eta^2].
\]

\[
J(T)=E_PT-E_QT-E_M[T^2]=1-E[(C-T)^2].
\]

\[
S-J(T)=E_M[(T-\eta)^2].
\]

最后一个等式给出了直接的研发目标：减少 posterior-regression error，而不是只提高训练 J。

在合成实验中同时报告：

- 原目标 S 的估计误差及置信区间；
- 后验回归误差 E_M[(T-η)^2]；
- 相对 Bayes risk 的额外误差 (S-J)/(1-S)，仅在 S<1 且分母有足够数值精度时报告；
- 对可微通道参数的梯度误差，例如 ∂ρJ 与 ∂ρS 的差异。

最后一项直接连接 SSL。函数值拟合准确与编码器获得正确梯度是不同的验证问题。真实数据没有 η 真值，独立 refit 的 J 增量只能作为欠拟合证据，不能当成完整 oracle gap。

## 4. 二次损失与 tanh 的明确权衡

取 T=tanh(f)。将二次损失与 logistic 损失的梯度在 T=0 处对齐：

\[
\ell_V=\tfrac12(C-T)^2,\quad
\ell_{JS}=-\log\sigma(2Cf).
\]

\[
\partial_f\ell_V=(T-C)(1-T^2),\qquad
\partial_f\ell_{JS}=T-C.
\]

因此两者对同一 pair 的 logit 梯度比为 1-T²。|T|=0.9 时为 0.19，|T|=0.99 时为 0.0199。抑制高置信配对可以减弱部分错误配对的影响，也会减弱对高置信错误的纠正。对于参数梯度，还需乘上 ∇f；有界输出本身并不约束该 Jacobian。

在理想 critic 下还有：

\[
E_M[1-(T^*)^2]=1-S.
\]

这是平均门控因子，不是完整梯度范数。它说明强依赖与 tanh 回归学习信号之间存在结构性权衡。当前 critic 未必处于该理想状态，应分别记录正负 pair 的门控、损失残差、编码器梯度和估计误差。

当前代码中的 negative detach 保留数值 J，但移除了负配对第二端的 encoder 梯度。在共享编码器、相同视图边缘、对称 critic 和可交换采样条件下，负项的期望 encoder 梯度约为完整梯度的一半；有限 batch 的实现未必逐次精确等于一半，BatchNorm 等耦合也需单独考虑。下一轮的 JS 控制应使用同样的梯度路径，以分离损失形式与更新规则。

## 5. 优先方法一：有界残差校正

### 5.1 保持原目标，改进输出空间拟合

已有 bounded critic T0，新候选 U 也满足 [-1,1]。令：

\[
T_\lambda=(1-\lambda)T_0+\lambda U,\quad 0\le\lambda\le1.
\]

定义：

\[
A=E[(C-T_0)(U-T_0)],\qquad B=E[(U-T_0)^2].
\]

直接展开二次风险：

\[
J(T_\lambda)-J(T_0)=2\lambda A-\lambda^2 B.
\]

当 B>0 时，给定两位 critic 的最优线段步长为：

\[
\lambda^*=\Pi_{[0,1]}(A/B).
\]

B=0 表示两位 critic 在 M 下相同，保留 T0 即可。这里没有给不同 pair 重新赋予权重，没有改变 P、Q 或 J；优化的是两个有界函数的组合。

### 5.2 扩展为小型 critic 字典

\[
T_w=\sum_m w_mT_m,\quad w_m\ge0,\quad\sum_mw_m=1.
\]

\[
J(T_w)=\sum_mw_mJ(T_m)+E_M\!\left[\sum_mw_mT_m^2-T_w^2\right].
\]

最后一项是非负的 critic 分歧项。固定候选输出后，权重拟合是一个小规模凸二次问题；拟合目标不低于任何单个候选，但独立评估上的收益必须另测。

候选先用 3–5 个小型 tanh critics：余弦、元素乘积与绝对差、低秩交互、一个小残差网络。每个基础 critic 保持 tanh 输出。凸组合仍有界，并在输出位于开区间时可等价写成 tanh(atanh(Tw))；这是一种 critic 参数化变化，不是另一个散度。

首轮使用冻结的现有 VCS/SimCLR 表示及可知 η 的合成数据。FIT 拟合基础 critic，独立部分选择/拟合组合，EVAL 报告 J 与误差；数据较小时使用交叉拟合。样本划分以图像或独立观测为单位，不能把共享图像的 pair 随机拆到不同 split。

这条路线与此前加宽 MLP、增加 critic steps 不同：它提供明确的残差方向与输出空间求解。共同饱和或高度相关的错误会限制其收益。先证明估计改进，再决定是否用于 SSL 的在线训练。

## 6. 优先方法二：观测尺度下的依赖估计

### 6.1 强依赖的固有分辨率

考虑 X=Y 且在 n 个状态上均匀分布。联合分布在对角线上为 1/n，边缘乘积为 1/n²。代入定义得：

\[
S=\frac{n-1}{n+1}.
\]

n=100 时 S≈0.9802，n=1000 时 S≈0.9980。这是精确的离散例子，说明 pair identity 足以产生很大的目标值；它不是对真实 CIFAR 训练机制的直接证明。

### 6.2 保持同一个 estimator，明确改变被观测的变量

对固定尺度的表示定义：

\[
\widetilde Z_1=Z_1+\sigma\epsilon_1,\qquad
\widetilde Z_2=Z_2+\sigma\epsilon_2,
\]

其中噪声独立。仍用原来的 J 估计：

\[
S_\sigma=S(P_{\widetilde Z_1\widetilde Z_2},P_{\widetilde Z_1}P_{\widetilde Z_2}).
\]

对于未再归一化的高斯加性观测，较大的方差可由较小的方差再加独立噪声得到；数据处理给出 Sσ 的单调下降。critic 可以是这些变量上的受限函数，但函数类受限与拟合误差会使实际估计偏离该单调性。

这个方案保留混合参考理论与二次 estimator，测量目标则明确变成扰动后的变量。首先固定编码器，在少数预设噪声尺度上测量 Sσ 曲线、critic 误差和实际梯度门控。单位范数或冻结的尺度标准化用于防止编码器仅靠放大表示逃避噪声；不要自适应地把噪声优化回零。

只有曲线显示更好的可分辨区间与稳定的拟合后，才做小型 SSL 对照。VCS 和 JS/InfoNCE 使用相同观测噪声、增强、梯度路径和预算。需要同时保留 nuisance 与 class/instance 信号，防止噪声仅把所有信息一起抹掉。

这条路线的潜在价值是：用同一 estimator 测量和学习在指定观测误差下仍可识别的依赖。收益目前尚未得到真实训练验证。

## 7. 优先方法三：把解释性写成依赖增量

在同一个平衡 P/Q 实验中，设 A 是 B 的确定性压缩：

\[
T_A^*=E[C\mid A],\quad T_B^*=E[C\mid B].
\]

由条件期望的正交性：

\[
S_B-S_A=E_M[(T_B^*-T_A^*)^2].
\]

这是同一参考实验中两个可观测信息集的差异，不是 Shannon MI 的链式法则。

一个直接应用是条件 nuisance 分析。令：

\[
P=P_Y P_{H,N\mid Y},\qquad Q=P_Y P_{H\mid Y}P_{N\mid Y}.
\]

取 B=(H,N,Y)，A=(F(H),N,Y)，其中 F 为已训练模型的 logits。则：

\[
S(H;N\mid Y)-S(F(H);N\mid Y)
=E_M[(T_B^*-T_A^*)^2].
\]

它衡量表示中保留、在 logits 中被压缩掉的 nuisance 依赖。这里的条件 S 是上述 P/Q 下的二次依赖，不应与 Shannon CMI 混写。对真实依赖与实际预测使用的关系，继续采用可控 nuisance 改变及预测变化作验证。

有限 critic 的两个 J 相减会同时包含两个逼近误差：

\[
J_B-J_A=(S_B-S_A)-\mathcal E_B+\mathcal E_A,
\quad \mathcal E_j=E[(T_j-T_j^*)^2].
\]

所以解释性研究必须首先改善 critic 拟合。若使用 critic 差的平方作为增量估计，另有：

\[
\left|\sqrt{E[(T_B-T_A)^2]}-\sqrt{S_B-S_A}\right|
\le\sqrt{\mathcal E_B}+\sqrt{\mathcal E_A}.
\]

该界说明误差如何传入依赖定位；真实数据中没有直接的 oracle 误差，应通过合成控制、重复拟合、交叉拟合以及独立干预评估其可靠性。

## 8. 对现有 E/R/T/S 计划的具体修改

### E：从横向排名转为 estimator 改进实验

保留 Gaussian、可逆非线性变换、非高斯混合与 oracle。加入原 VCS、有界残差 VCS、同后验参数化的 JS。主比较为共同后验误差、各自目标误差、梯度误差与预算曲线。InfoNCE 的原生分数不是直接的平衡后验，不能未经归一化就纳入共同后验误差。

修订当前指标：J 是无量纲量，不能使用“超过 5 nats 的跳变”作为其稳定性判据。对有界 J，幅度超过全取值范围的 spike 条件没有鉴别力。learned/oracle 比值在 S≈1 时也不够敏感，增加相对 Bayes risk 的误差。

比较不同目标的分辨率时，可使用相邻依赖条件下估计值的排序概率/分辨 AUC；它对共同严格单调变换不变。已有 oracle 单点的 1.2 倍结果不能直接设为所有学习实验必须达到的结论。

JS 的最优散度有界，不代表任意 logit 的经验变分值双侧有界。应明确归一化：log2 + 1/2{EP log σ(f)+EQ log σ(-f)} 的上确界为 JS，有限 critic 的值可以任意负。

### R：围绕错误配对对学习的影响

同时报告对受污染真实目标 Sε 的估计误差，以及清洁任务上的性能保持。污染会改变 P，真实目标的变化不能混同为估计不稳定。

先验证可控错误配对中 VCS/JS 的梯度门控差异，再做相同条件下的 SSL 与图文适配。训练集错配识别加入交叉拟合或独立噪声评估，避免只测对训练 pair 的记忆。

对于固定表示，单调分数映射不会改变 AUROC；跨方法 AUROC 的改进应归因到所学表示或排序改变，校准需要 Brier、log loss、precision/coverage 等另行评价。

### T：保留公平控制，并增加条件依赖增量

继续 per-class bandwidth HSIC、学习核检验、连续分数 C2ST 和 JS。每次重复重抽 nuisance，保留独立的拟合与置换评估。

增加幅度估计的重复性和对可控 nuisance 强度的分辨能力。小样本功效接近时，使用预先指定的主比较和置信区间，而不以每个格子必须跨过固定百分比阈值作为是否继续研究的依据。

### S：使每项 SSL 改动对应 estimator 假设

当前配方作为保留基准。优先测试已在 E 上证实的拟合改进，以及已有曲线证据支持的观测尺度设置。同步运行同一后验参数化、同一梯度路径的 JS 控制。标准框架迁移用于最后验证实现和协议外部性。

官方 CIFAR 测试结果已经看过。新一轮选择只使用训练/验证数据；后续测试需如实披露这是后续研究，不能再称完全未触碰的一次性测试。关键新结论应增加此前未用于开发的数据集或其他独立确认。

## 9. RuLSIF 的精确关系与求解条件

RuLSIF 的相对密度比在 α=1/2 时为：

\[
r_{1/2}=p/M=1+T^*.
\]

取任意 g=1+T：

\[
\tfrac12E_M[g^2]-E_Pg=-\tfrac12-\tfrac12J(T).
\]

因此相同函数类下两者只差仿射变换。比较闭式解时，需要匹配函数类、常数项、边界约束和正则化。原始 ridge 解之后再包 tanh，是另一个函数类/求解过程，不能要求它仍和未经包裹的 RuLSIF 解逐点一致。

对 kernel RuLSIF 与 neural VCS 的差异，应研究高维函数近似、泛化与求解效率。不得将等价目标的参数化差异写成散度本身的优势。这个关系也允许直接借用相对密度比回归的成熟求解方式。

## 10. 次级可能性：已知配对噪声下恢复清洁目标

若 Pε=(1-ε)P+εQ 且边缘保持不变，在 ε 已知时有：

\[
T_\epsilon^*=\frac{(1-\epsilon)T^*}{1-\epsilon T^*},\qquad
T^*=\frac{T_\epsilon^*}{1-\epsilon+\epsilon T_\epsilon^*}.
\]

也可直接构造清洁目标的无偏矩形式：

\[
J(T;P,Q)=\frac{E_{P_\epsilon}[T-\tfrac12T^2]
-E_Q[T+\tfrac12(1-2\epsilon)T^2]}{1-\epsilon}.
\]

该方案需要额外的噪声率信息；误差与方差会被分母放大。它不是未知噪声下自动恢复清洁依赖的保证，也并非仅 VCS 才能进行混合分布校正。暂作为后续可能性，不排在当前前三条之前。

## 11. 本轮独立数值核对

`independent_checks.py` 使用 NumPy，固定 seed=20260928。

离散分布精确求和核对了变分 gap、凸组合恒等式、残差步长增益、RuLSIF 仿射关系、嵌套增量恒等式，绝对误差均不超过 3e-16。二次与 logistic 梯度用有限差分核对，误差小于 4e-11。

另对 20 维相关 Gaussian，以每个 P/Q 各 300,000 样本计算 oracle J：

| 用于设定相关性的 Shannon MI（nats） | oracle S 的 MC 估计 | MC 标准误 | oracle JS（nats） |
|---|---:|---:|---:|
| 2 | 0.56725 | 0.00102 | 0.35030 |
| 4 | 0.80218 | 0.00080 | 0.52711 |
| 6 | 0.91203 | 0.00056 | 0.61648 |
| 8 | 0.96280 | 0.00038 | 0.65972 |
| 10 | 0.98525 | 0.00024 | 0.67945 |

这里 Shannon MI 仅为合成通道参数的标尺，S 的真值对象仍是原二次依赖。此表展示强依赖区间的压缩，不是 learned estimator 或某种训练方法的性能比较。

## 12. 建议的执行顺序

1. 用少量合成条件复核 oracle、误差和梯度定义；读现有早/中/晚 checkpoint 的真实梯度与独立 refit 增量。
2. 在冻结表示上检验 3–5 个 critic 的有界残差组合，并保留同后验 JS；验证估计收益与代价。
3. 在冻结表示上测少数观测噪声尺度的依赖曲线；仅对有解释依据的设置进入训练。
4. 将同一 estimator 变体用于有限的 SSL、错误配对学习和条件增量分析，不同时铺开大量独立应用。

最终要回答的问题是：哪项 estimator 改进减少了哪类误差，它怎样改变了表示学习，怎样提高了依赖定位的可靠性。

## 来源

### 本次核对的仓库文件（均锁定同一提交）

- [官方测试结果 P68](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md)
- [下一轮 E/R/T/S 计划](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md)
- [SSL 技术说明](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/SSL_TECHNICAL_NOTE_20260926.md)
- [第二应用最终汇总](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/SECOND_APP_FINAL_SUMMARY_20260928.md)
- [冻结表示上的 critic 重拟合 P48](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/reports/P48_PRECHECK_B1_REPORT_20260927.md)
- [目标与梯度路径实现](https://github.com/W-Yinghao/VCS_QMI/blob/7b7402c05161c33d77a4301f6efc27bb55420e6a/src/vcs_ssl/objectives.py)

### 原始理论和外部文献

- 用户提供的《Variational CS-QMI – Research Plan》，2026 年 9 月，特别是变分表示、回归 gap 与 bounded critic 部分。
- Yamada et al. Relative Density-Ratio Estimation for Robust Distribution Comparison. NeurIPS 2011. https://papers.neurips.cc/paper_files/paper/2011/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html
- Reid & Williamson. Composite Binary Losses. JMLR 2010. https://www.jmlr.org/papers/volume11/reid10a/reid10a.pdf
- Reid & Williamson. Information, Divergence and Risk for Binary Experiments. JMLR 2011. https://jmlr.org/papers/v12/reid11a.html
- Song & Ermon. Understanding the Limitations of Variational Mutual Information Estimators. ICLR 2020. https://openreview.net/pdf/381bba14579e1a88d1b1fec45df52d0fa9dd9fc6.pdf
- Tschannen et al. On Mutual Information Maximization for Representation Learning. ICLR 2020. https://research.google/pubs/on-mutual-information-maximization-for-representation-learning/
