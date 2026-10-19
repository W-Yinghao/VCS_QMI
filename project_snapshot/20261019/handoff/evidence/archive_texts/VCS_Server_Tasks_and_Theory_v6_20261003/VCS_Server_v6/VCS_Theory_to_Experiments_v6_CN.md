---
title: "从配对证据到表示几何"
subtitle: "v6 理论与实验对应说明｜服务器任务书附录"
date: "2026-10-03"
lang: zh-CN
---

# 1. 理论边界与符号

本文件区分三类内容：**合作者已给出的结果**、**由原定义继续推导的关系**、**为 SSL 提出的实验候选**。第一类来自用户提供的 *Variational CS-QMI – Research Plan* §3–9；第二类由下述公式独立推导，并附精确求和/有限差分检查；第三类尚无完整训练结果。

用 $\theta$ 表示 encoder/projector 参数，$\phi$ 表示 measurement critic 参数。$s=z_1^Tz_2$ 是单配对余弦，$S(P,Q)$ 是总体依赖量，二者不能混写。

## 1.1 主理论：不依赖 SSL

对于任意两分布 $P,Q$，$M=(P+Q)/2$，令 $C$ 为平衡二元标签，

$$
\eta=E[C\mid W]=\frac{p-q}{p+q},\quad S=E_M\eta^2,
\quad J(T)=2E_M[\eta T]-E_MT^2.
$$

因此

$$
S-J(T)=E_M(\eta-T)^2,\qquad J=1-E(C-T)^2.
$$

在共同正密度区域，$f^*=\operatorname{atanh}\eta=\tfrac12\log(p/q)$；没有共同正密度时可在无穷 logit 的极限实现边界。$J(T)\leq S$ 是总体结论，有限 $\widehat J$ 仍有采样波动。

## 1.2 SSL 特有选择

当 $P$ 为同图两视图分布、$Q$ 为其边缘乘积时，$f_\phi$ 读出配对证据。使用单位表示内积 $s$ 和 $f=a(s-\kappa)$ 是函数类选择，不是主定理要求。固定刻度既可限制测量能力，也可给 encoder 施加一种几何要求；学习和测量必须分别评价。

# 2. 嵌套观测分解及其可识别范围

若 $W\to Z\to s$ 均为确定性映射，且 $T$ 只依赖 $s$，定义

$$
\eta_Z=E[\eta_W\mid Z],\quad \eta_s=E[\eta_W\mid s].
$$

将 $\eta_W-T=(\eta_W-\eta_Z)+(\eta_Z-\eta_s)+(\eta_s-T)$ 展开，各交叉项因条件期望为零，于是

$$
S_W-J(T)=L_{W\to Z}+L_{Z\to s}+L_{s\to T},
$$

其中三项分别为相应差值的 $L^2(M)$ 范数平方。

最后一项是**评分拟合误差**，不是 $s\to T$ 的信息损失：若 $T=\tanh[a(s-\kappa)]$ 且 $a>0$，它在无限精度下对 $s$ 严格单调、可逆。它能保留 $s$ 的全部可测信息，却仍然给出错误的后验数值刻度。这是“可读信息”与“概率标定”的重要区别。

对于一般受限 critic 类 $\mathcal T$，

$$
\sup_{T\in\mathcal T}J(T)=S-\inf_{T\in\mathcal T}\|T-\eta\|_2^2.
$$

**不能从两个有限拟合器的差值直接识别压缩项。** 若 $\widehat T_Z,\widehat T_s$ 误差分别为 $e_Z,e_s$，那么

$$
J(\widehat T_Z)-J(\widehat T_s)
=(S_Z-S_s)-e_Z+e_s.
$$

因此 V6-EVIDENCE 报告的是在指定函数类/预算下增加输入恢复了多少评分。精确分解只在有限真值模型中核对。

P117 使用 oracle 辅助的分箱条件均值估计，不是直接观测完整 $E[\eta\mid T_0]$；分箱误差可混入 A 项。分解残差很小是计算一致性检查，不单独证明 binning 已逼近真正条件期望。该限制不否定其“已测试标定器未带来明显收益”的直接实证结果。

# 3. 仿射评分与密度比相容性

对于标量分数分布 $P_s,Q_s$，若在其共同支持上

$$
f(s)=a(s-\kappa)=\tfrac12\log[p_s(s)/q_s(s)],
$$

则

$$
p_s(s)=e^{2a(s-\kappa)}q_s(s),\quad
\kappa=\frac{\log E_{Q_s}e^{2as}}{2a}.
$$

需要 $P_s\ll Q_s$ 且相应比值确实归一化；这里的指数矩是正确密度比的必要恒等式，不是训练建议。由于 $s$ 有界、固定 $a$ 有限，数学上该矩有限，但大增益时有限样本估计仍可能由尾部主导。

若评分还等于完整 joint/product 的半 log-ratio，则对于几乎处处 anchor，

$$
E_{Z_2\sim P_{Z_2}}e^{2f(z_1,Z_2)}=1.
$$

**标量后验正确，不必满足逐 anchor 条件。** 逐 anchor 条件针对完整比值。两类约束必须分开。常零 $f$ 同时通过全局/anchor 归一化，却可能完全漏掉依赖，所以“通过归一化”不是 estimator 正确的充分条件。

真实图像中的 $f$ 可能只是训练评分器，偏离这些相容式不构成代码错误。不要用该偏差强制校正训练，也不要按它选择 encoder。

# 4. 低维 Gram 几何中的拟合问题

取有限共同观测域，$P_{ij},Q_{ij}>0$，对称 $P,Q$ 与同一单位向量表示 $z_i$。令 $G_{ij}=z_i^Tz_j$，则

$$
G\succeq0,\quad G_{ii}=1,\quad \operatorname{rank}G\leq d.
$$

原始变分目标等价于

$$
\min_G\sum_{ij}M_{ij}
\left[\tanh(a(G_{ij}-\kappa))-\eta_{ij}\right]^2
$$

在上述可实现集合中的最小化。其逐元素理想目标为

$$
G^*_{ij}=\kappa+\frac{1}{2a}\log\frac{P_{ij}}{Q_{ij}}.
$$

$G^*$ 可能不是 PSD，也可能对角线不是 1、元素超出余弦范围。不能通过简单截特征值就称作原约束问题的精确解。toy 中优化单位向量只给可行上界，多起点不能证明全局最优。

若真实与模型 logit 均在 $[-B,B]$，由中值定理

$$
a^2\operatorname{sech}^4(B)\|G-G^*\|_M^2
\leq S-J(G)\leq a^2\|G-G^*\|_M^2.
$$

大 $B$ 下下界可能极弱，不可用来无条件证明几何误差和估计误差等价。若两视图域不一致，应用一般双塔 cross-Gram，而不是套用单一对称 PSD 约束。

在连续图片域上，相同视图的对角事件可为零测度；toy 的有限对角约束仅说明有限模型，不直接证明真实图片无法拟合。基于样本正负标签构成的 0/1 配对矩阵也不是总体 $P,Q$，不得把其 log-ratio 当真值。

# 5. 后验核与潜语义：一个受控模型，而不是现实假设

当两视图在给定潜变量 $Y$ 后独立，

$$
P(u,v)=\sum_y\pi_y p(u\mid y)p(v\mid y),
$$

则 Bayes 公式给出

$$
\frac{P(u,v)}{p(u)p(v)}
=\sum_y\frac{p(y\mid u)p(y\mid v)}{\pi_y}
=\langle q(u),q(v)\rangle,
\quad q(u)_y=\frac{p(y\mid u)}{\sqrt{\pi_y}}.
$$

于是最优 $f=\tfrac12\log\langle q(u),q(v)\rangle$。这解释了配对证据与语义后验几何的一种联系，但真实同图增强可能保留实例、背景等共享变量，$Y$ 不一定是下游类别。log-kernel 一般也不是 PSD。

V6-THEORY 用这个模型核对结构；不立即把训练换成 log-inner-product，也不宣称其为新的普适 SSL 最优评分。

# 6. 梯度作用区间

对于 $A_c=cT-T^2/2$、$T=\tanh f(s)$：

$$
\frac{\partial A_c}{\partial s}=(c-T)(1-T^2)f'(s).
$$

仿射情况下，正对幅度在 $T=-1/3$、负对幅度在 $T=1/3$ 最大。因此峰值位置和幅度是

$$
s_+=\kappa-\frac{\log2}{2a},\quad
s_-=\kappa+\frac{\log2}{2a},\quad
\max |\partial A_c/\partial s|=\frac{32a}{27},
$$

前提为相应峰值落在允许相似度区间。这个峰值不是 encoder 参数梯度峰值，后者还乘上归一化与网络 Jacobian，并会在配对求和后抵消。

令 $z_i=r_i/\|r_i\|$、$r_i\neq0$，

$$
\nabla_{r_i}s=\frac{z_j-sz_i}{\|r_i\|}.
$$

若 $s=1$，即使 tanh 导数不小，这条球面切向梯度也可能为零。非零的 scalar gate 不保证 encoder 仍有大梯度。

实际损失是 $-J$，方向与上式取负；正负各自均值的权重需在实现中保留。JS 使用相同 $f$、logistic logit $2f$ 时，单配对损失导数为 $T-c$，VCS 为 $(T-c)(1-T^2)$；两者在 $f=0$ 处匹配。

# 7. 固定端点曲率候选

提出

$$
f(s)=a(s-\kappa)+a\lambda s(1-s),\quad |\lambda|\leq1/4.
$$

它保持 $f(0)=-a\kappa$、$f(1)=a(1-\kappa)$，但一般不保持 $f(\kappa)=0$。实际零点应另记为 $s_0(a,\kappa,\lambda)$。

$$
f'(s)=a[1+\lambda(1-2s)],\quad f''(s)=-2a\lambda.
$$

在 $[-1,1]$ 上，$|1-2s|\leq3$，所以

$$
\frac a4\leq f'(s)\leq\frac{7a}{4}.
$$

这保证相似度排序不改变，改变的是输出刻度与梯度分布。与 affine 相比，它不是额外的密度定义或正则项；只是 $T=\tanh f$ 里的另一个有限函数选择。单调函数保留 scalar 信息不等于它已正确标定，端点相同也不保证训练结果相同。

两种曲率都必须报告，不能根据 EVAL 临时增加更多多项式阶数。统计标定中 $f$ 应逼近半 log-ratio，训练评分中同一形状是否改善下游由完整实验决定。

# 8. 已有结果与理论预测的对应

| 仓库观察 | 理论允许的解释 | 不能直接得出的结论 |
|---|---|---|
| P117 标定收益很小 | 当前 score 可能已丢失很多可读依赖；也含有限标定能力/分箱误差 | 所有一维评分都无用、真实不可恢复份额已精确知道 |
| P118 嵌套模型改善细端拟合 | 函数类包含关系帮助减少两端误差不平衡 | 所有数据处理增量都已准确测量、VCS 普遍优于 JS |
| 固定 affine 的 C10 accuracy 改善 | 训练中的几何刻度限制可能有用 | 更高估计精度导致更高 accuracy |
| P115 低 IoU 梯度份额增大 | 少数配对可能贡献很大更新，可在反事实 f 上检验 | 必须过滤这些配对或按 IoU 加权 |
| P119 迁移排序反转 | 分类可读性、保留信息和任务之间可能有取舍 | 已证明颜色/纹理是迁移收益的原因 |

# 9. 文献归属

主理论引用合作者研究计划中的参考测度构造、最优 critic 和回归 gap。一般损失与 link 的区分参考 Reid & Williamson, *Composite Binary Losses*, JMLR 2010；正配对核与下游预测背景参考 Johnson et al., arXiv:2210.01883 / ICLR 2023。这里列出的投影分解、密度比条件与梯度等式应按具体构造给证明，不声称它们作为一般统计思想首次出现。

服务器不需要先写完新的泛化定理才开展实验。先完成精确可核对的命题、有限模型实验和真实表示诊断，主理论、SSL 实例化与经验机制分别提交。
