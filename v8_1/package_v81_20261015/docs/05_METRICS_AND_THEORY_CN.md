# 05 · 数学对象与统计规范

## 1. 原理论不变，应用不偷换

原工具：

```text
M = (P + Q) / 2
eta = (p - q) / (p + q)
S = E_M[eta²]
J(T) = E_P[T] - E_Q[T] - 0.5 E_P[T²] - 0.5 E_Q[T²]
S - J(T) = E_M[(T - eta)²]
```

这是合作者的基础定义。[T1] 在V8中，每个P_k/Q_k对应人工评分是否达到阈值k，而不是自然图文joint/product；本包不把它重新命名为Shannon MI。精确gap是任意合适T的读数关系，包括BCE拟合后转换的T，不是平方拟合的独占证书。[T2]

V8自然尾概率u_k经过FIT先验pi_k转换：

```text
ell_k = logit(u_k) - logit(pi_k)
T_k   = tanh(ell_k / 2)
mu    = 1 + sum_k u_k  =  sum_{h=1..5} h q_h
```

S/J和经验的mu-MSE、RPS分别报告，不能因全部是平方项就混为同一个目标。

## 2. 本包直接推导：MSE分解

对一个固定候选池，非负权重v_i归一化为sum v_i=1，e_i=mu_i−Hbar_i：

```text
b = sum_i v_i e_i
MSE_pool = sum_i v_i e_i²
bias_part = b²
within_part = sum_i v_i (e_i - b)²
MSE_pool = bias_part + within_part
```

这是精确代数恒等式，不是新方法。普通prompt宏平均使用每池uniform v；若使用别的权重，分解三项必须完全一致。within_part也不是排名指标：正的线性尺度误差能增大它而不改变排序，所以必须并列报告pairwise与regret。

若真实尾概率为u*，则由Cauchy–Schwarz：

```text
(mu - mu*)² = (sum_k (u_k-u*_k))²
             <= 4 sum_k (u_k-u*_k)²
             = 16 mean_k (u_k-u*_k)² .
```

该界针对真实概率误差，不把有限标注投票均值当作已知u*。它也不是排名胜出的保证。

## 3. RPS三种对象

给定当前图像的投票频率y_k，预测尾概率u_k：

```text
random-rater RPS = mean_k [y_k(1-u_k)² + (1-y_k)u_k²]
CDF-fit squared  = mean_k [(u_k-y_k)²]
constant         = mean_k [y_k(1-y_k)]
random-rater RPS = CDF-fit squared + constant
```

constant在相同图像与相同分母下不影响方法比较，但影响绝对值。完整直方图CE = -sum_h hist_h log(q_h)，零概率仅为数值处理而截断，记录epsilon；投票计数3/6不改变每图自然权重。

## 4. 排序、并列与选择

- 相关性：Spearman用average ranks；常数序列或只有一个值返回null并标记。参考工具不产生传统大样本p值。[E1]
- 人类均值同分pair从pair accuracy分母排除；模型同分在有信息pair上记0.5。显式给出informative_pairs和零信息pool数。
- 选图主诊断用明确候选顺序。按模型原始全精度分数取argmax，严格相等才并列；不得把显示为三位小数相等当成真并列。敏感性可用固定阈值形成near-tie集合，但不替换原读数。
- regret = 当前候选池中最高标注均值 − 被选图标注均值。这里的最高者是有限标注参照，不是无噪声总体oracle。
- 阶段D02的pairwise agreement是两模型的顺序一致性，不是对人类顺序的准确率，表名不能混用。
- temperature作用在五类logit上能改变不同图的mu顺序；不要以为它总是单一score的单调变换。本包主诊断使用raw导出，已有校准列另标。

## 5. 统计单位和不确定性

主单位是prompt或已知近重复component；同prompt多个生成器/图片、同一图多个投票不是独立样本。逐seed报告；三种子不足以精确估计训练随机性，因此不把所有图片×种子当成巨大独立样本。

默认每个seed给完整level；方法配对差按共同cluster bootstrap（2000次、seed2718）估计区间。可以对每个cluster先平均三种子的指标再重采样，解释为在这组三种子固定条件下的prompt抽样区间；不等同重新训练的总体不确定性。

参考程序对可加的pool指标实现配对cluster bootstrap；不把“平均各prompt相关系数”称为全局Spearman。全局相关系数区间如需要，须在每次cluster重采样后重建成对行并重新排序，另写测试。

本轮区间属于开发诊断。默认不生成显著性标签或多个未经调整的p值，不以某个95%区间排除0自动批准新方法。既有V8-04/V8-05的正式家族与指标不被本包改变。

## 6. 预期与否定范围

我们不预期某个损失一定最好。假设H-A“只有刻度变化”、H-B“选模指标错位”、H-C“适配损害泛化”、H-D“输入/容量不足”分别收集支持与反对证据。可以多因素并存，也可以证据不足。

任何小修的收益首先归于被改变的组件；若CS与RPS/MSE同样受益，就不能说是CS专属。基础目标与RPC/相对密度比关系沿用已更正材料，不在诊断包中重新认领新颖性。[T2]
