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
