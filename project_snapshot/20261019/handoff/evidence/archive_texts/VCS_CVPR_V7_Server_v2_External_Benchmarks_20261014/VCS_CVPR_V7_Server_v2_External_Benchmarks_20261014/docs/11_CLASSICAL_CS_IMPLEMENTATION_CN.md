# 传统 CS / 密度比：具体怎么做

这里的A0是原生经典CS，A1是任务适配后的共同S。二者共享部分KDE计算，但不是同一个统计对象。下列公式是原有方法及直接数值实现，不认领为V7新理论。[X01–X04,F01]

## 1. 原生固定参考 CS-KDE
令归一化Gaussian核为phi_h，密度估计为 p_hat(z)=sum_i a_i phi_hP(z-x_i)、q_hat(z)=sum_j b_j phi_hQ(z-y_j)，a/b各自和为1。

由Gaussian卷积：
\[
A=\sum_{ii'}a_i a_{i'}\phi_{\sqrt2 h_P}(x_i-x_{i'}),\quad
B=\sum_{jj'}b_j b_{j'}\phi_{\sqrt2 h_Q}(y_j-y_{j'}),
\]
\[
C=\sum_{ij}a_i b_j\phi_{\sqrt{h_P^2+h_Q^2}}(x_i-y_j),
\quad \widehat D_{CS}^{Leb}=-2\log C+\log A+\log B.
\]

这是**拟合密度的CS**，含自身pair对角；不要称为无偏总体估计。若改成去对角估计，必须另列估计器及可能的负信息势处理。高维用logsumexp、块计算，保留Gaussian归一化常数；hP不等于hQ时常数不能假设抵消。

对已知Gaussian密度，积分∫pq=N(muP;muQ,SigmaP+SigmaQ)，自积分取2SigmaP/2SigmaQ，可计算本征真值。不同常数约定必须统一后再比较。强分离造成接近零overlap时报告log-domain结果与失败，不使用任意epsilon假装有稳定真值。

联合—边缘乘积Q可由独立池采样估计，也可按原论文的因子化信息势计算；这两个实现必须标记，不把简化采样的结果称作所有经典CS方法上限。原生经典CS仍是正式A0结果，即使它没有逐图概率输出。

## 2. KDE 后验 → 共同S与选图
同一输入表示下用稳定log密度差 ell=log p_hat - log q_hat：
\[
T=\tanh(\ell/2),\quad d_M=\sigma(\ell),\quad
p_{nat}=\sigma(\ell+\operatorname{logit}\pi).
\]

在独立评价池读取J、在语义标注上读取自然Brier。此行名为 **KDE-posterior + common-S (V7 adaptation)**，不是原始CS论文的任务结果。KDE密度中心只来自FIT；任何标准化/降维均按共同FIT变换执行。

EvalMuse的soft投票可给每个观测分配两组质量：a_i∝w_i y_i，b_i∝w_i(1-y_i)。每组单独归一化；一条存在分歧的观测同时出现在两组是对标注者实验的积分表示，不是复制出独立样本。不得把y阈值化后冒称同一事件。

## 3. RuLSIF（alpha=1/2）
\[
u_\alpha(z)=\frac{p(z)}{\alpha p(z)+(1-\alpha)q(z)},\quad
\min_u\ \tfrac12E_{M_\alpha}u^2-E_Pu.
\]
取u=Phi theta，H=alpha E_P(PhiPhi^T)+(1-alpha)E_Q(PhiPhi^T)，h=E_PPhi：
\[
\hat\theta=(H+\lambda I)^{-1}h.
\]

原论文的相对比、非负处理与CV路径应保留原生结果。[X03] alpha=.5时eta=u-1。有限拟合可能u<0或u>2：同时保存raw u与供Bernoulli概率用的[0,2]投影；投影不是原论文自动保证。

对0<alpha<1，p/q=(1-alpha)u/(1-alpha*u)在0<=u<1/alpha时给出有限密度比；u=1/alpha对应可能的无穷比值，应记录边界而非数值硬除。alpha=0时u本身就是p/q，即普通uLSIF，[X04]但其原生Pearson目标不等于S。alpha=1退化为p/p，不能使用这个逆映射恢复p/q。不要将任何加tanh的rLS配置冒称为完全原生RuLSIF。

## 4. RKHS平方回归
在统一特征空间估计eta：
\[
\min_T\ \tfrac12E_P(1-T)^2+\tfrac12E_Q(-1-T)^2+\lambda\|T\|_\mathcal H^2.
\]
可用Nyström近似的白化核特征求解。若采用普通基函数系数ridge，明确命名“kernel-basis ridge”；该改变可能影响容量与正则，必须给出原生/近似差异。输出投影到[-1,1]只用于共同有界后验读数，保留raw与clip比例。

## 5. 条件CS、CS-IB和CS-Aligner的位置
条件CS的TPAMI方法用KDE比较条件分布；CS-IB在回归信息瓶颈中同时处理压缩与预测；CS-Aligner比较图像和文本边缘分布并结合配对学习。[X01,X02,X11]

这些方法必须出现在相关工作/方法对应表，但不能把它们的批次散度直接当“图像满足提示的概率”。有已完成的适用旧实验时复用原生任务结果；否则V7仅做直接可实现的KDE/RuLSIF接口，额外训练型CS-IB/CS-Aligner任务适配需单独协议，不能伪装成原方法复现。

## 6. 实际拟合要求
提供FIT/CAL的native核风险与共同Brier两套选模记录；主语义接口按共同Brier选，原生复现按作者CV另列。记录是否位于网格边界、condition number、线性求解残差、clip比例、支持向量/中心数、P/Q有效质量。不能用“不可用”掩盖未尝试合适带宽，也不能为填行把OOM设置记作准确率0。
