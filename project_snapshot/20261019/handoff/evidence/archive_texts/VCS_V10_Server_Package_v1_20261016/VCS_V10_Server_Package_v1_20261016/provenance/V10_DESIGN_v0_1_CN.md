# V10 设计稿 v0.1：Moment-Factorized VCS

日期：2026-10-16。状态：**候选方法设计与 CPU 数学参考已完成；尚无自然数据性能、GPU计时或新服务器训练。**

## 0. 决定与一句话主线

保留合作者的混合参考 S、原始 J 和最优 critic；重新设计实际 critic，使二次目标可以用完整负对的矩运算求值，并使固定特征下的关系头成为可检验求解误差的凸问题。

工作名 **Moment-Factorized VCS（MF-VCS）** 仅用于讨论，不是已确认新颖性的论文方法名。

**主张候选：在相同数据、计算与内存预算下，以结构化的有界 critic 更充分地利用配对数据，降低负对抽样误差和内层求解误差，最终改善误差—成本曲线。**是否改善总体估计或视觉任务必须另行验证。

V8/V8.1 保持收尾；V9-N2/G1/G2 不替换、不重算、不抢占。V10 是方法级设计，不是 V8 的新损失、V9 的新噪声率或另一个标定任务。本稿不提交 GitHub、不授权未知预算、不启动 GPU 网格。

## 1. 起点：已知证据与本轮假说

本次 GitHub 截点：main df4ced4；V8 cacb906；V8.1 260752c；V9 27117c9。依据 V81_08、C1 以及 G1 的实际训练代码。

| 来源中支持的结论 | V10 的处理 |
|---|---|
| V8 的损失变化与换 seed 对选图的影响在相近量级；C1 部分适配没有改善 | 不再以相同宿主换损失作为主要研究 |
| 旧材料已把残差拟合、字典组合联系到 L2 boosting / stacking | 不把同一套路改名为 V10 |
| 原始 J 与平衡 RPC 精确重合；RuLSIF 也是近邻 | 研究新增的表示、求解与配对计算，不重新认领泛函 |
| V9-G1 采用局部图像表示、图内随机 permutation 负对，InfoNCE 显式构建全部候选 | 已有真实接口可以检验矩计算，而不是先搭新大型 VLM |
| P157/P159 的有限 critic 结果曾提示拟合规则可能存在区制差异 | 仅作为研究动机；新参数化没有原输出 tanh 的 redescending 梯度，不继承其鲁棒性结论 |

本轮假说：原二次目标的计算结构尚未被一般 concat-MLP+tanh critic 充分利用。此假说**不是**已经查明的 V8 失败根因。

## 2. 不改变的理论对象

M=(P+Q)/2，eta=(p-q)/(p+q)，S=E_M eta^2。

\[
J(T)=E_PT-E_QT-\tfrac12E_PT^2-\tfrac12E_QT^2,
\quad S-J(T)=E_M(T-\eta)^2.
\]

D_CS^(M)=2 log((1+S)/(1-S)) 仍是原来的变换，不把 S 称为 Shannon MI。

下文计算主例是 P=P_XY、Q=P_X P_Y。一般 P/Q 的 J 仍成立，但只有 Q 的乘积/可分结构才能使用对应的快速矩公式；一般负分布没有免费的分解。

原 T=tanh(f) 是一种实现参数化，而非 J 恒等式必须使用的唯一函数形式。本候选将其改为结构性有界；任意内部值 T 仍可写成 tanh(atanh(T))，但不再通过该链式参数化训练。

经验 J 不是逐次样本的总体下界。正对 n 条生成的 n(n-1) 负对也不是 n(n-1) 个独立样本。所有估计性能需要独立基础样本或正确的分组评价。

## 3. 有界、可分离的 critic

两侧特征分别属于概率单纯形：
\[
a_\theta(x)\in\Delta^{r-1},\quad b_\psi(y)\in\Delta^{s-1}.
\]
采用
\[
T_{\theta,\psi,W}(x,y)=a_\theta(x)^T W b_\psi(y),\qquad -1\le W_{kl}\le1.
\]
因为 a_k b_l 非负、且其总和为1，T 是 W 元素的凸组合，所以对所有输入都有 |T|<=1。a、b 是归一化基函数权重，不自动具有语义类别或后验解释。

### 3.1 这提供什么

- 输出有界不再依赖最后一层 tanh；内层 W 可以直接在闭盒内求解。
- 关系是多个两侧特征之间的交互，不是原始 cosine 的两标量校准。
- r,s 控制函数类与计算量；W 无需正定，也不要求对称。它是关系矩阵，不称 PSD kernel。
- 两侧映射仍可以是神经网络，并可训练完整图像编码器。第一轮数学实验冻结特征只为隔离计算，不替代完整应用训练。

### 3.2 不能附带宣称什么

有限 r,s 不保证包含任意 eta。r=s=1 时 T 是常数，最优 J=0，即便真实依赖很强。固定类求解完美与总体估计准确是两回事。

若允许不断细分的可测 x/y 分区，硬单纯形特征可表示矩形网格上的任意 [-1,1] 值；在适当可测空间与分区逼近条件下可讨论 L2 稠密性。有限神经 softmax 网络是否实现这些分区，需要单独的逼近条件，不在此无条件声明。

softmax 特征本身仍可能饱和；此设计只去掉输出 tanh 的链式因子，不承诺所有参数都不存在梯度退化。去掉此因子也可能损失已有 tanh 的抗污染行为。

## 4. 完整乘积负对：精确矩计算

设正对共 n 条，A 的第 i 行是 a_i^T，B 的第 i 行是 b_i^T，t_i=a_i^TWb_i。令
\[
u=\sum_i a_i,\quad v=\sum_i b_i,\quad G_A=A^TA,\quad G_B=B^TB.
\]
注意本节的 u,v 是向量和，不是额外随机变量。

### 4.1 包含对角的经验乘积（V-statistic）
\[
q_1^V={u^TWv\over n^2},\qquad
q_2^V={\operatorname{tr}(W^TG_AWG_B)\over n^2}.
\]
这严格等于显式计算所有 n^2 个 T_ij 及 T_ij^2 再平均，不是近似。

### 4.2 排除原始正对对角的经验乘积（U-statistic）
\[
q_1^U={u^TWv-\sum_i t_i\over n(n-1)},
\]
\[
q_2^U={\operatorname{tr}(W^TG_AWG_B)-\sum_i t_i^2\over n(n-1)}.
\]
\[
\widehat J_U={1\over n}\sum_i(t_i-\tfrac12t_i^2)-q_1^U-\tfrac12q_2^U.
\]
对独立基础正对、固定 critic，非对角 (X_i,Y_j) i!=j 的边际为 P_X P_Y。若 critic 在同一批数据上学得，经验最大值不再无偏；必须独立评估。

### 4.3 独立边缘池与分组条件分布

独立 x 池大小 n_x、y 池大小 n_y 时分母换成 n_x n_y，通常不删除同号索引，因为两池本来独立。

若 Q 条件于场景/图像，逐场景计算 q1/q2，再按原场景权重平均。禁止把它静默改为跨场景乘积。

若 Q 排除空间邻域等，必须减去同一个禁止集合，并采用同一个归一化。均匀 eligible-edge 与每行分别均匀不是一回事。每行 allowed 数为 n-d_i 时，可令 alpha_i=1/[n(n-d_i)]，用加权 G_A 和 u 计算全行项，再减 alpha_i 加权的禁止边。额外成本必须计入；禁止集合稠密或依赖任意模型输出时，优势可能消失。

### 4.4 当前 G1 接入的一个重要事实

本次读取 g1.py：负对由 g.permutation(n_pos) 产生，**允许固定点**；空间半径排除用在污染正对应生成中，不等于负 permutation 的对角排除。

所以复现当前有限目标，应使用包含对角的 V 形式；它等于随机普通 permutation 的条件平均。U 形式应独立命名、给所有对照同样变化。不能为了得到更干净的公式而悄悄改 Q。

### 4.5 计算复杂度

r=s=K 时，对固定 W，一次 J 及特征梯度计算需要约 O(nK^2+K^3)，特征和矩中间量约 O(nK+K^2)。显式全部双线性分数为 O(nK^2+n^2K)。分块 dense 可以降低显存，但仍须计算全部 pair；应作为真实基线。

整个神经系统仍有编码器成本。W 的 t 次内层迭代通常增加约 t 次头部计算；本式不是整个训练的总成本。K 增长、低维吞吐和 GPU kernel 开销均可能抹去理论节省。只报告实测性能，不预测固定倍数。

## 5. 可证明的方差性质：只消除一种随机性

给定固定特征与 W，以及一批数据 D，均匀随机 derangement pi 满足：
\[
E_\pi[\widehat J_\pi\mid D]=\widehat J_U.
\]
因此（在相应固定 critic 与平方可积条件下）
\[
\operatorname{Var}(\widehat J_\pi)
=\operatorname{Var}(\widehat J_U)
+E\operatorname{Var}(\widehat J_\pi\mid D).
\]
普通 permutation 的相同结论对应 V 形式。

这是条件期望/Rao–Blackwell 化，不是新统计定理。它减少的是给定基础数据后的配对抽样方差，不增加独立样本数，也不保证训练后的 estimator MSE 必然更低。若 T 本身根据同一 pi 训练，不能未经处理套用该等式。固定迭代/冻结 critic 的梯度比较才是第一项核对。

## 6. 固定特征下的凸关系头

令 phi=a\otimes b，w=vec(W)，delta=E_P phi-E_Q phi，H=E_P phi phi^T+E_Q phi phi^T，则
\[
J(w)=\delta^Tw-\tfrac12w^THw,\qquad H\succeq0.
\]
因此固定 a,b 后最大化经验 J 是盒约束凹二次优化。

不显式构建 K^2 x K^2 Hessian；使用矩公式与正对收缩计算 HVP。原型可用 projected gradient / FISTA / box-QP。附带 SciPy dense 解只供小维度核对。

令 g=grad_W F(W)，F 为本轮实际求解的目标，则
\[
\mathcal G(W)=\sum_{kl}|g_{kl}|-\langle g,W\rangle,
\quad 0\le\max_{V\in[-1,1]}F(V)-F(W)\le\mathcal G(W).
\]
这是标准 Frank–Wolfe 一阶最优性 gap 在盒约束下的特例，不能认领为新原理。

### 6.1 证书到底覆盖什么

只覆盖当前固定特征、当前经验目标、当前约束内的优化不足。它不证明 S 被准确估计、不证明特征类足够强、不证明任务风险受控。

若增加 ridge F=J-lambda||W||_F^2/2，gap 认证的是 F；必须同时报告未正则的 J 和 ridge 偏差，不偷偷更改 S。第一次数学核对同时检查 lambda=0 和显式正则版本；正式选择规则另定。

### 6.2 两种训练模式

**计算模式 M0（主首发实现）**：保留常规 AdamW 更新 feature maps 和 W，W 每次投影回盒；每步使用精确矩 J。先隔离结构与完整配对效果。

**求解模式 M1（条件增强）**：固定当前 minibatch 的 feature tensors，高精度求解 W；然后使用同一 tensors、固定已解 W 的 J 更新 feature maps。这是 variable projection / profiled optimization 思路。

M1 的 outer objective 必须与 inner objective 同一数据、同一目标，才能使用包络定理直接忽略 dW*/dtheta；若 outer 改成独立验证 J，需要相应隐式微分，不能继续照搬 stop-gradient。

只固定特征内层是凸的；神经特征训练仍然非凸。正则保证唯一性时可进一步分析误差传播；无唯一性时应按 Danskin 的方向导数条件处理。不能把整个模型称为“全局凸训练”。

特征计算中的 dropout、BatchNorm、随机增强状态在同一次内层求解期间固定；禁止缓存过期特征矩后仍声称精确当前目标。W=0 可作内层初值，a/b 不能全部初始化成恒定相同的单纯形向量，否则依赖信号与外层梯度都可能塌缩。

## 7. 为什么这次不只是换损失

| 旧做法 | V10 候选 |
|---|---|
| concat 两侧表示，MLP，tanh | 两侧学习特征，显式双线性关系矩阵，全域有界 |
| 抽几个负对或显式建 N^2 表 | 精确压缩完整乘积负对的一二阶矩 |
| 总体无界于训练误差的非凸黑箱 | 固定特征内层有明确优化 gap；外层仍如实非凸 |
| 主要比较 BCE/CS 哪个应用分高 | 主要比较相同预算的估计误差、完整能量与最终任务 |

旧残差、stacking、nested equality 都不作为 V10 新算法。当前候选组合是否足够新颖，还需逐式对照下节工作；若被已有方法完整覆盖，则降为有用实现，而不是改名继续认领。

## 8. 先行工作与比较矩阵

| 来源 | 已有内容 | V10 必须分清的部分 |
|---|---|---|
| RPC (ICLR 2021) | 基础二次目标、相对参数、最优有界比及实验 | alpha=beta=gamma=1 与本 J 相同；同目标同实现只是一行，不算两个基线 |
| RuLSIF (NeurIPS 2011)、LSMI | 相对密度比、最小二乘与核求解 | 不能把二次头/闭式路线称为首次；用调好的核、适当正则和相同采样比较 |
| Spectral Contrastive (NeurIPS 2021) | 双线性关系、平方负项与谱解释 | 原始 S/J 还含正对平方与负对线性项；不能以“有二阶矩”声称新颖 |
| Fisher GAN (NeurIPS 2017) | 神经特征与二阶矩约束 | 要区分 L2 约束的 IPM 与这里的逐点有界 critic、固定 CS 目标 |
| FedSC (ICML 2024) | 用相关矩阵传递谱对比信息 | “传矩不传所有pair”已经有先例，本轮不另开联邦任务 |
| Variable Projection (TNN 2008 等) | 消去线性输出层、优化非线性特征 | profiled head 是已有思想，新增内容必须在具体结构和有效结果 |
| Frank–Wolfe (ICML 2013) | 一阶 gap 与凸约束求解 | 本文盒约束 gap 是特例 |
| Deep-kernel two-sample tests (ICML 2020) | 学习适应数据的核，而非只调带宽 | 核路线不是只能使用固定弱特征；需要保留强学习型核对照 |
| Spectral conditional independence (ICML 2026) | 学习谱表示、把表示误差联系到检验 | 若后续转条件检验，必须直接对照；不是第一轮主应用 |

本轮检索没有证实“所有论文里绝无该组合”，也没有完成所有论文的逐式全文审查。已核的 RPC 目标与本 J 精确重合；Spectral Contrastive 全文已定位；其余核验深度见 sources.json。

## 9. 应用选择：先接密集跨模态关系/配准，不重开 V8

建议第一自然应用沿用 BF–SHG 的独立 specimen、输入、原始配准器与几何指标，在新 V10 目录中比较新的 learned relation critic。完成各自 full image encoder 的训练及完整配准，不以冻结 probe 替代。EPFL 是第二数据集接续候选，不自动使用 V9 封存测试作 V10 调参。

原 G1 是固定训练 critic，在候选变换上读取经验能量；它不是对每个变换重新求总体最优 S。V10 也明确保持 W 在测试时固定，避免每个变换重新拟合到自身配对，造成错误位置也容易判别。

在配准中，完整矩能量可以在每个变换的对应 descriptor 上计算；继续遵守 warp-then-encode，不能把它偷换成 encode-then-warp 并假设等价。固定位置、固定 permutation 的原方法并没有每一步重抽负对；我们的改动去掉的是对最初 permutation 的依赖，而不是修复一个并不存在的迭代随机采样 bug。

### 一个条件性的估计—几何连接

对固定 critic 的总体能量 F(phi)，若 |Fhat(phi)-F(phi)|<=epsilon 在候选域上同时成立，并且搜索器对 Fhat 的优化残差<=tau，则
\[
F(\phi^*)-F(\hat\phi)\le2\epsilon+\tau.
\]
只有再有几何可识别条件 F(phi*)-F(phi)>=kappa d(phi,phi*)^2，才得到 d<=sqrt((2epsilon+tau)/kappa)。这些是直接的优化误差传播事实，不是无条件配准保证。降低某个固定 phi 的负对方差，不自动给出统一 epsilon，更不保证真配准是 F 的极大点。

外部主对照：原 MLP relation critic、InfoNCE、CoMIR 的适用原生流程、MI/NMI、调好的核/相对密度比路线；MIND 等补充接入按完整协议。MMD/HSIC 只进入相应的分布或独立性实验，不把位置置换下不变的单纯边缘 MMD 冒充配准能量。

## 10. 首轮实验任务

### T00：重复性与新颖性

记录旧 residual / stacking / ridge / nested 方案；逐式映射 RPC/RuLSIF/Spectral，标注哪些只是代数或标准优化工具。结果：method_delta.md。禁止以旧报告未经更正的“所有score不能排序”作动机。

### T01：数学与代码参考

已在本地完成 19 项 CPU 检查。本地也已补普通 permutation 对 V 形式的平均、每行归一化禁止边的一二阶矩和无正则头部求解；服务器进一步核对真实分组/禁止边、特征塌缩、fp32/fp64 消减误差，以及小规模 dense-QP 对 HVP-solver。不能只看 loss 数值相同，还要看特征、W 和几何变换梯度。

### T02：固定模型、固定样本的计算实验

比较 sampled K=1/8、完整 dense（含分块）、精确矩。相同 A/B/W、相同目标和对角规则。N in {128,512,2048,8192}；K in {8,16,32} 是初始设计表，具体范围由代码支持与内存实测填实，不按性能胜负删格。

报告值/梯度误差、配对重抽方差、时间、峰值显存、encoder成本。速度只在等价目标下归因；dense 与 moments 应得到相同预测质量，它们的差别应是计算，不是期望某种神秘精度提升。

### T03：受控真值的误差—预算曲线

复用已有生成器但使用独立的新随机数据：可计算 p/q 的低秩有限表、高频/高秩表、Gaussian/cubic 可逆通道、混合多峰。有限潜在标签真值不能在不可恢复特征上继续冒称观测真值。

主指标：后验 MSE、S 的独立读数偏差/方差/RMSE、同数据重抽负对的误差、内层 gap、总时间/内存。函数类受限误差与 solver误差分开；oracle只评价，不选超参。

主对照：调好的传统 CS-KDE（原生真值另列）、KDE->共同S、RuLSIF、Nyström/ARD、原MLP-VCS、学习型核、MF-VCS。MINE/DV、NWJ、SMILE、InfoNCE按原生目标和共同任务比较，不强行用 S 真值给它们算误差。匹配 BCE 是一项归因，不是唯一门槛。

两种预算表都需要：相同独立数据数；相同总计算/内存资源。完整pair数不写成独立样本数。

### T04：端到端学习与机制分离

最小机制矩阵：
1. 原 MLP+tanh，原随机负对；
2. 原 MLP+tanh，增加负对预算/完整分块；
3. 新 bounded bilinear，随机负对、普通优化；
4. 新 bounded bilinear，完整矩、普通优化（M0）；
5. 新 bounded bilinear，完整矩、profiled求解（M1）；
6. 相关非CS/传统方法的完整对照。

这里 1->3 不仅换计算，还换函数类；3->4 才隔离负对利用；4->5 才隔离内层求解。固定样本 dense 等价只作实现对照，不重复全训练冒充独立方法。

编码器训练完整；保存初始化、逐epoch/checkpoint、优化器与实际步数。学习率网格等信息在真实计时后对称填写，不给新方法无限调参。不以“先赢BCE”作为是否完成其他方法的门槛。

### T05：完整配准与可展示结果

在同一任务、同一搜索器和同一数据划分下画 success/median/p90 error 对总计算成本曲线，报告单幅结果和失败，不只展示训练J。按 specimen/scene统计，不将所有patch/transform当成独立人群。优化误差、配准能量错误与搜索错误分开。

在旧DEV上的新方法比较是开发；V9已读/正在确认的集合不能又被宣布为V10全新独立确认。确认来源在方法选择前单独锁定。无需为设计阶段新申请数据。

## 11. 预期优势与止损依据

| 假说 | 理论已支持的范围 | 完整实验需要补的证据 |
|---|---|---|
| 精确利用全部负对，避免枚举 | 在指定双线性类和Q结构下代数等价 | GPU实际成本；强核/谱方法也得到各自合理加速 |
| 固定模型的配对方差降低 | 条件期望恒等式 | 训练后总MSE与下游风险是否改善 |
| 固定特征求解更充分 | 凸内层及gap | 较小gap是否值得额外时间，是否仍受逼近误差主导 |
| 同预算估计更好 | 尚未证明 | 独立真值误差及强方法完整对照 |
| 配准更有效 | 尚未证明 | 真实几何结果、成本、外部系统 |

若有精确加速但函数类表达不足，保留计算结果并讨论结构，不堆多个新模块。若低维矩方案与现有谱/核实现相同且没有新增效果，则不主张新算法。若M1没有比M0改善成本—误差，只保留M0。若完整应用没有改善，不将仅J改善改称任务改进。

## 12. 当前交付与接续

附带 v10_reference.py、MATH_CHECK_RESULTS.json、sources.json。代码只操作显式传入的数值特征/小型合成矩阵，不访问网络、不读服务器数据、不改仓库、不发作业。

本设计稿可供合作者和server评审实现；真正的服务器全量包还需填实：当前V9结束状态、可复用数据/代码、实际GPU计时、明确比较单元和确认源。本轮已经开始并完成了具体方法设计，不是已经运行自然数据或批准全部大型实验。

## 13. 三种误差不能混成一份证书

固定两侧特征和其 critic 类 F 时，若 sup_F |Jhat-J|<=epsilon，关系头的经验优化不足<=gamma，则标准 ERM 比较给出
\[
S-J(\widehat T)\le\inf_{T\in\mathcal F}E_M(T-\eta)^2+2\epsilon+\gamma.
\]
第一项是逼近误差，第二项是统计误差，第三项是求解误差。本方案的头部 gap 只直接控制第三项；完整负对消除一部分条件配对噪声，不消除基础样本误差。两侧特征一起学习时，不能把头部 gap 当成整个神经函数类的全局优化 gap。

## 14. 本地参考结果

19项检查通过；完整矩与dense目标最大差 8.33e-17，特征/W梯度最大差 6.94e-17。5个样本的44种derangement平均与U目标差2.78e-17；普通120种permutation平均与V目标也通过；可行盒头的正则化求解gap为1.49e-9。

这些只是有限维、float64的独立核对，不是统计样本效率、自然任务优势、GPU加速或新颖性证明。

## 来源登记

**[P0] Collaborator: Variational Cauchy–Schwarz Quadratic Mutual Information, Research Plan** — Conversation-provided PDF, September 2026

核验范围：Equations (1)-(17), original S/J/gap and explicit non-Shannon target; fully provided text

**[P1] Project corrections, October 10, Appendix 4** — Conversation-provided VCS_QMI_Progress_Problems_Literature_NextSteps_20261010.md

核验范围：RPC overlap; nonexclusive gap; corrected noise/prior/ordering claims; not current execution status

**[P2] Supplement to estimator research/server v1** — Conversation-provided supplement dated 2026-09-28

核验范围：Residual=boosting and dictionary=stacking; some exclusive-property wording superseded by P1

**[R0] Current repository branches** — https://api.github.com/repos/W-Yinghao/VCS_QMI/branches?per_page=100

核验范围：Read-only branch snapshot this turn

**[R1] V81_08_PHASE_REPORT** — https://github.com/W-Yinghao/VCS_QMI/blob/260752ceecc669177db7fd41f8d9ee61b9994d61/reports/V8_1_DIAGNOSTICS/V81_08_PHASE_REPORT.md

核验范围：Current closeout/first-section facts, not a new rerun

**[R2] V9-G1 training implementation** — https://github.com/W-Yinghao/VCS_QMI/blob/27117c9ae78d64d39a6548c40b00ff58f70cc4d4/src/v9/reg/g1.py

核验范围：Actual cmd_train, within-scene ordinary permutation; perturbation mechanism distinct

**[R3] Original REG model and energy implementation** — https://github.com/W-Yinghao/VCS_QMI/blob/27117c9ae78d64d39a6548c40b00ff58f70cc4d4/src/v9/reg/r6_reg_pilot.py

核验范围：CNN descriptors, concat-MLP, tanh J, same-image pairing and fixed permutation energy

**[L1] Self-supervised Representation Learning with Relative Predictive Coding** — https://arxiv.org/html/2103.11275v3

核验范围：Primary full text; Eq.2 and optimal critic explicitly checked

**[L2] Relative Density-Ratio Estimation for Robust Distribution Comparison** — https://arxiv.org/abs/1106.4729

核验范围：Primary paper page/full displayed text; relative-density-ratio and least-squares prior art

**[L3] Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss** — https://arxiv.org/html/2106.04156

核验范围：Primary full-text available; spectral/quadratic comparator, not an identical S target

**[L4] Fisher GAN** — https://research.ibm.com/publications/fisher-gan

核验范围：Primary institutional publication abstract; second-moment constrained critic

**[L5] Training Two-Layered Feedforward Networks With Variable Projection Method** — https://doi.org/10.1109/TNN.2007.911739

核验范围：Primary publisher abstract; neural variable projection is established

**[L6] Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization** — https://proceedings.mlr.press/v28/jaggi13.html

核验范围：Primary proceedings page; duality-gap certificates

**[L7] FedSC: Provable Federated Self-supervised Learning with Spectral Contrastive Objective over Non-i.i.d. Data** — https://proceedings.mlr.press/v235/jing24b.html

核验范围：Primary proceedings abstract; correlation-matrix summaries already used

**[L8] Learning Deep Kernels for Non-Parametric Two-Sample Tests** — https://proceedings.mlr.press/v119/liu20m.html

核验范围：Primary proceedings abstract; learned adaptive kernel baseline

**[L9] Toward Scalable and Valid Conditional Independence Testing with Spectral Representations** — https://proceedings.mlr.press/v306/frohlich26a.html

核验范围：Primary 2026 proceedings abstract; relevant only if extending to CI testing; no claimed replication

