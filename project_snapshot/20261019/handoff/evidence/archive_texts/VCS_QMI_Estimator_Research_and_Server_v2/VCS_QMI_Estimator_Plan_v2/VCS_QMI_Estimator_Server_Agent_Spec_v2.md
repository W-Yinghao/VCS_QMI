---
title: "VCS-QMI：估计器与完整 SSL 实验规范"
subtitle: "服务器文档 v2｜接续现有分支，建立 CS 估计路线的直接证据"
author: "Server Agent Brief"
date: "2026 年 9 月 28 日"
---

# 0. 执行目标与优先级

**目标：完成一套以变分 CS-QMI estimator 为核心、以完整视觉 SSL 为主应用的实验。** 首要比较是神经 VCS 与已有 CS 估计路线；MINE/JS 等作为神经估计参照，SimCLR/VICReg 作为视觉参照。

主工作：

1. 复用现有 estimator benchmark，补齐同目标核对照、误差和成本评价。
2. 完成原 VCS 的完整视觉 SSL，并运行 CS 路线的直接比较。
3. 完成方法×增强强度的公平对照，以及现有 checkpoint 的机制诊断。

辅助工作：有界组合／残差、观测尺度和条件依赖增量。这些工作不成为开展完整 SSL 的前提。P72 只作已有结果诊断，不扩建图文项目。

本文与 `VCS_QMI_Estimator_Research_Plan_v2_CN.md` 配套。原始理论定义来自 [P1]；实验状态和修改依据来自 [S1] 与新增读取的 [R84]；以下新网格、容差、目录和训练组织是**本版实施建议**。已有明确批准的任务继续按其授权执行；本文件本身不触发提交、取消或重启作业。

# 1. 先接续已有分支，不建立平行项目

## 1.1 来源与时点

```text
repository: W-Yinghao/VCS_QMI
v1_reference: 7b7402c
supplement_reference: af7172c
additional_report_reference:
  d3b4d5ffb1539733314c83ab459535202e94c295
additional_report:
  reports/P84_FROZEN_DIAG_REPORT_20260928.md
```

本版在最后一个提交上核对了 P84。没有把所有分支的提交状态等同于实时队列状态。服务器启动前记录实际 HEAD、工作树差异、现有作业和结果文件。

| 分支 | 来源支持的状态 | 本轮接法 |
|---|---|---|
| P68 官方 CIFAR test | 已完成 | 引用完成结果，不重复当作新验证 |
| P69 E 基准 | 补充记录为已批准、当时未提交 | 查实际代码与输出；复用生成器、oracle、六种 estimator |
| P71/P72 错配 | 补充记录 R2/R3 完成 | 保留结果；只读已有 adapter 做诊断 |
| P73 T1 | 补充记录已冻结 | 收取实际输出，补强控制；不重复注册同一任务 |
| P75 标准 SSL | 补充记录已批准，S2 有专门 test 授权 | 保留授权边界；继续实际移植和完整训练 |
| P81/P82 探针 | P84 提及已存在 Gaussian probe | 查实际报告、配置和输出；不另写一套 probe |
| P83/P84 冻结诊断 | 本轮读取 P84 完成报告 | 复用十个 checkpoint 的结果，不重复同一字典与预算 |

若编号或路径在 HEAD 已变化，用文件内容与提交历史确认映射，记录真实路径。不要从本文推断某个尚未发现的 CLI 已存在。

## 1.2 首份交付：衔接清单

输出 `V2_RECONCILIATION.md` 和机器可读清单，逐项列：实际分支/文件、提交、授权来源、数据 manifest、任务状态、已有结果、本轮增量。重新生成的结果保留新文件名，不覆盖旧版报告。

补充中对 S2 官方测试的许可仅用于相应已批准协议，不自动覆盖 S4、噪声实验或所有新数据集。新开发依然使用 fit/selection；最终确认另行冻结并披露官方 CIFAR test 已看过。

# 2. 固定数学契约与共同损失

## 2.1 VCS

$$
P=P_{XY},\quad Q=P_XP_Y,\quad M=\tfrac12(P+Q),\quad
\eta=\frac{p-q}{p+q},\quad T=\tanh f.
$$

$$
\widehat J=
\operatorname{mean}(T_+-\tfrac12T_+^2)
+\operatorname{mean}(-T_--\tfrac12T_-^2),
\qquad L_V=-\widehat J.
$$

正负分开取均值。每个 anchor 的负样本数 K 不改变 P/Q 的总权重。主要目标为 $S=\mathbb E_M\eta^2$；不裁剪负 J，不用 CS 对数变换训练。[P1]

同时计算

$$
R=\tfrac12\operatorname{mean}(1-T_+)^2
+\tfrac12\operatorname{mean}(-1-T_-)^2,
\qquad J=1-R.
$$

## 2.2 匹配后验的 JS

共有 logit f，令 $q=\sigma(2f)=(1+T)/2$。匹配 $L_V=-J$ 在 $f=0$ 的梯度尺度：

$$
L_{JS}=\mathbb E_P\operatorname{softplus}(-2f)
+\mathbb E_Q\operatorname{softplus}(2f).
$$

原生 JS 的变分报告量是 $\log2-\tfrac12L_{JS}$。它上界为 $\log2$，任意有限 critic 的值并非双侧有界。

JS 的训练比例保持 1:1 总权重；另行分析历史 1:K logistic 时，记录先验及其 log K 截距修正，不把先验差异归因于损失函数。

MINE、NWJ、InfoNCE、SMILE 继续使用各自原生目标与输出。隐藏网络能力可匹配，不能把所有输出都强行 tanh。InfoNCE 原生分数没有自动确定的平衡后验；MINE 的加性常数也不能未经处理就用于 posterior MSE。

## 2.3 固定内容与新增分析边界

保持等权 M、原 J、tanh、有实际边缘意义的负样本。主方法不加入 hard-negative selection、类别排除、pair 重加权、额外方差／重建损失或新梯度算法。

同类标签仅用于诊断。条件 nuisance 分析单独定义条件 P/Q；其中允许同值 nuisance 被独立抽到，不能强制二元翻转冒充乘积分布。

# 3. 直接 CS 对照的实施

## 3.1 对照命名与评价范围

| ID | 定义 | 用途 |
|---|---|---|
| VCS-N | 神经 tanh critic，原 J | 原主方法 |
| CS-K-native | 经典固定参考测度核 CS-QMI | 完整 SSL 与自己的目标误差 |
| S-KDE | KDE 构造 $\widehat\eta$ | 同目标 S 的离线估计 |
| S-Kernel | 固定核特征后接可训练 tanh 回归，原 J | 同目标误差与端到端 SSL |
| rLS | RuLSIF $\alpha=1/2$ 对应的原始回归 | 离线同目标基线／一致性检查 |
| rLS-tanh | raw ridge 拟合后经 tanh 包裹与调尺度 | 有界后验候选，不称闭式 tanh 最优解 |

原始 VCS 是论文主方法，组合或噪声版本采用不同后缀。RPC 的平衡特例用于一致性测试，已调优 RPC 可以作为完整方法参照，不能把相同 J 的两个实现当成独立性能优势。经典核 CS 的真值与 S 分开，不能写入同一 RMSE 列。

## 3.2 经典核 CS-QMI

在配对表示上计算有效核 $K_{ij}=k(z_{1i},z_{1j})$、$L_{ij}=l(z_{2i},z_{2j})$。基本 plug-in 统计为

$$
A=\frac1{B^2}\sum_{ij}K_{ij}L_{ij},\quad
B_q=\left(\frac1{B^2}\sum_{ij}K_{ij}\right)
\left(\frac1{B^2}\sum_{ij}L_{ij}\right),
$$

$$
C=\frac1{B^3}\sum_i\left(\sum_jK_{ij}\right)
\left(\sum_jL_{ij}\right),\quad
D_{CS}=\log A+\log B_q-2\log C.
$$

使用与 CS-IB 原方法相符的核/带宽约定，明确该核是积分后的有效核还是原 KDE 核。先做数值与梯度检查：$K=L=\mathbf1\mathbf1^T$ 给 0；$K=L=I$ 给 $\log B$。后者是 plug-in 对角线效应，不把它报告成真实依赖。

密集核可分块计算，不将原生核方法故意限制为低效内存实现。保留原生自动微分路径，不能强行给三个核统计量添加一个定义不清的“负配对 detach”。带宽可选择固定的 FIT 中位距离倍数，具体网格见 §5；运行前记录数值下溢保护及激活频率。

## 3.3 同目标 KDE

对独立 support/query 构造密度模型。可使用联合配对的乘积 Gaussian KDE：

$$
\widehat p(x,y)=\frac1{N}\sum_i k_h(x-x_i)l_b(y-y_i),
$$

$$
\widehat q(x,y)=\left[\frac1{N}\sum_i k_h(x-x_i)\right]
\left[\frac1{N}\sum_i l_b(y-y_i)\right].
$$

所有核都带相应归一化常数，稳定地计算 log-density。取

$$
\widehat\eta=\tanh\!\left[\tfrac12(\log\widehat p-\log\widehat q)\right].
$$

在独立 query 上报告 $E_M\widehat\eta^2$、$J(\widehat\eta)$、posterior MSE。不能把前两者混成一个量。带宽根据非 EVAL 数据上的共同二次风险选择，并另保留方法原生的选择结果作敏感性比较。

S-KDE 首先用于离线估计；不未经定义就把它的所有 density support/query 路径移植成在线 SSL。

## 3.4 同目标核特征回归

首轮采用 RFF 作为可控的核特征近似：

$$
\phi(w)=\sqrt{2/m}\cos(\Omega w+b),\quad
f(w)=\theta^T[\phi(w);1],\quad T=\tanh f.
$$

$w=[z_1;z_2]$；$\Omega,b$ 固定，$\theta$ 按原 J 学习。基础表示可以反传，频率矩阵不学习。RFF 随机 seed、频率尺度、特征数和插入位置全部记录。

建议离线从 $m\in\{256,1024,4096\}$、带宽倍数 $\{0.5,1,2\}$ 选小网格；完整 SSL 冻结一个从开发数据选出的配置。它是同目标的核特征回归参照，不是原生 KDE-CS 的别名。

另外提供小样本精确核或 Nyström 参照，避免把近似误差全部归因于核方法；先测单元成本再决定扩展。

## 3.5 RuLSIF 一致性与 ridge 候选

令 $g=1+T$。验证

$$
L(g)=\tfrac12E_Mg^2-E_Pg=-\tfrac12-\tfrac12J(T).
$$

函数类、截距、正规化和正则一起映射。对 raw 线性 $T=\theta^T\phi$，

$$
\widehat\theta=\tfrac12(\widehat G+\lambda I)^{-1}\widehat d,
\quad G=E_M\phi\phi^T,\quad d=E_P\phi-E_Q\phi.
$$

用稳定求解器，不用显式逆。截距是否受罚须记录。tanh 包裹后用独立选择数据拟合标量；raw 解只作单独参照，有界字典用 wrapped 版本。对两参数 tanh-cosine 的收敛拟合是数值优化，不称“闭式刷新 a,b”。

# 4. 数据角色与统计单位

优先复用已注册 P69/P73/P75/P83 的 manifest。需要添加组合权重时，先审计现有数据角色，再建立兼容的 FIT/TUNE/SELECT/EVAL 映射；不擅自替换正在运行实验的划分。

| 角色 | 允许用途 |
|---|---|
| FIT | 网络、核参数化的拟合；无标签预处理 |
| TUNE | 组合权重、步长等待估参数；已有子协议可对应其 CAL |
| SELECT | 配置与 checkpoint 选择、提前固定的判断 |
| EVAL | 固定模型的最终评价，不反馈选择 |

组合拟合和 checkpoint 选择的具体顺序写入依赖图。交叉拟合在基础观测上划 fold，不能随机拆分共享图像的 pair 行。基础 critic、核 support、product pool、P/Q query 之间的共享关系明确记录。

固定表示分析可以复用已用于 SSL 的图像，结论是新 critic 的泛化；外部 encoder 验证需要对应的未参与开发数据。当前已有 P84 的 EVAL 不重新用于选择字典。

正负共享 anchor 时，bootstrap 按整个配对块；K 个负样本不等于 K 倍独立样本。对同一图像多视图、多层和不同模型做配对比较，全部共用基础身份重采样。

# 5. E 线：接续受控估计，增加直接 CS 对照

## 5.1 复用与新增

复用 P69 的 Gaussian、非高斯和可逆变换代码，以及真实密度和 oracle。先检查 P81/P82 是否已经覆盖 v1 的三条件探针。已有六类神经 estimator 不删除，已完成 probe 不重复。

新增内容限定为：S-KDE、S-Kernel/rLS 对照；posterior MSE 与尺度正确的梯度；无关维度压力测试；实际预算曲线。

建议完整实验分两个轴，不做所有维度的全排列：

| 轴 | 建议网格 | 固定项 |
|---|---|---|
| 无关维度 | 每侧总维度 2/10/20/50/100 | 有效信号维度 2，真实 S 固定 |
| 独立样本量 | N=256/1024/4096/16384 | 选定维度与中等依赖，batch 固定 |
| 优化 batch | B=64/256/1024 | 固定数据池及样本暴露量／更新次数分别评价 |
| 依赖强度 | 使用 P69 已定义阶梯 | 独立样本量与维度固定 |

这些是新增实验的默认建议；P69 已冻结网格优先保留。首次只对两个维度、两个 N 做原 VCS/核对照的小试，测完内存与耗时再提交完整预算。

## 5.2 oracle 与真值

独立噪声维度不得进入 signal log-density ratio。Gaussian 参数 d 表示每侧维度，pair 总维度为 2d。各坐标独立不意味着 S 相加，必须由整体密度比计算 S。

对无均值差、等相关 Gaussian，经典 CS 的独立解析检查为

$$
D_{CS}^{\rm Leb}
=d\log(1-\rho^2/4)-\frac d2\log(1-\rho^2).
$$

该式只用于经典固定测度 CS 的表，不作为 S 的真值。坐标立方变换可能使经典密度不平方可积；此时保留 mixture-S 实验，将经典 CS 的真值比较从该单元移出，并说明原因。

## 5.3 必须保留的输出

每个方法记录原生目标及自身真值、S 的共同二次评价（适用时）、posterior MSE（可确定后验时）、相对 Bayes risk、尾部分位数、数值失败、拟合/评价时间和显存。

跨目标比较使用依赖阶梯排序概率、检验功效和成本；不把相同原始误差阈值强加给不同量纲。$1-S$ 小于 oracle 不确定性时，将相对风险标为不可分辨，保留绝对误差。

通道导数评价固定 critic 参数，梯度穿过新的数据生成器；不反传 critic 的拟合过程。oracle 有限差分采用共同随机数，检查步长和 MC 精度。若参数改变两个边缘，负分布导数也要计算。

# 6. S 线：完整 SSL 的必做实验

## 6.1 不以辅助路线成功作为启动条件

P75 已批准的标准框架移植继续按实际状态运行。完整 SSL 用原始 estimator；组合、噪声和增量分析未完成不阻断它。

将完整视觉证据分成：

- **S-CS：直接 CS 路线比较。** VCS-N、S-Kernel、CS-K-native。
- **S-context：常规视觉参照。** SimCLR、VICReg 与必要的 JS 控制。
- **S4：方法×增强强度。** 见 §7。

完整方法表按 native 训练语义；仅 S-Kernel 与 VCS-N 的同目标对照强制匹配配对和 detach。CS-K-native 保留自己的核统计图，不伪造“相同负分支”的声明。

## 6.2 实现对齐

当前 CIFAR 基准：ResNet-18 CIFAR stem，512 维 h，512→512→128 projector，余弦–tanh critic，K=8 非零循环移位，四视图全六组配对，负配对右侧 detach，AdamW 与既有 800-epoch 日程。以实际已完成 YAML 为准，不凭文字重建一个近似版本。

共有组件和各自差异在配置里分开。核特征参照至少保留一个与神经参照相同总可训练预算的版本；原生核计算可分块。所有方法记录实际 encoder 前向次数、pair 次数、step、wall-clock、显存。`1×` 标签不是实际 FLOPs 或时间。

## 6.3 完整训练与外部确认

| 场景 | 本版安排 |
|---|---|
| CIFAR-10 既有基准 | 完整原方法结果直接复用；新增 CS 对照从头训练，不把短跑填入主表 |
| P75 标准协议 | 保留其已批准框架、优化器和完整训练时长，三种子；映射新基线须独立记录 |
| CIFAR-100 | 建议作为下一组独立视觉验证；从头训练、三种子、完整预定日程 |
| ImageNet-100 | 资源允许后的规模确认；固定类别列表，预算测量后单独提交 |

不能把“完整 SSL”写成自动运行所有数据集。提交每组的预计成本和明确配置后执行，但研究计划中的全量训练承诺不能退化为只有20-epoch探针。

统一使用 projector 之前的 h 做冻结线性评价，kNN 辅助。记录 probe 的实际训练数据、标准化、checkpoint、超参数和种子。原 test 已经查看，后续确认明确标注研究阶段，不重述为首次盲测。

# 7. S4：强增强公平对照

## 7.1 四个主单元

| 方法 | 标准增强 | 强增强 |
|---|---|---|
| VCS 原配置 | 三种子，能完全匹配则复用 | 三种子，已有精确匹配的 seed 可复用 |
| SimCLR | 三种子，能完全匹配则复用 | 三种子 |

四视图、800 epochs、B、共享架构与各方法已冻结超参数保持一致。强增强使用已完成 VCS 配置的真实 YAML（补充提到 crop 最小面积 0.08、jitter 0.8），复制整套操作、概率和顺序，不只改两个文字数字。

以相同 seed 的增强差值比较，计算

$$
\Delta_{\rm int}=(Acc_{V,s}-Acc_{V,b})-(Acc_{C,s}-Acc_{C,b}).
$$

同时报告每方法绝对结果和提升，不只给差中差。三种子提供描述性误差；有限种子不要配上夸大精确度的显著性语言。

## 7.2 机制记录

在预定早/中/晚时点，记录 J、正负 score、实际 $1-T^2$、残差、输入梯度、余弦中位数、h 的 alignment/uniformity 和有效秩。$1-J$ 不是实际门控的替代品。

SimCLR 用相同样本记录原生负权重与输入梯度；核 CS 对照若后续加入强增强，继续保持同一变换。使用标准控制的训练设置不等于强制相同最佳温度；本轮不重新为每种增强开展不对等搜索。

S4 默认以开发/selection 为新比较终点。官方 test 的新访问需覆盖本单元的明确授权；P75 的许可不自动移用。跨 CIFAR-100 的确认用于检验该交互能否重复。

# 8. D 线：现有模型的低成本机制诊断

## 8.1 P72 优先

从现有 P72 checkpoint inventory 提取 VCS、JS、InfoNCE 和 logistic，优先 $m=0,0.4$ 的早/末期（有哪个用哪个，不补训）。保留原训练的主题配对定义，并将与评价任务不一致的问题写在结果解释中。

按注入/干净配对拆分：

| 指标 | 解释 |
|---|---|
| score、残差、实际 gate 分布 | 不同样本在什么工作区间 |
| 每 pair 对 logit 与表示的梯度 | 统一尺度下的局部影响 |
| 学习率、步数、梯度累计与有效更新 | 累积漂移的实际来源 |
| 初始与最终特征、adapter 参数位移 | 整体漂移 |

分析采用两种视角：同一冻结 score 与 pair 上替换损失的局部比较；各方法真实 checkpoint 上的原生比较。二者结果分开，不将局部恒等式直接解释为两条不同训练轨迹的因果。

JS/logistic 读取原生 logit、类别比例和截距；必要先验修正只用于指定的概率解释，不改变训练记录。

## 8.2 SSL 同类负配对

先复用已有 VCS、SimCLR 同阶段 checkpoint。在完全相同的基础图像和负候选上，计算：同类负对率、score、gate、每 anchor 的同类梯度绝对量和份额。公式可用

$$
F_{same}(i)=\frac{\sum_{j:Y_j=Y_i}\|g_{ij}\|}
{\sum_{j\ne i}\|g_{ij}\|+\epsilon}.
$$

分子和分母都报告；按图像对 bootstrap。范数和不能替代梯度向量相加后的总更新，另记录同/异类梯度与正项的方向关系，说明是否存在抵消。

SimCLR 使用真实分母中的负 softmax 权重。若另用 K 匹配候选，明确是诊断式，而非替代其原生训练损失。标签只拆分报告，不筛除负样本。

## 8.3 P84 复用及有限补充

直接读取已完成的十个 checkpoint 结果。补充只包括：实际 gate、字典分歧、收敛 cosine 的拟合轨迹以及与原 critic 相同单位的 score/Jacobian。P84 的 2,000-step C0 与收敛 C0 不混写成同一最优函数类。

拟合较陡的 critic 可提高冻结 J，但不自动提高 SSL。若提出在线刷新，先以当前固定 checkpoint 比较 refit 前后实际表示梯度，再提出单独的有限训练实验。

# 9. 辅助 C 线：有界组合／残差

## 9.1 复用成员，先看上界

字典包含已有基础 tanh critics，可增加实际收敛的 cosine 和 P48 的 tanh-ridge。raw ridge 单独列出，不进入有界组合。选择与收敛条件只用非 EVAL 数据。

收集 TUNE 上成员输出并计算 $G=E_MT T^T$。固定权重的分歧为

$$
D(w)=\operatorname{diag}(G)^Tw-w^TGw.
$$

它是该权重组合相对最佳单模型收益的上界。筛查所有权重时可解 $\max_{w\in\Delta_m}D(w)$；也可采用无需再次优化的上界

$$
U_{dict}=\min\left\{
\frac14E_M(\max_jT_j-\min_jT_j)^2,
\frac12(1-1/m)\max_{jk}E_M(T_j-T_k)^2
\right\}.
$$

该上界限于固定字典与当前经验分布。先报告它和 MC 不确定性，再决定是否新增候选。补充中的 0.01/0.03 是研究预测，不是隐藏的删结果阈值。新残差候选会改变字典，不受旧字典上界约束。

## 9.2 求解与对照

VCS 组合解 $\max_w d^Tw-w^TGw$，$w\ge0,\sum w=1$。输出真实求解残差、目标和时间。即使没有独立性能收益，QP 本身的成本比较仍可报告。

JS 采用概率混合 $q_w=\sum_jw_jq_j$，优化平衡 log loss；稳定计算 log 概率，使用可靠的凸优化实现。不要把它描述成不可求解或非凸问题。

残差候选默认 $\lambda_0=0.5$，优化原 $J((1-\lambda_0)T_0+\lambda_0U)$。等价伪目标 r 在 $[-3,3]$，不裁剪 r。候选训练后在 TUNE 计算 A/B 的精确线段步长；用 SELECT 做固定方案选择，EVAL 最终评价。

只有新增收敛成员／残差在独立误差或成本上出现有意义的改善，才扩展额外种子与在线应用。P84 已测试的 mix/residual 不重复。这一条件仅限制辅助 C 线，不限制完整 SSL 主线。

# 10. 辅助 O 线：观测噪声标度与标定

## 10.1 数学与字段

单位向量上用 $u=z+\tau\epsilon/\sqrt d$；每坐标标准差为 $\sigma=\tau/\sqrt d$。另一端为 $v=w+\tau\epsilon_2/\sqrt d$。给定单位 z,w，两端独立噪声下

$$
\operatorname{Var}(u^Tv-z^Tw)=\frac{2\tau^2+\tau^4}{d}.
$$

必须记录 `noise_total_rms`、`noise_coordinate_sd`、d 和协方差。Gaussian 原始坐标的方差为1时，优先按每坐标 σ 解释，并核对 $\rho'=\rho/(1+\sigma^2)$。Gaussian 噪声实验只是 oracle 一致性检查。

## 10.2 FIT 内标定，不把拟合分数写成真值

拟合侧分为 `FIT-CRITIC` 与 `FIT-CAL`。固定一个已说明函数类和预算的校准 critic；闭式 raw ridge 加 tanh 的版本在这里是求解近似，不是真实 oracle。

目标刻度为 $J_{cal}\in\{0.95,0.85,0.70,0.50\}$。这是固定代理分数刻度，记录为 `calibration_target=J_proxy`。先用非 EVAL 数据扫 $\tau\in\{0,0.1,0.3,0.6,1.0,1.5,2.0\}$，在可达区间做固定次数细化。有限拟合可能不单调，保留原始曲线，选择最接近目标的候选，不强制二分法或事后单调化。

预先限定每目标容差与最多试验次数，例如绝对 J 距离0.03、最多10次细化；若不可达则记为不可达，不看 EVAL 扩围。所有 estimator 共享最终冻结的 τ，不各自用不同噪声宣称同目标误差优势。

## 10.3 子空间与评价

各向同性与 PCA 子空间分别报告。PCA 只用 FIT 无标签表示确定并固定，可用90%解释方差且上限64维的默认规则；实际秩写入 manifest。子空间总 RMS 使用 $\tau/\sqrt r$。整个扫描固定基底，不随 τ 或 EVAL 改变。

噪声后不再归一化；只看点积的 critic 测量的是受限 score 空间，不能标作完整 noisy pair 的真实 S。对新参数化的每个目标分别拟合/评价 critic。

先报告冻表示上的分辨率、误差或重拟合收益，以及类别/nuisance 信息保留。没有这一层结果前不默认将噪声加入主 SSL。在线噪声实验属于后续新观测目标 $S_\sigma$，不是更准确估计 $S_0$。

# 11. 辅助 T 线：条件依赖与增量分析

## 11.1 与 P73 衔接

接收 P73 实际结果；控制包括逐类带宽 HSIC、学习核 HSIC、连续分数 C2ST 和匹配 JS。每次独立重复重新抽 N，所有拟合/核参数先于 EVAL 固定。

功效相近时不宣布二次结构失去分析价值，也不依据 EVAL 功效选择幅度 critic。幅度研究按独立回归风险选择，并保留 VCS、JS 各自拟合后代入共同平方评价的结果。

## 11.2 同一条件实验和真实嵌套

$$
P=P_YP_{H,N|Y},\quad Q=P_YP_{H|Y}P_{N|Y}.
$$

完整 B=(H,N,Y)，压缩 A=(F(H),N,Y)，F 为确定性 logits。critic 显式接收 Y，或每类拟合再按固定 $P_Y$ 汇总。负 N 在类别内独立抽样，允许抽到相同值。

逐层分析必须使用确定前馈链上的完整状态或确有确定映射的特征。两个层各自做 global pooling 后，一般不再严格嵌套；这种情况下改为描述性依赖 profile，不引用正交增量定理。

## 11.3 输出与有信息量的自检

报告

$$
\Delta_J=J_B-J_A,\qquad
D_T=E_M(T_B-T_A)^2,
$$

$$
r_{BA}=\Delta_J-D_T
=2E_M[(C-T_B)(T_B-T_A)].
$$

差值 J 的望远镜求和是代数必然，不作为自检。对完整→压缩的链 $T_0,T_1,\ldots,T_L$，令 $d_l=T_l-T_{l+1}$，报告

$$
R_{orth}=\sum_lE_Md_l^2-E_M(T_0-T_L)^2
=-2\sum_{l<k}E_Md_ld_k.
$$

oracle 的 Rorth 为零；有限估计的系统偏差提示拟合或嵌套条件需检查。用同一基础图像的 paired bootstrap 给区间，同时区分固定 critic 的采样误差与重新拟合的重复误差。多个层/属性同时检查时报告多重比较处理。

$\Delta=0$ 的检验不自动继承条件独立检验的 permutation p值。正的平方差也可能由两个 critic 误差产生。用离散粗化 oracle 与可控 nuisance 干预验证，再报告真实层间结果。

# 12. 本轮代码与数值测试

复用既有 `reference/` 和 P69/P81/P83 的实现。新增文件优先放在既有 estimator 模块；在不知道实际路径前，不创建一套同名平行包。

本包 `support/checks_v2.py` 只作数学/尺度回归测试，不含完整数据生成、训练或调度。服务器整合时至少核对以下项目：

| 测试 | 检查内容 |
|---|---|
| 后验与风险 | J=1−R；VCS/JS 初始梯度尺度匹配 |
| 同目标 kernel | 独立 support/query；S-plug 与 J 两个字段 |
| 原生 kernel-CS | 全1与单位核极端，自动微分，带宽约定 |
| Gaussian | 噪声相关性变换；无关维度下 S 不变 |
| 组合 | 精确增益、固定字典上界、simplex 权重 |
| residual | 原 J 与伪目标回归的等价；lambda边界 |
| 噪声 | 总RMS/坐标尺度换算，点积方差含四次项 |
| 嵌套 | oracle增量、有限残差式、平方增量正交检查 |
| 统计单位 | 图像身份/基础样本角色互斥、配对块bootstrap |
| 数据权限 | test授权限定到具体协议，已完成单元不重复 |

精确代数采用 float64 数值容差；MC 测试以预测标准误和预设容差判断，不要求随机结果逐位重现别人的表格。所有核和神经路径都做 PyTorch 梯度检查；新代码只读原 checkpoint 的副本，不更新其中 BN 状态。

# 13. 执行次序、交付和扩展范围

## 13.1 推荐顺序

**第一批：接续与补缺。** 输出衔接清单；复用 P69/P81/P82 数据与 P84 诊断；增加直接 kernel-CS 与同目标 kernel-S 的数值测试；读取 P72 的已有 adapter 做机制分解；提交 S-CS/S4 的精确配置和预计成本。

**第二批：主证据。** 继续已批准的 P69/P73/P75；完成直接 CS 估计基准；按确认后的配置运行完整 SSL 的 CS 对照与 S4，不依赖辅助方法成功。

**第三批：独立视觉确认。** 将冻结配置放到事先确定的第二视觉数据集，完成多种子全量训练与成本记录。

**辅助批：有需要才做。** 收敛字典的分歧/组合、FIT标定噪声曲线、条件增量。先做低成本现有数据分析；在线修改和新增任务另行确认。

本文不授权中断既有任务、不修改既有门槛与评估结果，不自动提交完整图文、EEG、配准或新队列项目。

## 13.2 每个单元交付

配置、commit、来源与身份hash；方法原生输出和共同评价；每seed原始JSON/CSV；kernel支持集与带宽、critic预算与停止轨迹；时间/显存/样本与pair计数；与假设直接相关的一页结论。

主要结果结构如下：

\Needspace{25\baselineskip}

```yaml
experiment_family: direct_cs_or_full_ssl_or_diagnostic
protocol_id: <actual immutable id>
source_commit: <actual training/evaluation commit>
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural_or_kernel_cs_or_kernel_S_or_other
estimand: S_or_native_CS_or_JS_or_InfoNCE_B_or_S_sigma
evaluation_readout: native_or_J_common_or_S_plugin
loss_scale: <explicit factor>
reference_measure: mixture_equal_or_native_base
critic_class: <actual class and output>
gradient_routing: full_or_negative_right_detach_or_native
n_independent_units: <count>
n_positive_pairs: <count>
n_negative_pairs: <count>
split_manifest_hash: <hash>
noise_target_kind: none_or_J_proxy
noise_tau: <value if used>
noise_sigma_coordinate: <value if used>
fit_seconds: <measured>
evaluation_seconds: <measured>
status: <actual status>
```

E表同时列目标误差与计算；SSL表同时列完整预算与准确率；机制表按相同基础样本做对照；补充方案的未执行状态只在工作报告中出现，不填成0分或伪造论文结果。

# 14. 文档来源与修改归属

[P1] 合作者《Variational CS-QMI – Research Plan》，原目标、tanh、回归gap和CS参考测度。

[S1] `inputs/VCS_QMI_Estimator_Package_v1_Supplement.md`，用户补充原件，特别是 §§1、3.6–3.7、5。本文采纳其新结果、原有分支复用、完整神经参照、强增强公平比较、噪声标度与二次分析方向。

[R84] [P84 冻结表示诊断报告](https://github.com/W-Yinghao/VCS_QMI/blob/d3b4d5ffb1539733314c83ab459535202e94c295/reports/P84_FROZEN_DIAG_REPORT_20260928.md)，本轮另行读取；它更新了 v1 中“尚需运行冻结组合诊断”的状态。

完整文献见研究稿 §12。补充中的独占性、门控换算、望远镜检查与噪声公式，在研究稿 §§6、8–9、11 中逐项给出本版限定与推导，未作为补充的原结论照抄。

本文新增的实施默认包括：同目标核参照、无关维度网格、噪声代理分数标定细节、真正有信息量的增量残差，以及主/辅实验的重新排序。它们是本版建议，不是已完成的服务器实验。
