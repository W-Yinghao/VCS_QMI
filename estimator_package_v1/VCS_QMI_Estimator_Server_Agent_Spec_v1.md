---
title: "VCS-QMI：估计器优化的服务器执行规范"
subtitle: "误差诊断、有界组合、观测尺度与条件依赖增量"
author: "Server Agent Brief · v1"
date: "2026 年 9 月 28 日"
---

# 0. 本轮要完成的工作

本轮优化 **estimator**，不继续展开一般性的 CIFAR 超参数搜索。理论依据仍是合作者的等权混合参考测度、原始二次目标和 tanh 有界 critic。

执行顺序：

> **建立误差测量 → 诊断现有 critic → 拟合有界组合和残差候选 → 测量观测尺度曲线 → 将有效改进用于 SSL 和条件依赖增量。**

配套研究稿：`VCS_QMI_Estimator_Research_Plan_v1_CN.md`。

本文件分为两种内容：

- **继承定义**：来自合作者原计划和上一轮研究分析。
- **实施默认**：本文件为了可执行而补充的数据量、网络、划分、优化步骤和交付格式。这些是起始配置，不是理论要求，也不是已经验证的最佳配置。

**收到明确的“按本文执行”指令后，首批只完成阶段 0–2：代码核对、数值测试、三个 Gaussian 条件的 seed-0 小试和可用 checkpoint 的诊断。提交结果后再确定重复种子、观测尺度和在线训练。** 本文件本身不启动服务器作业。

# 1. 固定内容与可改内容

## 1.1 固定的理论与目标

$$
P=P_{XY},\quad Q=P_XP_Y,\quad M=\frac12(P+Q),
$$

$$
S=\mathbb E_M[\eta^2],\quad
\eta=\frac{p-q}{p+q},\quad
T_\phi=\tanh f_\phi,
$$

$$
J(T)=\mathbb E_PT-\mathbb E_QT
-\frac12\mathbb E_PT^2-\frac12\mathbb E_QT^2,
\qquad \mathcal L_V=-J.
$$

正负样本各自取均值，不能通过负样本数量改变两类的总权重。训练不使用 CS 对数变换，不裁剪负的 $J$。[P1, §§3–7, 9–10]

## 1.2 本轮的新实现

| 项目 | 本轮处理 |
|---|---|
| 基础 tanh critic 的函数类与优化 | 可以在本文列出的有限候选中比较 |
| 有界凸组合与残差更新 | 本轮的新拟合方案，单独命名和记录 |
| 独立高斯观测误差 | 明确定义新的 $S_\sigma$，先冻结表示研究 |
| 条件 nuisance 分析 | 明确条件 $P/Q$，只用于分析任务 |
| SSL 原配置 | 保留为基准；在线改动后续单独确认 |
| hard negatives、pair 重加权、额外正则 | 不加入首批实现 |
| 修改 $M$ 权重、替换 tanh、梯度估计新算法 | 不属于本轮任务 |
| 队列、EMA、更多视图或更长 schedule 搜索 | 不重复开展 |

有界组合不是对样本加权。组合权重作用于同一配对上的多个 critic 输出。

# 2. 先核对已有代码和资料

上一轮研究依据的快照为：

```text
repository: W-Yinghao/VCS_QMI
reference_commit: 7b7402c05161c33d77a4301f6efc27bb55420e6a
```

服务器曾报告以下位置，运行前逐项验证，不直接假定仍然有效：

```text
repo:    /home/infres/yinwang/CS_QMI/ssl_pilot
outputs: /home/infres/yinwang/CS_QMI/outputs
env:     /home/infres/yinwang/CS_QMI/env
```

需要阅读的已有文件：

```text
reports/PLAN_CONSTRAINTS_20260925.md
reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md
reports/P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md
reports/P48_PRECHECK_B1_REPORT_20260927.md
reports/SECOND_APP_FINAL_SUMMARY_20260928.md
src/vcs_ssl/objectives.py
src/vcs_ssl/models/critic.py
scripts/precheck_d_tests.py
reference/ssl_core.py
```

记录实际 HEAD、工作树改动、原文件 hash、依赖版本和可用设备。若 HEAD 已前进，先列出与本任务有关的差异；不要覆盖新代码或重启已完成的任务。

已有 `oracle_analysis/` 是否存在应由实际目录确认。附件包含可重跑的合成数值检查，不能把文档中的 oracle 表当作服务器已经生成的本地文件。

# 3. 评价量和损失归一化

## 3.1 所有 VCS 实现共用同一个样本目标

给定 $n_+$ 个正分数与 $n_-$ 个负分数：

$$
\widehat J=
\operatorname{mean}(t_+-\tfrac12t_+^2)
+\operatorname{mean}(-t_--\tfrac12t_-^2).
$$

同时记录

$$
\widehat R=\tfrac12\operatorname{mean}(1-t_+)^2
+\tfrac12\operatorname{mean}(-1-t_-)^2,
\qquad \widehat J=1-\widehat R.
$$

不要先把所有 pair 混在一起做不加权均值；当 $n_+\ne n_-$ 时，那样会改变参考混合比例。

## 3.2 必需的 JS 控制：相同后验参数化

统一使用未压缩的网络输出 $f$：

$$
T=\tanh f,\qquad q=\sigma(2f)=\frac{1+T}{2}.
$$

VCS 保持 $\mathcal L_V=-\widehat J$。用于匹配初始 logit 梯度的 JS/logistic 控制为

$$
\mathcal L_{\mathrm{JS,match}}=
\mathbb E_P\operatorname{softplus}(-2f)
+\mathbb E_Q\operatorname{softplus}(2f).
$$

它等于两倍的平衡 BCE。正项和负项分别平均，与 VCS 在 $f=0$ 时的梯度尺度相同。报告其原生 JS 数值时使用

$$
\widehat{\mathrm{JS}}=\log2-
\tfrac12\mathcal L_{\mathrm{JS,match}}.
$$

这样区分训练尺度和报告量。另行调参的 native BCE 可以在扩展实验中比较，但名称和倍数必须明确。

匹配后验对照允许直接比较 $\mathbb E_M[(T-\eta)^2]$。MINE、NWJ 和 InfoNCE 保留各自原生输出及目标；不要为了“相同 critic”把其输出也强行截成 tanh。

## 3.3 核心指标

每个合成条件必须记录：

| 字段 | 定义 |
|---|---|
| `J_eval` | 独立 EVAL 上的原始 VCS 分数 |
| `S_truth`, `S_truth_se` | 解析值或高精度 MC 参照及误差 |
| `signed_value_error` | $\widehat J-S$ |
| `posterior_mse` | $\frac12 E_P(T-\eta)^2+\frac12 E_Q(T-\eta)^2$ |
| `excess_to_bayes_risk` | $\mathrm{posterior\_mse}/(1-S)$，仅在分母可靠时 |
| `gate_pos`, `gate_neg` | $1-T^2$ 的均值、分位数 |
| `fit_time`, `eval_time` | 同一设备上的拟合和评估耗时 |
| `n_base_fit`, `n_pair_fit` | 独立基础观测数与实际 pair 数，分开记录 |
| `trainable_parameters` | 所有基础 critic 的总参数量 |

EVAL 上“oracle 的样本 $J$ 减学习模型的样本 $J$”与后验 MSE 在有限样本下不要求逐次相等。精确 gap 是总体关系；若使用精确离散权重，则应逐项精确一致。

$\widehat J\in[-3,1]$ 是数值范围检查，不是性能指标。不使用“超过 5 nats 的跳变”或任意跨量纲阈值。接近零的 oracle 值不计算 learned/oracle 比值。

# 4. 数据角色：基础观测先划分，pair 后生成

## 4.1 统一四角色

| 角色 | 允许用途 |
|---|---|
| FIT | 拟合基础 critic、残差候选和预处理参数 |
| TUNE | 拟合组合权重、残差步长、预先列出的校准参数 |
| SELECT | 选择 checkpoint、字典轮次、ridge 或已列出的候选 |
| EVAL | 一次计算固定模型的最终评价量 |

同一图像、受试者或相关试次的全部派生 pair 必须属于同一基础观测角色。按 pair 行随机拆分不合格。

oracle 的 $\eta$、$S$ 和真实导数只用于诊断与评价，不能用于挑选网络、权重或早停点。

## 4.2 合成数据的实施默认

首批每个条件使用以下互相独立的数据流：

```yaml
fit_joint_observations: 4096
tune_joint_observations: 1024
select_joint_observations: 1024
eval_pairs_per_distribution: 32768
truth_pairs_per_distribution: 300000
batch_size: 256
pilot_training_seed: 0
replication_seeds_after_review: [0, 1, 2]
```

P/Q 各自生成；P 的真实联合样本、Q 的独立边缘样本和 truth 样本使用独立随机流。报告 `N_fit` 和 batch size，不能将它们都写作 N。

## 4.3 冻结图像表示的实施默认

从原 45k fit 图像中按身份固定划分 27k/6k/6k/6k，分别用于 FIT/TUNE/SELECT/EVAL；所有模型和 checkpoint 使用同一份 manifest。若这些具体数据角色与现有未完成实验冲突，先回报，不另挑一份看起来更好的划分。

这些图像可能已被冻结 encoder 用于 SSL。当前实验评价的是固定表示上的新 critic 泛化，不把它写成 encoder 对未见图像的外部验证。

首批冻结 EVAL 使用独立配对块：每个块包含一个 anchor 图像的两个增强视图，以及另一张不共享身份的 partner 图像。不同块不重复基础身份。正负项可以共用 anchor，但方差和 bootstrap 按整个块计算。FIT 可使用现有 K=8 循环移位；EVAL 的 K=1 独立块仍估计相同 P/Q。

## 4.4 product-of-marginals 的规则

普通依赖问题使用独立身份的边缘配对。相同语义类别的负配对仍然有效，不按类别排除。

条件 nuisance 问题在 $Y$ 内独立抽取 $N$。允许 $N_{\mathrm{neg}}=N_{\mathrm{pos}}$；强制把二元 nuisance 翻成相反值会改变 Q。条件分析是单独定义的测量任务，不能把其标签采样规则偷偷用于原 SSL。

每个角色使用该角色内独立准备的 Q 数据或 pool，不把 FIT/TUNE 的 product pool 共享给 EVAL。条件类别权重由实验设计或 FIT 预先确定并记录，不能根据 EVAL 重新平衡。

共享样本形成的多负配对不是独立样本。标准误、bootstrap、置换与复杂度统计均按实际基础观测和配对结构处理。

# 5. 阶段 0：数学与实现检查

先复用附件中的小型参考模块，补进服务器测试环境；附件不是完整训练系统。

必须通过以下检查：

| 测试 | 要检查的内容 |
|---|---|
| `risk_identity` | 对任意分数、不同正负数量，$J=1-R$ |
| `finite_gap` | 精确离散 P/Q 下，$S-J=E_M(T-\eta)^2$ |
| `residual_step` | $\Delta J=2\lambda A-\lambda^2B$，包括 $B=0$ 和边界系数 |
| `convex_mixture` | 有界性、组合恒等式和拟合集上的目标改进 |
| `simplex_weights` | 非负、和为 1、目标与直接 scores 计算一致 |
| `gradient_scale` | VCS 与 JS 匹配式在 $f=0$ 的梯度一致 |
| `rulsif_identity` | 同一函数类、相同常数约定下的仿射等价 |
| `nested_information` | 精确粗化下的依赖增量恒等式 |
| `sample_roles` | 基础身份互斥，生成 pair 不跨角色 |
| `noise_independence` | 两端噪声独立，噪声后无隐式归一化 |

精确代数测试采用 float64；误差容差由浮点精度设置，不使用“达到 1% 就算一致”。torch 实现另外检查梯度到基础 critic 和组合前输出的路径。测试失败先修复实现，不启动训练。

# 6. 阶段 1：建立受控 estimator benchmark

## 6.1 首批 Gaussian 条件

每侧维度 $d=20$，即 pair 总输入维度为 40。

$$
X\sim\mathcal N(0,I_d),\qquad
Y=\rho X+\sqrt{1-\rho^2}E,
\quad E\sim\mathcal N(0,I_d),\quad E\perp X.
$$

使用 $I\in\{0.5,4,8\}$ nats 设定相关性：

$$
\rho=\sqrt{1-e^{-2I/d}}.
$$

这里 I 是生成器标尺，不是 VCS 的真值。Q 为两端独立的标准高斯。

真实 log-density ratio 为

$$
\ell_\rho(x,y)=-\frac d2\log(1-\rho^2)
+\frac{2\rho x^\top y-\rho^2(\|x\|^2+\|y\|^2)}{2(1-\rho^2)},
$$

$$
\eta_\rho=\tanh(\ell_\rho/2).
$$

用独立 truth 数据计算 $S=E_M\eta^2$，另算 oracle $J(\eta)$ 作为 MC 一致性检查。原始 Gaussian 输入不做 L2 归一化。FIT 均值/标准差可作为固定可逆预处理，但需在 oracle 中正确对应。

增加一个单独的二次特征 critic 诊断：$[x^\top y,\|x\|^2,\|y\|^2,1]$ 后接线性层和 tanh。其函数类包含 Gaussian oracle；它是识别优化误差的工具，不据此声称一般高维数据同样容易。

## 6.2 第一批基础 critic 字典

下列配置均是本轮实施默认。所有方法使用相同候选、初始化规则、数据和每候选更新预算。

| ID | $f(x,y)$ | 作用 |
|---|---|---|
| C0 | $a\,x^\top y+b$ | 两参数参照；仅单位向量输入时称 cosine |
| C1 | $\sum_{r=1}^{16}(u_r^\top x)(v_r^\top y)+b$ | 有限低秩交互 |
| C2 | `concat → 256 ReLU → 256 ReLU → 1` | 一般非线性参照 |

全部输出 $T=\tanh f$。C1 的两端投影分别学习，不强制正定或共享；没有额外 concat 分支。维度小于 16 时 rank 取 $\min(16,d)$。它是明确的候选类，不将过去其他 bilinear 实现的结论直接套用到这里。

首批每候选 Adam、lr=$5\times10^{-4}$、weight decay=0、batch=256、最多 2000 updates。每 100 updates 在 SELECT 上评估一次原生风险，选择固定模型。JS 控制使用 §3.2 的匹配梯度尺度。

以上是小试起点。学习率敏感性扩展才使用预设网格 $\{10^{-4},5\times10^{-4},2\times10^{-3}\}$，不得先让某种方法失败后只为另一种方法加预算。

## 6.3 第二批合成扩展

首批核对后再增加：

- Gaussian 的 $d\in\{2,20,50\}$ 与弱、中、强依赖，分别改变样本量和 batch；
- 各坐标立方变换 $U=X^3,V=Y^3$，oracle 用逐坐标立方根恢复后计算；共同 Jacobian 在密度比中抵消；
- 对称混合的异号相关分布：每个坐标对为 $\tfrac12\mathcal N_2(0,\Sigma_\rho)+\tfrac12\mathcal N_2(0,\Sigma_{-\rho})$，各坐标对独立。使用稳定 log-sum-exp 计算联合密度及 oracle；各边缘仍为标准高斯。

最后一类允许协方差为零但存在非线性依赖。整体 $S$ 不等于各坐标 $S$ 的和，必须由整体密度比计算。暂不加入更多数据生成器。

## 6.4 对通道参数的梯度评价

固定学习好的 critic 参数，在新 base noise 下生成 $Y_\rho$。计算 $\partial_\rho J_\rho(T_\phi)$ 时，梯度只穿过数据生成器；不穿过 critic 的训练过程。

oracle 参考采用共同随机数的中心差分，分别在 $\rho\pm\delta$ 重新计算真实 $S$。检查多个 $\delta$ 和 MC 重复，确认数值误差后再比较学习梯度。也可以实现 envelope 形式：oracle critic 的显式 $\rho$ 系数在当前点固定，保留对输入的导数；两种参照在 MC 误差内核对。

Gaussian 生成器下 Q 的边缘不随 $\rho$ 变化；若改为噪声尺度参数，P 与 Q 都可能变化，不能一律丢掉负项梯度。零梯度附近报告绝对误差，不计算不稳定的相对误差或符号结论。

# 7. 阶段 2：有界组合和残差校正

## 7.1 第一版：固定字典的凸组合

FIT/SELECT 完成各基础 critic 的拟合和 checkpoint 选择后，冻结它们，在 TUNE 上收集

```text
T_pos: [n_pos, m]
T_neg: [n_neg, m]
```

令

$$
d=\operatorname{mean}(T_+)-\operatorname{mean}(T_-),
$$

$$
G=\frac12\operatorname{mean}(T_+T_+^\top)
+\frac12\operatorname{mean}(T_-T_-^\top).
$$

其中每个乘积均为单个 pair 的 $m$ 维输出外积。求解

$$
\max_{w\geq0,\;\mathbf1^\top w=1}
\quad d^\top w-w^\top Gw.
$$

首批 $m=3$，可用附件的小维 simplex 枚举求解器或等价的可靠 QP 实现。不要使用未经检查的 inverse；记录求解残差、目标值和权重。

**训练集或 TUNE 上的改进只是优化检查；性能只在 EVAL 上评价。** 在 TUNE 上求得的权重原样用于 EVAL，不根据 EVAL 重新退回最佳单模型。

推断时直接计算 $T_w=\sum_jw_jT_j$，不执行 `atanh → tanh`，也不对最终分数额外 clipping。softmax 权重与 simplex QP 不是同一个数值求解器，不能用默认 softmax 优化替代而仍称精确 QP。

## 7.2 第二版：一个残差候选

先由 SELECT 确定基础模型 $T_0$。冻结它，初始化与 C2 相同的小型候选 $U=\tanh g$。

首批固定 $\lambda_0=0.5$，在 FIT 上训练 U：

$$
\min_g-\widehat J((1-\lambda_0)T_0+\lambda_0U).
$$

这里最终训练量仍是原始 J。最多 2000 updates，SELECT 选择候选 checkpoint。然后在 TUNE 上计算

$$
\widehat A=\tfrac12\operatorname{mean}[(1-T_{0,+})(U_+-T_{0,+})]
+\tfrac12\operatorname{mean}[(-1-T_{0,-})(U_--T_{0,-})],
$$

$$
\widehat B=\tfrac12\operatorname{mean}(U_+-T_{0,+})^2
+\tfrac12\operatorname{mean}(U_--T_{0,-})^2.
$$

令 $\lambda=\Pi_{[0,1]}(\widehat A/\widehat B)$；$\widehat B$ 为零或低于数值容差时令 $\lambda=0$ 并记录原因。EVAL 使用固定 lambda。

首批只做一个残差候选，不自动迭代到多轮 boosting。若实现一阶残差方向优化，应另列变体，不能替换这里直接优化 J 的第一版。

## 7.3 必须有的对照

| 对照 | 排除哪类解释 |
|---|---|
| 各个单 critic 与 SELECT 选出的最佳单 critic | 组合是否仅等于选择较好的函数类 |
| 相同数量的独立初始化、选最佳单模型 | 改善是否仅来自重启次数 |
| 同总参数量或同总拟合成本的大单模型 | 改善是否仅来自容量或算力 |
| 同字典的 JS 后验组合 | 收益是否来自二次拟合，还是一般集成 |

首批先完成单 critic、字典组合、一个残差候选和 JS 字典控制。容量/预算确认在候选出现稳定收益后补齐，不能在正式贡献中省略。

JS 字典由相同结构的基础网络按 §3.2 训练。权重优化其原生平衡 log loss；混合后 $q_w=\sum_jw_j\sigma(2f_j)$。用 log-sum-exp 计算 $\log q_w$ 和 $\log(1-q_w)$，避免概率舍入成 0 后再靠 clipping 获得虚假的稳定性。

报告两种组合的后验 MSE、各自原生风险和成本。若把 JS 输出放入 J 评价，该列明确命名为“共同后验的 VCS 回归评价”，不称 JS 的原生估计值。

# 8. 并行小任务：现有 SSL checkpoint 的 estimator 诊断

先读取 checkpoint inventory，选同一运行的早、中、晚三个实际存在的时间点。建议优先使用已保存的 epoch 20/100/800；若不存在，报告并使用最近的已保存时间点，不为补这个诊断重训 encoder。

优先模型：当前四视图 VCS 主配置、相同预算可用的 SimCLR，以及一个早期 VCS 配置。每个固定 checkpoint 完成：

1. 记录训练时 critic 的 J、正负残差、门控和实际输入梯度。
2. 冻结 encoder/projector，独立重拟合基础 critic、字典和残差候选。
3. 报告 EVAL 上相对原 critic 的 $\Delta J$、不同候选的差异与成本。

当前训练已包含 BN 与共享视图前向。固定评估使用 `eval()` 和复制后的模型；若专门研究训练态梯度，另用独立副本重现训练 batch，记录这是 `train-mode diagnostic`，避免污染 checkpoint 的统计量。

正负配对分别记录：score 分位数、$|T|>0.95$ 的正确端/错误端比例、$1-T^2$、$(T-C)(1-T^2)$，以及 `grad_left` / `grad_right`。negative detach 导致右侧梯度为零应保留，不补造该路径。

没有原 critic 的 SimCLR checkpoint，只拟合测量 critic，不把新拟合模型称为“训练时的 estimator”。

# 9. 阶段 3：观测尺度实验

此阶段在首批报告确认后执行，首先只冻结表示。

## 9.1 噪声定义

对单位范数的基础表示 $z\in\mathbb R^d$：

$$
u=z+\frac{\tau}{\sqrt d}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I_d).
$$

$\tau$ 是总 RMS 噪声幅度；研究稿中的每坐标标准差为 $\sigma=\tau/\sqrt d$。首批尺度预设为

$$
\tau\in\{0,0.1,0.3,0.6\}.
$$

这是实施默认，不因某个 EVAL J 更好而临时增加、删除尺度。

两个视图的 epsilon 独立。所有方法共用记录好的随机流；同一尺度的 P/Q 使用相同边缘噪声规律。比较方法时可复用同一组噪声以降低配对差异。

**加噪之后不再 L2 归一化。** 接收原始 noisy pair 的 critic 测量 $S_\sigma$。若保留只读取相似度的参照，它只是该目标下的受限 critic，不能把相似度压缩后的估计当成完整 oracle。

尺度较大的噪声可按方差差独立增量构造，用于核对 Gaussian 噪声通道的嵌套关系。有限 critic 的 J 无须强制单调。

## 9.2 每个尺度的输出

合成数据：重新计算该尺度的 $S_\sigma$、posterior MSE 和门控；不要始终拿 $S_0$ 作真值。

冻结表示：报告独立 J、refit 增益、不同种子波动、类别信息与 nuisance 信息的测量。先确认改善的拟合没有仅仅来自把所有信息抹掉。

SSL 尚不在本阶段启动。用于训练的尺度只能在开发数据选择，并为 JS/InfoNCE 设置完全相同的观测变换。

# 10. 阶段 4：条件依赖增量

## 10.1 首先复用现有 nuisance 任务

沿用已记录的颜色/模糊植入和分类器，不改变其任务定义以制造更明显的效果。先核对实际脚本、强度、类别关系、模型与身份划分。

需要构造的同一实验为

$$
P=P_YP_{H,N\mid Y},\qquad Q=P_YP_{H\mid Y}P_{N\mid Y}.
$$

完整观测 $B=(H,N,Y)$；压缩观测 $A=(F(H),N,Y)$。固定分类头 F，在 eval 模式输出完整 logits，关闭 dropout 等随机性。

A 必须真的是 B 的确定性函数。任意两层各自 pooled features 不一定具有这个关系，不能直接引用嵌套定理。

## 10.2 两套 critic 共享比较基础

两套模型使用相同基础身份、相同正负配对和相同 nuisance 分布。critic 输入显式包含 Y，或采用逐类拟合并按预先固定的 $P_Y$ 权重汇总。不能仅在采样中使用 Y、却仍声称遗漏 Y 的 critic 达到了完整条件 oracle。

完整侧可把压缩侧的固定输出 $T_A(F(H),N,Y)$ 放入自己的候选字典，再加入依赖 H 的新候选。这样完整侧函数类包含压缩侧，便于解释受限函数类的比较。TUNE 上的单调性不等于 EVAL 差值必然非负。

## 10.3 同时报告三个值

$$
\widehat S_H=\widehat J_B,\qquad
\widehat S_F=\widehat J_A,\qquad
\widehat\Delta_J=\widehat J_B-\widehat J_A.
$$

以及

$$
\widehat\Delta_T=
\tfrac12\operatorname{mean}_P(T_B-T_A)^2
+\tfrac12\operatorname{mean}_Q(T_B-T_A)^2.
$$

两个增量在总体 oracle 处相同；有限 critic 下不要求相等。原样保留负的 $\widehat\Delta_J$，并报告其不确定性。不要把非负的 $\widehat\Delta_T$ 自动解释为真实信息损失。

先用离散确定性粗化和 Gaussian 线性压缩验证 oracle 增量，再对真实表示报告重复拟合、独立 EVAL 和相同配对的差值区间。

## 10.4 检验与解释各自的评价

现有 within-class permutation 检验继续评价 $H\perp N\mid Y$ 或 $F(H)\perp N\mid Y$。所有 critic、权重、早停和核参数在 EVAL 之前固定，null 重复重新生成 nuisance 的随机赋值。

**条件独立的置换检验，不自动成为 $\Delta=0$ 的检验。** 没有另外推导零增量假设下的分布前，不报告“增量显著”的置换 p 值。

HSIC 使用逐类带宽或相应的合理控制；学习核、JS、连续分数 C2ST 使用相同拟合和选择数据。把功效、错误率、增量幅度与成本分开列出，不用单一功效表代替 estimator 评价。

同步评估修改 nuisance 后的 logits、预测和准确率变化。测量依赖与模型实际使用之间的联系，不将依赖检出直接等同于有害 shortcut。

# 11. 污染实验与 SSL 的衔接

## 11.1 先完成可控污染

在中等依赖 Gaussian 条件下，首轮扩展优先使用

$$
P_\varepsilon=(1-\varepsilon)P+\varepsilon Q,
\qquad \varepsilon\in\{0,0.1,0.2,0.4\}.
$$

它保持边缘分布不变，受污染 oracle 可准确计算。比较原 VCS、有效的组合版本和匹配 JS。

同时记录：受污染真值误差、相对清洁目标的变化、干净/错配样本的门控和梯度贡献。后者两项回答不同问题，不混写成一个“稳定性”。

一般 outlier 或重尾污染可能改变边缘；加入时必须重新定义 Q 为受污染联合的实际边缘乘积，或者明确任务已变成一般 P/Q 散度估计。

若做图文错配识别，至少采用交叉拟合或独立的带噪评估关系。全部在训练 pair 上评分会混入记忆。AUROC 评价排序；Brier、log loss 与精度/覆盖率评价概率或决策校准。

已知 epsilon 的清洁目标恢复公式放在研究稿备选段。本轮不把“知道噪声率”的结果写成未知噪声鲁棒性。

## 11.2 进入在线 SSL 的条件

至少一项 estimator 改动已在相同预算或明确成本下，改善独立后验误差、梯度误差或目标分辨率，并在独立重复中保持相同方向。进入在线训练前，写明要检验的具体机制；不要求每个数据格都超过人为设定的固定百分比。

在线最小对照按两种因素分开：

| 方法 | 原始观测 | 选定的观测尺度 |
|---|---|---|
| 原 VCS critic | 基准 | 尺度作用 |
| 已验证的 VCS 拟合改进 | 拟合作用 | 二者的交互，后做 |
| 匹配的 JS | 同路径基准 | 匹配尺度控制 |

先分别测试一个因素，交互组合在前两项有依据后再做。保持相同 encoder、projector、增强、视图数、K、negative detach 和预算。

若组合权重在当前 encoder 上内层拟合，外层更新时固定 critic 参数和权重，保留它们对表示输入的导数。不要因把 weights detach 而连输入梯度一起截断，也不要引入两点梯度或替代反向传播算法。

外层是否采用当前 negative detach 或全路径梯度，应预先列出；第一版沿用基准路径，并给 JS 相同路径。缓存特征随 encoder 改变会过时，内层拟合数据必须用当前 encoder 生成。具体在线日程在离线结果后确认。

**首批不重开官方 CIFAR test。** 新方法继续开发集选择；最终需要另行固定确认方案，并披露已有测试已被查看。

# 12. 对已有 E/R/T/S 草案的修改清单

| 原草案中的安排 | 本轮处理 |
|---|---|
| 六种 estimator 直接全网格比较 | 先原 VCS/JS/拟合变体，验证核心问题后扩展 |
| VCS “5 nats spike” | 移除；使用有意义的误差、分位数和门控 |
| learned/oracle 为主要指标 | 补 posterior MSE 和相对 Bayes risk |
| oracle 单点 1.2× 作为整体门槛 | 作为待复核现象，不设成所有实验的预定成功标准 |
| raw ridge 与 tanh-wrapped 解要求一致 | 分开函数类；只对真正等价目标做一致性测试 |
| 训练集错配 AUROC | 增加独立或交叉拟合评分 |
| 条件检验只看功效 | 加入幅度与增量的估计精度和成本 |
| 先花大预算迁移完整 SSL 框架 | 放到 estimator 改进后的外部确认 |

# 13. 建议的代码组织和命令接口

新增独立模块，尽量复用现有数据与日志，不改写原 reference 核心：

```text
src/vcs_estim/
  objectives.py       # J, balanced risk, matched JS, score diagnostics
  synthetic.py        # generators, densities, posterior oracles
  pairing.py          # split-aware independent and conditional pairing
  critics.py          # finite candidate classes; tanh outputs
  convex_mix.py       # fixed-output simplex fit and residual coefficient
  fitting.py          # candidate / residual training and selection
  observation.py      # fixed-scale independent Gaussian observation
  increments.py       # nested-observation estimands and evaluations
  evaluation.py       # own-target, posterior, gradient, cost metrics
  io.py               # manifests, result schema, hashes
```

下列为**待实现或映射的接口**，不是声称仓库已经存在的 CLI：

```text
python -m vcs_estim.run --config <config> --stage estimator_probe
python -m vcs_estim.run --config <config> --stage frozen_diagnostics
python -m vcs_estim.run --config <config> --stage bounded_refit
python -m vcs_estim.run --config <config> --stage observation_scan
python -m vcs_estim.run --config <config> --stage dependence_increment
```

配置必须记录 `loss_scale_convention`、`combination_fit_role`、`target_kind`、`noise_coordinate_sd`、`noise_total_rms`、`negative_construction` 和 `gradient_routing`。没有噪声时也显式写 `sigma=0`。

# 14. 首批规模和阶段交付

## 首批范围

1. 阶段 0 的数值与身份检查。
2. $d=20$、I=0.5/4/8 的 seed-0 Gaussian 小试：C0/C1/C2 对 VCS 与 JS，字典组合，一个 VCS 残差候选。
3. 一个实际存在的 VCS 运行的早/中/晚 checkpoint 诊断。若 checkpoint 缺失，报告现有时点；不补训。
4. 测量单元耗时后，提交下一批三种子、非高斯扩展和观测尺度的成本估计。

GPU 工作沿用服务器现有 SLURM 与配额管理。首批小试先单卡执行，不从本文推断新的配额或并发权限。不要自动启动 1000-epoch 标准框架迁移、ImageNet、EEG 或完整图文训练。

## 每个阶段必须交付

| 交付 | 具体要求 |
|---|---|
| 配置与代码变化 | 实际 git commit、diff、测试记录、依赖版本 |
| 数据 manifest | 基础身份、关系定义、各角色、随机种子和哈希 |
| 数值结果 | 每个运行的原始 JSON/CSV，不只汇总最优值 |
| 机制诊断 | J、posterior error、门控、实际梯度、组合权重 |
| 资源记录 | 样本、pair、步数、参数量、耗时、显存 |
| 阶段解读 | 哪类误差减少了；效果在哪些条件成立；下一步测什么 |

不得因 EVAL 的结果删除差的 seed、回退权重或改选 checkpoint。候选无改善时，仍报告其拟合误差、权重和成本；数学错误先修复，科学上的负结果用于缩小下一轮问题。

# 15. 结果字段与汇总表

建议每个 evaluation JSON 至少包含以下字段：

```yaml
run_id: <unique id>
code_commit: <actual commit>
source_reference_commit: 7b7402c05161c33d77a4301f6efc27bb55420e6a
status: completed_or_failed
setting: gaussian_or_frozen_or_conditional
seed: 0
target_kind: S_or_S_sigma_or_conditional_S
estimator: vcs_single_or_vcs_mix_or_vcs_residual_or_js
critic_family: <recorded family or dictionary>
loss_scale_convention: minus_J_or_matched_JS
roles_hash: <manifest hash>
negative_construction: independent_units_or_cyclic_K
n_independent_units: <count>
n_positive_pairs: <count>
n_negative_pairs: <count>
S_truth: <value or absent with reason>
S_truth_se: <value or absent with reason>
J_eval: <raw value>
posterior_mse: <value or absent with reason>
excess_to_bayes_risk: <value or undefined with reason>
selected_checkpoint: <id>
combination_weights: <fixed weights if used>
fit_seconds: <measured>
eval_seconds: <measured>
```

研究计划中没有真实结果的字段，不能填写为 0。出现数值异常时保存输出和失败原因。表格同时列单模型、组合和匹配控制，避免只报告最大的 J。

# 16. 本轮希望得到的结论

最终报告只需清楚回答四件事：

**第一，当前 estimator 的主要误差在哪里？** 是函数类不足、拟合不足、泛化不足，还是目标已进入饱和区间？

**第二，哪种修改确实减少了误差？** 有界组合、残差拟合、观测尺度分别带来什么效果，付出多少成本？

**第三，同样能力的 JS 控制如何表现？** 二次结构的收益与一般增加容量/集成的收益怎样区分？

**第四，哪些结果足以进入 SSL 或条件增量应用？** 下一批只扩展已经有明确依据的机制。

# 来源与附件

**理论依据 [P1]：** 用户提供的《Variational CS-QMI – Research Plan》，§§3–7、9–10，特别是原始 J、tanh critic 和精确回归 gap。

**研究依据 [A1]：** 配套 `VCS_QMI_Estimator_Research_Plan_v1_CN.md`，由上一轮 estimator 优化分析整理。当前文件中的样本数、网络尺寸、lambda0、噪声网格、阶段规模均为实施默认。

**仓库依据：** §2 所列文件，参考快照 `7b7402c`。主要来源的完整链接见研究稿来源节；执行时另行核对实际 HEAD。

**附件：** `support/bounded_regression_core.py`、`support/validate_spec.py`、`support/validation_report.json`，以及原样保留的上一轮 `independent_checks.py/.json`。小型参考模块仅覆盖代数和固定输出组合，不包含数据集、神经训练或服务器调度。
