---
title: "VCS 完整 SSL 性能优化：服务器执行规范"
subtitle: "v3｜固定几何刻度、完整配对与随机目标的精确计算"
author: "Server Agent Brief / VCS-QMI"
date: "2026 年 9 月 30 日"
---

# 1. 本轮交付与执行范围

**目标：在保留原始 VCS 目标的情况下，争取超过同协议充分调优的 SimCLR。首批候选必须经过完整的 800-epoch 预训练，不以合成 probe、较高 J 或短跑 kNN 代替。**

本轮围绕三个实现因素：固定 critic 的几何刻度；完整视图配对；相同观测噪声目标的多次独立采样平均。配套分析稿 `VCS_SSL_Optimization_Analysis_v3_CN.md` 给出动机和推导；本文件给出实际数据流、配置差异和结果要求。理论出处为合作者原计划 §§1、4–10；代码与资料索引见配套分析稿第 11 节。

参考仓库：`W-Yinghao/VCS_QMI`，证据截点 `e466fd548daf3ce53b97f157f4f91a75aa5c3d8b`。到该截点，P95 的后续种子已安排，P98/P101/P102 是已冻结的新任务。本文件不把配置、提交脚本或排队记录当成完成结果。

**收到按本文件执行的指令后：**先检查实际 HEAD、复用已有实现、完成数值与集成测试，然后按当前 GPU 限额安排首批八组完整训练。训练性能的结论在运行完成后给出；早期低 J 或低有效秩不触发科学性终止。

所有新任务只使用现有训练／选择划分。遵循已有 normal QOS、非 runfill 的调度要求；ImageNet 暂不提交。不要取消、覆盖或重启已经批准的 P95、P97–P102 等任务。

## 1.1 对已有分支的衔接

| 已有工作 | 本轮处理 |
|---|---|
| P86 estimator / kernel 比较 | 复用结果和代码；不重跑原完整网格 |
| P88 神经／核 CS 完整 SSL | 作为原方法证据与实现回归检查 |
| P90 强增强四格 | 复用标准／强增强 SimCLR 参照 |
| P95 噪声 τ=0.3、JS、组合、刷新 | 收齐已有种子；本轮不复制其 R=1 或无噪声 JS |
| P98 probe robustness | 使用其统一完成结果，不为单个方法另选 probe |
| P99 solo-learn scalar optimizer | 独立协议继续；不与本轮冻结 linear 混合 |
| P101/P102 | 保持已批准范围；不是本轮完整 SSL 的前置门槛 |

若实际 HEAD 已有与本文完全相同的完整实验，给出 config diff、seed、模型初值、数据划分和评价证据后复用。历史短训练或不同 critic 下的结果不当作相同运行。

# 2. 不变的数学定义

$$
P=P_{XY},\quad Q=P_XP_Y,\quad M=\tfrac12(P+Q),
$$

$$
J=E_P(T-\tfrac12T^2)+E_Q(-T-\tfrac12T^2),\quad T=\tanh f.
$$

训练 `loss = -J`；两类各自平均。正负数量不同，也各占一份总权重。J 不裁剪，不通过 CS 对数变换训练。

当前原配置使用

$$
f(z_1,z_2)=a(z_1^\top z_2)+b,
$$

输入为 L2 归一化 projector 输出。新 G 组只改变 a,b 的可学习范围；U 组只改变配对期望的计算；N 组改变／重复计算已明确的 noisy-observation 目标。

不加入监督训练标签、hard-negative 筛选、类排除、负样本重加权、额外 variance/covariance loss、teacher 或替代梯度估计器。

## 2.1 数据流与指标角色

主指标是 `h_before_projector` 的 final-epoch 冻结线性评价。kNN 同时报告。critic accuracy、J、rank 都是诊断，不用于单独宣布性能胜出。

固定 angular critic 限制了函数类：它仍是原 S 的合法有限 critic，但不保证表示任意分布的最优 η。噪声训练估计 S_sigma，干净训练估计 S；两种目标的 J 不以绝对大小排名。

# 3. 原配置：从实际文件继承

基础 YAML：`configs/cifar10_hpK_a5_views4_800ep_vcs_seed{s}.yaml`。先核对文件存在及当前内容，继承全部未列出的字段，不从本文重新手写一个近似配置。

| 项目 | 预期值，实际执行前核对 |
|---|---|
| 数据 | CIFAR-10，45k fit / 5k selection；official test 不开放 |
| Backbone | CIFAR ResNet-18，3×3 stride-1 stem，无初始 maxpool |
| Projector | 512→512(BN,ReLU)→128，末层无 BN，随后 L2 |
| 视图 | 4 个同分布独立增强；标准块保持 P35 原样 |
| Batch | 256 基础图像，drop_last=True |
| 时间 | 800 epochs，warm-up 10，原 cosine 日程 |
| 优化 | 原 AdamW 1e−3，矩阵 wd 1e−4，critic wd=0 |
| 配对 | 六组 a<b 视图对，各 K=8 不同非零 shift |
| 梯度 | 正 pair 双侧；负 pair 右侧 detach |
| 测试 | final h，原冻结 probe 与 kNN；不按中间精度选 checkpoint |

每 epoch 175 updates；总 140,000 updates；35,840,000 次基础图像呈现；143,360,000 次编码视图。重复 noise/更多 pair 不增加 `encoded_views`，需单独记录 critic 计算量。

基础参考 B0：P35 seeds 0/1/2 = 86.42/87.16/87.44 linear-val，均值 87.01±0.53；kNN 85.46±0.12。统一调优的标准增强 SimCLR = 88.32±0.30，强增强 SimCLR = 89.49±0.36。两种增强分别比较，不混作一个阈值。

\clearpage

# 4. 首批运行清单

`configs/experiments_v3.yaml` 是**逻辑实验清单**，不是仓库当前可直接加载的完整 YAML。Agent 需将 overrides 映射到当前 config schema，生成 resolved YAML 和差异表；参考代码仅覆盖损失核心。

| ID | a,b | 配对 | 负梯度 | 噪声 / 重复 | 本轮要回答的问题 |
|---|---|---|---|---|---|
| G1 | 固定 1,0 | 原 K8 跨视图 | 右侧 detach | 无 | 去掉自由仿射标定 |
| G2 | 固定 2,−1 | 原 K8 跨视图 | 右侧 detach | 无 | 另一种平缓固定余量 |
| G3 | 固定 1,0 | 原 K8 跨视图 | 完整 | 无 | G1 与完整负梯度的交互 |
| G4 | 原可学习 5,0 初始化 | 原 K8 跨视图 | 完整 | 无 | 完整梯度的控制 |
| U1 | 原可学习 5,0 初始化 | 所有视图 token | 右侧 detach | 无 | 配对覆盖和角色平衡 |
| U2 | 固定 1,0 | 所有视图 token | 右侧 detach | 无 | 配对与几何刻度结合 |
| N1 | 原可学习 5,0 初始化 | 原 K8 跨视图 | 右侧 detach | τ=0.3，R=4 | 同 S_sigma 的 Monte Carlo 改进 |
| N2 | 原可学习 5,0 初始化 | 原 K8 跨视图 | 右侧 detach | τ=0.3，R=1 | 匹配 JS 噪声控制 |

G/U/N1 使用原 VCS；N2 是明确命名的控制。每组 seed 0，完整 800 epochs。建议先排 G1、G3、U1、N1 与 N2，再安排余下几组；顺序只是资源安排，不因早期分数删掉后面的候选。

## 4.1 固定刻度获益后的必要初始化控制

G1/G2 同时改变了初始化和学习方式。若这类候选获益，需要补一个**完全相同初始 a,b、但允许 a,b 学习**的完整控制；对应梯度路径和采样方式保持一致。

例如 G1 对照使用可学习 (a0,b0)=(1,0)。这样才能区分“较好的起点”与“全程固定几何刻度”。该控制在确认机制前必须完成；不要求它先于所有首批候选。

# 5. G 组实现

## 5.1 参数化

$$
T=\tanh(a s+b),\qquad s=\sum_k z_{1k}z_{2k}.
$$

固定 G1/G2：a、b 注册为 buffers，不列入 optimizer，不通过一个附加可学习线性层恢复自由度。G1=(1,0)，G2=(2,−1)。encoder/projector 正常训练，norm 规则不变。

原 critic 的 build/optimizer 逻辑可能假定存在可训练 critic 参数。需要允许固定 scorer 有零个可训练参数：optimizer 仅更新 encoder/projector；critic grad norm 的缺失或零是预期状态，不是训练失败。

接口可保留 `.scale`、`.bias`、`forward(left,right)`、`logits(left,right)`。统计中显式记录 `trainable_affine=false`，不要仅从 `model.parameters()` 推断总可训练参数。

## 5.2 梯度路径

G1/G2 使用原有负右端 detach。

G3/G4 的正、负 pair 都通过两端反向传播。这是原 J 的普通完整梯度，不做乘 2 补偿，不用自定义 backward，也不在 critic 参数上乘不同损失系数。

每个 pair 的分类标签只有人工的正负配对来源，不读取 CIFAR 类别。

## 5.3 必需记录

分开记录正项、负项的 encoder/projector 梯度范数及内积；还记录合并梯度。不能把每个样本范数之和当作聚合梯度范数。

G3/G4 与 detach 组的前向 J 在相同张量上应完全相同，输入梯度则按路由不同。critic 参数梯度在有参数的组不因 detach 直接变化。

# 6. U 组实现：全部视图 token 的原始二次目标

## 6.1 明确 token 与 mask

编码仍只做一次拼接 VB 前向。保持输出为 `[V,B,D]`，展平顺序采用 view-major：

```text
flat_index = view_id * B + image_index
base_image_id = flat_index % B
view_id = flat_index // B
```

正 mask：同一 base image 且不同 token。负 mask：不同 base image。自身 token 不参加任一集合，同图不同 view 不能进负集。

$$
N_P=VB(V-1),\qquad N_Q=VB\,V(B-1).
$$

B=256，V=4 时 N_P=3072，N_Q=1,044,480。所有都是有序 pair；P 的两种方向分数相同，但均值后权重保持正确。

## 6.2 两类分开平均

$$
L=-\frac{1}{N_P}\sum_P(T-\tfrac12T^2)
  -\frac{1}{N_Q}\sum_Q(-T-\tfrac12T^2).
$$

绝不能将 P、Q 合并后按总 pair 数平均。那样会改变两类权重。

正分数使用 `flat @ flat.T` 的相应位置，两端反传。负右端 detach 时，用 `flat @ flat.detach().T` 计算负分数。自由 a,b 仍保留正常参数梯度。

首批 chunk_size=256，按 anchor 行分块累积总和，最后除以全局 N_P/N_Q。不要先对各块均值再不加权平均，以免尾块权重错误。

## 6.3 与原 all_pairs_matrix 的区别

现有 `vcs_pair_loss_all_matrix` 对一组跨视图 `[B,D]` 计算全部 off-diagonal。本轮的 U 组是整个 `[VB,D]`，还包含相同增强编号、不同基础图片的负 pair。

视图必须来自相同的增强分布。不得顺手改成 global/local 不同增强分布的 multi-crop。若后续要引入那类设置，应重新定义 P/Q。

## 6.4 噪声 matrix hook 的陷阱

当前 `NoisyCosineCritic` 的噪声发生在 `forward()`；继承的 `score_matrix()` 可能绕开它。本轮 U1/U2 是无噪声组，N1/N2 继续逐 pair 路径。

在配置层显式拒绝“noisy critic＋旧 score_matrix hook”。不允许默默运行一个名义带噪、实际未加噪的目标。后续若合并 U 与 N，必须另写显式噪声前向并测试其期望和梯度。

# 7. N 组实现：保持 S_sigma，平均每次损失

## 7.1 固定采样规则

沿用 P95 的逐 pair occurrence 加噪规则：每次正、负配对打分，两端各自生成独立 ε；正负块之间也用新的随机数。

$$
\widetilde z=z+\frac{0.3}{\sqrt{128}}\epsilon.
$$

先完成原 L2 normalization，再加噪；加噪后不再 normalize。负右端先按原规则 detach，再加入与参数无关的噪声。

N1 在同一组编码特征和同一组 shift 上，重复 R=4 次独立噪声抽样：

```python
loss = 0
for r in range(R):
    t_pos, t_neg = score_with_new_independent_noise(...)
    loss += -J(t_pos, t_neg)
loss = loss / R
```

实现可以向量化，但平均的位置不能改变。不要计算 `J(mean_r(t_pos), mean_r(t_neg))`。也不要重复四次编码器前向。

N2 使用 R=1、相同噪声和相同 pair 路由，损失为

$$
L_{JS}=\operatorname{mean}_P\operatorname{softplus}(-2f)
+\operatorname{mean}_Q\operatorname{softplus}(2f).
$$

这里 f 是 noisy dot-product 的线性 logit。评价共同 J 时用 T=tanh(f)，不从 sigmoid 再做一次 tanh。

## 7.2 噪声随机数与恢复

使用单独的训练 noise generator 或现有可恢复 counter-based stream，保存在 checkpoint；评价 generator 独立。不能复用 dropout/global augmentation RNG，不能因加入诊断改变训练随机轨迹。

由于 N1 每步消耗更多噪声随机数，它不要求与 P95 在训练中逐步使用相同的全部噪声；但数据加载、增强和 shift 流应相同。估计条件方差时使用单独预先固定的诊断样本流。

## 7.3 clean 与 noisy estimator 评价

编码器、projector 和 BN 保持 eval。用显式开关在 critic 输入上采噪声，不将整个网络切回 train。

所有 N 组共同报告：

| 量 | 规则 |
|---|---|
| clean-J | 无噪声输入，固定同一评价图像与配对 |
| noisy-J | τ=0.3，所有方法共同 R_eval=16 |
| clean / noisy gate | 直接平均 1−T²，不用 1−J |
| noisy posterior risk | 同 noisy pair 计算平方风险；真实 S_sigma 未知时不写 oracle 误差 |
| frozen linear / kNN | 干净 h，规则与基准一致 |

N1 与 P95-R1 的最终 noisy-J 必须使用相同评价重复数，避免将测试时多抽样混作训练改善。

## 7.4 条件梯度方差诊断

在同一模型、同一固定 batch、同一组 shift 上重复采样噪声，测量 loss 及对 z 的梯度波动。分别构造 R=1 与 R=4 的平均，报告条件方差和耗时。基础图像／增强不变时，这是噪声方差；跨图像变化另外记录。

优先对输入 z 梯度使用在线 Welford 统计，避免保存大量完整 encoder 参数梯度。这里的诊断不选 checkpoint、不决定提前停止，也不把 1/R 的条件方差关系误写成全部训练方差缩小 R 倍。

# 8. 测试要求与已有参考代码

附件 `support/vcs_perf_core.py` 提供独立的 PyTorch 参考核心，不是仓库训练器的替代品。`support/validate_core.py` 已在 CPU/float64 运行。服务器先运行同一脚本，再把等价逻辑接入当前模块。

| 测试 | 应满足什么 |
|---|---|
| 风险恒等式 | 正负数量不同时仍有 J=1−R |
| 固定 critic | a,b 不进 optimizer；encoder/projector 仍有梯度 |
| 原路径回归 | 不启用新配置时，原模型初始 hash、配对、损失、首步梯度不变 |
| U mask | 自己和同图 pair 不进 Q；正负数量匹配公式 |
| U chunk | dense 与 chunk 的值、输入和 critic 参数梯度一致 |
| U 路由 | 相同 forward；完整有序 Q 上 detach 输入梯度恰为完整的 1/2 |
| 视图置换 | U 组重排视图不改变 loss 和对应输入梯度 |
| 噪声平均 | 平均每次 J 与 J(平均 T) 的差符合平方方差恒等式 |
| RNG 恢复 | 相同状态重放 noise；stop/resume 与不中断一致 |
| 噪声目标 | R=1 退化为 P95 期望定义，τ=0 退化为干净路径 |
| 评价状态 | clean/noisy 诊断均不更新 BN、不改变训练 generator |
| 梯度差分 | 完整可微路径的 finite difference 与 autograd 一致 |

对故意 detach 的路径，finite difference 要按被冻结变量处理；不能用“把同一输入两端一起扰动”的有限差分来要求它等于完整梯度。

单元测试后，每个新代码路径做真实 CIFAR 的 100-step smoke；测 steady-state 时间、峰值显存、梯度有限性、恢复一致性。固定 scorer 的 critic grad=0 不属于错误。

# 9. 记录能解释性能的诊断

按已有保存安排记录 epoch 0/20/100/400/800。中间诊断不读取 official test。

## 9.1 公共诊断

记录 a,b、零分数阈值（a≠0 时 −b/a）、正负 cosine 的均值／分位数、T 分布、实际门控、正负回归残差。

几何记录分开 h、p_raw、z：均值向量范数、中心化协方差谱、effective rank、alignment 与 uniformity。它们是解释数据，不是优化目标或停止门槛。

## 9.2 梯度组织

保存正项 G_pos、负项 G_neg 的范数、内积／cosine、总梯度范数；额外记录各视图收到的输入梯度范数。对每次诊断，分项梯度和合并梯度在同一固定模型与同一批数据上计算，避免更新模型后再比较。

G_pos、G_neg 的符号约定始终使用最小化 loss 的梯度。报告聚合梯度与样本梯度范数之和的比值，区分“很多大更新互相抵消”和“原本没有更新”。

## 9.3 来源与成本

每个结果记录基础图像数、编码视图数、正／负 pair 数、噪声重复数、critic 前向次数、encoder updates、critic updates、实际训练秒数和显存。

U 组增加 pair 计算；N1 增加噪声和 scorer 计算。不能只用 epochs 相同就称实际成本严格相同。既要报告相同完整训练日程，也要报告时间—性能取舍。

# 10. 性能比较与后续确认

## 10.1 首批选择

首批八组全部完成后，以 final h 的 linear-val 排序选择最多两个 VCS 候选。kNN 放在相邻列；linear 差距小于 0.1 点时，优先选择实际计算更低、结构更简单的配置。

选择后冻结全部超参数，不根据 seeds 1、2 改变配置。匹配 JS 控制按对应机制补齐：噪声候选进入确认时，N2 必须至少三个种子；固定刻度候选进入确认时，补相同初始化的 free-a,b 控制。

## 10.2 完整确认

第一层：同一标准增强、同一训练与评价协议，候选与 P41 标准增强 SimCLR 的 seeds 0/1/2 比较。

第二层：对稳定正向候选，补 seeds 3/4，并给对照相同种子；再在双方完全相同的强增强块上确认。不要只给 VCS 更强的数据增强或更多 encoder 视图。

报告逐 seed 准确率、成对差值、均值、标准差和区间。不要因少于某个任意 1-point 门槛而删掉可重复的小改善，也不要将一个选中的 seed-0 极值称为胜出。

第三层：在冻结配置之后做 CIFAR-100 确认，沿用 P98 的统一 probe 处理或共同预先固定的充分收敛 probe。图像测试集的访问仍按各协议既有授权；新任务默认只读 selection。

## 10.3 早期训练规则

候选只有出现非有限数值、数据／实现错误或基础设施故障时才暂停。弱 kNN、低 J、低 rank、未达到预设“学习门槛”不终止本轮已批准的完整训练。

基础设施中断从合法 checkpoint 恢复，并记录耗时与重做步数。预算不足时排队或报告缺口，不把未完成的模型作为 800-epoch 结果。

# 11. 进一步优化的触发条件

这些是后续候选，不与首批八组同时展开。

**固定刻度有效：**做相同初始化 free 控制，然后考虑有界可学习 a、κ。推荐参数范围 $a\in[0.5,4]$、$\kappa\in[-0.5,0.5]$，但需要另行固定配置；禁止隐藏的自由输出 affine 取消这个限制。

**完整 pair 有效：**比较 current K8、跨视图 all-pairs、所有视图 all-pairs，区分覆盖增加与角色平衡；直接测量各自配对采样方差。无需新 encoder。

**噪声有效但 kNN 弱：**先看 noisy-J 和条件梯度方差，再比较 R=1/4，必要时继续少量 R。不要只因为噪声较小便一次性扩大十几个 τ。

**自由 critic 的求解仍是瓶颈：**考虑同训练态特征的小步阻尼更新。原 P95 的 eval-mode 完全刷新加旧 Adam 状态不直接复制。此路线排在 G/U/N 后面。

**多个头值得尝试：**使用原始 J 的平均，而不是 J(平均 T)；明确两者的分歧项差异。分组头可能冗余，必须通过完整训练确认其效用。

# 12. 当前代码需要额外注意的两处具体问题

**噪声评价路径。** `NoisyCosineCritic` 只在 training 状态加噪，而 `critic_holdout` 使用 eval。既有 clean-J 保留原名，新增加 `heldout_J_noisy`；不要覆盖历史数值，也不要把历史 clean-J 解释成 noisy estimator 的估计值。

**吞吐量和刷新成本。** 当前训练器部分 `views_per_s` 仍用 2×batch，四视图时应使用真实 V。P95 refresh 的时间应单列并计入方法总训练成本，不能被 epoch timer 之外的更新遗漏。修改只影响新汇总字段，不重写旧结果；为历史数字另附计数纠正表。

这些问题不直接改变已报告准确率，但影响新的效率结论。

# 13. 交付格式

每个阶段提交：实际 commit、代码 diff、完整 resolved config、初始化 hash、数据 manifest hash、测试日志、全部运行的机器可读结果与短报告。

结果表至少包含：

```yaml
run_id: <unique>
logical_variant: G1|G2|G3|G4|U1|U2|N1|N2
code_commit: <actual>
objective: original_vcs_J|matched_js
estimand: S|S_sigma
critic_affine: {mode: fixed_or_learned, a_init: <a>, b_init: <b>}
negative_routing: full|negative_right_detach
pair_scope: cross_view_K8|all_view_tokens
noise: {tau: <tau>, draws_train: <R>, draws_eval: 16}
pretrain_seed: <seed>
epochs: {planned: 800, completed: <actual>}
linear_feature: h_before_projector
selection_linear_top1: <result or null with reason>
selection_knn_top1: <result or null with reason>
clean_J: <result>
noisy_J: <result or null>
encoded_views: <actual count>
critic_pair_evaluations: <actual count>
training_seconds_total: <including extra updates>
peak_memory_mb: <measured>
status: completed|failed|interrupted
```

报告按三个问题组织：**哪项改变提高了最终表示？数据是否支持预定的机制解释？下一轮需要哪一个有限对照？** 不只交付最大精度，也不把负面候选从汇总表移除。

# 14. 本轮开始前给主 Agent 的简短指令

> 读取本规范及配套分析稿，核对实际 HEAD 与已执行分支。复用 P95/P98/P99，不覆盖历史配置。保持原 J、tanh 和无标签 product negatives，按 G/U/N 三条线生成八组完整 CIFAR-10 训练。先通过单元、恢复与真实数据 smoke，再按现有 GPU 权限排队执行 800 epochs；不要以早期 J、rank 或 probe 门槛删掉有效运行。主报告使用冻结 h 的最终线性精度及成对种子比较，机制指标单独解释。
