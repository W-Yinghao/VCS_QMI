# 源码审查：事实、影响假说与修正方式分开

## R8-01 参数冻结 ≠ 确定性 critic

**源码事实**：[SRC-02] 的 `Block.forward` 同时调用 `F.dropout(... training=self.training)` 与 `nn.Dropout`；`predict_r` 结束后无条件 `model.train()`。`train` 进入 phase B 时只把 `estimator_parameters` 的 `requires_grad` 设为 False，没有把 encoder blocks 切到 eval。[SRC-03] 的 Qwen `train40` 具有相同路径。D1-FULL wrapper 复用原训练器；因此不能只检查 10k 的日志而忽略 full-data 版本。[SRC-04]

**范围**：这是被审查函数的默认状态链，尚未在服务器真实 checkpoint 上记录过 dropout 贡献。若外层 wrapper/hook 在实际路径另外控制 mode，请给确切位置，不能由审查自行假设。

**性能假说**：训练 gate 时，它可能同时最小化观测删除的影响与随机 critic 的方差。方向可能有利也可能有害，不能写成既定 bug 导致全部差距。

固定输入与固定 mask 时，令 B/A 输出的随机性来自 dropout：

```text
E[(T_B − T_A)²]
= (E[T_B] − E[T_A])² + Var(T_B) + Var(T_A) − 2 Cov(T_B,T_A).
```

只有 dropout 独立时才省去 covariance。实际 gate mask 又依赖全路输出时，不能不加条件地代入这个简化式；先固定 mask 做诊断。

**最小核对**：记录所有模块 training flag；同输入无压缩时两次独立前向；eval 模式下重放；冻结参数哈希不变；mask/gate 梯度有限且在非退化例子中非零；旧残留 `.grad` 清零后再核对。

**建议修正（需要新运行版本）**：phase B 用确定性 frozen estimator，gate 仍可训练；保留显式 Gumbel 策略探索。`model.eval(); model.gate.train()` 适用于当前 gate 无 Dropout 的结构，但实际 patch 应逐模块明确状态，防止后续 `model.train()` 又打开 blocks。对 mask 梯度的路径不能包在 `no_grad()` 或 `inference_mode()` 中。[EXT-01]

`no_grad` 可用于不依赖 mask/gate 的固定 full-pass 缓存，但不能用于压缩前向。不得修改 live 模块供正在运行作业继续 import。

## R8-02 现有 soft–hard gap 混合多个变化

**源码事实**：`soft_hard_gap` 把 Gumbel 扰动后的软 mask 与无扰动的 hard top-k 比较；Qwen 的同类函数须按当前版本核对。它不是同一潜在排序下的松弛误差。[SRC-02, SRC-03]

现有读数继续保留，但改名/标注为 `soft_noise_minus_hard_deterministic`。新增四格读数：

```text
HD = risk(hard_topk(g))
SD = risk(soft_topk(g, τ))
HN = risk(hard_topk(g + ξ))
SN = risk(soft_topk(g + ξ, τ))        # HN/SN 使用同一 ξ

relax_det   = SD − HD
relax_noise = SN − HN
noise_hard  = HN − HD
legacy_gap  = SN − HD = relax_noise + noise_hard
```

eval critic 下做主分解；train critic 为随机性诊断，须共享/独立 dropout 的实验条件明确，不通过不同前向的浮点噪声伪造精确分解。

另外，有限温度下 `sum(m)` 不必等于 k，`log(m+1e-6)` 的 floor 让软路径不等于真正删除。记录 mask mass、有效支持、阈值两端间隙、hard-noise/deterministic 的 Jaccard。**不预设这是错误设计，也不先改 eps 或温度。**

## R8-03 本轮新增：Spearman 对 ties 与常数输入处理不正确

**源码事实**：当前 `spearman` 对输入各做两次稳定 argsort，生成的是按原始顺序打破 ties 的 ordinal ranks；随后检查的是这些 rank 的方差，而非原输入是否常数。它被 phase-A checkpoint 和学习率选择使用。[SRC-02, SRC-10, SRC-11]

```python
ra = np.argsort(np.argsort(a, kind="stable"), kind="stable")
rb = np.argsort(np.argsort(b, kind="stable"), kind="stable")
```

**独立反例**：`a=[0,0,0,0]`、`b=[0,0,0,0]` 返回 1；标准 Spearman 对常数输入未定义。并列值通常应使用平均名次，而非给相同分数分配不同排名。[EXT-02, EXT-03]

**不能推断**：这不说明当前所有 CAL 数值偏高，也不说明已选 checkpoint 必然错误。若数据几乎无 ties，差异可能很小。已有官方回答准确率仍是相应 frozen 模型的真实读数。

**必须补的影响分析**：
- 输入 r/h 的 unique fraction、最大 tie block、零值/饱和值比例；不按任意 eps 合并近似值。近似 ties 单列精度诊断。
- 同一数组的 legacy 与平均名次 Spearman；同步置换行顺序后是否变化。
- 当前选定 checkpoint 的复算；若完整历次预测存在，再复算选择路径。只有 scalar logs 无法反推出完整 tied ranks，必须写明。
- 若模型/学习率选择确实改变：保留 legacy 选择结果；corrected selection 是独立敏感性实验，不能把测试上更好的那个反选回主表。

本包 `audit_core.spearman_average` 是严格处理 finite inputs/ties/constants 的参考；官方 SciPy 是交叉核对，不要求更新服务器所有依赖到当前线上版本。

## R8-04 pooled CAL 相关性没有直接评价每次选择

**源码事实**：选择准则把所有问题、mask 和预算展平后计算相关性。此方式可以因预算差异而得到较高相关，却在同一问题的可选 mask 上不做正确选择。[SRC-02]

**这是指标与用途的差别，不是原准则违法或无效**。原冻结规则不覆盖，新加同问题同预算的 Spearman（仅 3 masks 时仅描述）、成对方向准确率、有限 mask 集合后悔值和 top-choice 实际 h。tie/no-spread 子集需单列分母，不加随机噪声制造排序。

## R8-05 策略产生 mask 后，测量分布可能改变

**源码事实**：phase A 在预先生成的 random/grid/qsim ×4 budgets 上拟合；phase B 用可学习 gate 最小化已冻结 critic 的风险。[SRC-02, SRC-12]

**性能假说**：policy-induced mask 可能被系统性低估。证明需要在相同图像与预算下比较原 mask 和策略 mask 的预测误差/宿主实际风险，不是看训练 loss 降了就宣布“reward hacking”。

如果要新增 on-policy 数据：只在 FIT 图像收集，记录完整采样政策、host 调用与类别权重；新策略导致的观测分布改变必须写清。不能把原 mask 下的 Bayes 后验解释无条件搬到新策略上。

## R8-06 小 critic、宿主、真实标签是三个对象

S1/S2 的 T 在 PCA token + frozen word embedding 上预测 GQA binary label；它不是宿主内部预测，也不是自然数据总体 Bayes 后验。S3 的目标是宿主实际风险变化。S1 的平方差与 h 不同，不是因为 S/J 恒等式错误。[SRC-02, SRC-12; H01]

后续宿主校正建议见数学文档。不能同时以“学习自己的后验”作为目标，却默认把平方差当成精确的宿主压缩损失。

## R8-07 Full-data 改变样本量，没有改变全部约束

D1-FULL/E2 仍是 binary GQA、相同特征降维与小 transformer、20k+5k steps。若结果不变，只能评价这个 data-scale extension；不能说任意更充分优化都无效，也不能在结果前承诺“数据更多一定修复”。[SRC-13, SRC-16]

## R8-08 单位、模型状态与报告一致性

检查 loss 标度：按平衡权重对全部正负样本做 weighted mean 的 `a_plus/a_minus` 值可为总体 J 的 1/2。只有取决于共同常数时，最优解不变；学习率/梯度尺度和 native 图表单位仍要标明。**这不是本轮已经认定的新主 bug**，先以手算有限 batch 对齐，避免 raw native value 被误写为公共 J。

原 `.train()` / `.eval()` helper 应保存进入状态并恢复，而不是永久改变调用者的子模块 mode。历史 helper 的无条件 `model.train()` 已知；新 audit helper 应用 context manager 恢复各子模块 flag。

## R8-09 已修复：R7-50 顶层 pass 检查

`54e1715` 改为解析 JSON 顶层 `pass is True`，并等待 NFS 文件；旧 grep 可能被嵌套 passing 行误触发。提交说明 phase L 的顶层本就为 true。[SRC-01]

本包接受它为源码层已修复。服务器再核对 phase-F 实际使用 code hash 和 gate 记录；不据此追溯宣布所有 official evaluations 无效，也不重做相同修复。
