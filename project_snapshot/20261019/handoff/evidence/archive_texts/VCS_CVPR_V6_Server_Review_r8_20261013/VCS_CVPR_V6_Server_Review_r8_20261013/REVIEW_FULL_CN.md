# VCS-QMI / CVPR V6：服务器审查全稿 r8

截点：54e1715e29dd2caaa6be7ed8e670829310951d39；2026-10-13。


---

<!-- Source file: docs/08_DELIVERY_UPDATE_CN.md -->

# 交付前增量核对：54e1715 → 3e2fe59

**最后核对：`3e2fe59c4861abae46537574c4c59f9bedfe91c6`，2026-10-13 18:51:49 UTC。**

正文源码审查基线仍为 `54e1715`，保留证据的版本边界。准备材料期间又出现 4 个提交；本补充记录它们对接续任务的影响。

## 当前实际差异

- R7-50 将 timing gate 放在 manifest 前，尝试提前启动 phase F 的 integrity 工作；随后发现 integrity-prepare 仍需要全部 reference shard markers，提前启动在 CPU 准备步骤退出，没有 GPU 单元/新测试文件读取。流程恢复 marker-triggered 接续。
- D1-FULL 的 ChartQA shard 1 重试允许 L40S/H100；新提交给后续提交的 eval wrapper 设置 `expandable_segments:True`。旧已提交 Slurm 脚本不会自动继承这个变量；该重试若再次 OOM，继续保留已有逐项预测。
- 提交说明把分配器修改定为执行配置，不改变分析或方法。它不提供新的完整全量性能结果。本包没有实测该分配器是否消除 OOM。
- **两个重点 critic 文件的 blob 在 3e2fe59 均与基线相同**：LLaVA `706a464…`，Qwen `9a68d1c…`。因此 dropout/mask/Spearman 审查仍指向同一源码，不是重复审查已修代码。

## 给服务器的影响

接入时以实际 job / marker 状态为准，不再直接照正文较早的“H100 重试”安排重新提交。phase F 当前依赖完整参考数据；不取消或复制已授权的接续流程。

审查辅助脚本认可两个已核对 HEAD（54e1715 与 3e2fe59），仍要求关键文件 blob 匹配；更晚版本需要显式 `--allow-newer` 并报告差异。

## 来源

- [比较记录（4 commits）](https://github.com/W-Yinghao/VCS_QMI/compare/54e1715e29dd2caaa6be7ed8e670829310951d39...3e2fe59c4861abae46537574c4c59f9bedfe91c6)
- [phase-F 提前启动与 reference guard](https://github.com/W-Yinghao/VCS_QMI/commit/86b5bc259ba56e01ab5451ad6620adbc6be34075)
- [恢复 marker-triggered 接续](https://github.com/W-Yinghao/VCS_QMI/commit/eaeb94a011d6cbd062d39abee71b2814a1f3a1bc)
- [ChartQA 重试与 allocator 记录](https://github.com/W-Yinghao/VCS_QMI/commit/3e2fe59c4861abae46537574c4c59f9bedfe91c6)

Git commit 时间与文内手写时刻略有差别，本包截点使用 Git commit UTC，不据手写时刻推断额外执行事件。


---

<!-- Source file: docs/02_REVIEW_CN.md -->

# r8 总体审查：完成现有验证，先修 critic—决策接口

## 0. 结论与来源范围

**建议：继续完成已批准的全量与外部对照，同时启动一个有限的 critic 审查。暂不增加新 VL 方向或大规模超参数网格。**

审查截点 `54e1715e29dd2caaa6be7ed8e670829310951d39`，上次分析 `3a5e2f0b19d5d32a3913c3916413bae05753eab2`。基线相对上次新增的是 R7-50 运行门控修复，不是新性能结果。交付前另核对到 3e2fe59：关键 critic blob 未变，执行增量见 docs/08_DELIVERY_UPDATE_CN.md。进度来自提交和报告，不是本地实时 Slurm 查询。[SRC-00, SRC-01]

本包区分五种陈述：
- **仓库结果**：原报告中的数值、实验条件和完成状态。
- **源码事实**：可从当前函数实现直接核对的行为。
- **独立推导／小例子**：本包重新推导或本地合成验证，不是自然数据结果。
- **性能假说**：可能解释当前结果，但没有被因果实验确认。
- **建议实验**：服务器评估后执行，不篡改历史协议。

用户提供的 10-10 综合文档有版本与附录勘误；它只作为研究历史，不能替代 10-13 的最新报告。原 S/J 和统计解释沿用合作者计划，不将新审查意见写成合作者已提出的理论。[H01, H02]

## 1. 已有实验的成熟度

| 实验线 | 本次已核对的状态 | 当前可支持的结论 | 下一步角色 |
|---|---|---|---|
| r5 grounding | 5 个官方 test 划分完成；query-level Acc@0.5 | 查询内归一化改善原 VCS，RefCOCO A/B 通过原多重检验；对 softmax 未建立普遍领先 | 结果收尾，不再默认扩展 |
| REG | BF–SHG 与 EPFL RGB–NIR 完成 | 所测共享搜索下，学习型局部 critic 的配准成功率较高；VCS 非独有 | 正面应用证据 |
| GAIN | AV-MNIST、CREMA-D 完成 | 实际收益回归有用途；有限模型平方分歧不等于收益 | 边界分析与附录 |
| D1 / LLaVA 10k | DEV/val 和官方 GQA/ChartQA 均完成 | 紧预算下 S1 多数低于 S2、SparseVLM，有时低于 grid | 不把它描述为成功的压缩方法 |
| R7-40 / Qwen 10k | DEV/val 和官方 GQA/ChartQA 均完成 | 官方测试 S1 各预算低于 S2 和 grid；TextVQA val 的局部相对优势不能替代全局结论 | 明确宿主/任务交互 |
| D1-FULL | 已进入剩余评价和计时，ChartQA OOM 后转 H100；最终报告未读到 | 尚无全量自然结果可引用 | 按原冻结协议完成 |
| E2 / Qwen full | 教师、缓存及后续流程已冻结并推进 | 无完整结果可引用 | 原批准范围内继续 |
| E1 | MMTok/HiPrune 24 单元完成门控和汇总流程已设置 | 尚无完整对照表可引用 | 补外部参考 |

来源：[SRC-05–SRC-16]。服务器接入时必须更新状态，已有完成报告优先于本表；不得把“提交脚本存在”写成训练已完成。

## 2. 主要结果快照

### 2.1 区域定位：critic 结构的正面结果

三个种子均值；全部官方表达 query-level Acc@IoU 0.5。VQ 是查询内归一化 VCS；JQ 是同结构 Logistic；D/E 分别为分布式/集合 softmax。

| 方法 | RefCOCO A | RefCOCO B | RefCOCOg | RefCOCO+ A | RefCOCO+ B |
|---|---:|---:|---:|---:|---:|
| VCS | 73.07 | 56.77 | 72.77 | 79.21 | 60.71 |
| VQ | 78.64 | 65.50 | 73.60 | 80.56 | 63.35 |
| JQ | 78.09 | 65.49 | 73.66 | 80.61 | 63.03 |
| D | 79.74 | 67.39 | 74.82 | 81.26 | 63.73 |
| E | 80.08 | 67.29 | 74.66 | 81.30 | 63.24 |

VQ−VCS 在 RefCOCO A/B 为 +5.57/+8.73，按原规则确认；其他三个正差未全部通过家族检验。VQ 与 JQ 未清晰分离；“区间含零”不是等效性定理。保留融合行作为单独、相同检测器信息的对照，不能只挑 VNU 高于未融合 E 的比较。[SRC-05]

### 2.2 Token 选择：已完成的首轮没有显示预期优势

为直观展示，以下固定 **25% token**，不是插值时延主表。S1=VCS 后验差、S2=Logistic 后验差、S3=实际风险回归。

| 官方任务/宿主 | grid | S1 | S2 | S3 |
|---|---:|---:|---:|---:|
| LLaVA / GQA testdev | 58.70 | 54.76 | 57.74 | 55.46 |
| LLaVA / ChartQA test | 14.80 | 14.91 | 15.57 | 13.77 |
| Qwen / GQA testdev | 58.48 | 53.67 | 55.07 | 54.22 |
| Qwen / ChartQA test | 54.84 | 38.59 | 51.87 | 39.49 |

LLaVA 的适配 SparseVLM 对应 25% 行为 58.79/16.56。Qwen 没有达到冻结的加速工作点，不作加速主张。TextVQA val 的 Qwen S1=66.01，高于 S2=51.19 和 S3=57.46，但仍低于 grid=73.18；LLaVA TextVQA 上的 S1/S2 方向不同。因此这是局部交互，不是“VCS 擅长 OCR”的结论。[SRC-06–SRC-09]

### 2.3 配准：正面但不能独占归因

| DEV 刚体配准成功率 | VCS | Logistic | InfoNCE | CoMIR 表示+同一优化器 | MI |
|---|---:|---:|---:|---:|---:|
| BF–SHG，<42 px | 94.2 | 93.1 | 94.9 | 90.0 | 74.1 |
| EPFL RGB–NIR，<10.5 px | 99.3 | 99.2 | 99.4 | 96.2 | 93.8 |

阈值按各数据尺度定义，不能跨行直接比较精度。EPFL 上约半数 MI 失败为该共享优化器的搜索失败；CoMIR 不是完整原生系统。仍然是可用的视觉应用，但不写成一般的“CS 优于 MI”或“VCS 优于所有神经损失”。[SRC-14, SRC-15]

## 3. 审查优先级

| 编号 | 优先级 | 本轮意见 |
|---|---|---|
| R8-01 | 高 | 参数冻结与 eval 状态分开；阶段 B 的 dropout 必须实测并记录 |
| R8-02 | 高 | soft–hard 诊断分离 Gumbel、松弛与 dropout，不再用混合差直接解释机制 |
| R8-03 | 高 | **新增**：Spearman ties/常数处理错误需要局部修正及选择影响审查 |
| R8-04 | 高 | pooled ranking 不够：同问题同预算、候选 mask 选择与实际宿主风险必须相连 |
| R8-05 | 中 | gate 生成的 mask 可能超出 critic 训练分布；须检验而非默认成立 |
| R8-06 | 中 | 小 critic 的任务后验差并非宿主风险差；主张和后续模型应明确连接 |
| R8-07 | 中 | 固定训练步数的 full-data 只检验数据规模，不保证优化已收敛 |
| R8-08 | 中 | 原测试、时延和效应量口径保留；新结构不得冒充盲确认 |
| R8-09 | 已修复 | 最新 integrity 顶层检查已修；核对运行记录，不再重复开修复任务 |

## 4. 本轮不建议改变的内容

原理论 S/J、不含新噪声的主训练分布、已完成官方测试行、已冻结的 full-data 训练日程和既有作者决定均保留。当前证据足以评价**已实现的方案**，不足以否定全部潜在 VCS 应用；反之，发现实现风险也不足以使已观察到的负结果失效。

不要把重点误变成“寻找 VCS 必胜的设置”。本轮要回答的是：同一个已训练 critic，在正确的测量状态和相同 mask 下是否提供可靠的决策信息；若不可靠，失败来自哪里。

## 5. 第一批交付

先交接入记录、源码/合成验证、CAL ties 影响、四格 mask 诊断、同问题同预算评估。每一项允许反驳审查：给证据即可。只有证据指向明确可修因素，才提交有限的新训练批准单。详细定义见其余文档。


---

<!-- Source file: docs/03_CODE_FINDINGS_CN.md -->

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


---

<!-- Source file: docs/04_DIAGNOSTIC_PROTOCOL_CN.md -->

# 第一批诊断：先量化影响，再批准修改训练

## 0. 统一约定

这是一轮新的审查，不是原始预注册的补写。记下首次看到当前结果的时间。本包不修改主任务权重、最终测试评价或正在运行作业。所有新增输出写独立目录，如 `outputs/REVIEW_R8/`；仓库报告编号先查重，建议前缀 `REV8-`。

首批自然数据只用现有 FIT/CAL，禁止以官方 GQA testdev、ChartQA test 和 RefCOCO test 选择审查结论或超参数。DEV/val 可用于事先写明的描述性复验，但已经不是全新独立证据；默认先不扩大。

采样单位：image/chart → questions → masks → stochastic repeats。不同 mask、Gumbel 或 dropout 重复不当作独立图片。S1/S2 用同一 image/question/mask 列表；可复用 S3 作任务级控制。缺失、NaN、常数、全同 h 必须保留计数。

## REV8-00 接入与作用范围

**目的**：确认被审查的是正在使用的代码，而不是一个已经被修好的旧版本。

交付：
- HEAD、worktree dirty 状态、两个 critic 模块 blob/SHA256；live job 的 code hash、resume version；不要只记录 main 的名字。
- D1-FULL/E1/E2/phase F 的 completed/running/queued/unknown 与依据，不编造进度百分比。
- 官方测试的模型/规则冻结记录；哪些测试已读；新修改需要何种 owner decision。
- 本轮审查用的阶段 A checkpoint 是否存在。只有最终 gate 权重时，不能声称已经恢复同一 A 起点和 gate 初始化。

辅助命令 `repo_source_audit.py` 只读 AST、git 状态和文件字节；`repo_runtime_probe.py` 仅在显式允许时导入原模块，使用微型合成输入记录训练模式。

## REV8-01 模式、随机性、梯度路径

### 样本与范围

先用合成输入检验定义；再对每宿主每目标 seeds 0–2 的已选阶段 A critic 做自然数据诊断。初始自然子集固定为 CAL 哈希最小的 128 张图、每图最多 4 个现有二元问题；数据不足用全部并记录。这个数是审查预算建议，不是科学效应预测。每个 stochastic condition 重复 32 次；主预算先用 25% 和 50%，全保留作零改动 sentinel。

### 条件

1. `eval / fixed hard mask`：同输入重放，重复输出与梯度（必要时容差，不强求跨 GPU bit-identical）。
2. `train / fixed hard mask`：固定 mask，独立 dropout。
3. `train / same-input B=A`：量化没有观测改变时的自分歧。
4. `train / common RNG`：仅诊断 common randomness 影响；明确 shapes/forward 次序相同，否则不能说 dropout 精确耦合。
5. `eval / soft mask + autograd`：冻结参数、可训练 gate，mask/gate 梯度有限且非零；parameter checksum 不变。清除旧 A.grad 后再判断无新 critic 梯度。

输出 per question 的 mean/std(T_B,T_A)、mean squared gap、covariance、self-gap、grad norm、mask-grad、module modes。重复估计的方差作为诊断，不用 32 次随机前向得到一个“独立样本 n=32”的自然性能 p 值。

**判断**：可以确认 train-state stochasticity 的量级；只有再训练 gate 的配对干预，才能判断它是否造成任务差距。若差异接近数值误差，就如实结束这条解释。

## REV8-02 四格 mask 与预算偏差

保持 critic eval、同输入固定 full output/无 dropout，取相同 gate logits。对每问题预算生成一次 ξ 后比较 HD/SD/HN/SN，另重复 ξ 得到稳定分布。包括 τ=当前最终温度 0.1；τ=1 与接近硬选择的值只作预先指定诊断，不能据任务成绩挑温度。

记录：
- 四个 risk 与 `SN-HD = (SN-HN)+(HN-HD)` 数值检查。
- `sum(m)-k`、Σ m(1−m)、m>0.5 的数量、有效 key 支持；mask floor 总质量约 `(N−k)*eps` 只是输入质量，不等于实际 attention 质量。
- HD/HN 交并比、排序一致性、不同预算 nesting 是否符合当前设计。
- deterministic soft/hard 输入对应的 pooled output、T_A 与 mask 梯度。

**关键**：旧 gap 同时换噪声/松弛，继续原样报告但改注释；只要发现 gap 不能就自动切 hard/ST 训练。硬前向 STE 的偏梯度是一种新求解器，需独立规范和对照。

## REV8-03 Spearman 审查与选择影响

使用现有 CAL predictions 的精确未四舍五入数组。主结果：legacy ordinal Spearman vs average-tie Spearman；constant 返回 `null` 并给 reason。tie 统计按真实存储精度计算；只用于展示的 round 后数值不可用于复算。

顺序：
1. 先对已选模型重新预测原固定 mask，检查读数变化。
2. 对相同数组共同置换 20 次，记录 legacy 和 corrected 的变化；正确 average Spearman 应对共同置换不变（浮点容差）。
3. 有逐 checkpoint 预测才复算历史 argmax；只有 selected/last checkpoint 就报告 incomplete coverage，不从相关性 scalar 反推重选。
4. 记录对 LR 排名、selected step 的潜在改变，**不访问测试结果来选择**。

替代代码必须通过：无 ties 一致；两边常数返回未定义；一边常数返回未定义；ties 使用平均名次；common permutation invariant；NaN/Inf/长度不符显式处理。不能加随机 jitter。

如果调整选择准则，视为新选模规则。原论文原结果与 corrected-sensitivity 并列；仍然能报告旧 frozen 模型的任务结果。

## REV8-04 同问题同预算的决策评估

定义每个 `(image_id, qid, k)` 的有限候选 mask 集合 M_qk，现有三种 mask 是起点。不同 masked inputs 若字节完全相同，记录 duplicate 关系；不把复制项当作新增证据。

```text
r_j = critic 预测的压缩风险
h_j = 该 mask 上实际宿主平方风险相对完整宿主的变化
j_hat = argmin_j r_j
j_oracle = argmin_j h_j                # 用标签的经验参考，不可部署
regret = h[j_hat] − min_j h[j] >= 0
```

预测并列时用预先固定的 mask_id 字典序，另报随机平均 tie 作为敏感性可选；不能用 h 打破预测 ties。

主要读数：
- 图像宏平均 regret、相对 uniform/grid 的实际 h 变化；同时展示各策略 h、d 与 residual，不只给差。
- within-query/within-budget pairwise direction accuracy：对 h 非零差的 pair，预测 tie 记 0.5；h tie 数量单列。
- within-group Spearman（3 masks 很粗，只描述，不用大量小组 Pearson p 值）；no-spread 子集与缺失组单列。
- 按预算、任务类型、宿主、正确/错误 full prediction、视觉质量分层，不挑最好格作核心结论。

`min h` 只是有限 mask 集合、一个真实标签下的经验 oracle；不是 Bayes-optimal compression，也不代表这份收益可从输入预测出来。

### 策略 mask 的外推诊断

在同一 CAL 子集加入已有 gates 的 hard-deployed mask（S1/S2/S3），宿主只补尚未存在的前向。宿主调用数量 = unique missing (question,mask) 对数，预算先测。

比较 `r-h` 的 signed bias、MAE、h−d、实际错误率；h 可为负，不截断。若 r 小但 h 大，才讨论潜在低估；不要因为某一条差很多就宣布整个 critic 被利用。

另在 FIT 留出的审查子集以同规则复核，区分训练数据过拟合与 mask-policy shift。不可把新 CAL 结果用于多轮 gate 调参后继续称它独立确认。

## REV8-05 表示覆盖与接口（次优先）

只有前四项无法解释失败时才扩大：记录 token 坐标覆盖、网格空洞、空间聚集、OCR 区域保留（需独立可用标注/检测，不新增不可核对标签）。先用 grid/random 作等预算参照。

Qwen 的可变 N、padding、原 M-RoPE 位置与 selector 因子位置表应维持已验证接口。检查 PCA 投影在不同任务/位置的误差可以作为诊断，**不可据此自动改为更高维或更大模型**。

## 首轮统一输出

- `intake.json`、`source_audit.json`、`runtime_probe.json`（若运行）。
- `mode_probe.csv`、`mask_fourcell.csv`、`rank_audit.json`、`mask_decisions.csv`。
- `SERVER_RESPONSE_CN.md`：逐条证据、接受/反驳意见、缺口、是否建议新实验。
- `run_manifest.json`：hash、数据 id 范围、种子、dtype、GPU、采样权重、耗时，不存用户访问令牌。

本包参考 CSV 汇总器只覆盖定义与分组，不实现宿主推理或完整统计报告。服务器须对接本地已有模块，不能把脚本样例输出当自然结果。


---

<!-- Source file: docs/05_CONDITIONAL_EXPERIMENTS_CN.md -->

# 条件修改实验：只改变被证据指向的因素

## A. 当前正在跑的冻结实验照常完成

D1-FULL / E2 / E1 / 原授权 phase F 的代码、训练长度、CAL 选择和测试行不因审查静默变动。若代码问题严重到负责人决定暂停，须独立记录 affected units、尚未执行部分、保留预测和授权；本包不是自动暂停命令。

新审查分支用独立 namespace 与输出根。不得在同一 `S1F` 行混用旧/新 mode、Spearman、gate 求解器或不同模型版本。

## B. 最小可解释干预：相同阶段 A，改变 B 的 critic mode

**触发**：REV8-01 确认 phase-B dropout 有非平凡贡献，并且服务器认为它与部署差异值得干预。不是要求先看到任务结果变好才运行。

**首轮范围建议**：LLaVA，一个已完成尺度（优先 10k，避免 full-data 并行更改）；VCS/Logistic × seeds 0,1,2。所有六个训练对都完整运行，不按 seed 0 胜负筛选。每个方法/seed 两个 arms：

| arm | phase A | gate 初始权重 | critic mode | Gumbel/预算/步数 |
|---|---|---|---|---|
| L：legacy restart | 同一已选 A state | 同一预记录 gate init | train | 原 5k 与原 mask 求解 |
| E：deterministic critic | 同一个 A state | 同一个 gate init | eval，gate 可训练 | 同 L |

这是 **6 个新增 E + 至多 6 个匹配 legacy 重放**，不是“只需 6 次”无条件承诺。若现有 legacy 的阶段 B 状态、初始化和随机流可严格复现，才复用原 L 并先做一致性检查。否则按最多 12 个 B-only 单元给预算。

配对输入 schedule（qid/k/Gumbel seed）独立于 dropout RNG；不通过重设所有全局随机种子把 dropout 人为耦合成一种不同算法。初始 A weights/gate init/hash、optimizer、dtype、训练步数、CAL 动态阈值过程分别记录。

### 不同时改什么

不改 Spearman、不改 A checkpoint、不改 warm-up、不加覆盖正则、不换数据、不改 soft top-k。否则不能把结果归因于 mode。若 R8-03 需要 corrected selection，它是另一个单独的 sensitivity experiment。

### 评价

先完整宿主 GQA DEV 二元 + 原开放式任务；TextVQA/ChartQA val 为已知 transfer 复验。报告 fixed-budget accuracy、Brier/h、grid/SparseVLM 等预算参考、总时延（包括 selector）和 mask 覆盖。

主要效应是 within-method E−L，VCS/Logistic 的交互为次级，不能只拿新 VCS 与旧 JS。旧 official test 只作已完成历史结果；新测试访问需负责人指定新的确认方案。

## C. 若真正问题是软硬求解

先完成四格诊断，再选一个方案，而不是同时试五种温度：
- 硬前向 + 有明确说明的 surrogate backward；或
- 在固定预算的有限 hard mask 集合内作监督选择/排序；或
- 将软训练 mask 加入 critic 训练，使输入机制相符。

三者都属于不同算法选择。STE 的反向一般不是硬选择真实梯度；不得把它称作“精确梯度修复”。主控制仍为相同 critic 的旧 solver 与匹配 Logistic。每条自然结果都需要完整宿主前向。

## D. 若 Spearman 确实影响选模

先保存当前被选模型、历史表和读取过的测试集。corrected rank 指标只在原 CAL 上重新计算，单独报告 selection sensitivity；缺失的历史 checkpoint 不补写。新的 A 训练若需要重做，预算与比较单列。

不以 corrected selection 在测试上表现更好为采纳依据。实质上改选模之后，旧盲测试已经被使用，新的独立确认需要额外设计或明确标为 repeated-test sensitivity。

## E. 若 critic 没能估计宿主损失

先看同问题同预算排名及 policy-mask 外推；再考虑宿主校正风险读数。新方案至少包含：
1. 现有后验差 S1/S2；
2. 直接实际风险回归 S3；
3. 预测宿主行为的平方差；
4. 加校正项的读数（见数学文档）。

不只更换损失，需定义每个头预测什么、训练标签哪里来、部署前可见什么。允许第二方案最终不如 S3；这否定该结构的实用收益，不是否定 S/J 恒等式。

## F. 若 mask-policy shift 成立

仅用 FIT 图像做一轮固定预算的数据聚合：原 mask、当前 policy mask、grid/random 共同收集；三目标收到相同集合。宿主新调用成本列入训练成本，CAL 不回流 FIT。新增策略不能读取标签/宿主答案才决定是否获取完整输入。

## G. 若空间覆盖是主要问题

同一总 k 下，固定少量 grid 覆盖 + 学习剩余预算。覆盖配额在新训练之前决定，不根据 val 看哪一种最优后隐藏探索。S1/S2/S3 都能使用相同规则，保留纯 grid 和原 learned gate。此方案是任务归纳偏置，不预称 VCS 专属创新。

## 终止／收束规则

- 完成一次定义清楚的干预后，若模式/solver 的任务影响很小，就结束该解释；不反复改变阈值直到显著。
- 若完整数据与合理实施下仍低于 grid/公开方法，当前 D1 作为边界结果保留。REG/r5 的正面发现不被它抵消，也不被它借用来宣称 D1 成功。
- 不从任何一项失败推出“必须退出 CVPR”或“所有应用不可做”。范围始终对应实际实验。


---

<!-- Source file: docs/06_MATH_AND_CRITIC_OPTIONS_CN.md -->

# 数学与 critic 设计：哪些是原理论，哪些是本轮建议

## 1. 保留的统计对象

合作者计划给出一般 P/Q 下：

```text
M = (P+Q)/2
η = (p−q)/(p+q)
S = E_M[η²]
J(T) = E_P[T−T²/2] + E_Q[−T−T²/2]
S−J(T) = E_M[(T−η)²]
D_CS^(M) = 2 log((1+S)/(1−S)).
```

当 P 为联合、Q 为边缘乘积时才称该实例为依赖/QMI。当前 D1 将 C=±1 定为任务标签，P/Q 是任务类条件观测，不应称“图文联合对乘积的 MI”。Logistic 拟合后 T=2D−1，同样能使用 J 的 gap；求解器与读数分开。[H01, H02]

基础二次泛函与平衡 RPC 的关系已经由项目承认；本包没有重新做排他性新颖性检索，不把归一化、常见条件期望恒等式或 dropout 控制宣称为新理论。[H02]

## 2. 归一化 critic 的用途（已有结果，不扩展至所有任务）

```text
Z = Σ_j ν_j exp(2f_j)
g_j = f_j − (1/2)log Z
ρ_j = ν_j exp(2f_j)/Z
T_j = tanh(g_j) = (ρ_j−ν_j)/(ρ_j+ν_j).
```

ν_j>0 且总和为 1。对条件候选分布，Σν exp(2g)=1，与真实密度比约束相容。有限 a·cos+b 类仍未必容纳真实后验。固定 f 时归一化减去公共常数，但训练梯度耦合不同；不要用固定 argmax 不变否认训练改进。[SRC-05]

这个结构适合“候选间有竞争”的任务，不应机械搬到“多个视觉区域都必需”的压缩问题。

## 3. 随机 critic 的平方分歧

固定输入、固定 hard mask 下，一般有：

```text
E[(X−Y)²] = (EX−EY)² + Var X + Var Y − 2Cov(X,Y).
```

如果 B=A 而 dropout 独立、均值相同，期望平方差为 2Var X，而不是 0。若 gate 的 mask 依赖 X，本式仍对联合随机变量成立，但不能将条件均值当作固定 mask 下的均值；更不能从实际 gate loss 直接减去一个估计方差声称得到无偏信息损失。本轮因此先把 mask 固定后做诊断。

## 4. 平方增量与有限宿主风险

原条件期望投影：固定观测机制，A 是 B 的可测函数，在同一个平衡任务分布上：

```text
η_B = E[C|B], η_A = E[C|A]
S_B−S_A = E[(η_B−η_A)²] = R_A*−R_B*.
```

有限宿主输出 b(B), a(A) 时则为：

```text
h = (C−a)²−(C−b)²
d = (b−a)²
h = d + 2(C−b)(b−a).
```

b,a,C∈[−1,1]（C=±1）时 h∈[−4,4]、d∈[0,4]。h<0 合法：压缩可能纠正完整宿主的错误。把 h 截为正数会改变问题。

mask 由固定策略 K=π(B,U) 产生，U 在给定 B 后与 C 独立。令 μ(B)=E[C|B]，则：

```text
E[h|B,K] = (b−a_K)² + 2(μ(B)−b)(b−a_K).
```

该式不要求 b 是 Bayes，但要求同一概率法则、固定宿主，以及上述条件独立。若用 balanced label weights 训练 μ，解释也在该 reweighted law 下；不能自动对自然部署分布的 host 风险成立。

## 5. 条件设计建议：宿主校正的风险 critic

**尚未执行、不是合作者原计划已有方法**。

可以分别学习：
- `uψ(B)`：真实标签条件均值 μ 的任务 critic（VCS/Logistic 匹配对照）；
- `b̃ω(B), ãω(A_K)`：固定宿主 b/a 的代理；
- 组合风险 `d̃ + 2(uψ−b̃)(b̃−ã)`，并与直接回归 h 比较。

不要将 uψ 自己的 B/A 输出差直接等同宿主 b/a 差。实际部署只计算选择器允许看到的输入；宿主完整输出只用于训练/评价，不能为省算力先调用完整宿主。

### 一个基本误差检查（独立推导，不作为新颖性）

令 δ=b−a、δ̃=b̃−ã。真实条件风险 H=δ²+2(μ−b)δ，预测为 H̃=δ̃²+2(μ̃−b̃)δ̃。在所有均值/预测均界于 [−1,1] 时：

```text
|H−H̃| ≤ 4|μ−μ̃| + 12|b−b̃| + 8|a−ã|.
```

这是用 |δ|,|δ̃|≤2 的保守三角不等式，不保证可从无标签数据知道右侧。用途只是提醒：三个头的误差都会进入最终风险，增加头不等于增加可靠性。包内随机数值检查仅检查公式一致性，不证明自然数据上优势。

## 6. Policy-induced observation 的边界

原 mask 分布下学到的 E[C|A]，不一定适用于 gate 依赖完整 B 选择后的 A。K 本身是观测的一部分，也可能编码已删除区域的信息。记录位置索引是必要的，但仍需区分：
- 对固定外生 mask 机制的后验拟合；
- 对新策略诱导观测的后验；
- 宿主实际有限网络在该输入上的表现。

因此 off-policy/on-policy 诊断是后续假说。只要 oracle gap 在一个分布上小，就不能保证任意选择器都无法利用局部拟合误差。

## 7. 梯度控制的一项 caution

源码 `relaxed_topk` 将 threshold detach。对该 surrogate 求导时，自动微分按固定 threshold 计算。重算 threshold 的端到端有限差分与它不同，可能是**设计的 surrogate 梯度**，而非 autograd 错误。梯度校验必须声明固定哪些量。

同时，eval 的所有模块可仍然对输入/soft-mask 求导；冻结参数只是排除它们的参数更新，不排除 mask 的计算图。[EXT-01]

## 8. “更好的 critic”应怎样判断

不只看 held-out J 或整体预测准确率。压缩决策至少要回答：同问题同预算哪个 mask 较好；策略生成 mask 是否低估风险；是否保留实际视觉证据；其成本是否小于所节省宿主计算。

并列的 source target、surrogate loss 和 deployment metric 要分开。S3 以 h 为训练目标，使用 h 排名作选择与任务需求相符，不能仅因此叫“不公平”；若要比较后验估计，应另外报告公共概率风险，而不是换掉部署终点保护某方法。


---

<!-- Source file: docs/07_REPORTING_AND_TEST_USE_CN.md -->

# 结果报告、测试使用与审查回复

## 1. 原结果与新诊断必须并存

原 10k/full-data/official-test 报告代表对应 frozen 方法；发现 critic 实施风险不自动撤销其评价。新 mode、corrected Spearman、新 gate solver 和新风险头都必须有独立标识。

r5 的 positive normalization effect 可以保留；VQ≈JQ 应改为“未检出差异”，不声称等效。REG 的优势是学习 critic 类和搜索器的共同结果；GAIN 的风险—成本操作点不是无损加速。D1 的局部 TextVQA 相对优势须同时显示 grid。[SRC-05–SRC-09, SRC-14, SRC-15]

旧 10-10 综合文件中已被附录修正的“JS 必然更准”“固定单调评分使训练不能改变排名”等句子不作为本包依据；最新报告优先于过时的“test 未读”。[H02]

## 2. 统计单位

图片/图表是重采样 cluster；同一图片多个问题、每题多个 mask、每 mask 多次 dropout 不扩充独立 N。每个训练 seed 的配对差先保存，再报告 seed variation 与 image sampling 两种来源。

可用两层/交叉 bootstrap 时明确重采样操作；不能把三 seeds×数千 masks 当成数千个独立训练重复。源码诊断是描述性，除非新协议明确声明，不另做几十个 post-hoc 显著性裁决。

同一问题的 mask oracle 使用真实标签，只是有限候选的解释参照。它的 regret 不是学习策略能达到的理论最大收益。常数 rank、无差别 h 的组单列，不用只丢掉难以定义的组使平均值提高。

## 3. 成本与资源

保持当前 H100/L40S/RTX 等执行约束；服务器实测再填写预算。记录总训练成本、宿主补前向成本、selector runtime 和完整宿主 latency。显存不足/重试/混 GPU 用完整记录说明，不根据成功分片删掉失败计算量。

时延是固定 profiling workload 的读数，不自动等于每个测试集的实际平均。Qwen 无既定 B* 时继续不作加速主声明。排查 dropout 或 rank 修复无需重测所有历史时延；新算法改变路径时才以相同 workload 重测。

## 4. 测试暴露与新实验

官方 GQA testdev、ChartQA test、RefCOCO test 已经产生结果。受这些结果启发的修正，即使只在 CAL 调参，也不能把同一个 test 再称为从未观察的独立确认。

允许的处理：
- 原 frozen test 如实保留；
- 新方案在 FIT/CAL 开发并在已有 DEV/val 报 repeated development evidence；
- 为最终新主张由负责人另定未使用数据/确认方案；若只剩重复 test，明确披露并限制结论。

本包不解除任何 test-closed 规则，也不取消原已批准 phase F 的冻结执行。该授权只覆盖已声明方法，不自动覆盖本包未来新方法。

## 5. 第一轮回复模板的要求

每条问题填写：
1. 审查主张（原文简述）；
2. 接受/修改/反驳/已解决/无法核对；
3. 源码位置和实际运行路径；
4. 实测数据与统计单位；
5. 影响范围（哪些 seeds、hosts、scale、报告）；
6. 修复/敏感性建议及是否需新批准；
7. 哪些结论尚不能得出。

若关键判断不成立，明确写反例或新代码，不能为符合审查而制造“确认”。若重算不改变选择，也应记录，这是有价值的影响边界。

## 6. 收束会议需要回答的三句话

- 已批准的全量/外部验证现在完成到哪里，还有哪些完整报告没出？
- 本轮源码问题有无实际影响，是否足以解释部分压缩失败？
- 下一次新训练只改变哪一个因素，它将支持或否定哪一个有限主张？

不要再从一个没有增益的细节自动派生下一批大网格。应用、估计器和理论贡献分别整理；“修复一段常见实现”不等于绕过 RPC 的新颖性边界。


---

<!-- Source file: sources/SOURCE_INDEX_CN.md -->

# 来源、范围与复核深度

包截点 `54e1715e29dd2caaa6be7ed8e670829310951d39`。凡在父提交 `3a5e2f0b19d5d32a3913c3916413bae05753eab2` 读取的文件，最新单提交 diff 未修改它们，因此链接统一定位本包 pin；这不表示本轮重新逐行读取了所有旧报告。

本包不使用 10-10 历史批评的旧数字替代最新结果；不把历史文献新颖性判断重新审计一遍。公开文档只核对软件语义，推导和改进建议在正文单列。

## SRC-00 — main HEAD
- 范围：branch metadata
- 说明：本轮直接读取，2026-10-13 18:36:55 UTC
- 入口：[查看原始来源](https://api.github.com/repos/W-Yinghao/VCS_QMI/branches/main)

## SRC-01 — Latest commit / integrity gate
- 范围：commit diff
- 说明：本轮读取完整相关 diff
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/commit/54e1715e29dd2caaa6be7ed8e670829310951d39)

## SRC-02 — LLaVA selector implementation
- 范围：L63–206; L289–441; functions Block/Selector/train/soft_hard_gap/spearman
- 说明：本轮重新读取；blob 706a464b494f8458b49026322030347a1d9aeaaf
- 文件：`scripts/r7_d1_selectors.py`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/scripts/r7_d1_selectors.py)

## SRC-03 — Qwen selector implementation
- 范围：本轮 L245–265 与 L284–356；其余 encode/estimator_terms/gate_terms/soft_hard_gap40 在上一轮读取
- 说明：本轮重新核对 train40 定义和 B 模式路径；其余上一轮读取；blob 9a68d1c0926cdb46c06e84a38b8ab57251c1011c
- 文件：`scripts/r7_d1_40_selectors.py`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/scripts/r7_d1_40_selectors.py)

## SRC-04 — D1-FULL training wrapper
- 范围：L1–180
- 说明：本轮重新读取，确认复用 frozen D1 trainer
- 文件：`scripts/r7_d1full_train.py`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/scripts/r7_d1full_train.py)

## SRC-05 — r5 official test report
- 范围：§1–6, official score table
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R5/R5_FINAL_EVAL_REPORT_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R5/R5_FINAL_EVAL_REPORT_20261012.md)

## SRC-06 — LLaVA DEV/val result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_D1_30_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_D1_30_REPORT_20261013.md)

## SRC-07 — LLaVA official result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_50_PHASE_L_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_50_PHASE_L_REPORT_20261013.md)

## SRC-08 — Qwen DEV/val result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_40C_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_40C_REPORT_20261013.md)

## SRC-09 — Qwen official result
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_50_PHASE_Q_REPORT_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_50_PHASE_Q_REPORT_20261013.md)

## SRC-10 — LLaVA CAL readout
- 范围：full report
- 说明：上一轮读取；相关性数字为原代码读数，未在本包纠正
- 文件：`reports/R7/R7_D1_20_CAL_READOUT.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_D1_20_CAL_READOUT.md)

## SRC-11 — Qwen CAL readout
- 范围：full report
- 说明：上一轮读取；相关性数字为原代码读数
- 文件：`reports/R7/R7_40B_CAL_READOUT.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_40B_CAL_READOUT.md)

## SRC-12 — D1 route / task definitions
- 范围：§1–6 excerpt
- 说明：草稿只用于原始问题说明；执行以已冻结协议为准
- 文件：`reports/R7/R7_ROUTE_DECISION.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_ROUTE_DECISION.md)

## SRC-13 — LLaVA full-data freeze
- 范围：full file incl df_4
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_D1_FULL_STAGED_FREEZE_20261012.yaml`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_D1_FULL_STAGED_FREEZE_20261012.yaml)

## SRC-14 — REG second modality
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R6/R6_REG_P3_REPORT_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R6/R6_REG_P3_REPORT_20261012.md)

## SRC-15 — REG/GAIN synthesis
- 范围：full report
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R6/R6_SYNTHESIS_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R6/R6_SYNTHESIS_20261012.md)

## SRC-16 — Qwen full-data freeze
- 范围：full file
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_E2_STAGED_FREEZE_20261013.yaml`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_E2_STAGED_FREEZE_20261013.yaml)

## SRC-17 — E1/E2 decision
- 范围：full file
- 说明：上一轮读取；最新提交未改
- 文件：`reports/R7/R7_EXTENSIONS_DECISION_20261013.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/R7/R7_EXTENSIONS_DECISION_20261013.md)

## SRC-18 — Owner decisions
- 范围：decision history and updated actions
- 说明：上一轮读取相关内容；最新提交未改；接入时需再查实时授权
- 文件：`reports/OWNER_DECISIONS_20261012.md`
- 入口：[查看原始来源](https://github.com/W-Yinghao/VCS_QMI/blob/54e1715e29dd2caaa6be7ed8e670829310951d39/reports/OWNER_DECISIONS_20261012.md)

## EXT-01 — PyTorch Autograd mechanics: eval vs grad modes
- 范围：Evaluation Mode; requires_grad; no-grad
- 说明：本轮公开官方文档核对；不要求服务器升级依赖
- 入口：[查看原始来源](https://docs.pytorch.org/docs/stable/notes/autograd)

## EXT-02 — SciPy spearmanr
- 范围：ConstantInputWarning: constant coefficient undefined
- 说明：本轮官方文档核对
- 入口：[查看原始来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)

## EXT-03 — SciPy rankdata
- 范围：average/ordinal tie rules
- 说明：本轮官方文档核对
- 入口：[查看原始来源](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)

## H01 — Collaborator original September 2026 plan
- 范围：§4–7: S/J/gap and classification; §8: prior-art; §10–12: claims and experiments
- 说明：用户提供资料，完整内容已在对话中读取；本包不复制整个 PDF

## H02 — 10-10 consolidated audit with Appendix 4
- 范围：附录4优先于相冲突的正文；只作历史定位
- 说明：用户提供资料；不把其旧状态当成 10-13 进度

## SRC-19 — 交付前增量

最后核对到 3e2fe59；4 个新增提交，两个关键 critic blob 未变。正文 pin 不变，执行状态由 `docs/08_DELIVERY_UPDATE_CN.md` 补充。

- [固定比较](https://github.com/W-Yinghao/VCS_QMI/compare/54e1715e29dd2caaa6be7ed8e670829310951d39...3e2fe59c4861abae46537574c4c59f9bedfe91c6)


---

<!-- Source file: reference/README_CN.md -->

# 审查辅助代码

这些脚本不是训练实现，不替代服务器原有 host、dataset、metrics 和统计模块。

| 文件 | 能做什么 | 不能做什么 |
|---|---|---|
| audit_core.py | 平均 ties 的 rank、四格分解、风险恒等式、有限 mask 决策、模式 context | 自然数据性能或因果解释 |
| repo_source_audit.py | 本地源码 AST、blob、HEAD、函数范围，只读检查 | 证明实际 GPU 作业走了相同路径 |
| repo_runtime_probe.py | 显式允许后，用微型合成输入导入生产 LLaVA selector，检查 mode/梯度/constant rank | 真实 checkpoint、Qwen 可变 N 运行、完整 VLM |
| audit_rows.py | 对预先导出的 FIT/CAL 逐 mask CSV 计算描述性 rank/有限 regret | 模型拟合、独立性检验、读取 official test |

依赖：Python 3.10+、NumPy、PyTorch；本地测试用 SciPy 交叉检查。不要求服务器改变既有训练环境。脚本在本包目录运行，实际 repo 路径由参数提供。输出只写用户指定文件，不提交 Git、不修改源码、不调用 sbatch。

`--allow-newer` 只表示明确允许检查另一版本，输出记录实际 blob；不表示旧审查结论已针对新代码验证。

`legacy_spearman` 故意复现被审查实现的两次 argsort，仅用于测量差异，不应导入生产训练。

本包作者只运行了参考单元测试；没有本地挂载该私有仓库源码，所以未执行 `repo_runtime_probe.py` 的真实生产模块导入路径。交付前做了语法和 CLI 拒绝路径检查。服务器应先审代码再执行。
