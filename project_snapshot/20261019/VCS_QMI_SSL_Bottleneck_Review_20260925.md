# VCS-QMI SSL：执行审查、瓶颈定位与下一轮优化建议

版本：2026-09-25。审查快照：`W-Yinghao/VCS_QMI@398c17f6b1057c939aa93bc5ead77b7c762f65d9`（最后一次可见提交 2026-09-25 13:33:17 UTC）。

## 0. 阅读范围、结论与边界

本报告读取了已提交的阶段报告、结果表、配置计划，以及 `reference/ssl_core.py`、`objectives.py`、`models/critic.py`、`models/__init__.py`、`train.py`、`evaluate.py`、`diagnostics.py`。未连接服务器实时 Slurm，未访问未提交的 checkpoint / 逐步日志 / 特征缓存，未重新运行图像训练。

**主要判断：目前最强的证据指向 critic 的结构偏置与表示几何之间的配合问题，而不是单纯的 critic 容量不足、负样本数量不足或训练未启动。** 拼接 MLP 产生了可用于分类、但谱能量高度集中的表示；cosine 和显式交互 critic 改善了表示谱与 kNN，然而还没有关闭与对照的准确率差距。

原始等权混合参考测度、二次目标 J、外层 tanh 保持不变。这里建议优化的是 f 的参数化、其初始化/尺度、投影头、常规训练配方和实现效率。没有提出平滑目标、两点梯度或额外损失。

本报告不是新的自动执行授权。已有 P24 任务的服务器状态应先核查；不要重复启动已经在跑的实验。

## 1. 当前数据真正说明了什么

所有准确率都是官方训练部分内的固定 45k fit / 5k selection 上的 frozen-h 线性验证成绩，不是 CIFAR-10 官方 test。大部分后期搜索是 seed 0；不能当成多种子确认结果。

| 设置 | epoch / seeds | linear-val (%) | kNN (%) | h effective rank | heldout-J | 来源 |
|---|---|---:|---:|---:|---:|---|
| 原 MLP，K=1 | 200 / 3 | 74.34 ± 0.46 | 63.93 ± 0.62 | 13.18 ± 0.30 | 0.899 ± 0.002 | P7 |
| SimCLR matched | 200 / 3 | 86.09 ± 0.38 | 83.98 ± 0.24 | 89.96 ± 0.83 | 不适用 | P7 |
| VICReg matched-128 | 200 / 3 | 85.46 ± 0.26 | 81.92 ± 0.40 | 75.66 ± 0.37 | 不适用 | P7 |
| MLP，K=8 | 200 / 3 | 76.74 ± 0.78 | 67.09 ± 0.31 | 15.57 ± 0.18 | 0.927 ± 0.001 | P11 |
| MLP，K=64 | 200 / 3 | 76.80 ± 0.31 | 67.53 ± 0.61 | 15.93 ± 0.18 | 0.932 | P11 |
| MLP，K=8，critic lr×10 | 200 / 1 | 77.44 | 68.40 | 21.4 | 0.940 | P17 |
| 相同组合 | 800 / 1 | 79.56 | 74.28 | 34.8 | 0.964 | P17，最新阶段报告 |
| cosine，K=8 | 200 / 1 | 78.32 | 72.76 | 57.7 | 0.968 | P19 |
| 交互 MLP，K=8 | 200 / 1 | 77.34 | 70.74 | 39.3 | 0.962 | P19 |

最干净的 critic 对照是 P19 的同 seed、同 K=8、同 lr 基线：76.12 / 67.44 / h-rank 15.6 → cosine 78.32 / 72.76 / 57.7。不能把 78.32 与三种子均值的差异当成配对检验。

### 1.1 更新滞后的汇总表

`ALL_RUNS.md` 仍将 `P16_vcs_k8_clr10_800ep_seed0` 标为 RUNNING，但较晚的 P17 阶段结果已给出 COMPLETED 与 79.56。引用结果时按相应阶段的时间、配置和结果文件核对，不直接信任任一旧汇总表。

P24 在本快照中只有预注册、配置和较早 RUNNING 状态，没有完整的已提交最终结果。因此不能评价 cosine 800ep、cosine K64/K255、EMA 或 scale10 的最终效果。

### 1.2 已有结论的适用范围

- K=1→8 的收益在三种子上出现；K=8→64 在该 MLP 配方下没有明显进一步收益。
- 2048 宽 MLP、更多层、更宽 projector 等没有解决 MLP 配方的准确率差距。这不证明所有 critic 或 projector 结构都无效。
- 弱增强使 J 更高、精度明显更低；更强增强降低 J 也未自动提高精度。J 大小不能代替表示质量。
- P19 使 critic 结构成为优先方向，但单种子不能证明“唯一瓶颈已定位”。
- 有效秩是谱能量分散程度，不是信息量、语义量或精确线性秩。不得写“只剩七维信息”。
- 原结果有低维集中，但没有恒定表示坍塌：74% 的线性验证仍显著高于同协议随机初始化表示。

## 2. 先修正一处诊断实现：target / predictor 分支的 heldout-J

### 2.1 静态代码路径

`objectives.forward_features_target` 与 `compute_objective_target` 在训练时使用：

- stop-grad：online 与 detached target；
- EMA：online 与 EMA encoder/projector；
- predictor：online 侧先经过 predictor，再与 target 配对。

但 `evaluate.evaluate_run` 与 `Trainer.epoch_evaluation` 只恢复/使用 encoder、projector、critic 来调用 `diagnostics.critic_holdout`。该函数没有 teacher、predictor 或 target_branch 参数，总是计算原始 online-projector 的两种视图，再传入 critic。

**结论：对于 EMA / predictor 变体，现有 heldout-J 不对应训练时的同一变量对。** 对共享无 predictor 基线，这条问题不适用；纯 stopgrad 在 eval 模式下数值路径可与共享路径一致，需单独测试，不能一概作废。

`P22_vcs_sg_pred_seed0` 的 linear 76.94 与 heldout-J −0.976 并不构成已证实的估计器失效；后者很可能受错路径影响。原始负值仍应保留，不允许裁掉。

### 2.2 修复要求

1. 从 checkpoint 恢复 predictor 和 teacher 的完整参数与 buffer。
2. 复用一个统一的“特征构造 + 配对评分”函数；训练/评估显式传入 target 分支、predictor、feature_source 和方向策略。
3. 固定输入、shift、模块状态与模式时，训练目标的无梯度评估与 heldout helper 的输出应相同。
4. eval 必须冻结所有实际参与评分的模块；单独说明 teacher BN 的训练/评估模式。当前训练 teacher 使用 train-mode BN、每步复制 student buffer，这是一种可记录的选择，不能静默换掉。
5. 保存 `heldout_J_train_wiring`；原来的 online-online 分数可以保留为单独的 transfer diagnostic，但必须改名。
6. predictor 变体的谱记录应区分 projector 输出、predictor 输出、teacher 输出；不要把 pre-predictor 的 z-rank称作 critic 输入秩。
7. critic-on-h 变体的 projector 不接收目标梯度。其 projector 输出谱不能作为“训练后的 projector 质量”来解释。
8. 优先用已保存 checkpoint 重新评估，不为修复一个评估路径立即重跑所有训练。

这处修复不直接改变 student h 的冻结线性结果；因此不能用它解释所有共享分支 VCS 与 SimCLR 的差距。

## 3. Bottleneck A：自由配对判别器不一定产生好的共享度量

### 3.1 直接证据

当前拼接 MLP：`T=tanh(MLP([z1,z2]))`。它可以通过复杂的非线性判别边界区分正负配对；损失并不显式要求配对信息必须以共享 embedding 的简单几何关系存在。

P19 的 cosine 只含两个参数，却使 h-rank 从 15.6 到 57.7、kNN 从 67.44 到 72.76；已有显式交互 MLP 也改善。最合理的工作假设是：**几何约束比增加判别器自由度更有帮助。** 这与依赖学习中 critic / encoder 归纳偏置重要的文献一致，但该文献不是本项目机制的证明 [R1]。

### 3.2 已试过与尚未试过必须区别

- `InteractCritic` 实际输入是 `[z1,z2,z1*z2,abs(z1-z2)]`。第四项是逐元素绝对值，不是报告中可能被理解成的单个 L2 距离。
- 已有 bilinear 是 `z1^T W z2 + MLP([z1,z2])`，不是纯 bilinear / shared metric。它失败不能否定纯 metric 结构。
- P19 cosine 仍是单种子发现；先补种子 1、2，再将它当作稳定的新基线。

### 3.3 新候选 A：interaction-only critic

`T=tanh(MLP([z1*z2, abs(z1-z2)]))`，建议先 `256→256→1`，不输入原始 z1/z2；或者使用与已有交互 MLP匹配的宽度以隔离特征删除的作用。先锁定一种，避免把宽度变化混入主要比较。

目标：让 critic 的输入直接是两种表示的关系，不让单侧坐标成为主要通道。函数关于交换视图天然对称。它仍有足够的非线性，但比自由拼接更受限制。

### 3.4 新候选 B：纯共享度量 critic

`w_v=normalize(W z_v)`；`T=tanh(a * dot(w1,w2)+b)`。

W 使用同一个 128×128 线性矩阵，初始化为单位阵，无额外 concat-MLP。这样起点等于 cosine，再测试适量的共享度量自由度是否有帮助。必须记录 W 的谱/条件数，防止又只使用少数方向。不要把带输出归一化的这种结构与任意不对称 bilinear 混称为同一算法。

这些变化保持 J 和 tanh，但改变函数类。有限 critic 的 supremum 只是 S 的下界，不宣称其仍能表示任意分布的 Bayes critic。

## 4. Bottleneck B：cosine 的尺度与偏置可能让分数先饱和

### 4.1 需要实测，而不是只看 |T|>0.95

当前 `CosineCritic` 为 `T=tanh(a*s+b)`，a、b 都可学习，a 初始为1、b为0，a未约束正号。P24 在测试 a_init=10 和 critic lr×10，但这并不等于控制训练后期的 a。

定义 `a_c(t)=c*t-t²/2`，c=±1。则

`d a_c(tanh u) / du = (c-tanh u)*(1-tanh² u)`。

对正确正样本：t=0 时系数1；t=.95时.004875；t=.99时.000199。

错误负样本也可能饱和：t=.999时梯度系数约−.003996。整体网络梯度还包含 a、归一化导数与网络 Jacobian，不能凭这些标量直接断言完整梯度消失。

P19 中高 J 与较高秩并存，说明“高饱和率必然导致低秩”太强；但它没有排除高饱和率限制后续语义分离。cosine 在100→200epoch的kNN仅71.72→72.76，仍需分析剩余有效梯度。

### 4.2 必须增加的日志

对正、负分布分别保存：

- cosine similarity s 的分位数；a(t)、b(t)、阈值 −b/a（a接近0时记null）；
- logits u 的分位数；正确端饱和与错误端饱和四项：P:t>.95 / t<−.95；Q:t<−.95 / t>.95；
- |(c−T)(1−T²)| 的均值、分位数、零附近比例；
- 每样本实际 `dL/dz`、`dL/dp_raw` 的范数，诊断时恢复类别均值因子，避免K改变后单对1/K缩放混淆；
- 贡献集中度 `N_part=(sum w)^2/(sum w²+eps)`，分别按类计算，再除以该类配对数。它是“梯度权重参与率”，不是独立样本有效样本量；
- 正负 embedding 的 alignment、uniformity、均值向量范数，以及 h/z 的谱。alignment/uniformity只用于诊断，不加入训练损失 [R2]。

更改这些日志不改目标，不需要声称新的梯度方法。

### 4.3 新候选 C：固定尺度或有上界的尺度

先测试 `a=5`、`a=10`，b仍学习；全部训练保持原始J。为隔离“是否允许a继续增长”，固定与自由两组必须使用相同a0和相同b0，例如 `a0=5且继续学习` 对比 `a=5固定`；不能只拿free-a初始1与fixed-a初始10作机制结论。尺度10作为第二档，在第一档有信息后再执行。

如果固定a有效，可进一步考虑一个显式区间内的可学习a；这属于函数类限制，不能声称与任意自由critic有完全相同的拟合能力。

不要只把a初始化得更小/更大，然后允许它无约束增长，再宣布“尺度已控制”。也不要因为a变小造成绝对梯度变小就声称性能改善。

### 4.4 新候选 D：训练数据上的偏置初始化

P19 cosine 初始 heldout-J 为−0.5686。对于正负cosine都接近1且a=1,b=0的简化情形，有J≈−tanh²(1)=−0.5800。两者相近提示初始表示可能在一个窄角锥内，但真实相似度分布仍需读取，不能直接视为测量结论。

在仅来自fit的固定校准mini-batch上估计 `mu=0.5*(mean(s_pos)+mean(s_neg))`，设 `b0=−a0*mu`，只做一次、只改变初始化，随后正常优化J。

必须记录校准UID和RNG，明确BN模式并保存/恢复未授权变更的BN buffers；不读取selection/test；不把逐batch重新减均值混成同一个试验。

先配对比较 `(a0=1,b0=0)` 与 `(a0=1,b0=calibrated)`；再在a0=10时比较。特别是a0=10,b0=0可能把初始错误负配对直接推到tanh饱和端，不能据此直接否定“较大尺度”本身。

## 5. Bottleneck C：投影头的训练几何，而不是仅仅它的宽度

当前 projector：Linear512→512、BN、ReLU、Linear512→128、有bias、无末层BN，然后L2 normalize。

P13 的宽度/维度搜索均在 concat-MLP 基础上，未证明“cosine条件下投影头已最优”。P24已包含depth3及on_h，勿重复。

### 新候选 E：最后一层BN后再L2

保留前层，最后线性层可设置bias=False，接 `BatchNorm1d(128, affine=False)`，再L2。把bias去除作为明确记录的同一结构变体，不偷称只有一个数值字段变化。

目的：减少某些坐标因尺度占优导致的方向集中；它不保证 decorrelation，也不保证不坍塌。官方SimCLR实现的最后投影线性层使用BN且无ReLU [R4]，可作为结构参照，不声称原版正是affine=False。

BN让训练表示依赖同batch样本；理论中固定逐样本映射的统计界不因此直接适用于训练batch。下游评估冻结running stats，来源只允许fit。

可选的第二级实验：samplewise LayerNorm 或分组L2归一化，避免额外方差损失。每组可以重复相同信息，因此不能将它们称为独立性保证。

### 先做廉价定位

同一checkpoint分别在 h、projector隐藏层、p_raw 上做公平的冻结线性读出，仍以h为正式主结果。若差异巨大，可定位信息在何处变得不易线性提取；不能择最好层替换主指标而不声明。

现有fit标准化probe多数只改变不到1点，已经削弱“十点差距只是probe尺度问题”的解释。个别高lr配置提升2.4点，仍远未补齐差距。

## 6. 更多优化方向：明确条件后再做

### 6.1 四个同分布视图，增加正配对覆盖

对同一图像独立抽4次相同增强分布，按预定的视图配对集合计算同一个J并取平均，Q仍来自不同UID。各配对来自同一个P/Q时，这只是增加相同目标的视图Monte Carlo覆盖；不是新损失。

比较时记录encoder前向、图像呈现数、optimizer steps与wall-clock。4视图优于2视图但计算翻倍不能自动称高效。先不要同时加入异质multi-crop、不同teacher和额外正则。

### 6.2 常规优化器与尺度控制

当前多数搜索都在AdamW下。若结构/尺度确认后仍有明显早期迟滞，可在同一cosine基线测试SGD+momentum用于encoder/projector，critic仍采用明确配置的优化器，或对归一化前参数使用不同但预先固定的decay规则。

不要先做大范围优化器×lr×batch笛卡尔积。相同epoch但batch改变会改变optimizer步数，P17不能证明B=256对所有预算最优。

### 6.3 增强再检查应基于cosine，而不是重复旧MLP全网格

P17中弱增强显著变差，强增强也未单独改善，不支持“直接加大所有增强”。新基线可优先做两项可解释变化：原配方+有限的grayscale/颜色变化控制，或约束极端不重叠裁剪。裁剪重叠控制会改变view分布，是应用设计而不是原数据分布不变的估计技巧；与对照共同使用并记录。

先做只读nuisance诊断：从表示预测平均颜色/亮度、增强参数，或对颜色扰动检查critic分数与检索是否异常变化。发现实际依赖后再改增强。

### 6.4 匹配分数结构的对照，定位剩余差距

在同一cosine分数、相同batch/K/增强下，添加balanced logistic pair classifier和固定K的InfoNCE作为**诊断基线**，不是把它们换成VCS主方法。

如果同几何、同配对预算下logistic明显优于VCS，说明tanh+平方目标的训练权重值得更集中分析；若两者相近且均落后原SimCLR，问题更可能还包括pair coverage / projector /训练配方。

不得把替换损失后的成绩填入VCS行。也不把两点梯度、辅助variance/covariance损失或hard-negative mining作为这一阶段方案。

## 7. 吞吐量：把可比计算做好，而不是靠更多epoch掩盖问题

### 7.1 cosine的全off-diagonal矩阵实现

对单位范数Z1,Z2，直接算 `C=Z1 @ Z2.T`，`T=tanh(a*C+b)`。

正项使用对角线均值；负项使用所有非对角项均值：

`J=mean(diag(T)-diag(T)^2/2)+mean_offdiag(-T-T^2/2)`。

它与K=B−1的全部非零shift在函数值与梯度上等价；不使用softmax，不对大分数重加权，不排除同类别负对。若只用K个shift，可从C按原来的索引gather。

本地CPU float64小张量检查（B=7,D=5）中，完整shift与矩阵形式J差为0，四组梯度最大差约1.94e−16。该检查不是服务器集成测试，也不构成速度测试。

理论复杂度仍随B²增长；B=256的score矩阵有65536个元素，但“输出矩阵小”不等于训练所有激活都免费。对实际GPU单独测量。

### 7.2 逐步CPU同步

`objectives.compute_objective`每step将多个stats转换成float，`train_step`还读取scalar score和做grad检查。它们可能引入CPU/GPU同步。建议先profile：训练forward/backward、统计同步、数据加载与评估各自耗时，再将非必要日志转换延迟到记录步。

必须保留明确的非有限值检测，不以提速为理由静默跳过异常batch。不能在不同GPU上比较几秒差异后宣称方法更快。

## 8. 建议的下一轮，按顺序而不是一次性全跑

### A. 无新训练的审计

- 同步ALL_RUNS与最新阶段结果；固定快照与结果来源。
- 修复EMA/predictor的评估路径；从既有checkpoint补正确heldout-J。
- 提取MLP-K8、cosine-K8、interaction-K8、SimCLR的共同诊断：alignment/uniformity、score/gradient参与率、h/p谱、各层probe；cosine初始及后期a/b。
- 核对P24实际完成状态，只等已有任务汇总，不重复提交。

### B. 小规模确认

cosine-K8补seed1、2，保持200epoch配方。interaction如预算允许也补seed1、2。汇报与匹配MLP、SimCLR的逐seed差值；不要只报告选中seed0的增量。

### C. 新筛选，不重复P24

候选清单：

| ID | 唯一主要问题 | 改动 |
|---|---|---|
| N1 | 初始共同偏移 | a0=1不变，仅fit校准b0 |
| N2 | 尺度对照 | a0=5继续学习，b学习，fit校准b0 |
| N3 | 晚期尺度自由度 | a=5固定，b学习；a0/b0与N2完全相同 |
| N4 | 输出坐标尺度 | projector末层BN+L2 |
| N5 | 自由单侧信息通道 | interaction-only，不含原始z1/z2 |
| N6 | 受约束的可学度量 | 纯共享W+cosine，无concat MLP |

先根据A阶段诊断选其中3–4项，不自动提交全部；N2/N3成对选择，否则不能单独归因于是否冻结尺度。以同一cosine基线开展，不能每发现一个好点就无记录地改变其余实验的基线。

可用固定较短预算做明确失败筛查，但不能用20epoch的排序断言200epoch/800epoch的最终优劣；存活配置按一致200epoch重新训练。最终保留不超过2个候选，用新种子复核后再组合有理论动机的两个因素。

### D. 停止扩散条件

- 高J、低rank、低linear：回到几何诊断，不继续堆critic容量。
- 降低饱和但linear不升：不把“梯度更活跃”当成功。
- rank升而linear不升：检查各层可读性与共享nuisance，不继续仅按rank选配置。
- cosine多seed收益不稳定：先缩小效果主张，不立刻转新任务来回避。
- 现有配置搜索已反复使用selection：最终锁定配方，再以统一规则在官方test与第二视觉数据集确认。新seed不会消除对同一个selection集的适配。

## 9. 建议server下一次必须回报什么

1. 本轮新增/修复代码差异、来源commit、测试与重评估覆盖范围。
2. 已有P24完整状态；未提交的结果补齐，旧汇总保留快照时间。
3. 一张与原对照一致协议的精度/成本表，标注seed数量与selection而非test。
4. 一张cosine a/b、正负similarity、四类饱和率和梯度参与率的时间表。
5. 对每个新候选，只回答对应的单一假设是否得到支持；不写“理论已证明cosine最优”等过度结论。
6. 当只变f参数化时，说明是同J不同函数类；若额外改变target/routing，另列算法变体，不能合并成一个无差别VCS结果。

## 10. 来源索引

所有本仓库文件均以开头的commit为准：

- [S0] `reports/PLAN_CONSTRAINTS_20260925.md`
- [S1] `reports/P7_confirm200_report.md`
- [S2] `reports/P11_kstudy_report.md`
- [S3] `reports/P13_hparamA_report.md`
- [S4] `reports/P17_hparamB_set_report.md` 及该阶段results_table
- [S5] `reports/P19_critic_variants_report.md` / `P19_critic_variants_results_table.json`
- [S6] `reports/P24_COSINE_BASE_PREREG_FROZEN_20260925.md`
- [S7] `reports/ALL_RUNS.md`（有时间滞后）
- [S8] `reports/PROBE_STANDARDIZED.md`
- [C1] `reference/ssl_core.py`
- [C2] `src/vcs_ssl/models/critic.py`
- [C3] `src/vcs_ssl/objectives.py`
- [C4] `src/vcs_ssl/train.py`
- [C5] `src/vcs_ssl/evaluate.py`
- [C6] `src/vcs_ssl/diagnostics.py`
- [C7] `src/vcs_ssl/models/__init__.py`

外部文献用于提供解释与诊断工具，不替代项目实证：

- [R1] Tschannen et al. *On Mutual Information Maximization for Representation Learning*, ICLR 2020. `https://research.google/pubs/on-mutual-information-maximization-for-representation-learning/`
- [R2] Wang and Isola. *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020. `https://proceedings.mlr.press/v119/wang20k.html`
- [R3] Jing et al. *Understanding Dimensional Collapse in Contrastive Self-Supervised Learning*, ICLR 2022. `https://openreview.net/pdf?id=YevsQ05DEN7`
- [R4] SimCLR official implementation, `google-research/simclr/model_util.py`, `projection_head`, fetched blob `e9caf634dd3f4698258266ce9d1533f0b3b23ab2`. `https://github.com/google-research/simclr/blob/master/model_util.py`

**最终建议：保留原理论和J；把下一轮重点从“继续扩大搜索”转成“验证cosine结构收益，并控制它的初始化、尺度和投影几何”。先修正观测工具，再根据真实梯度与几何证据选择少量实验。**
