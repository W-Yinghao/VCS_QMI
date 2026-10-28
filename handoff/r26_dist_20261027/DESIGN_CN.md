Publication copy (sample names aliased; numbers and conclusions unchanged).

# R26 实验设计：真实候选对应分布 + VCS 二阶学习（2026-10-27）

依据包：`r26_retrieval_arch/VCS_Retrieval_R26_Architecture_20261027`（zip sha256 e31faf7e…，MANIFEST 核对通过）。
用户指示：「这个包需要设计实验并开始，不要设置阈值让实验直接不启动」。因此本设计**不设任何结果门槛来决定是否启动后续单元**；唯一的启动条件是技术门禁（实现正确性、数值核对、真实数据 smoke）。各阶段只是排队顺序，结果好坏都完整跑完。成功/失败的解释标准沿用包内 §7、§10 与 R25 协议，只用于事后解释。

## 1. 共同设置（所有 R26 臂相同）

- 骨干：R25 的 IN2R-code（commit 909e5329）及其 RUN_LOCK、噪声映射、full DEV、共用工程修复、batch centering、评测批顺序；从 R25 锁定代码派生一个 R26 patch，不改 R25 的运行代码。
- 保留：双编码器、GPO、原生 GMM 划分、clean ranking、intra 约束、双 peer、高置信 FIFO、训练长度与学习率、最终余弦检索与 peer 聚合。
- 修改（所有 R26 臂一致）：noisy 修正分支去掉可训练 GraphRefiner；候选由另一 peer 的无梯度前向及同一 peer 的高置信队列构造，K=5、均匀权重；候选向量、候选选择与图在当前更新中停止梯度、队列快照冻结；重复候选按 (peer, slot, write_version) 合并并累加质量；W=r·A，c=W 列和，Q=r cᵀ（不剔除正边、不用均匀队列参考）；两个方向各自建图，不用转置重复计数。无效行按共同规则跳过并计数。
- critic：C0 cosine，F = X Yᵀ（单位化 query 与固定单位化候选）。
- 外系数：在共同原生状态上用固定 FIT batch 匹配 noisy 分支 query 梯度量级，一次标定后冻结；每数据集各自标定一次；记录 raw J、缩放后损失、梯度比。
- 训练不使用 oracle 噪声真值；真值只用于事后诊断。TEST 不在本轮训练/选择中读取，入口留给主会话在锁定后执行。

## 2. 实验臂

| 臂 | 支持 | 目标 | 回答 |
|---|---|---|---|
| D-VCS | 完整真实候选图 | 原始 VCS J | 主方法 |
| D-BCE | 同一图、同 W/Q/F | balanced BCE（P、Q 各半质量） | 二次目标是否有额外价值 |
| M-VCS | 同一候选袋的未归一化均值点 | VCS（对应的联合/乘积） | 保留二阶分散是否有益 |
| N-VCS | 同一候选袋的归一化均值点 | VCS | 连接常见 prototype 实现 |
| D-sNCE | 同一图 | 数学附件的 marginal-aware soft NCE | NCE 类目标获得同样信息时如何 |
| （复用）H0 | R25 原生 IN2R-code | 原生 SCE + GraphRefiner | 完整强系统参照 |

N-VCS 与 D-sNCE 原为“有信号后补”，按用户指示直接纳入首批，不设门槛。

## 3. 运行矩阵（全部完整训练长度）

- **阶段 A（技术门禁通过即提交）**：Flickr30K 40% 与 60% × seed1001 × 5 臂 = 10 单元。
- **阶段 B（紧接 A 排队，不等 A 的结果）**：Flickr30K 40%/60% × seed1002/1003 × 5 臂 = 20 单元。
- **阶段 C（Flickr 30 单元全部提交后排队，不设门槛）**：COCO 40% 与 CC152K（真实噪声，不注入）× D-VCS / D-BCE / M-VCS × 3 seed = 18 单元；同条件原生 H0 × 3 seed × 2 数据集 = 6 单元（R25 若届时已有则复用，不重复）。COCO 同一 checkpoint 同时报 1K 与 5K。外系数按各数据集 FIT 重新标定一次，其余冻结。

合计最多 30 + 24 = 54 个完整训练单元（每单元含两个 peer）。共同原生前缀：Flickr40%/1001 复用 R25 的 COMMON_PREFIX；其余 (数据集, 噪声, seed) 各生成一个原生前缀，所有 R26 臂共用，且供 R25/R26 的同格原生运行复用，不重复训练。

## 4. 技术门禁（唯一的启动条件）

1. CPU：PyTorch 实现与包内 NumPy 参考一致（J 稠密/矩形式值与梯度、平方分解、均值点恒等式、BCE 共同最优、重复候选合并后 W/r/c/Q 一致、Q 含对角）。
2. 真实数据 GPU smoke：候选与图无梯度、query 有梯度、队列快照在更新中不变、无 GraphRefiner 参数参与 noisy 分支、跳过行计数、完整保存恢复后首步一致、raw J 有限。
3. 外系数标定记录写出后才提交该数据集的正式单元。

门禁失败只修技术问题后重测，不改科学设计。

## 5. 报告

每 run：六项 recall、rSum、best DEV epoch、末轮/末五轮、GPU 时间、峰值显存、跳过计数、raw J 轨迹。主比较（逐 seed 配对、图像分组 bootstrap）：D-VCS−H0、D-VCS−D-BCE、D-VCS−M-VCS、D-VCS−N-VCS、D-VCS−D-sNCE；外部方法只列论文报告值。机制诊断按包 §8（候选覆盖、合法多样性 vs 错误混入、query 方向协方差、hub、同类别难例、队列年龄）。
