Publication copy (sample names aliased; numbers and conclusions unchanged).

# R27 实现设计：共同候选接口 + VCS 二次伪目标（2026-10-29）

依据：`r27_review_design/VCS_R27_Review_and_Design_20261029/REVIEW_AND_R27_DESIGN_CN.md`（用户批准，§4 接口与生成器，§5 矩阵），数学参考 `math_checks.py`（同目录）。
宿主：R25 锁定的 IN2R-code（909e5329）+ R25 驱动 `r25_co_train.py`（文件原样复制，sha256 与 R25 `R25_CODE_PINNED_SHA256.txt` 一致，见 `R26_CODE_COPIED_SHA256.txt`）。R25/R26 目录未改动。
本文件记录每一项实现决定；**实验结果门槛一律不设**，只有技术门禁（§6）。

## 0. 矩阵

| 臂 | 代码名 | 设置 | 本轮新跑 |
|---|---|---|---|
| A0 | — | 原生 IN2R-code H0（同前缀、同预算） | 否：复用 R25 的 6 个 H0（`r25_in2r_20261027/groups/<cell>/H0`，结果 `RESULTS_PER_RUN_CORE.json`） |
| A1 | `a1_iface` | 共同身份 + 一致编码接口，仍原生 GraphRefiner + SCE | 6 |
| A2 | `a2_mean` | A1 + λ·SCE(query, sg normalize(μ_i)) | 6 |
| A3 | `a3_vcs` | A1 + λ·SCE(query, sg normalize(z*_VCS)) | 6 |
| A4 | `a4_bce` | A1 + λ·SCE(query, sg normalize(z*_BCE))，同图同球 | 6 |
| A5 | `a5_snce` | A1 + λ·SCE(query, sg normalize(z*_sNCE))，同图同球 | 6 |

Flickr30K 40%/60% × seed 1001/1002/1003，共 30 个新完整单元。共同前缀即 R25 H0 与 R26 使用的同一份 `prefix_state.pt`（`CELLS.json`，从 R26 CELLS 抄出 6 个 Flickr 格，路径不变；f40_s1003/f60_s1002/f60_s1003 的 R26 路径与 R25 副本 sha256 相同，见 `r25_in2r_20261027/PREFIX_REUSE_R26.json`）。完整 40 轮（前缀 epoch 0 + 本轮 epoch 1–39），主终点 best full-DEV rSum 不变，无 TEST。

**A0 复用的条件**：同一前缀文件、同一噪声映射、同一 RUN_LOCK 超参、同一 full DEV、同一 40 轮预算；唯一不同是硬件节点（H0 当时在 L40S，本轮按三分区调度），报告时披露。

## 1. 不变部分（所有臂与原生逐行一致）

`r27_co_train.train` 是 `r25_co_train.train` 的逐行副本（与 R26 相同的做法），以下保持不变：warmup/前缀、eval_train + GMM 划分、clean ranking + intra 两项、高置信 FIFO 写入规则（`batch_prob > avg_clean_prob`）、grad clip（只裁编码器参数）、AdamW、LR 表、full-state 保存/续跑、full DEV 验证与 peer 平均读出。已知原生缺陷防护保留并对所有臂生效：labeled batch <2 跳过（R25 NATIVE_FIDELITY §3）、空 caption 保留规则（`data.py` R25 修复版，原样复制）。noisy batch <2 或 unlabeled 为空时跳过整个 noisy 分支（R25 共同规则）。

## 2. 共同接口（A1–A5 完全相同，`r27_interface.py`）

### 2.1 队列身份缓冲（开放点 1）

- 原生 `MemoryQueue` 只存嵌入。每个队列附一个并行缓冲 `q.r27_ids`（GPU long 张量，长度 = queue_size=65536）：`cap` = 全局训练 caption 下标；`img` = 该 caption 的**观测**图像下标（训练可见的 `t2i_index`，可能是注入错配后的）；`wepoch`/`witer` = 写入时的 co-train epoch 与迭代；`peer` = 写入者（0=A，1=B）。
- 写入：`enqueue_with_ids` 先调用原生 `enqueue`（作者代码不改），再按同一指针算术写 ID，并断言写后 `ptr/is_full` 与预期一致（含回绕）。labeled loader 的 batch 下标经 `pred.nonzero()` 映射到全局 caption 下标（`get_loader` 包装里附加 `dataset.r27_gidx`，并逐 997 条抽查 caption 字符串一致）。
- 注入噪声真值（`dataset._t2i_index`、`_labels`、loader 输出的 `batch_clean_labels_*`）**从不进入训练器**；`img` 只取观测下标。GPU 门禁 `truth_isolation` 用噪声文件核对：所有带 ID 的槽 `img == 观测下标`，包括与真值不同的注入错配槽。
- 保存/续跑：`model_full_state` 包装把 ID 缓冲（CPU 副本）加入 `state_last.pt` 的 `model_A/model_B['r27_queue_ids']`；`load_full_state` 包装在原生恢复（含 RNG，最后一步）之后恢复 ID，不消耗 RNG。
- **前缀中的无 ID 槽（决定）**：前缀队列由原生 epoch 0 写入，没有 ID，也无法从张量恢复。所有这类槽记 `cap=-1`，**在搜索中排除，直到被覆盖**；A1–A5 完全相同。不重建、不回放前缀（那会给接口额外加一次非原生的全数据前向）。
  - 影响（CPU 门禁 `prefix_queues` 读出 6 个前缀的 ptr/is_full）：40% 前缀每个 peer 在 epoch 0 写入约 77k 条（>65536，队列已满且回绕到 ~12k）；60% 写入约 49k 条（未满）。A 在每轮先训练，搜索 B 的队列；B 的队列在 A 的 epoch 1 期间全是前缀槽 → **peer A 在 epoch 1 没有任何可用候选（noisy 分支对所有臂整体跳过，计为 `i2t/t2i_no_proto`）**。B 在 epoch 1 搜 A 的队列：40% 约 100%、60% 约 75% 可用；epoch 2 起两者 ~100%。注意原生自身在 co-train epoch 0 也有同样的“A 首轮无候选”（B 队列尚空，R26/H0 EPOCHS 中 epoch0 `i2t_no_proto=666`），R27 把这一现象在 epoch 1 再重复一次，属于接口的一部分，计入 A1−A0。每轮可用比例写入 EPOCHS（`{i2t,t2i}_usable_frac_mean`）。

### 2.2 候选身份（与 R26 相同的搜索规则）

另一 peer（net2，eval 模式，no grad）编码当前 noisy batch，在**它自己的**队列里做同模态余弦 top-K（K=5，均匀权重 1/K），屏蔽无 ID 槽。原子按样本 ID 合并（i2t：caption ID；t2i：观测图像 ID —— 同一图像的不同 caption 槽合并成一个图像原子），质量累加：`A[i,j] = (#指向原子 j 的槽)/K`。可用 ID 槽 <K 时该方向跳过（与原生 `proto is None` 同处理）。

### 2.3 一致编码（开放点 2）

- 查看 `vse.py` `forward_emb`：两个模态各自 `enc(...)` → 减 **batch 内均值** → `l2norm`。因此单独编码一批候选并用候选自己的均值中心化，会落在与 query 不同的中心化上下文里。
- 决定：query 保持原生 `forward_emb` 数值（`forward_emb_ctx` 是逐行副本，只额外返回 detach 的两个预中心化均值；门禁 `forward_ctx` 核对与原生 `forward_emb` **逐位相同**）。候选由**当前 peer**（net）在 **eval 模式、no grad** 下从原始数据重编码（图像：全部 36 个区域、不做 20% 区域丢弃；caption：`process_caption(drop=False)`，不做 token 增广，不消耗任何 RNG），减去**同一 noisy batch 同一模态**的预中心化均值（i2t 候选是文本 → 减文本均值；t2i 候选是图像 → 减图像均值），再 `l2norm`。候选、均值、A、背景 c 全部停止梯度。
- 为什么 eval 模式：设计要求“以当前 peer 的固定状态重新表达”；train 模式会更新 BN running stats（改变状态）并消耗 dropout RNG（改变数据/模型随机流）。代价：query 的均值来自 train 模式前向（dropout + BN batch 统计），候选是 eval 前向，两者存在系统差异。为量化这一差异，每 50 步对 query batch 自身做一次 eval 重编码、用同一均值中心化，记录与原生 query 的平均余弦（`selfcos_img_mean / selfcos_txt_mean`）。
- 工程：训练图像特征（29000×36×2048 fp32，8.55GB）一次性放到 GPU（从 TMPDIR 分块拷贝），caption token 按 ID 缓存。

### 2.4 A1 的精确定义（开放点 3）

A1 = 原生 GraphRefiner + 原生 SCE（`vse.infonce`，correction_weight 0.5，每方向单独 backward），只把输入的 K 个邻居换成接口给出的候选：`neigh[i,k] = Y[atom(i,k)]`（[B,K,d]，重复槽保留，与原生 [B,K,d] 输入形状一致）。相对 A0 的**全部**差异：

1. 候选身份：原生用当前 peer 的 train 模式 query（带梯度张量）去搜另一 peer 队列里的缓存嵌入（跨 peer 坐标比较）；A1 用另一 peer 的 eval 模式 query 搜它自己的队列（同 peer 坐标）。
2. 候选向量：原生是另一 peer 过去某个时刻 train 模式下、在它自己的 labeled batch 中心化后的缓存嵌入；A1 是当前 peer 的 eval 重编码、用当前 noisy batch 的均值中心化。
3. 前缀无 ID 槽排除（§2.1）。
4. 队列写入多一份 ID（不改变嵌入写入）。

其余（refiner 参数、优化器参数组、dropout RNG 消耗——refiner 调用次数与形状相同、裁剪、权重 0.5）不变。A1−A0 只能归因于这一整套接口，不能拆成单因素。

## 3. 生成器（A2–A5，`r27_targets.py`）——先写数学，后写代码

对一个方向的一次更新，记 B 个 query、M 个原子 y_j（单位、停止梯度），A ∈ R^{B×M} 行和为 1，r_i = 1/B，
c_j = Σ_i r_i A_ij（背景边缘），μ_i = Σ_j A_ij y_j，μ_Q = Σ_j c_j y_j，δ_i = μ_i − μ_Q，
H_i = Σ_j (A_ij + c_j) y_j y_jᵀ，F_ij(z) = zᵀ y_j。共同可行域 𝔹 = {‖z‖ ≤ 1}（保证 |F| ≤ 1，即有界余弦 critic）。
W_ij = r_i A_ij，Q_ij = r_i c_j（与 R26 `coupling` 相同，含对角）。三个生成器分别是 R26 冻结目标在“第 i 行 query 表示自由”时的逐行版本：

- **A2 mean**：t_i = normalize(μ_i)。无内层问题。
- **A3 VCS**：f_V(z) = Σ_j (A_ij − c_j)F_j − ½ Σ_j (A_ij + c_j)F_j² = δ_iᵀz − ½ zᵀH_i z，z*_V = argmax_𝔹 f_V。Σ_i r_i f_V(x_i) = R26 `j_graph(W,Q,XYᵀ)`（CPU 门禁 `objective_identity` 核对到 1e-17）。
- **A4 BCE**：f_B(z) = ½ Σ_j A_ij log((1+F_j)/2) + ½ Σ_j c_j log((1−F_j)/2)。Σ_i r_i f_B(x_i) = −R26 `bce_graph`（未 clip 时）。在 𝔹 内凹；A_ij>0 的原子同时 c_j>0，所以对它们 F→±1 时目标 → −∞（log 障碍），最优点严格在 log 定义域内。
- **A5 sNCE**（marginal-aware soft-NCE，τ=0.05 = 宿主 SCE 温度）：f_N(z) = Σ_j A_ij F_j/τ − log Σ_j c_j exp(F_j/τ)。Σ_i r_i f_N(x_i) = −R26 `snce_graph`。凹（线性 − log-sum-exp）。

共同性质：∇f_V(0) = δ_i，∇f_B(0) = δ_i/2，∇f_N(0) = δ_i/τ —— 三者在零点方向相同（CPU 门禁 `grad_at_zero`），这对应 §3.2 “F 近 0 时尺度匹配后梯度相近”；差别全部来自二阶及以上结构。

**共同初始化**：z⁰ = 0（三个目标都有限、可行，且是唯一不偏向任何一个目标的点）。A2 的归一化均值点不作初始化，因为 t2i 中五个槽可能是同一图像，归一化均值恰好等于某个原子，BCE 的 log(1−F) 在该点为 −∞。

**求解器**（全部 float64）：
- VCS（精确）：H_i = H_Q + V_iV_iᵀ，H_Q = YᵀCY 所有行共享，V_i = Y_{S_i}ᵀ diag(√A_i)（≤K 列）。每方向每步一次 M×M 特征分解（Gram 形式 C^{1/2}YYᵀC^{1/2}，相对 1e-12 截断零空间）得到 range(Yᵀ) 的正交基 U，所有 δ_i、H_i 都在该子空间内；每行用秩 K 的 Woodbury 精确求 (H_i+νI)⁻¹δ_i；ν=0 若 ‖H_i⁺δ_i‖ ≤ 1（内点解），否则 Moré–Sorensen Newton（安全括号 + 二分回退）解 ‖z(ν)‖ = 1。**不为每个样本建立或分解 1024×1024 矩阵**。最终用精确矩乘 H_i z = Yᵀ[(A_i+c)⊙(Yz)] 在全空间计算 KKT 残差。
- BCE / sNCE：批量非单调谱投影梯度（SPG，BB1/BB2 交替步长，Grippo 非单调 Armijo，拒绝非有限目标即 BCE 越出 log 域），z⁰=0。因为 z⁰=0 且所有梯度在 range(Yᵀ) 内、球投影只做缩放，迭代精确地留在 span(Yᵀ)；用 Yᵀ 的约化 QR 基计算（精确，不是近似）。线搜索对 8 个步长分数一次性批量评估（T_c = T + λ·DYᵀ，每迭代只有 2 次矩阵乘、无主机同步），每 10 次迭代精确重算 T 并检查收敛；上限 4000 次。原型中 FISTA 的 sNCE 迭代数约为 SPG 的 1.5–2 倍，故生产用 SPG，FISTA 只作 CPU/GPU 门禁里的独立对照求解器。（曾试过 fp32 预求解 + fp64 收尾：fp32 在 1e-5 处 Armijo 停滞，放弃；生产全程 fp64。CPU 门禁 1073236 运行的是逐迭代同步线搜索的早期 SPG 版本，生产版本由 GPU 门禁 `solver_real` 与 smoke 统计核验，并在本地小问题上复核收敛。）
- **统一残差与停止规则**：g = ∇f(z)；若 ‖z‖ = 1：ν̂ = max(zᵀg, 0)，res = ‖g − ν̂z‖；否则 res = ‖g‖。rel = res/‖∇f(0)‖。停止：rel ≤ 1e-6（三者相同；VCS 精确解实际 ~1e-13）。对球上的凹最大化有证书 f* − f(z) ≤ 2·res，CPU/GPU 门禁都按此证书核对与通用优化器（SciPy SLSQP）/更紧 FISTA 的差距。每行记录迭代数、是否触顶（cap）、ν̂、是否内点。
- 输出：t_i = normalize(z*_i)（任务读出）。‖z*‖ < 1e-6 或非有限 → 该行回退到 A2 的均值目标（所有臂同规则，计数）；均值也无效（‖μ_i‖ ≤ 1e-6）→ 该行关闭辅助项（只关辅助，不删主任务样本）。
- **归一化声明**：内层最优性证书只对未归一化的 z* 成立；normalize 是任务读出，可能改变（降低）原目标值。因此每步同时记录 f_V 在 z*（`Jv_raw`）、在 normalize(z*)（`Jv_norm`）、在均值目标（`Jv_mean`）的值，以及各臂自己的目标在 z*、normalize(z*)、均值、零点的值（`obj_raw/obj_norm/obj_mean/obj_zero`）和 ‖z*‖。BCE 在归一化目标恰好等于某个候选时目标为 −∞，均值只对有限项求平均并计数（`cnt_*_obj_*_nonfinite`）。

**一个需要知道的数值事实（CPU 门禁实测）**：sNCE 目标在最优点附近很平。rel KKT 1e-6 时，SPG 解与 SLSQP / 紧 FISTA 解的目标差 <1e-6（在 2·res 证书内），但方向在个别小维度构造问题上相差可达 0.28 rad。也就是说 A5 的伪目标方向在这一容差下本身不唯一。GPU 门禁 `solver_real` 在真实图上报告 SPG 解与 1e-9 FISTA 解的最大夹角，用来判断真实数据上的方向松弛有多大（见 §6.3 结果与 §9 待决）。

## 4. 辅助项与梯度路径

L_total = L_native(A1 接口) + λ · [SCE(X_img, sg t_text) + SCE(X_txt, sg t_img)]/2，SCE = 宿主 `vse.infonce(XTᵀ)`（对角为正对，B×B）。实现上每方向 `loss = 0.5·SCE(X, refiner(neigh)) + 0.5·λ·SCE(X, t)`，单独 backward（与原生每方向单独 backward 相同）。梯度：query（当前 peer 编码器）+ 当前 peer 的 GraphRefiner（原生路径）；候选、邻居、目标、中心均值无梯度；另一 peer 的编码器、refiner、队列与 ID 不变（GPU 门禁 `grad_paths` 核对）。推理不变：peer 平均余弦检索，无测试期图。

## 5. λ 标定（开放点 4）

- 一个 λ 供 A2–A5 共用，在共同前缀 f40_s1001（与 R26 SCALE_flickr 同一格）一次标定后冻结，`LAMBDA_flickr.json`；Flickr 的 6 格全部使用。
- 规则：λ = median_{(peer,batch)} ‖∇_θenc L_nat‖ / ‖∇_θenc L_mean‖，其中 L_nat = ½SCE_i2t + ½SCE_t2i 是 **A1 接口下**的原生 noisy 分支（GraphRefiner 原型），L_mean = ½(SCE_i2t + SCE_t2i) 是均值目标辅助项；θenc = 当前 peer 编码器参数（net.params，不含 refiner）。含义：λ·aux 的编码器梯度量级与它所伴随的原生 noisy 分支相当。固定 FIT 批：每 peer 前 32 个 unlabeled batch（generator seed 20261029），共 64 个比值。
- 参照选 A1 接口下的原生分支而不是原 A0 分支，因为它才是与辅助项同处一次更新的那个原生项；原 A0 路径（当前 query 搜另一 peer 原前缀队列的缓存嵌入）的比值一并记录（`diagnostic_ratio_native0_over_mean`、`diagnostic_ratio_nat_over_native0`）。
- 前缀无 ID 的处理（仅标定）：前缀队列没有 ID，接口在前缀上没有候选。标定时在内存里为每个 peer 重建一个带 ID 的队列：用该 peer 自己 epoch-1 labeled loader（seed 20261030）跑一遍 train 模式前向（no grad，高置信规则同原生），之后恢复该 peer 编码器状态（BN 统计），不写回任何东西。这是“标定用重构队列”，与训练中的真实队列不同，已标明。
- 同时记录（不用于 λ）：vcs/bce/snce 目标辅助项的同类比值、query 梯度比、cos(∇L_nat, ∇L_aux)、λ·aux 与 clean 损失的编码器梯度比、生成器在真实 FIT 图上的残差/迭代/时间。
- **λ 网格建议（未运行，待主会话决定）**：见 §9。

## 6. 技术门禁（唯一启动条件，无结果门槛）

### 6.1 CPU（`r27_cpu_check.py` → `CPU_CHECK.json`，job 1073236，ALL_PASS）
VCS 精确解 vs 包内参考 `quadratic_ball_refiner`（用 ast 从 math_checks.py 原样取函数源码，不执行该模块，避免改写包内 math_checks.json）：z 差 ≤3.9e-13、ν 差 ≤4.1e-13、KKT ≤5.9e-13，17 个内点行 + 109 个边界行；vs SLSQP 与 FISTA；矩乘恒等式；三目标与 R26 冻结 `j_graph/bce_graph/snce_graph` 的恒等式；零点梯度；有限差分；BCE/sNCE 的 SPG vs SLSQP/FISTA（证书 gap ≤ 2·res）；ID 缓冲在作者 `MemoryQueue.enqueue`（CPU 张量）上 9 次回绕对齐、torch.save/load 往返、前缀加载全 −1、搜索排除无 ID 槽、可用 <K 时返回 None；停止梯度；6 个前缀的 ptr/is_full。

### 6.2 真实数据 GPU smoke + gate（`r27_smoke.sbatch` → `smoke_<job>/GATE.json`）
真实 f40_s1001 前缀，五臂各跑 2 个 co-train epoch × 30 iter（λ=1 占位），然后 `r27_gate.py`：`forward_ctx`、`grad_paths`、`current_peer`、`truth_isolation`、`state_roundtrip`（含 ID 缓冲、优化器、RNG）、`resume`（加载 vs 加载-保存-加载后首步参数差 <1e-4，宿主 R26 门禁同一容差）、`solver_real`、`timing`（同一状态、同 30 iter：R25 原生、R26 d_vcs、R27 五臂 ms/step）、`arms_ran`。

### 6.3 λ 标定（`r27_calib.sbatch` → `LAMBDA_flickr.json`）

结果见 §10。

## 7. 每轮日志（EPOCHS.json，R.main 原有字段 + 下列）

每 peer、每方向（i2t/t2i）：`usable_frac_mean`（另一 peer 队列中带 ID 的比例）、`no_proto`、`n_atoms`、`shared_frac`、`max_col_mass`、`cand_age_epochs`、`self_hit`（候选恰为该 query 自己的观测 caption/图像的比例，训练可见信息，仅诊断）、原生 SCE `raw_loss`、`aux_loss`、`J_query`（当前 query 在该图上的 VCS J）、`Jv_raw/Jv_norm/Jv_mean`、`z_norm` 与 `min_znorm`、`obj_raw/obj_norm/obj_mean/obj_zero`、`iters` 与 `max_iters`、`rel_res` 与 `max_rel_res`、`cnt_*_cap_hits`、`cnt_*_interior`、`nu_hat`、`cnt_*_fallback`、`cnt_*_invalid_rows`、`cnt_*_aux_skipped`、`cnt_*_near_log_boundary`（BCE）、`backtracks`、`max_range_loss`（VCS 零空间截断丢失）、`cos_to_mean`。另：`selfcos_img/txt_mean`、`time_*_mean_s`（每 25 步同步计时：另一 peer 前向、搜索、重编码、生成器、整个 noisy 分支、整步）、`train_seconds`；宿主原有 split_A/B（数量、precision/recall，仅事后）、六项 recall、lr、epoch 秒数、峰值显存。

## 8. 提交行与成本

`LINES_SEED1001.txt`（10 行）、`LINES_REST.txt`（20 行），每行 `sbatch --job-name=r27_<arm>_<cell> --time=… --nice=40 sbatch/r27_unit.sbatch <cell> <arm>`。单元脚本：TMPDIR 数据暂存、`timeout` 自续跑（≤3 次，不用 --dependency）、λ 从 `LAMBDA_flickr.json` 读、输出 `/projects/.../outputs_v9/r27_qtarget_20261029/runs/<cell>/<arm>`。**本代理不提交 30 个正式单元**。成本估计见 §10。

## 9. 偏离、风险与待主会话决定的点（详见 §10）

## 10. 门禁结果、λ、成本（2026-10-29）

- CPU 门禁 **1073236 ALL_PASS**（`CPU_CHECK.json`）。
- λ 标定 **1073239**（RTX PRO 6000，`LAMBDA_flickr.json`）：**λ = 1.1448**（64 个比值 1.057–1.293，四分位 1.105/1.194）。cos(∇L_nat, ∇L_mean) 中位 0.747；原 A0 路径与 A1 接口原生项的编码器梯度方向几乎不同（cos 中位 0.173），‖∇L_nat(A1)‖/‖∇L_native(A0)‖ 中位 1.39（0.82–2.08）。λ·aux 与 clean 的编码器梯度比中位 0.091（与原生 noisy/clean 0.093 相当）。重构标定队列每 peer 65536 条（~18 s）。
- GPU smoke **1073252 失败**（`summarize` 包装自调用 → RecursionError，第一轮 30 iter 后；训练步本身无误），修复后 **1073255 全部 PASS**（`SMOKE_GATE_1073255.json`，原件 `outputs_v9/r27_qtarget_20261029/smoke_1073255/GATE.json`）：forward_ctx 逐位相同；grad_paths 五臂（当前 peer 编码器 + 当前 peer refiner 有梯度，另一 peer 0 个参数有梯度，队列/ID 不变，A1 无辅助、A2–A5 有）；current_peer 当前 peer 重编码逐位复现（maxdiff 0），另一 peer 重编码差 ~0.37；truth_isolation（214 个注入错配的带 ID 槽均存观测下标）；state_roundtrip（含 ID、优化器、RNG）；resume 首步参数差 0.0；solver_real（真实图 B=128, M=590：VCS 精确 rel 2e-13、Newton ≤5 次、与 SPG(1e-10) 夹角 2e-8；BCE SPG 10 次、rel 1.1e-8；sNCE SPG ≤430 次（均 161）、rel ≤1e-6、与 4 万次 FISTA 解夹角 ≤0.0037 rad、目标差 1.5e-8，均在 2·res 证书内）；arms_ran 五臂两轮有限。
- 协调者要求的一个数：同一 ID 下“当前 peer 重编码候选”与“另一 peer 队列缓存向量”的平均余弦 **i2t 0.166、t2i 0.186**（smoke 状态 = 前缀 + 2×30 iter；前缀本身无 ID 无法直接测）。与 R26 固定图诊断的跨 peer 不相容（0.12–0.19）一致。
- selfcos（query 的 eval 重编码、同均值中心化 vs 原生 train query）：图像 ~0.93、文本 ~0.96 —— train/eval 模式差异存在但不大。
- smoke 中 peer A 第 1 轮 `no_proto=30`（预期，§2.1）；B 第 1 轮可用比例只有 5%（30 iter 只写入 ~3.5k 条，正式运行一轮写入 ~77k/49k，按 §2.1 推算）。
- **每步耗时**（同一状态、同 30 iter、RTX PRO 6000，含数据加载，A 的一步）：R25 原生 90 ms、R26 d_vcs 94 ms；A1 124（R26 的 1.32×）、A2 122（1.30×）、A3 170（1.80×）、A4 163（1.72×）、A5 592 ms（**6.26×**）。A5 的代价来自 sNCE 平坦目标需要数百次 SPG 迭代，每方向 ~0.25 s。
- **成本估计**（以 R26 L40S 单元 13,010 s/40 轮、约 300 s/轮、其中约 2/3 为训练步推算；RTX 更快）：A1/A2 ≈ 4–5 h、A3/A4 ≈ 5–6 h、A5 ≈ 12–17 h/单元。30 单元合计约 **150–210 GPUh**（A5 六个约 70–100 GPUh）。A5 在 10 h 时限下预计需 2 次自续跑（≤3 次，总可用约 27 h）。

### 待主会话决定 / 风险

1. **peer A 在 epoch 1 无 noisy 分支**（前缀无 ID 槽排除）。所有臂相同，计入 A1−A0。若不接受，备选是在训练开始前按标定方式重建带 ID 队列（额外一次全数据前向，非原生），需另行决定；本实现未做。
2. **A5 成本 6.3×、方向松弛**：rel KKT 1e-6 下真实图方向松弛 ≤0.004 rad（toy 构造中可达 0.28 rad）。若嫌贵，可选（须对 A4/A5 同时适用）：容差放到 1e-5，或接受现状；本实现保持 1e-6 不变。
3. **λ 网格建议（未运行）**：若要 CAL，建议所有 A2–A5 共用 {0.5, 1, 2}×λ̂，只在 f40_s1001 一格、seed1001、full DEV 上选一个共同 λ，选择预算单列；默认不做。
4. eval 模式重编码 vs train 模式 query 中心化均值不一致（selfcos 0.93/0.96），已记录，不修正。
5. A0 硬件：R25 H0 在 L40S；R27 三分区调度，报告时披露。
6. `CODE_SHA256.txt` 记录了本次 smoke 时的代码哈希；正式单元运行时会把代码哈希写入各自 RUN_*.json，可与之核对。
