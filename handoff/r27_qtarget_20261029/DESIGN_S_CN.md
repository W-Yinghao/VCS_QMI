Publication copy (sample names aliased; numbers and conclusions unchanged).

# R27-S 实现设计：单目标、任务锚定的 VCS 修正（2026-10-30）

依据：`r27_gpu_supplement/vcs_gpu_supplement_20261030/GPU_EXPERIMENT_PLAN_CN.md`、`EXPERIMENT_MATRIX.json`、`math_checks.py`、`CPU_CHECKS.json`（zip sha256 3bbe7af7…，包内没有 MANIFEST）。用户批准 12 个完整单元。
代码：新快照 `code_v12/`，从 `code_v11/` 原样复制后只增加 R27-S 部分。`code/`、`code_v11/` 未改动：在跑的 A5/A6 及其续跑作业仍会导入这两个目录。

## 1. 矩阵与不变部分

S1–S6 × Flickr 40%/60% × seed1001 = 12 个单元，从共同前缀开始（epoch0 是前缀，本轮跑 epoch1–39）。A1、A3 复用作参照，不重训。以下与 R27 完全相同：

- 接口：候选身份由另一 peer 在自己队列中检索；候选由当前 peer 重编码并用同批中心化；队列 ID 与前缀槽排除。
- 结构与超参：K=5、P/Q、两个 peer、clean ranking 与 intra 损失、原 GraphRefiner 结构、warmup/prefix、优化器与 LR、数据增强、每方向系数 0.5、full-DEV 选择、推理路径。
- λ：`LAMBDA_flickr.json` 中冻结的 1.1448，只有 S1 使用。
- 单元脚本：`sbatch/r27s_unit.sbatch` 与 `r27_unit_v11.sbatch` 只差代码目录和续跑脚本名；10h 时限，超时自续跑，最多 3 次。
- 不读 TEST。

## 2. 梯度拆分（方案 §3，`r27_co_train.s_branch`）

每方向只做一次 refiner 前向，得到 P（单位原型）；Pd = stopgrad(P)。
- refiner 项：`L_ref = SCE(stopgrad(X), P)`，只更新当前 peer 的 refiner。
- 编码器项 `L_enc`（目标全部 stop-grad）：
  - S0（仅门禁用）：`SCE(X, Pd)`。
  - S1：`[SCE(X, Pd) + λ·SCE(X, T_free)] / (1+λ)`。
  - S2：`SCE(X, T_free)`。
  - S3–S6：`SCE(X, T_cap)`。
- 每方向合计 `0.5·L_ref + 0.5·L_enc`，一次 backward。两组参数不重叠。梯度裁剪（只裁编码器）、AdamW 与每步一次 `optimizer.step` 都不变。
- 门禁要求：S0 必须与 A1 完全一致。
- T_free 来自 R27 自由球 VCS 生成器（与 A3 同一函数）。若某行无效，沿用 R27 规则在有效行子矩阵上计算；计数写入日志。
- 日志里的 `raw_loss` 是 L_ref 的值，数值上等于原生 SCE；`enc_loss` 单独记录。两者相加不能与旧 loss 比较。

## 3. 受限生成器（`code_v12/r27s_targets.py`）

可行域 C_i = {‖z‖≤1, p_iᵀz ≥ γ}，γ = cos15°，p_i = Pd_i 归一化后的值。四个臂的目标沿用 R27 的逐行定义：S3 为 δᵀz，S4 为 f_V，S5 为 f_B，S6 为 f_N。J 和 P/Q 都不变，只改可行集合。

- **约化空间包含 span(Y, p_i)**：设 Q 为 span(Yᵀ) 的正交基，写 p_i = Q a_i + b_i e_i（e_i ⊥ span(Y)），z = Q u + v e_i。于是 Yz = Yr u，‖z‖² = ‖u‖² + v²，pᵀz = [a,b]·[u,v]。所有行共享 (M+1) 维空间，结果精确映回 d 维；不复用只含 Y 的旧 QR 基。门禁中 p 确实落在 span(Y) 之外。
- **S3**：闭式解。若 δ̂ 满足 p·δ̂ ≥ γ，取 δ̂；否则取锥边界上的点 γp + √(1−γ²)·w。
- **S4/S5/S6**：与 R27 `solve_spg` 相同的 SPG（同线搜索、同上限 4000、每 10 步检查一次）。投影换成精确的球∩半空间投影 `proj_ball_half`，原样复制自 `r27_audit_20261029/audit_constrained.py`，其 KKT 精确性见 `PROJECTION_KKT_20261030.json`。三者共同初值 z0 = (1+γ)/2·p_i：可行，在球内部，且 |F| < 1，保证 BCE 从 log 定义域内启动。
- **残差（四者同一定义）**：在活跃约束上求 min_{ν,κ≥0} ‖g − νz + κp‖（判活跃的容差 1e-9），rel = res/‖∇f(0)‖。停止条件 rel ≤ 1e-6，与 R27 生成器相同。凹最大化在凸集上仍满足证书 f* − f(z) ≤ 2·res。
- **输出**：T = normalize(z*)。raw 范数 ≥ γ，所以 T 仍在 15° 锥内。失败（非有限、范数过小或违反约束）时回退到 T = p_i，即原任务目标，S3–S6 同一规则，计数记录。
- **每步日志**：
  - 目标：与 p 的夹角（均值/最大）、raw 范数、cap/ball 约束激活比例。
  - 求解：迭代数与最大值、残差与最大值、上限命中、违反量、回退次数、ν̂、κ。
  - 目标值：J_v 与各臂自身目标在 z*、normalize(z*)、p 处的值。
  - 任务 margin（目标与 p 两种）、cos(T, p)。
  - 每 25 步的梯度范数：编码器 clean 部分、编码器 noisy 增量、refiner；用前后快照计算，不做额外 autograd。

## 4. 门禁（只有技术门禁，不设结果门槛）

- **CPU 门禁 `r27s_cpu_check.py` → `S_CPU_CHECK.json`**：
  - 受限求解器与独立参考 `audit_constrained.solve_constrained`（全维 FISTA、紧容差）及 SciPy SLSQP 比较，p 在 span(Y) 之外。检查：可行性、rel ≤ 1e-6、无上限命中、证书 gap ≤ 2·res、夹角 ≤ 15°。
  - 增广基精确性。
  - A1 恒等：用作者 GraphRefiner 与作者 `vse.infonce`（CPU 上把 `.cuda()` 置为空操作），比较梯度、一步 AdamW 后的参数、优化器状态、RNG。本地预跑全部为 0 差。
- **GPU smoke `r27s_smoke.sbatch` → `smoke_s_<job>/GATE_S.json`**：六臂各跑 2 轮 × 30 iter，然后 `r27s_gate.py` 检查：
  - `a1_identity`：同一真实状态下 A1 与 S0 各跑一步，比较编码器与 refiner 梯度、步后参数、AdamW 状态、torch/cuda RNG。
  - `grad_paths`：逐臂检查梯度路径。
  - `solver_real`：真实图加真实原型，与参考求解器比较。
  - `timing`：各臂每步耗时。
  - `arms_ran`：各臂跑完且损失有限。

## 5. 门禁结果（2026-10-30）

- **CPU 1075953：ALL_PASS**（`S_CPU_CHECK.json`）。
  - 增广基误差 ≤2.5e-16，p 落在 span(Y) 之外 0.29。
  - A1 恒等（作者 refiner 与 SCE）：梯度、步后参数、Adam 状态的差全部为 0，RNG 相同。
  - 受限求解器：可行性 ≤4.4e-16；rel ≤1e-6；无上限命中，无回退；与参考 FISTA 和 SLSQP 的差距都在 2·res 证书之内；夹角 ≤15°。
- **GPU smoke 1075954**（RTX PRO 6000，六臂各 2 轮 × 30 iter）：六臂、grad_paths、solver_real、timing、arms_ran 全部 PASS，但 a1_identity FAIL。
  - 失败现象：梯度差 0、RNG 相同，步后参数却差 7e-4。
  - 原因在门禁脚本本身：`restore()` 中 `optimizer.load_state_dict(snapshot)` 对同设备张量是别名，A1 那一步原地改写了快照，S0 因此从被污染的 Adam 状态出发。
  - 修复：在 `restore()` 里对快照做 deepcopy；训练代码不变。原 GATE 存为 `S_SMOKE_GATE_1075954_identity_artifact.json`。
- **门禁重跑 1075975**（沿用同一 smoke 输出，`sbatch/r27s_gate_only.sbatch`）：**all_pass**（`S_SMOKE_GATE_1075975.json`）。
  - **a1_identity 逐位相同**：编码器与 refiner 梯度、步后参数、AdamW 状态的差均为 0.0，torch 与 cuda RNG 相同。
  - **grad_paths**：六臂的编码器、当前 peer refiner 都有梯度；另一 peer 0 个参数有梯度；队列不变。cap 臂的 cos(T,p) 最小 0.965926 = cos15°。
  - **solver_real**（B=128，M=590，真实原型，p 在 span(Y) 外 0.26）：

    | 臂 | 迭代 | 最大 rel | 每方向耗时 | 与参考夹角 |
    |---|---|---|---|---|
    | S3 闭式 | 0 | 3e-15 | — | — |
    | S4 VCS | 10 | 7e-9 | 0.024 s | — |
    | S5 BCE | 10 | 1.5e-9 | 0.025 s | — |
    | S6 sNCE | 最多 280（均 63） | 9.6e-7 | 0.27 s | ≤0.0033 rad |

    cap 与 ball 两个约束在全部行上同时激活（激活比例 1.0）。
- **每步耗时**（相对 A1 的 126.8 ms）：S3 0.99×；S1、S2、S4、S5 约 1.35×；S6 6.4×。
- **成本估计**：S1–S5 每单元约 5–6 h，S6 约 15–18 h（需要 1–2 次自续跑）。12 个单元合计约 90 GPUh。
