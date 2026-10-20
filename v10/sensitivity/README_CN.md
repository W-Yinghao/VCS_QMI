# V10 看过 DEV 之后的敏感性诊断 D1–D3：预登记

状态：**预登记，写于 2026-10-19 12:10 UTC，早于任何 D1/D2/D3 作业和任何真实数据的试运行。**
机器可读版本：同目录 `PREREGISTRATION.json`（与本文内容一致，以 JSON 为准）。
分支：`v10/sensitivity-diagnostics`，从 `ddd3bcb` 开出；工作树 `ssl_pilot_v10_diag`。

## 0 标签与边界

- 这三项都是**看过 DEV 结果之后才设计的敏感性诊断（post-DEV sensitivity）**。设计前已经看过：补充方法冻结的 DEV32/DEV24 行；独立审查 `handoff/review_claude_20261019/REVIEW_CN.md`；R6 REG-P2 与 V9-G1 的 CoMIR CAL/DEV 记录。
- 任何 D1/D2/D3 结果都**不替换、不重排、不重选**冻结主表中的任何一行。它们不进入方法选择，不进入 TEST_REPEAT 的冻结条件，也不作为方法优劣、强基线或新颖性结论的依据（决定 §5）。
- 不做新训练。D1 只加载已保存的 CoMIR 每 epoch checkpoint；D2 不需要训练；D3 运行补充方法自己的 gate，按其定义在 FIT32 上走 3 步优化器以检查续跑。这 3 步只用于验证，不是科学 seed。
- 数据角色：FIT（只经由未改动的 trainer 构造和 D3 gate）、CAL8、DEV32；DEV24 是 DEV32 行的派生子集。TEST134 维持 HOLD 不读；EPFL 未激活不读；`evaluator_only` 不读；CONFIRM=0。数据门用冻结的 `v10.adapter.DataGuard`，manifest sha256 为 `cc309cec…`，数据锁为 BF1 的 `DATA_LOCK.json`（`e65c60a3…`）。
- 复用的补充方法代码一律从冻结工作树只读导入，导入前逐文件核对 sha256（对照冻结的 RELEASE.json 或 validation MANIFEST）。冻结工作树和已有 bundle 不写入任何文件。
- 输出位置：
  - 记录 bundle：`/home/infres/yinwang/CS_QMI/handoff/v10_sensitivity_20261019/`
  - 逐配准行：`/projects/EEG-foundation-model/yinghao/outputs_v10/sensitivity_20261019/`
  - D3 gate：`<bundle>/D3/validation/`
- 失败即停：身份、哈希、覆盖或数据角色任何一项不符，该单元以非零退出并写失败记录。不盲目重试，不把部分结果当完整结果报告。

## 1 冻结输入

| 用途 | 文件 | sha256 |
|---|---|---|
| 数据锁 | `handoff/v10_bf1_20261017/DATA_LOCK.json` | `e65c60a3…72c5` |
| 父协议 | `handoff/v10_bf1_20261017/PROTOCOL.json` | `de4c0c39…5e9d` |
| CoMIR release（D1） | `handoff/v10_native_geometry_recovery_20261019/families/4/RELEASE.json`，源码树 `ssl_pilot_v10_native_recovery_20261019` @ `ddd3bcb`，code `3ec64cb5…` | `a8cd6bf3…ef8d` |
| MIND release（D2） | `handoff/v10_method_pipeline_20261019/families/6/RELEASE.json`，源码树 `ssl_pilot_v10_pipeline_20261019` @ `ecd8a95`，code `3f1405ed…` | `188204c3…f961` |
| 补充方法 validation manifest（D3） | `handoff/v10_method_pipeline_20261019/validation/MANIFEST.json` | `8befdabb…292b` |
| 已接受的 L40S gate | `validation/runs/1047322/MLP_VCS_perm8/GATE.json`；`runs/1047323/InfoNCE_host[J_perm1]/GATE.json` | `d86ff365…`；`590db6c6…` |

D1 的四个冻结单元及 COMPLETE.json 的 sha256 见 JSON 的 `frozen_inputs.comir_release.units`。

## 2 公共规则（与冻结行一致）

- 搜索器：未改动的 R6-REG-P2 `Q.optimise_rigid`。21 个角度（步长 3°，|θ|≤30°）× 508 个平移（4 px 圆盘，|t|≤51），取前 3 个起点，各做 100 步 Adam（lr 0.5），最终取精确能量最大者。
- 扰动：`Q.perturbations_p2(core)`，每个 core 10 个，与所有冻结行相同。位置：每个 (core, 扰动) 用 `seed_of('R6-REG-P2|pos|core|i')` 生成 4096 个支撑盘上的位置，与冻结行共享。
- 误差：`Q.corner_error_full`，即中心 834 px 区域四角平均误差，全分辨率像素。
- 指标：冻结的 `records.metrics`，即 success<42、success<9（配准次数的比例）和中位误差。
- 选择规则：决定 §8.1 第 1–4 项，依次为 success<42 高者、success<9 高者、中位误差低者、较早的 checkpoint。第 5 项候选顺序在这里不起作用，因为 CoMIR_native 只有一个冻结候选。实现直接用冻结的 `records.selection_key`。
- DEV24：沿用冻结 runner 的规则，排除与 FIT/CAL 同 TMA 位置的 DEV core。被排除的 8 个是 1B_F4、1B_H4、1B_H9、2B_D5、2B_G3、2B_G5、2B_H2、2B_I5。
- 统计：只做描述。以 core 为单位，同一 core 的 10 个扰动不当作独立样本；不同变体之间按 core 配对。用 95% 百分位 bootstrap 对 core 重采样，10000 次，`numpy default_rng(20261019)`。DEV32 的 32 个 core 正好落在 32 个 TMA 位置组里，所以按 core 重采样与按位置组重采样是一回事。
- 旋转分层：按 |θ*| 分为 [0,10)、[10,20)、[20,30]，最后一层包含上界；θ* 取记录中的 `theta`。

## 3 D1：CoMIR_native 在 CAL 上选 checkpoint

**问题。** 冻结的 CoMIR 只在第 23 epoch 读 CAL，但 23 个 epoch 的 checkpoint 全部保存了；学习型宿主 critic 则是在 12 个 checkpoint 中按 CAL 选。如果 CoMIR 也用公共规则在 CAL 上选 epoch，它的 DEV32/DEV24 会不会变？

**单元。** 四个冻结单元：调参运行 CAL_1017（seed 1017），以及最终运行 FINAL_1001、FINAL_1002、FINAL_1003。

**checkpoint，运行前写定。**

| epoch | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 | 576 | 640 | 704 | 736 |

- 取法是每隔一个 epoch 取一个，再加上最后的第 23 epoch。这样 12 个点的分布与 BF1 的 12 个 checkpoint 一致：BF1 在 250…3000 步，即每 1/12 训练长度一个点。
- 每 epoch 32 步。第 23 epoch（step 736）就是冻结行所用的点。
- 其余 11 个 epoch 此前只保存过，从未在 CAL 或 DEV 上评估过。

**评估与选择。**

- 对 12 个 step，逐一用冻结的 `checkpoints.load` 加载冻结单元的 `checkpoints/step_XXXX.pt`。加载时必须核对身份、文件 sha256、解析后的配置和恢复后的模型哈希。checkpoint 不作任何改动。
- 用冻结的 `run.evaluate` 做 CAL8 × 10 配准，内部是未改动的 `RegistrationSession`：先编码再配准，参考图 latent 每个 core 编码一次，移动图 latent 每个扰动编码一次，两者都计时并计入 `full_registration_seconds`。
- 每个单元在 12 行 CAL 结果上取冻结 `selection_key` 的最小者，写入 `SELECTION.json`。这一步必须在读取任何 DEV 之前完成。

**DEV。**

- 只评估 FINAL_1001/1002/1003，而且只在各自 CAL 选中的 step 上做 DEV32 × 10，并由这些行计算 DEV24 和逐 core 指标。CAL_1017 不做 DEV。
- 如果选中的仍是 step 736，DEV 照样评估，因为它就是 CAL 选中的 checkpoint；这些行同时用作冻结 DEV 行的重放核对。

**QC 哨兵。**

- Q1 身份：冻结 release 的 sha256、全部源码文件、冻结树的 git HEAD、冻结单元 COMPLETE（`records.verified_complete`）、CONFIG 一致，以及 checkpoint 的身份、sha 和模型哈希。
- Q2 覆盖：12 次完整 CAL 读数，每次 80 次配准；DEV 320 行；DEV24 为 24 个 core、240 行。
- Q3 CAL 重放：把 D1 在 step 736 的 CAL 行与同一单元冻结的 `CAL/step_0736.jsonl` 比较，报告 err_full、θ̂、t̂、F_hat、F_true 完全相同的行数和最大绝对差。预期在 L40S 上完全相同。若不相同，予以披露，但 D1 不中止，仍按 D1 自己的 12 次读数做选择。
- Q4 数据角色：DATA_READS 的角色只能在 {FIT, CAL, DEV} 之内；DEV 只在 `SELECTION.json` 写出之后读取，且只用于选中的 checkpoint；不读任何 held core。

**报告内容。**

1. 每个单元的 12 行 CAL 结果（success42、success9、中位误差）及选中的 epoch；
2. 每个最终 seed 在选中 epoch 上的 DEV32、DEV24，与冻结的第 23 epoch 行对照；
3. DEV32 success<42 的逐 core 配对差（选中减冻结），分 seed 给出，再给三 seed 平均，附 32 core 的 bootstrap CI；success<9 同样处理；
4. 两种读数的 DEV32 success<42 旋转分层；
5. Q1–Q4 和实测成本。

**解读网格，预先写定。**

- G1：三个最终 seed 都选第 23 epoch。结论是，在 CAL 上选 epoch 不会改变冻结的 CoMIR 行，CoMIR DEV 偏低不是只读最后 epoch 造成的。
- G2：至少一个 seed 选了更早的 epoch，并且三 seed 平均的 DEV32 success<42 配对差 ≥ +0.03、CI 在 0 之上。结论是，若在 CAL 上选 epoch（看过 DEV 之后），CoMIR 的 DEV32 会高出这么多，单 epoch 读数对 CoMIR 不利。该结果放在冻结行旁边报告，不替换冻结行。
- G3：选了更早的 epoch，但平均差落在 (−0.03, +0.03) 内，或 CI 含 0。结论是，程序上的不对称确实存在，但对这次 DEV 读数影响不大。
- G4：平均差 ≤ −0.03、CI 在 0 之下。结论是，用 CAL8 选 epoch 反而会让 CoMIR 在 DEV 上变差，说明 CAL8 在这里的预测力弱；保留冻结行并披露。
- 关于与早先宿主 CoMIR 的差距：如果三个最终 seed 的 D1 DEV32 success<42 都 ≥ 0.89（早先 R6 宿主 CoMIR 的范围是 0.891–0.9125），则 checkpoint 时点可以解释这一差距；否则差距仍不能由 checkpoint 时点解释，与下面的记录核查一起报告，不作因果结论。

**D1b 记录核查，不做新计算。** 只看代码和记录，不训练、不配准、不提交 GPU 作业。要回答的问题：

- 早先宿主 CoMIR（R6 REG-P2 comir_s0..s2；V9-G1 comir frac1 eps0 s901..s903）是否用了同样的 DEV core、同样的 CAL/DEV 扰动真值、同样的预处理与输入缓存、同样的训练和配准代码路径？
- 有哪些不同之处（seed、硬件、软件、非确定性）？差异落在 DEV 的哪些 core 上？

结果写入 `<bundle>/D1/COMIR_GAP_RECORDS.json` 和同目录的 `COMIR_GAP_RECORDS_CN.md`。这两个文件在本预登记之后生成；本文件提交后不再改动。

## 4 D2：MIND_2D_warp_then_describe

**问题。** 冻结 MIND_2D_adapted 行的 success<42 随 |θ*| 下降：0–10° 为 0.804，10–20° 为 0.759，20–30° 为 0.577。这是不是因为移动图像只在它自己的坐标轴框架下描述一次（先描述后变换，describe-then-warp）？

**变体定义。** 只改一处：移动图的描述子改为先变换到参考框架再计算（每个网格角度一次，每个精确候选一次），而不是在变换之前计算一次。

- 参考图：与冻结行相同，即 BF 亮度参考按冻结 MINDConfig 计算 `mind2d`，每个 core 一次，在支撑位置 x 上采样。
- 网格阶段：对搜索的 21 个角度 θ，各自把移动 SHG 强度图（冻结 MIND 的输入）在宿主网格框 BOX2=[59,452] 上只按 R(θ) 旋转重采样，算一次描述子（`mind2d`，冻结配置）。该角度下的 508 个平移 t 都复用这一描述子，在 x + R(−θ)t 处采样。这与宿主 critic 的网格机制相同。
- 精确阶段：refinement 的每一步，以及最终的 F_hat 和 F_true，都把移动图按完整的 (θ, t) 在宿主精确框 BOX1=[110,401] 上重采样，再算描述子，再在 x 处采样。这一阶段可微，供 Adam refinement 使用。
- 不变的部分：MINDConfig 为 radius 1、sigma 0.8、offset 1、方差下限 1e-6，即同样的邻域、Gaussian patch、方差规则和最大值归一化；能量 F 为 4096 个位置 × 4 个通道上描述子平方差均值的负数。支撑、位置、扰动、搜索器、双线性采样约定、输入归一化和数据角色也全部不变。
- 运行硬件为 L40S，与冻结 MIND 行一致；冻结的 `run.runtime_environment` 不改。

**运行。** 只有一个确定性的、不需要训练的配置，没有 seed：

- (a) 用未改动的 session 重放冻结 MIND 的 CAL 读数，作为 QC；
- (b) 变体的 CAL8 × 10 读数；
- (c) 变体的 DEV32 × 10，以及 DEV24 和逐 core 指标。

**QC。**

- Q1 身份：冻结 MIND release、源码、单元 COMPLETE、固定 checkpoint step_0000 的身份。
- Q2 重放：冻结变体的 CAL 重放与冻结的 `CAL/step_0000.jsonl` 比较，预期完全相同；如不同则披露。
- Q3 覆盖：CAL 80 行，DEV 320 行，DEV24 240 行。
- Q4 角色：只读 CAL 和 DEV；不读 FIT，不读任何 held core。

**报告内容。**

1. 两个变体的整体 CAL、DEV32、DEV24：success42、success9、中位误差；
2. 两个变体按 |θ*| 分层的 DEV32 success42/success9/中位误差，CAL 和 DEV24 也同样分层；
3. DEV32 success<42 的逐 core 配对差（变体减冻结），附 bootstrap CI；
4. 失败配准（err ≥ 42）按 F_hat ≥ F_true（能量最优点错了）与 F_hat < F_true（搜索漏掉了更高的 F）分开计数；
5. 实测成本。

**解读网格。**

- G1：变体在 [20,30] 层的 DEV32 success<42 比冻结行高 ≥ 0.10，并且变体三层的极差不超过冻结行极差（0.804−0.577=0.227）的一半。结论是，旋转依赖主要来自在自身坐标轴框架下描述；先变换后描述能消除大部分。
- G2：[20,30] 层提高 ≥ 0.03，但达不到 G1。结论是，方向只解释了一部分趋势。
- G3：[20,30] 层变化 < 0.03。结论是，这一趋势不能用通道方向解释。
- G4：整体 DEV32 success<42 下降 ≥ 0.03、CI 在 0 之下。无论分层结果如何，都报告该变体整体更差。
- 无论哪种情况：主表行仍是 MIND_2D_adapted。变体是一个命名的、看过 DEV 之后才设的敏感性变体；它不是原三维 MIND 的复现，也不是 MIND-SSC。

## 5 D3：补充方法 GPU gate 的跨硬件运行

**问题。** 补充方法自己的 GPU gate 包括：CUDA 数值、完整 FIT32 trainer 的 checkpoint/optimizer/RNG 续跑、几何梯度、完整公共搜索。对于 MLP_VCS_perm8 和 InfoNCE_host[J_perm1]，这个 gate 在 RTX PRO 6000 和 H100 上能否通过？种子复现中有部分单元正是跑在这两种 GPU 上。

**代码。**

- 使用冻结流水线树 `ssl_pilot_v10_pipeline_20261019` @ `ecd8a95` 的 `scripts/v10_supplement/validate.py gate --family F`，不改动导入。已接受的 L40S gate 1047322 和 1047323 用的就是这份代码。
- 把流水线 validation `MANIFEST.json`（sha256 `8befdabb…`）逐字节复制到 `<bundle>/D3/validation/MANIFEST.json`。未改动的 gate 只会在它旁边的 `runs/<job>` 下写文件，不写别处。

**唯一改动。**

- 只在作业进程内，把 `run.runtime_environment` 中“gpu_name 必须是 NVIDIA L40S”这一条，换成：SLURM 分区属于 {RTX6000PRO, H100}，并且 gpu_name 分别以 `NVIDIA RTX PRO 6000` 或 `NVIDIA H100` 开头。做法与种子复现的 `hardware.py` 相同。
- 其余检查照旧，对照父协议：torch、cuda_runtime、cudnn、torch_threads，十项精度设置，以及 CUBLAS/TF32 覆盖变量。

**作业。** 两个方法 × 两种分区，共 4 个：MLP_VCS_perm8 与 InfoNCE_host[J_perm1]，分别在 RTX6000PRO 和 H100 上。

**判定。**

- 以 gate 自己的 PASS/FAIL 为准。
- 如果 CUDA 设备无法初始化（例如 node53 出现过的“busy or unavailable”），在 gate 开始之前记为 INFRA_CUDA_UNAVAILABLE，不算 gate FAIL。作业先打印 nvidia-smi，然后明确失败退出。

**对比。** 与同一方法已接受的 L40S gate 作描述性差值比较。两者 seed 都是 1117，都是第一个候选 lr 1e-4。比较的量：

- 第 1 步、第 2 步和续跑后第 2 步的 loss 与梯度范数；
- 几何梯度及其有限差分；
- 完整搜索的 F_hat、θ̂、t̂；
- 计时；
- 峰值显存。

**解读。**

- PASS：补充方法在实现层面的数值恒等式、续跑和搜索机制在该 GPU 型号上成立。这并不说明与 L40S 逐位或任务层面等价，也不能使混合硬件的复现 seed 可以互换。
- FAIL：该 GPU 型号上这两个方法的复现单元缺少这一 gate；照实报告，不把它们视为与 L40S 硬件等价。

## 6 作业、硬件与成本

公共 sbatch 参数：`--qos=normal --gres=gpu:1 --exclude=node51,node60 --nice=0`，作业名以 `v10d_` 开头。D1 和 D2 用 L40S，D3 用 RTX6000PRO 和 H100。

预计成本依据的实测：

- CoMIR CAL 作业 1047391 共分配 326.8 s，即 0.091 GPU-h。其中约 273 s 是 736 步训练，80 次 CAL 配准只占 45.1 s。
- CoMIR DEV 320 次配准为 177.2 s。
- MIND CAL 作业分配 31.0 s；MIND FINAL 作业分配 218.6 s，其中 CAL 43.8 s，DEV 168.7 s。
- 已接受的 gate 作业 1047322 和 1047323 批处理时间分别为 15.0 s 和 12.9 s。

| 诊断 | 作业 | 预计 | 保守上界 |
|---|---|---|---|
| D1 | 4 × L40S | 每个约 10–14 min，合计约 0.8 GPU-h | 按每组 checkpoint 0.091 GPU-h、每次 DEV 0.364 GPU-h 计：4.37 + 1.09 = 5.5 GPU-h |
| D2 | 1 × L40S | ≤ 约 0.3 GPU-h（变体单次配准至多约为冻结 MIND 的 3 倍；提交前用 CPU 微基准细化） | 同左 |
| D3 | 2 × RTX6000PRO，2 × H100 | 每个 1–3 min，合计约 0.1–0.2 GPU-h | 同左 |

合计预计约 1.2 GPU-h，上界约 6 GPU-h。

## 7 表述要求

**必须写明：**

- 看过 DEV 之后设计的敏感性诊断（post-DEV sensitivity）；
- 这是开发数据（DEV32/DEV24），不是 TEST，也不是确认；
- D2 是命名变体 MIND_2D_warp_then_describe，是一种适配，不是 MIND 或 MIND-SSC；
- D3 是在另一种 GPU 型号上运行的 gate，不是等价性结论。

**禁止：**

- 用 D1/D2 结果替换或重排任何冻结主表行；
- 由 D1/D2 得出方法优劣、强基线或新颖性结论；
- 说“CoMIR 的差距由硬件或 seed 造成”（记录只能给出相关性）；
- 说“RTX PRO 6000 / H100 与 L40S 数值等价”；
- 把 D1/D2/D3 的任何输出用于 TEST_REPEAT 冻结或其他方法的选择。

## 8 执行顺序与停止规则

1. 本预登记先单独提交，早于任何 D1/D2/D3 作业和任何真实数据试运行。
2. 在本分支上实现代码，并用合成输入做 CPU 测试。用真实 FIT/CAL 输入的 CPU 试运行只经 Slurm cpu-high 进行，不碰 DEV。
3. bundle 的 FREEZE.json 固定每个诊断源码文件的 sha256 和全部冻结输入；任何不符，作业都拒绝运行。
4. 由主会话审核后提交。D1 各单元、D2、D3 彼此独立。
5. 失败的单元连同失败记录一起报告，不盲目重提。重提时只有通过身份核对的已完成行才会被复用。
6. 分析脚本只写中性的 RESULTS；如何表述，等负责人决定。
