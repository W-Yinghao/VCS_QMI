# V10 MF-VCS 第一阶段（T00–T02）阶段报告

生成：`scripts/v10/phase1_report.py`，2026-10-17T02:33:14Z；分支 `v10/mf-vcs`，生成时 HEAD `11746dd`，结果提交 `11746dd`。全部数值、表格与 git 事实在生成时读自 §11 所列文件、包文件、git 对象与作业日志（另有一次只读 squeue）；手写的只有文字、决定选项与建议，以及明确标为“主会话陈述”的条目。

**范围**：第一阶段只做了预检、数学/宿主接入验证与计时。**没有测量任何配准精度、S 估计质量或训练后任务表现**；本报告不含性能、加速或新颖性结论，计时表只是成本输入。

**待 owner 决定 5 项（§6；每项列选项、证据与标为“建议”的默认值，本报告不作决定）**：① MF-M1 头停止规则；② TF32 策略；③ InfoNCE 配准能量；④ T04/T05 前尚缺的基线；⑤ 数据角色（TEST、CONFIRM、EPFL 激活）。决定 1–4 之前不提交 T04/T05 作业。

## 1 当前真实提交

| 项 | 值 |
|:---|:---|
| 分支 / 生成时 HEAD | `v10/mf-vcs` / `11746ddb914e1fb370215f13026fadb2dd4c571c`（2026-10-17T01:49:55Z）；测量代码、作业脚本、测试、v10/T00–T02 与 V9 路径无未提交改动；本生成器与本报告：`scripts/v10/phase1_report.py`（未跟踪），二者在此 HEAD 之后的同一提交中加入 |
| 结果提交 | `11746dd`：最后一个改动 v10/T00–T02、src/v10、scripts/v10、slurm/v10 或 tests/v10 的提交（不计本生成器与本报告） |
| 测量代码身份 | code hash `bf8942645f0167b0`（src/v10 + scripts/v10 已跟踪文件，定义同 src/v10/adapter.py code_hash）：由 git 对象在 a8f4c5d、4c6bbe9、11746dd 重算均得此值，等于结果作业 1043668、1043695 作业头的 code= 与 T01/T02 JSON 的 provenance；作业 pin（另含 slurm/v10）`1bcc4a5eb2f3cd2d` |
| 宿主与后端 | 宿主代码 hash `6c6f094684fbc50a`（V9-G1 g1.py + r6_reg_pilot.py + r6_reg_p2.py，只读导入）；自包提交 `f652f1e` 起 src/v9、scripts/v9、slurm/v9 的 git diff 为空；后端 src/v10/mf_vcs.py sha256 `cdf5df2e43bacb74`… = 包 core/mf_vcs.py |
| 包 | `v10/package_v10_server_v1_20261016`（`f652f1e` 原样提交）；包 MANIFEST_SHA256.json 所列 134 个文件全部一致 |
| 本报告提交的影响 | 生成器位于 scripts/v10，code hash 的定义覆盖该目录：提交后树的 code hash 变为 `30f03898aa5782b9`、pin 变为 `757839edb1057cd3`；其余测量代码不变（去掉生成器重算 = `bf8942645f0167b0`，等于结果）。生成时（2026-10-17T02:33:17Z）队列中 V10 作业 0 个，没有按旧 pin 排队的作业。提交后若在新 HEAD 重跑 `t01_validate.py merge`，会把这些 part 记为 provenance 不一致；核对时应以去掉生成器的 hash 为准 |

V10 提交（包提交起，git log；code hash 由 git 对象重算）：

| 提交 | UTC | code hash | 改动路径 | 说明（提交标题） |
|:---|:---|:---|:---|:---|
| `f652f1e` | 2026-10-16T19:55:25Z | —（无代码文件） | v10/package_v10_server_v1_20261016 | V10 package as received: VCS_V10_Server_Package_v1_20261016 (owner 2026-10-16… |
| `97119bd` | 2026-10-16T20:22:31Z | `09cee4daeab116a3` | scripts/v10, tests/v10 | V10 T00: metadata-only data-manifest builder (BF-SHG / EPFL from the R6 / V9 … |
| `d8364a5` | 2026-10-16T20:23:06Z | `09cee4daeab116a3` | v10/T00 | V10 T00 data manifest + exposure record (pass 1, before the preflight): 206 B… |
| `ac7eb7d` | 2026-10-16T20:23:46Z | `595122f5a195045c` | scripts/v10 | V10 T00 builder: the notes' preflight line also gives the declared group coun… |
| `78e019f` | 2026-10-16T20:24:59Z | `61d9476bb649cd54` | scripts/v10 | V10 T00 builder: P60's EPFL input described exactly (square centre crop resiz… |
| `0d52397` | 2026-10-16T20:25:19Z | `61d9476bb649cd54` | v10/T00 | V10 T00 preflight (package scripts/preflight.py, read-only, unmodified sha256… |
| `50cbfab` | 2026-10-16T20:25:46Z | `61d9476bb649cd54` | v10/T00 | V10 T00: host-adapter audit of the V9 registration host (read-only) — HOST_AD… |
| `e3b4618` | 2026-10-16T21:16:05Z | `d5222ad6c5e9a45a` | scripts/v10, slurm/v10, src/v10, tests/v10 | V10 phase-1 code: MF-VCS backend copy, real-host adapter on the V9-G1 host, T… |
| `b83cdc3` | 2026-10-16T22:37:53Z | `da1975a1da5bd7b1` | scripts/v10, slurm/v10, src/v10, tests/v10 | V10 phase 1: fixes from the three-reviewer audit of e3b4618 (M1 degeneracy, T… |
| `d89f018` | 2026-10-16T22:58:39Z | `00e2f06c99f415f1` | scripts/v10, tests/v10 | V10 T01: fix the M1 training-mode row (keyword collision) and characterise th… |
| `f0c661c` | 2026-10-16T23:11:38Z | `00e2f06c99f415f1` | v10/T01, v10/T02 | V10 T01 / T02 CPU-phase results (job 1043657 at d89f018, code hash 00e2f06c99… |
| `a8f4c5d` | 2026-10-16T23:16:22Z | `bf8942645f0167b0` | scripts/v10 | V10 T01: fp32 dense-MLP gate compares the blocked/checkpointed implementation… |
| `4c6bbe9` | 2026-10-16T23:35:49Z | `bf8942645f0167b0` | v10/T01, v10/T02 | V10 T01 / T02 CPU-phase results, job 1043668 at a8f4c5d (code hash bf8942645f… |
| `11746dd` | 2026-10-17T01:49:55Z | `bf8942645f0167b0` | v10/T01, v10/T02 | V10 T01 / T02 GPU-phase results, job 1043695 (L40S node50, code hash bf894264… |

作业（作业头、结束行与 part 行解析自 slurm 日志；T00 行解析自 PREFLIGHT_RUN.log）：

| 作业 | 内容 | 提交 / code hash | 节点 | 开始 – 结束（UTC） | 结果 | 日志 |
|:---|:---|:---|:---|:---|:---|:---|
| 1039458 | T00 预检（在已存在的 CPU 作业内运行，T00 审计记为 no-shell CPU；非 V10 提交） | `d8364a5` | nodecpu10 | 2026-10-16T20:23:26Z – 2026-10-16T20:23:27Z | exit 0；INVENTORY_WRITTEN_REQUIRES_HOST_REVIEW | v10/T00/PREFLIGHT_RUN.log |
| 1043653 | T01 CPU + T02 CPU 数值头 | `b83cdc3` / `da1975a1da5bd7b1` | nodecpu01，cpu-high | 2026-10-16T22:38:01Z – 2026-10-16T22:55:54Z | T01 5 个 part：adapter_cpu FAIL，其余 PASS；T02 head PASS；作业内合并 TORCH_VALIDATION FAIL、GEOMETRY NOT_RUN、host_adapter_verified false（4 个 CUDA part 当时待跑）；已作废（§7） | v10_t01_cpu_1043653.out |
| 1043657 | T01 CPU + T02 CPU 数值头 | `d89f018` / `00e2f06c99f415f1` | nodecpu01，cpu-high | 2026-10-16T22:58:43Z – 2026-10-16T23:11:03Z | T01 5 个 part：adapter_cpu FAIL，其余 PASS；T02 head PASS；作业内合并 TORCH_VALIDATION FAIL、GEOMETRY NOT_RUN、host_adapter_verified false（4 个 CUDA part 当时待跑）；已作废（§7） | v10_t01_cpu_1043657.out |
| 1043667 | T01 CUDA + T02（GPU） | pin 为该提交之前的代码 | — | 排队中取消，未启动 | 无日志；见 `a8f4c5d` 提交说明：“GPU job 1043667 (pending, pinned to the old code) was cancelled before it started;” | — |
| 1043668 | T01 CPU + T02 CPU 数值头 | `a8f4c5d` / `bf8942645f0167b0` | nodecpu01，cpu-high | 2026-10-16T23:16:25Z – 2026-10-16T23:35:01Z | T01 5 个 part：全部 PASS；T02 head PASS；作业内合并 TORCH_VALIDATION NOT_RUN、GEOMETRY NOT_RUN、host_adapter_verified false（4 个 CUDA part 当时待跑）；**当前结果** | v10_t01_cpu_1043668.out |
| 1043695 | T01 CUDA + T02 全部 + BUDGET | `4c6bbe9` / `bf8942645f0167b0` | node50，L40S，NVIDIA L40S, 46068 MiB | 2026-10-17T01:36:37Z – 2026-10-17T01:48:35Z | T01 4 个 part：全部 PASS；T02 head PASS、head_extra PASS、train PASS、register PASS；作业内合并 TORCH_VALIDATION PASS、GEOMETRY PASS、host_adapter_verified true；**当前结果** | v10_t01_t02_gpu_1043695.out |

## 2 已完成代码/验证

| 项 | 状态 | 计数 / 要点 | 证据 |
|:---|:---|:---|:---|
| T00 预检（包 scripts/preflight.py，未改动） | PASS | 状态 INVENTORY_WRITTEN_REQUIRES_HOST_REVIEW（只表示清单已写出）；errors 0，warnings 0 | v10/T00/PREFLIGHT.json、PREFLIGHT_RUN.log |
| T00 数据 manifest | PASS | 构建检查 43/43；683 行，sha256 cc309cece34b574b… = MANIFEST_SHA256.txt = 预检记录 | v10/T00/V10_DATA_MANIFEST.csv、MANIFEST_BUILD_SUMMARY.json、EXPOSURE_NOTES.md |
| T00 宿主源码映射 | PASS（已写出） | 映射状态 WRITTEN_REQUIRES_REVIEW；与包假设的差异 9 项（D1–D9）登记未改；数值接入见 T01 | v10/T00/HOST_ADAPTER_AUDIT.md、host_map.json |
| T01 verify_torch（cpu） | PASS | 4 项：必需 3 全 PASS；诊断 1，其中 FAIL 1 | 作业 1043668，2 s |
| T01 verify_torch（cuda） | PASS | 4 项：必需 3 全 PASS；诊断 1，其中 FAIL 1 | 作业 1043695，4 s |
| T01 core_tol（cpu） | PASS | 654 项：必需 654 全 PASS；诊断 0 | 作业 1043668，111 s |
| T01 core_tol（cuda） | PASS | 654 项：必需 654 全 PASS；诊断 0 | 作业 1043695，9 s |
| T01 adapter（cpu） | PASS | 285 项：必需 269 全 PASS；诊断 16，其中 FAIL 10 | 作业 1043668，360 s |
| T01 adapter（cuda） | PASS | 285 项：必需 269 全 PASS；诊断 16，其中 FAIL 8 | 作业 1043695，345 s |
| T01 host_replay（cpu） | PASS | 8 项：必需 7 全 PASS；诊断 1 | 作业 1043668，15 s |
| T01 geometry（cpu） | PASS | 90 项：必需 86 全 PASS；诊断 4 | 作业 1043668，208 s |
| T01 geometry（cuda） | PASS | 94 项：必需 86 全 PASS；诊断 8 | 作业 1043695，5 s |
| T01 合并 | PASS | TORCH_VALIDATION PASS、GEOMETRY_GRADIENT_CHECK PASS、host_adapter_verified true；provenance 不一致 0、pending 0 | v10/T01/TORCH_VALIDATION.json、GEOMETRY_GRADIENT_CHECK.json |
| T02 head（cpu） | PASS | fp32 + fp64，各 12 格 × 5 法；OOM 0 | v10/T02/T02_TIMING_cpu.json |
| T02 head_extra（cpu） | NOT_RUN | 按设计只在 GPU 作业中运行 | v10/T02/T02_TIMING_cpu.json |
| T02 train（cpu） | NOT_RUN | 按设计只在 GPU 作业中运行 | v10/T02/T02_TIMING_cpu.json |
| T02 register（cpu） | NOT_RUN | 按设计只在 GPU 作业中运行 | v10/T02/T02_TIMING_cpu.json |
| T02 head（cuda） | PASS | fp32 + fp64，各 12 格 × 5 法；OOM 0 | v10/T02/T02_TIMING_cuda.json |
| T02 head_extra（cuda） | PASS | 分块保留图 + 排列条件方差；包抽样重放核对 12/12 格，不一致 0 | v10/T02/T02_TIMING_cuda.json |
| T02 train（cuda） | PASS | 18 个配置，encoder 在内；完成 18 | v10/T02/T02_TIMING_cuda.json |
| T02 register（cuda） | PASS | 5 个能量，完成 5 | v10/T02/T02_TIMING_cuda.json |
| BUDGET | MEASURED_PROJECTION | 已测 4 族：MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1、MLP_VCS_perm1；部分测得：MLP_VCS_perm8（仅训练步） | v10/T02/BUDGET.json |
| UNIT_LEDGER（= 包 EXPERIMENT_UNITS.csv 的单元） | PASS（T02 格）/ NOT_RUN（其余） | 114 个单元：T02 计算格 60 个 completed（fp32 + fp64 参考）；T04/T05 54 个 NOT_RUN | v10/T02/UNIT_LEDGER.csv |
| 单元测试 tests/v10（pytest，作业内记录、不门控） | PASS | 1043653 64 passed；1043657 66 passed；1043668 66 passed；1043695 66 passed | 作业日志 |
| 包作者本地检查 | NOT_RUN（服务器未重跑） | 包记录：NumPy 数学 47、synthetic 138、语法与 manifest 15 项通过 | v10/package_v10_server_v1_20261016/PACKAGE_STATUS.json |
| T03 受控真值 / T04 完整训练 / T05 完整配准 | NOT_RUN | 第一阶段不启动（§5 为提议） | v10/T02/UNIT_LEDGER.csv |

T01 合计 2078 项（9 个 part）：**必需 2031 项全部 PASS**（BLOCKED 0、NOT_RUN 0）；诊断 47 项（不门控），其中 20 项 FAIL，按类别如下。

| 类别 | part | FAIL 行 | 关键数值 | 说明 |
|:---|:---|:---|:---|:---|
| 包脚本误差 vs 设计 atol | verify_torch_cpu、verify_torch_cuda | 2 | 每次 3/150 行高于设计 atol 1e-10；最大 9.88e-9，包自身限 1e-8 | 包脚本不记录参考量级；超出的行都是 solver gap 行（在包自身 1e-8 限内）；core_tol 以参考量级重跑这些恒等式，全部 PASS |
| 可检测性（错误规则的效应是否超过等式界） | adapter_cpu 7、adapter_cuda 7 | 14 | effect/界 = 4.70e-5–94.4（门槛 100） | host_init（J 约 1e-6）与 fp32（界 ≥ 1e-5）下效应按构造不足；这类检查只在 fp64 sharpened_x10 状态门控，那 6 行均为必需且 PASS |
| fp32 全图 dense-MLP 累加误差 | adapter_cpu | 2 | blocked_checkpointed_vs_full_grads_float32 2.94e-5；full_float32_grads_vs_float64_value_same_inputs 2.91e-5 | 两侧都是 fp32，全图对 262144 对求和；adapter_cuda 同名 2 行 PASS、PASS（2.98e-8、4.14e-8）；门控行（分块 fp32 vs fp64）adapter_cpu PASS 2.94e-7；adapter_cuda PASS 1.64e-8（§8 主会话决定） |
| M1 协议容差从 W = 0 起步 | adapter_cpu、adapter_cuda | 2 | ‖Δ‖₁ = 3.1758e-5 / 3.1760e-5 < 1e-4；迭代 0，\|W\|max 0，encoder 梯度 0 | 协议设置的诊断，见 §6 决定 1 |

T01 覆盖（必需 / 诊断，PASS/总数；按检查名分组，每项检查只计一次）：

| part | 内容 | CPU 必需 | CPU 诊断 | CUDA 必需 | CUDA 诊断 |
|:---|:---|---:|---:|---:|---:|
| verify_torch | 包 core/verify_torch.py（未改动）；后端与脚本 sha256 = 包清单 | 3/3 | 0/1 | 3/3 | 0/1 |
| core_tol | 设计容差核心数学：V/U/独立池矩 vs dense 的值与梯度、头部值/梯度/HVP、盒 QP 对独立 oracle、NaN/inf/不可行 W/非 simplex 拒绝 | 654/654 | — | 654/654 | — |
| adapter | 数据门、宿主/manifest 固定、V9 写入 tripwire、宿主 batch RNG | 10/10 | — | 10/10 | — |
| adapter | 真实 FIT batch：矩 vs dense 的值与 A/B/W、encoder + feature map 梯度（含分块 checkpoint） | 36/36 | 2/2 | 36/36 | 2/2 |
| adapter | 全部排列的均值 = V（值与梯度） | 8/8 | — | 8/8 | — |
| adapter | 场景权重、不等 n、n = 1 的 V | 26/26 | 2/6 | 26/26 | 2/6 |
| adapter | Q 模式：U 与独立池 | 21/21 | 0/3 | 21/21 | 0/3 |
| adapter | M1 头：严格解、盒约束与 detach、外层梯度 = 固定 W 偏导、训练模式记录 gap、上限强制步 | 30/30 | 0/1 | 30/30 | 0/1 |
| adapter | 宿主 MLP critic：perm1 = 宿主公式、dense = 排列均值、分块 checkpoint、perm8 | 14/14 | 0/2 | 14/14 | 2/2 |
| adapter | 配准侧 warp 链：宿主位置与排列、批量能量 = 包矩 / perm1、几何梯度 moments vs dense | 68/68 | — | 68/68 | — |
| adapter | 测试时固定 W：状态 hash 不变、eval 与冻结、几何梯度有限非零、拒绝可训练模型 | 16/16 | — | 16/16 | — |
| adapter | checkpoint 往返：(encoder, feature maps, W) 状态 hash、能量与几何梯度 | 20/20 | — | 20/20 | — |
| adapter | 一步训练：MF_perm1 与 M0 同批同场景、W 投影可行、encoder 与 feature map 实际更新 | 16/16 | 2/2 | 16/16 | 2/2 |
| adapter | weight decay 分组（W 为 0） | 4/4 | — | 4/4 | — |
| host_replay | 宿主 replay：未改动的 G.cmd_train vs V10 MLP_perm1 trainer | 7/7 | 1/1 | — | — |
| geometry | 几何：采样点与插值折点的距离、零跨越 | 14/14 | — | 14/14 | — |
| geometry | 几何：dense vs moments 值与几何梯度、P/Q 各项梯度非零 | 24/24 | — | 24/24 | — |
| geometry | 几何：fp64 中心差分 vs autograd（P/Q 各项；moments / dense / perm1；步长 1.0e-3、2.5e-4） | 48/48 | — | 48/48 | — |
| geometry | 几何：fp32 vs fp64（默认 TF32 卷积与 IEEE 卷积，诊断） | — | 4/4 | — | 8/8 |

代码清单（当前树）：

| 文件 | 行 | sha256[:16] | 作用 |
|:---|---:|:---|:---|
| `scripts/v10/phase1_report.py` | 1542 | `b345dbacc2a6c38d` | 本报告生成器（不参与测量） |
| `scripts/v10/t00_build_manifest.py` | 751 | `7070b3ff7a7250d6` | T00 manifest 构建（只读元数据） |
| `scripts/v10/t01_host_replay.py` | 56 | `82de1d3fb6fe54f3` | T01 宿主 replay |
| `scripts/v10/t01_validate.py` | 1336 | `ddaf05fed570e0cd` | T01 验证与合并 |
| `scripts/v10/t02_timing.py` | 711 | `1a51f69bd8f98218` | T02 计时、BUDGET、UNIT_LEDGER |
| `slurm/v10/submit.sh` | 27 | `29682e6eede8c2ba` | 作业脚本 |
| `slurm/v10/t01_cpu.sbatch` | 30 | `536550ca24c7e74d` | 作业脚本 |
| `slurm/v10/t01_t02_gpu.sbatch` | 34 | `256f6b800f4e358b` | 作业脚本 |
| `slurm/v10/v10_header.sh` | 28 | `4adaf3745d088962` | 作业脚本 |
| `src/v10/__init__.py` | 6 | `1d84649b206398c3` | 包初始化 |
| `src/v10/adapter.py` | 863 | `70096b5662c21531` | 真实宿主 adapter（V9-G1 只读导入） |
| `src/v10/mf_vcs.py` | 540 | `cdf5df2e43bacb74` | 包后端原样（sha256 = 包） |
| `tests/v10/test_mf_vcs_core.py` | 234 | `f174fdd97290cf8f` | 单元测试 |
| `tests/v10/test_t00_manifest.py` | 116 | `f512e4d18a477bbe` | 单元测试 |
| `tests/v10/test_v10_adapter.py` | 544 | `47abc84cfeb58e0b` | 单元测试 |

## 3 仍缺的输入

1. **owner 决定 1–5**（§6）。决定 1、2 影响所有已测族的单元；决定 3、4 决定哪些基线能进入同一比较；决定 5 决定 TEST 与 EPFL 的使用。
2. **尚未实现或未接入 V10 的方法**（BUDGET `not_measured_in_T02`）：InfoNCE_host、CoMIR_native、RuLSIF_kernel、MI_native / NMI_native / MIND_2D_adapted、MLP_VCS_perm8 registration energy（逐项现状见 §6.4）。V10 trainer 只接受 MF_perm1, MF_M0, MF_M1, MLP_perm1, MLP_perm8, MLP_dense（src/v10/adapter.py），InfoNCE、CoMIR、核方法都未接入。
3. **EPFL**：协议激活规则为 `BF_SHG_primary_EPFL_after_manifest_and_alignment_audit`；V10 adapter 的数据门只开放 BF_SHG 行，EPFL 宿主路径（V9-G2 / REG-P3）尚未接入或审计；BUDGET 中的 EPFL 成本是代理值。
4. **T03 代码**：包只提供离散真值生成器与检查脚本（synthetic/README.md：不含神经训练器）；V10 的 T03 训练/CAL/评估脚本尚未编写。
5. **冻结的 V10 T04/T05 协议**：包协议状态为 `designed_not_launched`，`execution.full_training_budget_seconds` = null，`independent_confirmation_source` = null；BUDGET 已给出实测投影，但单元的冻结协议（种子、CAL 规则、停止规则、TF32、能量定义）要在决定 1–5 后写定。
6. **CONFIRM 数据**：T00 结论为 `no CONFIRM source in this round`（可登记组 0）。

## 4 实际测量表

### 4.1 数值头时间（L40S，fp32，中位 ms）

设备 NVIDIA L40S（node50，作业 1043695），torch 2.9.0+cu128；包 `core/benchmark_torch.py --full`（seed 10101）：每格同一组随机 simplex 特征 A/B 与 W、单场景 V 目标；warmup 5、重复 30，计时 = 前向 + autograd.grad(A, B, W)，每次调用前后 CUDA 同步；**不含 encoder**。分块 dense 每块 128 行：“ckpt 重算”= 非重入 checkpoint、反向重算；“保留图”= 同样分块但保留各块反向图（head_extra，按包的抽样顺序重放核对 12/12 格）。

| N | K | perm1 | perm8 | dense | 分块 dense（ckpt 重算） | 分块 dense（保留图） | moments |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 0.886 | 0.901 | 0.984 | 1.52 | 0.996 | 1.38 |
| 128 | 16 | 0.886 | 0.897 | 0.984 | 1.53 | 0.995 | 1.38 |
| 128 | 32 | 0.883 | 0.895 | 0.985 | 1.52 | 0.995 | 1.39 |
| 512 | 8 | 0.888 | 0.915 | 1.00 | 4.15 | 2.13 | 1.38 |
| 512 | 16 | 0.886 | 0.920 | 0.999 | 4.15 | 2.13 | 1.38 |
| 512 | 32 | 0.889 | 0.916 | 1.00 | 4.14 | 2.14 | 1.39 |
| 2048 | 8 | 0.890 | 0.928 | 1.01 | 14.8 | 6.66 | 1.39 |
| 2048 | 16 | 0.895 | 0.927 | 1.01 | 14.7 | 6.63 | 1.39 |
| 2048 | 32 | 0.894 | 0.927 | 1.01 | 14.7 | 6.65 | 1.40 |
| 8192 | 8 | 0.917 | 0.927 | 6.45 | 56.9 | 24.7 | 1.40 |
| 8192 | 16 | 0.904 | 0.917 | 6.43 | 56.8 | 24.7 | 1.40 |
| 8192 | 32 | 0.907 | 0.921 | 6.38 | 56.9 | 24.7 | 1.40 |

### 4.2 数值头峰值显存（L40S，fp32，MiB）

CUDA max_memory_allocated，**含进程基线**：本 run 各格调用前的基线 allocated 为 64.0–66.5 MiB（原始 benchmark JSON，sha256 与 T02_TIMING_cuda.json 的记录一致；保留图一列由 T02 脚本单独测，峰值同样含基线）。数值头与 head_extra 共 132 行，非 completed 0 行，OOM 0 行。

| N | K | perm1 | perm8 | dense | 分块 dense（ckpt 重算） | 分块 dense（保留图） | moments |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 64.0 | 64.1 | 64.3 | 64.3 | 64.3 | 64.0 |
| 128 | 16 | 64.1 | 64.2 | 64.3 | 64.3 | 64.3 | 64.1 |
| 128 | 32 | 64.1 | 64.5 | 64.3 | 64.3 | 64.4 | 64.1 |
| 512 | 8 | 64.1 | 64.5 | 68.1 | 65.1 | 65.9 | 64.1 |
| 512 | 16 | 64.3 | 65.0 | 68.2 | 65.2 | 66.0 | 64.3 |
| 512 | 32 | 64.5 | 65.9 | 68.3 | 65.4 | 66.2 | 64.5 |
| 2048 | 8 | 64.6 | 66.0 | 128 | 68.5 | 83.4 | 64.6 |
| 2048 | 16 | 65.0 | 67.8 | 129 | 68.8 | 83.9 | 65.0 |
| 2048 | 32 | 65.9 | 71.5 | 129 | 69.4 | 84.7 | 65.9 |
| 8192 | 8 | 66.4 | 72.0 | 1090 | 81.8 | 334 | 66.3 |
| 8192 | 16 | 68.1 | 79.3 | 1091 | 83.0 | 336 | 68.0 |
| 8192 | 32 | 71.6 | 93.8 | 1093 | 85.6 | 339 | 71.5 |

### 4.3 fp32 与 fp64（L40S，数值头，中位 ms）

| N | K | moments fp32 | moments fp64 | dense fp32 | dense fp64 | 分块(ckpt) fp32 | 分块(ckpt) fp64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 1.38 | 1.41 | 0.984 | 0.998 | 1.52 | 1.53 |
| 128 | 16 | 1.38 | 1.40 | 0.984 | 0.997 | 1.53 | 1.51 |
| 128 | 32 | 1.39 | 1.40 | 0.985 | 0.993 | 1.52 | 1.52 |
| 512 | 8 | 1.38 | 1.44 | 1.00 | 1.06 | 4.15 | 4.19 |
| 512 | 16 | 1.38 | 1.44 | 0.999 | 1.06 | 4.15 | 4.21 |
| 512 | 32 | 1.39 | 1.42 | 1.00 | 1.03 | 4.14 | 4.17 |
| 2048 | 8 | 1.39 | 1.87 | 1.01 | 1.95 | 14.8 | 15.4 |
| 2048 | 16 | 1.39 | 1.87 | 1.01 | 1.96 | 14.7 | 15.4 |
| 2048 | 32 | 1.40 | 1.41 | 1.01 | 1.26 | 14.7 | 15.1 |
| 8192 | 8 | 1.40 | 1.72 | 6.45 | 19.3 | 56.9 | 57.6 |
| 8192 | 16 | 1.40 | 1.43 | 6.43 | 19.5 | 56.8 | 57.5 |
| 8192 | 32 | 1.40 | 1.43 | 6.38 | 21.0 | 56.9 | 57.5 |

精确 V 的其他实现（dense、分块 ckpt；fp32 另有保留图）相对 moments 的最大 |ΔJ|：fp32 9.31e-9、fp64 3.47e-17；A/B/W 梯度最大绝对差：fp32 7.45e-9、fp64 5.55e-17。包 benchmark 的正确性门限 fp32 3e-5、fp64 3e-11，超限格记 correctness_failed 且不计时（本表 0 格）。perm1/perm8 与 V 的差是抽样偏差（4.5）。

### 4.4 数值头时间（CPU，中位 ms）

nodecpu01（cpu-high，作业 1043668，torch 线程 16），同一包 benchmark；CPU 不测显存。† = 该格中位数比同方法、同 dtype、N 与 K 都不更小的另一格高 10 倍以上（违反成本随规模单调的预期）：CPU 全部 120 行中 26 行（本表只列 fp32 5 法与 fp64 2 法），L40S 0/132 行；原因未诊断。CPU 数字不进入 BUDGET（BUDGET 的来源是 T02_TIMING_cuda.json）。

| N | K | perm1 fp32 | perm8 fp32 | dense fp32 | 分块(ckpt) fp32 | moments fp32 | moments fp64 | dense fp64 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 72.0† | 26.3† | 105† | 130† | 104† | 106† | 102† |
| 128 | 16 | 70.0† | 76.0† | 104† | 0.842 | 0.506 | 0.513 | 104† |
| 128 | 32 | 0.467 | 0.723 | 0.564 | 0.877 | 0.624 | 0.644 | 0.646 |
| 512 | 8 | 0.488 | 1.11 | 2.02 | 3.41 | 0.563 | 108† | 146† |
| 512 | 16 | 0.475 | 0.838 | 1.94 | 3.47 | 0.553 | 0.613 | 145† |
| 512 | 32 | 0.613 | 0.974 | 2.12 | 3.68 | 0.752 | 178† | 2.93 |
| 2048 | 8 | 0.732 | 1.71 | 24.0 | 24.2 | 0.769 | 0.727 | 28.7 |
| 2048 | 16 | 0.822 | 1.44 | 23.5 | 22.5 | 0.760 | 0.946 | 28.1 |
| 2048 | 32 | 1.01 | 1.51 | 21.0 | 22.7 | 1.03 | 1.36 | 32.2 |
| 8192 | 8 | 1.99 | 3.25 | 387 | 251 | 1.61 | 1.28 | 364 |
| 8192 | 16 | 1.33 | 2.99 | 346 | 227 | 1.26 | 1.69 | 357 |
| 8192 | 32 | 1.84 | 4.05 | 354 | 235 | 1.93 | 2.82 | 381 |

### 4.5 排列的条件抽样偏差（L40S，fp32）

固定 4.1 的 A/B/W，独立重抽普通排列（含固定点）：perm1 256 次、perm8 64 次；sd 为 J 的条件标准差（同一组特征上的抽样偏差，不是独立样本方差）；W 梯度相对误差 = 前 32 次抽样的 ‖∇W J_perm − ∇W J_V‖ / ‖∇W J_V‖ 均值。

| N | K | J_V | sd perm1 | sd perm8 | W 梯度相对误差 perm1 | perm8 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | -8.502e-3 | 4.58e-3 | 1.50e-3 | 0.733 | 0.249 |
| 128 | 16 | -2.078e-3 | 2.72e-3 | 9.73e-4 | 0.940 | 0.333 |
| 128 | 32 | -1.355e-3 | 1.57e-3 | 5.47e-4 | 0.921 | 0.323 |
| 512 | 8 | -2.566e-2 | 2.09e-3 | 7.69e-4 | 0.169 | 0.0581 |
| 512 | 16 | -2.824e-3 | 1.40e-3 | 5.50e-4 | 0.504 | 0.178 |
| 512 | 32 | -6.549e-4 | 8.46e-4 | 2.77e-4 | 0.833 | 0.297 |
| 2048 | 8 | -9.988e-3 | 1.22e-3 | 4.65e-4 | 0.157 | 0.0570 |
| 2048 | 16 | -3.382e-3 | 6.55e-4 | 2.46e-4 | 0.701 | 0.244 |
| 2048 | 32 | -1.111e-3 | 4.01e-4 | 1.19e-4 | 0.500 | 0.175 |
| 8192 | 8 | -1.398e-2 | 5.26e-4 | 2.12e-4 | 0.0841 | 0.0281 |
| 8192 | 16 | -6.601e-3 | 3.15e-4 | 1.02e-4 | 0.102 | 0.0356 |
| 8192 | 32 | -1.248e-3 | 1.98e-4 | 7.73e-5 | 0.421 | 0.149 |

### 4.6 含 encoder 的训练步（L40S，fp32，ms）

宿主 batch：16 个 FIT crop × 128 位置，宿主 encoder，AdamW + clip 1.0（CSV notes）；warmup 10、计时 50 步（中位、p90）；阶段时间（目标、反向）来自另外 10 步的同步打点。MF 的 W-box/排列检查只在首步（另有 2 行每步检查）。M1 的 capbound 行 = tolerance 1e-300、require_converged False，迭代预算每批用满；`MF_M1_K16_it30_tol1e-04` 行只记录协议设置，不进 BUDGET（§6 决定 1）。

| 配置 | 中位 | p90 | 峰值 MiB | 目标 | 反向 | M1 迭代 | M1 gap 中位 | M1 起始 gap 中位 | HVP 次数 | 收敛比例 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MF_perm1_K16` | 56.0 | 56.3 | 921 | 3.85 | 29.8 | — | — | — | — | — |
| `MF_M0_K16` | 64.3 | 64.4 | 921 | 6.90 | 34.7 | — | — | — | — | — |
| `MF_M1_K16_it30_capbound` | 367 | 369 | 921 | 311 | 32.7 | 30 | 1.39e-2 | 7.45e-2 | 92 | 0 |
| `MF_M1_K16_it10_capbound` | 172 | 173 | 921 | 116 | 32.9 | 10 | 2.82e-2 | 6.24e-2 | 32 | 0 |
| `MF_M1_K16_it100_capbound` | 1052 | 1056 | 921 | 999 | 32.8 | 100 | 4.37e-3 | 8.58e-2 | 302 | 0 |
| `MF_M1_K16_it30_tol1e-04`（记录行） | 367 | 369 | 921 | 311 | 32.8 | 30 | 1.54e-2 | 8.26e-2 | 92 | 0 |
| `MF_perm1_K8` | 56.1 | 56.4 | 921 | 3.84 | 29.8 | — | — | — | — | — |
| `MF_M0_K8` | 64.3 | 64.5 | 921 | 6.88 | 34.8 | — | — | — | — | — |
| `MF_M1_K8_it30_capbound` | 366 | 367 | 921 | 311 | 32.8 | 30 | 9.25e-3 | 6.95e-2 | 92 | 0 |
| `MF_perm1_K32` | 56.0 | 56.3 | 922 | 3.83 | 29.8 | — | — | — | — | — |
| `MF_M0_K32` | 64.5 | 64.6 | 922 | 6.93 | 34.9 | — | — | — | — | — |
| `MF_M1_K32_it30_capbound` | 372 | 373 | 922 | 316 | 32.9 | 30 | 1.43e-2 | 7.69e-2 | 92 | 0 |
| `MF_perm1_K16_checks_every_step` | 57.6 | 57.8 | 921 | 5.29 | 29.9 | — | — | — | — | — |
| `MF_M0_K16_checks_every_step` | 64.4 | 64.6 | 921 | 6.94 | 34.9 | — | — | — | — | — |
| `MLP_perm1` | 39.3 | 39.7 | 921 | 0.973 | 23.6 | — | — | — | — | — |
| `MLP_perm8` | 40.0 | 40.1 | 921 | 1.46 | 23.6 | — | — | — | — | — |
| `MLP_dense` | 39.4 | 39.6 | 1148 | 2.09 | 23.0 | — | — | — | — | — |
| `MLP_dense_block32_ckpt` | 44.3 | 44.5 | 922 | 2.77 | 26.5 | — | — | — | — | — |

### 4.7 完整配准（L40S，fp32，每次配准秒数）

宿主共享搜索器不变（网格候选 + 网格最优 3 个起点各 100 步 Adam 精修，Q.optimise_rigid；host_map.json），**未训练的固定模型，只计时**（不计算误差或真值量）；3 个 DEV core × 2 个扰动 = 每能量计时 6 次（另 1 次 warmup 不计）。参考侧 encoder 一次：0.036 s（含读取参考图）。

| 能量 | n | 中位 | 范围 | 网格阶段 | 精修 + 终值 | 精确能量调用 | 网格候选 | 位置 N | 峰值 MiB |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MF_moments_K16` | 6 | 1.024 | 1.023–1.030 | 0.155 | 0.869 | 303 | 10668 | 4096 | 238 |
| `MF_perm1_K16` | 6 | 0.942 | 0.939–0.948 | 0.107 | 0.835 | 303 | 10668 | 4096 | 238 |
| `MF_moments_K8` | 6 | 1.025 | 1.023–1.031 | 0.149 | 0.876 | 303 | 10668 | 4096 | 238 |
| `MF_moments_K32` | 6 | 1.052 | 1.044–1.053 | 0.172 | 0.879 | 303 | 10668 | 4096 | 266 |
| `MLP_perm1_host` | 6 | 1.574 | 1.574–1.576 | 0.607 | 0.967 | 303 | 10668 | 4096 | 329 |

### 4.8 V9 参考成本（只读，BUDGET.json `v9_reference_costs`）

| V9 单元 | 设备 / 作业 | 配准次数 | s/次 CAL | s/次 DEV | 训练 s | 训练峰值 GiB |
|:---|:---|---:|---:|---:|---:|---:|
| `V9-G1-vcs-frac1-eps0-s901` | RTX PRO 6000 Blackwell, job 1041939 | 1280 | 0.767 | 0.734 | 83.0 | 0.967 |
| `V9-G2-vcs-frac1-eps0-s901` | L40S, job 1042139 | 1940 | 1.477 | 1.479 | — | — |

## 5 下一批具体单元与预计资源（提议，未提交）

### 5.1 T03 受控真值（CPU / 小 GPU；T03 代码尚未编写）

- 数据：包生成器 `synthetic/src/v10_synthetic.py`（sha256 3c5d7d22e129eea7…，= 包清单）的默认参数：seed 20261016，two_block、diagonal_mixture，m ∈ {8, 32}，ρ ∈ {0, 0.25, 0.75}；每条件 FIT/CAL/EVAL = 4096/2048/8192 个独立基本观测，12 个条件共 172032 个。输出写 `outputs_v10/T03/`，再跑包的 `synthetic/tests/check_synthetic.py`（包示例套件上同一脚本 138 项 passed）。truth 只交给评估器。
- 方法（建议网格）：MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1 × K ∈ {8, 16, 32}，加 MLP_VCS_perm1、MLP_VCS_perm8（整数观测 one-hot 输入）；CAL 在调参 seed 1017 上按 CAL 经验 J 从 lr ∈ {1e-4, 3e-4, 1e-3} 中选，最终 seed 1001/1002/1003：12 条件 × 11 配置 → 最终训练 396 次、CAL 候选 396 次。M1 行等决定 1。
- 目标模式：包 §2.1 把 U 留作“受控 iid 估计实验”；建议 V（与配准一致）为主，U 作命名的敏感性行。其他估计器按包 §6.3 “MINE/DV/NWJ/SMILE/InfoNCE以各自原生量和共同下游任务比较，不用S真值给非S输出算RMSE”，不在本提议内。
- 指标（包 §6.3）：E_M(T−η)²（evaluator-only truth）、EVAL 经验 J 对 S 的 bias / variance / RMSE（跨 seed）、同一基础数据上的排列方差、M1 gap、总时间。
- 需先写：`scripts/v10/t03_synthetic.py`（训练只经 `load_samples` 读 FIT/CAL，评估器单独 `load_evaluation_truth`）、单元测试与 smoke。
- 资源：CPU（cpu-high）为主，**未实测**；第一个作业 = 生成 + 检查 + 1 个条件的 smoke 计时，再写预算。

条件（精确总体 S 读自包示例套件的 truth.json；S 只取决于 family、m、ρ）：

| 条件 | family | m | ρ | S |
|:---|:---|---:|---:|---:|
| `two_block__m008__rho0` | two_block | 8 | 0 | 0 |
| `two_block__m008__rho0p25` | two_block | 8 | 0.25 | 0.01587 |
| `two_block__m008__rho0p75` | two_block | 8 | 0.75 | 0.1636 |
| `two_block__m032__rho0` | two_block | 32 | 0 | 0 |
| `two_block__m032__rho0p25` | two_block | 32 | 0.25 | 0.01587 |
| `two_block__m032__rho0p75` | two_block | 32 | 0.75 | 0.1636 |
| `diagonal_mixture__m008__rho0` | diagonal_mixture | 8 | 0 | 0 |
| `diagonal_mixture__m008__rho0p25` | diagonal_mixture | 8 | 0.25 | 0.06667 |
| `diagonal_mixture__m008__rho0p75` | diagonal_mixture | 8 | 0.75 | 0.4345 |
| `diagonal_mixture__m032__rho0` | diagonal_mixture | 32 | 0 | 0 |
| `diagonal_mixture__m032__rho0p25` | diagonal_mixture | 32 | 0.25 | 0.1136 |
| `diagonal_mixture__m032__rho0p75` | diagonal_mixture | 32 | 0.75 | 0.5525 |

### 5.2 T04/T05 单元（UNIT_LEDGER.csv；GPU-h 取 BUDGET.json，L40S；全部 NOT_RUN）

每单元 = 3000 训练步 + 配准 1280 次（BF-SHG：12 checkpoint × 8 CAL × 10 + 32 DEV × 10）/ 1940 次（EPFL：12 × 10 CAL × 10 + 74 DEV × 10，代理）；M1 按 `MF_M1_K16_it30_capbound`（迭代上限用满，上界）计。已测合计 **29.78 GPU-h**（BF-SHG 6.18 + EPFL 代理 8.69 + BF CAL 调参 14.90）；“依赖决定”列为 §6 的编号。

| 方法 | 数据 | 阶段 | 单元 | 每单元 GPU-h | 小计 | 依赖决定 | 状态 |
|:---|:---|:---|:---|---:|---:|:---|:---|
| `MLP_VCS_perm1` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 0.59 | 1.78 | 2 | NOT_RUN |
| `MLP_VCS_perm8` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 未测（仅训练步） | — | 2、4 | NOT_RUN |
| `MF_VCS_perm1` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 0.38 | 1.14 | 2 | NOT_RUN |
| `MF_VCS_M0` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 0.42 | 1.25 | 2 | NOT_RUN |
| `MF_VCS_M1` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 0.67 | 2.01 | 1、2 | NOT_RUN |
| `InfoNCE_host` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 未测 | — | 2、3、4 | NOT_RUN |
| `CoMIR_native` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 未测 | — | 2、4 | NOT_RUN |
| `RuLSIF_kernel` | BF_SHG | T04_T05 | 3（seed 1001/1002/1003） | 未测 | — | 4 | NOT_RUN |
| `MLP_VCS_perm1` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 0.88 | 2.64 | 2、5 | NOT_RUN |
| `MLP_VCS_perm8` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 未测（仅训练步） | — | 2、4、5 | NOT_RUN |
| `MF_VCS_perm1` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 0.55 | 1.66 | 2、5 | NOT_RUN |
| `MF_VCS_M0` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 0.61 | 1.82 | 2、5 | NOT_RUN |
| `MF_VCS_M1` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 0.86 | 2.57 | 1、2、5 | NOT_RUN |
| `InfoNCE_host` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 未测 | — | 2、3、4、5 | NOT_RUN |
| `CoMIR_native` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 未测 | — | 2、4、5 | NOT_RUN |
| `RuLSIF_kernel` | EPFL_RGB_NIR | T04_T05 | 3（seed 1001/1002/1003） | 未测 | — | 4、5 | NOT_RUN |
| `MI_native` | BF_SHG | T05 | 1（非学习，无 seed） | 未测 | — | 4 | NOT_RUN |
| `NMI_native` | BF_SHG | T05 | 1（非学习，无 seed） | 未测 | — | 4 | NOT_RUN |
| `MIND_2D_adapted` | BF_SHG | T05 | 1（非学习，无 seed） | 未测 | — | 4 | NOT_RUN |
| `MI_native` | EPFL_RGB_NIR | T05 | 1（非学习，无 seed） | 未测 | — | 4、5 | NOT_RUN |
| `NMI_native` | EPFL_RGB_NIR | T05 | 1（非学习，无 seed） | 未测 | — | 4、5 | NOT_RUN |
| `MIND_2D_adapted` | EPFL_RGB_NIR | T05 | 1（非学习，无 seed） | 未测 | — | 4、5 | NOT_RUN |
| CAL 调参（已测族） | BF_SHG | T04_T05 | 4 族 × 9 候选（调参 seed 1017） | — | 14.90 | 1、2 | NOT_RUN |

### 5.3 建议的批次（均为提议；每批提交前再按队列规则核对）

- **批 0（不需 owner 决定）**：T03 数据生成 + 检查 + T03 代码与 smoke（CPU）；成本未测。
- **批 1（决定 1、2 之后）**：BF-SHG CAL 调参 14.90 GPU-h，再跑已测 4 族 × 3 seed（12 单元，6.18 GPU-h）；合计 21.09 GPU-h，V10 同时最多 2 个作业时墙钟约 10.5 h（不含排队）。
- **批 2（决定 5 / EPFL 审计之后）**：EPFL 已测 4 族（12 单元，8.69 GPU-h，代理值）；EPFL 的 CAL 调参未计入 BUDGET。
- **批 3（决定 3、4 与实现之后）**：其余 30 个单元（MLP_VCS_perm8、InfoNCE_host、CoMIR_native、RuLSIF_kernel、MI_native、NMI_native、MIND_2D_adapted）；先做 T01 式核对与 T02 式计时再定 GPU-h。
- M1 迭代上限的成本（BF-SHG 每单元 GPU-h，按 BUDGET 同一公式由 T02 步时推算；BUDGET 只含 `MF_M1_K16_it30_capbound`）：`MF_M1_K16_it10_capbound` 0.51；`MF_M1_K16_it30_capbound` 0.67；`MF_M1_K16_it100_capbound` 1.24。包把 max_iter ∈ {10, 30, 100} 也算作调参，BUDGET 的 CAL 调参未含此维度。

## 6 待 owner 决定（OPEN DECISIONS；本报告不作决定，“建议”仅为默认提议）

### 6.1 决定 1：MF-M1 头的停止规则（影响全部 MF_VCS_M1 单元与 T03 的 M1 行）

- **问题**：协议 `head.M1_gap_absolute_target` = 1e-4（绝对 FW gap；包 §7：“gap目标1e-4是起始工程容差，达到iteration cap时报告不足，不硬称精确包络梯度”），包的起点 W = 0，W = 0 处的 FW gap 恰为 ‖Δ‖₁（Δ = F 在 W = 0 处对 W 的梯度，即 P 与 Q 的一阶矩差 E_P[abᵀ] − E_Q[abᵀ]）。‖Δ‖₁ 低于目标时求解器第 0 次迭代即停：W 保持 0，外层目标恒为 0，feature map 与 encoder 的梯度恰为 0（只剩 AdamW 衰减）。
- **证据 T01**（真实宿主初始化，K16）：‖Δ‖₁ = 3.1758e-5（CPU）/ 3.1760e-5（CUDA）→ 迭代 0、|W|max 0、encoder 梯度范数 0、J 0（诊断行，记为 FAIL、不门控）。同一 batch（CPU）强制用满 30 次迭代：gap 3.176e-5 → 3.147e-5，|W|max 0.0143，converged false（必需行 PASS）。
- **证据 T02 协议记录行** `MF_M1_K16_it30_tol1e-04`（seed 10101，逐步文件 `train_steps_per_step.json` sha256 592d47e167cc0554…）：前 10 步（恰为全部 10 个 warmup 步，不在计时窗口内）全部 0 次迭代、W = 0、J = 0，各步 ‖Δ‖₁ ∈ [2.63e-5, 8.97e-5]；第 11 步（index 10）‖Δ‖₁ = 1.08e-4 超过目标后才开始迭代；之后计时 50 步中 W = 0 的 0 步、收敛 0 步。即起步时刻由 batch 间 ‖Δ‖₁ 的波动决定。
- **证据 T02 首个 batch**（W = 0 起步）的 ‖Δ‖₁：K8 3.70e-5、K16 7.73e-5、K32 6.00e-5，均 < 1e-4。
- **证据 T02 上限强制行**（tolerance 1e-300）：`MF_M1_K16_it30_capbound` 计时步 gap 中位 1.392e-2、最小 8.737e-3；`MF_M1_K16_it10_capbound` 计时步 gap 中位 2.824e-2、最小 1.206e-2；`MF_M1_K16_it100_capbound` 计时步 gap 中位 4.372e-3、最小 1.642e-3；`MF_M1_K8_it30_capbound` 计时步 gap 中位 9.250e-3、最小 3.042e-3；`MF_M1_K32_it30_capbound` 计时步 gap 中位 1.431e-2、最小 7.136e-3。计时步中没有一步的 gap 到达 1e-4：一旦起步，迭代上限总是先到。
- **选项**：
  - A 尺度相对目标：G(W) ≤ ε_rel·‖Δ‖₁（同一 batch、同一目标；W = 0 时相对 gap 为 1，至少迭代一次）。以 |F| 作分母不合适（W = 0 时 F = 0）。
  - B 非零起点 W：例如现有的 seeded uniform 初始化（目前只用于检查与计时）或取 M0 的 W。现实现已在 batch 间热启动（沿用上一 batch 的 W），问题只出在训练起点 W = 0；非零起点使 M1 与 M0 / perm1（均从 W = 0 起）的初始化不对称，须登记。
  - C 更小的绝对目标：仍随 K、温度与特征尺度变化；按上面的 gap，一旦起步，上限总是先到。
  - D 维持 1e-4：起步延迟取决于 batch 噪声（T02 记录行中为 10 步）。
- **建议默认（建议，未执行）**：A，ε_rel = 1e-2；迭代上限沿用包的 {10, 30, 100} 作为 CAL 维度；require_converged = False，逐批记录 gap 与 ‖Δ‖₁；W 起点保持 0。需改 `M1Config`（src/v10/adapter.py）并重跑 T01 的 M1 行（小）；BUDGET 的 M1 成本（迭代上限用满）在 A 下仍是上界。

### 6.2 决定 2：TF32 策略（全部 GPU 方法）

- **问题**：torch 2.9.0+cu128 默认 cuDNN fp32 卷积用 TF32（作业环境 `cudnn_conv_fp32_precision = tf32`），fp32 matmul = `highest`；宿主文件中没有任何精度开关（T00）。第一阶段全部 CUDA 计时与 T01 的 CUDA fp32 行都在此默认下运行。
- **证据**：几何梯度 fp32 对 fp64 的最大相对误差（T01 geometry 诊断行，同一 fp32 舍入权重；全部在 fp32 门限 atol 1e-5 + rtol 1e-4 内，但 host_init 行的参考梯度低于 atol，该门限在此不敏感）：

| 状态 / 变换 | 参考 max\|g\| | 参考 / atol | CPU | L40S 默认（TF32 卷积） | L40S IEEE 卷积 |
|:---|---:|---:|---:|---:|---:|
| host_init T1（+16.26°） | 9.33e-8 | 0.00933 | 3.60e-4 | 1.02e-2 | 1.08e-4 |
| host_init T2（-22.62°） | 6.45e-8 | 0.00645 | 3.72e-5 | 7.19e-4 | 3.11e-5 |
| sharpened_x10 T1（+16.26°） | 3.14e-5 | 3.14 | 1.82e-4 | 1.59e-3 | 2.01e-4 |
| sharpened_x10 T2（-22.62°） | 2.05e-5 | 2.05 | 1.75e-4 | 7.30e-4 | 1.81e-4 |

  V9 记录：学习型 critic 的配准不逐位可重复（同设备最大 1.7608 px RTX PRO 6000、0.9817 px L40S），原因未定位（host_map.json）。BUDGET 的全部 GPU 成本在 TF32 卷积下测得，IEEE 卷积下的成本未测。包 §6.1：“若大规模fp32消减错误超限，使用显式更高精度积累并把成本计入”；T01 的 fp32 门控行没有超限。
- **选项**：A 全部方法保持 torch 默认并写进协议（作业环境已逐次记录这些开关）；B 全部方法关闭 TF32（cuDNN conv fp32_precision = ieee）：几何梯度误差降到上表 IEEE 列的量级，需先重测 T02 train / register；C 训练用默认、测试与配准能量用 IEEE：训练与测试数值设置不同，须登记。
- **建议默认（建议，未执行）**：A——所有方法同一设置，与 V9 宿主和已测 BUDGET 一致；若 owner 更看重跨 GPU 型号的数值可重复性，选 B 并先补测成本。

### 6.3 决定 3：InfoNCE 的配准能量

- **问题**（T00 差异 D3）：V9 宿主用与 vcs 相同的 J(tanh f) perm1 能量为 logistic 与 InfoNCE 训练的 critic 配准（host_map.json D3：the same J(tanh f) perm1 energy as vcs (Q.cmd_register critic branch L329-348 passes kind 'critic' for vcs / logistic / infonce; j_critic L173-176)）；包的 InfoNCE_host 行把“原生目标 / 候选归一化”列为改变的因素。
- **证据**：V10 trainer 只接受 MF_perm1, MF_M0, MF_M1, MLP_perm1, MLP_perm8, MLP_dense，InfoNCE 训练未接入；宿主每次配准 N = 4096 个位置、网格 10668 个候选（T02_LONG_cuda.csv）；T02 未测任何 InfoNCE 能量。
- **选项**：A 宿主能量：InfoNCE 训练 + J(tanh f) perm1 能量（与 V9 定义相同，表中命名 InfoNCE_host[J]）；B 原生 InfoNCE 能量：每个候选需要 N×N 打分与候选归一化，是新定义，需实现、T01 式核对与 T02 式计时；C A 进主表，B 作单列命名变体。
- **建议默认（建议，未执行）**：A（命名明确，表注披露 D3）；B 只在实测成本后作为单列变体（即 C）。

### 6.4 决定 4：T04/T05 之前尚缺的基线

| 方法 | V10 现状 | BUDGET 记录 |
|:---|:---|:---|
| RuLSIF / 核（精确矩） | 未实现。V9 宿主的核行是 RFF 岭回归闭式解、每个 core 一个排列的负对，不是包 §5 的可分核精确矩 | kernel fit + registration not implemented in V10 phase 1 |
| MIND-2D 适配 | 未实现；src/v9/reg 中含 MIND 的文件 0 个 | non-learned; registration cost only (V9 measured MI / NMI) |
| MLP perm8 配准能量 | 未实现；训练侧已有（T02 步时 40.0 ms） | a perm8 registration energy is not implemented in phase 1 |
| CoMIR 原生 | V9 有完整流程，未接入 V10（seed、输出路径、数据门） | external system: own training / latent / registration costs (see V9 G1 / G2 results) |
| MI / NMI | V9 有，未接入 V10 | non-learned; registration cost only (V9 measured MI / NMI) |
| InfoNCE | 待决定 3；V10 trainer 未接入 | native InfoNCE energy undecided (T00 D3); host trains InfoNCE critics then scores J(tanh f) |

- **选项**：A 全部实现并核对后再启动任何 T04 单元；B 分批：已测 4 族（MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1、MLP_VCS_perm1）在决定 1–2 后先跑，各基线并行实现 → T01 式核对 → T02 式计时 → 在同一冻结协议下补跑；全部基线完成前不写任何比较结论；C 删减基线（包要求强基线与 same-data / same-total-cost 比较，不建议）。
- **建议默认（建议，未执行）**：B。

### 6.5 决定 5：数据角色（TEST、CONFIRM、EPFL）

- **证据（T00）**：BF-SHG 206 个 core 全部已被读过（72 个 train/validation 由 R6/V9 使用；134 个官方 test 由 V9-G1 做过一次确认读取，读取开始 2026-10-15T23:36:09Z）；EPFL 477 对全部被 REG-P3 审计读过（其中 60 对另被 P60 读过）；可登记为 CONFIRM 的组 0 个。TEST 在 V10 manifest 中为 HOLD（role DEV、inclusion exclude，134 行）。86 个 TMA 位置同时出现在 1B 与 2B 两张切片，是否同一组织块未记录。EPFL 保留 420 个窗口（对齐排除 33、尺寸排除 24），发布对齐的残差约 2 native px，严格阈值 2.25 native px 接近该限。V10 adapter 的数据门只开放 BF_SHG 行（src/v10/adapter.py DataGuard）。第一阶段实际读取：FIT 32 个 core、DEV 3 个 core（仅 T02 计时，未训练模型，不计算误差）、CAL 0、TEST 0。
  包 §4.3：“V9已读或参与方法设计的集合，在V10仍是开发/重复benchmark。”
- **选项**：A BF-SHG 只用 DEV（32 个 core）作开发主表；B 另把 134 个 TEST core 作为单独的重复基准块（V10 协议冻结后、新 manifest 版本；与 DEV 分表、不合并）；C 寻找新的 CONFIRM 数据（超出现有授权，需单独提案；T00 未评估服务器上的其他数据）。EPFL：在 V10 的 EPFL 宿主路径（V9-G2 / REG-P3）adapter 与对齐审计完成后激活，单独分表。
- **建议默认（建议，未执行）**：A 立即；B 在 V10 协议冻结后；EPFL 审计后激活；全部结论标为开发 / 重复基准结果；C 只作独立提案。

## 7 失败尝试（保留原因与原日志）

| 作业 | 提交 / code hash | 失败内容 | 原因 | 相关提交（git） | 日志与原始文件 |
|:---|:---|:---|:---|:---|:---|
| 1043653 | `b83cdc3` / `da1975a1da5bd7b1` | adapter_cpu FAIL（必需失败 5 行）：4 行 `real_m1_*:exception`：TypeError("Recorder.add() got multiple values for argument 'status'")；`MLP_dense_blocked_checkpointed_vs_full_grads_float32`：max_abs_err 2.94e-5 = 2.83 × fp32 界。其余 4 个 T01 part PASS | exception 行：检查代码 bug（关键字参数冲突），不是数值失败；fp32 行：两侧都是 fp32 实现，误差来自 fp32 全图实现（同一输入对 fp64：全图 2.91e-5、分块 2.94e-7，当前 T01 CPU） | `d89f018` V10 T01: fix the M1 training-mode row (keyword collision) and charact…；`f0c661c` V10 T01 / T02 CPU-phase results (job 1043657 at d89f018, code hash 00… | v10_t01_cpu_1043653.out（sha256 64b743a3312f…）；/projects/EEG-foundation-model/yinghao/outputs_v10/T01/runs/cpu_1043653、/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/cpu_1043653 |
| 1043657 | `d89f018` / `00e2f06c99f415f1` | adapter_cpu FAIL（必需失败 1 行）：`MLP_dense_blocked_checkpointed_vs_full_grads_float32`：max_abs_err 2.94e-5 = 2.83 × fp32 界。其余 4 个 T01 part PASS | fp32 行：两侧都是 fp32 实现，误差来自 fp32 全图实现（同一输入对 fp64：全图 2.91e-5、分块 2.94e-7，当前 T01 CPU） | `f0c661c` V10 T01 / T02 CPU-phase results (job 1043657 at d89f018, code hash 00…；`a8f4c5d` V10 T01: fp32 dense-MLP gate compares the blocked/checkpointed implem… | v10_t01_cpu_1043657.out（sha256 94c976f84385…）；/projects/EEG-foundation-model/yinghao/outputs_v10/T01/runs/cpu_1043657、/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/cpu_1043657 |
| 1043667 | — | 未运行（排队中取消） | pin 指向门控修改之前的代码 | `a8f4c5d`：“GPU job 1043667 (pending, pinned to the old code) was cancelled before it started;” | 无日志 |

这些作业的结果文件已被结果作业取代（结果提交链见 §1），日志与 part 文件原样保留；没有任何失败行被删除或改判，fp32 门控的修改见 §8。

## 8 已执行的主会话决定与已登记的实现选择

- **主会话决定** `a8f4c5d`（2026-10-16T23:16:22Z）：V10 T01: fp32 dense-MLP gate compares the blocked/checkpointed implementation with fp64 (main-session decision)
  - 内容：fp32 的 dense-MLP 门控改为“分块 / checkpoint 的 fp32 实现 vs 同一输入的 fp64 值”（必需）；fp32 全图实现与 fp32 分块 vs 全图之差保留为诊断；fp64 仍门控分块 vs 全图（adapter_cpu PASS、adapter_cuda PASS）。容差未改：fp64 atol 1e-10 / rtol 1e-8，fp32 atol 1e-5 / rtol 1e-4。
  - 证据（当前 T01）：CPU 上 fp32 全图（对 262144 对求和）对 fp64 的最大误差 2.91e-5，分块 fp32 为 2.94e-7；CUDA 上分别为 4.14e-8 与 1.64e-8。
  - 影响：作业 1043657 的唯一必需失败由此消除；当时排队的作业 1043667 因 pin 过期被取消（§7）。

已登记的实现选择（编码会话按包“常规实现选择可自行处理”所作，未经 owner 决定，列出以便审阅）：
- M1：`M1Config` 保留协议值 1e-4；T02 以 tolerance 1e-300（迭代预算每批用满）计时，另留一行协议记录不进 BUDGET（b83cdc3）。训练中 W 在 batch 间热启动（src/v10/adapter.py：“warm start of the next batch; saved with the encoder”）。
- clip：`clip_norm` = 1.0，作用范围按 src/v10/adapter.py 注释“host clip over all optimiser-trained parameters (MF: incl. W for perm1 / M0)”（T00 差异 D7 要求登记 W 是否进入 clip norm）。
- 可检测性检查只在 fp64 sharpened_x10 状态门控，host_init 与 fp32 行作为诊断（b83cdc3）。
- 计时配置：训练步与配准以 K16 为主（BUDGET 单元映射 `MF_M0_K16` 等），K8/K32 另测；配准计时用未训练的固定模型（BUDGET assumptions：搜索成本与权重无关）。

## 9 队列与优先级处理

- 规则（BUDGET.json `resources`）：每用户 GPU 上限 8，normal-QOS 提交上限 30；“V10 holds at most 2 normal-QOS jobs while V9-N2 keeps the count at 27; never above 29”。`slurm/v10/submit.sh` 强制：V10 已有 ≥ 2 个排队/运行作业或 normal-QOS ≥ 29 时拒绝提交，并把代码 pin 写入作业；作业头在代码改变、未 pin 或脚本与 HEAD 不符时拒绝运行。
- V10 作业资源：GPU 作业 partition L40S、--exclude=node51,node60、--nice=0、--time=02:00:00；CPU 作业 partition cpu-high、--nice=0、--time=02:00:00。V10 与 V9-N2 共用每用户 GPU 上限，V9 作业未取消。
- 主会话陈述（2026-10-17 任务说明；仓库、日志与结果文件中无记录）：第一阶段为让 V10 GPU 作业（--time=02:00:00）启动，主会话临时把排队中的 V9 作业 nice 最多调高 1300，并在该作业启动时（2026-10-17T01:36:48Z）恢复原值。可核对的时间：作业 1043695 作业头开始 2026-10-17T01:36:37Z，首个 CUDA part 开始 2026-10-17T01:36:49Z（T01 JSON）。
- BUDGET 生成时的队列快照（2026-10-17T01:48:35Z）：normal-QOS 27，GPU 分区上运行中（全部 QOS）10，排队 19。
- 本报告生成时（2026-10-17T02:33:17Z）：V10 作业 0 个；normal-QOS 28；v9n2_* 运行 7、排队 17。
- 本报告未提交、取消、hold 或修改任何作业。

## 10 限制

1. **只测了一种 GPU**：NVIDIA L40S（node50，作业 1043695）。V9 使用的其他 GPU 分区 RTX6000PRO、H100 未测；V9 的同类数字来自不同代码与时间，不是受控比较（V9-G1 vcs 在 RTX PRO 6000 Blackwell, job 1041939：0.734 s/次 DEV；T02 宿主 MLP_perm1 在 L40S：1.574 s/次，未训练模型）。
2. **EPFL 成本是代理**：按与 BF-SHG 相同的每窗口成本计；V9-G2 vcs 在 L40S, job 1042139 实测 1.479 s/次（DEV）。EPFL 的 CAL 调参未计入 BUDGET。
3. **M1 成本是上界**：BUDGET 取 `MF_M1_K16_it30_capbound`（迭代上限用满）；其他迭代上限另有实测（§4.6、§5.3）；M1 在 max_iter 维度上的调参未计入 BUDGET。
4. **配准计时**：未训练的固定模型，3 个 DEV core × 2 个扰动，每能量 n = 6；假定搜索成本与权重无关（BUDGET assumptions）。
5. **训练步**：只测宿主 batch（K16 为主，K8/K32 另测）；完整训练、checkpoint 读写、数据加载、CAL 选择、排队与失败重跑都未计入（BUDGET assumptions）。
6. **数值头**：随机合成 simplex 特征、单场景、固定排列，不是真实 descriptor；峰值显存含进程基线；CPU 计时有未诊断的成段跳变（§4.4），CPU 不测显存。
7. **T02 未测**：InfoNCE 能量、CoMIR、RuLSIF、MIND、MI/NMI、MLP perm8 配准能量、任何 EPFL 计时、TF32 关闭时的成本、其他 GPU 型号、多 GPU。
8. **T00 只读元数据**：未重算图像 hash；BF-SHG 的患者 / 组织块独立性未验证（core 是可核验的最细单位）。
9. **没有任务结果**：没有配准误差、成功率、S 估计误差或训练曲线；本报告的任何数字都不支持性能、加速或新颖性结论。

## 11 文件清单

结果文件（已提交；“包要求”= 「首批回传格式」列出的附件）：

| 文件 | 字节 | sha256[:16] | 包要求 |
|:---|---:|:---|:---|
| `v10/T00/EXPOSURE_NOTES.md` | 18714 | `8131cb1ba7808092` |  |
| `v10/T00/HOST_ADAPTER_AUDIT.md` | 41909 | `3b2b21b19bd3219e` | 是 |
| `v10/T00/MANIFEST_BUILD_SUMMARY.json` | 40828 | `4e18ad3c36c4e378` |  |
| `v10/T00/MANIFEST_SHA256.txt` | 182 | `7a4b7020433a318c` | 是 |
| `v10/T00/PREFLIGHT.json` | 6619 | `6c07f780968fa263` | 是 |
| `v10/T00/PREFLIGHT_RUN.log` | 983 | `6d8e07fccd27f5d2` |  |
| `v10/T00/V10_DATA_MANIFEST.csv` | 663565 | `cc309cece34b574b` | 是 |
| `v10/T00/host_map.json` | 46752 | `681bffa26681e64d` |  |
| `v10/T01/GEOMETRY_GRADIENT_CHECK.json` | 318315 | `845bd484bdeb7049` | 是 |
| `v10/T01/TORCH_VALIDATION.json` | 963463 | `f8cdb1cb86e4d306` | 是 |
| `v10/T02/BUDGET.json` | 8860 | `f104b335dd6b4854` | 是 |
| `v10/T02/T02_LONG_cpu.csv` | 59508 | `bba63630a2ce1249` | 是 |
| `v10/T02/T02_LONG_cuda.csv` | 111167 | `80af6fd52b33af4b` | 是 |
| `v10/T02/T02_TIMING_cpu.json` | 6215 | `ee4d5b0fe3e3da79` |  |
| `v10/T02/T02_TIMING_cuda.json` | 29415 | `c3b493c73bd2da81` |  |
| `v10/T02/UNIT_LEDGER.csv` | 22747 | `fdc37d242d538a2d` | 是 |

本报告读取的仓库外文件（未提交；日志与原始 run 目录原样保留）：

| 文件 | 字节 | sha256[:16] | 用途 |
|:---|---:|:---|:---|
| `/home/infres/yinwang/CS_QMI/slurm_logs/v10_t01_cpu_1043653.out` | 155367 | `64b743a3312f4ddb` | 作业 1043653 日志 |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v10_t01_cpu_1043657.out` | 155311 | `94c976f843856a10` | 作业 1043657 日志 |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v10_t01_cpu_1043668.out` | 155249 | `6ec224b33266f543` | 作业 1043668 日志 |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v10_t01_t02_gpu_1043695.out` | 158975 | `db42c1c6f397e557` | 作业 1043695 日志 |
| `/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/cuda_1043695/benchmark_torch/BENCHMARK_TORCH_cuda_float32_full.json` | 100349 | `2458513ab49f26e3` | 包 benchmark 原始 JSON（sha256 一致） |
| `/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/cuda_1043695/benchmark_torch/BENCHMARK_TORCH_cuda_float64_full.json` | 100449 | `5dd19b1a9032736c` | 包 benchmark 原始 JSON（sha256 一致） |
| `/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/cuda_1043695/train_steps_per_step.json` | 425567 | `592d47e167cc0554` | T02 训练逐步记录（§6.1） |

原始 part 与 run 目录：`/projects/EEG-foundation-model/yinghao/outputs_v10/T01/{parts,runs,host_replay}/`、`/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/`；T00 审计证据：`/projects/EEG-foundation-model/yinghao/outputs_v10/T00/`。
本报告：`v10/V10_PHASE1_STAGE_REPORT_20261017.md`；生成器：`scripts/v10/phase1_report.py`（sha256 b345dbacc2a6c38d…，1542 行）。仓库公开：报告不含数据、图像块、checkpoint 或逐图标签。
