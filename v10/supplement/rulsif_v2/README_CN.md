# RuLSIF_exact_v2：V10 核精确矩基线的第二个注册版本

分支 `v10/rulsif-exact-v2`（自 `ddd3bcb`），开发工作树 `ssl_pilot_v10_rulsif2`。本文件不在源码身份清单内，修改它不改变 code_sha256。

**状态（2026-10-19 UTC）：未提交任何 GPU 作业。** v2 未通过原预登记 FIT 功能门禁（第 4 节）。主会话（owner 授权：“你需要把实验做全，你自己决定范围”）随后决定采用**门禁之后的修订 A1**（第 11 节）：以层级 A 判定资格，层级 B 作诊断，候选 8 排除，order 0–7 进入 CAL。修订在已知 v2 FIT 门禁数值之后、在任何 v2 CAL/DEV 结果出现之前作出（至今没有任何 v2 CAL/DEV 行）。只运行了 CPU Slurm 作业。

## 1 v1 失败证据（v1 保留，报告为失败适配）

v1 = `RuLSIF_exact`（family 5，代码 ddd3bcb/ecd8a95，结果 `handoff/v10_method_pipeline_20261019/families/5`）。

- CAL 9 个候选：success<42 为 0–1/80，success<9 为 0/80，median 246–292 full-res px；按规则选中 ×0.5、ridge 1e-2。
- 全部 720 行 CAL：F_true < 0；F_hat ≥ F_true。各候选最大 F_true（ridge 1e-4/1e-3/1e-2）：×0.5 −0.193/−0.156/−0.143，×1 −0.048/−0.037/−0.044，×2 −0.0055/−0.0070/−0.0097。
- 代数经独立审查正确（与暴力 N² V-mode J(g−1) 相差 7.8e-16；解是 J(g−1) − λ|θ|² 的驻点）。问题在函数类和输入尺度，不在代数。
- FIT 交叉拟合复现（本分支，仅 FIT）：v1 规则在留出核心上 mean J −0.11…−0.42，J>0 的比例 0，真位姿高于施加扰动的比例 0.40–0.48（低于随机）；连 in-sample J 也为负（−0.07…−0.26）。

FIT 证据给出的三个机理：

1. **没有常数基**：T = g − 1 无法取 0，J(T≡0)=0 不可达；远离中心处 g→0、T→−1、J→−1。
2. **联合中位数带宽被 SHG 主导**：配准圆盘上 BF patch 中位距离 6.77，SHG 46.8，SHG 占联合平方距离 87%；BF 侧核值几乎恒定（×0.5 中位 0.965，×1 中位 0.991）。乘积核因此几乎不含 BF 信息，能量随移动 patch 的边缘覆盖变化，而不是随对齐变化。
3. **FIT 位置覆盖整幅图（5–506 px）**，而配准只在半径 129 的圆盘上评估。

## 2 v2 定义及依据

保持不变：RuLSIF、α=½、场景内精确 V 矩（U 矩已实现，BF 不用）、H=½H_P+½H_Q、fp64 累积与求解、θ/g/T 都不裁剪、fp32 特征与配准、宿主强度路径 9×9 patch、成对中心（每个 FIT 核心等量、无放回）、FIT-only 中位数规则、共同协议（FIT32/CAL8/DEV32、DataGuard、seed 1017 调参、1001–1003 正式、R6-REG-P2 搜索/支撑/扰动、字典序 CAL 选择、DEV24 敏感性、TEST134/EPFL 不读）。

改动（定义写在 `v10/supplement/RULSIF_V2_PREREGISTRATION.json`；证据见同目录 `DESIGN_EVIDENCE.json`：15 个 CPU 作业、230 个配置，4 折核心交叉拟合 sorted(FIT)[j::4]，seed-1117 随机流，只读 FIT）：

| 编号 | 定义 | FIT 证据（留出核心） |
|---|---|---|
| a 常数基 | g = θ₀ + Σθ_l a_l(x)b_l(y)；常数是“两侧核都恒为 1”的成对中心，精确矩公式不变；θ₀ 不进 ridge：(H+λD)θ=h_P，D=diag(0,1,…,1)，因此 FIT 上 E_{(P+Q)/2}[g]=1 精确成立，ridge 把核部分收缩到 T=0 | 无常数基的所有变体 J<0（v1 规则 −0.11…−0.42；只改支撑 −0.05…−0.24；只改带宽 −0.09…−0.38），J>0 比例 0–0.06。θ₀ 加不加 ridge：×0.5 时 |ΔJ|≤0.0002、各比例差 ≤0.02；×1、×2 且 ridge 1e-2 时比例差最大 0.13（两者都远低于阈值，不在网格内） |
| b 支撑 | FIT 位置取配准圆盘（半径 129，Q.RigidEnergy 的分布与抽样顺序），每核心 4096 个（与一次配准的位置数相同） | 圆盘比整图更好：×0.5 时真位姿高于施加扰动 0.68–0.74 vs 0.52–0.65；4096 比 2048 位置 0.83/0.81 vs 0.81/0.77 |
| c 分模态带宽 | h_x、h_y 分别为 BF、SHG 中心两两距离中位数 × 倍数 | 强度路径、带常数基：联合带宽下真位姿高于扰动 0.23–0.48（×0.5、ridge≤1e-2 时 0.23–0.29）；分模态后同一设置为 0.68–0.78 |
| e 逐图 ECDF | 每幅整图（FIT 图、参考图、每次配准的移动图）先换成自身的 mid-rank 经验分布函数值，再取 patch；单调变换，MI 不变 | 留出 SHG 亮核心的那一折训练时 SHG 中位距离 29–34（其余 40–45），J 为负；log1p SHG 无改善。ECDF 后（16 中心/核心，×0.25，1e-2）mean J −0.0011→+0.0011，J>0 0.57→0.72，真位姿>扰动 0.71→0.76，局部检验 0.53→0.70，四折均为正 |
| f 中心数 | 每 FIT 核心 32 个成对中心（共 1024；v1 为 128） | 8/16/32 中心：真位姿>扰动 0.79/0.79/0.83，局部检验 0.73/0.74/0.75 |

试过但未采用：log1p SHG（无增益）；ridge ≥3e-2（判别下降，λ≥0.1 时真位姿处依赖项变负）；×0.125（J≈0，判别 0.32–0.60）；ECDF+联合带宽（J>0 比例 9/9 个配置低于分模态，局部检验 8/9 个配置更低）。

## 3 预登记网格（第二阶段冻结，早于任何 v2 CAL）

固定：α 0.5、常数基、θ₀ 不进 ridge、圆盘支撑、4096 位置/核心、32 中心/核心、分模态中位数带宽、ECDF、SHG 强度路径、patch 半径 4。网格 digest `09777dbf195aa20dece4ed36cced09b92fe9f383ef7e134d29dd117180de3791`。

| order | 带宽倍数 | ridge |
|---:|---:|---:|
| 0 | 0.25 | 1e-4 |
| 1 | 0.25 | 1e-3 |
| 2 | 0.25 | 1e-2 |
| 3 | 0.35 | 1e-4 |
| 4 | 0.35 | 1e-3 |
| 5 | 0.35 | 1e-2 |
| 6 | 0.5 | 1e-4 |
| 7 | 0.5 | 1e-3 |
| 8 | 0.5 | 1e-2 |

`contract.candidates("RuLSIF_exact_v2")` 只从冻结记录读取网格（状态、顺序、digest、固定定义逐项校验）。CAL 按 success<42 降序、success<9 降序、median 升序、更早 checkpoint、候选顺序选择。

## 4 FIT 功能门禁（定义于 4c495a7，早于任何 v2 数值）

- 数据只读 FIT；样本 =（FIT 核心, 扰动 i=0..9），位置按配准规则（seed `R6-REG-P2|pos|core|i`），参考为 BF 亮度，移动图为宿主移动图流程在零扰动下的输出，施加扰动为冻结的 `Q.perturbations_p2(core)[i]`。
- 统计：F_true（真对齐处能量 = J）、F_imposed（按施加扰动错位时的能量）；±6°、±8 px 六个邻近位姿仅作诊断。
- 两个层级：A 为部署拟合（全部 32 个 FIT 核心、CAL 的 seed 流）在 FIT 核心的配准位置上评估（与求解用的位置独立）；B 为同一拟合程序的 4 折核心交叉拟合。
- 阈值：mean F_true > 0；F_true>0 的比例 ≥0.75；F_true>F_imposed 的比例 ≥0.75。规则：9 个候选在 seed 1017 下两个层级都通过才算 PASS；任一失败即不得提交任何 v2 CAL，不增删候选。

结果：见下方“门禁结果”（由 CPU 作业生成，报告在 `handoff/v10_rulsif_v2_20261019/gate/`）。

### 门禁结果（seed 1017，提交 72407e7，code_sha256 `5918f294…`；v2 作业 1048980，v1 作业 1048981；只读 FIT）

**结论：v2 未通过预登记门禁。按规则不得提交任何 v2 CAL。** 摘要见同目录 `FIT_GATE_SUMMARY.json`，完整报告在 `handoff/v10_rulsif_v2_20261019/gate/`。

| v2 候选 | 层级 A：mean J / J>0 / 真>扰动 / 局部 | 层级 B：mean J / J>0 / 真>扰动 / 局部 | 结果 |
|---|---|---|---|
| 0 ×0.25, 1e-4 | +0.0162 / 0.75 / 0.87 / 0.80 | +0.0070 / **0.62** / 0.78 / 0.75 | FAIL |
| 1 ×0.25, 1e-3 | +0.0094 / 0.76 / 0.89 / 0.83 | +0.0046 / **0.66** / 0.80 / 0.75 | FAIL |
| 2 ×0.25, 1e-2 | +0.0040 / 0.84 / 0.90 / 0.84 | +0.0015 / **0.64** / 0.78 / 0.72 | FAIL |
| 3 ×0.35, 1e-4 | +0.0371 / 0.78 / 0.87 / 0.78 | +0.0097 / **0.66** / 0.78 / 0.69 | FAIL |
| 4 ×0.35, 1e-3 | +0.0234 / 0.75 / 0.84 / 0.81 | +0.0063 / **0.65** / **0.74** / 0.69 | FAIL |
| 5 ×0.35, 1e-2 | +0.0129 / 0.78 / 0.85 / 0.81 | +0.0047 / **0.66** / 0.76 / 0.72 | FAIL |
| 6 ×0.5, 1e-4 | +0.0564 / 0.82 / 0.88 / 0.77 | +0.0194 / **0.68** / 0.82 / 0.74 | FAIL |
| 7 ×0.5, 1e-3 | +0.0379 / 0.80 / 0.87 / 0.75 | +0.0097 / **0.63** / 0.78 / 0.69 | FAIL |
| 8 ×0.5, 1e-2 | +0.0221 / **0.73** / 0.84 / 0.75 | +0.0023 / **0.62** / **0.72** / 0.66 | FAIL |

v1 的同一门禁：9 个候选两个层级全部三项不通过。层级 A mean J −0.108…−0.414、层级 B −0.088…−0.347；J>0 比例均为 0；真>扰动 0.41–0.52；局部 0.07–0.11。共享拟合代码对 v1 的 9 个部署拟合逐位复现（中心、带宽、系数完全相同）。

失败位置：层级 B 中，留出 SHG 亮核心那一折（1B_A5、1B_E3、1B_G3、1B_I7、1B_M2、2B_B8、2B_E5、2B_H4）在 8/9 候选上为负；32 个留出核心中有 10–12 个平均 J<0。真位姿处依赖项平均为正（+0.002…+0.064），在这些核心上被二次项抵消。层级 A（部署拟合）也有 5–9 个核心平均 J<0。第二阶段记录在门禁之前已写明“预计 cross-fit J 符号条件不通过”，门禁之后没有改动任何阈值、层级或规则。

参考（仅供决定，不用于推翻规则）：能用于配准的 V9 RFF 核 critic，在其 CAL 行中真对齐处 J>0 的比例也只有 46%（mean −0.0010），CAL success<42 为 0.25。也就是说，真位姿处 J 的符号比“能量能否用于配准”更严格。

## 5 已运行的内容（全部为 CPU Slurm，无 GPU）

- 设计证据：1048595、1048598、1048602、1048606–1048608、1048631–1048633、1048644、1048645、1048659、1048660、1048760–1048762（输出在 `/projects/EEG-foundation-model/yinghao/outputs_v10/rulsif_v2/design/`）。
- 正式门禁：v2 1048980（FAIL，exit 3），v1 1048981（FAIL）。
- CPU 回归：在冻结执行工作树 `ssl_pilot_v10_rulsif2_exec`（72407e7）中 1049001：5 个 tests/v10 文件共 116 项全部通过，前后工作树干净。1048982 在 git-archive 快照中跑，2 项失败只因快照没有 `.git`、且 `PROJECT` 被解析为 `/tmp`（环境假象，不是代码问题）。
- 已验证 release 构建脚本在门禁 FAIL 时拒绝生成 GPU 门禁 manifest（`REFUSED: FIT functional gate is not PASS`），没有写任何文件。
- 登录节点只跑过小张量的 pytest（2 线程）。

## 6 剩余步骤（主会话审核后执行；都在冻结执行工作树中）

```bash
SRC=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_rulsif2_exec   # 冻结执行工作树（detached，修订提交 bdca7fa）
cd $SRC
# 1) 修订后 FIT 门禁 PASS 后生成 GPU 门禁 manifest，并打印提交命令（已执行，见第 12 节）
/home/infres/yinwang/CS_QMI/env/bin/python scripts/v10_supplement/rulsif_v2_release.py prepare-gpu-gate \
  --fit-gate /home/infres/yinwang/CS_QMI/handoff/v10_rulsif_v2_20261019/gate/FIT_GATE_RuLSIF_exact_v2_amended.json
# 2) GPU 门禁（L40S，排除 node51,node60，nice 0；约 2–5 分钟 GPU 工作）
V10_VALIDATION_SOURCE=$SRC V10_VALIDATION_MANIFEST=<打印的路径> V10_VALIDATION_SHA256=<打印的 sha> \
  sbatch --parsable --export=ALL $SRC/slurm/v10_supplement/rulsif_v2_gpu_gate.sbatch
# 3) GPU 门禁 PASS（exit 0）后生成 release，并打印 CAL/final 命令
/home/infres/yinwang/CS_QMI/env/bin/python scripts/v10_supplement/rulsif_v2_release.py release \
  --fit-gate /home/infres/yinwang/CS_QMI/handoff/v10_rulsif_v2_20261019/gate/FIT_GATE_RuLSIF_exact_v2_amended.json --gpu-gate-job <job>
# 4) CAL（8 个合格候选串行，seed 1017，8 L40S GPU-h 上限）与三个正式 seed（afterok CAL）
ENV="V10_SUPPLEMENT_SOURCE=$SRC V10_SUPPLEMENT_RELEASE=/home/infres/yinwang/CS_QMI/handoff/v10_rulsif_v2_20261019/RELEASE.json V10_SUPPLEMENT_EXPECT_RELEASE=<release sha>"
CAL=$(env $ENV sbatch --parsable --job-name=v10r2_cal --time=06:00:00 --export=ALL $SRC/slurm/v10_supplement/rulsif_v2_run.sbatch cal RuLSIF_exact_v2)
for S in 1001 1002 1003; do env $ENV sbatch --parsable --job-name=v10r2_f$S --time=03:00:00 --dependency=afterok:$CAL \
  --export=ALL $SRC/slurm/v10_supplement/rulsif_v2_run.sbatch final RuLSIF_exact_v2 --seed $S; done
```

修订后 CAL 只跑 8 个合格候选（order 0–7，原冻结顺序）；第 6 节命令里的 CAL 即指这 8 个。

release 为本族单独的最小显式 release（`contract.require_release_v2`，状态 `FROZEN_PREREGISTERED_GATED`）：与 v1 的检查相同（源码身份、父协议、方案、数据锁、ACTIVE_SCHEDULING_POLICY 指针、暂停标记），验收换成 FIT 门禁 PASS + CUDA T01/执行门禁 PASS；端到端 T02/runner 验收由第一个正式 CAL 候选承担（2026-10-19 用户指出重复全量验证的成本，不再单独跑 seed 1117 全量验证）。结果目录 `handoff/v10_rulsif_v2_20261019/results` 链接到 `/projects/EEG-foundation-model/yinghao/outputs_v10/rulsif_v2/results`；成本账本、退出回执在 handoff 目录。

## 7 预期 GPU 成本（L40S，估计，GPU 门禁会实测单次配准时间）

v1 单次配准 8.1 s；v2 每个候选位姿多一个 1025×1025 的二次型，估计约 10–11 s。GPU 门禁约 0.1 GPU-h；修订后 CAL 8 候选 × 80 次配准约 2.0 GPU-h（上限 8）；每个正式 seed（CAL8 + DEV32 共 400 次配准）约 1.2 GPU-h；合计约 5.7 GPU-h。

## 8 报告规则

- `RuLSIF_exact`（v1）留在所有表中，标为失败适配，保留其 CAL/DEV 结果。
- `RuLSIF_exact_v2` 是在 v1 CAL 失败后、依据 FIT-only 证据与 v1 的 CAL 诊断新注册的变体，不是 v1 的重新调参；它**未通过原预登记的层级 B 判据**（cross-fit J>0 比例 0.62–0.68 < 0.75），在门禁后修订 A1 下运行（层级 A 资格、候选 8 排除）。所有 v2 结果都必须带这个标注，并写明 a/b/c/e/f 五项改动。
- 设计和门禁从未读取 DEV 行；v2 的 CAL 尚未运行。

## 9 需要 owner 决定的事项（2026-10-19 已由主会话按 owner 授权决定，见第 11 节）

1. **门禁 FAIL 之后怎么办。** (a) 维持规则：v2 不运行，RuLSIF 报告为“v1 CAL 失败的适配 + v2 未过预登记 FIT 门禁”。(b) 门禁之后的修订（必须标为门禁后修订：已经看到 FIT 门禁数值，没有看过任何 CAL/DEV）。与最初任务表述最接近的版本：只要求层级 A 的三项（部署的 T 在求解未用过的 FIT 数据上），层级 B 改为诊断，未过层级 A 的候选排除，得到 order 0–7 共 8 个候选。做法：修改记录（`functional_gate.required_levels=["deployed_position_heldout"]`；`candidate_grid` 取 0–7 并重算 digest；新增带 owner 批准的 amendment 条目）→ 提交 → 执行工作树切到新提交 → 重跑 FIT 门禁（CPU 约 25 分钟；随机流相同，数值相同）→ 再按第 6 节执行。按 8 个候选估计约 5.7 GPU-h。
2. **θ₀ 不进 ridge。** 用 λD 而不是任务中写的 (H+λI)。在网格所在的 ×0.5 区域 FIT 上无实质差别（|ΔJ|≤0.0002，比例差 ≤0.02；×1–×2 且 ridge 1e-2 时比例差最大 0.13），保留它是为了 E_{(P+Q)/2}[g]=1 精确成立。
3. **GPU 分区。** 按主会话要求，脚本只用 L40S（run.py/validate.py 也检查 `NVIDIA L40S`，每族预算以 L40S GPU-h 定义）。用户 09:31 UTC 的指示是后续 GPU 任务统一 `RTX6000PRO,L40S,H100`；如果对 v2 也适用，需要类似复现池的硬件修订（改代码，然后重跑两个门禁）。
4. v2 的改动比“常数基 + 支撑”更多（还有 c、e、f 三项），每项都有 FIT 证据，并在第二阶段记录中分别列出。报告时需要逐项说明。

## 10 勘误（第二阶段记录中的四处表述；记录本身不改，因为其 sha256 已被门禁报告固定）

以下数字由 `DESIGN_EVIDENCE.json` 重新计算，不影响任何定义、阈值或结论：

1. `a_constant_basis.fit_evidence` 写“|delta mean J| <= 0.0006, same fractions within 0.02”。准确：ridge≤1e-3 的所有带宽和 ×0.5 的所有 ridge 成立（|ΔJ|≤0.0002）；×1/1e-2 时 J>0 比例差 0.13，×2/1e-2 时 |ΔJ| 0.0009、真>扰动差 0.11。
2. `c_per_modality_widths.fit_evidence` 写“Joint width with constant: 0.23-0.48; per-modality: 0.59-0.80”。准确：0.23–0.48 只指强度路径（a_only、ab_c4/c8/c16）；log1p+联合最高 0.61，ECDF+联合最高 0.77。分模态在 ×0.5、ridge≤1e-2 时为 0.68–0.78，在 ×2 时同样很差（0.21–0.26）。
3. `design_evidence.expected_gate_outcome` 写“0.57-0.72”。准确：分模态+常数基、×0.25–×0.5、ridge≤1e-2 的 50 个配置中，cross-fit J>0 比例为 0.49–0.72（最高 0.716，ECDF、16 中心、×0.25、1e-2）。
4. `tested_not_adopted.joint_width_with_ecdf` 写“below on J > 0 fraction and local test”。准确：J>0 比例 9/9 更低；局部检验 8/9 更低，×0.25/1e-2 例外（0.71 对 0.68）。

## 11 门禁后修订 A1（`v10/supplement/RULSIF_V2_AMENDMENT_20261019.json`）

- 性质：**门禁后修订**。作出时已知 v2 FIT 门禁数值；尚无任何 v2 CAL/DEV 结果。决定者：主会话（owner 原话“你需要把实验做全，你自己决定范围”）。预登记记录不改（第二阶段 sha256 `48f8211d…`，第一阶段 `43a274fd…`），修订记录引用它们以及原门禁报告（v2 `0edd0ecd…`，v1 `4da26598…`）。
- 新资格规则：只要求层级 A（部署拟合在新的 FIT 位置上），条件就是原门禁已定义的层级 A 判定，不变：mean J>0、J>0 比例 ≥0.75、真>扰动比例 ≥0.75。层级 B 照常报告，作诊断。未通过层级 A 的候选排除，合格候选保持冻结顺序：order 0–7 合格；8（×0.5，1e-2）层级 A J>0 比例 0.728（233/320）排除。
- 理由（原文）：the gate exists to stop degenerate energies like v1 (J < 0 everywhere, ranking at or below chance); v2 ranks the true pose above the imposed perturbation in 0.84–0.90 (A) / 0.72–0.82 (B); the per-sample J > 0 condition at level B is stricter than what a working kernel critic achieves here (V9 RFF kernel critic: J > 0 in 46% of CAL rows while registering)。对应数字：v1 层级 A 真>扰动 0.41–0.52、J>0 比例 0；V9 RFF 核 critic 的 CAL 行中 37/80 J>0，CAL success<42 0.25。
- 公开的偏离：θ₀ 不进 ridge（λD，不是字面的 λI），证据见第 9 节第 2 条；硬件只用 L40S（预算单位与其他补充基线的 CAL 预算可比），**有意**不同于 09:31 UTC 的三分区指示（那条指示针对 seed 复现池）。
- 执行：修订后的代码在提交 bdca7fa；CPU 门禁在该提交上重跑（作业 1049157），要求与原运行的每个层级判定完全一致，并恰好得到 order 0–7；release 检查和构建脚本都要求修订后的报告并固定修订记录。

## 12 修订后的 CPU 门禁重跑（作业 1049157，提交 bdca7fa，code_sha256 `f6ea2af6…`）

- 结果：**PASS**（修订规则）。order 0–7 合格，8 排除（层级 A J>0 比例 0.728）。报告 `handoff/v10_rulsif_v2_20261019/gate/FIT_GATE_RuLSIF_exact_v2_amended.json`（sha256 `cd8447e9…`），摘要见同目录 `FIT_GATE_AMENDED_SUMMARY.json`。
- 与原运行（1048980）对照：两次都用 24 线程，但节点不同（nodecpu02 对 nodecpu08），所以**不是逐位相同**：5760 行中 57 行完全相同，能量最大绝对差 6.0e-7（fp32 求和顺序）；连续汇总量最大差 3.6e-7；系数和的相对差 ≤4e-8；中心、两侧带宽、常数系数完全相同。**所有比例、计数、各层级判定和合格集合完全相同**（verdict_differences 为空）。边界候选 0 和 4 的层级 A J>0 恰为 240/320，但其最小 |F_true| 为 4.1e-5 和 2.4e-4，远大于上述数值差。
- 只读 FIT；未读 CAL、DEV、TEST、EPFL。
- 之后：冻结执行工作树切到分支最新提交（代码身份不变，只多了文档），在其中执行了 `prepare-gpu-gate`；GPU 门禁 manifest 的路径、sha 和提交命令写在 `handoff/v10_rulsif_v2_20261019/STATUS.json`。

## 13 跨硬件 final 版本（owner 决定 2026-10-19：「final 放开，CAL 留 L40S」）

- owner 选择（原文）：「final 放开，CAL 留 L40S」。当时给出的选项（原文）：“EPFL final、BF 和 EPFL 的 RuLSIF v2 final、K8/K32 剩下的 6 个单元，改为三分区，记录实际硬件、分层报告。CAL 和计时留在 L40S，保持每族 8 个 L40S GPU 小时的同等预算。先跑跨硬件门禁，再撤下未开始的作业重新提交。” 依据 09:31 UTC 的常设指示（原文）：“后续待运行GPU任务统一 --partition=RTX6000PRO,L40S,H100，不要再按seed、方法或GPU型号人为分组/限卡；记录实际硬件与成本即可”。
- 起点：CAL release `RELEASE.json`（41313c39…，提交 2e6ea58，冻结工作树 `ssl_pilot_v10_rulsif2_exec`），两者都不改。CAL 作业 1049724 以 exit 0 结束；`SELECTION.json`（cea56cbc…）选中 order 5（×0.35、ridge 1e-2）：CAL success<42 0.75，<9 0.625，median 6.05 px；8 个候选都完成，未触及预算上限（实耗 1.52 L40S GPU-h）。
- 新版本：提交 eaaef1c3，冻结工作树 `ssl_pilot_v10_rulsif2_xhw`，code_sha256 `1dfedd30…`，release `RELEASE_XHW.json`（sha256 `12e37693…`）。
  - **唯一的差异**：final 接受 `NVIDIA L40S`、`NVIDIA RTX PRO 6000 Blackwell Server Edition`、`NVIDIA H100 NVL`（即 CUDA 设备报告的名字）；CAL 仍只接受 L40S，且在新 release 下不能运行；其余环境和精度检查不变（torch、CUDA runtime、cuDNN、线程数、精度设置、TF32 覆盖变量）。
  - 差异审查 `xhw/DELTA_REVIEW.json`：改动 `run.py`、`validate.py`，新增 `rulsif_v2_xhw.py`、`rulsif_v2_xhw_release.py`、`test_rulsif_v2_xhw.py`；`src/**`、`v10/supplement/*.json`、`slurm/` 和原有测试与 2e6ea58 **逐字节相同**（31 个科学文件）。每个 final 启动时都会重新比对。
  - CAL 选择只读取、不重选：final 校验 SELECTION.json 的 hash，并在 41313c39 下核对 CAL 的 exit 回执和 8 个候选单元的身份。
  - 已在 41313c39 下开始或完成的 seed 保持有效，新版本拒绝该 seed，不重跑（两个版本共用 seed 锁）。
  - 每个型号都要先过门禁：L40S 用 CAL release 的门禁 1049287（科学代码逐字节相同）；RTX PRO 6000 和 H100 NVL 需要在固定的跨硬件 manifest（`xhw/validation/MANIFEST.json`，f5e18730…）下有 PASS 门禁和 exit-0 回执，final 启动时查找并记录。
  - 每个 final 只在一种 GPU 上运行（续跑必须同型号），记录 GPU 型号、节点、分区（`HARDWARE.json`、成本账本 `RUNTIME_VERIFIED`、`ENV_<job>.json`）以及分配时间（exit 回执）。
- 报告规则：逐个 final 记录硬件；结果按硬件分层报告，同时报告合并结果；不声称跨硬件等价。CAL 与预算仍按 L40S（每族 8 个 L40S GPU-h）。
- 执行（主会话）：先提交两个门禁（RTX6000PRO、H100，排除 node51,node53,node60，限时 30 分钟），两者都 PASS 后提交三个 final（`--partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --time=03:00:00`）。准确命令见 `handoff/v10_rulsif_v2_20261019/STATUS.json`。`deps_watch.sh` 仍可能按 41313c39 在 L40S 提交 final：要么先暂停它（写 PAUSE），要么接受它提交的 seed 留在 41313c39 下。两版作业同名（`v10r2_f<seed>`），所以 watcher 会跳过已在队列中的 seed。
- `preflight` 已在真实记录上通过（release、选择绑定、三个 seed 均未开始、门禁：只有 L40S）。

## 14 事后硬件敏感性诊断：seed 1001 在 L40S 上重跑同一单元

- 性质：**事后硬件敏感性诊断**（owner：「final 放开，CAL 留 L40S」「什么快什么来吧」）。三个正式 final 已在 RELEASE_XHW 下完成：seed 1001（作业 1050357）和 1002（1050376）在 RTX PRO 6000 上，1003（1050407）在 L40S 上。本诊断把 seed 1001 作为**完全相同的单元**重跑一次：身份与主单元 `IDENTITY.json` 逐字段相同（RELEASE_XHW 的科学代码、CAL 选择 cea56cbc、seed 1001），只在 L40S 上运行。它不是第四个 seed，不与主结果合并，也不替代主单元。
- 代码：提交 6859cd7d，冻结工作树 `ssl_pilot_v10_rulsif2_hwrep`，code_sha256 `057e5f80…`；release `hw_replicates/RELEASE_HWREP.json`（sha256 `619b2bf8…`）。差异审查 `hw_replicates/DELTA_REVIEW.json`：只改 `run.py`（诊断模式分派），新增 `rulsif_v2_hwrep.py` 和其测试；`src/**`、`v10/supplement/*.json`、`slurm/` 和已有测试与 eaaef1c3、2e6ea58 逐字节相同。
- 输出：`handoff/v10_rulsif_v2_20261019/hw_replicates/results/FINAL_DEV/RuLSIF_exact_v2/seed_1001__L40S_replicate`。锁、成本账本和退出回执都放在 `hw_replicates/` 下，绝不写入主单元。
- 一致性记录（作业结束时写入 `hw_replicates/AGREEMENT_seed_1001.json`，也可用 `rulsif_v2_hwrep.py agreement` 重算），相对 RTX 主运行逐次配准比较：err_full 完全相同的比例、最大绝对差、中位绝对差；θ̂、t̂、F_hat、F_true 的同类统计；success<42、<9 阈值翻转数；DEV32、DEV24、CAL8 的 success<42、<9、median 差（重跑减主运行）；拟合模型 hash 是否相同。
- 提交（主会话）：`--partition=L40S --exclude=node51,node60 --time=03:00:00`，作业名 `v10r2_hwrep1001`；准确命令见第 13 节同目录 STATUS 或下方汇报。
