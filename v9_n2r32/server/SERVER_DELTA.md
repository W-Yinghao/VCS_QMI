# V9-N2R32 服务器接入：与包的差异记录

包 `round_20261021_application_adaptation/VCS_QMI_Server_Round_20261021_v1`（zip sha256 `29aa3285…`）的 `v9/`、`shared/`、`SOURCE_LOCK.json`、`PACKAGE_MANIFEST.json` 原样提交于 `v9_n2r32/package/`（c66822c6），17 个文件与 PACKAGE_MANIFEST 逐字节一致。本目录只有调度包装，**包内文件一个字节都没改**，门槛、阈值、检查都没改。

## 路径（`env.sh`）

| 变量 | 值 | 依据 |
|---|---|---|
| `V9_REPO` | 本工作树 `ssl_pilot_v9_n2r32`（分支 v9/n2r32-20261021，基于 v9/n2p-c-20261020 01dab37d） | SOURCE_LOCK 的 13 个源文件在此树与 ea0535b5 逐字节一致（登录节点 sha256 核对） |
| `N2P_AFFINE_ROOT` | `outputs_v9_n2p/c_20261020` | 旧 affine C0/C1；只读 |
| `N2R_ROOT` | `outputs_v9_n2p/round_20261021_n2r32` | 负责人此前的 N2P 输出放 home 的决定 |
| `V9_HISTORY`、`V9_DATA`、`V9_VOCAB`、`V9_TOKENS` | 与 C1（`ssl_pilot_v9_n2p_c/v9_n2p/c_server/c1_student.py`）相同 | 同一学生池、DEV、冻结 token cache |
| `PY_CPU` | `envs/v9_sgr`（Python 3.10.20，NumPy 2.2.6） | 原 affine fit 1052397 用的环境；`check_affine_selection` 要逐位复现标准化和 affine SELECT J |
| `PY_GPU` | `CS_QMI/env`（Python 3.12.12，Torch 2.9.0+cu128，NumPy 2.5.2） | C1 学生环境 |

## 包装

- `cpu_checks.sbatch`（cpu-high）：包的唯一 CPU 验包入口 `tests/run_cpu_checks.py --repo`。
- `fit_export.sbatch`（cpu-high）：V9_SERVER_CN.md §5 的 fit/export 命令（trace 名 `fit_s<seed>_<ATTEMPT>.json` 等），三个 export 都在时接着跑 `decide_residual.py`。已完成的 fit/export 跳过；重试要用新的 `ATTEMPT`。
- `accept.sbatch`、`seed.sbatch`（GPU）：`student_runtime.py accept` / `run`，命令行与 §5 相同。`seed.sbatch` 加了 `trap TERM`，让 runtime 在时限处先写完 INTERRUPTED 记录，再写作业结束行。
- `submit_when_room.sh`：normal 作业数低于 28 才 sbatch（与其他会话共用上限）。
- 成本：sacct 不可用，每个作业在 `$N2R_ROOT/cost/JOBS.tsv` 写开始和结束行（节点、GPU 型号与驱动、rc）；没有结束行的作业按成本未知保留。

## 调度选择

- 技术验收 `accept` 用项目默认的 `--partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60`。
- **3 个 seed 学生作业固定在 RTX6000PRO**（同样排除 node51、53、60），这是与默认三分区规则的一处偏离，理由：
  1. `compare_architectures.py` 只有在新旧学生的 `backend_sha256` 相同时才把 residual−affine 当作身份匹配的容量比较；backend 包括 GPU 型号和驱动。旧 affine C1 的 6 个单元全部在 RTX PRO 6000（驱动 615.71.09，node58/59/61）上，backend_sha256 都是 `bbe781ba…`。新学生落在 L40S/H100 上，该 seed 的容量比较就只能标为上下文，且一旦开跑无法挽回。
  2. 同一 seed 的两臂和所有续跑必须同型号、同驱动（runtime 不符就以 76 退出）；按 C1 的实测，只有 RTX 能在一个作业内（时限 23:59:00）跑完两臂。
  - 这与 C1 绑定 RTX 的理由相同。排队中的作业可以撤下、改成三分区重交，没有损失；若负责人要三分区，照此处理即可。
- 续跑：同一命令，仍固定 RTX6000PRO。RTX 三个节点驱动相同，不需要固定节点。每个 seed 最多因基础设施原因续跑 2 次，每次 attempt 都记录。

## 执行结果（本线到决定为止）

- CPU 验包 1055943（nodecpu01，19 s）：PASS，12/12，无跳过。
- 拟合与导出 1055952（nodecpu01，92 s，ATTEMPT=attempt1）：24 次拟合各 400 步、无续跑；6 份后验都有效（raw rho 0.367–0.403，gap ≥ 0.93），6 个权重文件已导出；7 份 trace 都是 PASS。
- `decide_residual.py`：**NO_NEW_STUDENTS_FROM_N2R32**（RESIDUAL_DECISION.json sha256 `e636bc07…`）。residual−affine SELECT J：VCS −0.0332/−0.0757/−0.0370（均值 −0.0486，0/3 为正），BCE −0.0332/−0.0357/−0.0198（均值 −0.0296，0/3 为正）。
- 按包的规定关闭 N2R32：不做技术验收，不跑新学生，不降低 .01、不扩网格。上文的 accept/seed 包装没有用到。
- 小型记录复制在 `../records/`（与 `$N2R_ROOT` 同路径结构；权重 CSV、分数 npz 和 checkpoint 只留在 `$N2R_ROOT`）。
