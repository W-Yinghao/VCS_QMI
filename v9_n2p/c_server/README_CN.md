# V9 N2P C0/C1：服务器接入

包 `../VCS_V9_N2P_V10_Review_Server_Package_v1_20261020/` 原样提交于 4e7f6da0（zip sha256 `6d69b903…`）。本目录只有服务器接入：C0 直接调用包内脚本，C1 只加一个把 rich 权重接到原 `v9.n2.train.WeightedStudent` 的适配器。包内文件和 17 个锁定源文件都不改。

| 文件 | 作用 |
|---|---|
| `env.sh` | 路径与两个 Python 环境 |
| `trace_open.py` | 读文件证据：原样运行脚本，用 audit hook 记录打开的文件；C0 禁 DEV/TEST/eval_only/噪声文件，C1 允许 DEV |
| `c0_extract.sbatch` | GPU：`torch_crosscheck.py`，再对 901/902/903 运行 `extract_sgr.py --cache-mode new-cache` 和 `validate-cache` |
| `c0_fit.sbatch` | CPU：每 seed `fit`（16 个冻结候选，共 48 次）、`export --include-native-sensitivity`、`student_handoff.py`（rich_vcs、rich_bce），最后 `decide_rich_pilot.py` |
| `c1_student.py` | C1 适配器：`accept`（小型无 TEST 技术验收）、`run`（一个 seed 的两臂）、`summarize` |
| `c1_accept.sbatch`、`c1_seed.sbatch` | C1 的 GPU 作业 |
| `test_c_server.py` | CPU 合成检查，只验证代码路径 |

## 环境

- C0 用 `envs/v9_sgr`（Python 3.10.20、Torch 2.9.0+cu128、NumPy 2.2.6、nltk 3.10.3）。`extract_sgr.py` 要求空的输出目录，会用 nltk 重新分词；而且冻结 teacher 和历史分数（precheck 1042133，node58，RTX PRO 6000）都出自这个环境。
- C1 用 N2F/N2M 的学生环境 `CS_QMI/env`（Python 3.12.12、Torch 2.9.0+cu128、NumPy 2.5.2），从冻结的 N2F token cache 读取 token。
- 所有 Python 均以 `-I -B` 运行。

## C1 适配器做什么

- 权重只经包内 `student_handoff.validated_handoff` 读取，即原 `read_weight_lock` 加 cache/selection/export/observed 身份核对；而且必须是 `PILOT_DECISION.json` 里 `verified_weight_identities` 列出的文件。
- 用训练器自己的向量检查：A 权重恰为 1，U 权重等于锁文件，`mean_v_full = (n_A+Σ_U w)/n_pool`。
- 用训练器自己的 `iterate_batches` 检查 145000/14500/130500、每 epoch 1133 步（最后一批 104，只丢单样本批）、共 45320 步。训练后再按日志核对 40×1133 步、数值有限、best 为最早的最大值。
- HostConfig 取自冻结的 HOST_LOCK，并与协议配方逐项比对。`loss_scale="sum"`，lr_mult 1，从第 0 步起加权。
- 精度设置按协议：deterministic、cuDNN deterministic、benchmark 关、TF32 关、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、无 AMP。实际生效值写入记录，与协议不符就拒绝运行。
- 一个 seed 的两臂在同一个作业、同一张 GPU 上按协议顺序串行：901 VCS→BCE，902 BCE→VCS，903 VCS→BCE。两臂 seed 相同，所以初始模型、每 epoch 批次顺序相同。初始模型 hash、批次顺序 hash、optimizer 配置 hash、backend hash 都写入 `unit_result.json`。
- 续跑的作业若与该 seed 先前 attempt 的 GPU 型号、driver 或库版本不同，就退出 76。
- 只读 train 和完整 DEV（1014 图/5070 caption），代码里没有 TEST loader。

## 门槛

- `accept` 要求 `PILOT_DECISION.json` 为 `ELIGIBLE_PENDING_TECHNICAL_ACCEPTANCE`。
- `run` 还要求 `c1/C1_ACCEPTANCE.json` 为 PASS。
- 判定为 INCOMPLETE、HOLD_INVALID_POSTERIOR 或 NO_FULL_TRAINING_FROM_THIS_PILOT 时，C1 不运行，这些都作为有效结果回传。

## 不覆盖已有结果

- 已完成的步骤会跳过。
- 失败的 seed 保留原目录。重跑时用 `SEEDS=<seed> CACHE=<新名>`，c0_fit 使用同一个 `CACHE`。
- 每次技术验收写入新的 `c1/acceptance/<job>_<t>/`。
- 学生单元因时限或故障中断时，用同一行重交：已完成的臂跳过，未完成的臂从最后一个完整 epoch 续跑。每次 attempt 都记录在 `attempts.jsonl`。

## 读取证据的范围

trace 记录 Python 层的 open 事件，包括打开失败的尝试。这台服务器在数据角色之间没有挂载或权限隔离，所以以 trace 作为读取证据。`fit` 会对 7 个 NPZ 全部计算 hash、读取 ID，但只反序列化 FIT/SELECT 的特征；这是包的设计（README_zh §2）。
