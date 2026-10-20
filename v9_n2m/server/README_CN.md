# V9 N2M：本服务器上的执行接入

包 `VCS_V9_N2M_Server_Package_v1_20261019` 原样位于 `../VCS_V9_N2M_Server_Package_v1_20261019/`，提交于 8fe6f8a，zip 的 sha256 为 `cb1da9b4…`。本目录只放服务器侧的作业脚本，包内文件一律不改动，因此包的代码身份和 `PACKAGE_MANIFEST.json` 都保持不变。

## 路径（env.sh）

| 变量 | 值 |
|---|---|
| N2M_PACKAGE | `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019` |
| N2M_BASE | `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017`：原 N2F 执行包，FROZEN/GROUPS 与包内 provenance 一致 |
| N2M_OUT | `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1` |
| N2M_PYTHON | `/home/infres/yinwang/CS_QMI/env/bin/python`：Python 3.12.12、Torch 2.9.0+cu128、NumPy 2.5.2 |

## 与包内示例命令的差异（只改调度参数，不涉及科学内容）

1. **作业名。** 改用 `n2m_*`，不用包模板里的 `v9n2m_*`。原因：仍在运行的 N2F 收尾控制器 1047107（`v9_ess_resume_20261018/reconcile.py:42`）和 1046165（`v9_overlap_v10_20261018/bridge.py:116`）会一直等到没有任何名字以 `v9` 开头的作业。若沿用 `v9n2m_*`，约 19 小时的新训练会把 N2F 收尾一起拖住。作业名在 sbatch 命令行上覆盖，模板文件不变。早先的 `v9n2m_verify`、`v9n2m_cpucheck`、`v9n2m_drvprobe` 在 1046654 仍在运行时已经结束，没有造成影响。
2. **分区。** 统一使用 RTX6000PRO（node58/59/61），这也是包的默认值。包要求核对作业和所有新三臂处于同一环境，runner 在每次训练开始时都会比较 GPU 型号、driver、NumPy、cuDNN、线程等，不一致就拒绝运行。三台节点的 driver 均为 615.71.09（2026-10-19 11:08 UTC 实测），所以 Slurm 在这三台中任选都可以。这一点与 10-19 09:31 的通用指令“在 RTX/L40S/H100 三分区中由 Slurm 调度”不同；N2M 的主比较是同一 seed 内三臂的配对，混用型号会改变比较条件，因此以包的规定为准。
3. **日志。** 统一写到 `/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out`。
4. **成本记录。** sacct 不可用（slurmdbd 拒绝连接）。`record_slurm.sh` 在作业存活期间每 30 秒只读保存一次 `scontrol show job -o` 快照，存于 `N2M_OUT/slurm_records/`，作为 allocation 开始、结束时间和退出码的现场凭据。`accounting` 需要等 sacct 恢复后再运行，或传入 `--sacct-text`。

## 实际执行记录

| 步骤 | 作业 / 命令 | 结果 |
|---|---|---|
| 包核对 | 登录节点 `verify-package` | PASS，44 个文件 |
| bootstrap | 登录节点 | PASS；只改了 `src/v9/n2/train.py`；effective digest `7cfe4f86…` |
| CPU 核对 | 1048421（cpu-high） | inspect NOT_READY（预期）：32/33 完整且 checkpoint 已核对，缺 903_1 VCS SUM（1046654 仍在运行）；源文件 49 个与输入 15 个全部 PASS；单元测试 119 项 OK；validate_local --full-shape PASS |
| CUDA 核对 | 1048422（RTX6000PRO，node58） | 24/24 检查 PASS；SERVER_VERIFICATION.json `status=PASS` |
| driver 探针 | 1048423–25 | node58/59/61 均为 615.71.09 |

## 后续顺序

1. 原 N2F 33/33 完成后（1046654 结束），执行 `sbatch prepare.sbatch`（冻结 6–9 次矩阵；auto 预计 replay9）。
2. 执行 `./submit_n2m.sh seed 901`、`seed 902`、`seed 903`，每个 seed 一个 GPU 作业，按冻结顺序串行跑该 seed 的三臂。作业退出码为 75 时，用同一命令续跑。
3. 执行 `./submit_n2m.sh loss 901|902|903`，导出固定学生的损失；导出完成后在 CPU 上运行 `score-fixed-losses`。
4. 执行 `sbatch score.sbatch`，做三个 seed 的离线后验评分。
5. 全部完成后运行 `summarize`；`accounting` 在 sacct 可用时运行。
