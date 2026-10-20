# D1b CoMIR 差距的记录核查（不含新计算）

标签：post-DEV sensitivity 的附属记录核查。只读结果 JSON、训练日志、Slurm 日志首行和缓存哈希；不训练、不配准。
生成：`scripts/v10_sensitivity/audit_comir_gap.py`，2026-10-19T12:32:20Z；机器可读版本 `<bundle>/D1/COMIR_GAP_RECORDS.json`。

## 结论

- DEV cores, CAL cores and all 320 DEV / 80 CAL perturbation truths are identical across R6, V9-G1 and V10.
- Preprocessing and inputs are identical: same R6 cache files (data-lock hashes match now; file mtimes 2026-10-11T21:39:48Z..2026-10-11T21:42:13Z, before the first R6 CoMIR training job started at 2026-10-11T22:45:00Z), same Q.Inputs path; R6 and V9-G1 training FIT lists equal the V10 FIT32.
- Training recipe, RNG call order, FIT order and the encode-then-register path are the same by code reading; the only code difference (efficient=False at R6 registration) changes no value under no_grad.
- Differences: training seeds (0-2 / 901-903 / 1001-1003 and 1017); training GPU (R6: RTX PRO 6000 x2 and H100 NVL x1; V9-G1 and V10: L40S node52); registration GPU (R6: L40S x2, RTX PRO 6000 x1; V9-G1, V10: L40S node52). Same torch 2.9.0+cu128 environment (installed 2026-09-24).
- CoMIR training is not bitwise reproducible on this stack: two V9-G1 jobs of the same unit on the same node and with the same hashed host code (repository commits 915de3c and e04a336 otherwise differ) logged epoch-1 losses 4.4081 and 4.4057.
- The run-to-run differences sit on 12 variable DEV cores; 18 cores succeed in all nine runs and 2 fail in all nine.
- No cause can be attributed from records: three runs per generation, seeds and hardware confounded. D1 tests only the checkpoint-epoch factor.

## 各次运行

| 代 | seed | DEV32 <42 | DEV32 <9 | CAL <42 | 训练日志首行 | 配准日志首行 |
|---|---|---|---|---|---|---|
| R6_REG_P2 | 0 | 0.9125 | 0.6031 | 0.8750 | host=node59 NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 | host=node39 NVIDIA L40S, 46068 MiB start=2026-10-12T01:01:21 |
| R6_REG_P2 | 1 | 0.8906 | 0.6156 | 0.9875 | host=node58 NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 | host=node50 NVIDIA L40S, 46068 MiB start=2026-10-12T01:12:41 |
| R6_REG_P2 | 2 | 0.8969 | 0.6406 | 1.0000 | host=nodesumo01 NVIDIA H100 NVL, 95830 MiB start=2026-10-11T | host=node61 NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 |
| V9_G1_frac1_eps0 | 901 | 0.8875 | 0.6250 | 0.8750 | host=node52 NVIDIA L40S, 46068 MiB start=2026-10-15T15:49:15 | host=node52 NVIDIA L40S, 46068 MiB start=2026-10-15T15:49:15 |
| V9_G1_frac1_eps0 | 902 | 0.8250 | 0.5594 | 0.8750 | host=node52 NVIDIA L40S, 46068 MiB start=2026-10-15T15:49:15 | host=node52 NVIDIA L40S, 46068 MiB start=2026-10-15T15:49:15 |
| V9_G1_frac1_eps0 | 903 | 0.8281 | 0.5719 | 0.8125 | host=node52 NVIDIA L40S, 46068 MiB start=2026-10-15T15:49:15 | host=node52 NVIDIA L40S, 46068 MiB start=2026-10-15T15:49:15 |
| V10 | 1001 | 0.7406 | 0.5000 | 0.7500 | node52 NVIDIA L40S | node52 NVIDIA L40S |
| V10 | 1002 | 0.8750 | 0.5656 | 0.8750 | node52 NVIDIA L40S | node52 NVIDIA L40S |
| V10 | 1003 | 0.7781 | 0.5188 | 0.7500 | node52 NVIDIA L40S | node52 NVIDIA L40S |
| V10 调参 | 1017 | — | — | 0.3500 | node52 NVIDIA L40S | 同左 |

## 逐 core 模式（DEV32 success<42，9 次运行）

- 9 次全部成功：18 个 core
- 9 次全部失败：1B_M3, 2B_H10
- 随运行变化：1B_H4, 1B_H9, 1B_K6, 2B_A10, 2B_A6, 2B_C1, 2B_D8, 2B_E9, 2B_F5, 2B_H2, 2B_H5, 2B_I5

| core | R6_REG_P2:0 | R6_REG_P2:1 | R6_REG_P2:2 | V9_G1_frac1_eps0:901 | V9_G1_frac1_eps0:902 | V9_G1_frac1_eps0:903 | V10:1001 | V10:1002 | V10:1003 |
|---|---|---|---|---|---|---|---|---|---|
| 1B_H4 | 1.0 | 0.9 | 0.9 | 1.0 | 0.7 | 1.0 | 0.1 | 1.0 | 1.0 |
| 1B_H9 | 1.0 | 0.7 | 0.8 | 0.5 | 0.4 | 0.2 | 0.0 | 0.1 | 0.6 |
| 1B_K6 | 1.0 | 0.9 | 1.0 | 1.0 | 0.5 | 0.5 | 0.3 | 1.0 | 0.3 |
| 2B_A10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 |
| 2B_A6 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2B_C1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 1.0 | 1.0 |
| 2B_D8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 | 0.9 | 1.0 | 0.7 |
| 2B_E9 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2B_F5 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 1.0 | 0.7 | 1.0 | 0.0 |
| 2B_H2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 0.6 | 0.0 | 1.0 | 0.2 |
| 2B_H5 | 1.0 | 1.0 | 1.0 | 1.0 | 0.7 | 0.7 | 0.6 | 0.9 | 0.3 |
| 2B_I5 | 1.0 | 1.0 | 1.0 | 0.9 | 0.9 | 0.5 | 0.2 | 1.0 | 0.9 |

## 非确定性证据

- 单元 V9-G1-comir-frac1-eps0-s901，两次作业 host-code 哈希相同（6c6f094684fbc50a (both job headers)）：
  - 被取消的重复作业：['[comir V9-G1-comir-frac1-eps0-s901] epoch 1 loss 4.4081 (13s)']
  - 保留的作业：['[comir V9-G1-comir-frac1-eps0-s901] epoch 1 loss 4.4057 (12s)']

