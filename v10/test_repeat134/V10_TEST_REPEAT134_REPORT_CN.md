# V10 TEST_REPEAT134：BF 重复基准（单独报告，不是独立确认；v4 同作业配对）

**状态：最终单元 0/42 完成，0 个 BLOCKED，其余 pending。** 本文由 `scripts/v10_test_repeat/analyze.py` 从冻结记录与已完成的作业输出生成；同一输入两次运行逐字节相同。

> **读法。** TEST_REPEAT134 (repeated benchmark; BF/SHG official test cores read once before by V9-G1; not an independent confirmation)。规则原文：reported as its own table, separate from DEV32; DEV32 / DEV24 shown beside it；DEV and TEST_REPEAT are never pooled (no 166-sample confirmation)；no independent-confirmation claim; prior exposure V9-G1 stated；TEST112 sensitivity reported beside TEST134；RuLSIF_exact (v1) carries its failed-adaptation label; RuLSIF_exact_v2 carries amendment A1's label；hardware recorded per unit; secondary replication by hardware；no method-superiority, strong-baseline or novelty claim while the required baselines are not confirmed complete (decision 5)；intervals are descriptive; no significance tests；TEST134 / TEST112 are reported beside the same-job DEV32 re-evaluation (paired reference) and beside the recorded DEV32；an evaluation-noise section reports every DEV re-evaluation pair; method differences smaller than this noise are not interpretable, on DEV as well as TEST (including the main V10 DEV tables)；BLOCKED units are reported with their failed criteria; TEST is never read for them。

## 1 负责人决定与冻结记录

- 启动（2026-10-20，UTC about 06:45）问题原文：「BF 上 V10 全部方法和 seed 的选择已经齐全，可以做 TEST_REPEAT134（读 held-out 的 TEST134）。现在开始吗？」选项：「开始 (Recommended)」——「先把所有方法和 seed 选中的 checkpoint 冻结成新 manifest，冻结完成前不读 TEST；然后在 TEST134 上评估。结果单独报告为重复测试，另附 112 个 core 的敏感性集，不当作独立确认。」
- v4（2026-10-20）问题原文：「评估本身有噪声（同一 checkpoint 重评，<9 变动 0.6–2.2 个百分点），TEST_REPEAT134 怎么继续？哪个方案都先登记、后读 TEST。」选项：「同作业配对，评一次 (Recommended)」——「每个单元在同一个作业里先重评 DEV32、再评 TEST134，TEST 和这次 DEV 并列报告。门禁改为：checkpoint、输入、扰动和真实位姿能量必须逐位一致，成功次数只粗查，用来拦流程错误。重评数据披露为噪声下限。三个分区都可以跑，约 50 GPU 小时。」未选方案：确定性评估（启用确定性算法 / cuDNN 确定性后重评 DEV32 与 TEST）；重复评估取平均（多次重评以平均评估噪声）。
- `v10/test_repeat134/OWNER_DECISION.json` sha256 `bd84057928a49b48f101ca8f92e2d53437f493435956c50c1826b651527d648c`
- `v10/test_repeat134/OWNER_DECISION_V4.json` sha256 `19018ce893cfb2264c21d9fa9653b6112cfabb5652f455a22453342a607642ac`
- `v10/test_repeat134/TEST_REPEAT134_MANIFEST.json` sha256 `9dfda63905998413999743f2f7b7e5424903d5a32b78dd8f6f177ab308d1dcab`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL_V4.json` sha256 `8f6133c3b28bac449eabe4e01cee4860fddcc9753e318519d1358f6491a34505`
- `v10/test_repeat134/TEST_REPEAT134_AMENDMENT_V4.json` sha256 `531a11f8056e00b873a481726d6de001e987a88a0d7cf9ce407b03952e9a6efe`
- `v10/test_repeat134/TEST_REPEAT134_DELTA_REVIEW_V4.json` sha256 `30fc407b81a5c89d0476deff07ba8a88fc54ea645a234c649d9756a6761e3342`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL_V2.json` sha256 `31897806539f6d5b066599e0e41ef38833a29b3b976152ceac4e82462a8d79e7`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION_V2.json` sha256 `3aa948917e710daaffc0fbc56adaccfce580cfad1541fdcaf5e69f32cc8a6808`
- `v10/test_repeat134/TEST_REPEAT134_GATE_AMENDMENT_V2.json` sha256 `f64016098142bc94cf4c81d2290abb4bd7531878641b52f9589f5c8fc4f2a4e3`
- `v10/test_repeat134/TEST_REPEAT134_DELTA_REVIEW_V2.json` sha256 `7e6b3af7d1e71d400c98a9bf15ea90bd1912febaf204a6d489eb09fdca0fe6f0`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL.json` sha256 `cdcba1e7d578ab3524945e005c0a3214ac41c46712a32a4af430edb568d18506`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION.json` sha256 `5b46a0ae613dcd533a00a387e4d49b7b66f57aaf8000011ea45c6509fa5078d1`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION_V4.json` sha256 `b477870e323353561732d2bf359dd74bbfc498a2d96d4da1629f11928c3d1e80`
- TEST134 core 列表 sha256 `58681663df2d2f6ebdc7003f0338c9fec3ec5ae0cef2710f6daccb587c3e4ae2`；TEST112 敏感性集 sha256 `007cd25b46a5a8e0952039395aa5fed352fee341fa63f850201b2392d61a7b84`。先前暴露：V9-G1; reading began 2026-10-15T23:36:09Z。扰动：the frozen DEV rule, applied to the TEST core ids: Q.perturbations_p2(core) = REG-P1 translations (seed sha256('R6-REG-P1|<core>')[:16], magnitude strata) with theta uniform on [-30, 30] deg (seed sha256('R6-REG-P2|<core>')[:16]); per-registration position stream seed_of('R6-REG-P2|pos|<core>|<i>'); MLP extra stream seed_of('V10|MLP-extra-registration-perms|<core>|<i>')。V9-G1's sealed-test read used new seed tags (V9-G1-TEST-20261015|P1 / |P2); TEST_REPEAT134 uses V10's DEV rule unchanged, so the perturbations are not V9-G1's。
- 作业内检查：(a) identity, configuration (trainer rebuild and checkpoint load), checkpoint sha256, input hashes (DATA_LOCK), perturbation truths theta / t and n_grid are bitwise equal to the recorded unit, and every value is finite；(e) F_true is bitwise equal in all 320 rows (the frozen energy is reproduced)；(f) coarse gross-error band only: |delta count<42| <= 12, |delta count<9| <= 12 (of 320), |delta median err_full| <= 0.5。the band was set with the v1 / v2 gate re-evaluations in view (DEV only; no TEST result exists); the largest changes seen were |delta count<42| 2, |delta count<9| 7, |delta median| 0.110 over 9 re-evaluations

## 2 TEST 输入锁

pending（INPUT_LOCK.json 尚不存在）。

## 3 逐单元：TEST 与同作业 DEV32 并列，并附记录的 DEV32（不合并）

| 族 | seed | 层 | 标签 | GPU | TEST134 s<42 / s<9 / 中位 | TEST112 | DEV32（同作业重评） | DEV32（记录） | 作业 |
|---|---|---|---|---|---|---|---|---|---|
| MF_VCS_perm1 | 1001 | primary | frozen | L40S | pending | pending | pending | 0.9656 / 0.6031 / 7.013 |  |
| MF_VCS_perm1 | 1002 | primary | frozen | L40S | pending | pending | pending | 0.9438 / 0.6094 / 7.567 |  |
| MF_VCS_perm1 | 1003 | primary | frozen | L40S | pending | pending | pending | 0.9688 / 0.5813 / 7.074 |  |
| MF_VCS_M0 | 1001 | primary | frozen | L40S | pending | pending | pending | 0.9063 / 0.6313 / 5.891 |  |
| MF_VCS_M0 | 1002 | primary | frozen | L40S | pending | pending | pending | 0.9344 / 0.6250 / 6.679 |  |
| MF_VCS_M0 | 1003 | primary | frozen | L40S | pending | pending | pending | 0.9469 / 0.6344 / 6.927 |  |
| MF_VCS_M1 | 1001 | primary | frozen | L40S | pending | pending | pending | 0.9625 / 0.6438 / 5.230 |  |
| MF_VCS_M1 | 1002 | primary | frozen | L40S | pending | pending | pending | 0.9656 / 0.6469 / 5.452 |  |
| MF_VCS_M1 | 1003 | primary | frozen | L40S | pending | pending | pending | 0.9438 / 0.6313 / 6.126 |  |
| MLP_VCS_perm1 | 1001 | primary | frozen | L40S | pending | pending | pending | 0.9500 / 0.6531 / 5.826 |  |
| MLP_VCS_perm1 | 1002 | primary | frozen | L40S | pending | pending | pending | 0.9594 / 0.6531 / 5.715 |  |
| MLP_VCS_perm1 | 1003 | primary | frozen | L40S | pending | pending | pending | 0.9500 / 0.6313 / 5.104 |  |
| MLP_VCS_perm8 | 1001 | primary | frozen | L40S | pending | pending | pending | 0.9656 / 0.6875 / 5.651 |  |
| MLP_VCS_perm8 | 1002 | primary | frozen | L40S | pending | pending | pending | 0.9531 / 0.6813 / 5.622 |  |
| MLP_VCS_perm8 | 1003 | primary | frozen | L40S | pending | pending | pending | 0.9188 / 0.6719 / 5.666 |  |
| InfoNCE_host[J_perm1] | 1001 | primary | frozen | L40S | pending | pending | pending | 0.9719 / 0.6969 / 5.231 |  |
| InfoNCE_host[J_perm1] | 1002 | primary | frozen | L40S | pending | pending | pending | 0.9406 / 0.6813 / 6.606 |  |
| InfoNCE_host[J_perm1] | 1003 | primary | frozen | L40S | pending | pending | pending | 0.9563 / 0.6906 / 5.940 |  |
| MI_native | 1001 | primary | frozen | L40S | pending | pending | pending | 0.7406 / 0.4938 / 9.329 |  |
| NMI_native | 1001 | primary | frozen | L40S | pending | pending | pending | 0.7594 / 0.5063 / 8.936 |  |
| CoMIR_native | 1001 | primary | frozen | L40S | pending | pending | pending | 0.7406 / 0.5000 / 8.568 |  |
| CoMIR_native | 1002 | primary | frozen | L40S | pending | pending | pending | 0.8750 / 0.5656 / 6.531 |  |
| CoMIR_native | 1003 | primary | frozen | L40S | pending | pending | pending | 0.7781 / 0.5188 / 7.161 |  |
| RuLSIF_exact | 1001 | primary | failed_adaptation | L40S | pending | pending | pending | 0.0938 / 0.0344 / 272.970 |  |
| RuLSIF_exact | 1002 | primary | failed_adaptation | L40S | pending | pending | pending | 0.0969 / 0.0344 / 270.006 |  |
| RuLSIF_exact | 1003 | primary | failed_adaptation | L40S | pending | pending | pending | 0.0969 / 0.0406 / 267.803 |  |
| MIND_2D_adapted | 1001 | primary | frozen | L40S | pending | pending | pending | 0.7094 / 0.3500 / 15.054 |  |
| RuLSIF_exact_v2 | 1001 | primary | post_gate_amendment_A1 | RTX6000PRO | pending | pending | pending | 0.7875 / 0.5813 / 7.602 |  |
| RuLSIF_exact_v2 | 1002 | primary | post_gate_amendment_A1 | RTX6000PRO | pending | pending | pending | 0.7906 / 0.5906 / 7.443 |  |
| RuLSIF_exact_v2 | 1003 | primary | post_gate_amendment_A1 | L40S | pending | pending | pending | 0.7813 / 0.5719 / 7.868 |  |
| MF_VCS_perm1 | 1004 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9469 / 0.5531 / 7.982 |  |
| MF_VCS_perm1 | 1005 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9281 / 0.6281 / 6.827 |  |
| MF_VCS_M0 | 1004 | secondary | frozen | H100 | pending | pending | pending | 0.9281 / 0.5875 / 6.537 |  |
| MF_VCS_M0 | 1005 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9531 / 0.6469 / 6.541 |  |
| MF_VCS_M1 | 1004 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9688 / 0.6531 / 6.369 |  |
| MF_VCS_M1 | 1005 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9625 / 0.6469 / 6.447 |  |
| MLP_VCS_perm1 | 1004 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9656 / 0.6938 / 6.327 |  |
| MLP_VCS_perm1 | 1005 | secondary | frozen | H100 | pending | pending | pending | 0.9688 / 0.6719 / 5.635 |  |
| MLP_VCS_perm8 | 1004 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9500 / 0.6938 / 5.551 |  |
| MLP_VCS_perm8 | 1005 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9469 / 0.6656 / 5.776 |  |
| InfoNCE_host[J_perm1] | 1004 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9688 / 0.6531 / 6.545 |  |
| InfoNCE_host[J_perm1] | 1005 | secondary | frozen | RTX6000PRO | pending | pending | pending | 0.9688 / 0.6594 / 5.809 |  |

## 4 族的 seed 均值（主体 seeds 1001–1003）

| 族 | seed 数 | GPU | TEST134 | TEST112 | DEV32（同作业） | DEV32（记录） |
|---|---|---|---|---|---|---|
| MF_VCS_perm1 | 3 | L40S | pending | pending | pending | 0.9594 / 0.5979 / 7.218 |
| MF_VCS_M0 | 3 | L40S | pending | pending | pending | 0.9292 / 0.6302 / 6.499 |
| MF_VCS_M1 | 3 | L40S | pending | pending | pending | 0.9573 / 0.6406 / 5.603 |
| MLP_VCS_perm1 | 3 | L40S | pending | pending | pending | 0.9531 / 0.6458 / 5.548 |
| MLP_VCS_perm8 | 3 | L40S | pending | pending | pending | 0.9458 / 0.6802 / 5.646 |
| InfoNCE_host[J_perm1] | 3 | L40S | pending | pending | pending | 0.9563 / 0.6896 / 5.926 |
| MI_native | 1 | L40S | pending | pending | pending | 0.7406 / 0.4938 / 9.329 |
| NMI_native | 1 | L40S | pending | pending | pending | 0.7594 / 0.5063 / 8.936 |
| CoMIR_native | 3 | L40S | pending | pending | pending | 0.7979 / 0.5281 / 7.420 |
| RuLSIF_exact | 3 | L40S | pending | pending | pending | 0.0958 / 0.0365 / 270.260 |
| MIND_2D_adapted | 1 | L40S | pending | pending | pending | 0.7094 / 0.3500 / 15.054 |
| RuLSIF_exact_v2 | 3 | L40S，RTX6000PRO | pending | pending | pending | 0.7865 / 0.5813 / 7.638 |

## 5 以 core 为单位的配对差（success<42，A − B；描述性）

| 对照 | TEST134 [core 95%] · 位置组 | TEST112 | DEV32（同作业） | DEV32（记录，与第二阶段报告相同的键） |
|---|---|---|---|---|
| MF_VCS_M1 − MF_VCS_M0 | pending | pending | pending | +0.0281 [+0.0000, +0.0698] · 位置组 [+0.0000, +0.0698] |
| MF_VCS_M1 − MF_VCS_perm1 | pending | pending | pending | -0.0021 [-0.0313, +0.0250] · 位置组 [-0.0313, +0.0250] |
| MF_VCS_M1 − MLP_VCS_perm1 | pending | pending | pending | +0.0042 [-0.0219, +0.0344] · 位置组 [-0.0219, +0.0344] |
| MF_VCS_M1 − MLP_VCS_perm8 | pending | pending | pending | +0.0115 [-0.0156, +0.0500] · 位置组 [-0.0156, +0.0500] |
| MF_VCS_M1 − InfoNCE_host[J_perm1] | pending | pending | pending | +0.0010 [-0.0313, +0.0365] · 位置组 [-0.0313, +0.0365] |
| MF_VCS_M1 − CoMIR_native | pending | pending | pending | +0.1594 [+0.0750, +0.2594] · 位置组 [+0.0750, +0.2594] |
| MF_VCS_M1 − MI_native | pending | pending | pending | +0.2167 [+0.1031, +0.3438] · 位置组 [+0.1031, +0.3438] |
| MF_VCS_M1 − NMI_native | pending | pending | pending | +0.1979 [+0.0854, +0.3209] · 位置组 [+0.0854, +0.3209] |
| MF_VCS_M1 − MIND_2D_adapted | pending | pending | pending | +0.2479 [+0.1427, +0.3615] · 位置组 [+0.1427, +0.3615] |
| MF_VCS_M0 − MF_VCS_perm1（机制） | pending | pending | pending | -0.0302 [-0.0844, +0.0000] · 位置组 [-0.0844, +0.0000] |
| MLP_VCS_perm1 − MF_VCS_perm1（机制） | pending | pending | pending | -0.0063 [-0.0156, +0.0000] · 位置组 [-0.0156, +0.0000] |
| MLP_VCS_perm8 − MLP_VCS_perm1（机制） | pending | pending | pending | -0.0073 [-0.0198, +0.0000] · 位置组 [-0.0198, +0.0000] |
| InfoNCE_host[J_perm1] − MLP_VCS_perm1（机制） | pending | pending | pending | +0.0031 [-0.0021, +0.0104] · 位置组 [-0.0021, +0.0104] |

这些区间不是检验，不能读作优劣；RuLSIF_exact（v1）与 RuLSIF_exact_v2（A1）不进入配对。

## 6 评估噪声（同一冻结 checkpoint 的 DEV32 重评，对照记录的 DEV32）

评估本身有噪声：同一冻结 checkpoint 在同一 GPU 型号上重评，单次配准结果和成功计数都会变动。小于这一噪声的方法差，在 TEST 和 DEV 上都不可解释；这同样适用于 V10 的主 DEV 表。

| 族 | GPU | 来源 | 作业 | Δ<42 | Δ<9 | 翻转 <42 / <9 | Δ中位 | |Δerr| q50 | q90 | max | 逐位相同行 | F_true 逐位 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InfoNCE_host[J_perm1] | RTX6000PRO | gate (PASS_TOLERANCE) | 1051292 | +0 | +0 | 0 / 8 | +0.0045 | 0.075 | 0.672 | 42.975 | 7 | 是 |
| MF_VCS_M0 | L40S | gate (FAIL) | 1051030 | +0 | +1 | 0 / 3 | -0.0064 | 0.050 | 0.731 | 31.009 | 73 | 是 |
| MF_VCS_M0 | RTX6000PRO | gate (PASS_TOLERANCE) | 1051263 | -2 | -2 | 4 / 6 | +0.0857 | 0.097 | 1.910 | 63.706 | 7 | 是 |
| MF_VCS_M1 | L40S | gate (FAIL) | 1051031 | +0 | -2 | 0 / 2 | +0.0802 | 0.058 | 0.377 | 9.319 | 18 | 是 |
| MF_VCS_M1 | RTX6000PRO | gate (FAIL) | 1051266 | +0 | +4 | 0 / 6 | +0.0852 | 0.068 | 0.385 | 15.549 | 4 | 是 |
| MF_VCS_perm1 | L40S | gate (FAIL) | 1051029 | +0 | +4 | 0 / 8 | +0.1102 | 0.000 | 2.084 | 9.180 | 157 | 是 |
| MF_VCS_perm1 | RTX6000PRO | gate (FAIL) | 1051151 | +1 | +7 | 3 / 35 | +0.0187 | 1.326 | 4.866 | 92.310 | 8 | 是 |
| MLP_VCS_perm1 | L40S | gate (FAIL) | 1051032 | +0 | -1 | 0 / 5 | +0.0350 | 0.052 | 0.318 | 7.069 | 7 | 是 |
| MLP_VCS_perm1 | RTX6000PRO | gate (PASS_TOLERANCE) | 1051274 | +0 | -3 | 0 / 5 | +0.0695 | 0.041 | 0.270 | 24.229 | 8 | 是 |
| MLP_VCS_perm8 | RTX6000PRO | gate (PASS_TOLERANCE) | 1051277 | +0 | -2 | 0 / 2 | +0.0486 | 0.048 | 0.382 | 13.158 | 10 | 是 |
| RuLSIF_exact_v2 | RTX6000PRO | gate (PASS) | 1051295 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |

| 族 | GPU | 对数 | Δ<42 范围 | Δ<9 范围 | <9 翻转最大 | |Δ中位| 最大 | q50 中位 | q90 最大 | max | 逐位相同行范围 |
|---|---|---|---|---|---|---|---|---|---|---|
| InfoNCE_host[J_perm1] | RTX6000PRO | 1 | +0…+0 | +0…+0 | 8 | 0.0045 | 0.075 | 0.672 | 42.975 | 7…7 |
| MF_VCS_M0 | L40S | 1 | +0…+0 | +1…+1 | 3 | 0.0064 | 0.050 | 0.731 | 31.009 | 73…73 |
| MF_VCS_M0 | RTX6000PRO | 1 | -2…-2 | -2…-2 | 6 | 0.0857 | 0.097 | 1.910 | 63.706 | 7…7 |
| MF_VCS_M1 | L40S | 1 | +0…+0 | -2…-2 | 2 | 0.0802 | 0.058 | 0.377 | 9.319 | 18…18 |
| MF_VCS_M1 | RTX6000PRO | 1 | +0…+0 | +4…+4 | 6 | 0.0852 | 0.068 | 0.385 | 15.549 | 4…4 |
| MF_VCS_perm1 | L40S | 1 | +0…+0 | +4…+4 | 8 | 0.1102 | 0.000 | 2.084 | 9.180 | 157…157 |
| MF_VCS_perm1 | RTX6000PRO | 1 | +1…+1 | +7…+7 | 35 | 0.0187 | 1.326 | 4.866 | 92.310 | 8…8 |
| MLP_VCS_perm1 | L40S | 1 | +0…+0 | -1…-1 | 5 | 0.0350 | 0.052 | 0.318 | 7.069 | 7…7 |
| MLP_VCS_perm1 | RTX6000PRO | 1 | +0…+0 | -3…-3 | 5 | 0.0695 | 0.041 | 0.270 | 24.229 | 8…8 |
| MLP_VCS_perm8 | RTX6000PRO | 1 | +0…+0 | -2…-2 | 2 | 0.0486 | 0.048 | 0.382 | 13.158 | 10…10 |
| RuLSIF_exact_v2 | RTX6000PRO | 1 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |

## 7 成本

- 预估（recorded DEV seconds of the unit on its own GPU type x (1340 TEST + 320 DEV) / 320 + 2 x 180 s trainer rebuilds; native allocated GPU-h; prepare 0.25 h）：L40S 34.52，RTX6000PRO 9.21，H100 1.44 GPU-h；输入准备 0.25；合计 45.42 GPU-h。
- 实际（exit receipt 的已分配时间）：pending。

## 8 局限

- TEST134 已在 V9-G1 读过一次；这是重复基准，不是独立确认；不与 DEV 合并；不作方法优劣、强基线或新颖性结论（决定 §5）。
- 评估本身有噪声（第 6 节）：小于该噪声的方法差，在 TEST 和 DEV（包括 V10 主 DEV 表）上都不可解释。
- 每个单元在产生其 DEV32 的 GPU 型号上评估；不声称跨硬件等价。

