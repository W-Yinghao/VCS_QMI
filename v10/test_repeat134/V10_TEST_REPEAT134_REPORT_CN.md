# V10 TEST_REPEAT134：BF 重复基准（单独报告，不是独立确认；v4 同作业配对，v5 输入）

**状态：最终单元 42/42 完成，0 个 BLOCKED，其余 pending。** 本文由 `scripts/v10_test_repeat/analyze.py` 从冻结记录与已完成的作业输出生成；同一输入两次运行逐字节相同。

> **读法。** TEST_REPEAT134 (repeated benchmark; BF/SHG official test cores read once before by V9-G1; not an independent confirmation)。规则原文：reported as its own table, separate from DEV32; DEV32 / DEV24 shown beside it；DEV and TEST_REPEAT are never pooled (no 166-sample confirmation)；no independent-confirmation claim; prior exposure V9-G1 stated；TEST112 sensitivity reported beside TEST134；RuLSIF_exact (v1) carries its failed-adaptation label; RuLSIF_exact_v2 carries amendment A1's label；hardware recorded per unit; secondary replication by hardware；no method-superiority, strong-baseline or novelty claim while the required baselines are not confirmed complete (decision 5)；intervals are descriptive; no significance tests；TEST134 / TEST112 are reported beside the same-job DEV32 re-evaluation (paired reference) and beside the recorded DEV32；an evaluation-noise section reports every DEV re-evaluation pair; method differences smaller than this noise are not interpretable, on DEV as well as TEST (including the main V10 DEV tables)；BLOCKED units are reported with their failed criteria; TEST is never read for them；the TEST inputs follow v5 variant A; v4's INPUT_LOCK failure and the input diagnostic are reported。

## 1 负责人决定与冻结记录

- 启动（2026-10-20，UTC about 06:45）问题原文：「BF 上 V10 全部方法和 seed 的选择已经齐全，可以做 TEST_REPEAT134（读 held-out 的 TEST134）。现在开始吗？」选项：「开始 (Recommended)」——「先把所有方法和 seed 选中的 checkpoint 冻结成新 manifest，冻结完成前不读 TEST；然后在 TEST134 上评估。结果单独报告为重复测试，另附 112 个 core 的敏感性集，不当作独立确认。」
- v4（2026-10-20）问题原文：「评估本身有噪声（同一 checkpoint 重评，<9 变动 0.6–2.2 个百分点），TEST_REPEAT134 怎么继续？哪个方案都先登记、后读 TEST。」选项：「同作业配对，评一次 (Recommended)」——「每个单元在同一个作业里先重评 DEV32、再评 TEST134，TEST 和这次 DEV 并列报告。门禁改为：checkpoint、输入、扰动和真实位姿能量必须逐位一致，成功次数只粗查，用来拦流程错误。重评数据披露为噪声下限。三个分区都可以跑，约 50 GPU 小时。」未选方案：确定性评估（启用确定性算法 / cuDNN 确定性后重评 DEV32 与 TEST）；重复评估取平均（多次重评以平均评估噪声）。
- v6（2026-10-20）选项：「修复后补跑这 3 个单元 (Recommended)」——「在任何 CoMIR TEST 读取之前冻结 v6：只修这个检查，允许路径在固定 CoMIR 目录下的命名空间子包；模型、checkpoint、评估器和判定标准都不变。这 3 个单元仍在同一作业里先重评 DEV 再评 TEST，在 L40S 上跑。」适用单元：CoMIR_native__1001、CoMIR_native__1002、CoMIR_native__1003（在 results/TEST_REPEAT_V6 中重新同作业重评 DEV 再评 TEST）；其余 39 个单元仍属 v5。
- v5（2026-10-20）问题原文：「TEST_REPEAT 的输入锁卡在 1B_D8（另外 133 个 core 与 V9-G1 缓存逐位一致），最可能是 CPU 不同导致 numpy 走了不同的 SIMD 路径。继续需要读 1B_D8 的 TEST 原始文件做诊断（不做评估），并冻结 v5。怎么处理？」选项：「先诊断，按结果处理 (Recommended)」——「诊断确认是 CPU 路径造成的，就按 V9-G1 的 CPU 路径重新准备输入，判定标准不变，预计 134/134 通过。不是的话，就采用原始哈希已核对的 V10 重建版 1B_D8，另加一个排除它的 TEST133 敏感性集。」未选方案：直接排除 1B_D8；先停在这里。规则在诊断结果之前确定；按规则得到变体 A：A (CPU-dispatch artefact): REG-P1's frozen builder rebuilds all 134 cores into /projects/EEG-foundation-model/yinghao/outputs_v10/test_repeat134_20261020/inputs_v5 with numpy's float64 log1p / add kernels pinned to baseline(X86_V2) / X86_V3 (NPY_DISABLE_CPU_FEATURES='X86_V4'), the kernels of the diagnostic rebuild 'no_X86_V4' that reproduced V9-G1's 1B_D8 file bitwise; criterion unchanged (all 134 cores bitwise equal to V9-G1's prepared cache); TEST134, TEST112 and the 42 units unchanged
- `v10/test_repeat134/OWNER_DECISION.json` sha256 `bd84057928a49b48f101ca8f92e2d53437f493435956c50c1826b651527d648c`
- `v10/test_repeat134/OWNER_DECISION_V4.json` sha256 `19018ce893cfb2264c21d9fa9653b6112cfabb5652f455a22453342a607642ac`
- `v10/test_repeat134/OWNER_DECISION_V5.json` sha256 `6ab10af341e9463ab0cc670f63884d91046307ba1f3e7015afb0fc013898bd3f`
- `v10/test_repeat134/OWNER_DECISION_V6.json` sha256 `c9ce816cde131840bd8dcbe26158b287df891d293f2ffe4c2be9858efdc6179c`
- `v10/test_repeat134/TEST_REPEAT134_MANIFEST.json` sha256 `9dfda63905998413999743f2f7b7e5424903d5a32b78dd8f6f177ab308d1dcab`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL_V5.json` sha256 `c09df0c1280397980918ca3e9b7b4713c961998c849268d5aa8eefe16eaf1618`
- `v10/test_repeat134/TEST_REPEAT134_AMENDMENT_V5.json` sha256 `49457a987bf35f0a9d7c266faf5f8376d96f2c7b28e3e6e11dd4376f32ecc802`
- `v10/test_repeat134/TEST_REPEAT134_DELTA_REVIEW_V5.json` sha256 `0cafcab09dc7cadfb7ddc3de494ac7d4d4880e4eca977f2427bb89d9442b99c0`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION_V5.json` sha256 `c724c56b5ec37edddddf52dc1ff4f30d1dd2197c6735c947d63db6d148f2d926`
- `v10/test_repeat134/TEST_REPEAT134_AMENDMENT_V6.json` sha256 `3b46107c7a144547c7a23cf0a55f61e4b84bb18bc476031d7683433e79da91ea`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL_V4.json` sha256 `8f6133c3b28bac449eabe4e01cee4860fddcc9753e318519d1358f6491a34505`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION_V4.json` sha256 `b477870e323353561732d2bf359dd74bbfc498a2d96d4da1629f11928c3d1e80`
- `v10/test_repeat134/TEST_REPEAT134_AMENDMENT_V4.json` sha256 `531a11f8056e00b873a481726d6de001e987a88a0d7cf9ce407b03952e9a6efe`
- `v10/test_repeat134/TEST_REPEAT134_DELTA_REVIEW_V4.json` sha256 `30fc407b81a5c89d0476deff07ba8a88fc54ea645a234c649d9756a6761e3342`
- `v10/test_repeat134/DIAGNOSTIC_AUTHORIZATION_V4D.json` sha256 `fe4e6093dfbaf85ece802d620346ec66bc6e37972230e66dee373a0ba5ad660a`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL_V2.json` sha256 `31897806539f6d5b066599e0e41ef38833a29b3b976152ceac4e82462a8d79e7`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION_V2.json` sha256 `3aa948917e710daaffc0fbc56adaccfce580cfad1541fdcaf5e69f32cc8a6808`
- `v10/test_repeat134/TEST_REPEAT134_GATE_AMENDMENT_V2.json` sha256 `f64016098142bc94cf4c81d2290abb4bd7531878641b52f9589f5c8fc4f2a4e3`
- `v10/test_repeat134/TEST_REPEAT134_DELTA_REVIEW_V2.json` sha256 `7e6b3af7d1e71d400c98a9bf15ea90bd1912febaf204a6d489eb09fdca0fe6f0`
- `v10/test_repeat134/TEST_REPEAT134_PROTOCOL.json` sha256 `cdcba1e7d578ab3524945e005c0a3214ac41c46712a32a4af430edb568d18506`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION.json` sha256 `5b46a0ae613dcd533a00a387e4d49b7b66f57aaf8000011ea45c6509fa5078d1`
- `v10/test_repeat134/TEST_REPEAT134_ACTIVATION_V6.json` sha256 `74ddc5892b60bcf17e863f0bf610bbf9598d871d2627f9a149405e47897397fa`
- TEST134 core 列表 sha256 `58681663df2d2f6ebdc7003f0338c9fec3ec5ae0cef2710f6daccb587c3e4ae2`；TEST112 敏感性集 sha256 `007cd25b46a5a8e0952039395aa5fed352fee341fa63f850201b2392d61a7b84`。先前暴露：V9-G1; reading began 2026-10-15T23:36:09Z。扰动：the frozen DEV rule, applied to the TEST core ids: Q.perturbations_p2(core) = REG-P1 translations (seed sha256('R6-REG-P1|<core>')[:16], magnitude strata) with theta uniform on [-30, 30] deg (seed sha256('R6-REG-P2|<core>')[:16]); per-registration position stream seed_of('R6-REG-P2|pos|<core>|<i>'); MLP extra stream seed_of('V10|MLP-extra-registration-perms|<core>|<i>')。V9-G1's sealed-test read used new seed tags (V9-G1-TEST-20261015|P1 / |P2); TEST_REPEAT134 uses V10's DEV rule unchanged, so the perturbations are not V9-G1's。
- 作业内检查：(a) identity, configuration (trainer rebuild and checkpoint load), checkpoint sha256, input hashes (DATA_LOCK), perturbation truths theta / t and n_grid are bitwise equal to the recorded unit, and every value is finite；(e) F_true is bitwise equal in all 320 rows (the frozen energy is reproduced)；(f) coarse gross-error band only: |delta count<42| <= 12, |delta count<9| <= 12 (of 320), |delta median err_full| <= 0.5。the band was set with the v1 / v2 gate re-evaluations in view (DEV only; no TEST result exists); the largest changes seen were |delta count<42| 2, |delta count<9| 7, |delta median| 0.110 over 9 re-evaluations

## 2 TEST 输入

- v4 输入锁（作业 1051404）：FAIL_V9_CACHE_MISMATCH；268 个原始成员 sha256 核对；133/134 个 core 与 V9-G1 预处理缓存逐位相同，不同：1B_D8。该锁按原样保留。
- 输入诊断（作业 1051581，nodecpu01；只有数字，不评估）：原始成员与全部清单一致 是；两次重建逐位相同 是；与 V9-G1 不同的数组：shg_log_n 2 个元素（最大 604353734 ULP，最大绝对差 2.9e-17）；与 V9-G1 文件逐位相同的重建：no_X86_V4、baseline_X86_V2；与 V10 文件逐位相同的重建：default_1、default_2；V9-G1 文件在其预处理时间窗内 是。
- v5 变体 A：A (CPU-dispatch artefact): REG-P1's frozen builder rebuilds all 134 cores into /projects/EEG-foundation-model/yinghao/outputs_v10/test_repeat134_20261020/inputs_v5 with numpy's float64 log1p / add kernels pinned to baseline(X86_V2) / X86_V3 (NPY_DISABLE_CPU_FEATURES='X86_V4'), the kernels of the diagnostic rebuild 'no_X86_V4' that reproduced V9-G1's 1B_D8 file bitwise; criterion unchanged (all 134 cores bitwise equal to V9-G1's prepared cache); TEST134, TEST112 and the 42 units unchanged
- v5 输入锁：状态 PASS（作业 1051656，nodecpu01；268 个原始成员 sha256 核对；与 V9-G1 预处理缓存不同的 core：无）。

## 3 逐单元：TEST 与同作业 DEV32 并列，并附记录的 DEV32（不合并）

| 族 | seed | 层 | 标签 | GPU | TEST134 s<42 / s<9 / 中位 | TEST112 | DEV32（同作业重评） | DEV32（记录） | 作业 |
|---|---|---|---|---|---|---|---|---|---|
| MF_VCS_perm1 | 1001 | primary | frozen | L40S | 0.9851 / 0.7075 / 5.959 | 0.9821 / 0.6982 / 5.950 | 0.9656 / 0.6250 / 6.897 | 0.9656 / 0.6031 / 7.013 | 1051674 |
| MF_VCS_perm1 | 1002 | primary | frozen | L40S | 0.9784 / 0.6948 / 6.338 | 0.9741 / 0.6938 / 6.300 | 0.9438 / 0.6156 / 7.654 | 0.9438 / 0.6094 / 7.567 | 1051675 |
| MF_VCS_perm1 | 1003 | primary | frozen | L40S | 0.9828 / 0.7306 / 5.954 | 0.9795 / 0.7313 / 5.954 | 0.9688 / 0.5906 / 7.145 | 0.9688 / 0.5813 / 7.074 | 1051676 |
| MF_VCS_M0 | 1001 | primary | frozen | L40S | 0.9746 / 0.7164 / 5.341 | 0.9696 / 0.7125 / 5.497 | 0.9063 / 0.6313 / 5.772 | 0.9063 / 0.6313 / 5.891 | 1051677 |
| MF_VCS_M0 | 1002 | primary | frozen | L40S | 0.9784 / 0.7343 / 5.087 | 0.9741 / 0.7268 / 5.079 | 0.9344 / 0.6250 / 6.588 | 0.9344 / 0.6250 / 6.679 | 1051678 |
| MF_VCS_M0 | 1003 | primary | frozen | L40S | 0.9791 / 0.6948 / 6.001 | 0.9750 / 0.7045 / 6.010 | 0.9469 / 0.6406 / 6.919 | 0.9469 / 0.6344 / 6.927 | 1051679 |
| MF_VCS_M1 | 1001 | primary | frozen | L40S | 0.9918 / 0.7560 / 4.835 | 0.9902 / 0.7500 / 4.738 | 0.9625 / 0.6438 / 5.195 | 0.9625 / 0.6438 / 5.230 | 1051680 |
| MF_VCS_M1 | 1002 | primary | frozen | L40S | 0.9776 / 0.7373 / 5.247 | 0.9732 / 0.7321 / 5.278 | 0.9656 / 0.6563 / 5.432 | 0.9656 / 0.6469 / 5.452 | 1051681 |
| MF_VCS_M1 | 1003 | primary | frozen | L40S | 0.9828 / 0.7463 / 5.276 | 0.9795 / 0.7420 / 5.289 | 0.9438 / 0.6281 / 6.122 | 0.9438 / 0.6313 / 6.126 | 1051682 |
| MLP_VCS_perm1 | 1001 | primary | frozen | L40S | 0.9851 / 0.7724 / 4.923 | 0.9821 / 0.7670 / 4.888 | 0.9500 / 0.6531 / 5.842 | 0.9500 / 0.6531 / 5.826 | 1051785 |
| MLP_VCS_perm1 | 1002 | primary | frozen | L40S | 0.9799 / 0.7567 / 4.835 | 0.9759 / 0.7554 / 4.839 | 0.9594 / 0.6656 / 5.729 | 0.9594 / 0.6531 / 5.715 | 1051788 |
| MLP_VCS_perm1 | 1003 | primary | frozen | L40S | 0.9866 / 0.7269 / 4.404 | 0.9839 / 0.7188 / 4.345 | 0.9500 / 0.6344 / 5.062 | 0.9500 / 0.6313 / 5.104 | 1051789 |
| MLP_VCS_perm8 | 1001 | primary | frozen | L40S | 0.9813 / 0.7590 / 5.190 | 0.9777 / 0.7625 / 5.207 | 0.9656 / 0.6969 / 5.645 | 0.9656 / 0.6875 / 5.651 | 1051790 |
| MLP_VCS_perm8 | 1002 | primary | frozen | L40S | 0.9821 / 0.7672 / 4.864 | 0.9786 / 0.7634 / 4.917 | 0.9531 / 0.6813 / 5.668 | 0.9531 / 0.6813 / 5.622 | 1051793 |
| MLP_VCS_perm8 | 1003 | primary | frozen | L40S | 0.9821 / 0.7761 / 4.429 | 0.9786 / 0.7741 / 4.451 | 0.9188 / 0.6844 / 5.674 | 0.9188 / 0.6719 / 5.666 | 1051823 |
| InfoNCE_host[J_perm1] | 1001 | primary | frozen | L40S | 0.9784 / 0.7776 / 4.316 | 0.9741 / 0.7759 / 4.278 | 0.9719 / 0.6781 / 5.291 | 0.9719 / 0.6969 / 5.231 | 1051824 |
| InfoNCE_host[J_perm1] | 1002 | primary | frozen | L40S | 0.9828 / 0.7642 / 5.163 | 0.9795 / 0.7670 / 5.163 | 0.9406 / 0.6813 / 6.439 | 0.9406 / 0.6813 / 6.606 | 1051826 |
| InfoNCE_host[J_perm1] | 1003 | primary | frozen | L40S | 0.9784 / 0.7433 / 5.462 | 0.9741 / 0.7464 / 5.412 | 0.9594 / 0.6969 / 6.020 | 0.9563 / 0.6906 / 5.940 | 1051829 |
| MI_native | 1001 | primary | frozen | L40S | 0.7769 / 0.5739 / 6.715 | 0.7902 / 0.5821 / 6.639 | 0.7406 / 0.4938 / 9.329 | 0.7406 / 0.4938 / 9.329 | 1051830 |
| NMI_native | 1001 | primary | frozen | L40S | 0.7910 / 0.5739 / 6.816 | 0.7991 / 0.5768 / 6.632 | 0.7594 / 0.5063 / 8.936 | 0.7594 / 0.5063 / 8.936 | 1051904 |
| CoMIR_native | 1001 | primary | frozen | L40S | 0.7664 / 0.5806 / 6.319 | 0.7759 / 0.5821 / 6.018 | 0.7406 / 0.5000 / 8.568 | 0.7406 / 0.5000 / 8.568 | 1052306（v6） |
| CoMIR_native | 1002 | primary | frozen | L40S | 0.8955 / 0.6858 / 5.674 | 0.8929 / 0.6946 / 5.674 | 0.8750 / 0.5656 / 6.531 | 0.8750 / 0.5656 / 6.531 | 1052307（v6） |
| CoMIR_native | 1003 | primary | frozen | L40S | 0.8179 / 0.6537 / 5.334 | 0.8179 / 0.6598 / 5.331 | 0.7781 / 0.5188 / 7.161 | 0.7781 / 0.5188 / 7.161 | 1052313（v6） |
| RuLSIF_exact | 1001 | primary | failed_adaptation | L40S | 0.0366 / 0.0216 / 292.826 | 0.0411 / 0.0259 / 292.826 | 0.0938 / 0.0344 / 272.970 | 0.0938 / 0.0344 / 272.970 | 1051990 |
| RuLSIF_exact | 1002 | primary | failed_adaptation | L40S | 0.0396 / 0.0224 / 291.536 | 0.0446 / 0.0250 / 290.859 | 0.0969 / 0.0344 / 270.006 | 0.0969 / 0.0344 / 270.006 | 1052032 |
| RuLSIF_exact | 1003 | primary | failed_adaptation | L40S | 0.0381 / 0.0209 / 291.408 | 0.0429 / 0.0241 / 291.821 | 0.0969 / 0.0406 / 267.803 | 0.0969 / 0.0406 / 267.803 | 1052043 |
| MIND_2D_adapted | 1001 | primary | frozen | L40S | 0.7657 / 0.3336 / 15.369 | 0.7679 / 0.3268 / 15.451 | 0.7094 / 0.3500 / 15.054 | 0.7094 / 0.3500 / 15.054 | 1052044 |
| RuLSIF_exact_v2 | 1001 | primary | post_gate_amendment_A1 | RTX6000PRO | 0.7716 / 0.6164 / 6.139 | 0.7571 / 0.6107 / 6.487 | 0.7875 / 0.5813 / 7.602 | 0.7875 / 0.5813 / 7.602 | 1052651 |
| RuLSIF_exact_v2 | 1002 | primary | post_gate_amendment_A1 | RTX6000PRO | 0.7664 / 0.6112 / 6.432 | 0.7536 / 0.6045 / 6.648 | 0.7906 / 0.5906 / 7.443 | 0.7906 / 0.5906 / 7.443 | 1051755 |
| RuLSIF_exact_v2 | 1003 | primary | post_gate_amendment_A1 | L40S | 0.7687 / 0.6082 / 6.498 | 0.7545 / 0.5982 / 6.694 | 0.7813 / 0.5719 / 7.868 | 0.7813 / 0.5719 / 7.868 | 1052053 |
| MF_VCS_perm1 | 1004 | secondary | frozen | RTX6000PRO | 0.9769 / 0.6888 / 6.275 | 0.9723 / 0.6911 / 6.275 | 0.9531 / 0.5813 / 7.737 | 0.9469 / 0.5531 / 7.982 | 1051749 |
| MF_VCS_perm1 | 1005 | secondary | frozen | RTX6000PRO | 0.9716 / 0.6821 / 5.982 | 0.9661 / 0.6813 / 5.971 | 0.9313 / 0.6125 / 6.459 | 0.9281 / 0.6281 / 6.827 | 1051747 |
| MF_VCS_M0 | 1004 | secondary | frozen | H100 | 0.9664 / 0.6873 / 5.799 | 0.9598 / 0.6741 / 5.888 | 0.9313 / 0.5844 / 6.558 | 0.9281 / 0.5875 / 6.537 | 1051733 |
| MF_VCS_M0 | 1005 | secondary | frozen | RTX6000PRO | 0.9821 / 0.7082 / 5.839 | 0.9786 / 0.7080 / 5.954 | 0.9469 / 0.6375 / 6.601 | 0.9531 / 0.6469 / 6.541 | 1051723 |
| MF_VCS_M1 | 1004 | secondary | frozen | RTX6000PRO | 0.9806 / 0.7507 / 5.316 | 0.9768 / 0.7473 / 5.367 | 0.9688 / 0.6531 / 6.346 | 0.9688 / 0.6531 / 6.369 | 1051721 |
| MF_VCS_M1 | 1005 | secondary | frozen | RTX6000PRO | 0.9791 / 0.7172 / 5.899 | 0.9750 / 0.7170 / 6.035 | 0.9625 / 0.6469 / 6.523 | 0.9625 / 0.6469 / 6.447 | 1051719 |
| MLP_VCS_perm1 | 1004 | secondary | frozen | RTX6000PRO | 0.9888 / 0.7881 / 5.151 | 0.9866 / 0.7911 / 5.148 | 0.9656 / 0.6875 / 6.302 | 0.9656 / 0.6938 / 6.327 | 1051712 |
| MLP_VCS_perm1 | 1005 | secondary | frozen | H100 | 0.9896 / 0.7515 / 4.890 | 0.9875 / 0.7518 / 4.865 | 0.9688 / 0.6594 / 5.610 | 0.9688 / 0.6719 / 5.635 | 1051711 |
| MLP_VCS_perm8 | 1004 | secondary | frozen | RTX6000PRO | 0.9813 / 0.7687 / 4.450 | 0.9777 / 0.7652 / 4.315 | 0.9500 / 0.6906 / 5.593 | 0.9500 / 0.6938 / 5.551 | 1051705 |
| MLP_VCS_perm8 | 1005 | secondary | frozen | RTX6000PRO | 0.9866 / 0.7851 / 4.965 | 0.9839 / 0.7848 / 4.881 | 0.9469 / 0.6688 / 5.723 | 0.9469 / 0.6656 / 5.776 | 1051690 |
| InfoNCE_host[J_perm1] | 1004 | secondary | frozen | RTX6000PRO | 0.9925 / 0.7634 / 5.713 | 0.9911 / 0.7768 / 5.744 | 0.9688 / 0.6656 / 6.445 | 0.9688 / 0.6531 / 6.545 | 1051688 |
| InfoNCE_host[J_perm1] | 1005 | secondary | frozen | RTX6000PRO | 0.9776 / 0.7873 / 5.040 | 0.9732 / 0.7830 / 4.945 | 0.9688 / 0.6688 / 5.677 | 0.9688 / 0.6594 / 5.809 | 1051684 |

## 4 族的 seed 均值（主体 seeds 1001–1003）

| 族 | seed 数 | GPU | TEST134 | TEST112 | DEV32（同作业） | DEV32（记录） |
|---|---|---|---|---|---|---|
| MF_VCS_perm1 | 3 | L40S | 0.9821 / 0.7109 / 6.083 | 0.9786 / 0.7077 / 6.068 | 0.9594 / 0.6104 / 7.232 | 0.9594 / 0.5979 / 7.218 |
| MF_VCS_M0 | 3 | L40S | 0.9774 / 0.7152 / 5.477 | 0.9729 / 0.7146 / 5.529 | 0.9292 / 0.6323 / 6.426 | 0.9292 / 0.6302 / 6.499 |
| MF_VCS_M1 | 3 | L40S | 0.9841 / 0.7465 / 5.119 | 0.9810 / 0.7414 / 5.102 | 0.9573 / 0.6427 / 5.583 | 0.9573 / 0.6406 / 5.603 |
| MLP_VCS_perm1 | 3 | L40S | 0.9838 / 0.7520 / 4.721 | 0.9807 / 0.7470 / 4.691 | 0.9531 / 0.6510 / 5.544 | 0.9531 / 0.6458 / 5.548 |
| MLP_VCS_perm8 | 3 | L40S | 0.9818 / 0.7674 / 4.828 | 0.9783 / 0.7667 / 4.858 | 0.9458 / 0.6875 / 5.662 | 0.9458 / 0.6802 / 5.646 |
| InfoNCE_host[J_perm1] | 3 | L40S | 0.9799 / 0.7617 / 4.980 | 0.9759 / 0.7631 / 4.951 | 0.9573 / 0.6854 / 5.916 | 0.9563 / 0.6896 / 5.926 |
| MI_native | 1 | L40S | 0.7769 / 0.5739 / 6.715 | 0.7902 / 0.5821 / 6.639 | 0.7406 / 0.4938 / 9.329 | 0.7406 / 0.4938 / 9.329 |
| NMI_native | 1 | L40S | 0.7910 / 0.5739 / 6.816 | 0.7991 / 0.5768 / 6.632 | 0.7594 / 0.5063 / 8.936 | 0.7594 / 0.5063 / 8.936 |
| CoMIR_native | 3 | L40S | 0.8266 / 0.6400 / 5.776 | 0.8289 / 0.6455 / 5.674 | 0.7979 / 0.5281 / 7.420 | 0.7979 / 0.5281 / 7.420 |
| RuLSIF_exact | 3 | L40S | 0.0381 / 0.0216 / 291.923 | 0.0429 / 0.0250 / 291.835 | 0.0958 / 0.0365 / 270.260 | 0.0958 / 0.0365 / 270.260 |
| MIND_2D_adapted | 1 | L40S | 0.7657 / 0.3336 / 15.369 | 0.7679 / 0.3268 / 15.451 | 0.7094 / 0.3500 / 15.054 | 0.7094 / 0.3500 / 15.054 |
| RuLSIF_exact_v2 | 3 | L40S，RTX6000PRO | 0.7689 / 0.6119 / 6.356 | 0.7551 / 0.6045 / 6.610 | 0.7865 / 0.5813 / 7.638 | 0.7865 / 0.5813 / 7.638 |

## 5 以 core 为单位的配对差（success<42，A − B；描述性）

| 对照 | TEST134 [core 95%] · 位置组 | TEST112 | DEV32（同作业） | DEV32（记录，与第二阶段报告相同的键） |
|---|---|---|---|---|
| MF_VCS_M1 − MF_VCS_M0 | +0.0067 [-0.0027, +0.0216] · 位置组 [-0.0027, +0.0214] | +0.0080 [-0.0033, +0.0247] · 位置组 [-0.0031, +0.0251] | +0.0281 [+0.0000, +0.0698] · 位置组 [+0.0000, +0.0698] | +0.0281 [+0.0000, +0.0698] · 位置组 [+0.0000, +0.0698] |
| MF_VCS_M1 − MF_VCS_perm1 | +0.0020 [-0.0025, +0.0082] · 位置组 [-0.0025, +0.0083] | +0.0024 [-0.0030, +0.0098] · 位置组 [-0.0029, +0.0100] | -0.0021 [-0.0313, +0.0250] · 位置组 [-0.0313, +0.0250] | -0.0021 [-0.0313, +0.0250] · 位置组 [-0.0313, +0.0250] |
| MF_VCS_M1 − MLP_VCS_perm1 | +0.0002 [-0.0010, +0.0022] · 位置组 [-0.0010, +0.0021] | +0.0003 [-0.0012, +0.0024] · 位置组 [-0.0012, +0.0026] | +0.0042 [-0.0219, +0.0344] · 位置组 [-0.0219, +0.0344] | +0.0042 [-0.0219, +0.0344] · 位置组 [-0.0219, +0.0344] |
| MF_VCS_M1 − MLP_VCS_perm8 | +0.0022 [-0.0005, +0.0072] · 位置组 [-0.0005, +0.0073] | +0.0027 [-0.0006, +0.0086] · 位置组 [-0.0006, +0.0087] | +0.0115 [-0.0156, +0.0500] · 位置组 [-0.0156, +0.0500] | +0.0115 [-0.0156, +0.0500] · 位置组 [-0.0156, +0.0500] |
| MF_VCS_M1 − InfoNCE_host[J_perm1] | +0.0042 [-0.0002, +0.0109] · 位置组 [-0.0003, +0.0110] | +0.0051 [-0.0003, +0.0134] · 位置组 [-0.0003, +0.0132] | +0.0000 [-0.0313, +0.0333] · 位置组 [-0.0313, +0.0333] | +0.0010 [-0.0313, +0.0365] · 位置组 [-0.0313, +0.0365] |
| MF_VCS_M1 − CoMIR_native | +0.1575 [+0.1124, +0.2052] · 位置组 [+0.1130, +0.2057] | +0.1521 [+0.1039, +0.2045] · 位置组 [+0.1042, +0.2037] | +0.1594 [+0.0740, +0.2563] · 位置组 [+0.0740, +0.2563] | +0.1594 [+0.0750, +0.2594] · 位置组 [+0.0750, +0.2594] |
| MF_VCS_M1 − MI_native | +0.2072 [+0.1493, +0.2692] · 位置组 [+0.1506, +0.2691] | +0.1908 [+0.1289, +0.2595] · 位置组 [+0.1299, +0.2574] | +0.2167 [+0.1000, +0.3490] · 位置组 [+0.1000, +0.3490] | +0.2167 [+0.1031, +0.3438] · 位置组 [+0.1031, +0.3438] |
| MF_VCS_M1 − NMI_native | +0.1930 [+0.1368, +0.2532] · 位置组 [+0.1371, +0.2522] | +0.1818 [+0.1214, +0.2455] · 位置组 [+0.1229, +0.2448] | +0.1979 [+0.0875, +0.3281] · 位置组 [+0.0875, +0.3281] | +0.1979 [+0.0854, +0.3209] · 位置组 [+0.0854, +0.3209] |
| MF_VCS_M1 − MIND_2D_adapted | +0.2184 [+0.1756, +0.2642] · 位置组 [+0.1757, +0.2629] | +0.2131 [+0.1661, +0.2622] · 位置组 [+0.1683, +0.2603] | +0.2479 [+0.1417, +0.3625] · 位置组 [+0.1417, +0.3625] | +0.2479 [+0.1427, +0.3615] · 位置组 [+0.1427, +0.3615] |
| MF_VCS_M0 − MF_VCS_perm1（机制） | -0.0047 [-0.0172, +0.0022] · 位置组 [-0.0173, +0.0024] | -0.0057 [-0.0205, +0.0030] · 位置组 [-0.0208, +0.0029] | -0.0302 [-0.0865, +0.0000] · 位置组 [-0.0865, +0.0000] | -0.0302 [-0.0844, +0.0000] · 位置组 [-0.0844, +0.0000] |
| MLP_VCS_perm1 − MF_VCS_perm1（机制） | +0.0017 [-0.0015, +0.0060] · 位置组 [-0.0015, +0.0063] | +0.0021 [-0.0018, +0.0074] · 位置组 [-0.0018, +0.0075] | -0.0063 [-0.0156, +0.0000] · 位置组 [-0.0156, +0.0000] | -0.0063 [-0.0156, +0.0000] · 位置组 [-0.0156, +0.0000] |
| MLP_VCS_perm8 − MLP_VCS_perm1（机制） | -0.0020 [-0.0055, +0.0000] · 位置组 [-0.0055, +0.0000] | -0.0024 [-0.0065, +0.0000] · 位置组 [-0.0067, +0.0000] | -0.0073 [-0.0198, +0.0000] · 位置组 [-0.0198, +0.0000] | -0.0073 [-0.0198, +0.0000] · 位置组 [-0.0198, +0.0000] |
| InfoNCE_host[J_perm1] − MLP_VCS_perm1（机制） | -0.0040 [-0.0100, +0.0000] · 位置组 [-0.0099, +0.0000] | -0.0048 [-0.0116, +0.0000] · 位置组 [-0.0117, +0.0000] | +0.0042 [+0.0000, +0.0115] · 位置组 [+0.0000, +0.0115] | +0.0031 [-0.0021, +0.0104] · 位置组 [-0.0021, +0.0104] |

这些区间不是检验，不能读作优劣；RuLSIF_exact（v1）与 RuLSIF_exact_v2（A1）不进入配对。

## 6 同一 checkpoint 重评的观察变动（DEV32 重评，对照记录的 DEV32）

冻结协议的原文：评估本身有噪声：同一冻结 checkpoint 在同一 GPU 型号上重评，单次配准结果和成功计数都会变动。小于这一噪声的方法差，在 TEST 和 DEV 上都不可解释；这同样适用于 V10 的主 DEV 表。

**更正（2026-10-20 审查包 V10 第 4 节）**：下表是有限几次重评的观察值，不是已标定的噪声分布、噪声下限、等价界或显著性阈值；不能据此把较小的方法差判为不可解释，也不能把较大的差判为显著。同一阈值下的双向翻转和净变化要一起看，两个阈值分开解释；固定 checkpoint 的重评不是新的训练 seed。逐配准审计见 v10/review-actions-20261020 的 v10/review_20261020/repeat_with_v5/。

| 族 | GPU | 来源 | 作业 | Δ<42 | Δ<9 | 翻转 <42 / <9 | Δ中位 | |Δerr| q50 | q90 | max | 逐位相同行 | F_true 逐位 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CoMIR_native | L40S | in-job (v4) | 1052306 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| CoMIR_native | L40S | in-job (v4) | 1052307 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| CoMIR_native | L40S | in-job (v4) | 1052313 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| InfoNCE_host[J_perm1] | L40S | in-job (v4) | 1051824 | +0 | -6 | 0 / 6 | +0.0602 | 0.042 | 0.203 | 11.353 | 32 | 是 |
| InfoNCE_host[J_perm1] | L40S | in-job (v4) | 1051826 | +0 | +0 | 0 / 0 | -0.1677 | 0.043 | 0.204 | 8.236 | 29 | 是 |
| InfoNCE_host[J_perm1] | L40S | in-job (v4) | 1051829 | +1 | +2 | 1 / 6 | +0.0798 | 0.059 | 0.367 | 74.330 | 35 | 是 |
| InfoNCE_host[J_perm1] | RTX6000PRO | gate (PASS_TOLERANCE) | 1051292 | +0 | +0 | 0 / 8 | +0.0045 | 0.075 | 0.672 | 42.975 | 7 | 是 |
| InfoNCE_host[J_perm1] | RTX6000PRO | in-job (v4) | 1051684 | +0 | +3 | 0 / 7 | -0.1329 | 0.064 | 0.786 | 5.983 | 7 | 是 |
| InfoNCE_host[J_perm1] | RTX6000PRO | in-job (v4) | 1051688 | +0 | +4 | 0 / 6 | -0.1006 | 0.076 | 0.576 | 45.692 | 11 | 是 |
| MF_VCS_M0 | H100 | in-job (v4) | 1051733 | +1 | -1 | 1 / 7 | +0.0211 | 0.086 | 1.449 | 106.610 | 5 | 是 |
| MF_VCS_M0 | L40S | gate (FAIL) | 1051030 | +0 | +1 | 0 / 3 | -0.0064 | 0.050 | 0.731 | 31.009 | 73 | 是 |
| MF_VCS_M0 | L40S | in-job (v4) | 1051677 | +0 | +0 | 0 / 2 | -0.1189 | 0.039 | 0.730 | 34.069 | 83 | 是 |
| MF_VCS_M0 | L40S | in-job (v4) | 1051678 | +0 | +0 | 0 / 4 | -0.0905 | 0.015 | 0.808 | 53.260 | 115 | 是 |
| MF_VCS_M0 | L40S | in-job (v4) | 1051679 | +0 | +2 | 0 / 6 | -0.0086 | 0.061 | 0.727 | 46.478 | 73 | 是 |
| MF_VCS_M0 | RTX6000PRO | gate (PASS_TOLERANCE) | 1051263 | -2 | -2 | 4 / 6 | +0.0857 | 0.097 | 1.910 | 63.706 | 7 | 是 |
| MF_VCS_M0 | RTX6000PRO | in-job (v4) | 1051723 | -2 | -3 | 2 / 3 | +0.0599 | 0.098 | 1.632 | 262.151 | 5 | 是 |
| MF_VCS_M1 | L40S | gate (FAIL) | 1051031 | +0 | -2 | 0 / 2 | +0.0802 | 0.058 | 0.377 | 9.319 | 18 | 是 |
| MF_VCS_M1 | L40S | in-job (v4) | 1051680 | +0 | +0 | 0 / 2 | -0.0349 | 0.058 | 0.293 | 12.331 | 18 | 是 |
| MF_VCS_M1 | L40S | in-job (v4) | 1051681 | +0 | +3 | 0 / 3 | -0.0199 | 0.082 | 0.620 | 98.709 | 7 | 是 |
| MF_VCS_M1 | L40S | in-job (v4) | 1051682 | +0 | -1 | 0 / 1 | -0.0042 | 0.055 | 0.209 | 2.661 | 9 | 是 |
| MF_VCS_M1 | RTX6000PRO | gate (FAIL) | 1051266 | +0 | +4 | 0 / 6 | +0.0852 | 0.068 | 0.385 | 15.549 | 4 | 是 |
| MF_VCS_M1 | RTX6000PRO | in-job (v4) | 1051719 | +0 | +0 | 0 / 2 | +0.0758 | 0.067 | 0.719 | 12.866 | 5 | 是 |
| MF_VCS_M1 | RTX6000PRO | in-job (v4) | 1051721 | +0 | +0 | 0 / 4 | -0.0229 | 0.073 | 0.499 | 13.619 | 3 | 是 |
| MF_VCS_perm1 | L40S | gate (FAIL) | 1051029 | +0 | +4 | 0 / 8 | +0.1102 | 0.000 | 2.084 | 9.180 | 157 | 是 |
| MF_VCS_perm1 | L40S | in-job (v4) | 1051674 | +0 | +7 | 0 / 9 | -0.1167 | 0.014 | 2.546 | 13.064 | 152 | 是 |
| MF_VCS_perm1 | L40S | in-job (v4) | 1051675 | +0 | +2 | 0 / 16 | +0.0868 | 0.092 | 1.198 | 359.768 | 31 | 是 |
| MF_VCS_perm1 | L40S | in-job (v4) | 1051676 | +0 | +3 | 0 / 7 | +0.0714 | 0.070 | 1.591 | 288.116 | 23 | 是 |
| MF_VCS_perm1 | RTX6000PRO | gate (FAIL) | 1051151 | +1 | +7 | 3 / 35 | +0.0187 | 1.326 | 4.866 | 92.310 | 8 | 是 |
| MF_VCS_perm1 | RTX6000PRO | in-job (v4) | 1051747 | +1 | -5 | 1 / 25 | -0.3679 | 1.284 | 5.118 | 166.506 | 8 | 是 |
| MF_VCS_perm1 | RTX6000PRO | in-job (v4) | 1051749 | +2 | +9 | 2 / 41 | -0.2454 | 1.590 | 5.697 | 41.266 | 5 | 是 |
| MIND_2D_adapted | L40S | in-job (v4) | 1052044 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| MI_native | L40S | in-job (v4) | 1051830 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| MLP_VCS_perm1 | H100 | in-job (v4) | 1051711 | +0 | -4 | 0 / 6 | -0.0242 | 0.058 | 0.432 | 8.820 | 4 | 是 |
| MLP_VCS_perm1 | L40S | gate (FAIL) | 1051032 | +0 | -1 | 0 / 5 | +0.0350 | 0.052 | 0.318 | 7.069 | 7 | 是 |
| MLP_VCS_perm1 | L40S | in-job (v4) | 1051785 | +0 | +0 | 0 / 6 | +0.0161 | 0.054 | 0.495 | 30.163 | 7 | 是 |
| MLP_VCS_perm1 | L40S | in-job (v4) | 1051788 | +0 | +4 | 0 / 4 | +0.0133 | 0.054 | 0.225 | 6.585 | 7 | 是 |
| MLP_VCS_perm1 | L40S | in-job (v4) | 1051789 | +0 | +1 | 0 / 1 | -0.0412 | 0.046 | 0.226 | 5.983 | 9 | 是 |
| MLP_VCS_perm1 | RTX6000PRO | gate (PASS_TOLERANCE) | 1051274 | +0 | -3 | 0 / 5 | +0.0695 | 0.041 | 0.270 | 24.229 | 8 | 是 |
| MLP_VCS_perm1 | RTX6000PRO | in-job (v4) | 1051712 | +0 | -2 | 0 / 8 | -0.0252 | 0.048 | 0.254 | 4.281 | 5 | 是 |
| MLP_VCS_perm8 | L40S | in-job (v4) | 1051790 | +0 | +3 | 0 / 5 | -0.0057 | 0.054 | 0.302 | 24.107 | 8 | 是 |
| MLP_VCS_perm8 | L40S | in-job (v4) | 1051793 | +0 | +0 | 0 / 6 | +0.0460 | 0.052 | 0.239 | 16.437 | 6 | 是 |
| MLP_VCS_perm8 | L40S | in-job (v4) | 1051823 | +0 | +4 | 0 / 6 | +0.0074 | 0.044 | 0.167 | 8.856 | 10 | 是 |
| MLP_VCS_perm8 | RTX6000PRO | gate (PASS_TOLERANCE) | 1051277 | +0 | -2 | 0 / 2 | +0.0486 | 0.048 | 0.382 | 13.158 | 10 | 是 |
| MLP_VCS_perm8 | RTX6000PRO | in-job (v4) | 1051690 | +0 | +1 | 0 / 7 | -0.0530 | 0.054 | 1.223 | 195.903 | 4 | 是 |
| MLP_VCS_perm8 | RTX6000PRO | in-job (v4) | 1051705 | +0 | -1 | 0 / 3 | +0.0428 | 0.054 | 0.242 | 13.618 | 10 | 是 |
| NMI_native | L40S | in-job (v4) | 1051904 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact | L40S | in-job (v4) | 1051990 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact | L40S | in-job (v4) | 1052032 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact | L40S | in-job (v4) | 1052043 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact_v2 | L40S | in-job (v4) | 1052053 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact_v2 | RTX6000PRO | gate (PASS) | 1051295 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact_v2 | RTX6000PRO | in-job (v4) | 1051755 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |
| RuLSIF_exact_v2 | RTX6000PRO | in-job (v4) | 1052651 | +0 | +0 | 0 / 0 | +0.0000 | 0.000 | 0.000 | 0.000 | 320 | 是 |

| 族 | GPU | 对数 | Δ<42 范围 | Δ<9 范围 | <9 翻转最大 | |Δ中位| 最大 | q50 中位 | q90 最大 | max | 逐位相同行范围 |
|---|---|---|---|---|---|---|---|---|---|---|
| CoMIR_native | L40S | 3 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |
| InfoNCE_host[J_perm1] | L40S | 3 | +0…+1 | -6…+2 | 6 | 0.1677 | 0.043 | 0.367 | 74.330 | 29…35 |
| InfoNCE_host[J_perm1] | RTX6000PRO | 3 | +0…+0 | +0…+4 | 8 | 0.1329 | 0.075 | 0.786 | 45.692 | 7…11 |
| MF_VCS_M0 | H100 | 1 | +1…+1 | -1…-1 | 7 | 0.0211 | 0.086 | 1.449 | 106.610 | 5…5 |
| MF_VCS_M0 | L40S | 4 | +0…+0 | +0…+2 | 6 | 0.1189 | 0.045 | 0.808 | 53.260 | 73…115 |
| MF_VCS_M0 | RTX6000PRO | 2 | -2…-2 | -3…-2 | 6 | 0.0857 | 0.097 | 1.910 | 262.151 | 5…7 |
| MF_VCS_M1 | L40S | 4 | +0…+0 | -2…+3 | 3 | 0.0802 | 0.058 | 0.620 | 98.709 | 7…18 |
| MF_VCS_M1 | RTX6000PRO | 3 | +0…+0 | +0…+4 | 6 | 0.0852 | 0.068 | 0.719 | 15.549 | 3…5 |
| MF_VCS_perm1 | L40S | 4 | +0…+0 | +2…+7 | 16 | 0.1167 | 0.042 | 2.546 | 359.768 | 23…157 |
| MF_VCS_perm1 | RTX6000PRO | 3 | +1…+2 | -5…+9 | 41 | 0.3679 | 1.326 | 5.697 | 166.506 | 5…8 |
| MIND_2D_adapted | L40S | 1 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |
| MI_native | L40S | 1 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |
| MLP_VCS_perm1 | H100 | 1 | +0…+0 | -4…-4 | 6 | 0.0242 | 0.058 | 0.432 | 8.820 | 4…4 |
| MLP_VCS_perm1 | L40S | 4 | +0…+0 | -1…+4 | 6 | 0.0412 | 0.053 | 0.495 | 30.163 | 7…9 |
| MLP_VCS_perm1 | RTX6000PRO | 2 | +0…+0 | -3…-2 | 8 | 0.0695 | 0.045 | 0.270 | 24.229 | 5…8 |
| MLP_VCS_perm8 | L40S | 3 | +0…+0 | +0…+4 | 6 | 0.0460 | 0.052 | 0.302 | 24.107 | 6…10 |
| MLP_VCS_perm8 | RTX6000PRO | 3 | +0…+0 | -2…+1 | 7 | 0.0530 | 0.054 | 1.223 | 195.903 | 4…10 |
| NMI_native | L40S | 1 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |
| RuLSIF_exact | L40S | 3 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |
| RuLSIF_exact_v2 | L40S | 1 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |
| RuLSIF_exact_v2 | RTX6000PRO | 3 | +0…+0 | +0…+0 | 0 | 0.0000 | 0.000 | 0.000 | 0.000 | 320…320 |

## 7 成本

- 预估（recorded DEV seconds of the unit on its own GPU type x (1340 TEST + 320 DEV) / 320 + 2 x 180 s trainer rebuilds; native allocated GPU-h; the v5 inputs job runs on CPU）：L40S 34.52，RTX6000PRO 9.21，H100 1.44 GPU-h；输入准备 0.00；合计 45.17 GPU-h。
- 实际（exit receipt 的已分配时间）：H100 1.266，L40S 29.953，RTX6000PRO 10.522。

## 8 局限

- TEST134 已在 V9-G1 读过一次；这是重复基准，不是独立确认；不与 DEV 合并；不作方法优劣、强基线或新颖性结论（决定 §5）。
- 同一 checkpoint 重评时，单次配准和成功计数会变动（第 6 节）。观察到的变动只是描述，不作为判定方法差能否解释的阈值（审查包 2026-10-20）。
- 每个单元在产生其 DEV32 的 GPU 型号上评估；不声称跨硬件等价。
- 输入：输入按 V9-G1 的 CPU 路径重建（numpy float64 log1p / add 内核固定为 baseline(X86_V2) / X86_V3），134 个 core 与 V9-G1 预处理缓存逐位相同才放行；v4 的输入（1B_D8 不同）不使用。

