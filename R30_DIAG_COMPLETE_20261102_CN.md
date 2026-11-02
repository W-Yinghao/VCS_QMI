# R30 诊断完成（2026-11-02 第二十八版）

接在第二十七版（`R30_COMPLETE_20261102_CN.md`）之后。只发布数字表与代码，不附解读。

## 内容

- `handoff/r30_diag_20261101/MARGIN_DIAG.md`、`PUBLIC_COVERAGE_V3.json`：R30 共同状态探针全部完成——7 个父状态、固定 E10 30/30、E40 12/12（三层更新、竞争者身份审计、固定参考清单 J/S/gap）。78MB 完整 JSON 不发布。
- `handoff/test_eval_20261101/TEST_TABLES.md`：R30 36/36 全部通过 DEV 复算门禁。60% seed1003 H→H 先在 V100-16GB 评测时门禁失败（i2t R@1 差 2 个 query）；在训练所用 GPU 类型（L40S）上只读 DEV 复核与训练日志一致（rSum 496.1 vs 496.07，六项 Recall 一致），随后在 L40S 上重做该单元 TEST 一次（DEV 复算差 0，TEST rSum 496.44，与 V100 值相同）；V100 结果作为被替代记录保留，不进表。
- 同表首次出现 R32（NRCS-v1，GPO 双塔）TEST 行：每单元 epoch-20 主 checkpoint 与 SELECT-J 次 checkpoint（读取 TEST 前登记）。0% seed1001 三个 ratio 臂（各两个 checkpoint，共 6 行）DEV 复算门禁失败（0.2–0.8pp，复算高于日志），未进表；只读 DEV 诊断（分数并列与两种并列规则的比较）在进行。
- TEST 增量与探针作业改用 normal（RTX6000PRO/L40S/H100）。
- `EXPERIMENT_LIST_R30_R33.md` 更新。

## 范围说明

DEV 在开发中被反复使用；TEST 只做选出 checkpoint 的一次性评测，不用于任何选择。设计包与包内源码不发布。
