# R30 完成（2026-11-02 第二十七版）

接在第二十六版（`R30_R33_LIST_20261102_CN.md`）之后。分析点：R30 v1.1 + 修订 A1 全部 36 项完成，并全部做了 TEST 一次性评测。只发布数字表与代码，不附解读。

## 内容

- `handoff/r30_pairwise_20261101/R30_STAGE_E10.md`：22 条阶段路径在固定 completed_cotrain_epochs=10 的 full-DEV rSum、E0/E1/E2/E5、E10−E0、累计更新数，及同格配对比较（60% 三 seed 正式均值：HL−HN −15.83、HL−HB −2.18、LN−LL +40.22、HN−LN +8.47、HL−HH −24.62、交互 I +24.40）。
- `R30_FULL40.md`：全部 24 个 40 轮单元的 40 轮内 best full-DEV rSum 与同 epoch 六项 Recall、final、last5，及同格配对差——Ls−L0、PV−Ls、PV−PN、PV−PB、PN/PV/PB/L0/Ls/HN/HB−H0、L0−HN、L0−HB、60% seed1002/1003 的 PV−PN、PV−PB、PN/PV/PB−HH。`PUBLIC_COEFFICIENTS.json`：四格静态系数。`PUBLIC_TRIGGER_EVAL.json`：修订 A1 的两对反馈干预触发均不成立，干预未运行。
- `handoff/test_eval_20261101/TEST_TABLES.md`：R30 全部 36 单元的一次性 TEST（Flickr30K 1K），35 个通过 DEV 复算门禁；60% seed1003 H→H 在 V100-16GB 复算 DEV 时 i2t R@1 差 2 个 query（0.197pp），不进入表格，训练所用 GPU 类型（L40S）上的只读 DEV 复核在进行。`handoff/r30_r33_list_20261102/EXPERIMENT_LIST_R30_R33.md`：逐单元 DEV 与 TEST 并列。
- TEST 工具增加 `nrcs` 宿主（R32 GPO 双塔，epoch-20 主 / SELECT-J 次 checkpoint，读取 TEST 前登记），并修正一处：对没有该宿主单元的旧 manifest 跳过其评测代码核对。
- `handoff/r30_diag_20261101/MARGIN_DIAG.md`、`PUBLIC_COVERAGE_V3.json`：共同状态探针覆盖（固定 E10 27/30、E40 4/12）；其余 E40 状态的探针已排队，完成后随下一版发布。

## 范围说明

单 seed（seed1001）为主，60% 另有 seed1002/1003 的 PN/PV/PB/HH 配对确认；配对差不给显著性结论。DEV 在开发中被反复使用；TEST 只做选出 checkpoint 的一次性评测，不用于任何选择。设计包与包内源码不发布。
