# R37C CC152K（2026-11-02 第三十四版）

接在第三十三版（`R37_RESULTS_20261102_CN.md`）之后。只发布数字表与代码，不附解读。

## 内容

`handoff/r37c_cc152k_20261102/`（开发实验，不读 TEST）：

- 父模型：R31 原生 IN²R `H0_default/cc_s1001`（最佳 epoch 38，DEV rSum 389.70），与 Flickr 的 R30-H 父模型不同系列（`PUBLIC_PARENT_BINDING.json`）。6 个单元共用同一冻结父状态。
- CC152K 映射与监督组件全部重建：原生 loader 的一图一文本身份（record k = image k = caption k）、DEV 1000×1000、全池 top-32 候选表（观测伙伴在 top-32 内：i2t 71.9%、t2i 68.7%）、父模型原生 GMM 可信子集（137,675 条，91.8%）、公共标定 a0 21.564 / b0 −4.323；无注入噪声、无真值。
- 6 项（固定 8000 步，相对父模型 389.70）：N-G J 392.30（+2.60）、BCE 393.10（+3.40）、NCE 387.50（−2.20）；T-G J 5.20、BCE 3.10、NCE 2.50。`RESULTS.md`、`RESULTS_FIXED.csv`、`CURVES.csv`、`BOOTSTRAP.csv`（按图像分组，描述性）、`NULL_READOUTS.csv`、`PARTICIPATION.csv`、`COST_LEDGER.csv`。
- 每单元 step 0 两阶段排名与原生排名一致（6/6）；cache 构建要求原生 DEV 精确复现（rSum 389.7，六项 Recall 一致；peer A 单独在 V100S 上 385.9 vs 日志 385.8，peer 平均读出精确一致）。独立 GPU 门禁按用户在 R37 的决定豁免。
- 代码（`r37c/`、`tests/`、`sbatch/`、`tools/`）、冻结协议、CPU 测试。
- TEST 汇总表与实验清单更新。

## 范围说明

单父模型、单 seed 评分头；配对差不给显著性结论；与 Flickr 结果分开报告，只比较相对各自父模型的增益。CC152K TEST 曾在 R31 读过；本线不读 TEST。设计包与作者代码副本不发布。
