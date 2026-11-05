# CR5 与 CR5_P1105（2026-11-05 第四十九版）

接在第四十八版（`NR5_CR4_COMPLETE_20261105_CN.md`）之后。只发布数字表与代码，不附解读。用户：「先push CR的内容吧」。NR6-A / NR6-B 仍在运行，本版不含。

## CR5：RECT 与 COMPLETE-PC 的 seed 确认（`handoff/cr5_20261105/`，`RESULTS_CR5.md` 与 CSV）

- 主会话按用户授权「下来这两个你都自己来分析分析结果，设计新一轮的方向，不要让GPU空着」设计（理由见 `PUBLIC_PROTOCOL_FROZEN_CR5.json`）。
- 48 格 = {Flickr60, CC152K_RECOVERED_20261104} × 6 张固定图（OBS、J_MASS_OBS、BASE / J / BCE / NCE_KEEP_TOP1）× {RECT, COMPLETE_PC} × seed {1002, 1003}；head seed = s，sampler seed = 20261104201 + (s − 1001)；seed 1001 取自 CR4（PC 单元与同组重跑的 RECT 参照）。图、候选、目标、校准与优化器冻结同 CR4，8000 步。
- 每个条件 × seed 一个组作业、同一 GPU；CC OBS / J_MASS_OBS 的 PC 与 RECT 等价（单正例）。
- 验收（L40S）：seed 1001 下新训练器与 CR3 RECT、CR4 PC 训练器逐位一致；同 seed 内 RECT 与 PC 边抽样相同；断点续跑逐位一致；CC OBS / J_MASS_OBS 单正例。
- 修订 CR5_A1（`PUBLIC_AMENDMENT_CR5_A1.json`）：用户「好，CR组加上V100-32GB」；V100-32GB 上重做验收（全部通过）后，仍排队的组作业分区加入 V100-32GB。实际 GPU：Flickr60 s1002 / s1003 与 CC s1002 为 L40S，CC s1003 为 V100S（`PUBLIC_GPU_ENV.json`）。组内 RECT / PC 同卡；跨组比较可能跨 GPU 型号。
- 全 DEV、基础模型 Top100、全部 query；不读 TEST。

## CR5_P1105：COMPLETE-PC 的 student seed 2002 / 2003（`handoff/cr5p_20261105/`，`RESULTS_CR5P.md` 与 CSV）

- 包 `VCS_NR5_CR4_Action_Server_20261105`（证据 e3971599）中的 CR5 部分；与上方 CR5 名称冲突且配置不等价，按包规则登记为 CR5_P1105（单元名映射 `PUBLIC_UNIT_ID_MAPPING.json`），不复用 CR5。
- 24 格 = {Flickr60, CC152K_RECOVERED_20261104} × 6 张 CR4 固定图 × COMPLETE_PC × student seed {2002, 2003}（sampler seed 202611052002 / 202611052003）；新 G 自条件共同 a、b 起，末层残差零初始化，CR4 优化器，8000 步；训练器为 CR5 的 COMPLETE_PC 路径（只读导入）。
- 汇总：每个 student seed（1001 CR4、1002 / 1003 CR5、2002 / 2003 CR5_P1105）分数据集给出 J_KEEP_TOP1 相对 OBS、J_MASS_OBS、BASE、BCE、NCE 的差，不跨数据集平均；每格实际曝光（不同边数、最大重复、被查询图像数、图像查询熵）由冻结 sampler 在 CPU 上回放并与训练边哈希收据核对（全部一致）。
- 修订 CR5_P1105_A1（`PUBLIC_AMENDMENT_CR5_P1105_A1.json`）：同上 V100-32GB；四个组作业均在 V100S。
- 代码：`code/site/`、`sbatch/`。设计包本身（`package/`）不发布。不读 TEST。

## R33 TEST 表

- `handoff/test_eval_20261101/TEST_TABLES.md` 按 R33 27/27 完成后的增量重新汇总。

## 范围说明

配对差不给显著性结论。Flickr 与 CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
