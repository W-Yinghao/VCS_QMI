# NR6-A（2026-11-05 第五十版）

接在第四十九版（`CR5_RESULTS_20261105_CN.md`）之后。只发布数字表与代码，不附解读。用户：「那就push」。NR6-B、NR6A_P1105 / NR6B_P1105 与 CR6 仍在运行或准备中，本版不含。

## NR6-A：TAIL_LORA8 与 FROZEN 的 seed 确认（`handoff/nr6a_20261105/`，`RESULTS_NR6A.md` 与 CSV）

- 主会话按用户授权「下来这两个你都自己来分析分析结果，设计新一轮的方向，不要让GPU空着」设计（理由见 `PUBLIC_PROTOCOL_FROZEN_NR6A.json`）。
- 24 项 = {Flickr60, CC152K_RECOVERED} × J/BCE/NCE × {FROZEN, TAIL_LORA8} × 训练 seed 3003（12）+ COCO60 × J/BCE/NCE × {FROZEN, TAIL_LORA8} × seed {2002, 3003}（12）。配置同 NR5-A，只换 seed（训练 / sampler / adapter）与条件；自 NR4 同一 stage-0 前缀（folded NR3-B NCE），4000 步；只复用 NR4 前 10 层隐藏态缓存。
- 汇总按条件 × 目标给出三个 seed（1001 来自 NR4 TAIL / NR3-B FROZEN；2002 来自 NR5-A（Flickr60、CC）或 NR6-A（COCO60）；3003 来自 NR6-A）的 TAIL−FROZEN 与目标间差。
- 修订 NR5_A1 规则：每个比较组（条件 × seed）同一 GPU 型号，提交时按可用性选择（`PUBLIC_GROUP_GPU_NR6A.json`）。
- 全 DEV、基础模型 Top100、全部 query；不读 TEST。
- 代码：`code/nr6a/`、`sbatch/`。训练器沿用未发布的 NR5 训练器（不发布）。

## 范围说明

配对差不给显著性结论。Flickr、COCO、CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
