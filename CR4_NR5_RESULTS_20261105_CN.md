# CR4 与 NR5（部分）（2026-11-05 第四十七版）

接在第四十六版（`CR3_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。用户：「你来push一版，CR4和NR5」。

## CR4：固定 CR3 图上的正例互斥学生损失（`handoff/cr4_20261104/`，`RESULTS_CR4.md` 与 CSV）

- 包 `VCS_CR3_Review_CR4_Server_20261104`（协议 CR4_FIXED_GRAPH_POSITIVE_EXCLUSIVE_V1，证据 e156e2f1）。冻结 CR3 的 16 张图；零新估计器、零新图、零编码器训练；复用 CR3 FixedRelation、同 64 条边抽样、COMPLETE 候选并集、完整 targets 与 query 权重，只把最终条件损失换为正例互斥损失 PC（每条正边只与零图权重候选比较）。kind='CE' 时逐位复现 CR3 COMPLETE。
- 矩阵：14 个新训练 + 2 个恒等检查（CC OBS / J_MASS_OBS 单正例，PC 与 CE 等价）。条件 Flickr00、Flickr60、CC152K_RECOVERED_20261104；seed 1001、8000 步、读数 0/250/500/1000/2000/4000/8000；全 DEV、基础模型 Top100、全部 query；不读 TEST。
- 每个条件一个组作业、同一 GPU 型号（CR3_A1 原则）；组作业未落在与 CR3 相同的 GPU / 驱动上时，在同一作业内用未改动的 CR3 代码重跑该条件的 RECT 与 COMPLETE-CE 参照（`PUBLIC_REFERENCE_RECEIPTS.json`）。
- 验收：包 CPU 测试 29/29；真实小批次验收 P1–P8 通过。修订 CR4_A1（`PUBLIC_AMENDMENT_CR4_A1.json`）：首次三个组作业在设置阶段因组运行脚本中多余的字典键查找而失败（未训练，记录保留），新组运行脚本只删去该行，冻结协议与科学设置不变。
- 训练前只读审计（CE 分解与梯度分量）已完成；训练后审计在本版构建时尚在排队，本版不含其结果。
- 代码：`code/site/`、`sbatch/`。设计包本身（`package/`）与其字节副本 `code/pkg/` 不发布。

## NR5（部分，26/30 单元；`handoff/nr5_20261104/`，`RESULTS_NR5.md` 与 CSV）

- 包 `VCS_NR4_Review_NR5_Server_20261104`（证据 e2e1509b）。D0：NR4 12 个 TAIL 单元固定 stage 4000 的只读分解。A（12）：Flickr60、CC152K_RECOVERED × J/BCE/NCE × {FROZEN, TAIL_LORA8}，自 NR4 同一 stage-0 前缀，只换随机性（训练 seed 2002、sampler 102002、adapter 2002），4000 步。B（18）：Flickr00 / Flickr60 / CC152K_RECOVERED × {COS, G_ZERO} × J/BCE/NCE，自原始 SigLIP2 + 零 LoRA，不读 NR2/NR3 任务头或 NR4 适配权重，P = 观测联合、Q = 乘积边缘，8000 步。
- 修订 NR5_A1（`PUBLIC_NR5_A1.json`、`PUBLIC_NR5_A1_ADDENDUM_1.json`）：用户「至少我们要能在L40S上跑」；允许 RTX / L40S / H100，每个比较组同一 GPU 型号，非 RTX 上 t0 改为登记容差。
- 本版构建时 CC152K_RECOVERED 的 B 组 4 个单元（COS NCE、G_ZERO J / BCE / NCE）尚未完成，表中列为缺失；完成后在下一版补齐。不读 TEST。
- 代码：`code/nr5/`、`code/nr5a1/`、`sbatch/`。设计包本身、与包参考相同的 `__init__.py`、改写自未发布训练器的 `train.py` 与实现 diff 不发布；`code/nr5a1/check_frozen_a1.py` 被修订文件哈希绑定且运行中的单元在用、含行尾空格，本版不发布（不改动原文件）。

## 范围说明

单 seed（NR5-A 为第二 seed）；配对差不给显著性结论。Flickr、COCO、CC152K_RECOVERED 分开报告。ORACLE 图只作诊断。DEV 在开发中被反复使用。
