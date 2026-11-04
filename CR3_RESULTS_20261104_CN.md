# CR3：固定关系图的完整监督传递（2026-11-04 第四十六版）

接在第四十五版（`CR2_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。NR5 仍在运行，本版不含。

## CR3（`handoff/cr3_20261104/`，`RESULTS_CR3.md` 与 CSV）

- 用户：给出 `VCS_CR2_Review_CR3_Server_20261104` 包（协议 CR3_FIXED_GRAPH_SUPPORT_COMPLETE_V1，证据 e08cdf50）；「cr3先push」。
- 固定图：OBS、CR2 的 BASE / J / BCE / NCE KEEP_TOP1、已登记 KEEP_ORACLE；新增 J_MASS_OBS（行质量与 J_CAP 相同）与隔离的 FULL_TRUE_ORACLE（仅 Flickr60，独立进程）。不重训关系估计器，不改候选、图权重或温度。
- 两种 consumer 同图、同 64 条 P_R 边抽样（边曝光哈希一致）、同初值与优化器：RECT 原样；COMPLETE 为每个原 query 加入其在固定图中全部正权重邻居、保留原矩形负候选，补入候选不升级为新 query，目标在完整邻居上归一化。seed 1001、sampler 20261104201、8000 步、CR1 G 与共同校准、soft-target NCE。32 格，条件 Flickr00、Flickr60、CC152K_RECOVERED_20261104。
- 先导出 CR2 审计中尚未公开的支持交叉表、raw / 校准排序接缝与专家分数尺度（逐字取自已有审计 JSON，`cr2_audit_export/`）；无真值的图→batch 传递审计（`TRANSFER_STATS.csv`）。
- 验收：包内单元测试 26/26；真实数据验收 A1–A7（RECT 与 CR1/CR2 soft-NCE 等价、同抽样、COMPLETE 目标与邻居完整、score 不变性、两正例 0.5/0.5、断点恢复）先在 RTX PRO 6000、后在 L40S 各通过一次。12 个历史 RECT 单元在锁定 GPU 上 250 步逐位核对均不一致，全部重跑。
- 修订 CR3_A1（`PUBLIC_AMENDMENT_CR3_A1.json`）：用户「至少我们要能在L40S上跑」；GPU 锁由全局 RTX 改为每个条件一组、在同一张卡上顺序运行（实际三组均在 L40S、同一驱动 / torch / CUDA）；原冻结协议未改；原 RTX 上的 Flickr00 OBS RECT 结果留档。
- 成本：每次更新的 pair 数、每个 query 事件的 pair 数、缓存读取、同步训练时间与显存（`RESOURCE_LEDGER.csv`）。全 DEV、基础模型 Top100、全部 query；Flickr 真值只在隔离评价器与两张 ORACLE 图中；CC 无 oracle；不读 TEST。
- 代码：`code/site/`、`sbatch/`。设计包本身（`package/`）与其字节副本 `code/pkg/` 不发布。

## 范围说明

单 seed；配对差不给显著性结论。ORACLE 图只作诊断，不作部署方法、不称上界。Flickr 与 CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
