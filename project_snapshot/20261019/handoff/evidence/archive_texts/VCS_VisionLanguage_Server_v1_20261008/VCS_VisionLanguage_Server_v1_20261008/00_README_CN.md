# VCS Vision–Language：服务器交接包 v1

**日期：2026-10-08。论文仍为V6；本包编号独立。**

本轮把之前的vision–language讨论变成一套可实施的候选方案：区域—短语主任务、18个数据/任务条目、12类失败诊断、5条有限备用路线。默认只启动一个主数据集，不是18套数据的全网格。

## 文件

- `01_SERVER_AGENT_PROMPT_CN.md`：发给服务器agent的接入消息。
- `docs/02_SERVER_PLAN_CN.md`：完整任务、数学对象、模型、对照、选择与评价。
- `docs/03_DATASET_ATLAS_CN.md`：数据入口、用途、划分、去重与获取限制。
- `docs/04_FALLBACKS_AND_DECISIONS_CN.md`：失败时先查什么、改变什么、如何保留原结论。
- `docs/05_SOURCES_AND_RELATED_WORK_CN.md`：原计划、历史结果与公开文献/数据来源。
- `contracts/task_manifest.yaml`：逻辑任务与参数建议，不是可直接运行的仓库trainer配置。
- `templates/`：数据/结果/决定记录模板。
- `reference/vl_pair_reference.py`、`tests/`：数学与权重参考实现，非完整VL训练器。
- `checks/`：本地测试与有限模型数值核对。
- `VCS_VisionLanguage_Server_Plan_v1_CN.pdf`：主任务书＋源索引阅读版；MD为服务器编辑来源。

## 第一轮只做清楚的事情

先读取已有结果与数据库存，确认Flickr实体数据可用，建立图像级划分，验证P/Q/多正例/均匀区域先验，再拟合小型评分器。默认36个结果单元是四条路线×三个训练图像数×三个初始化，不是36个大模型预训练；超参数候选开销另计。轻量adapter首轮建议9个单元，需实际预算确认。

## 本地检查

在材料包根目录：

```bash
python -m unittest discover -s tests -v
python reference/export_math_checks.py
```

依赖NumPy、PyTorch。所有检查使用人工有限分布与CPU张量；不需要GPU、图片、模型权重或网络。它们不会连接GitHub、提交训练、改动项目仓库或删除文件。

## 版本和边界

本次核对main为`6f30f9b5cf85b8af1b4780a24c2103824d96c3cb`（2026-10-08 11:10:11 UTC），针对性重读P80并沿用已有P72/P79记录。公共数据/文献入口于同日核查。没有完整审计最新仓库所有文件，没有下载新数据、复评checkpoint或运行新VL实验。

合作者原始S/J保持不变；本轮的条件候选采样、统一scorer、数据网格及备用路线是新建议，不冒充原理论直接指定的最优方案。
