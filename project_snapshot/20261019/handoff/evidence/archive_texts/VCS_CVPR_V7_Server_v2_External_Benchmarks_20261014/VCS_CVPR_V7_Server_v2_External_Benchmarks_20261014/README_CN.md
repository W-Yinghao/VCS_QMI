# V7 · CS 神经估计的外部比较与图文选图应用
## 服务器交接包 v2 / 2026-10-14

**本轮改动：把传统 CS、非 CS 估计方法和公开视觉方法放回主要比较；Logistic 留作必要的归因对照，不再成为全篇唯一的成败判据。**

研究版本仍是 **V7**。应用仍是细粒度图文一致性估计与固定候选池 best-of-N 图像选择；原始 S/J、元素事件、数据角色、完整模型训练与已有结果都保留。本包不是把 V7 改成另一项任务，也不自动重开已关闭的 SSL。

### 三类主要证据
| 层次 | 对手 | 主要回答 |
|---|---|---|
| A：传统 CS 与相对密度比 | 原生 CS-KDE、共同 S 的 KDE、RuLSIF、核平方回归 | 相对既有 CS / 密度比路线，估计精度与成本推进了什么？ |
| B：非 CS 依赖估计 | MINE/DV、NWJ、InfoNCE、SMILE；适用检验中的 MMD/HSIC | 在各自正确的目标和共同任务上，相对替代工具有什么价值？ |
| C：完整图文应用 | FGA-BLIP2、VQAScore、TIFA、DSG、PN-VQA、可用的新评分器 | 相同候选池中，是否选出人工评价更好的图像？ |

之后用共享 L1 / Logistic / VCS、critic 和聚合消融解释贡献来源。**不删掉不利的 Logistic 结果，也不以“必须先赢 Logistic”为其他实验的准入门。**

### 先读
1. `00_V1_TO_V2_DELTA_CN.md`：增量与接续；已接入 v1 的服务器从这里开始。
2. `01_SERVER_AGENT_PROMPT_CN.md` 与 `docs/02_SERVER_PLAN_CN.md`：执行顺序。
3. `docs/10_EXTERNAL_BENCHMARK_PROTOCOL_CN.md`：原生目标、共同目标与任务指标怎样分表。
4. `docs/11_CLASSICAL_CS_IMPLEMENTATION_CN.md`、`docs/12_NONCS_AND_RATIO_ADAPTERS_CN.md`：可实现的数学接口。
5. `docs/13_EXTERNAL_SYSTEMS_CN.md`、`literature/EXTERNAL_BASELINES_REVIEW_CN.md`：公开方法与文献定位。

`INDEX.html` / `V7_FULL_PLAN_CN.html` 可离线阅读，公式保留 LaTeX 源式，不依赖网络脚本。`V7_OVERVIEW_CN.pdf` 是执行摘要；完整技术内容在 Markdown/HTML。

### 接入快照
本次读取到 `W-Yinghao/VCS_QMI@aeb56b5c7b3c238565dcf9a43804cca130cec9f4`（提交时间 2026-10-13 21:58:14 UTC）。最新提交在 `scripts/data_fetch.py` 登记了 V7 数据和初始权重；**这证明接入清单已写入，不证明下载、训练或评价已经完成。** [G09]

### 第一轮实际工作
保留 v1 的 12 个完整多模态训练单元及 9 个调参单元。并行完成外部评分器复现、旧估计器产物覆盖盘点，以及 V7 共同特征上的传统核 / KL 比较。缺失的旧实验先列明确差额和预算，再补；不重新运行整套历史网格。既有 V7、D1-FULL、E1/E2、r8 作业不自动取消、重启或改配置。

本文将 `[Fxx]` 标为用户原始材料，`[Pxx/Gxx/Txx/Xxx]` 标为文献或仓库来源；“v2设计”是本次建议。源码核验不等于自然数据复现，CPU公式测试不等于模型性能结果。
