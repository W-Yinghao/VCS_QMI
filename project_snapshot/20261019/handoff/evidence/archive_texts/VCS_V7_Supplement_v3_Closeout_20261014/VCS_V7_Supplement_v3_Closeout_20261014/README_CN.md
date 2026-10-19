# V7 补充与收尾包 v3
## 完成外部竞争评价，拆开估计、聚合与选择三个问题

**本包不更换 V7 方法，不重选已冻结的主输出，也不把它改写成新一轮成功故事。**它补齐当前尚缺的实验与分析，形成可以进入论文的完整结论。下一轮方法另见独立 V8 包。

建议先读 `01_SERVER_AGENT_PROMPT_CN.md` → `docs/02_TASK_PLAN_CN.md` → `docs/03_DIAGNOSTICS_CN.md` → `docs/04_EXTERNAL_BENCHMARK_CN.md`。`FULL_PLAN_CN.html` 是聚合离线阅读版；`V7_SUPPLEMENT_OVERVIEW_CN.pdf` 是概览。

| 本包必须完成 | 复用内容 | 新训练 |
|---|---|---|
| V7外部主家族、标准N=1/3/9选图与成本 | 已有完整模型和全部冻结预测 | 不默认重训 |
| 元素标签→总体评分的对应检查 | EvalMuse CAL/DEV原始投票、E/O/H预测 | 无 |
| 同提示排序、最大错误候选与解析分解 | 既有checkpoint重新导出必要字段 | 无 |
| 传统CS/密度比与非CS比较的完整收束 | X11/X21/PCA/N0特征敏感性 | 只补缺失/失败单元 |
| 二次读数、外部系统、归因三套结论 | 原v1/v2协议与结果 | 不以BCE胜负控制其他面板 |

已有记录表明完整M2和M1未分离；共同特征的KDE较弱而Nyström/RuLSIF并非都弱。这些是本轮起点，不是要通过删基线消除的问题。[G05,G07,G08]

本包包含本地数学/聚合参考实现，不包含完整GPU训练器、数据、权重或新的自然数据结果。引用键对应 `literature/SOURCES_CN.md`。

交付前增量与最新双分支快照见 `docs/00_DELIVERY_UPDATE_CN.md`；不替换旧冻结协议。
