# 交付前增量核对：54e1715 → 3e2fe59

**最后核对：`3e2fe59c4861abae46537574c4c59f9bedfe91c6`，2026-10-13 18:51:49 UTC。**

正文源码审查基线仍为 `54e1715`，保留证据的版本边界。准备材料期间又出现 4 个提交；本补充记录它们对接续任务的影响。

## 当前实际差异

- R7-50 将 timing gate 放在 manifest 前，尝试提前启动 phase F 的 integrity 工作；随后发现 integrity-prepare 仍需要全部 reference shard markers，提前启动在 CPU 准备步骤退出，没有 GPU 单元/新测试文件读取。流程恢复 marker-triggered 接续。
- D1-FULL 的 ChartQA shard 1 重试允许 L40S/H100；新提交给后续提交的 eval wrapper 设置 `expandable_segments:True`。旧已提交 Slurm 脚本不会自动继承这个变量；该重试若再次 OOM，继续保留已有逐项预测。
- 提交说明把分配器修改定为执行配置，不改变分析或方法。它不提供新的完整全量性能结果。本包没有实测该分配器是否消除 OOM。
- **两个重点 critic 文件的 blob 在 3e2fe59 均与基线相同**：LLaVA `706a464…`，Qwen `9a68d1c…`。因此 dropout/mask/Spearman 审查仍指向同一源码，不是重复审查已修代码。

## 给服务器的影响

接入时以实际 job / marker 状态为准，不再直接照正文较早的“H100 重试”安排重新提交。phase F 当前依赖完整参考数据；不取消或复制已授权的接续流程。

审查辅助脚本认可两个已核对 HEAD（54e1715 与 3e2fe59），仍要求关键文件 blob 匹配；更晚版本需要显式 `--allow-newer` 并报告差异。

## 来源

- [比较记录（4 commits）](https://github.com/W-Yinghao/VCS_QMI/compare/54e1715e29dd2caaa6be7ed8e670829310951d39...3e2fe59c4861abae46537574c4c59f9bedfe91c6)
- [phase-F 提前启动与 reference guard](https://github.com/W-Yinghao/VCS_QMI/commit/86b5bc259ba56e01ab5451ad6620adbc6be34075)
- [恢复 marker-triggered 接续](https://github.com/W-Yinghao/VCS_QMI/commit/eaeb94a011d6cbd062d39abee71b2814a1f3a1bc)
- [ChartQA 重试与 allocator 记录](https://github.com/W-Yinghao/VCS_QMI/commit/3e2fe59c4861abae46537574c4c59f9bedfe91c6)

Git commit 时间与文内手写时刻略有差别，本包截点使用 Git commit UTC，不据手写时刻推断额外执行事件。
