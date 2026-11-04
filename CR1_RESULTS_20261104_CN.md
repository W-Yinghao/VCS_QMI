# CR1：对应关系重建——独立并行检索路线（2026-11-04 第四十三版）

接在第四十二版（`NR3_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。NR4 仍在运行，本版不含。

## CR1（`handoff/cr1_20261104/`，`RESULTS_CR1.md` 与 CSV）

- 用户：「准备这个包，并行进行」（`VCS_CR1_Server_20261104`，协议 CR1_20261104_V1）；「还是先push CR系列的内容吧」。
- 链路：观测联合 P 与乘积边缘 Q → VCS(J) / BCE / NCE 关系估计 → 固定的部分软匹配（ROW：仅行约束；CAP：行 + 容量约束）→ 统一 soft-target NCE 检索 student。条件 Flickr00、Flickr60、CC152K_RECOVERED_20261104；冻结 SigLIP2 内容缓存（NR1 锁定），seed 1001。
- 54 个关系估计 fit：3 条件 × J/BCE/NCE × 6 个两端留出专家（内容哈希四折；被重建 pair 的两端都不在该专家的训练与校准中），各 8000 步。12 个交叉拟合评分导出、18 个无梯度匹配求解（KKT 收据）。21 个下游 student：3 条件 × 7 监督来源（OBS、BASE_ROW、BASE_CAP、J_ROW、J_CAP、BCE_CAP、NCE_CAP），相同 soft-target NCE、相同新 G 初值、各 8000 步。
- 评价：全 DEV、基础模型 Top100、全部 query，六 Recall、rSum、相对基础模型 / OBS 之差、救回 / 丢失、曲线；关系图覆盖、质量、新边质量、容量约束激活（Flickr 的正确边比例只在隔离评价器内计算；CC 无注入真值）；成本。不读 TEST，训练进程不读干净对应或噪声掩码。
- 验收：包内 CPU 检查 31/31；真实数据验收 9/9（目标与 NR1 生产实现一致、采样、非零 head 不变性、两端留出、匹配 KKT、soft-NCE、DEV 桥接、断点恢复、内容键）；两次因测试脚本自身问题取消的验收记录保留（`acceptance/ACCEPTANCE_LOG.md`）。57/57 个流程节点一次完成。
- 代码：`code/site/`、`sbatch/`。设计包本身（`package/`）、包源码的字节副本 `code/src/cr1/` 及相对包源码的 diff 不发布。L2RM 最近邻基线的接入清单已准备，未运行。

## 范围说明

单 seed；配对差不给显著性结论。Flickr 与 CC152K_RECOVERED 分开报告。CAP 不是 L2RM 复现。DEV 在开发中被反复使用。
