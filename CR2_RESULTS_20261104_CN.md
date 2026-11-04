# CR2：保留观测锚点的有限关系修复（2026-11-04 第四十五版）

接在第四十四版（`NR4_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。

## CR2（`handoff/cr2_20261104/`，`RESULTS_CR2.md` 与 CSV）

- 用户：「先push CR2的内容吧」（`VCS_CR1_Review_CR2_Server_20261104`，协议 CR2_ANCHOR_PRESERVATION_CONCENTRATION_V1，证据 bf3c4380）。
- 0 个新关系估计器 fit；复用冻结的 CR1 CAP 图、交叉拟合分数、专家与内容缓存。
- 只读关系审计：观测伙伴是否落在无标签 Top32（H）的交叉表，raw / 校准候选排序及校准接缝，条件熵 / ESS，每行质量与图像负载（Flickr 正确性只在隔离评价器内；CC 无注入真值）。
- 32 个新 student fit：24 个修复 = {Flickr00, Flickr60, CC152K_RECOVERED_20261104} × 来源 {BASE, J, BCE, NCE} × {KEEP_SOFT, KEEP_TOP1}；每个文本行总质量精确保持为 CR1 旧图的 m_j，有自然 Top32 支持的观测锚点保留全部质量，其余行最多转移一半质量；6 个观测筛选对照 {SUPPORT_OBS, RANDOM_SUPPORT}；2 个隔离 oracle 诊断（Flickr00、Flickr60，独立进程读取真值，不作部署方法、不称上界）。全部新 G 初值、统一 soft-target NCE、8000 步，沿 CR1 的采样、优化与评价。
- 不重新求解 CAP 容量问题；组合图的实际图像负载只作报告，不沿用 KKT 证书。
- 验收：包内 CPU 测试 28/28；真实小样本验收 7/7（稀疏图与行质量、CSR 采样与 CR1 一致、soft-NCE 展开等价、训练路径逐位一致、score 不变性、DEV 桥接）。冻结协议 `PUBLIC_PROTOCOL_FROZEN_CR2.json`。全 DEV、基础模型 Top100、全部 query；不读 TEST。
- 代码：`code/site/`、`sbatch/`。设计包本身（`package/`）、包参考实现的字节副本（`code/reference/`、`code/site/build_relation_pkg.py`）不发布。

## 范围说明

单 seed；配对差不给显著性结论。Flickr 与 CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
