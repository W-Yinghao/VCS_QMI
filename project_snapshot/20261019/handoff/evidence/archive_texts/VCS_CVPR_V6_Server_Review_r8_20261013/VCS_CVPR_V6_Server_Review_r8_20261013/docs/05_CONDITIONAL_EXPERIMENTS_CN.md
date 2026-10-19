# 条件修改实验：只改变被证据指向的因素

## A. 当前正在跑的冻结实验照常完成

D1-FULL / E2 / E1 / 原授权 phase F 的代码、训练长度、CAL 选择和测试行不因审查静默变动。若代码问题严重到负责人决定暂停，须独立记录 affected units、尚未执行部分、保留预测和授权；本包不是自动暂停命令。

新审查分支用独立 namespace 与输出根。不得在同一 `S1F` 行混用旧/新 mode、Spearman、gate 求解器或不同模型版本。

## B. 最小可解释干预：相同阶段 A，改变 B 的 critic mode

**触发**：REV8-01 确认 phase-B dropout 有非平凡贡献，并且服务器认为它与部署差异值得干预。不是要求先看到任务结果变好才运行。

**首轮范围建议**：LLaVA，一个已完成尺度（优先 10k，避免 full-data 并行更改）；VCS/Logistic × seeds 0,1,2。所有六个训练对都完整运行，不按 seed 0 胜负筛选。每个方法/seed 两个 arms：

| arm | phase A | gate 初始权重 | critic mode | Gumbel/预算/步数 |
|---|---|---|---|---|
| L：legacy restart | 同一已选 A state | 同一预记录 gate init | train | 原 5k 与原 mask 求解 |
| E：deterministic critic | 同一个 A state | 同一个 gate init | eval，gate 可训练 | 同 L |

这是 **6 个新增 E + 至多 6 个匹配 legacy 重放**，不是“只需 6 次”无条件承诺。若现有 legacy 的阶段 B 状态、初始化和随机流可严格复现，才复用原 L 并先做一致性检查。否则按最多 12 个 B-only 单元给预算。

配对输入 schedule（qid/k/Gumbel seed）独立于 dropout RNG；不通过重设所有全局随机种子把 dropout 人为耦合成一种不同算法。初始 A weights/gate init/hash、optimizer、dtype、训练步数、CAL 动态阈值过程分别记录。

### 不同时改什么

不改 Spearman、不改 A checkpoint、不改 warm-up、不加覆盖正则、不换数据、不改 soft top-k。否则不能把结果归因于 mode。若 R8-03 需要 corrected selection，它是另一个单独的 sensitivity experiment。

### 评价

先完整宿主 GQA DEV 二元 + 原开放式任务；TextVQA/ChartQA val 为已知 transfer 复验。报告 fixed-budget accuracy、Brier/h、grid/SparseVLM 等预算参考、总时延（包括 selector）和 mask 覆盖。

主要效应是 within-method E−L，VCS/Logistic 的交互为次级，不能只拿新 VCS 与旧 JS。旧 official test 只作已完成历史结果；新测试访问需负责人指定新的确认方案。

## C. 若真正问题是软硬求解

先完成四格诊断，再选一个方案，而不是同时试五种温度：
- 硬前向 + 有明确说明的 surrogate backward；或
- 在固定预算的有限 hard mask 集合内作监督选择/排序；或
- 将软训练 mask 加入 critic 训练，使输入机制相符。

三者都属于不同算法选择。STE 的反向一般不是硬选择真实梯度；不得把它称作“精确梯度修复”。主控制仍为相同 critic 的旧 solver 与匹配 Logistic。每条自然结果都需要完整宿主前向。

## D. 若 Spearman 确实影响选模

先保存当前被选模型、历史表和读取过的测试集。corrected rank 指标只在原 CAL 上重新计算，单独报告 selection sensitivity；缺失的历史 checkpoint 不补写。新的 A 训练若需要重做，预算与比较单列。

不以 corrected selection 在测试上表现更好为采纳依据。实质上改选模之后，旧盲测试已经被使用，新的独立确认需要额外设计或明确标为 repeated-test sensitivity。

## E. 若 critic 没能估计宿主损失

先看同问题同预算排名及 policy-mask 外推；再考虑宿主校正风险读数。新方案至少包含：
1. 现有后验差 S1/S2；
2. 直接实际风险回归 S3；
3. 预测宿主行为的平方差；
4. 加校正项的读数（见数学文档）。

不只更换损失，需定义每个头预测什么、训练标签哪里来、部署前可见什么。允许第二方案最终不如 S3；这否定该结构的实用收益，不是否定 S/J 恒等式。

## F. 若 mask-policy shift 成立

仅用 FIT 图像做一轮固定预算的数据聚合：原 mask、当前 policy mask、grid/random 共同收集；三目标收到相同集合。宿主新调用成本列入训练成本，CAL 不回流 FIT。新增策略不能读取标签/宿主答案才决定是否获取完整输入。

## G. 若空间覆盖是主要问题

同一总 k 下，固定少量 grid 覆盖 + 学习剩余预算。覆盖配额在新训练之前决定，不根据 val 看哪一种最优后隐藏探索。S1/S2/S3 都能使用相同规则，保留纯 grid 和原 learned gate。此方案是任务归纳偏置，不预称 VCS 专属创新。

## 终止／收束规则

- 完成一次定义清楚的干预后，若模式/solver 的任务影响很小，就结束该解释；不反复改变阈值直到显著。
- 若完整数据与合理实施下仍低于 grid/公开方法，当前 D1 作为边界结果保留。REG/r5 的正面发现不被它抵消，也不被它借用来宣称 D1 成功。
- 不从任何一项失败推出“必须退出 CVPR”或“所有应用不可做”。范围始终对应实际实验。
