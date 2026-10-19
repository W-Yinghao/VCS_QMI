# VCS-QMI / CVPR V6 — 服务器审查意见包 r8

**交付增量：最后核对到 `3e2fe59`（2026-10-13 18:51:49 UTC）；关键 critic blob 未变，接续状态见 `docs/08_DELIVERY_UPDATE_CN.md`。**

**日期：2026-10-13｜审查截点：`54e1715e29dd2caaa6be7ed8e670829310951d39`（18:36:55 UTC）**  
性质：对当前实现、结果解释和后续实验的独立审查；不是一套已执行的新算法，也不是对冻结协议的静默替换。

## 核心意见

现有结果保留：r5 候选归一化已在官方测试中得到正面确认；REG 在两个模态对上展示学习型关系 critic 的实用性；10k 图像尺度的 D1 token 选择在两个宿主上没有建立预期优势。继续完成已经批准的 D1-FULL、E1、E2 和相应冻结评价，不重复提交。

下一步先检查 critic 的**训练状态、mask 使用方式、选择指标和宿主风险对应关系**，而不是立即扩大模型或改动合作者的 S/J。

本轮三个重点：
1. 参数冻结后，阶段 B 仍可能处于 train 模式，dropout 仍开启；这是源码行为，影响大小待实测。
2. 原 soft–hard 读数同时改变 Gumbel 扰动和软硬 mask，不能单独归因为软硬松弛。
3. **本轮新增**：源码 Spearman 使用 ordinal ranks，不处理 ties，常数输入也可能返回 1；需量化它对 CAL 读数与选择的影响，不得直接推断所有成绩无效。

## 阅读顺序

- `01_SERVER_AGENT_PROMPT_CN.md`：给服务器 agent 的接入指令。
- `docs/02_REVIEW_CN.md`：总体审查与已完成／待完成实验。
- `docs/03_CODE_FINDINGS_CN.md`：逐项源码证据、适用范围、验证办法。
- `docs/04_DIAGNOSTIC_PROTOCOL_CN.md`：本轮第一批只读与短运行诊断。
- `docs/05_CONDITIONAL_EXPERIMENTS_CN.md`：需单独确认的有限修改实验。
- `docs/06_MATH_AND_CRITIC_OPTIONS_CN.md`：推导、假设、critic 后续设计。
- `docs/07_REPORTING_AND_TEST_USE_CN.md`：结果口径、统计单位、测试暴露与回复要求。
- `sources/SOURCE_INDEX_CN.md`：每项来源、读取范围与公开文档。

`REVIEW_FULL_CN.html` 是离线全稿，`REVIEW_OVERVIEW_CN.pdf` 是简版审查总览。机器可读问题清单、结果快照和模板见 `tables/`、`templates/`。

## 附带代码的地位

`reference/` 是审查辅助代码：只读源码检查、数学与指标参考、合成输入的可选生产模块探针、逐 mask CSV 的诊断汇总。**不是自然数据训练 runner，不读取宿主权重，不调用 Slurm，不自动修改仓库。**本地测试覆盖参考代码及确定性小例子；未运行服务器真实 checkpoint。

先在隔离 worktree／目录运行；仓库探针只有在显式 `--allow-import` 时才导入本地生产模块。执行前核对当前 HEAD 与本包截点。没有相应源码版本就停止归因并记录差异，不拿旧审查给新代码定罪。

## 不做的事

不更换合作者 S/J、不重开 SSL／旧 R 线、不取消已授权作业、不修改正运行任务的共享导入模块、不删除历史预测、不重新挑选已经读取官方测试上的最好模型、不将缺失结果写为零。

本包没有实际修改 GitHub、读取服务器队列、下载新数据或提交训练。来源事实、源码事实、独立推导、性能假说和建议执行项在正文分别标记。
