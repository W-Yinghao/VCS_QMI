# 标准逐图行格式

参考CLI读取JSONL，所有行是明确提供的输入，不自动搜索服务器目录。

必需字段：`dataset, role, method, seed, checkpoint_id, prompt_id, pool_id, image_id, candidate_index, score`。

可选：`cluster_id`（默认prompt）、`score_kind`（rating_mean或native_score）、`q`（五概率）、`u`（四尾概率）、`human_hist`（五投票比例）、`human_mean`、`votes`、`source_sha256`。

- 同一dataset/role/method/seed/checkpoint/pool/image不得重复；同pool候选index唯一。
- `q`若存在则sum=1，非负，score_kind=rating_mean时必须与score一致；u若存在须与q一致。
- 标签可以是完整hist，或仅human_mean；只有mean的公开预测比较可以做MSE/排序，但不得制造五档标签。
- `score_kind=native_score`的外部方法可以做排序，不计算1–5 MSE。
- 任何label字段在label-free模式都拒绝；避免外部dummy均匀hist误入评价。
- labelled模式仅允许FIT_AUDIT/CAL_SELECT/CAL_PROB/DEV；其他角色拒绝。该限制是流程护栏，不能替代真正的数据隔离和权限控制。
- 三种子不要事先平均；ensemble须另命名，例如U-CS-ensemble，而不是覆盖seed0。

严格并列按浮点原值，不按显示精度；主选择按candidate_index升序后的首个最大分。模型缺候选默认停止并输出说明，不静默削掉该图。
