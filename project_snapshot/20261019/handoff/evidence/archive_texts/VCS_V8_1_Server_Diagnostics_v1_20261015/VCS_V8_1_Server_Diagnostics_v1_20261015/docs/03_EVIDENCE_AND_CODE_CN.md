# 03 · 证据、源码与需核对的解释

本节分三类。**已提交结果**复述来源，不是本包重新计算；**源码事实**以本次读取提交为准；**影响解释**尚需真实job snapshot和产物检验。

## 1. 已提交V8结果 [R2]

| U-CS对外差值：N9选中图像人工评分 | DALLE_3 | SDXL_Base |
|---|---:|---:|
| FGA released | −0.025，Holm未显著 | +0.039，Holm p=0.046 |
| VQAScore CLIP-FlanT5-XXL | −0.121，Holm显著 | −0.046，Holm未显著 |
| TIFA | +0.043，Holm未显著 | +0.130，Holm显著 |
| DSG本地适配 | +0.026，Holm未显著 | +0.147，Holm显著 |

这些是1–5量表的评分差，不是百分点。“未检测到差异”不能写为等效；RANK/MSE/CS不同风险尺度不能直接比较训练loss数值。

八臂×三种子已导出。五臂选到原LR网格下沿；这说明被测优化范围有限，不证明更低LR一定改善。U-RANK的CAL均值MSE约0.52–0.60，其他约0.40；排序臂在该回归选模规则下表现差，但应核对原生排序行为。[R1,R2]

## 2. 重要更正：epoch 0不是pre-update模型 [R3]

本次源码顺序：`for epoch ...` → 遍历完整训练batch → `predict`和CAL → 保存`best.pt`/`resume.pt`。因此，数组索引0代表第一轮完成后的模型。

原报告“epoch 0是带自然直方图bias的初始化head”与该循环不相符。提交一份证据对照：

| 字段 | 必须记录 |
|---|---|
| source_commit / snapshot_hash | 真实运行使用的源码，不只当前HEAD |
| epoch_index | 原始0-based值 |
| completed_epoch_count | 经日志核对后为epoch_index+1，不改旧字段 |
| planned_batches / attempted_steps | 原计划与日志步骤 |
| successful_updates | 若AMP历史没记录，填unknown，不以计数代替成功更新 |
| images_seen | 对应轮次是否完整，是否smoke或被截断 |
| resume_history | 从哪轮恢复、是否有数据/随机态缺失 |
| checkpoint_stage | INIT_PREUPDATE / AFTER_EPOCH / RECONSTRUCTED_INIT / UNKNOWN |

仅更正报告解释时，不改变数值、选中的epoch或主结论。若job snapshot实际不同，以快照为准，说明本包核对范围。

要比较真实初始状态，可用相同上游权重、seed、FIT直方图重新构造 **RECONSTRUCTED_INIT**。没有存下原初始化权重就不能冒称bitwise复现；未经训练的随机质量头不是强评分基线。

## 3. 保存策略限制 [R3]

源码持续覆盖`best.pt`与`resume.pt`；每epoch日志被保留，不能据此推定每epoch权重或逐图预测都在。先列实际文件：

- 存在某epoch checkpoint：可以重新推理开发角色，记录来源。
- 只有该epoch的聚合日志：只能做曲线分析，不能计算其逐promptTop-1。
- 只有最终best/resume：不启动整轮重训来“补历史”，缺项报告为not_saved。
- 没保存RNG的恢复路径：不声称精确配对重放已成立；未来条件实验必须验证恢复一致性。

## 4. 不把D1的问题搬到V8 [R3,E2]

V8训练时`model.train()`且冻结视觉骨干转eval；预测时`model.eval()`。D1曾出现的phase-B冻结critic/dropout问题不构成V8有同一错误的证据。

本轮检查的是实际freeze mask、模块mode和AMP成功更新。诊断梯度时允许eval-mode自动微分；不要对需要梯度的路径套整体`no_grad()`。推理重读则保持全部eval，且不写回任何训练状态。

## 5. 排序项与尺度 [R4]

U-RANK = 归一化均值MSE + pairwise BCE；U-LIST = 同一MSE + listwise CE，tau=1、lambda=1。无信息人评分pair在训练中可作为0.5 target，而评价通常排除同分pair。这是原设计，不直接判bug。

核对batch含多少有效subgroup/pair、点项和排序项的梯度量级/夹角、mu是否饱和、同一prompt是否跨组、归一化是否与R1一致。不能从“rank loss数值更大”推断梯度主导，也不因梯度不同就自动修改lambda。

## 6. 确认集与数据决定 [R6–R8]

V8-05已提交1042053，850非Rank提示×6生成器，原主指标是pooled Spearman，另有同提示pairwise；不是N9。其主模型是三种子mu均值，另报逐seed。这与V8-04的逐seed选图均值不同，不能混合成同一估计量。

main已记录负责人现有数据使用决定。本包沿用，不再建立申请/许可核查的实验门槛。标签隔离仍保持：数据可用与允许用确认标签调参是不同问题。


**交付状态更新见 docs/09：V8-05确认已完成。上述协议保护不变；不再把等待该作业作为前置条件。**
