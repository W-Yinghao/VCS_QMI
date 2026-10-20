# 训练遥测补丁与服务器核对

本包使用已经实际执行的 N2F `source/src/v9/n2/train.py`，其 SHA256 固定为 `cafac01f989b0bbdd88f4d21c51e7475227367d8d4750057897b77832a61302c`。`patch_train_source` 只接受这个文件的完整文本，在新 runtime 副本上施加补丁；文件内容不同、重复应用补丁、补丁定位不唯一都会报错。原有源码、结果和权重保持原身份。

## 修正的具体问题

原代码续跑时用 `len(log["steps"])` 恢复更新次数，但该日志每 20 次更新才记录一次。补丁改为核对并累加各完整 epoch 的 `steps_in_epoch`。新的 checkpoint 另存 `global_step`，并要求它与 epoch 累计值一致。每条新抽样记录包含 `epoch`、`step_in_epoch` 和 `global_step`，其中兼容字段 `step` 同样是真实累计更新数。旧 checkpoint 的缺失字段从完整 epoch 记录恢复，不从稀疏记录条数猜测。旧的抽样记录不重写。

这使每个完整 40 epoch 训练的 45,320 次更新能被核对，也使“实际最后 200 次更新中的抽样 U-risk”能正确定位。补丁没有增加训练步数，没有改变样本顺序或抽样间隔；修正的是续跑后抽样发生的累计步位置。

JSON 完成文件和最终训练日志改为先写临时文件、再原子替换。原有 `best.pt`、`last.pt` 的原子替换保留。训练遇到非有限权重、分母、损失、梯度或验证相似度会明确失败，不产生一个看似有效的排名结果。已有 `done.json` 必须与完整 epoch 日志、主终点选择和所需产物一致；不一致时报告错误，不启动另一次训练覆盖证据。

新 checkpoint 保存本单元 `identity.json` 与 `run_manifest.json` 的 SHA256。生产续跑必须与两个文件同时相符；旧臂、旧协议或缺失身份字段的 checkpoint 不能放入新目录后被接纳。只有不提供生产 identity 的合成核对才可读取旧格式 checkpoint，以检验原先更新计数错误的修复。实际 runner 每次生产训练都创建并核对 identity。

## 数值训练部分

以下代码表达式保持原样：SGR 建模与初始化、Adam、每 epoch 的独立 shuffle RNG、batch 迭代、双向逐 anchor hinge、warmup/hard 切换、SUM 损失、固定全池分母、梯度裁剪、optimizer 更新、学习率衰减、每 epoch 完整 DEV 验证以及“最大 rSum、并列选更早”选择。

有效输入上的补丁不消耗额外 Python、NumPy 或 Torch 随机数。不共享一个已经训练过的 warmup checkpoint 作为新对照的起点。增加的有限性检查会引入少量同步和检查成本，因此新包报告各 attempt 的真实时间，而不声称与旧日志有相同耗时。

## Slurm 退出与恢复

外层 runner 收到预退出信号后，在该单元目录写 `STOP_REQUESTED`。训练在当前 epoch 的训练、验证、best/last checkpoint 全部保存后退出，返回 `status=INTERRUPTED`、已经完成的 epoch 数、目标 epoch 数和真实累计更新数，并保存训练日志；它不写 `done.json`。若第 40 个 epoch 已经完成，则正常写完成结果。

明确重启该单元时，外层 runner 归档旧停止请求，再从最后一个完整 epoch 恢复。若剩余时间不足以完成当前 epoch，作业仍可能被 Slurm 终止；最后一个已经保存的 epoch 是恢复边界，未保存的计算需重做。这个机制不会声称保存了 epoch 中间的 optimizer 状态。预退出信号应留出足够完成一个 epoch 的时间。

## 核对的范围

纯 CPU 单元测试核对补丁来源、语法、科学表达式不变、45,320 更新计数、稀疏日志恢复、停止状态以及完成文件一致性。本地还使用 Torch 2.9.0+cpu 实际执行了下述小型 SGR 核对，具体报告随包提供。CPU 核对通过不等于服务器 CUDA 核对已经运行；服务器仍需在所选实际环境执行它。

服务器 `verify-torch` 使用真实 SGR 模块，但只生成小型合成 region features 和 token IDs。它不读取原训练数据、DEV、TEST、teacher 或机制标签。缩小后的 fixture 为 30 对样本、每 batch 8、最后 batch 6、4 个 epoch，覆盖两个 warmup epoch、两个 hard epoch、一次学习率变化和稀疏日志。

核对包括：

- 原版与补丁版各自从头训练后的 model、Adam、RNG、best 和完整科学轨迹相同；
- 补丁版连续训练与在 epoch 1 保存后停止、恢复训练相同；
- 原版格式、没有 `global_step` 的 checkpoint 被补丁版恢复后相同；
- 正常带身份的 checkpoint 能续跑，复制来的其他单元 checkpoint 以及没有新身份字段的生产 checkpoint 被拒绝；
- 实际第一批 SGR 反向传播中，每个 row/column hinge 的系数与 `v_i / mean_v_full_pool` 一致；
- 三个原生 SGR 部件确有参数更新，阶段切换、学习率和最后一个非满 batch 正确；
- 已经完成的单元再次调用不会创建新 host，改用不相符的目标 epoch 会报错。

同一确定性设备上的 model、optimizer、RNG 和科学轨迹使用精确相等比较。逐系数检查用独立 float64 表达式作参考（先遵循训练的 float32 权重转换），允许 `4 × eps(float32) × max(1, |系数|)` 的舍入差，完整记录各系数容差和误差。这个小型实现检查不证明跨 GPU、跨 Torch 版本逐位相同，也不产生方法优劣证据。

报告状态只有 `PASS`、`FAIL`、`NOT_RUN`。Torch 或请求的 CUDA 设备不存在时必须为 `NOT_RUN`；失败时保留具体检查和错误。新训练引用服务器核对报告、运行时文件身份和实际环境记录。环境变更应重新核对，不能沿用一个不同设备上的通过标签。
