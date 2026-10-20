# V9 N2M v1：总质量与内容关联的完整训练对照

日期：2026-10-19。机器参数以 `config/protocol.json` 为准。本包已经固定研究规则；实际 server 输入、GPU 环境和参照复用决定由 `prepare` 写入不可变的 `experiment.json`。本协议不声称本轮训练已经执行。

## 1. 主问题与推断范围

原 N2F 的 VCS U 均权重约为 .61，而 A 权重为 1。这使可信 A 在目标中的质量占比从 10% 提高到约 15%。原有带权方法相对 Unweighted 的改善，尚不能单独证明逐样本可靠度估计起了关键作用。

本轮预定两项主比较：

1. **VCS − Constant-U**：保持 A/U 总质量后，非恒定样本权重是否带来增量？
2. **VCS − Shuffle-VCS**：保持 VCS 权重分布、质量和 ESS，打断其与样本内容的关系后，结果是否改变？

同 seed、SUM、best full DEV rSum 为主终点，所有 Recall 从同一被选 checkpoint 读取。DEV 已多次使用；这里的 seed 是三次配对随机重复，不是独立确认集。原 Unweighted、RankMass、BCE、RuLSIF、OracleMask/Drop 只在原 N2F 完整表中作为上下文。

## 2. 固定输入与完整配方

| 项 | 固定值 |
|---|---|
| 数据 | Flickr30K precomp，train 29,000 images /145,000 pairs；DEV 1,014 images /5,070 captions |
| A / U | 14,500 /130,500 pairs；原 10% anchor 与原角色划分 |
| 噪声 | design rho .4，复用原 seed 901/902/903 observed 映射 |
| 学生 | 原 SGR 单模型全部可训练部分，每臂从本 seed 的随机初始化开始 |
| 训练 | 40 epochs，batch128，最后 batch104，每轮1,133更新，共45,320更新 |
| 优化 | Adam lr2e-4，clip2，margin.2，epoch30 起 LR×.1 |
| 阶段 | epoch0–4 原 mean-negative warmup，epoch5–39 原 hard-negative hinge |
| 聚合 | SUM，全池固定分母；从第一步应用权重 |
| 选择 | 每个完整 epoch 做 full DEV；rSum 最大，并列取更早 |
| 数值 | 原 Torch2.9.0+cu128/CUDA12.8/Python3.12.12；确定性开启，TF32关闭，cuDNN benchmark关闭 |

不共享已经按某一臂权重训练过的 epoch4 checkpoint。所有低权重 pair 仍留在对应 minibatch 的负候选池。A 的权重恒为1。

令每个 anchor 的双向 hinge 和为 `ell_i`，全池系数均值为

\[
\bar v=\frac{14500+\sum_{i\in U}w_i}{145000}.
\]

每个 minibatch 的目标严格沿用原 N2F SUM：

\[
L_B=\frac{\sum_{i\in B}v_i\ell_i}{\bar v},\qquad
v_i=\begin{cases}1&i\in A\\w_i&i\in U.\end{cases}
\]

权重 detach。分母不是该批次的总权重；SUM 也不额外除以 batch size。最后一批的行为保持原实现。

## 3. 处理定义

**Constant-U**：按 pair ID 与原 observed U 对齐，读取原 CSV 已保存的浮点值，以 float64 求均值 mu，再把每个 U 设为 mu。保持原全池分母和总质量。正的恒定权重使 ESS_U 等于 n_U；不要求它等于原 VCS 的 ESS。

**Shuffle-VCS**：按 U 的 observed_image_id 分为26,100个五行块。图像 ID 按整数优先、否则词典序排序，整数相同以原字符串打破并列；块内 caption_id 用同一规则排序。生成唯一一次

```python
Generator(PCG64(SeedSequence([seed, 20261019, 9101]))).permutation(26100)
```

目标块 k 得到来源块 pi[k] 的五个权重，再写回目标 pair ID。允许固定点，保存 `permutation.npy` 和 SHA256，40 epoch 不重洗。核对完整 multiset、总质量和 ESS；若不是五行块，报结构错误，不能默默换成逐 pair shuffle。

这个处理保留接收图像块内的权重结构，不保证保留所有 caption 原始来源依赖。生成器不读取 evaluator 标签，也不能根据离线分数挑选一次更有利的置换。三臂 CSV 与置换完成读回核对后才发布。

## 4. 矩阵与参照

九个表格单元固定为三个 seed × 三臂。Constant-U、Shuffle-VCS 六个单元必须新跑；VCS 三个单元逐个标为新跑或历史复用。

历史复用要求完整40轮、best/last与元数据 SHA、原 VCS 权重 SHA、host cfg、原协议和环境证据一致。当前旧 hardware 没有完整记录 driver/cuDNN/NumPy，因此 `auto` 在没有额外当时证据时会选同环境的新 VCS。`replay9` 可直接冻结九次；`reuse6` 只有证据充分才通过。

三臂轮换顺序固定为：901 VCS→Constant→Shuffle；902 Constant→Shuffle→VCS；903 Shuffle→VCS→Constant。同 seed 的前臂须完整，后臂才开始；不同 seed 可按现有配额独立推进。最多九次完整训练，不扩展到12次或更大网格。

旧 N2F 33单元须按原规则收完再开始本轮完整训练。复核截止提交的副本仍有 BCE SUM902、VCS SUM903、BCE MEAN903 缺少最终结果；server 以实时文件核对其最新状态，不把此历史列表永久当作待办。

## 5. 后验和固定风险诊断

独立 scorer 才可读取 `pairs.eval_only.json`。clean=未被注入机制实际错配，它不是语义兼容性真值。输出 rho_raw/projected/used/realized、原矩记录、Brier、logloss、AUROC、average precision、固定10箱校准、平均权重/ESS、clean与mismatch平均保留、残余噪声质量占比。

AUROC 给并列正负对半分；AP 合并相同分数，恒定分数的 AP 等于 clean 比例。单类、空质量等情况按指标定义给 null 与原因。logloss 的数值裁剪阈值为1e-12并披露。

原 CSV 保存 score/w 到10位小数。生产后验重建使用原 JSON 的完整 `rho_used`，允许由保存精度和导数上界推出的舍入误差；不把新的任意性能容差当作数值核对。固定同一 `T=tanh(a*s+b)`，将 rho 换为实际机制比例只用于 oracle-rho 敏感性分析。

原 rho 区间只保留为无覆盖保证的诊断：P/Q 参考图像重叠、原 Q 聚类标识及角色前缀不能建立独立性，U donor 联系也可跨原估计/留出划分。不输出 pair-IID 的“高精度”区间。

补充固定风险诊断预定用原 N2F VCS SUM 的 best checkpoint，置为 eval 模式。取原完整 observed 池在 `default_rng([seed,9020,5])` 下的前8个完整batch，保留其原负候选，导出每 anchor hard hinge。三个控制和 clean oracle 用同一份 ell 重加权。这个诊断反映同一模型和负候选下的目标变化，不等于各臂重新训练后的结果，也不恢复全 clean 负候选池。

上述分数不作训练资格的性能门槛。身份错位、权重变更、非有限值是实现问题；指标弱则保留为研究结果。

## 6. 续跑、完整性和成本

唯一宿主改动为独立副本中的训练遥测补丁。累计更新从已完成 epoch 的实际 steps 求和，并与 checkpoint 的 global_step 相核对。checkpoint 绑定本臂 identity 与 run_manifest 的摘要，旧格式 checkpoint 不能成为新控制的热启动。

run_manifest 的权重身份和配方逐字段精确一致；ESS、总质量和分母的纯 float64 派生诊断允许2e-12相对、1e-12绝对的舍入差。保留既有 manifest 的原始字节，避免再次出现1 ULP 的诊断差导致正确训练无法恢复。这没有修改权重、训练系数或优化配方。

每个 attempt 独立记录开始/结束、Slurm job ID、硬件、状态和时长。被强制杀死而没有结束记录的时长为未知；`sacct` 另行汇总整个 allocation，并对同一 job 内多臂去重。最终 `done.seconds` 不能替代全部成本。

预退出信号只请求在完整 epoch 的 DEV 与 checkpoint 保存后停止；中途设备故障或硬时限依靠前一个完整 epoch 恢复。不会将部分epoch、当前最好分数或提前停止写成完整结果。原结果、失败记录、恢复记录均保留。

## 7. 结果表与投入决定

每 seed 主表：VCS、Constant-U、Shuffle-VCS 的 best full DEV rSum，以及两项配对差。三对齐全后才报告主均差、最小/最大差、正向seed数。缺失、失败或 checkpoint 未核验的单元单独标记；不填补，不删seed。

同时报告六Recall、mR、best epoch、epoch4/5、最差hard值、last、最后5轮均值、梯度/clip窗口、严重塌陷与成本。末值不能替换已经固定的主终点。

严重塌陷：连续两完整 hard epoch 的 DEV 均低于本臂 epoch4 的50%，且这两轮实际最后200更新中的抽样 U-risk≥.39 比例各至少90%。新日志直接记录 step_in_epoch；旧 resume 日志根据实际历史恢复位置重建，未知则输出 UNKNOWN。该诊断不提前停止或删除训练。

**工程继续规则**：两项 VCS 主差均值都至少+3rSum，各至少2/3 seed为正，严重塌陷不更多，才继续投入“内容关联的可靠度”扩展。这仅为资源决定，不是显著性检验；未通过也不证明等价。

| 观察模式 | 下一步解释 |
|---|---|
| Constant 与 VCS 接近 | 逐样本分配的增量有限，优先分析 A/U 再平衡 |
| VCS 同时优于两个控制 | 内容相关分配有用，再看校准或更有信息的输入 |
| VCS 只优于 Constant | 权重分布/ESS/梯度效应仍是解释，不能直接归因可靠度 |
| VCS 优于 Shuffle，Constant 接近 VCS | 错误分配会伤害，但复杂逐样本估计未显示足够增量 |
| Brier 更好，完整终点没有相应改善 | 保留估计结果，重审应用接口或结束此方向扩展 |

这些是预定的解释分支，不是对未执行结果的预测。本轮不增加 MEAN、LR、rho、anchor、COCO、REG 或额外 Unweighted 完整训练；任何后续扩展由本轮实际证据决定。
