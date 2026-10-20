# N2M 评分与结果分析说明

本轮的两个问题是：VCS 的逐样本权重比相同 A/U 总质量的 Constant-U 多带来多少完整训练收益？保留权重分布、打破接收图像内容对应后的 Shuffle-VCS 又改变多少收益？概率评分用于解释这两项对照，不替代完整训练终点。

## 1. 数据边界与锁定顺序

控制生成器只读 observed pair 表和原生产 VCS 的锁定 CSV/lock，不读机制标签。`score_weights` 先核对控制 manifest 与调用输入的 observed、生产 CSV、生产 lock 哈希，再核对三臂 CSV/lock 哈希及 U 的 ID 覆盖；通过后才打开单独的 EVAL_ONLY 标签文件。观察映射和标签必须按 pair_id 精确覆盖同一池。`mismatched` 必须是 JSON 布尔值，并与 `observed_image_id != original_image_id` 一致；A 中不允许错配。

若提供 noise manifest，还核对 observed/evaluator/mapping 哈希、样本数、实际错配数和实际 rho。不同 seed 可以有相同的 pair ID，因此仅核对 ID 集合不足以识别同一噪声映射。

标签表示注入机制是否真的更改了图像索引。它不回答不同图像是否也能与该 caption 语义相容。所有结果明确使用 `clean = not mismatched` 为正标签。

## 2. 离线输出

`offline_metrics.json` 保留完整来源、rho/moment 原记录、每臂指标、同 critic 的 oracle-rho 诊断和重建误差。`offline_metrics.csv` 是每臂主表，`calibration_10bins.csv` 是预先固定的校准表，`OFFLINE_READOUT_CN.md` 可直接阅读。

每臂输出：

- mean(w)、U 总质量、ESS、全池归一化分母和 A 在保留质量中的比例。
- Brier 和 logloss。logloss 仅在计算对数时截断到 `[1e-12,1-1e-12]`，同时报告截断数；不改写生产权重。
- AUROC，给并列正负样本各半分。只有一类标签时记 null/NA 并说明原因。
- AUPRC 使用非插值 average precision：同分数整体作为一个阈值。因此恒定分数的 AP 等于 clean 比例，不因 CSV 的行顺序变化。
- 10 个等宽固定区间 `[0,.1),...,[.9,1]`，报告每箱数量、平均权重、实际 clean 比例和 ECE；空箱记 null。
- clean/mismatched 的平均保留率、保留总质量，以及 `sum(w*mismatched)/sum(w)` 残余噪声质量占比。零总质量的比例记 null，不伪造为 0。

以上 pair 数字均为描述性诊断。代码不按 130,500 个 pair 独立同分布来生成区间，也不把已有 N2 区间解释为具备覆盖保证：参考 P/Q 复用图像，原 Q 聚类和角色前缀没有证明独立性；U 的 donor 关系也可能跨原 estimation/held-out 划分。

## 3. 相同 T 的 oracle-rho 敏感性

从原锁定 CSV 的 `score` 以及原 lock 的 scalar critic 参数恢复：

`T = tanh(a * score + b)`

然后用该 seed 的实际机制错配率代入原后验公式：

`w_oracle = (1-rho_actual)*(1+T) / ((1-rho_actual)*(1+T)+rho_actual*(1-T))`。

整个过程不重新拟合 critic，不选新置换，不将 oracle-rho 写成学生训练权重。缺少 score 或 a/b 时将该诊断标为不可用，不重新训练一个模型替代它。

原 CSV 的 score 和 w 都被写为 10 位小数；JSON lock 的 `rho_used` 保留实际计算值，而 CSV 的 `rho_hat` 只保留 8 位。重建使用 JSON 中的 rho，绝不使用低精度 CSV rho。对生产 w 的一致性容差来自已有量化误差：

`5e-11 + (|a|/2)*5e-11 + 128*eps_fp64`。

第一项对应 w 的舍入，第二项使用后验导数 `|dw/ds| <= |a|/2` 对 score 舍入传播的界，第三项仅覆盖 fp64 舍入。另做 w 舍入区间反推 T 的一致性诊断；接近端点时不套统一 T 误差阈值，因为反演会病态。rho 位于边界时 T 不可由后验识别，明确记不可用。

指标较弱不阻止本轮完整对照。若输入哈希、ID、标签映射或公式重建不相容，应先查清具体身份或数值问题，不能把错位结果解释为方法效果。

## 4. 可选固定学生风险诊断

`score_fixed_losses(observed_path, labels_path, controls_dir, loss_csv, out_dir)` 接受同一学生、同一批次和同一负候选库导出的逐 anchor `ell`，CSV 列为 `batch_id,pair_id,ell`。同一批次内的 pair 不可重复，所有 pair 必须属于冻结 observed 表。可以有多个固定批次，同一 pair 出现在不同批次是允许的。

伴随的 `<loss_csv>.meta.json` 必须给出：

- `checkpoint_sha256`：固定学生 checkpoint 的 64 位 SHA256。
- `observed_sha256`：冻结 observed 表的 SHA256。
- `batch_definition`：批次顺序和构造说明。
- `ell_definition`：逐 anchor hinge 的定义及 warmup/hard 阶段。
- `same_negative_pool: true`：所有比较共享同一负候选库。

本包 exporter 还给出 `loss_csv_sha256`、`seed`、`experiment_sha256`，以及每个 batch 的 `batch_id,n,pool_indices`。评分器在打开标签前核对 CSV 摘要、所有 batch 的数量和 pair 顺序与冻结 observed 索引一一对应；声明的摘要或顺序不符时拒绝该诊断。外部旧导出若没有 CSV 摘要或完整 batch 索引，将明确输出未核验身份的警告，不能视为科学导出已完整核验。

Python API 可用 `expected_experiment_sha256=` 与 `expected_checkpoint_sha256=` 额外绑定调用方冻结的实验和参考 checkpoint。CLI 应传当前 experiment 的 SHA256 和原 N2F VCS best 的已核验摘要，而不能仅信任导出文件自身的声称。评分 JSON 的 `identity` 分开记录 CSV、batch、observed、experiment 与 checkpoint 的核对范围；`scientific_training_complete` 始终为 false，因为离线风险诊断不是一项完整训练结果。

评分器对同一份 `ell` 分别应用三臂系数和 clean oracle 系数，输出 `sum(v_i*ell_i)/mean_v_full_pool` 及其除以 batch size 的形式。分母始终由完整冻结池计算。该功能只比较固定学生下的重加权风险；它不代表各臂独立训练后的收益，也不代表已经去除错配负候选的 clean-pool 风险。外部导出器必须诚实保证元数据；评分器未加载模型，因此不会声称独立核实了这些 ell 来自所写的 checkpoint。

## 5. 完整训练主分析

`summarize_experiment(experiment_path, out_dir)` 要求冻结清单恰为 VCS、ConstantU、ShuffleVCS × seed901/902/903。VCS 可按已冻结身份复用历史结果，其余两臂必须新跑。每行保留 RUN/REUSE 和结果来源路径。

汇总入口内置 `verify_analysis_metadata`：先核对同目录 `EXPERIMENT.sha256`、本包 protocol 摘要、experiment 中的 `analysis_parameters` 与 protocol 的一致性，并在提供时核对 frozen package code。它不调用训练入口的完整输入核验，因此结果分析机器无需挂载原始特征、vocab 或 runtime；仅需当前冻结包、experiment/receipt 和要读取的结果文件。它不会因此跳过结果或 checkpoint 自身的完整性核对。文件路径仍沿用 manifest 的路径，复制结果时应保持可访问的对应路径；不要直接改写已冻结 experiment。

完整单元必须具有 full_result、train_log、run_manifest、identity 和 best/last checkpoint。检查 JSON 与 checkpoint 哈希、40 个有序且无重复的 epoch0..39、每 epoch 1133 更新、总计45320、六项 Recall 及其 rSum/mR 一致性、同一 checkpoint 的最大 DEV rSum、相同值选更早 epoch。新结果核对 protocol/experiment/观测/权重/runtime 身份与恢复后的真实 global step。历史复用结果核对冻结 N2F 身份及保存的 result 哈希。

未完成、失败、失去 checkpoint 或哈希不符的单元，主终点均为 null。部分训练的即时最好分只出现在 `current_best_diagnostic`，不会填入主结果。明确未拷贝 checkpoint 的元数据可以检查，但仍标为 `COMPLETE_METADATA_ONLY`，不视为已经完整核验。

仅有两项预定主差：

1. VCS − ConstantU。
2. VCS − ShuffleVCS。

报告每 seed 的差、三 seed 的均差、差范围、正/零方向数量；三组尚未齐全时不输出选择性完成子集的主均差。n=3 不生成参数 p 值、pair-IID 区间或等价检验结论。

同时输出同 checkpoint 的六 Recall/mR、best epoch、epoch4/5、最差 hard 值、epoch39、末5轮平均、梯度窗口、恢复次数及成本未知项。主終点保持 best full DEV；末值只解释稳定性。

本 v1 协议的工程投入规则是两项均差都至少 +3 rSum、各至少 2/3 seed 为正、严重塌陷不更多。代码从已核验的 `analysis_parameters.investment_rule` 读取数字并在结果中保存这些参数，不另写一组可能与协议漂移的阈值。没有满足它不能解释为两方法等价，也不能据它提前删除 seed。

## 6. 严重塌陷、恢复和成本

连续两个完整 hard epoch 满足下列条件才记严重塌陷：DEV < 本臂 epoch4 的50%；各自实际最后200更新里每20步抽样的 U-risk>=.39 比例至少90%。报告样本位置和数量。新日志直接使用 `step_in_epoch`；旧日志依据 `resumed_at` 重建当时从抽样日志条数开始的错误计数器，再恢复 epoch 内的真实位置。不能简单取最后10行或把旧 `step` 当全局更新数。

本函数专用于本轮的完整145000池，固定每 epoch1133更新；旧 OracleDrop 的缩小池不能直接用本轮函数判定完整性。若需要分析它，应使用原始池大小和相应更新数的已有 N2F 工具。

新结果逐个读取 attempts 账本，保存状态、job ID、起止时间、已知秒数和未闭合记录。缺失时长不记0。历史 epoch `seconds` 是每次 fit 内累计值；仅累加各 segment 的末值，作为已保存完整 epoch 的时间下界。最终 done.seconds 不能代替跨恢复总成本。Slurm allocation 成本在单独会计步骤按唯一 job 汇总，避免同一 job 内依次执行几臂时重复计费。

## 7. 本地验证的界限

本地 unittest 使用合成标签、权重、日志、路径和不是真实模型的 opaque checkpoint 字节，仅验证评分公式、舍入界、绑定顺序、完整性状态、paired seed 汇总与恢复位置算法。测试不会训练 SGR，不会验证 CUDA、完整数据文件或真实模型二进制。服务器的真实运行与原/补丁 trainer 数值等价检查仍须在分配的 GPU job 中完成。
