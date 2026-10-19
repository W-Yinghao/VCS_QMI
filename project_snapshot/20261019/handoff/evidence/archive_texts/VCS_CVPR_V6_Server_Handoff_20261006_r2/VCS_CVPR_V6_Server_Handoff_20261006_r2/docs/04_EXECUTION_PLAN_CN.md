# 04｜围绕全文的执行计划

本文件把 V6 中文定位说明 §8 的三项后续工作变成接入任务。逻辑任务编号为本包新增 `[PROPOSED]`，不是仓库已有 P 编号，也不是既有批准记录。

## 1. 总体顺序

先做 `MV6-00` 接入；随后 `MV6-E0` 已有证据整理与规模化 SSL 衔接可以并行。测量端从 `MV6-I0` 分成两支：`MV6-I1` 完整测量器表，以及 `MV6-R0` 已有最终 SSL 表示分析表。R0 不必等新的 kernel/HSIC 全部补齐才整理已报结果；后续扩展在 `MV6-I2` 保留同样的两种比较轴。真实压缩 `MV6-C1` 在映射与 oracle 核对后可独立运行。不需要等 ImageNet 训练结束才做轻量测量，也不允许用大量测量实验挤掉规模化主证据。

| ID | 问题与动作 | 依赖 | 交付 | 默认是否需要新训练 |
|---|---|---|---|---|
| MV6-00 | V6→HEAD、运行状态、权限、数据与入口 | 无 | intake + reconciliation + 任务映射 | 否 |
| MV6-E0 | 复核 P86、P118，整理 estimator 图表 | 00 | fig/table source、claim ledger | 否；有具体缺口再开小单元 |
| MV6-S0 | 复核现有 CIFAR 与全部对照 | 00 | selected/shared/strong/readout 分开的表 | 默认否 |
| MV6-S1 | 衔接规模化 SSL 工程与完整训练 | 00、原授权 | 吞吐与梯度核对、完整协议、训练结果 | 可能；未授权则只做方案 |
| MV6-I0 | 原 P116 与 P109/P118 分别建档 | 00 | fixture registry、原数字复用表 | 否 |
| MV6-I1 | 固定 encoder，比较测量器 | I0、null代码核对 | Table3：VCS/JS/kernel/HSIC × 成本/检测 | 仅补缺测量器拟合；encoder冻结 |
| MV6-R0 | 固定测量流程，分析最终 VCS 与 SimCLR | I0 | Table4 已有配对条件表、null与来源核对 | **默认否；先复用已完成行** |
| MV6-I2 | 两步测量在外部 encoder／C100 上的最小扩展 | I1/R0协议、库存 | 测量器对照与编码器分析分表 | 首选已有 checkpoint；仅补缺行 |
| MV6-C1 | 真正 h→r→z 或 h→task-output 的依赖变化 | E0、映射单测 | signed decrements、D_T、拟合敏感性 | 仅测量器拟合 |
| MV6-W0 | 对齐论文、更新图表与补充 | 各已完成单元 | source patch、change log、完整可追溯表 | 否 |

“完成”指协议执行和证据齐全，不要求胜过 baseline。主比较相当、反向或统计不确定都可完成。技术失败、测量无效与科学负结果分开标记。

## 2. MV6-00：第一轮接入

读取 actual checkout 与对应 report/prereg/config，记录基准 snapshot、当前 HEAD、dirty 改动以及每个工作树的负责范围。不要清理别人的未提交文件；不要为本包回退仓库。查现有 job registry，并用本机调度器只读状态确认；未有权限就标注 unknown。

检查 dataset identities、split manifests、checkpoint 文件及其 hashes、预处理与 classifier provenance。报告级“完成”与原始产物齐备分开。新编号在主会话协调后分配；本包不预占 P142 等任何看似空闲编号。

第一轮结论为每单元的 `REUSE`、`CONTINUE_EXISTING`、`DRAFT_NEW`、`BLOCKED` 或 `OUT_OF_SCOPE`。具体原因必须可核对，不用一个总 PASS 代替清单。

## 3. MV6-E0 / S0：先把已有论文证据完整收好

Estimator 图1/2和压缩表1复用原条件与结果。对核方法保留 same-target 约定，MI methods 按 own target 和 resolution/readout 分组。误差条只来自 raw repeats/seeds；没有可用数据就保留均值并说明，不能从图像猜标准差。

SSL 表按 dataset、augmentation、scorer selection policy、pretrain预算、representation site、probe预处理分组。主结果仅一个 VCS。C10 五种子确认和三种子跨 dataset 表分别保留。不得把 raw-h 标准 probe 和 standardized readout 混在一列。

可以补充转录与代码路径测试；不重新开启已结束的自由 scorer、门控、noise 或 queue sweep 来“补贡献”。若当前后续工作已有获批结果，先登记其与当前 V6 主实现的关系，再决定进入主表还是 supplement。

## 4. MV6-S1：规模化 SSL，先衔接再训练

### 快照事实 [SNAPSHOT]

P132 已检查 ImageNet-1k 数据、两进程配对／梯度以及单卡吞吐，未训练。其报告指出 CPU Gaussian blur 输入管线限制和多GPU scaling 尚缺；不能把单卡计算吞吐当完整训练吞吐。入口 `src/vcs_ssl/imagenet.py` 和 `scripts/p132_imagenet_preflight.py` 已由该报告定位，现 HEAD 的接口需重新读取。

### 要交给负责人的协议 [PROPOSED]

明确数据集是 ImageNet-1k 还是具名且可复核的 ImageNet-100 子集；后者不能改标题成前者。完整记录 ResNet-50、projector、V、global/per-device B、image sampler、分布式梯度约定、增强、precision、优化器、lr和warmup、epochs、probe、seed与选择流程。V6没有冻结这些所有字段，使用现有批准配置；缺的列入 decision block。

最小完整对照组围绕 VCS、匹配 balanced JS、SimCLR。共享结构比较和方法各自成熟配方比较必须区分，不用“相同optimizer”宣称实现与累计调参预算完全公平。baseline不足时列明，而不是改变目标使它更弱。

仅在需要时补输入管线工程与 2/4 GPU scaling 单元：先核对增强语义、单机／DDP loss及梯度、吞吐和峰值内存，再形成 GPU-hour 预算。不要根据旧报告的历史 GPU/路径可用性直接提交。

获批后正式训练从初始化按完整日程进行。pilot用于代码/成本，不作为最终效果证据；同一日程故障恢复允许。提交 batch/seed 前先检查已有任务和同配置 checkpoint，避免双份运行。

## 5. MV6-I1 / R0：先检查测量，再理解 SSL

### I1：固定模型，改变测量器 [V6-R]

保留原 Table 3 的 P116 三格：blur.5/n1000、blur.5/n2000、colour.2/n2000。VCS 与短预算 logistic 是已完成对照；核和条件 HSIC 按相同 fixture 补缺。先恢复真实样本/特征/选择与置换定义，能复用则直接复用，不能将新 fixture 的核结果横拼到旧表。

这一轴回答 estimator 的使用表现。每个模型单独固定 checkpoint 和 head；误差、检测、成本按各自目标和协议比较。

### R0：固定测量流程，改变 encoder [V6-R；任务编号为交接组织]

新正文 Table 4 已有最终 VCS 与 SimCLR、encoder seeds 1/2、colour.1 结果。执行内容是来源恢复、行对齐、单位换算、null关联、最终表输出，不是新的预训练。数据来源与 expected display values 见 `reference_v6/evidence/retention_source_rows.json` 和 `contracts/comparison_axes.json`。

固定 VCS 测量统计及其拟合/选择规则；对不同 encoder 分别拟合 critic。检测 n=2000，100 次重复；对应预测响应用8000个 held-out base images。记录两种样本量，不把抽样重复当 encoder seeds，也不把整池响应伪装成某个检测子样本上的响应。

保持每一行的 dataset、encoder checkpoint/seed、attribute操作/强度、N|Y、head、EVAL池来源相符。不要将 Table 3 的 colour.2/blur.5 与 Table 4 的 colour.1 合并成一个共同显著性结论。blur.25 的已报行进入 supplement。

第一批交付是两张不同轴的结果表及其来源核对；Table4 不需要等待新增外部方法的所有运行。

## 6. MV6-I2：围绕 SSL 的最小扩展

[V6-R] 最终 VCS 与 SimCLR 是表示分析的主对象；同协议的 VICReg 或选定 logistic encoder 尚未补齐，不能填预测结果。原 r1 的第二外部 encoder/第二 dataset 覆盖需求继续保留。

[PROPOSED] 优先使用已有 provenance 完整的 VICReg 或选定 logistic checkpoint，并扩到已有 CIFAR-100。先按库存与协议完整性决定最小补齐组，不按EVAL检测分数挑模型；新协议在实际运行前冻结。可以复用同一组特征同时交付 I1-style 与 R0-style 表，但各轴结果仍分开。

encoder不存在或来源不完整时，列出该缺口与所需授权，不为凑解释性表启动长预训练。没有必要完成所有模型×数据集×属性的全因子组合才交付已有Table4。按预定范围保留正/零/反向结果，默认不加入像素热图、响应噪声或可解释架构。

## 7. MV6-C1：压缩测量的最小完整单元

复用 nested_exact，不改变原 J。先在现有P118真值链重现必要代码核对，不默认再跑整个全网格；随后在实际图像模型验证 h→r→z 及 h→固定输出。不是宣称 VCS 最少保留 nuisance，也不要求每一步下降。

先用库存中的最终 VCS 或外部 SimCLR 的一条已注册映射验证完整产物，再按预定两模型范围扩展；按可复用性而非效应方向选择先后。对 independent/nested、VCS/JS 保持共同底层pair和表示；报告 signed Delta J、D_T、重复拟合离散度及 paired bootstrap。不增加响应噪声来迫使数值变化。

## 8. 预算、停止与范围

本包不捏造GPU小时或总核时上限。`contracts/work_items.json` 的新工作预算为 null；服务器用原运行计时或固定规模 smoke测量填表，经授权再freeze。可将已有CPU测量器任务与GPU预训练并行，但不要把当前配额、节点健康状态和数据路径当作历史报告保证。

已有预注册的统计读法照原规则；新增条件先写规则。技术检查失败，修实现并记录；null calibration失败，保留结果并标注对应power不作有效主张，不通过改强度删除失败。科学负结果不触发自动追加seed、延长训练或无限调参。
