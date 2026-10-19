# V9-N2F：全量复现实验与理论边界检验

2026-10-17。用户明确要求规划 8 GPU、持续安排全量实验；本协议取代本轮尚未提交的短程 risk probe 计划。原 V9-N2 冻结协议及其暂停标记保持独立。本协议先冻结、提交，再读取新增结果。既有 DEV 轨迹已参与本阶段设计，所以这是探索性机制复现，不是独立确认。

## 问题与已有证据

原 N2 在 rho=0.4 有多例 warmup→hard-negative 后崩塌；最新 A/A 重跑 C4 在 300 step 未复现崩塌，原尺度诊断按自己的规则不能给出 scale_fix_supported。seed917 调参最优 DEV rSum：RuLSIF 469.84、VCS 468.56、RankMass 467.36、BCE 465.05、不加权 439.90。带权有用途的迹象，但 VCS 特有优势尚无证据。

理论 J 的最优解是 eta=(p-q)/(p+q)，S-J=E_M(T-eta)^2。匹配 logistic 共享无限制最优解；有限样本参数拟合仍可不同。混合比例公式和 posterior 权重不自动保证完整 in-batch retrieval risk 的恢复：损失还读取学生表示、hard negatives 和被污染的负样本池。因此不能把完整训练成功归为某个已证明的无偏结论，也不能把崩塌直接归为 CS 的数学性质。

## 全量实验矩阵：33 次训练，15 个配对作业 + 3 个单次作业

只用 Flickr30K 全部 29,000 训练图像 / 145,000 captions、固定 10% clean anchors、rho=0.4、seeds 901/902/903。所有单位从头初始化，40 epoch，前 5 epoch mean-negative hinge，之后 max-violation；其余架构、lr=2e-4、batch=128、clip=2、epoch30 decay 均保留 HOST_LOCK。每 epoch 使用完整原 DEV 文件（1014 图像，保持原 runner 口径），按 DEV rSum 选 checkpoint；不读取 TEST/COCO。

主矩阵：VCS_Weight、BCE_Weight、RankMass、Unweighted_A × 3 seeds × {mean,sum} = 24 次。mean 是 mean(v*ell)/mean_v_full；sum 是 sum(v*ell)/mean_v_full。固定 x1 learning rate，不进行事后挑 lr。每个 method/seed 的两个尺度在同一个 Slurm job、同一 GPU 上顺序运行；交替尺度顺序是预先固定的，初始化和 shuffle seed 相同。新成对实验启用 deterministic algorithms、关闭 TF32/cuDNN benchmark，记录硬件与环境；不能把它与历史非确定性运行简单拼成一个随机实验。

附加全量参照：每个 seed 在 SUM 尺度各跑 OracleMask 和 OracleDrop，共 6 次。OracleMask 在 U 上使用 C=1{未错配}，A 权重为 1，错配对仍留在负样本池；OracleDrop 从训练池完全移除错配对，保留所有 A。两者是额外标签预算的不可部署参照，不能进入公平主排名。Drop 的每 epoch 样本量较小，40 epoch 的更新数随之减少，明确记录；它不是严格的性能上界。两者也同卡顺序运行。

提交前补充：RuLSIF 是现有调参中最强的对照，SUM 主尺度补入其 3 seeds 全量训练，避免独特优势判断遗漏强基线。它没有新的 MEAN 配对，不能从历史 MEAN 与新增 SUM 推断尺度因果效应。此前 30 次的草案尚未提交任何作业，冻结清单保留在 FROZEN_UNSUBMITTED_v1.json，修订原因记录于 PRE_SUBMISSION_AMENDMENT.json。

生产权重来自既有已锁定 CSV，绝不重拟合。oracle 标签仅在独立的 oracle 分支构造诊断权重/池后使用；传给通用 trainer 的行不含标签字段。每单位记录完整配置、输入 hash、源码 hash、协议 hash；COMPLETE 复用必须 identity 一致。所有实验产物写入本协议专属路径。

## 判断规则

主比较在每 seed 内报告 SUM−MEAN、VCS−BCE、VCS−RankMass、VCS−Unweighted 的最佳 DEV rSum 差及完整 40 epoch 轨迹；3 seeds 单独显示，报告均值、范围和方向一致性。DEV 的多次选择与 n=3 不支持宣称统计显著，不用图像 bootstrap 假装增加训练重复数。

预先定义“严重训练崩塌”：在 hard 阶段连续两个完整 epoch，DEV rSum < 同一次训练 epoch4 的 50%，同时该两 epoch 最后 200 training steps 的日志抽样点中 U-risk>=0.39 的比例各 >=90%。报告实际抽样点数。作为结果标注，仍保留训练到 40 epoch，除非非有限数、OOM、硬件故障或时间上限。失败必须计入分母，不从优胜表里删除。

投入决策是工程阈值而非统计显著性。SUM 预先定为新的主尺度，MEAN 为配对敏感性；二者冲突时必须报告，不取更有利者作为事先主分析：

* VCS 专有路线只有在完整 3 seeds 中 VCS−BCE、VCS−RankMass、VCS−RuLSIF 的平均差均 >=3 rSum、至少 2/3 seeds 同方向、且崩塌率不更高时，才值得设计独立确认。未满足则结束“VCS 独特优势”扩展。
* 通用可靠性加权路线只有在 VCS/BCE/RankMass/RuLSIF 中至少一种，相对不加权平均提升 >=5 rSum、至少 2/3 seeds 同方向、且不增加崩塌时，才值得后续独立确认。不可把这里选择的优胜者当预先选定方法。
* 若 OracleMask 明显好而估计权重不行，瓶颈是当前估计/使用权重的路径；若 OracleDrop 相对 Mask 平均提升 >=5 rSum，则污染的负样本池值得研究，但 Drop 同时改变池分布和更新数，不能宣称已孤立因果效应。
* 如果两种尺度下均无稳定带权收益，停止当前 full in-batch weighting 路线；不扩大 COCO 网格。即使有通用收益，也不据此声称新的 CS 泛函/新的比例估计理论。

## 8 GPU 调度和预算

当前 V9 已有 8 个 normal GPU 作业运行；其它项目在 runfill QoS 的 GPU 单独列示，不能混入本项目的 8 卡。已有 diag2、NCR-health、official-smoke 等排队任务继续。新增 feeder 只提交本协议组，最多维持 8 个本协议 GPU 作业在队列（含 pending）、其中最多 2 个 pending；normal QoS 的总 8 GPU 限额由 Slurm 强制。normal 队列作业数提交前须 <27，为记载的 30-job 限额保留余量，每次只提交 1 个长组，不一次占满 18 组。运行时观察全部 QoS 总队列已达31，因此不能把其它项目的 runfill 队列当作 normal 队列剩余槽位；这是调度更正，不改变科学矩阵。gres/gpu:0 的 feeder 也不计作 GPU。

RTX6000PRO、L40S、H100 均可运行这些单模型组，exclude=node51,node60。每组至多两次完整训练，time=23:59:00，8 CPU、64GB、1 GPU、nice=400。按实测单次成本 RTX 3.60h / L40S 6.47h / H100 NVL 7.21h，本矩阵约 115–230 GPU-h（OracleDrop 按约64%训练池估计，额外启动/验证和确定性运行开销另计）；异构 8 卡的日历时间由实际分配决定，不承诺 8 张 RTX 同时可用。各组运行时间不足时通过已保存 epoch checkpoint 续跑，最多一次基础设施重试；科学错误不自动换配方。

feeder 每 45 秒检查队列和 DONE/FAILED，槽位释放后补交。遇到提交结果不明确立即停止并留记录，避免重复任务。源码在提交前复制到 source/ 并由 SHA256 清单锁定，排队期间不跟随其它客户端修改。所有作业结束后自动汇总，最后据上述阈值形成下一次投入建议。
