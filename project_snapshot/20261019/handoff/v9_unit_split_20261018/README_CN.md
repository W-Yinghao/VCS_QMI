# V9 seed903 SUM/MEAN 独立调度

用户在查看V9尾部ETA后提出“能拆开吗？”。本次只拆VCS_Weight seed903和BCE_Weight seed903两组，不增加33次全量训练矩阵。CPU作业1046651先迁移，再作为唯一V9 feeder；旧1045056停止后不能重启。

| 训练 | 原组作业 | 独立作业 | 执行方式 |
|---|---:|---:|---|
| VCS_Weight903 MEAN | 1045063 | 1046652 | 从归档checkpoint接续到40轮 |
| VCS_Weight903 SUM | 1045063 | 1046654 | 原预定完整40轮 |
| BCE_Weight903 SUM | 1045064 | 1046653 | 从归档checkpoint接续到40轮 |
| BCE_Weight903 MEAN | 1045064 | 1046655 | 原预定完整40轮 |

迁移先在CPU Slurm验证原checkpoint中的模型、优化器、RNG、epoch序列和配置，再预排四个held作业。只有观察到下一轮last.pt原子落盘，才停止对应旧GPU作业；确认旧作业离开队列后，归档last/best、配置、进度、原硬件和退出记录并重新核验，随后自动release该组两个单元。短暂的JobHeldUser保护共享产物免受两个writer同时写入，不是V9整体完成依赖。迁移进度见TRANSITION_STATUS.json，实际完成的切换见ACTIVATION.json；未经这些记录核实，不应称已经完成。

训练核心继续使用原FROZEN.json绑定的WeightedStudent、全量FIT/DEV、原生产权重、seed、学习率和40epoch。新runner只选择单元，增加每单元独占锁、配置复用检查及独立退出记录；具体差异见RUNNER_DIFF.patch，独立调度代码由CONTROL_FREEZE.json冻结。8项控制流检查通过；真实checkpoint检查在CPU Slurm通过。它们用于确认迁移和结果管理正确，不替代全量实验结果。

原协议中这两组“同job、同GPU顺序运行”被显式覆盖。新作业保留RTX6000PRO/L40S/H100允许分区，可能更换物理GPU或型号。SUM—MEAN仍可报告为固定seed敏感性结果，但不能再称严格同卡对照或保证跨硬件逐位一致。原硬件和迁移checkpoint保存在checkpoints_before_split，终末硬件保存在每个full_result及units目录。科学冻结文件本身不修改，不把工程变更伪装成原运行方式。

新feeder继续处理全部18组，包括1044282到时限后的原配置一次基础设施续跑。只有记录到137/143、没有科学失败、未用完一次续跑时才自动补交；异常退出或无记录不会伪装成功。四个新单元全部用真实退出记录和产物哈希判定完成，一组两个单元都核验后才写组DONE。组SCHED_EXIT明确标记aggregate，job_id/utc保留最后完成GPU worker的真实退出时间，并嵌入两个单元退出记录；它不是CPU收尾时间。独立收尾1046165继续验证全部33次。

V10 dispatcher1046189及其已有作业照常，共享normal8张GPU。新feeder每30秒检查，目标最多8个V9pending、16个V9活跃GPU作业；normal全部CPU/GPU/数组任务逐项计数，最多29个提交作业。没有新增V9科学网格，不为凑pending重复提交，不改TEST134 HOLD。完整allocated GPU成本仍有会计缺口；原中断段与新单元的退出和硬件证据均保留，不把单个代表作业成本当整组成本。

拆分消除了串行的必要等待；实际结束时间取决于GPU排队和型号。已满载的8卡不会因拆分变成更多卡，不承诺结束时间减半。原20:24 CEST的14–18小时估计属于拆分前情景，迁移后应按新作业开始时间重算。

实际激活：2026-10-18 18:49:02 UTC。VCS保留34轮、BCE保留33轮，四个替代作业均已release且无依赖；1046653已在H100 NVL进入训练，其余等待配额。见ACTIVATION_VERIFIED.json。原作业SIGKILL后未必存在EXIT trap；CONTROLLER_ORIGINAL.json保留真实CANCELLED/ExitCode0:9，没有补造成功记录。额外CPU1046660只对1044282的真实TIMEOUT补齐可能缺失的退出证据，依据冻结脚本和5项边界检查，不改变原一次续跑上限。
