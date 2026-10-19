# V10 BF1 执行记录

最新优先级（2026-10-18 23:49 CEST）：用户要求“在不取消已经在运行的任务的前提下，先完成V9的所有实现，然后开始实现V10剩余的所有东西”。已保留全部8个在跑GPU作业（V9为5个、V10为3个），没有取消任何运行任务；V10未运行的1046917已hold，1046189通过PAUSE_SUBMISSIONS暂停新提交，M1三个正式seed保持未提交。V9已完成25/33，剩8次为5运行+3排队，空出的配额优先接V9；V9不足8个活跃单元时不再用新增V10补位。新CPU控制1047062等全部33次、无V9在跑/排队、独立收尾1046165的核验与分析PASS后，自动释放本次hold并恢复现有BF1调度。V10缺失适配器、后续全量实验及EPFL/TEST_REPEAT条件已列入完整工作单，仍待V9收尾后实际编码/验证；Slurm控制器只恢复既有冻结作业，不会自动编写新方法。见 [调度核验](handoff/v9_first_20261018/ACTIVATION_VERIFIED.json) 与 [后续工作单](handoff/v9_first_20261018/V10_REMAINING_WORK.json)。该决定覆盖此前“V9尾部释放GPU立即接V10”，保留已运行V10继续完成这一例外。

运行核验（2026-10-18 20:19 CEST）：normal8张GPU均有任务，V9运行7个、V10运行1个，另7个V10等待配额且无Dependency。V9已完成23/33次、11/18组；1044068于16:51 CEST正常结束，1044282的BCE902 SUM仅15/40轮且21:37到时，预计必须由原feeder按checkpoint续跑一次。V10 GPU gate1045975已PASS；perm1和M0两族CAL完成，M1正在第三个候选，合计8/18个全量CAL候选完成；6个最终seed已自动提交，尚无最终DEV结果。V9、V10 loop和独立收尾审计均正常，无记录到的失败。核验23份V9完整日志及69个元数据哈希、8份V10完整候选及424个JSON/JSONL哈希通过；此次未重新校验checkpoint二进制或重算预测。见 [阶段快照](handoff/evidence/runtime_20261018_1818/STAGE_SNAPSHOT.json) 和 [队列](handoff/evidence/runtime_20261018_1818/QUEUE.json)。下文较早运行数字保留为历史。

最新调度（2026-10-18 15:29 CEST核验）：按用户要求，V10可用并发上限改为8，与V9共享normal的8张GPU。旧四任务数组1045976已在RunTime=0时取消；gate1045975保留且无V9依赖。CPU调度器1046189已运行，四个CAL作业1046190–1046193已提交并依赖该gate。各族CAL仍按原顺序和每族8 L40S GPUh预算执行；该族CAL退出成功且产物核验通过后，自动补交seed1001/1002/1003三个独立全量FINAL作业，因此不再受“四个串行家族任务”限制。CAL阶段自身最多4路，CAL与已就绪FINAL混合及FINAL阶段最多8路；实际仍取决于空闲L40S。loop每30秒补位，可就绪任务足够时目标8个pending，本轮活跃GPU作业最多16、normal所有任务逐项计数最多29。源代码、18个CAL候选、12个FINAL、3000步与数据角色不改；额外单seed作业启动成本计入FINAL。V9原feeder1045056和独立审计1046165照常。见 [调度说明](handoff/v10_parallel8_20261018/README_CN.md)、[核验](handoff/v10_parallel8_20261018/ACTIVATION_VERIFIED.json) 和 [动态账本](handoff/v10_parallel8_20261018/LEDGER.json)。下文旧数组与较低并发是历史。

最新调度（2026-10-18 15:13 CEST）：按用户“V9尾部释放的GPU立即接V10”，现有gate1045975已解除对V9 CPU收尾的依赖，PENDING原因变为QOSMaxGRESPerUser；正式数组1045976保留afterok:1045975，并发上限2→4。原冻结execution.gpu_concurrency=2的覆盖见 [调度修订](handoff/v9_overlap_v10_20261018/SCHEDULING_AMENDMENT.json)，原科学文件和每族L40S预算不变。新独立V9审计1046165替代旧CPU1045974，继续核验分析但不限制V10。GPU任务仍为既有1045975/1045976，无重复提交；新状态见 [交叠激活](handoff/v9_overlap_v10_20261018/ACTIVATION.json)。下文旧顺序和%2是历史。

最新调度（2026-10-18 11:41 CEST）：按用户“也别只排4个任务，V9完成后可以提交V10”的指示，已预提交CPU收尾1045974、GPU gate1045975和数组1045976_0–3（0-3%2，18 CAL+12 final）。gate依赖1045974成功，数组依赖1045975成功；CPU收尾动态等待V9完整矩阵与续跑任务结束，核验并生成分析后退出0。当前V9仍8运行+4排队，V10另5个GPU任务等待Dependency；V10正式训练尚未开始。冻结科学代码和数据不变，TEST仍HOLD，旧watch不重启。权威记录见 [新提交](handoff/v9_to_v10_20261018/ACTIVATION.json)、[队列核验](handoff/v9_to_v10_20261018/ACTIVATION_VERIFIED.json) 和 [接续说明](handoff/v9_to_v10_20261018/README_CN.md)。下文旧ID及hold/取消记录保留为历史。

最新调度记录（2026-10-17 20:03 UTC）：原V10 gate1044420及数组1044424（4个任务）已在启动前取消，释放5个提交名额；代码、数据锁、科学冻结和提交参数均保留。V9当前保持8运行+8排队。V9收尾并分析后按 [重新提交计划](handoff/v10_bf1_20261017/RESUBMISSION_PLAN.json) 提交新作业及新的afterok依赖，旧ID不能再release；下文17:14等时点的hold记录属于历史。

最新启动条件：“让这些都结束吧，V9干净了再启动V10”。当前V9全部18组/33次训练及历史在跑任务1042382完成收尾、无相关排队/运行任务、异常核对和结果分析完成前，V10 gate及数组继续hold。剩余10组自动补交；[收尾条件](handoff/V9_CLOSEOUT_BEFORE_V10_20261017.json)已写入机器状态。

更新：2026-10-17 17:14 UTC / 巴黎19:14。用户最新明确要求：“不，还是要让V9的作业跑完”。现已恢复先完成当前V9全33次并分析、再启动V10 GPU实验的顺序。V10代码与冻结配置保留，17:07提交的GPU作业已暂缓，优先占位监控已停止。

## 已落地与已提交

| 项 | 当前状态 |
|---|---|
| BF1 driver | b8f59ea825dc7a2064457d47a628bef09ca4e185；新增 scripts/v10_bf1 和 slurm/v10_bf1 |
| accepted core | e9b4239450c4e2eb，未修改数值实现；新driver的hash另行冻结 |
| 隔离源快照 | handoff/v10_bf1_20261017/source，本地clone；队列作业不跟随开发工作树编辑 |
| 数据核验1044409 | PASS：72个已激活FIT32/CAL8/DEV32文件完整哈希、形状/有限值检查；TEST134未读取 |
| 控制流验证 | 5项通过：身份/产物拒绝、完整步数、CAL字典序、失败成本累计、网格/严格阈值 |
| 原GPU执行检查1044420 | 已在启动前取消；V9收尾后重新提交 |
| 原正式数组1044424 | 已在启动前取消；保留0–3%2及18 CAL+12 final配置，后续依赖新的gate ID |
| 原监控1044425 | 已停止；不得重启其优先占用V9槽位的策略 |
| 调度恢复1044429 | 已完成：暂缓V10、停止旧watch、解除V9 1044282的hold |

**已提交不等于已经运行，更不等于科学完成。** 本时点GPU检查尚未通过、完整训练尚无结果。

## 正式执行内容

四个数组任务依次对应 MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1、MLP_VCS_perm1。每族按已定顺序执行全部CAL候选，选择核验通过后，接续1001/1002/1003三个完整final seeds。不同族独立，最多两个GPU任务同时运行。

每次训练使用全部32个FIT cores、3000steps、每250steps一个checkpoint；每个checkpoint在完整CAL8×10扰动上评估。最终seed的lr/cap来自tuning选择，checkpoint按自己的CAL选取，之后评估DEV32×10并报告DEV24敏感性。任务不读取TEST134，不激活EPFL。

实际调用完整宿主rigid搜索，不以短程检查代替主实验。GPU gate只验证载入、续跑、几何搜索和恢复训练；不进入正式CAL候选或结果表。

完整规则：[PROTOCOL.json](handoff/v10_bf1_20261017/PROTOCOL.json)；身份：[FROZEN.json](handoff/v10_bf1_20261017/FROZEN.json)，SHA256 `a03e88277ba61159d4cd70255d72bf13515b8cac9843465329f6f9bcb15f58a2`。其中数据身份见 [DATA_LOCK.json](handoff/v10_bf1_20261017/DATA_LOCK.json)。

## 资源安排与故障边界

normal的8 GPU当前全部用于V9；19:57 UTC由新feeder1045056接替1043940，为已冻结的33次矩阵维持最多8个pending。新循环每30秒检查，normal数组逐项计数后最多29个作业；当前8运行+8排队、余2组待补交。旧V10 watch1044425已停止，V9 1044282的hold已解除。先完成并分析V9当前轮，再按保留计划重新提交V10 gate和数组；届时采用只读监控，不重启旧watch的优先占位策略。V10仍按最多2张L40S的冻结配置执行，以保持已测预算和数值环境一致。恢复证据见 [RESTORE_V9_PRIORITY.json](handoff/v10_bf1_20261017/RESTORE_V9_PRIORITY.json)。

每族CAL上限仍为8个L40S allocated GPU小时，包含启动、训练、checkpoints、CAL和失败尝试。阶段日志记录实际时间，Slurm整段allocation另行核对；CAL与final不得重复计费。预算中止的部分candidate保留；固定步数的主选择只使用3000步且12个CAL读数齐全的候选。没有完整候选时必须BLOCKED，不能以退出码或部分读数表示成功。

checkpoint保存完整模型对、optimizer、sampler及Torch RNG；复用COMPLETE需匹配配置、freeze、数据和产物哈希。失败保留；GPU gate失败时正式依赖不能放行。数值core原验证继续对应原identity，新driver的实际GPU路径需要本次检查。

## 接续入口

- [LAUNCH.json](handoff/v10_bf1_20261017/LAUNCH.json)：实际提交ID与依赖。
- [WATCH_STATUS.json](handoff/v10_bf1_20261017/WATCH_STATUS.json)：旧watch已停止的状态；不是实时监控。实时队列直接查询squeue，V9补位读FEED_STATUS。
- [READOUT.json](handoff/v10_bf1_20261017/READOUT.json)：进行中计数与读数；最终结论另需逐项验证及core/position分组不确定性分析。
- GPU检查完成后读取 `handoff/v10_bf1_20261017/GPU_GATE.json`，失败则读 `GPU_GATE_FAILURE.json` 与Slurm日志。
- 正式结果在 `handoff/v10_bf1_20261017/results/`，cost在 `COST_LEDGER.jsonl` 与 `SCHEDULER_ACCOUNTING.jsonl`。

四族结果仅回答首批结构/精确矩/头部优化问题。尚缺的原生与强基线适配仍按 [V10工作单](V10_TAKEOVER_CN.md)补齐，不能据首批四族提前作完整比较结论。
