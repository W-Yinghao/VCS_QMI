# CS_QMI 接手入口

- 最新故障修复与执行（2026-10-19 02:17 UTC）：原GPU gate1047204于00:59 UTC因checkpoint恢复将AdamW的CPU step计数器搬到CUDA而失败，旧验证1047202退出；旧正式接续1047235空等上游，造成GPU断档。已在ecd8a95改为CPU反序列化后由module/optimizer各自恢复设备，并补AdamW/SGD下一步、优化器值/设备及RNG回归；新独立方法控制器1047321运行，94项CPU+17项调度检查通过，单方法失败不再阻断其它方法。随后MI/CoMIR的粗步长有限差分跨插值边界；GPU诊断1047351显示fp64局部差分与autograd一致，ddd3bcb仅修验证标准：两个fp64局部步长交叉核验、实际fp32梯度容差保持原值、正式执行仍fp32。单独恢复控制器1047364仅负责MI/CoMIR，97项CPU+18项控制检查通过；其余五方法仍归1047321，所有旧失败和冻结源保留。七方法GPU gate现全部真实PASS，NMI/MIND完整验证与正式执行已接上，其余方法全量训练/CAL继续。旧BF1已16/16阶段完成（仍待完整科学分析）；V9当前30/33完成、余3个GPU任务，收尾审计继续且不再阻塞V10。没有取消在跑任务。当前入口为 `handoff/V10_CURRENT_EXECUTION.json`；必须按其中逐方法所有权读取两个实时LEDGER。主控制器账本中的MI/CoMIR旧失败/blocked条目由1047364接手，不能误作新失败或再提交。两个冻结执行源 `ssl_pilot_v10_pipeline_20261019` 和 `ssl_pilot_v10_native_recovery_20261019` 不可修改；TEST_REPEAT/EPFL状态不变。

- 最新调度授权（2026-10-19 00:36 UTC）：用户明确“把所有能启动的都启动”。新增正式接续控制器1047235已运行，15项调度/放行检查与Slurm CPU92项回归均PASS。当前策略 `v10_ready_dispatch_20261019` 允许新增V10按方法验证通过即接正式实验，不再依赖V9收尾；V9全部33次、异常处理和独立分析仍须完成。既有BF1 dispatcher1046189与验证controller1047202继续独占各自作业；GPU gate1047204保留原ID并把nice500调整为0，尚待GPU实际验证。后续七个完整验证由1047202提交；新1047235对逐方法真实退出、报告和完整产物核验后生成冻结release，接7个CAL作业（19候选、仍受每族8 GPUh上限）及15个最终作业。可就绪时共享8卡、最多8个pending、normal总作业最多29，不取消在跑任务。现阶段正式新增作业0、科学release0，不能把已启动控制器称为所有实验已开始。正式源为 `ssl_pilot_v10_ready_20261019` @3d3ffd4；与5c04719验证源仅release授权函数不同，科学代码逐字节一致，验收迁移明确登记；禁止修改两个冻结工作树。TEST_REPEAT仍HOLD，EPFL仍未激活。见 `handoff/v10_ready_dispatch_20261019/ACTIVATION_VERIFIED.json`、POLICY.json、DELTA_REVIEW.json及动态STATUS.json。下述旧V9等待条件已被本授权覆盖。

- 最新验证记录（2026-10-19）：代码 `5c04719`，CPU Slurm1047202的92项检查全部通过；GPU门禁1047204已提交、尚未验证PASS。控制器1047202在门禁真实PASS后自动接七个完整训练/CAL验证任务（合计2320次完整配准），共享8卡、最多8个pending，不创建科学release。固定源为 `ssl_pilot_v10_validation_20261019`，不得修改；账本见 `handoff/v10_validation_20261019/ACTIVATION_VERIFIED.json`、LEDGER.json及STATUS.json。

- 最新调度与验证授权（2026-10-19 CEST）：用户询问GPU空闲并明确“现在也可以测试了”。现改为V9优先、已有BF1全量任务补空位，新增V10数值/GPU测试现在允许。旧stage_bridge1047062已因ACTIVE策略变更自行退出，未取消运行任务；1046917已释放并运行，M1最终seed新ID1047185/1047186/1047187由原dispatcher1046189提交。共享normal最多8 GPU；原PAUSE已归档。策略见 `handoff/v10_fill_validate_20261019/POLICY.json`，其覆盖下述旧“全部V9收尾后才能测试/提交”的时间限制。V9恢复1047108、收尾1047107、feeder1046651与审计1046165继续。七种新增方法仍需真实验证与正式冻结，不将验证作业当作科学结果；新测试状态见 `handoff/v10_validation_20261019/`。TEST134和EPFL规则不变。

- 最新代码进展（2026-10-19 CEST）：V10剩余七种基线首版及完整CAL/DEV执行入口已写入独立工作树 `ssl_pilot_v10_remaining`，提交 `d4b234f`；仅静态检查通过，Torch/CUDA/续跑/全量实验均未运行，正式release未生成。19个CAL候选+15个最终单元是待验证草案；EPFL、TEST_REPEAT、T03 RuLSIF接入及最终分析仍在后续工作范围。说明见 [V10补充实现](ssl_pilot_v10_remaining/v10/supplement/README_CN.md)，状态见 [实现账本](handoff/v10_remaining_implementation_20261018/STATUS.json)。现有V9优先调度不变。

- V9恢复更新（2026-10-19 CEST）：1046652在训练前因诊断ESS的1 ULP差异触发manifest字节检查；其它字段精确一致，34轮checkpoint保留。已提交恢复GPU1047108和CPU收尾1047107，未取消任何在跑任务。旧feeder1046651会自然结束；其failed_groups可能仍显示903_1，需同时读取 `handoff/v9_ess_resume_20261018/STATUS.json`，不要重复恢复或手改旧账本。GPU真实成功、完整产物核验后且原feeder退出，1047107才聚合并重建FEED_FINISHED；独立审计1046165和切换1047062继续等完整条件。

- 最新实现安排（2026-10-18）：用户允许“V10的实现可以先开始，测试之后来”。V10剩余方法现在在独立工作树 `ssl_pilot_v10_remaining`、分支 `v10/remaining-implementation-20261018` 实现；数值/GPU测试和新实验仍等V9收尾。现有V9优先调度、在跑任务及V10暂停标记不变。新代码尚未测试，不计为已验证方法；授权记录见 [AUTHORIZATION.json](handoff/v10_remaining_implementation_20261018/AUTHORIZATION.json)。这仅覆盖此前“编码也等V9完成”的时间限制。

- 最新优先级（2026-10-18 21:49 UTC）：先完成冻结V9全部33次训练、异常处理和独立结果分析，再推进V10剩余工作；禁止取消已运行作业。V10在跑1046192/1046915/1046916正常完成；pending1046917已hold；1046189仍运行但 `handoff/v10_parallel8_20261018/PAUSE_SUBMISSIONS` 已暂停新提交。V9 feeder1046651和独立审计1046165继续，当前25/33完成、5运行+3排队。CPU1047062（stage_bridge）在V9全量收尾PASS后自动释放其持有的hold、移走自己的pause并恢复既有BF1；不要手动提前release或移除pause。新策略见 `handoff/ACTIVE_SCHEDULING_POLICY.json` 和 `handoff/v9_first_20261018/ACTIVATION_VERIFIED.json`；后续若改策略必须更新ACTIVE_SCHEDULING_POLICY以阻止旧controller误放行。剩余V10实现全清单见同目录V10_REMAINING_WORK.json，尚未实施的代码不能说已自动接上。新决定覆盖此前尾部交叠补位；V9尾部不足8可就绪单元时不新增V10填空，也不扩展V9矩阵。

- 最新拆分已激活（2026-10-18 18:49 UTC）：用户授权拆开V9串行尾部。唯一V9 feeder现为1046651，替代已停止1045056；原903_1/903_2作业1045063/1045064已在checkpoint边界停止。VCS MEAN34/40→1046652，VCS SUM全40→1046654；BCE SUM33/40→1046653（H100已训练），BCE MEAN全40→1046655。四个作业均已release且无Dependency，切勿重复提交、重启旧feeder或恢复旧组作业。动态状态以 `handoff/v9_unit_split_20261018/UNIT_LEDGER.json`、`STATUS.json` 和原V9 LAUNCH/FEED_STATUS为准；核验见ACTIVATION_VERIFIED.json。1046651也处理1044282时限后的原配置一次基础设施续跑；CPU1046660只在controller证实TIMEOUT且真实退出9/15时补录缺失证据，无提交权、不增加重试。原33次矩阵不扩展；两组903配对已明确取消严格同GPU条件，必须报告硬件差异。V10 dispatcher1046189与独立审计1046165不变。


- 最新运行核验（2026-10-18 18:19 UTC）：V9完成23/33，7 GPU运行；V10 gate1045975已PASS，2族CAL完成，M1在跑，MLP CAL和6个最终seed共7任务等配额。现有loop1045056/1046189均正常；不得重复提交。1044068已正常结束；1044282预计到21:37 CEST超时，当前attempt1，原V9 feeder已有137/143退出后一次原配置checkpoint续跑规则，需核对后续SCHED_EXIT/新ID。证据见 `handoff/evidence/runtime_20261018_1818/STAGE_SNAPSHOT.json`。

- 最新授权（2026-10-18 13:29 UTC核验）：V9尾部释放的GPU立即接V10，V10最多8并行。旧数组1045976已在从未运行时取消；gate1045975保留且无V9依赖。新CPU调度器1046189负责4个CAL作业1046190–1046193及其后12个独立全量FINAL seed，动态ID以 `handoff/v10_parallel8_20261018/LEDGER.json` 为准。CAL内候选仍串行且每族8 L40S GPUh，只有4族可同时CAL；每族CAL成功并核验后其3个seed独立补位，可使CAL/FINAL混合及FINAL达到最多8并行，与V9共享normal8 GPU。loop每30秒补位，足够可就绪单元时目标8pending、最多16活跃GPU作业，normal总提交最多29。原科学冻结不变，明确调度覆盖见同目录SCHEDULING_AMENDMENT.json；不要重复启动feeder、重提旧all数组或重启旧占位watch。V9 feeder1045056和独立审计1046165照常。PAUSE_SUBMISSIONS只暂停新V10提交，不停止已有GPU任务；旧独立审计目录的PAUSE_AUTO_START只暂停V9审计。

- 历史顺序“V9干净了再启动V10”已被上述尾部交叠授权覆盖；V9仍需完成18组/33次、核验历史1042382、处理异常并分析收尾，这些现在是独立义务。历史文件 `handoff/V9_CLOSEOUT_BEFORE_V10_20261017.json` 已标记SUPERSEDED并记录新条件。保持原wave1暂停，不扩展V9，不删除实验产物。

开始本项目工作时，先读根目录 `PROJECT_MEMORY_CN.md`、`CODE_REVIEW_CN.md` 和 `V10_TAKEOVER_CN.md`，再核对 `handoff/V10_TAKEOVER_STATE.json` 与当前 Git 状态。它们是 2026-10-17 接手快照；后续用户决定和新冻结协议优先。

- 根目录不是 Git 仓库；`ssl_pilot` 为主仓库，其它 `ssl_pilot_*` 是不同分支的 linked worktrees。不要把某个树的结果当成所有分支状态。
- 保留用户已有未跟踪文件、实验输出和历史更正；正式结果与科学代码按 commit/content hash/protocol/manifest 对应。
- 科学训练、Torch 数值门禁和耗费资源的科学统计作业通过 Slurm；轻量文本、文件完整性及记录算术检查可在当前环境进行。无法查询 controller 时，不推断队列为空，不转到登录节点训练。
- 队列查询直接通过 exec_command 调用 `squeue ...`。2026-10-17 对照显示直接调用成功，timeout/Python subprocess 包装超时；包装失败不能直接归因为 Slurm 不可达。
- 用户最新要求：V9 用全量训练结果作判断，规划 normal QoS 的8张GPU并持续补位。读 `V9_STAGE_REVIEW_CN.md` 和 `handoff/v9_full_20261017/FEED_STATUS.json`；当前N2F调度器1046651负责33次全量训练，已替代1045056；1045056此前替代1043940；每30秒补足最多8个pending、本轮运行+排队最多16，normal数组逐项计数后总数最多29。不可重复启动feeder或提交已有组，交接见 `handoff/v9_dispatch_pending8_20261017/TRANSITION.json`。runfill 属其它项目，CPU的gres/gpu:0不计GPU；V10也共享同一normal配额。
- 历史调度决定（2026-10-17 20:03 UTC）：原V10 gate1044420及数组1044424已在启动前取消，旧watch1044425已停止，旧ID不能release。2026-10-18尾部交叠授权已取代“收尾后才提交/执行”的安排。先查gate1045975、CAL1046190–1046193及parallel8动态账本，避免重复提交；1045976已被拆分调度替代，不能恢复；禁止重启旧watch的V10优先占位策略。V9 loop1046651继续原矩阵，剩余组不足8时不新增V9实验凑数。
- V10 自然 BF 为 scene 内 V，T03 主目标为 U；不能静默互换。最新 accepted M1 为 relative_zero_head，具体规则见协议。
- TEST134 当前 HOLD、CONFIRM0、EPFL 未激活；遵循数据角色与未来 TEST_REPEAT 冻结条件。不得用 `evaluator_only` 内容进行训练、CAL 选择或事后改动冻结网格。
- 新 runner 在复用 COMPLETE 前校验配置、身份及产物；任何执行阶段的 BLOCKED/FAIL 都不能仅靠退出码掩盖。
- 延续已有授权，不重复询问已定数据许可或常规可逆操作；对外发送消息、发布和未授权 held-out 读取另按用户实际授权处理。
