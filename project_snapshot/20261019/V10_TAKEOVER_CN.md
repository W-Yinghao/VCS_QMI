# V10 接手工作单

最新故障修复与执行（2026-10-19 02:17 UTC）：原GPU gate1047204于00:59 UTC因checkpoint恢复将AdamW的CPU step计数器搬到CUDA而失败，旧验证1047202退出；旧正式接续1047235空等上游，造成GPU断档。已在ecd8a95改为CPU反序列化后由module/optimizer各自恢复设备，并补AdamW/SGD下一步、优化器值/设备及RNG回归；新独立方法控制器1047321运行，94项CPU+17项调度检查通过，单方法失败不再阻断其它方法。随后MI/CoMIR的粗步长有限差分跨插值边界；GPU诊断1047351显示fp64局部差分与autograd一致，ddd3bcb仅修验证标准：两个fp64局部步长交叉核验、实际fp32梯度容差保持原值、正式执行仍fp32。单独恢复控制器1047364仅负责MI/CoMIR，97项CPU+18项控制检查通过；其余五方法仍归1047321，所有旧失败和冻结源保留。七方法GPU gate现全部真实PASS，NMI/MIND完整验证与正式执行已接上，其余方法全量训练/CAL继续。旧BF1已16/16阶段完成（仍待完整科学分析）；V9当前30/33完成、余3个GPU任务，收尾审计继续且不再阻塞V10。没有取消在跑任务。当前入口为 `handoff/V10_CURRENT_EXECUTION.json`；必须按其中逐方法所有权读取两个实时LEDGER。主控制器账本中的MI/CoMIR旧失败/blocked条目由1047364接手，不能误作新失败或再提交。两个冻结执行源 `ssl_pilot_v10_pipeline_20261019` 和 `ssl_pilot_v10_native_recovery_20261019` 不可修改；TEST_REPEAT/EPFL状态不变。

最新调度授权（2026-10-19 00:36 UTC）：用户明确“把所有能启动的都启动”。新增正式接续控制器1047235已运行，15项调度/放行检查与Slurm CPU92项回归均PASS。当前策略 `v10_ready_dispatch_20261019` 允许新增V10按方法验证通过即接正式实验，不再依赖V9收尾；V9全部33次、异常处理和独立分析仍须完成。既有BF1 dispatcher1046189与验证controller1047202继续独占各自作业；GPU gate1047204保留原ID并把nice500调整为0，尚待GPU实际验证。后续七个完整验证由1047202提交；新1047235对逐方法真实退出、报告和完整产物核验后生成冻结release，接7个CAL作业（19候选、仍受每族8 GPUh上限）及15个最终作业。可就绪时共享8卡、最多8个pending、normal总作业最多29，不取消在跑任务。现阶段正式新增作业0、科学release0，不能把已启动控制器称为所有实验已开始。正式源为 `ssl_pilot_v10_ready_20261019` @3d3ffd4；与5c04719验证源仅release授权函数不同，科学代码逐字节一致，验收迁移明确登记；禁止修改两个冻结工作树。TEST_REPEAT仍HOLD，EPFL仍未激活。见 `handoff/v10_ready_dispatch_20261019/ACTIVATION_VERIFIED.json`、POLICY.json、DELTA_REVIEW.json及动态STATUS.json。下述旧V9等待条件已被本授权覆盖。

- 最新验证记录（2026-10-19）：代码 `5c04719`，CPU Slurm1047202的92项检查全部通过；GPU门禁1047204已提交、尚未验证PASS。控制器1047202在门禁真实PASS后自动接七个完整训练/CAL验证任务（合计2320次完整配准），共享8卡、最多8个pending，不创建科学release。固定源为 `ssl_pilot_v10_validation_20261019`，不得修改；账本见 `handoff/v10_validation_20261019/ACTIVATION_VERIFIED.json`、LEDGER.json及STATUS.json。

- 最新调度与验证授权（2026-10-19 CEST）：用户询问GPU空闲并明确“现在也可以测试了”。现改为V9优先、已有BF1全量任务补空位，新增V10数值/GPU测试现在允许。旧stage_bridge1047062已因ACTIVE策略变更自行退出，未取消运行任务；1046917已释放并运行，M1最终seed新ID1047185/1047186/1047187由原dispatcher1046189提交。共享normal最多8 GPU；原PAUSE已归档。策略见 `handoff/v10_fill_validate_20261019/POLICY.json`，其覆盖下述旧“全部V9收尾后才能测试/提交”的时间限制。V9恢复1047108、收尾1047107、feeder1046651与审计1046165继续。七种新增方法仍需真实验证与正式冻结，不将验证作业当作科学结果；新测试状态见 `handoff/v10_validation_20261019/`。TEST134和EPFL规则不变。

最新代码进展（2026-10-19 CEST）：V10剩余七种基线首版及完整CAL/DEV执行入口已写入独立工作树 `ssl_pilot_v10_remaining`，提交 `d4b234f`；仅静态检查通过，Torch/CUDA/续跑/全量实验均未运行，正式release未生成。19个CAL候选+15个最终单元是待验证草案；EPFL、TEST_REPEAT、T03 RuLSIF接入及最终分析仍在后续工作范围。说明见 [V10补充实现](ssl_pilot_v10_remaining/v10/supplement/README_CN.md)，状态见 [实现账本](handoff/v10_remaining_implementation_20261018/STATUS.json)。现有V9优先调度不变。

最新实现安排（2026-10-18）：用户允许“V10的实现可以先开始，测试之后来”。V10剩余方法现在在独立工作树 `ssl_pilot_v10_remaining`、分支 `v10/remaining-implementation-20261018` 实现；数值/GPU测试和新实验仍等V9收尾。现有V9优先调度、在跑任务及V10暂停标记不变。新代码尚未测试，不计为已验证方法；授权记录见 [AUTHORIZATION.json](handoff/v10_remaining_implementation_20261018/AUTHORIZATION.json)。这仅覆盖此前“编码也等V9完成”的时间限制。

最新优先级（2026-10-18 23:49 CEST）：用户要求“在不取消已经在运行的任务的前提下，先完成V9的所有实现，然后开始实现V10剩余的所有东西”。已保留全部8个在跑GPU作业（V9为5个、V10为3个），没有取消任何运行任务；V10未运行的1046917已hold，1046189通过PAUSE_SUBMISSIONS暂停新提交，M1三个正式seed保持未提交。V9已完成25/33，剩8次为5运行+3排队，空出的配额优先接V9；V9不足8个活跃单元时不再用新增V10补位。新CPU控制1047062等全部33次、无V9在跑/排队、独立收尾1046165的核验与分析PASS后，自动释放本次hold并恢复现有BF1调度。V10缺失适配器、后续全量实验及EPFL/TEST_REPEAT条件已列入完整工作单，仍待V9收尾后实际编码/验证；Slurm控制器只恢复既有冻结作业，不会自动编写新方法。见 [调度核验](handoff/v9_first_20261018/ACTIVATION_VERIFIED.json) 与 [后续工作单](handoff/v9_first_20261018/V10_REMAINING_WORK.json)。该决定覆盖此前“V9尾部释放GPU立即接V10”，保留已运行V10继续完成这一例外。

拆分已落实（2026-10-18 20:49 CEST）：按用户“能拆开吗？”，VCS/BCE seed903两组拆为四个独立全量单元。VCS MEAN保留34/40轮→1046652，VCS SUM完整40轮→1046654；BCE SUM保留33/40轮→1046653（H100 NVL已进入训练），BCE MEAN完整40轮→1046655。四个任务均已解除hold且无Dependency；另外三个等待配额。旧串行1045063/1045064在新checkpoint落盘后停止并归档，旧feeder1045056由1046651替代，后者继续负责全18组和1044282原配置续跑。normal8卡仍满载：V9运行6、V10运行2，另V9排队3/V10排队6。原33次矩阵、科学代码及数据角色不变，但这两组不再严格同卡配对，分析必须保留硬件差异。1046660仅为1044282收集超时退出证据，不提交训练、不增加重试。独立收尾1046165和V10 dispatcher1046189继续。拆分前14–18小时ETA已标记历史，需按新任务实际分配重算，不能承诺减半。见 [拆分核验](handoff/v9_unit_split_20261018/ACTIVATION_VERIFIED.json)、[调度说明](handoff/v9_unit_split_20261018/README_CN.md)。

运行核验（2026-10-18 20:19 CEST）：normal8张GPU均有任务，V9运行7个、V10运行1个，另7个V10等待配额且无Dependency。V9已完成23/33次、11/18组；1044068于16:51 CEST正常结束，1044282的BCE902 SUM仅15/40轮且21:37到时，预计必须由原feeder按checkpoint续跑一次。V10 GPU gate1045975已PASS；perm1和M0两族CAL完成，M1正在第三个候选，合计8/18个全量CAL候选完成；6个最终seed已自动提交，尚无最终DEV结果。V9、V10 loop和独立收尾审计均正常，无记录到的失败。核验23份V9完整日志及69个元数据哈希、8份V10完整候选及424个JSON/JSONL哈希通过；此次未重新校验checkpoint二进制或重算预测。见 [阶段快照](handoff/evidence/runtime_20261018_1818/STAGE_SNAPSHOT.json) 和 [队列](handoff/evidence/runtime_20261018_1818/QUEUE.json)。下文较早运行数字保留为历史。

最新调度（2026-10-18 15:29 CEST核验）：按用户要求，V10可用并发上限改为8，与V9共享normal的8张GPU。旧四任务数组1045976已在RunTime=0时取消；gate1045975保留且无V9依赖。CPU调度器1046189已运行，四个CAL作业1046190–1046193已提交并依赖该gate。各族CAL仍按原顺序和每族8 L40S GPUh预算执行；该族CAL退出成功且产物核验通过后，自动补交seed1001/1002/1003三个独立全量FINAL作业，因此不再受“四个串行家族任务”限制。CAL阶段自身最多4路，CAL与已就绪FINAL混合及FINAL阶段最多8路；实际仍取决于空闲L40S。loop每30秒补位，可就绪任务足够时目标8个pending，本轮活跃GPU作业最多16、normal所有任务逐项计数最多29。源代码、18个CAL候选、12个FINAL、3000步与数据角色不改；额外单seed作业启动成本计入FINAL。V9原feeder1045056和独立审计1046165照常。见 [调度说明](handoff/v10_parallel8_20261018/README_CN.md)、[核验](handoff/v10_parallel8_20261018/ACTIVATION_VERIFIED.json) 和 [动态账本](handoff/v10_parallel8_20261018/LEDGER.json)。下文旧数组与较低并发是历史。

最新执行顺序（2026-10-18 15:13 CEST）：用户明确允许V9尾部与V10交叠。gate1045975已无V9依赖，等待空闲L40S；数组1045976只依赖gate1045975成功，并发上限为4。原协议execution.gpu_concurrency=2由 [单独调度修订](handoff/v9_overlap_v10_20261018/SCHEDULING_AMENDMENT.json) 覆盖，科学参数及每族预算不变。V9独立审计1046165替代旧1045974，继续完整核验和分析，不再阻止V10启动。现有GPU作业未重提，见 [激活记录](handoff/v9_overlap_v10_20261018/ACTIVATION.json)。下文“V9收尾后才执行”、旧CPU1045974和%2并发均属于历史。

2026-10-18 11:41 CEST接续授权更新：用户要求不只排4个任务，允许V9结束后接上V10。CPU配置作业1045973已成功预提交“CPU收尾1045974→GPU gate1045975→四族数组1045976”的afterok依赖链；CPU收尾作业RUNNING，5个V10 GPU任务均PENDING/Dependency，见 [接续记录](handoff/v9_to_v10_20261018/ACTIVATION.json) 与 [队列核验](handoff/v9_to_v10_20261018/ACTIVATION_VERIFIED.json)。仅预排时间改变，GPU仍等待V9的33次训练、完整性核验及结果分析。原科学冻结、18 CAL+12 final及数组0-3%2保持不变；禁止重复提交和重启旧watch。下文“收尾后重新提交”是旧安排，现以 [接续说明](handoff/v9_to_v10_20261018/README_CN.md) 为准。

最新调度记录（2026-10-17 20:03 UTC）：原V10 gate1044420及数组1044424（4个任务）已在启动前取消，释放5个提交名额；代码、数据锁、科学冻结和提交参数均保留。V9当前保持8运行+8排队。V9收尾并分析后按 [重新提交计划](handoff/v10_bf1_20261017/RESUBMISSION_PLAN.json) 提交新作业及新的afterok依赖，旧ID不能再release；下文17:14等时点的hold记录属于历史。

用户进一步明确“V9干净了再启动V10”：需完成当前18组/33次矩阵及历史在跑任务1042382，确认无相关V9排队/运行任务，处理异常并完成结果分析，再释放V10既有作业。剩余10组仍由原feeder补交。具体见 [收尾条件](handoff/V9_CLOSEOUT_BEFORE_V10_20261017.json)。

更新：2026-10-17；与 [项目记忆](PROJECT_MEMORY_CN.md)、[代码审查](CODE_REVIEW_CN.md) 配套。机器状态见 [V10_TAKEOVER_STATE.json](handoff/V10_TAKEOVER_STATE.json)。

**最新用户决定（17:14 UTC落实）：“不，还是要让V9的作业跑完”。先完成当前V9全量矩阵并分析，再启动V10 GPU实验。** BF1 driver和独立快照已完成，72个数据文件核验及5项控制测试通过；原gate1044420与四族全量数组1044424（18 CAL+12 final）已在20:03 UTC取消，待V9收尾后按保留配置重新提交。优先占位watch1044425已停止，V9 1044282的hold已解除，现由新feeder1045056按8个pending目标继续补位（19:57 UTC替代1043940）。GPU执行检查与完整训练尚未完成。当前权威入口见 [BF1 执行记录](V10_BF1_EXECUTION_CN.md)；下文原审计数字保留其对应身份。

**当前接手点：T03 已完成核验；BF1 执行链与冻结已准备，GPU作业暂缓等待V9收尾。** driver新增于 `scripts/v10_bf1/`，accepted数值core未改，独立source clone固定在b8f59ea。V9收尾后重新提交BF1前先核对队列和账本，避免重复提交；不重复启动T03主网格；不要重启旧watch的V10优先占位策略。

## 1. 本次实际完成

- 从原始理论、25 个压缩包、8 个工作树和各路线最终报告建立了项目记忆；保留已更正的负结果与数据曝光记录。
- 核验 T03 smoke 7、CAL 252、formal 960 单元；60/60 CAL 选择可从候选记录重建。对所有导出 manifest 条目及其列出的 2,466 个外部 artifact 条目检查哈希。
- 对 main/algo/capacity 的成员、描述性数字和配对差值独立复算；结果见 [T03 读数](handoff/evaluator_only/V10_T03_READOUT_CN.md) 及同目录 CSV。
- 用原函数的隔离控制流复现三个结果管理问题，未改变冻结的训练代码或旧结果。
- 生成 [BF1_NEXT_UNITS.csv](handoff/BF1_NEXT_UNITS.csv)：18 个 CAL 候选及 12 个最终 seed 单元，绑定最新预算；它是待执行台账，不是 runnable driver。
- 已将 BF1 工作树由 d57341e 快进到 accepted 4336bb6；只变更五个验证/计时/预算文件，core 内容身份未变，见 [同步记录](handoff/evidence/BF1_SYNC.json)。

## 2. accepted 身份与阶段状态

| 项 | 当前值 / 状态 | 权威入口 |
|---|---|---|
| 用户决定 | 2026-10-17 phase1 decisions | [CN 决定](V10_PHASE1_DECISIONS_20261017_CN.md)、[JSON](V10_PROTOCOL_DECISIONS_20261017.json) |
| 实现映射协议 | P2，`LANDED_NOT_FROZEN` | [P2](ssl_pilot_v10/v10/protocol/V10_PROTOCOL_P2_20261017.json) |
| accepted core 工作树 | `ssl_pilot_v10` @ `4336bb6` | 最新 T01/T02/预算 |
| BF1 工作树 | `ssl_pilot_v10_bf1` @ `b8f59ea` | 新增正式执行链，accepted core 身份不变；作业运行独立clone |
| T03 工作树 | `ssl_pilot_v10_t03` @ `0518984` | 正式结果提交1c13749；工作期间外部新增中文报告和生成器 |
| BF core code hash | `e9b4239450c4e2eb` | `src/v10 + scripts/v10` tracked 文件集合 |
| V9 host hash | `6c6f094684fbc50a` | 记录中的 host 桥接身份 |
| backend SHA-256 | `cdf5df2e43bacb74b963f7b6b519a962f91fc8e41f28b4f6a4e2696816024973` | 与服务器包原件相同 |
| T03 code / formal pin | `54606921cdd2bf15` / `64478d332c0a115a` | T03 独立内容身份；不改变 BF 的 hash 集合 |
| T01 | required 2,101/2,101 PASS | [Torch](ssl_pilot_v10/v10/T01/TORCH_VALIDATION.json)、[geometry](ssl_pilot_v10/v10/T01/GEOMETRY_GRADIENT_CHECK.json) |
| T02 | `MEASURED_PROJECTION`，relative-M1 已重测 | [BUDGET](ssl_pilot_v10/v10/T02/BUDGET.json) |
| T03 CAL | 252 COMPLETE；60 SELECTED | [manifest](ssl_pilot_v10_t03/v10/T03/CAL/CAL_RESULTS_MANIFEST.json) |
| T03 formal | 600 main +120 algo extra +240 capacity COMPLETE | [manifest](ssl_pilot_v10_t03/v10/T03/formal/FORMAL_RESULTS_MANIFEST.json) |
| BF1 正式运行 | 原数组1044424已在启动前取消，尚无结果 | driver/冻结完成；V9收尾后重新提交gate和数组，使用新的afterok依赖 |

T03 job：smoke 1043854，CAL 1043862，formal 1043864。后两者记录的 batch wall time 分别约 378s、1,127s，使用 `cpu-high`、32 个单线程进程。单元 wall time、总 CPU 时间与 batch elapsed 是不同量。

原项目最新 [T03 中文报告](ssl_pilot_v10_t03/v10/T03/V10_T03_REPORT_20261017.md) 在本次核对过程中新增；科学代码、结果文件和 aggregate SHA 均未因此改变。独立接手审计与它并存，不覆盖该外部工作。

旧 `V10_PHASE1_STAGE_REPORT_20261017.md` 是指定旧实现的历史报告，不在最新代码上重跑其生成器后覆盖原文。根目录决定中的 old cost projection 也已被新 T02 取代。

## 3. BF1 已定科学规则

| 维度 | 固定规则 |
|---|---|
| 四个首发方法 | MF_VCS_perm1、MF_VCS_M0、MF_VCS_M1、MLP_VCS_perm1 |
| 数据角色 | FIT32、CAL8、DEV32；CONFIRM0；TEST134 HOLD |
| 自然负对 | scene 内 V；普通均匀 permutation 允许 fixed points |
| 结构 | 既有审计 host encoders/MLP；MF K16、temperature1、无输出 tanh |
| 优化 | AdamW、clip1；非 W decay1e-4，W decay0，ridge0 |
| 批次 | 16 scenes×128 positions，crop160；共同 RNG/augmentation/geometry |
| 学习率 | 1e-4、3e-4、1e-3，所有由 optimizer 学习的模块绑定同一 lr |
| M1 | relative_zero_head，tol=0.01·当前批零头 gap；cap10/30/100；允许未收敛但必须记账 |
| 候选顺序 | family 内 lr 升序；M1 在每个 lr 内 cap 升序；共 3+3+9+3=18 |
| 训练 | 3,000 steps，每 250 step checkpoint，共12 |
| seed | tuning1017；最终1001/1002/1003 |
| CAL 选择 | <42 native px 成功率降序 → <9 成功率降序 → median corner error 升序 → 更早 checkpoint → 冻结候选序号 |
| 最终 seeds | 固定该族 tuning-CAL 选出的 lr/cap，仍各自在自己的 CAL 读数上按同规则选 checkpoint；不直接沿用 tuning seed 的 checkpoint |
| 预算 | 每族最多9候选或8 L40S allocated GPUh，先到为止；含 train/checkpoint/CAL/job 内开销；queue wait 单列 |
| 输出 | fixed-steps 表 + 按实际累计成本的视图；相同预算上限不等于相同实际花费 |

每个 tuning 候选有 12×8×10=960 次 CAL registrations；每个最终 seed 除自己的 CAL 外，再有 DEV32×10=320 次。registration 内固定 permutation，避免搜索过程中目标随抽样变化；不能为提高平滑性而暗改主协议。

### 最新成本预测

| 方法 | CAL 候选数 | 调参 GPUh 预测 |
|---|---:|---:|
| MF_VCS_perm1 | 3 | 0.8934 |
| MF_VCS_M0 | 3 | 0.9798 |
| MF_VCS_M1 | 9 | 6.4982 |
| MLP_VCS_perm1 | 3 | 1.3575 |
| 合计 | 18 | **9.7288** |

加入每族三个最终 seeds，在所选 M1 cap10/30/100 下，按对应 lr 的最大预测为 **15.4311 / 15.9285 / 17.6751 GPUh**。来源是最新 BUDGET 的 `projection`，不是 `old_timing_projection_reproduced` 或 `m1_cap_reference_projection`。

这不是实际已花费时间，也不包含缺失 baselines、EPFL、TEST_REPEAT、失败重跑等；BF1 driver 必须累计实际 allocated 时间并按上限停止，不能因结果不好延长。

## 4. 明确的下一执行顺序

1. **对齐 BF1 工作树证据：本次已完成。**accepted `4336bb6` 的最新 gate/timing/budget 已纳入 BF1，core hash 相同。后续共享实现若实际变化，必须补对应验证。
2. **正式 driver已实现，尚待GPU执行检查。**`scripts/v10_bf1/` 与 `slurm/v10_bf1/`，复用 `adapter.Trainer` 与完整registration搜索。含全量FIT训练、checkpoints、CAL选择、final-seed/DEV、成本累计、失败记录和身份/产物校验；本地5项控制测试通过。
3. **共同协议八项已冻结。**[PROTOCOL](handoff/v10_bf1_20261017/PROTOCOL.json) 与 [FROZEN](handoff/v10_bf1_20261017/FROZEN.json)绑定独立clone b8f59ea、58个source文件和72个激活数据文件。冻结SHA为a03e88277ba61159d4cd70255d72bf13515b8cac9843465329f6f9bcb15f58a2。
4. **原GPU执行检查1044420已在启动前取消。**V9当前轮完成并分析后重新提交gate与正式数组，登记新的依赖ID。四族在完整32 FIT驻留数据上核对checkpoint、RNG/optimizer续跑、拒绝失配、CAL完整搜索及评估后恢复训练；这只用于执行验证，不参与科学选择。正式数组依赖它afterok，同时runner检查相同freeze的GPU_GATE PASS。
5. **运行18个 CAL 候选。**保持每族候选顺序与预算规则，允许不同族在资源允许时独立调度。全部可用选择完成前不释放依赖它的最终单元；预算截断与失败如实标记。
6. **运行12个最终 seed 与 DEV。**报告 DEV32；另报排除 FIT/CAL 同 TMA position 的 DEV24 敏感性及 core/position-group 不确定性。所有变化以协议版本与结果来源可追踪。
7. **补齐比较方法。**见下节。四族首发允许技术进展报告；完整比较结论须等必需 baseline 齐全。

这个顺序是已授权工作的技术依赖，不是新的用户审批流程。本次没有把缺少 driver 的条目伪装成可提交任务。

## 5. 尚缺的方法和数据激活

剩余必需方法：MLP_VCS_perm8 registration；InfoNCE_host[J_perm1]；MI_native；NMI_native；CoMIR_native；exact-moment RuLSIF；MIND_2D_adapted。每个 adapter 需 targeted T01、适用的训练/配准 T02、冻结方法附件与同一共同协议。

InfoNCE_host 是 **用 InfoNCE 训练、用共同 J_perm1 energy 评估**，训练与评估对象都要写明。CoMIR_native 走 encode-then-register，计入训练和 latent 生成成本。V9 的 RFF+perm1 RuLSIF 不能替代 exact moments baseline。T03 RuLSIF follow-up 仍为后续任务。

BF TEST134 曾在 V9-G1 读取，当前仍 HOLD。须完整 V10 方法集合、共同协议及所有方法内选择全部冻结后，创建新 manifest，以 TEST_REPEAT134 单独报告；不能与 DEV32 合并为独立 confirmation。排除 FIT/CAL 同 position 的未来 TEST_REPEAT 敏感性集为112 cores；不能提前读取它来选方法。

EPFL 尚未激活。需补图像级 hash/解码、9×3 overlay QC、既有477→420筛除（33 alignment、24 size）的可追溯记录、重复组审查，以及 host replay/geometry/完整计时。V9 的运行记录不代替 V10 的激活材料。

## 6. 本会话的运行边界

2026-10-17 09:15 UTC 已更正连接判断：[直接 `squeue` 连续查询成功](handoff/evidence/SLURM_ACCESS.json)，同一时段 timeout/Python 包装查询超时。结合已批准命令前缀，判断是工具命令授权/沙箱路径差异；后续直接调用 `squeue`，不再把 Slurm 连接列为 BF1 阻塞项。17:14 UTC已确认V9使用8张normal GPU，BF1按用户要求暂缓，driver与冻结已完成，GPU路径验证待V9收尾后执行。当前工具对 `/projects/...` 可读但没有写权限；训练产物通过集群执行环境写入，不能绕到登录节点训练。

已经完成的独立工作包括文档、manifest/file audit、控制流复现、读数、候选台账，以及BF1 driver与冻结。当前先完成V9并分析，随后接续BF1 GPU执行检查与正式全量训练；不需要再次盘点全部历史，也不需要重复询问已有数据许可。

## 7. 后续更新本页的规则

新状态必须带 `UTC / worktree / commit / code identity / protocol / job / result manifest`。新结果出现后，先核验再更新阶段状态；不要只根据日志最后一行或进程退出码写 COMPLETE。历史报告保留，新数字从机器记录生成；对既有结论的更正写明旧说法和取代它的证据。
