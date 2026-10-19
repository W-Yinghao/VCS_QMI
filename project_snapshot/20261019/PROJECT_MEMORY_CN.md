# CS_QMI 项目记忆

最新故障修复与执行（2026-10-19 02:17 UTC）：原GPU gate1047204于00:59 UTC因checkpoint恢复将AdamW的CPU step计数器搬到CUDA而失败，旧验证1047202退出；旧正式接续1047235空等上游，造成GPU断档。已在ecd8a95改为CPU反序列化后由module/optimizer各自恢复设备，并补AdamW/SGD下一步、优化器值/设备及RNG回归；新独立方法控制器1047321运行，94项CPU+17项调度检查通过，单方法失败不再阻断其它方法。随后MI/CoMIR的粗步长有限差分跨插值边界；GPU诊断1047351显示fp64局部差分与autograd一致，ddd3bcb仅修验证标准：两个fp64局部步长交叉核验、实际fp32梯度容差保持原值、正式执行仍fp32。单独恢复控制器1047364仅负责MI/CoMIR，97项CPU+18项控制检查通过；其余五方法仍归1047321，所有旧失败和冻结源保留。七方法GPU gate现全部真实PASS，NMI/MIND完整验证与正式执行已接上，其余方法全量训练/CAL继续。旧BF1已16/16阶段完成（仍待完整科学分析）；V9当前30/33完成、余3个GPU任务，收尾审计继续且不再阻塞V10。没有取消在跑任务。当前入口为 `handoff/V10_CURRENT_EXECUTION.json`；必须按其中逐方法所有权读取两个实时LEDGER。主控制器账本中的MI/CoMIR旧失败/blocked条目由1047364接手，不能误作新失败或再提交。两个冻结执行源 `ssl_pilot_v10_pipeline_20261019` 和 `ssl_pilot_v10_native_recovery_20261019` 不可修改；TEST_REPEAT/EPFL状态不变。

最新调度授权（2026-10-19 00:36 UTC）：用户明确“把所有能启动的都启动”。新增正式接续控制器1047235已运行，15项调度/放行检查与Slurm CPU92项回归均PASS。当前策略 `v10_ready_dispatch_20261019` 允许新增V10按方法验证通过即接正式实验，不再依赖V9收尾；V9全部33次、异常处理和独立分析仍须完成。既有BF1 dispatcher1046189与验证controller1047202继续独占各自作业；GPU gate1047204保留原ID并把nice500调整为0，尚待GPU实际验证。后续七个完整验证由1047202提交；新1047235对逐方法真实退出、报告和完整产物核验后生成冻结release，接7个CAL作业（19候选、仍受每族8 GPUh上限）及15个最终作业。可就绪时共享8卡、最多8个pending、normal总作业最多29，不取消在跑任务。现阶段正式新增作业0、科学release0，不能把已启动控制器称为所有实验已开始。正式源为 `ssl_pilot_v10_ready_20261019` @3d3ffd4；与5c04719验证源仅release授权函数不同，科学代码逐字节一致，验收迁移明确登记；禁止修改两个冻结工作树。TEST_REPEAT仍HOLD，EPFL仍未激活。见 `handoff/v10_ready_dispatch_20261019/ACTIVATION_VERIFIED.json`、POLICY.json、DELTA_REVIEW.json及动态STATUS.json。下述旧V9等待条件已被本授权覆盖。

最新ETA（2026-10-19 02:10 CEST）：V9已27/33完成，剩6个GPU单元均在运行。按当前allocation内完整epoch耗时估计，VCS903 SUM（1046654）5/40、23.97分钟/轮，是总尾部，训练及单元落盘约还14小时，预计今天16:10 CEST；含独立核验与结果汇总按15–17小时规划，即17:00–19:00 CEST。此前10:30–14:30结束的估计已被本次实测取代：该SUM拆分后于22:02 UTC才开始，node52约24分钟/轮，明显慢于原node50约17分钟/轮。恢复任务1047108已37/40，ESS问题没有再次阻止训练。各作业均有足够Slurm时间余量，关键任务按当前速度仍余约7.8小时。此区间以不发生新失败/重启、吞吐近似稳定为条件；收尾0.5–1小时为规划余量。详见 [当前实测ETA](handoff/evidence/V9_ETA_20261019_0011.json)。

- 最新验证记录（2026-10-19）：代码 `5c04719`，CPU Slurm1047202的92项检查全部通过；GPU门禁1047204已提交、尚未验证PASS。控制器1047202在门禁真实PASS后自动接七个完整训练/CAL验证任务（合计2320次完整配准），共享8卡、最多8个pending，不创建科学release。固定源为 `ssl_pilot_v10_validation_20261019`，不得修改；账本见 `handoff/v10_validation_20261019/ACTIVATION_VERIFIED.json`、LEDGER.json及STATUS.json。

- 最新调度与验证授权（2026-10-19 CEST）：用户询问GPU空闲并明确“现在也可以测试了”。现改为V9优先、已有BF1全量任务补空位，新增V10数值/GPU测试现在允许。旧stage_bridge1047062已因ACTIVE策略变更自行退出，未取消运行任务；1046917已释放并运行，M1最终seed新ID1047185/1047186/1047187由原dispatcher1046189提交。共享normal最多8 GPU；原PAUSE已归档。策略见 `handoff/v10_fill_validate_20261019/POLICY.json`，其覆盖下述旧“全部V9收尾后才能测试/提交”的时间限制。V9恢复1047108、收尾1047107、feeder1046651与审计1046165继续。七种新增方法仍需真实验证与正式冻结，不将验证作业当作科学结果；新测试状态见 `handoff/v10_validation_20261019/`。TEST134和EPFL规则不变。

V9恢复更新（2026-10-19 CEST）：1046652在训练前因诊断ESS的1 ULP差异触发manifest字节检查；其它字段精确一致，34轮checkpoint保留。已提交恢复GPU1047108和CPU收尾1047107，未取消任何在跑任务。旧feeder1046651会自然结束；其failed_groups可能仍显示903_1，需同时读取 `handoff/v9_ess_resume_20261018/STATUS.json`，不要重复恢复或手改旧账本。GPU真实成功、完整产物核验后且原feeder退出，1047107才聚合并重建FEED_FINISHED；独立审计1046165和切换1047062继续等完整条件。

最新代码进展（2026-10-19 CEST）：V10剩余七种基线首版及完整CAL/DEV执行入口已写入独立工作树 `ssl_pilot_v10_remaining`，提交 `d4b234f`；仅静态检查通过，Torch/CUDA/续跑/全量实验均未运行，正式release未生成。19个CAL候选+15个最终单元是待验证草案；EPFL、TEST_REPEAT、T03 RuLSIF接入及最终分析仍在后续工作范围。说明见 [V10补充实现](ssl_pilot_v10_remaining/v10/supplement/README_CN.md)，状态见 [实现账本](handoff/v10_remaining_implementation_20261018/STATUS.json)。现有V9优先调度不变。

最新实现安排（2026-10-18）：用户允许“V10的实现可以先开始，测试之后来”。V10剩余方法现在在独立工作树 `ssl_pilot_v10_remaining`、分支 `v10/remaining-implementation-20261018` 实现；数值/GPU测试和新实验仍等V9收尾。现有V9优先调度、在跑任务及V10暂停标记不变。新代码尚未测试，不计为已验证方法；授权记录见 [AUTHORIZATION.json](handoff/v10_remaining_implementation_20261018/AUTHORIZATION.json)。这仅覆盖此前“编码也等V9完成”的时间限制。

最新优先级（2026-10-18 23:49 CEST）：用户要求“在不取消已经在运行的任务的前提下，先完成V9的所有实现，然后开始实现V10剩余的所有东西”。已保留全部8个在跑GPU作业（V9为5个、V10为3个），没有取消任何运行任务；V10未运行的1046917已hold，1046189通过PAUSE_SUBMISSIONS暂停新提交，M1三个正式seed保持未提交。V9已完成25/33，剩8次为5运行+3排队，空出的配额优先接V9；V9不足8个活跃单元时不再用新增V10补位。新CPU控制1047062等全部33次、无V9在跑/排队、独立收尾1046165的核验与分析PASS后，自动释放本次hold并恢复现有BF1调度。V10缺失适配器、后续全量实验及EPFL/TEST_REPEAT条件已列入完整工作单，仍待V9收尾后实际编码/验证；Slurm控制器只恢复既有冻结作业，不会自动编写新方法。见 [调度核验](handoff/v9_first_20261018/ACTIVATION_VERIFIED.json) 与 [后续工作单](handoff/v9_first_20261018/V10_REMAINING_WORK.json)。该决定覆盖此前“V9尾部释放GPU立即接V10”，保留已运行V10继续完成这一例外。

拆分已落实（2026-10-18 20:49 CEST）：按用户“能拆开吗？”，VCS/BCE seed903两组拆为四个独立全量单元。VCS MEAN保留34/40轮→1046652，VCS SUM完整40轮→1046654；BCE SUM保留33/40轮→1046653（H100 NVL已进入训练），BCE MEAN完整40轮→1046655。四个任务均已解除hold且无Dependency；另外三个等待配额。旧串行1045063/1045064在新checkpoint落盘后停止并归档，旧feeder1045056由1046651替代，后者继续负责全18组和1044282原配置续跑。normal8卡仍满载：V9运行6、V10运行2，另V9排队3/V10排队6。原33次矩阵、科学代码及数据角色不变，但这两组不再严格同卡配对，分析必须保留硬件差异。1046660仅为1044282收集超时退出证据，不提交训练、不增加重试。独立收尾1046165和V10 dispatcher1046189继续。拆分前14–18小时ETA已标记历史，需按新任务实际分配重算，不能承诺减半。见 [拆分核验](handoff/v9_unit_split_20261018/ACTIVATION_VERIFIED.json)、[调度说明](handoff/v9_unit_split_20261018/README_CN.md)。

ETA更新（2026-10-18 20:24 CEST）：按最新完整epoch耗时重算，剩余训练中央估计13.66小时，约10月19日10:03 CEST结束；连同续跑排队、吞吐波动及0.5–1小时核验/汇总，按14–18小时规划，约10月19日10:30–14:30 CEST收尾。决定总时长的是seed903的VCS/BCE两组，均还有当前训练尾部和另一条完整40轮。1044282预计在时限前保存到18/40，续跑剩22轮；按原慢L40S速度，若超时后等待2/4/6小时再启动，则分别约08:40/10:40/12:40 CEST完成。排队和目标GPU未知，此区间不是调度保证。没有改训练、重提任务或改变V10交叠安排。详见 [实测ETA](handoff/evidence/V9_ETA_20261018_1824.json)。

运行核验（2026-10-18 20:19 CEST）：normal8张GPU均有任务，V9运行7个、V10运行1个，另7个V10等待配额且无Dependency。V9已完成23/33次、11/18组；1044068于16:51 CEST正常结束，1044282的BCE902 SUM仅15/40轮且21:37到时，预计必须由原feeder按checkpoint续跑一次。V10 GPU gate1045975已PASS；perm1和M0两族CAL完成，M1正在第三个候选，合计8/18个全量CAL候选完成；6个最终seed已自动提交，尚无最终DEV结果。V9、V10 loop和独立收尾审计均正常，无记录到的失败。核验23份V9完整日志及69个元数据哈希、8份V10完整候选及424个JSON/JSONL哈希通过；此次未重新校验checkpoint二进制或重算预测。见 [阶段快照](handoff/evidence/runtime_20261018_1818/STAGE_SNAPSHOT.json) 和 [队列](handoff/evidence/runtime_20261018_1818/QUEUE.json)。下文较早运行数字保留为历史。

最新调度（2026-10-18 15:29 CEST核验）：按用户要求，V10可用并发上限改为8，与V9共享normal的8张GPU。旧四任务数组1045976已在RunTime=0时取消；gate1045975保留且无V9依赖。CPU调度器1046189已运行，四个CAL作业1046190–1046193已提交并依赖该gate。各族CAL仍按原顺序和每族8 L40S GPUh预算执行；该族CAL退出成功且产物核验通过后，自动补交seed1001/1002/1003三个独立全量FINAL作业，因此不再受“四个串行家族任务”限制。CAL阶段自身最多4路，CAL与已就绪FINAL混合及FINAL阶段最多8路；实际仍取决于空闲L40S。loop每30秒补位，可就绪任务足够时目标8个pending，本轮活跃GPU作业最多16、normal所有任务逐项计数最多29。源代码、18个CAL候选、12个FINAL、3000步与数据角色不改；额外单seed作业启动成本计入FINAL。V9原feeder1045056和独立审计1046165照常。见 [调度说明](handoff/v10_parallel8_20261018/README_CN.md)、[核验](handoff/v10_parallel8_20261018/ACTIVATION_VERIFIED.json) 和 [动态账本](handoff/v10_parallel8_20261018/LEDGER.json)。下文旧数组与较低并发是历史。

最新调度记录（2026-10-17 20:03 UTC）：原V10 gate1044420及数组1044424（4个任务）已在启动前取消，释放5个提交名额；代码、数据锁、科学冻结和提交参数均保留。V9当前保持8运行+8排队。V9收尾并分析后按 [重新提交计划](handoff/v10_bf1_20261017/RESUBMISSION_PLAN.json) 提交新作业及新的afterok依赖，旧ID不能再release；下文17:14等时点的hold记录属于历史。

最新补充：“让这些都结束吧，V9干净了再启动V10”。当前已提交8组、未提交10组（18次全量训练）均须收尾，历史在跑任务1042382也包括在内。V10保持hold，直到相关V9运行/排队任务清空、调度器结束、异常处理与完整结果分析完成。原wave1暂停和实验产物保留；[收尾条件](handoff/V9_CLOSEOUT_BEFORE_V10_20261017.json)是后续放行依据。

更新：2026-10-17 17:14 UTC。用途：后续接手者的入口，不替代冻结协议、原始结果或历史修订。项目主线是 **V10 Moment-Factorized VCS（MF-VCS）**。用户最新明确要求：“不，还是要让V9的作业跑完”。执行顺序为 **先完成当前 V9 的33次全量训练并分析，再启动 V10 GPU实验**；V9 信息扩展、重匹配、低标注预算等新网格仍不启动。V10 driver和冻结配置已准备好，原已提交的GPU作业已在启动前取消，待V9收尾后重新提交；提前为V10占位的调度器已停止，V9正常补位已恢复。V10正式入口和作业见 [BF1 执行记录](V10_BF1_EXECUTION_CN.md)，完整接续见 [V10 工作单](V10_TAKEOVER_CN.md)。

这是一条从“混合参考下的 CS 散度及其二次变分表示”，经过 SSL、估计器、条件信息诊断、视觉语言任务，再转向结构化 critic 与精确矩计算的研究链。项目尚未建立“VCS 普遍优于 logistic/InfoNCE”或“提出全新散度”的证据。V10 的可检验贡献是结构、计算和优化，而不是重新命名已有目标。

## 1. 现在到哪里了

| 工作 | 已有证据 | 接手状态 |
|---|---|---|
| SSL / A-P3 | 已有 CIFAR-10 官方测试评估；matched JS 同样很强 | 历史主线，按既有决定不重新开启 SSL 搜索 |
| 估计与条件诊断 | P156–P159 为较新的补充，不能只读 10/14 项目地图 | 保留有条件的正结果和冻结规则下的负结果 |
| R4/R5 定位与查询归一化 | 训练、DEV、官方 test 已完成 | 不再称官方 test 未读 |
| R6 REG / GAIN | 配准成为 V9/V10 host 的来源；GAIN 有数据与任务边界 | 复用已审计 host，不能直接移植旧结论 |
| R7/D1 token 选择 | 10/17 最终修订已收束，REV10 = MIXED | 读 closeout 与更正，不复述早期过强归因 |
| V7 T2I / V8 / V8.1 | V7 收束；V8.1 C1 未达到救援规则 | 已关闭，不能把 V8 epoch 0 当未训练模型 |
| V9 N1 / G1 / G2 | N1 比例估计未支持独特优势；G1/G2 有配准收益 | 结论限定在各自协议；TEST134 已在 V9 用过 |
| V9 N2 / N2F | 原 N2 可见40个 student 完成、6个未完成；带权有收益迹象，多方法崩塌 | N2F冻结33次全量训练；19:57 UTC新调度器1045056接手，16组已提交（8运行+8排队），2组待补交；新增完整训练3次 |
| V10 T00/T01/T02 | 数据角色明确；最新必需验证记录 PASS；relative-M1 重测完成 | 使用最新 JSON，旧阶段报告仍保留为历史快照 |
| V10 T03 | smoke 7、CAL 252、formal 960，共 1,219 单元完成 | 本次逐项核验通过，并补了独立读数报告 |
| V10 BF1 | 正式 driver 与独立快照已冻结；72个数据文件核验及5项控制测试通过 | **原gate1044420与四族数组1044424已取消；待V9收尾后重新提交18 CAL+12 final，仍须新gate通过；尚无GPU检查或正式结果** |

实时状态入口：[V10 接手工作单](V10_TAKEOVER_CN.md)、[机器状态](handoff/V10_TAKEOVER_STATE.json)。T03 的科学结果放在 [evaluator-only 读数](handoff/evaluator_only/V10_T03_READOUT_CN.md)，不作为训练或 CAL 的输入。

## 2. 证据优先级和本次覆盖

处理冲突时依次看：用户最新明确决定 → 冻结协议及更正 → 对应版本的机器结果、manifest 和日志 → 汇总报告 → 历史计划与 Claude 记忆。结果说明“实际发生了什么”；协议说明“允许如何解释”，两者都不可缺少。

本次建立了 25 个根目录 ZIP 的 CRC、逐文件 SHA-256 清单，共 964 个非目录成员；提取了 5 份根目录 PDF；盘点 8 个 Git 工作树。全量检查不等于逐行审查每份历史副本。深读重点是合作者理论、各轮路线变化、最终报告，以及 V10 backend、adapter、T01/T02、T03 训练/选择/评估/调度代码；历史训练实现做了有针对性的抽查。

- [压缩包清单](handoff/evidence/archive_inventory.json)：25 包 CRC 无失败，包含每个成员路径、哈希和 Markdown 标题。
- [PDF 清单](handoff/evidence/pdf_inventory.json)、[工作树清单](handoff/evidence/worktree_inventory.json)。工作树清单记录了本次开始时已有的未跟踪文件，不能清理掉它们。
- [V10 核验结果](handoff/evidence/V10_AUDIT.json)：1,219 单元的配置、代码身份、CAL 选择、数据/初始化配对、汇总算术一致；2,590 个导出文件条目和 2,466 个外部 artifact 条目通过哈希检查。外部条目合计 377,160,053 字节，可能包含跨快照重复条目。
- 旧路线数字以下均明确作为**历史报告结果**引用；没有重新训练所有历史实验。V10 本次是记录与文件完整性核验，没有重新执行 Torch 数值门禁或反序列化 checkpoint 做前向复算。
- Claude 旧记忆位于 `/home/infres/yinwang/.claude/projects/-home-infres-yinwang-CS-QMI/memory/`，可用于定位历史，但其中的日期、队列状态和中途判断不能覆盖上述证据。

## 3. 从合作者理论理解项目

最早入口是 [Variational CS-QMI Research Plan](<Variational_CS_QMI_Research_Plan (1).pdf>)；其后是 [Draft v2](VCS_QMI_CVPR_Draft_v2.pdf)、[Draft v3](VCS_QMI_CVPR_Draft_v3.pdf) 和 V6 包内 manuscript。`2409.04747v3.pdf` 是 *Explicit Mutual Information Maximization for Self-Supervised Learning*，是外部 SSL/MI 参考，不是合作者的原始理论。

### 3.1 核心对象及精确恒等式

对同一空间上的分布 P、Q，令 M=(P+Q)/2，密度 p、q 相对同一支配测度定义。定义

\[
\eta=\frac{p-q}{p+q},\qquad
S(P,Q)=\mathbb E_M\eta^2=\frac12\int\frac{(p-q)^2}{p+q}.
\]

零质量处的 η 可任意定义。S 是三角判别散度的一半，取值 [0,1]。令

\[
J(T)=\mathbb E_PT-\mathbb E_QT-\tfrac12\mathbb E_PT^2-\tfrac12\mathbb E_QT^2.
\]

配方直接给出

\[
S-J(T)=\mathbb E_M(T-\eta)^2,\qquad
\sup_T J(T)=S,\qquad T^*=\eta.
\]

平衡二分类平方风险为

\[
R(T)=\tfrac12\mathbb E_P(1-T)^2+\tfrac12\mathbb E_Q(-1-T)^2=1-J(T).
\]

这是项目最可靠的理论资产：**总体 J 的差距恰好是 posterior 的加权平方误差**。它没有自动保证 SGD 更好、学到的 encoder 更好，或下游任务更好。

原计划采用平方比值版本的 CS 定义。在参考测度取 M 时，p/M=1+η、q/M=1−η，交叉项为 1−S，两个平方项均为 1+S，因此

\[
D_{CS}^{M}(P,Q)=2\log\frac{1+S}{1-S}.
\]

这里的 **M 随 P、Q 改变**。它不能冒充固定 Lebesgue 参考下的传统 KDE-CS；比较时必须写清参考测度和定义中的系数。S 接近 1 时，上式导数 4/(1−S²) 会放大估计误差。

若 P=PXY、Q=PX⊗PY，则 η=tanh(PMI/2)，并在可逆处 p/q=(1+η)/(1−η)。S 衡量依赖性，但 **S 不是 Shannon MI**。有界 T 的 J 也可以为负（一般范围 [-3,1]）；不要为了画图或报告把负的经验 J 截到零。

S 作为 f-divergence 对共同随机映射满足数据处理不等式，对可逆观测变换保持不变。这说明压缩不能增加真实可辨别信息；它没有证明某个压缩算法保留了任务效用，也没有给训练得到的有限 critic 同样的精确不变性。

### 3.2 与已有方法的关系

balanced RPC 在 α=β=γ=1 时就是上述 J。RuLSIF 在 α=1/2、g=1+T 时，其平方风险满足

\[
\tfrac12\mathbb E_Mg^2-\mathbb E_Pg=-\tfrac12-\tfrac12J(T).
\]

这是直接代入得到的对应关系；不能把二次目标本身作为新方法优先权。原始文献：[RPC](https://arxiv.org/abs/2103.11275)、[RuLSIF](https://www.ms.k.u-tokyo.ac.jp/sugi/2013/RuLSIF.pdf)。项目 r4 对这些关系已作修订。

matched logistic 用 D=σ(2f)、T=tanh(f) 时也以 η 为总体最优。二次风险对 f 的梯度含 `(1−T²)`，logistic 的相应梯度没有这个额外因子（公共尺度随风险归一化变化）。因此“同一总体最优”可以解释大量归因对照，但**既不证明有限样本等价，也不证明任何一方永远不能胜出**。参数化限制、校准、正则、选择和优化都会改变实际结果。

### 3.3 统计与条件信息边界

总体下界 J≤S，不等于独立 EVAL 上每个随机经验值都≤S；更不等于在 FIT 上拟合后的经验值有无条件覆盖保证。重用同一批样本的所有 n² 对，不会创造 n² 个独立样本。

对 iid 配对观测 `(xi,yi)`：V 负对包括全部 n² 对；U 负对排除 i=j，分母 n(n−1)。普通均匀排列允许 fixed points，其条件均值是 V；均匀 derangement 的条件均值是 U。T03 用后者，自然配准沿用 host 的前者，不能“统一修正”为同一种。

条件/嵌套分辨率的 L² 投影恒等式对真实 posterior 成立。经验拟合的差值可能不单调、正交残差可能不为零；不能把一个看起来非负的曲线当成条件充分性的证明。scene 内 Q 也不等于全局 product；若 critic 未观察 scene 身份，混合分布目标通常也不等于逐 scene 理想最优值的平均。

## 4. 压缩包发展脉络

下表按研究演进排列；文件全名见链接。**SSL 的 v7、稿件 V6 的 r7、T2I 的 V7 是三套编号**。r1–r8 也不是八份独立模型。

| 包 | 关键变化及后续落点 |
|---|---|
| [SSL Starter v1](VCS_QMI_SSL_Agent_Starter_v1.zip) | 参考 J、tanh critic、cyclic negatives、SimCLR/VICReg；起初是可验证 reference，不是完整服务器结果 |
| [Estimator v1](VCS_QMI_Estimator_Research_and_Server_v1.zip) | 把“学 representation”和“准确估计 S”拆开 |
| [Estimator v2](VCS_QMI_Estimator_Research_and_Server_v2.zip) | 补优化、采样和传统 CS 对照，不以 SSL 胜出为前提 |
| [SSL Performance v3](VCS_SSL_Performance_Research_and_Server_v3.zip) | scorer 几何、多视图、负对设计；后来发展为 A-P3 |
| [Next Experiments v4](VCS_Next_Experiments_v4_Package.zip) | 学得更好、测得更准、测量有用三条证据线 |
| [Tasks/Theory v6](VCS_Server_Tasks_and_Theory_v6_20261003.zip) | 后验解释、曲率与几何、CIFAR-100 控制；不是 manuscript V6 |
| [SSL Plan v7](VCS_SSL_Server_Plan_v7_20261005.zip) | projector、PAIR/STRESS、ImageNet 探索 |
| [V6 r1](VCS_CVPR_V6_Server_Handoff_20261006_r1.zip) | 稿件交接：估计、SSL、条件审计、真实压缩任务 |
| [V6 r2](VCS_CVPR_V6_Server_Handoff_20261006_r2.zip) | 强化同 encoder 比估计器、同测量器比 encoder 的分工 |
| [V6 r3](VCS_CVPR_V6_Theory_Aligned_Server_20261007_r3.zip) | 回到误差、稳定性、分辨率；复用已有证据，P144 压缩未被自动解决 |
| [VL v1](VCS_VisionLanguage_Server_v1_20261008.zip) | region–phrase，RefCOCO 系列，显式 pair law |
| [V6 r4](VCS_CVPR_V6_Server_Revision_r4_20261010.zip) | RPC 对应、U-statistic 统计修复、proposal-aware VL |
| [V6 r5](VCS_CVPR_V6_Server_r5_VL_Alignment_20261011.zip) | 参考 ν、positive distribution vs set event、query normalization 与融合 |
| [V6 r6](VCS_CVPR_V6_Server_r6_New_Applications_20261011.zip) | 执行 REG 与 GAIN；其它方向仅研究卡片 |
| [V6 r7 atlas](VCS_CVPR_V6_Server_r7_VL_Research_Atlas_20261011.zip) | 六条 VL 路线，D1 token evidence 成为实际路线 |
| [V6 r8 review](VCS_CVPR_V6_Server_Review_r8_20261013.zip) | D1 train/eval/dropout、soft/hard 混杂、ties Spearman、干预修订 |
| [T2I V7 v1](VCS_CVPR_V7_T2I_Alignment_Server_20261013.zip) | EvalMuse 元素监督、BLIP2 Q-Former、best-of-N |
| [V7 v2](VCS_CVPR_V7_Server_v2_External_Benchmarks_20261014.zip) | 外部强基线与传统 CS；BCE 归因不是唯一成功门槛 |
| [V7 v3 closeout](VCS_V7_Supplement_v3_Closeout_20261014.zip) | 封账、归因和 element→utility 的边界 |
| [V8](VCS_V8_Utility_Aligned_CS_Server_v1_20261014.zip) | 1–5 人类评分的累积分布/期望效用，比较 BCE/RPS/CE/MSE/L1/ranking |
| [V8.1](VCS_V8_1_Server_Diagnostics_v1_20261015.zip) | epoch 语义、排名/尺度、输入/梯度检查、条件式 C1 救援 |
| [V9](VCS_V9_Reliable_Correspondence_Server_v1_20261016.zip) | N1 比例与可靠性，N2 加权检索，G1/G2 少样本与污染配准 |
| [V8.1/V9 Update](VCS_V81_V9_Update_20261016.zip) | 对审查意见作更正，ties Spearman、typed host config 等 |
| [V10 Design v0.1](VCS_V10_Moment_Factorized_Design_v0_1_20261016.zip) | simplex bilinear critic、精确矩、可证书的头部子问题；最初是设计/小型验证 |
| [V10 Server v1](VCS_V10_Server_Package_v1_20261016.zip) | 生产 backend、host contract、synthetic generator、T00–T05；包本身没有自然图像结果 |

包内 Markdown 已留在 `handoff/evidence/archive_texts/`，可以跨版本检索；未把包中的脚本当作命令自动执行。根目录 v1/v2/v5/v7 计划和 `RESULTS_20260928.md` 是对应时期快照，不能作为当前任务队列。

## 5. 各主线的最终或最新证据

### 5.1 SSL：从较大差距到同配方下接近

早期结果从约 74% 的 pilot，发展到 v2 四视图 800 epoch 的约 87% dev；后续 A-P3 采用固定 `tanh(2s−1)`、完整梯度及完整 view-pair 配方。不能只把提升归因于散度名称。

[P155 官方测试报告](ssl_pilot/reports/P155_AP3_OFFICIAL_TEST.md) 中：A-P3 五 seed linear **88.83±0.10%**，SimCLR 五 seed **88.20±0.13%**，matched JS-AP3 三 seed **88.92±0.29%**；kNN 分别 **87.75±0.23%、88.07±0.40%、88.04±0.06%**。A-P3 dev 89.02% 不是官方 test。官方 CIFAR 测试在更早阶段也用过，因此不能称整个项目第一次独立确认。报告标题残留 P68，定位应以 P155 文件和协议为准。

CIFAR-100、solo-learn/ImageNet 等没有形成普遍领先的证据。学习率、critic 几何、增广与特征源都需要同配方归因；不要继续以“必然击败 JS”为目标搜索。

### 5.2 估计、污染与嵌套读数

P85/P86、R4-E1 等支持“目标参数化与优化路径很重要”；表达力较强时 quadratic 没有稳定胜过 JS 的普遍结果。P108 已区分经验 J 与 plug-in 估计，R4-T2 已修复把 pair 当独立观测的推断问题。P134 的污染结果应按干净基线和下降量重读，不能直接称鲁棒性定理。

较新而容易漏掉的三份记录：

- [P157 SEP-2](ssl_pilot/reports/P157_SEP2_REPORT_20261015.md)：有真值的参数化 critic 污染实验，是后续 SEP-3 的背景。
- [P158 nested results](ssl_pilot/reports/P158/p158_results.md)：嵌套设计恢复部分结构恒等式，但冻结规则的最终读法仍为 **no clear reduction**；zero 分支是 shrinkage。不能把结构修复等同于准确性已确认改善。
- [P159 SEP-3](ssl_pilot/reports/P159_SEP3_NATURAL_REPORT_20261015.md)：320 单元、8 seeds，冻结结论 **FIT-ONLY**。污染下 clean-set J 的相对差 Q1=+0.0562，干净拟合 Q2=+0.0020；因此保留“特定参数化类的拟合性质”，不升级为 CS 特有鲁棒性、SSL 或 VL 准确率结论。clean 优势主要来自 CIFAR；quad-3 有 lr 上边界限制。

### 5.3 R4/R5：任务参考与候选竞争

R4/R5 的理论问题是：定位时 Q 的参考、query 内候选耦合、多个可接受 positive 的分布目标，是否与实际决策相容。实现入口为 [pairlaw.py](ssl_pilot/src/vcs_vl/pairlaw.py)。query normalization 是密度比归一化约束，不能包装为新的散度。

若 Q 定义为 P 的边缘乘积，它本来就可能给 positive-compatible pair 正质量；不能把这些项删掉再声称目标不变。多个 valid boxes 的分布目标与“集合中任一个命中”的事件目标也不同，这是 R4/R5 必须保留的对照。

[R5 QN DEV](ssl_pilot/reports/R5/R5_QNORM_REPORT_20261012.md) 中 VQ 相对旧 VCS 改善约 1.13/2.22 点，matched JS 有相近改善，softmax 仍很强。[正式测试](ssl_pilot/reports/R5/R5_FINAL_EVAL_REPORT_20261012.md) 的五 split 主检验中，RefCOCO testA/testB 的 +5.57/+8.73 点通过冻结双区间与 Holm 规则；另三 split 未在 family-wise 层面检测到差异。不能把五个正点估计写成五个显著确认，也不能把 VQ≈JQ 写成等价证明。

### 5.4 R6 和 D1：应用的成功与归因分开

[R6 synthesis](ssl_pilot/reports/R6/R6_SYNTHESIS_20261012.md) 将 REG 与 GAIN 分开。REG 的学习表示有用，VCS、logistic、InfoNCE 都可获益；V9/V10 延续的是这个配准 host。GAIN 的 CREMA-D 有有限用途，AVMNIST 的音频复用/重复问题不能当独立多模态证据。

[D1 最终报告](ssl_pilot/reports/R7/R7_D1_CLOSEOUT_REPORT_20261017.md) 已修订早期结论：S1 在匹配 latency 的 LLaVA claim cells 全部落后 SparseVLM，Qwen 对应格点也落后；不存在所有条件下固定的 S1/S2 排序；REV10 冻结读法为 **MIXED**。结论是这一设计与预算下没有形成主张所需的外部优势，不是“posterior 从原理上不可能用于 token 选择”。

### 5.5 V7、V8、V8.1

[V7 conclusion ledger](ssl_pilot_v7/reports/V7_CLOSEOUT/V7C_07_CONCLUSION_LEDGER.md)：与 FGA 相近，部分 SDXL 比较优于 TIFA/DSG，但落后 VQAScore；VCS 对 matched BCE 没有检测到差异。common-feature 下优于 KDE 不等于优于所有传统估计器，RuLSIF/Nyström 必须单列。

[V8 报告](ssl_pilot_v8/reports/V8_UTILITY/V8_04_EVAL_REPORT_20261015.md) 与 [V8.1 C1](ssl_pilot_v81/reports/V8_1_DIAGNOSTICS/V81_C1_REPORT_20261016.md)：从元素对应转为人类效用后，CS 特有改善有限，U-RANK 未获成功，C1 在冻结救援规则下改善 0/3，V8 关闭。`epoch0` checkpoint 是完成第一轮 2,379 steps 之后，不能作为未训练基线。部分 Holm 展示曾更正，不应复制旧 p 值。GenAI confirmation 与 TIFA secondary 的样本、标签语义不同。

### 5.6 V9

2026-10-18 15:13 CEST最新调度：用户明确改为“V9尾部释放的GPU立即接V10”。控制作业1046164已清除V10 gate1045975对整轮V9收尾的依赖；gate现为PENDING/QOSMaxGRESPerUser，可在L40S释放后竞争资源。四族数组1045976保留afterok:1045975，并发上限2→4；这是对原协议execution.gpu_concurrency的明确调度覆盖，原协议/源码、每族8 L40S GPUh预算及30次训练矩阵不改。旧CPU收尾1045974取消，新独立CPU审计1046165继续V9结果核验与分析，不阻止V10。V9 feeder1045056及所有GPU任务照常，查询时19/33完成、8运行、1排队。新的权威记录：[交叠激活](handoff/v9_overlap_v10_20261018/ACTIVATION.json)、[调度修订](handoff/v9_overlap_v10_20261018/SCHEDULING_AMENDMENT.json)。此前“整轮V9收尾后V10才执行”的状态已被本次指令覆盖；TEST134仍HOLD，旧占位watch不重启。

2026-10-18 11:40–11:41 CEST 接续调度更新：用户要求不只排4个任务，允许V9结束后接V10。已预提交CPU收尾作业1045974、V10 GPU gate1045975和四族数组1045976_0–3（并发上限2）。依赖为1045974成功→1045975成功→1045976；V9仍8 GPU运行、4组排队，另5个V10 GPU任务等待依赖，共9个GPU任务排队，normal总26。V9最新LAUNCH中的续跑新ID由CPU接续器动态跟踪；它在18组/33次全部完成、feeder退出、无V9活跃任务、产物哈希及完整结果分析通过后才退出0。原科学冻结与原feeder未改，旧watch1044425不重启，旧V10 ID仍取消。提交前49个V9代码文件、58个V10源文件及冻结输入/72个激活缓存检查通过，7项控制测试通过；这不代表未来科学结果或最终收尾已PASS。详见 [接续说明](handoff/v9_to_v10_20261018/README_CN.md)、[激活与依赖](handoff/v9_to_v10_20261018/ACTIVATION.json)、[队列核验](handoff/v9_to_v10_20261018/ACTIVATION_VERIFIED.json)。

2026-10-18 11:26–11:28 CEST 更新：N2F完整15/33次，seed901为11/11、seed902为4/11、seed903为0/11且已启动。18组全部提交，6组完成、8运行、4排队、0未提交；feeder1045056正常，无失败组，15个完成单位均未严重崩塌。seed901 SUM主尺度：VCS468.32、BCE464.20、RankMass466.25、RuLSIF467.00、不加权445.90，OracleMask478.54、Drop477.04；通用加权收益一致，VCS相对最强RuLSIF仅+1.32，仍待三seed结论。seed902 SUM VCS467.36、RankMass460.28，二者两seed均值差+4.58只是描述性线索。完整尺度配对已出现，SUM收益因方法而异；不再沿用凌晨“尚无配对”的状态。45个日志/manifest/identity哈希与6组DONE核验通过，未重算模型预测或读TEST。训练预计还需22–28小时，node52慢任务可能触发原配置checkpoint续跑，最终核验另计。详见 [阶段报告](V9_STAGE_REVIEW_CN.md) 与 [证据](handoff/evidence/v9_stage_20261018_1126/STAGE_SNAPSHOT.json)；V9完成分析后再启动V10。

2026-10-18 00:23–00:25 CEST 新阶段读数：N2F 完整5/33次，均为 seed901。最佳 DEV rSum 为 VCS MEAN467.53、RankMass MEAN457.46、BCE SUM464.20、不加权 SUM445.90、OracleMask SUM478.54；5次均未达到预定义严重崩塌条件。同尺度描述性差值为 BCE−不加权 +18.30、VCS−RankMass +10.08；方法间 GPU 型号不同且仅一个 seed，尚无完整尺度配对或 SUM 主尺度三 seed 结论。OracleMask 相对 BCE SUM 高14.34，但使用额外真值。保留通用加权的正向线索，不认定 VCS 独有优势，不据部分训练读数排名。完成现有33次并分析后转 V10，不扩 V9 网格。详见 [最新阶段分析](V9_STAGE_REVIEW_CN.md) 和 [证据快照](handoff/evidence/v9_stage_20261018_0023/STAGE_SNAPSHOT.json)。

[N1](ssl_pilot_v9/reports/V9/V9_N1_REPORT_20261015.md)：比例公式 `(a−c)/(a−b)` 对固定 score 的非退化仿射变换不变；不是对任意非线性 score 不变。VCS/BCE/BBSE-soft 的对应关系不构成新的比例估计器。EM、BBSE-hard 等很强，H1 未获支持。已知 shift 设计里未计算置信集，不等于系统已能检测未知真实 shift。

[G1 TEST](ssl_pilot_v9/reports/V9/V9_G1_TEST_REPORT_20261016.md)：134 cores、每 core 10 次扰动、54 个结果；full FIT clean VCS 的 <42 native px 成功率约 98.3%，BCE/InfoNCE 约 98.4%，MI 约 78.4%。相对 MI 的成功不等于 VCS 特有优势。TEST 首次读取记录为 **2026-10-15T23:36:09Z**，直接约束 V10 的独立确认说法。

[G2](ssl_pilot_v9/reports/V9/V9_G2_REPORT_20261017.md)：EPFL 477 windows 中保留 420（330 FIT/18 CAL/72 DEV）；52 个结果。VCS、BCE、InfoNCE clean 成功率都接近 99%，污染主比较相对 MI/CoMIR 有收益；仍不能认定 CS 特有鲁棒性。严格几何误差受残余对齐影响。

N2 是完整加权 retrieval 训练，不能用 N1 的比例估计结果代替。2026-10-17 用户明确要求分析阶段结果、设计并提交下一阶段，随后要求以全量训练作为依据、合理使用 8 GPU。已保存 [DEV 轨迹快照](handoff/evidence/v9_stage_20261017/CENSUS.json)：35个 tuning +5个 formal 标记单位已到40 epoch，另6个目录未完成。seed917 noisy 调参最优 rSum 为 RuLSIF469.84、VCS468.56、RankMass467.36、BCE465.05、Unweighted439.90；不能当三seed正式比较。

原 wave1 暂停文件保留，原GPU作业继续。本轮另冻结 [N2F 全量协议](handoff/v9_full_20261017/PROTOCOL_CN.md)：24次四方法×三seed×两尺度的同GPU配对，3次 RuLSIF SUM 对照，6次不可部署的 oracle 全量参照，共33次。全部40 epoch、完整Flickr train/DEV、TEST不读。原调度器1043940已于19:57 UTC由1045056接手，按8个pending目标自动补位；[实时机器状态](handoff/v9_full_20261017/FEED_STATUS.json)、[提交账本](handoff/v9_full_20261017/LAUNCH.json)、[读数](handoff/v9_full_20261017/READOUT.json)。本轮没有提交以短程 probe 为主的科学实验。

当前资源：normal QoS 8GPU 已满，runfill 为其它项目单独配额，不把 CPU 的 gres/gpu:0 当GPU，也不把全部QoS任务数当normal的提交余额。19:57 UTC按用户要求改为新feeder1045056：目标8个本轮pending、最多16个本轮运行/排队组；normal数组逐项展开后最多29个作业，每30秒检查。只补当前冻结矩阵，剩余任务不足时据实收尾。科学冻结保持原哈希，交接记录见 [调度更新](handoff/v9_dispatch_pending8_20261017/CHANGE_CN.md)。V10后续实验需共享这8张normal卡，不能另算8张。路线判断：通用加权值得本轮有退出条件的全量复现；VCS独特优势尚无证据，不扩COCO。详见 [阶段分析与停止条件](V9_STAGE_REVIEW_CN.md)。

2026-10-17 后续研究设计见 [V9_NEXT_RESEARCH_DESIGN_CN.md](V9_NEXT_RESEARCH_DESIGN_CN.md)：优先用全量 Oracle 对照区分 critic 信息不足与批次/负样本机制，再选择一个完整实验分支；稳定收益后研究固定训练噪声 manifest 下的 2%/5%/10%可信标注效率。补充了面向损失的加权 J 工作推导及其条件，明确不认领新的总体最优 posterior。该文是条件性设计，未提交新实验，不改变现有33次冻结矩阵。12:35 UTC 快照为8 GPU运行、N2F已提交7组（5运行、2等待）、新增完整40 epoch单位0个。

随后用户明确要求本轮结束后专心 V10：上段候选设计保留为研究记录，不继续执行。V9 收尾需核验完整三 seed 结果、尺度差异、Oracle 对照、失败/崩塌和实际成本，再记录结论；即使达到原继续投入门槛，也不自动扩展 V9。该优先级决定不取消当前33次矩阵、不改变冻结科学协议、不解除 V10 TEST134 HOLD。

17:07 UTC 曾提交V10 gate/正式数组，并启动watch1044425优先占位；该调度安排已被用户随后“还是要让V9的作业跑完”的明确要求覆盖。17:14 UTC恢复作业1044429停止旧watch、解除V9 1044282的hold、暂缓V10 gate1044420和数组1044424。8张normal GPU仍全部运行V9，原feeder1043940继续当前轮补位。证据见 [恢复记录](handoff/v10_bf1_20261017/RESTORE_V9_PRIORITY.json)。V10代码与冻结配置保留，V9收尾后再接续，不重启旧占位策略。

## 6. V10 的设计和实现

### 6.1 真正改变了什么

令 a(x)、b(y) 为非负且和为 1 的 K 维特征，W 的每个元素在 [-1,1]：

\[
T(x,y)=a(x)^\top Wb(y).
\]

T 是 W 元素的凸组合，天然有界，不再依靠输出 tanh。对同一 scene 的特征矩阵 A、B，设 u=Σai、v=Σbi、GA=AᵀA、GB=BᵀB，则 V 负对两项为

\[
Q_1=u^\top Wv/n^2,\quad Q_2=\langle W,G_AWG_B\rangle_F/n^2.
\]

U 版本从分子减去配对对角项，再除 n(n−1)。正对仍按配对样本直接计算。所有 scene 先各自归一化，再按显式权重汇总。

固定 K 下可避开 n² 个负对枚举；典型 head 矩计算为 O(nK²+K³)，矩阵存储 O(nK+K²)。这不是整个图像训练/配准 pipeline 的复杂度保证：encoder、空间采样、优化搜索和数据 IO 都必须计费。离散有限支持还允许对任意 critic 用计数聚合，不能以 T03 的表格加速独占为 MF 的优点。

### 6.2 三种 MF 方法

| 方法 | 训练负对 | W 更新 | 用途 |
|---|---|---|---|
| MF_VCS_perm1 | 一次采样排列（自然 V / T03 U 各按协议） | 与 encoder/feature map 一起 AdamW，之后 clamp | 同结构随机负对参照 |
| MF_VCS_M0 | 精确 moments | 同上 | 单独检验消除负对采样噪声 |
| MF_VCS_M1 | 精确 moments | 固定当前 feature 求 box 二次子问题，W detach 后 outer step | 单独检验头部求解策略 |

固定 features 的负 J 是 box 上凸二次函数，使用 HVP，避免显式构造 K²×K² Hessian。令梯度 g=∇L(W)，FW gap 为 `〈g,W〉+||g||₁`，可以上界该**经验、固定特征子问题**的最优性差。不能把它当总体 S 误差、encoder 收敛或泛化证书。M1 未精确求解时，detach 的 outer 梯度也不能声称是精确 profile objective 的梯度。

最初绝对容差 1e-4 在小初始 gap 下可立即停止。10/17 已改成 **relative_zero_head**：每批以零头 gap_ref=||Δ||₁ 为参考，阈值 0.01·gap_ref，cap=10/30/100；不是相对 warm-start 当前 gap。第一步 W=0，以后 warm start。Δ 精确为零单独处理；低 gap 的 fp64 诊断不能擅自 clamp/floor。容许 cap 耗尽，但必须如实记录未收敛。

保存的 encoder/W 是最后一次 inner 求解与随后 outer 更新形成的 checkpoint pair；没有对整个 FIT 再优化 W。评估不能偷偷 refit 头部。

### 6.3 代码入口

| 文件 | 责任 |
|---|---|
| [mf_vcs.py](ssl_pilot_v10/src/v10/mf_vcs.py) | simplex critic、P/Q moments、dense 对照、HVP、box solver；与服务器包原件字节相同 |
| [adapter.py](ssl_pilot_v10/src/v10/adapter.py) | 数据角色 guard、V9 host 桥接、共同采样、M1 新规则、Trainer、registration energy、checkpoint pair |
| [t00_build_manifest.py](ssl_pilot_v10/scripts/v10/t00_build_manifest.py) | data manifest 与 exposure |
| [t01_validate.py](ssl_pilot_v10/scripts/v10/t01_validate.py) | CPU/CUDA 数值、梯度、geometry 与 host 验证 |
| [t02_timing.py](ssl_pilot_v10/scripts/v10/t02_timing.py) | 实际 host 的训练/配准计时 |
| [T03 runner](ssl_pilot_v10_t03/scripts/v10_t03/t03_run.py) | synthetic 各单元、依赖、truth barrier、CAL 解析 |
| [T03 objective](ssl_pilot_v10_t03/scripts/v10_t03/t03_objectives.py) | 均匀 derangement、离散计数、U/V、饱和 FIT 表 |
| [T03 evaluator](ssl_pilot_v10_t03/scripts/v10_t03/t03_evaluate.py) | 真值、全支持 posterior 误差、EVAL、head gap 与诊断 |
| [T03 aggregate](ssl_pilot_v10_t03/v10/tools/t03_aggregate.py) | data replicate、algo variation、capacity 和配对差异分开汇总 |

精度策略为共同且完整记录的环境，不是“所有 TF32 均关闭”。当前记录包括 Torch 2.9.0+cu128、matmul highest、cuDNN convolution TF32 和相关 override 环境变量。自然训练主配置为 K16、temperature 1、ridge 0、W decay 0、其它 decay 1e-4、16 scenes×128 positions、crop160、AdamW、clip1；encoder/head 学习率绑定。

## 7. 工作树、存储和运行约束

| 目录 | 分支 | 本次快照 HEAD | 角色 |
|---|---|---|---|
| `ssl_pilot` | main | 450f91f | 实际主 Git 仓库；历史 SSL/estimation/R4–R7 |
| `ssl_pilot_v7` | v7/t2i-alignment | 2dc39b5 | T2I V7 |
| `ssl_pilot_v8` | v8/utility-aligned-cs | cacb906 | V8 |
| `ssl_pilot_v81` | v8.1/utility-diagnostics | 260752c | V8.1 |
| `ssl_pilot_v9` | v9/reliable-correspondence | abd281b | V9 |
| `ssl_pilot_v10` | v10/mf-vcs | 4336bb6 | 最新 accepted T01/T02 与预算 |
| `ssl_pilot_v10_bf1` | v10/bf1 | 4336bb6 | 本次由 d57341e 快进，同步五个 accepted 验证/计时/预算文件，core 未变 |
| `ssl_pilot_v10_t03` | v10/t03 | 0518984 | 结果在1c13749，工作期间外部新增中文报告及生成器，科学代码/结果未变 |

根目录本身不是 Git 仓库。其余 `ssl_pilot_*` 是 linked worktrees；不要认为在 main 改了代码就同步到了它们。上游 refs 未在本次 fetch，不能据此宣称远端当前状态。根目录新接手文档目前是本地交付，未自动 push。

`worktree_inventory.json` 保留开始盘点时的 BF1=d57341e；上表及 `V10_TAKEOVER_STATE.json` 记录接手同步之后的状态，迁移见 [BF1_SYNC.json](handoff/evidence/BF1_SYNC.json)。

最终核对时 T03 新增 [原项目中文报告](ssl_pilot_v10_t03/v10/T03/V10_T03_REPORT_20261017.md)（0518984，11:10:48 Europe/Paris）。它的生成器已加入 manifest、evaluator digest/status/ID 与汇总复算检查。本次独立读数保留作为交叉核验，不能把该新增提交说成本次 Codex 写入。

数据、模型和较新结果大量位于 `/projects/EEG-foundation-model/yinghao/{data,models,outputs_v9,outputs_v10}`；早期实验还有根目录 `outputs/`。源码、轻量结果和协议在仓库，checkpoint/逐步日志通常留在服务器。`slurm_logs/` 与 manifest 中 job ID 对应；日志存在不单独证明成功，需结合 result/status/hash。

延续现有执行约束：科学训练/数值门禁通过 Slurm；使用完整已定数据集，subset 只用于检查；先冻结、smoke，再正式运行。既有资源记录为 normal QOS、用户总作业上限 30、通常留余量到 27、本人 GPU≤8，V10 首轮≤2；提交前必须查实际资源。排除 node51/node60，node52 后来已恢复允许。自然 BF 当前预算以 L40S 计，不能未经协议更新换卡后把耗时直接混用。历史包里的“开并行代理”不是本次新授权。

2026-10-17 09:15 UTC 更正：**直接调用 `squeue` 连续成功**；同一时段 `timeout 5s squeue ...` 与 Python subprocess 包装均超时。结合当前已批准命令前缀，判断与工具的命令授权/沙箱执行路径有关，不能据此称 Slurm controller 不可达。后续直接调用 `squeue`；[最新查询与对照](handoff/evidence/SLURM_ACCESS.json) 保留完整证据，[初次失败记录](handoff/evidence/SLURM_ACCESS_INITIAL_FAILURE.json) 留档。当次28个作业中22 RUNNING、6 PENDING，N2仍有运行中作业；队列数字仅是该时点快照。没有在登录节点启动科学训练，也没有发送消息、取消其它任务、下载新数据或重读自然 TEST 图像。

## 8. 未来接手的固定规则

1. 首先读 [V10 工作单](V10_TAKEOVER_CN.md) 和 [代码审查](CODE_REVIEW_CN.md)，然后核对当前 git HEAD 与结果身份，不从旧 README 推导当前状态。
2. 本轮已核验的 BF code hash 是 `e9b4239450c4e2eb`；T03 code `54606921cdd2bf15`，formal job pin `64478d332c0a115a`。这些是路径集合的内容身份，与结果 commit hash 不是同一个概念。
3. 自然角色：FIT32/CAL8/DEV32、CONFIRM0、TEST134 HOLD；TEST 将来只能在完整方法与选择冻结后以 TEST_REPEAT 单列。DEV 和 TEST_REPEAT 不合成“166 个独立确认样本”。
4. 独立统计单位是 core，10 个扰动不是 10 个独立组织。另报 TMA position 的保守分组与 DEV24 敏感性；相同 position 不自动证明相同 specimen。
5. EPFL 需单独激活和图像级 QC，V9 已使用它不等于 V10 已完成激活。
6. 不用 T03 evaluator 真值或已读 DEV 的结果回头改变冻结 BF 网格；物质性 bug 需新版本并对受影响方法匹配重跑。
7. 保留负结果和旧错误记录，补更正与可追溯来源；从 JSON 生成数值，不能靠记忆抄数。
8. 新任务不要求重新询问已有授权；但必须完成 driver、冻结、门禁和实际成本记账等技术前置，不能把“用户允许继续”写成“实验已经 ready”。
