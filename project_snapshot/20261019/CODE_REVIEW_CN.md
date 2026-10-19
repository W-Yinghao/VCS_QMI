# CS_QMI 代码审查与接续风险

最新审查CR-15/16（2026-10-19）：确认并修复新增checkpoint恢复的设备语义错误：map_location=cuda会错误搬移非capturable AdamW的CPU step计数器；CPU加载后使用优化器自身load_state_dict设备规则，严格设备比较保留，新增AdamW/SGD下一步与RNG回放测试。ecd8a95下CPU94/调度17通过。原全局gate失败让全部后续任务停发且下游空等；现按方法gate→full→CAL→final隔离失败，明确FAILED及blocked后代，其它方法继续，旧1047235自然退出。另MI/CoMIR原h=.01跨越双线性插值边界，fp32更小h又出现差分消减；诊断1047351显示fp32-vs-fp64梯度最大差约2.01e-5/1.81e-6。ddd3bcb只修验证：fp64局部h与h/2两套差分、atol1e-6/rtol1e-3参考检查；实际fp32梯度仍用原atol5e-4/rtol.05，训练/配准精度、网格、候选均不改。CPU97/控制18及两方法GPU gate真实通过；七方法合计GPU gate7/7。控制器1047321负责五方法，1047364只恢复MI/CoMIR，不重复通过方法；各自源身份和实际exit0、完整产物校验仍为正式放行条件。原V9 49文件和BF1 58文件哈希不变。详见 `handoff/v10_method_pipeline_20261019/RECOVERY_VERIFIED.json`、`handoff/v10_native_geometry_recovery_20261019/EXECUTION_UNCHANGED.json`。全GPU门禁通过不等于全量实验完成或科学优势成立。

最新审查（2026-10-19）：新增正式接续已启动1047235，Slurm CPU92项回归与15项元数据控制检查PASS；GPU/full仍待真实结果。仅正式源contract.require_release增加显式、固定哈希且当前生效的调度授权，可替代V9收尾依赖；无授权的旧release仍要求33次V9收尾。现有GPU验证源5c04719保留；正式源3d3ffd4的其余36文件字节相同、contract其它AST相同，逐方法验收采用明确的证据迁移记录，不伪称在新commit重复运行CUDA。新控制器核对真实exit0、同代码/manifest/方法的T01/T02/RUNNER、完整训练/CAL及产物哈希后才创建release；正式CAL成功核验后才发最终seed。共享normal8卡，未取消运行任务；15项控制检查覆盖策略失效、未通过验收、旧V9路径、配额、失败退出、来源变动和重复提交。详见 `handoff/v10_ready_dispatch_20261019/`。CUDA/全量训练尚未完成，EPFL/TEST_REPEAT未激活；本更新覆盖下述旧测试延期和V9强制等待的执行状态。

最新实现安排（2026-10-18）：用户允许“V10的实现可以先开始，测试之后来”。V10剩余方法现在在独立工作树 `ssl_pilot_v10_remaining`、分支 `v10/remaining-implementation-20261018` 实现；数值/GPU测试和新实验仍等V9收尾。现有V9优先调度、在跑任务及V10暂停标记不变。新代码尚未测试，不计为已验证方法；授权记录见 [AUTHORIZATION.json](handoff/v10_remaining_implementation_20261018/AUTHORIZATION.json)。这仅覆盖此前“编码也等V9完成”的时间限制。

最新运行补充（2026-10-18）：BF1 GPU gate 已通过，首批全量作业正在执行；下文 10 月 17 日的“尚未运行”均保留为历史状态。用户最新决定要求保留全部在跑作业、V9 收尾后再继续 V10。新控制器 1047062 已激活，V9 当前 25/33 完成，V10 3 个在跑保留、1046917 已 hold、新提交暂停。调度核验与限制见 [说明](handoff/v9_first_20261018/README_CN.md)；这不表示 V9 已全部完成或 V10 缺失方法已实现。

审查日期：2026-10-17。起始基准：main `450f91f`、V10 core `4336bb6`、BF1 `d57341e`、T03 `1c13749`。BF1 本次已快进4336bb6；工作期间 T03 外部新增报告提交0518984，下面补充其影响。审查者：本次 Codex 接手。

**结论：当前归档的 T03 结果通过文件和记录一致性核验；已审查的 MF 矩公式与理论契约相符。需要修复的主要问题在结果复用和作业完成判定。BF1正式执行链与冻结现已补齐，GPU路径检查尚未运行；不应把 T01/T02 的通过写成 BF1 已验证或已完成。** 17:14 UTC按用户最新要求先完成V9，BF1 GPU作业暂缓；执行状态见 [V10_BF1_EXECUTION_CN.md](V10_BF1_EXECUTION_CN.md)。以下历史发现保留原身份与复现范围。

本次不修改已接受的训练代码、协议和历史结果，避免静默改变代码身份。新增了独立审计工具、可复现控制流案例和当前状态文档。下列 CR-01/02/03 是**已复现、尚未在原 runner 中修复**的问题；本次审计是对既有结果的补充防线。

## 1. 发现及优先级

### CR-12 · V10补充方法首版实现，尚未经过数值与执行验收

按用户“实现先开始、测试之后来”，独立分支`v10/remaining-implementation-20261018`提交`d4b234f`实现七种补充方法、统一CAL/DEV入口、数据角色限制、完整checkpoint状态和成本记录。见 [实现说明与待验证项](ssl_pilot_v10_remaining/v10/supplement/README_CN.md)。静态语法/格式检查通过；未执行Torch/CUDA、续跑或真实配准测试，也未生成正式release。旧V9的49个科学源文件和旧BF1的58个冻结源文件均核对未变。

已在代码审查中处理：InfoNCE loss不能冒充J；perm8须平均各负排列的一阶/二阶矩；RuLSIF按原文无约束ridge解保留负系数，不能静默截断；原生CoMIR与无训练方法须分别定义完整工作量；非有限配准能量须在宿主nan_to_num前报错。运行入口要求当前代码和方法的T01/T02/runner验收、版本冻结、V9收尾，并核对CAL退出及全套产物。

仍需实测：固定排列值/梯度、RuLSIF精确V/U与完整枚举一致性、MIND二维和旋转行为、CoMIR BatchNorm/dropout续跑、完整数据和几何路径、实际显存/成本。MIND、RuLSIF有限核中心及CoMIR末epoch CAL规则均为明确登记的实现草案；不能将首版代码、静态检查或19 CAL+15最终单元清单写成完成了科学比较。

### CR-13 · V9跨节点诊断ESS末位差异触发过强字节检查

1046652在resume前因manifest SHA变化退出；定位为`ess_U`诊断字段的1 ULP差异，非ESS字段完全一致，34轮last/best checkpoint哈希未变。旧训练源码把manifest在构造器中重写，split runner随后比较字节，因而把无训练意义的浮点诊断差异视作配方变化。

新 [恢复包](handoff/v9_ess_resume_20261018/README_CN.md) 只对该单元、已核实checkpoint允许ESS最多4 ULP的重算差异，所有其它字段必须完全相等，再保留原manifest字节并另存实际重算证据。训练公式和原源文件不改。6项纯元数据控制检查通过；GPU恢复1047108已提交，CPU1047107等待真实成功及原feeder自然退出后重建账本。旧失败保留；当前不是已恢复成功。V10数值测试继续延后。

### CR-11 · 阶段切换必须保留运行作业并验证 V9 收尾

新 [bridge.py](handoff/v9_first_20261018/bridge.py) 只对现有 V10 账本中的 PENDING 作业执行 hold，并在操作前再次查询；启动竞争中变成 RUNNING 的作业继续执行。代码不含取消路径。先用原 dispatcher 支持的 PAUSE_SUBMISSIONS 阻止新提交，再由独立 CPU 控制器管理本次 hold 的所有权，既有用户/管理员 hold 不自动解除。

自动恢复同时要求完整 33 次、无相关 V9 活跃作业、feeder 无失败、独立审计 PASS 且已退出，再比对精确单位矩阵、科学冻结、报告及结果哈希。任一失败均不自动恢复。ACTIVE_SCHEDULING_POLICY 防止后续用户修改顺序后旧控制器仍放行；控制器名称 stage_bridge 避免被 V9 审计自身的名称检测纳入等待集合。6 项控制测试通过，实际队列确认全部 8 个在跑 GPU 作业保留、仅未运行的 1046917 被 hold，原科学源未改。

边界：这是调度与产物核验，不替代全量训练结果。控制器只自动恢复已有 BF1 冻结作业；[剩余工作单](handoff/v9_first_20261018/V10_REMAINING_WORK.json) 中的新方法仍待 V9 收尾后编码、验证、冻结及完整实验。控制器异常记录 ERROR.json，不能靠错误退出或缺少队列输出把 V9 标为完成。

### CR-10 · V9串行尾部拆分的重复写入与完成判定

2026-10-18，按用户“能拆开吗？”新增 [独立调度包](handoff/v9_unit_split_20261018/README_CN.md)。旧runner在进入group时已加载两条单位，没有动态跳过或stop-after-unit开关；直接并行启动第二条会使旧作业稍后重复进入同一产物目录。因此采用checkpoint边界迁移，原writer离开队列后才释放新作业；每单元再加非阻塞独占锁。已有epoch、优化器、RNG及best/last在CPU Slurm检查并归档，完整40轮科学训练调用与原版一致。

新完成条件包含真实exit0、无科学FAILED、精确单位身份、固定产物哈希及40轮/最佳DEV规则。一条完成不能写组DONE；两条均核验后才生成聚合完成记录，保留两个真实GPU退出及代表作业的实际退出时间。8项控制测试覆盖重复writer、live/非零退出误判、部分组完成、配额逐项计数及原独立审计等待新任务；全部通过。原科学源49文件未改，调度差异另行冻结。V9 1044282的137/143一次基础设施续跑由替换feeder保留。

限制：这两组已显式取消同一GPU顺序执行条件，可能引入硬件差异，不能作严格同卡尺度因果解释；恢复训练的原步数日志计数器实现未改。完整allocation成本仍不能仅由组代表作业推算，需查归档原退出及各unit退出，sacct缺口继续声明。实际切换/运行状态见该包ACTIVATION.json和UNIT_LEDGER.json，控制测试通过不等于训练已完成。

### CR-01 · P1 · `COMPLETE` 复用没有核对科学配置及产物身份

位置：[t03_run.py:279](ssl_pilot_v10_t03/scripts/v10_t03/t03_run.py#L279)，关键提前返回在 284–286。

`run_unit()` 只要看到已有 `result.json` 的 `status == COMPLETE` 就返回成功。该分支位于 suite/gates/hyperparameter resolution/UnitConfig 构造之前；当前代码 hash 与提交 pin 的入口检查，不能证明旧结果由当前身份产生。

**具体触发：**在同一 output root、同一 unit ID 下切换代码版本，或把 `--dtype float32` 改成 `float64`。旧结果仍被记作 `COMPLETE (kept)`，甚至不会先检查 evaluator/checkpoint 是否存在。`_dep_results` 的身份检查只覆盖依赖解析；这个分支绕过了它。

**后果：**重提交可把旧科学配置的结果当成新运行的完成状态；结合 CR-03 可让 CAL batch 的退出码呈现成功。当前不是“任意文件都能骗过门禁”的泛化安全问题，而是日常续跑会遇到的版本混淆。

**复现：**`reproduce_review_findings.py` AST 提取原函数，用无副作用 stub 放入 old-code/float32 的 COMPLETE 记录，传入 new-code/float64，实际返回 COMPLETE。没有启动训练或修改原结果。

**建议修复：**在复用前比较 unit row、resolved hyperparameters、code/protocol/data signature、dtype/device、seed、steps/checkpoints，以及 evaluator 与关键产物哈希。不同 signature 拒绝复用并说明新 output namespace；保留旧记录。Git commit 可因结果文档提交变化，不宜仅比较 commit；job pin 的允许迁移也必须显式定义。

**必要回归：**相同配置可复用；代码变更、dtype 变更、依赖变化、缺文件、hash 不符均不能返回 COMPLETE。现有 [runner 测试](ssl_pilot_v10_t03/tests/v10_t03/test_t03_runner.py#L27) 覆盖正常 COMPLETE kept，没有覆盖这些失配。

**本轮影响核验：**1,219 单元的已记录 code hash 一致；formal/CAL pin 一致、smoke 使用其已记录旧调度 pin；配置与冻结 ledger、CAL 依赖、evaluator/checkpoint/log 哈希均匹配。没有发现该漏洞污染本轮结果的证据。后续新代码不能据此免除修复。

### CR-02 · P1 · 汇总器将存在的 evaluator 文件直接视为可用

位置：[t03_aggregate.py:66](ssl_pilot_v10_t03/v10/tools/t03_aggregate.py#L66) 的 `load_all()`，以及 [112](ssl_pilot_v10_t03/v10/tools/t03_aggregate.py#L112) 附近的 `_cell()`。

加载时没有对照 `result.evaluation.sha256`，也不检查 evaluator 的 status、unit ID、method、replicate identity 与 result/ledger 的一致性；计入条件仅为 result COMPLETE 且 evaluation 非空。

**具体触发：**结果导出/复制中混入另一个 unit 的 evaluation，或 evaluation 被修改、状态 FAIL，但 result 仍 COMPLETE。表格仍可得到 `n_complete == expected` 和合法外观的数字。

**复现：**临时目录中放入 COMPLETE result、错误 digest、`unit_id=DIFFERENT_UNIT/status=FAIL` 的 evaluator。执行原 loader 和 cell 控制流后 `n_complete=1`。统计函数用 inert stub 替代，因此这个复现只证明错误接纳路径，不声称运行了整个科学汇总流程。

**建议修复：**在 analysis 入口强制执行结果验证；不一致时返回非零并记录具体 unit，不应把损坏记录安静排除后继续发表完整表。汇总输出同时保存已验证输入 manifest 和汇总器 SHA。汇总器位于训练 hash 之外本身是合理设计，不能因此省略独立身份。

**本次补充：**[audit_v10.py](handoff/tools/audit_v10.py) 对导出 manifest、result↔evaluation、配置、CAL 依赖、配对和汇总算术建立独立检查；失败不生成新科学表。本轮导出文件全部匹配，未发现数据混淆。它不自动改变原汇总命令的行为，未来直接运行旧 `t03_aggregate.py` 仍需先验证输入。

**新增报告入口的缓解：**最终核对时出现外部提交 `0518984`，其 [t03_report.py:310](ssl_pilot_v10_t03/v10/tools/t03_report.py#L310) 已核对 evaluator ID、PASS 和 SHA，并验证导出 manifest、重算表格。因此新增正式中文报告有这层防线；CR-02 仍适用于直接使用底层 aggregate 的路径，不应误读为新报告没有验证。

### CR-03 · P2 · CAL batch 只检查 selection 命令退出码，不检查全部组已选定

位置：[cmd_select](ssl_pilot_v10_t03/scripts/v10_t03/t03_run.py#L405)、[t03_grid.sbatch:52](ssl_pilot_v10_t03/slurm/v10_t03/t03_grid.sbatch#L52)。

`cmd_select()` 可以输出 BLOCKED 组，但末尾恒返回 0。这与一个“生成状态报告”的命令语义可以相容，现有 [test_select_command_reports_every_group](ssl_pilot_v10_t03/tests/v10_t03/test_t03_runner.py#L146) 还明确要求 0/60 和 1/60 时返回 0。问题在于 batch 将此退出码当作 selection 阶段完成，没有检查 JSON 的 `n_selected == n_groups == 60`。

**触发：**上游单元看似完成但被 CAL 身份检查拒绝，或报告型 selection 在依赖缺失时被当成执行 gate。CR-01 的 stale COMPLETE 是具体可达路径。并非每次缺失依赖都会误报成功：正常训练 shard 已返回 FAIL/BLOCKED 时，batch 的 `rc_r` 仍会失败。

**复现：**所有组依赖均 BLOCKED 的 stub 场景返回 exit 0、n_selected 0。当前正式 CAL 是 60/60，故此问题未影响本轮实际选择。

**建议修复：**保留报告命令兼容性，增加 `select --require-all`，或在 sbatch 中严格校验 selection JSON。新的 batch gate 要覆盖 0/60、59/60、60/60 及 stale identity。不要只改 return 值而忽略原有报告模式和测试契约。

### CR-04 · P2 · 小样本方差区间的前提未在读数说明中显式限定

位置：[t03_aggregate.py:79](ssl_pilot_v10_t03/v10/tools/t03_aggregate.py#L79) 的 `_stats()`。

chi-square 方差区间依赖正态独立重复；R=10 的数据重复和尤其 R=3 的算法 seed 并不能自动满足该前提。经验误差在 null 条件下可能偏斜。t 区间在这种小样本下同样应明确假设，percentile bootstrap 也不是自动有效的有限样本覆盖保证。

原始数值计算本身没有发现算术错误；问题是把它们笼统称为“finite-replicate uncertainty”会掩盖解释范围。48 个 paired comparisons 也没有据此形成已冻结的多重检验结论。

**处理：**本次读数标明区间为对应假设下的探索性不确定性，方向计数不是显著性检验。保留原区间和冻结结果，不根据看到的结果另选检验。未来新增确认任务应在结果读取前规定统计单位、区间方法和多重性。

### CR-05 · P2 · 最新状态散落，BF1 分支结果材料落后一轮

`ssl_pilot_v10` 的 `4336bb6` 含最新 relative-M1 CUDA 门禁、T02 与预算；`ssl_pilot_v10_bf1` 仍为 `d57341e`。根目录 phase1 decisions、旧 stage report 和 P2 的 NOT_RUN 字段记录的是不同时间点。

这些历史记录并不应被整体覆盖为“错误”。风险是执行者误用旧绝对 M1 规则或旧预算，或误把 `LANDED_NOT_FROZEN` 当作冻结完成。

**本次已处理：**新增 [PROJECT_MEMORY_CN.md](PROJECT_MEMORY_CN.md)、[V10_TAKEOVER_CN.md](V10_TAKEOVER_CN.md) 和机器状态；先将干净的 BF1 分支从 d57341e 快进到 4336bb6，仅同步五个 accepted 验证/计时/预算文件，core hash 未变。后续b8f59ea补入正式driver和独立冻结快照，5项控制测试与72个激活数据文件核验通过。**仍待执行：**GPU实际路径检查与全量训练，当前依用户要求等待V9完成；新driver已做身份/产物复用防护，但不能把这些本地检查当作GPU执行验证。

### CR-06 · P3 · 历史报告的标题和表格标识存在残留

[P155_AP3_OFFICIAL_TEST.md](ssl_pilot/reports/P155_AP3_OFFICIAL_TEST.md) 标题仍为 P68；cell 名含未转义 `|`，会破坏 Markdown 表格分列。旧 project map 某些段落还说没有新官方 test，后续报告已覆盖这个状态。

这些不改变原 JSON 数值，但会使读者误判阶段和数据曝光。后续维护应修 report generator 的标题参数与表格 escaping，并显式链接更正。本次项目记忆已采用 P155 的实际结果与协议，未覆盖历史文件。

## 2. 已核对为一致的关键实现

| 检查点 | 代码证据 | 结论及边界 |
|---|---|---|
| 混合风险归一化 | `vcs_estim/objectives.py`；`mf_vcs.py:_finish` | J 的两个分布分别平均，R=1−J；没有发现额外 1/2 的错位 |
| simplex 有界性 | `mf_vcs.py:SimplexBilinearCritic` | softmax features + box W，T 自然有界；不是 tanh 之后再宣称 exact moments |
| V/U 两项 | `mf_vcs.py:_product_moments`、T03 objectives | q1/q2 使用相同 law；U 的对角扣除与分母一致 |
| 多 scene 权重 | `SceneBatch`、`_inputs`、adapter scene assembly | scene 先归一化再按显式权重组合；未静默改为全局 product |
| M1 子问题 | `QuadraticHead`、`solve_box_head` | 固定 feature 的凸二次、HVP 与 box gap；ridge0 的自然主协议未被额外 W decay 改写 |
| 新停止规则 | `adapter.py:_m1_relative` | 当批零头参考、warm start、cap 耗尽与 fp64 诊断单列；未将 warm-start gap 作为参考 |
| host 采样复现 | `HostBatchSampler`、T01 host replay | 共享 crop、位置、普通 permutation 的 RNG 消耗；允许 fixed points 是自然 V 语义 |
| 配准 energy | `MFRigidEnergy`、`mf_energy_rows` | reference features 缓存，moving features 随变换更新；一次 registration 内排列固定 |
| checkpoint pair | `save_pair/load_pair`、Trainer | 保存实际 encoder/head；评估不 refit W；gap 仅对 inner 当时的 features 成立 |
| T03 derangement | `t03_objectives.py:sample_derangement` | 拒绝含 fixed point 的均匀排列，得到均匀 derangement；没有使用有偏的“修复置换” |
| T03 选择隔离 | `t03_select.py`、`truth_barrier` | 训练/CAL 只读样本与 CAL J；truth/EVAL 单列，拒绝记录会进入 FAIL/BLOCKED |
| T03 配对 | RNG streams、data hashes、init hashes | 同条件/replicate 数据一致；方法之间及 data replicates 的指定算法流一致；本次记录核验通过 |

truth barrier 是防意外访问的 Python 进程内 guard，代码也明确说明它不是操作系统隔离。已经覆盖 audit events、若干 Torch C-level 读路径，并用静态限制补充；本次未将其宣传成对任意扩展模块的安全边界。

## 3. 数值门禁与已知例外

最新 [TORCH_VALIDATION](ssl_pilot_v10/v10/T01/TORCH_VALIDATION.json) 有 1,978 条外层检查，其中必需项 1,929/1,929 PASS；[GEOMETRY_GRADIENT_CHECK](ssl_pilot_v10/v10/T01/GEOMETRY_GRADIENT_CHECK.json) 有 184 条，必需项 172/172 PASS。合计 **2,101 个 required 项通过**；内部 reference 子检查计数不要再次重复加到这个总数里。

Torch record 中还有 20 个 nonrequired FAIL。它们被记录为诊断，例如把带相对项的设计容差误读成只看 atol、以及已拆分的完整 MLP fp32 梯度诊断。不能称“所有检查零失败”，也不能把所有诊断 FAIL 都当新 blocking bug。是否豁免以已采纳决定及对应 required 标记为准，后续新增失败不得照搬豁免。

新 T02 relative-M1 短程测试中，cap10/30 全部触顶，cap100 的触顶比例约 94–100%。这是已经披露的近似求解，不是“1% gap 全部达到”。T03 M1 的训练阶段收敛比例不能替代自然图像的收敛证据。

本次只读取并校验上述记录，未在本会话重新执行 Torch/CUDA tests。初次包装查询超时后，已在09:15 UTC用直接 `squeue` 连续成功查询；原先“当前会话无法连接”的解释已更正为调用方式/执行路径问题。记录中的 CPU job 1043836、CUDA job 1043855 是先前完成的验证，不能署为本次新跑。

## 4. 独立核验交付及复现

从项目根目录执行以下轻量文件/记录检查；它们不导入训练模块、不启动 Torch、不读取自然 TEST 图像、不改已有实验结果：

```bash
python3 handoff/tools/audit_v10.py --verify-external-artifacts
python3 handoff/tools/reproduce_review_findings.py
python3 handoff/tools/prepare_bf1.py
```

首次盘点还可运行 `python3 handoff/tools/inventory.py`，其 PDF 提取依赖 pypdf；audit/reproduction/BF1 preparation 均用标准库。`prepare_bf1` 只生成待执行清单，不会提交作业或把协议状态改为 FROZEN。

本次 audit 检查包含：

- 三份结果 manifest 的全部导出文件哈希/大小；外部 checkpoint、逐步日志等条目哈希。
- frozen ledger 的覆盖、配置、dtype/device、code/pin、训练步数、checkpoint 网格。
- result↔evaluation 身份、digest、PASS，已记录的理论恒等式残差。
- 每个单元从 CAL checkpoint 重建 argmax；从全部候选重建 60 个 group 选择；formal 超参数与依赖 digest。
- 数据哈希、初始参数哈希、permutation stream、饱和 FIT 表的一致性。
- main/algo/capacity 的成员归属、均值/方差/标准差/RMSE，以及 paired 差异的描述性算术。

输出：[V10_AUDIT.json](handoff/evidence/V10_AUDIT.json)、[控制流复现记录](handoff/evidence/REVIEW_REPRODUCTIONS.json)。本次为 **71,024 个程序检查条件通过**；这不是 71,024 个独立科学检验，也不替代实验复跑。

## 5. 尚未覆盖和仍需关闭的事项

没有逐行审查所有历史分支、所有外部 baseline 源码、数据解码器和训练循环；没有独立从 checkpoint 重算所有模型输出。已直接查询实时队列，但尚在写入的 N2 结果没有被当作完整最终结果核验。文件哈希证明与 manifest 一致，不证明生成过程绝对正确或统计主张成立。

BF1 release 前必须关闭执行链前置：immutable driver 与配置身份、实际 family budget 累计、CAL 选择完成 gate、角色/geometry/precision 冻结、完整路径 smoke、失败保留与安全续跑。missing baselines 的 targeted T01/T02 和 method attachments 仍是全 V10 比较结论的前置。

优先顺序是：保留已核验 T03 → 将 CR-01/02/03 的经验用于新 runner 并按版本修复旧 runner → 落地 BF1 四族 → 补齐剩余基线 → 按完整冻结规则考虑 TEST_REPEAT。不要为了消除代码审查中的“待办”而改变已经看过结果的科学网格。

## 6. V9 全量执行接手补充（2026-10-17）

用户最新要求 V9 全量实验优先，阶段证据与新实验见 [V9_STAGE_REVIEW_CN.md](V9_STAGE_REVIEW_CN.md)。以下不改变既有冻结权重或原实验归属。

### V9-CR-01：旧 student 的 COMPLETE 复用缺少身份/产物校验（P1，新入口已防护）

[train.py](ssl_pilot_v9/src/v9/n2/train.py:84) 的 `fit()` 看到 done.json 就返回；构造器又先写 run_manifest。单靠目录名不能保证 epochs、loss scale、输入或配置没有变。新 [run_full.py](handoff/v9_full_20261017/run_full.py) 在进入旧 trainer 前检查身份；复用 full_result 要核对 best/last checkpoint、训练日志、run manifest、identity 的 SHA256；原始 done 还需匹配 manifest，并有完整产物。完整40 epoch才产生full_result和组DONE。新源代码已复制并冻结，排队期间不会跟随原工作树编辑。

### V9-CR-02：Q 的“图像 cluster”实际上使用 pair ID（P2，影响区间解释）

[pipeline.py:69](ssl_pilot_v9/src/v9/n2/pipeline.py:69) 把 `ref_Q_img` 设置为 `q_of`，而 q_of 是 pair_id。当前 s917 锁记录显示 P=435 clusters、Q=2175 clusters，后者其实按caption/pair计数。再加上 P/Q 复用参考图像、U 错配 donor 引入的依赖，不能将这些区间当作已经验证的独立图像簇覆盖保证。原代码已把区间叫 diagnostic；这里明确登记其原因。该问题不改变点估计所用三个经验均值，不能因此宣布权重训练结果全部无效。未来正式区间需要重审 original/donor 图像依赖，而不是仅替换一个字段名。

### V9-CR-03：Q sampler 未保证真正排除 same-image（P2，冻结方法差异需披露）

[weights.py:115](ssl_pilot_v9/src/v9/n2/weights.py:115) 首个 donor 与原图相同时只前进一步；同图有多个caption时，第二个 donor 仍可来自同一图。例如图像序列 `[a,a,a,b]`、identity排列、shift=1，第一项会连续遇到两个 a。代码没有循环排除或结束后的断言，因此“Q 必定无同图对”的契约并未由实现保证。这是代码反例，尚未统计当前各冻结cell的实际发生率；它不等于已测得大幅性能偏差。新全量实验复用原锁定权重，不悄悄更改 Q 后将两种方法混为一个版本。

### V9-CR-04：mean/sum 的差异会通过裁剪改变优化；短程重跑尚未识别原因

[train.py](ssl_pilot_v9/src/v9/n2/train.py:98) 的尺度分支保持 per-anchor loss 和 fixed-pool normalizer，SUM=当前batch大小×MEAN。梯度裁剪阈值固定2，所以不能仅凭 Adam 的尺度性质认为两条训练完全等价。现有 A/A 在300 step不复现原崩塌，短程 C1 SUM 正常；证据不足以确认因果。新同 GPU/seed、固定其它配置的完整40 epoch配对专门检验这个问题，并把新确定性设置与历史设置区分。

### V9-CR-05：调度统计与实际授权范围

首次新feeder把全部QoS混为一条队列，且字符串匹配把 gres/gpu:0 计为GPU。实际 squeue 显示 runfill 属其它项目、normal 的8张卡已满。科学作业提交前修复 [feed.py](handoff/v9_full_20261017/feed.py)，以 normal 队列<27控制提交，解析GPU数量；4个纯元数据案例验证gpu:0/gpu:1/typed gpu:2/N/A。旧CPU feeder1043935已停止，当前1043940；修订前没有新增科学训练。保留前后冻结哈希和 [DISPATCH_REVISION](handoff/v9_full_20261017/DISPATCH_REVISION.json)。

### 新入口的验证和边界

[VALIDATION.json](handoff/v9_full_20261017/VALIDATION.json)：Slurm CPU上核对40个冻结输入；数值核对可手算hinge及loss/gradient尺度比例、oracle标签不进入通用trainer行、严重崩塌汇总的正反例；验证18组/33次完整训练矩阵。脚本语法检查通过。首批1043941/1043942已成功进入Slurm队列，当前等待已占满的8 GPU；**尚未验证新GPU完整训练成功**，不能将执行门禁写成科学PASS。

新full runner只读train/DEV，所有新结果与原N2隔离；oracle单列、失败留存，不自动改变科学配方。特征文件沿用既有数据路径/size/mtime身份检查，token cache和权重完整哈希；本轮没有重新对整份大型特征文件做全量SHA256，也没有独立重算全部旧模型预测。现有源码中的“hard loss从step0必崩”旧注释与历史更正不一致，不作为本轮实验动机的已证事实。

### V9-CR-06：八卡运行时缺少排队后备（已按用户要求替换调度器）

旧feeder同时要求本轮运行+排队<8、pending<2，导致8个组运行时无法提交后备任务；原队列查询未展开数组，也可能低估normal提交总数。19:57 UTC以独立冻结的 [新调度器](handoff/v9_dispatch_pending8_20261017/feed_pending8.py) 替代旧CPU循环，目标8 pending、本轮最多16个活动组、normal数组逐项计数后最多29。沿用原锁和提交账本，交接先停旧循环再启动新循环；原科学冻结与GPU作业保持身份。容量、数组计数、QoS隔离和CPU计数检查通过，实际已观察到8运行+8排队；科学结果仍需独立核验。详见 [验证](handoff/v9_dispatch_pending8_20261017/VALIDATION.json) 与 [交接](handoff/v9_dispatch_pending8_20261017/TRANSITION.json)。

### V9/V10-CR-07：续跑换ID与自动接续的依赖边界（2026-10-18）

只给V10绑定一组旧V9作业ID，无法覆盖超时续跑；只看空队列也会把失败当完成。新 [CPU接续器](handoff/v9_to_v10_20261018/bridge.py) 动态读取原LAUNCH、18组DONE、FEED_FINISHED和实时队列；在33次全量单位及best/last等产物哈希、最佳DEV选择、旧作业退出证据和冻结身份通过后生成分析报告并退出0。V10 GPU gate依赖该CPU作业afterok，四族数组依赖新gate afterok；不调用V9的hold/cancel，不更改科学配方。7项元数据控制测试通过，涵盖缺项/失败不放行、续跑新ID、数组计数、新依赖、checkpoint损坏、最早并列选择和已到末epoch的续跑计时；这些不是科学实验结果。实际激活记录见 [ACTIVATION](handoff/v9_to_v10_20261018/ACTIVATION.json)。完整33次终态尚未发生，不能提前声称最终收尾PASS。会计服务缺口与硬件迁移作为分析限制保留。

### V9/V10-CR-08：按新授权解除整轮依赖，保留V10自身执行检查（2026-10-18）

用户明确允许V9尾部与V10交叠。控制作业1046164已清除gate1045975的V9收尾依赖；数组1045976保留afterok:1045975，最大并发2→4。原PROTOCOL.execution.gpu_concurrency=2未静默篡改，而由[单独调度修订](handoff/v9_overlap_v10_20261018/SCHEDULING_AMENDMENT.json)覆盖。V9独立审计1046165继承原核验逻辑并替代旧1045974；旧CPU仅在gate依赖清除后取消，所有V9 GPU任务照常。源码派生差异限于说明、完成状态和报告后续段落；语法和派生一致性检查通过，控制器已核验gate无依赖、数组%4且仍依赖自己的GPU gate。新的[实时核验](handoff/v9_overlap_v10_20261018/ACTIVATION_VERIFIED.json)是当前依据，CR-07中的CPU收尾作为V10前置条件属于历史。未重提交GPU任务，模型、候选顺序、训练数据与每族L40S预算不变。


### CR-09 · V10真实8路并发的执行拆分（2026-10-18）

旧数组只有4个task，单改%8不能产生8路。已用冻结runner原有`cal FAMILY`和`final FAMILY --seed`接口改为4个CAL家族作业、12个独立FINAL种子作业；原58个源码文件和2个冻结输入哈希通过，原协议字节保持不变，并发和作业分组由 [显式调度修订](handoff/v10_parallel8_20261018/SCHEDULING_AMENDMENT.json) 覆盖。CAL候选顺序、8小时家族预算、完整3000步/12次CAL读数、数据角色与选择规则均保持。`Trainer.__init__`重设torch seed，batch及MLP额外随机流按seed初始化；每个FINAL输出和RUN.lock独立，三个seed绑定同一CAL选择哈希。额外进程启动成本归入FINAL；不新增逐bit确定性承诺。

CPU调度器以独占锁、先持久化提交意图再记录ID防止重复；模糊提交不得自动重试。只在CAL离开队列、batch exit=0、成本账本END/COMPLETE及选择和产物哈希全通过后提交FINAL。队列消失或退出0都不能单独代表完成。成本账本读取使用共享文件锁，兼容并发append。失败家族停止后续提交，已运行独立seed保留各自结果且家族仍需审查；不静默扩预算或改参数。GPU总上限由当前normal共享8GPU配额约束，loop目标8pending、活跃最多16、normal逐项计数最多29。

[7项控制测试](handoff/v10_parallel8_20261018/VALIDATION.json) 通过：精确矩阵、同族CAL依赖、失败/重复防护、数组展开与共享提交上限、队列消失、零退出缺少COMPLETE、gate身份/退出校验。这些是调度检查，不是科学结论。旧1045976在RunTime=0时取消，新1046189运行、CAL1046190–1046193已提交，gate1045975保留；[实队列核验](handoff/v10_parallel8_20261018/ACTIVATION_VERIFIED.json) 通过。CR-08的%4安排为历史。


执行核验补充（2026-10-18 20:19 CEST）：新拆分调度实际完成2族CAL、自动提交6个FINAL seed，1运行+7排队；GPU gate已PASS，修正主LAUNCH和接手状态中滞留的“gate等待资源”字段，未改运行中dispatcher或科学冻结。23份V9和8份V10全量候选通过本次元数据/JSON日志检查；具体边界、哈希计数和1044282续跑风险见 [运行快照](handoff/evidence/runtime_20261018_1818/STAGE_SNAPSHOT.json)。


## CR14：V10验证已启动及GPU补位修正（2026-10-19）

“V9优先”此前被实现为全V9收尾前禁止新V10启动，导致7个V9仍运行时第8个normal GPU名额未补。已通过新ACTIVE策略使旧stage_bridge1047062自然退出，CPU1047184只释放原策略拥有的hold1046917并归档其pause；没有取消运行作业。原dispatcher1046189提交1047185–1047187，当前7 V9 + 1 V10占用8个GPU名额。

用户进一步授权现在测试。独立提交5c04719新增公式/梯度和状态恢复测试、真实GPU执行检查，以及七种方法首候选的完整训练与全CAL验证自动调度。CPU Slurm1047202：92 passed（26.84s）；GPU1047204尚待执行。数值小张量检查只核对公式，不构成科学性能证据；完整验证调用实际runner，保持3000步/736步及所有CAL checkpoint，不提前读DEV或held TEST。

验证执行快照与代码哈希绑定，实际非零退出、缺失产物或身份不匹配都拒绝PASS；full验证先验证真实GPU gate成功退出。最多8项目GPU pending、normal29提交上限和normal8GPU共享配额；不重复启动BF1调度器、不自动重试。原V9 49文件与BF1 58文件哈希未改。

仍未验证：CUDA数值和真实几何梯度、CoMIR含Dropout/BatchNorm的实际续跑、RuLSIF真实FIT/CAL性能、全训练及全CAL时间/峰值显存、完整runner复用与失败路径。当前没有FROZEN_VALIDATED release，不能将CPU92通过或“已排队”写成新增方法已完成。证据见handoff/v10_validation_20261019。
