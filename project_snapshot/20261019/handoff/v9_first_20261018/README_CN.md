# V9 优先收尾、保留在跑任务

2026-10-18 21:48 UTC 激活核验。用户要求：不取消已经运行的任务，先完成 V9，再实现 V10 剩余工作。本次没有取消、抢占或停止任何运行作业；训练代码、实验网格与已有产物保持原冻结身份。

## 当前执行安排

| 项目 | 激活时状态 | 后续动作 |
|---|---|---|
| V9 全量训练 | 25/33 完成；5 运行、3 排队 | 原 feeder 1046651 继续原矩阵及已授权的异常恢复 |
| V10 已运行 | 1046192、1046915、1046916 | 正常完成，产物仍由 1046189 核验 |
| V10 尚未运行 | 1046917 | 已 hold；仅在 V9 收尾通过后释放 |
| V10 尚未提交 | M1 最终 seed1001/1002/1003 | 1046189 受 PAUSE_SUBMISSIONS 控制，暂不提交 |
| V9 独立收尾 | 1046165 | 等待全部训练完成，核验完整矩阵、退出和产物，生成结果分析 |
| 阶段切换 | 新 CPU 作业 1047062，stage_bridge | 每 30 秒检查；通过收尾条件后恢复现有 V10 调度 |

两阶段继续共享 normal 的 8 张 GPU。空出的配额优先让已提交 V9 启动；剩余 V9 不足 8 个可运行单元时，不再新增 V10 填空，也不增加 V9 实验凑数。runfill 属其它项目，不在本次调度范围。

## V10 放行条件

必须同时满足：冻结的 18 组、33 次全量训练全部成功；无 V9 运行或排队作业；feeder 无失败组；独立收尾审计 PASS 且已退出；当前完整矩阵、冻结科学代码、结果文件及分析报告哈希与审计记录一致。任一条件缺失就保持暂停。

控制器只释放本次策略登记的 hold，不解除已有用户或管理员 hold；随后归档自己写入的 PAUSE_SUBMISSIONS，让现有 1046189 恢复提交。后续新策略必须更新 [ACTIVE_SCHEDULING_POLICY.json](../ACTIVE_SCHEDULING_POLICY.json)，旧控制器据此停止放行。不要重复启动控制器、提前手动 release 或删除 pause。

控制器不依赖自己完成来通过 V9 检查，作业名使用 stage_bridge；V9 独立审计将名称以 v9 开头的作业视为尚未收尾。控制器在查询异常时不据此判定队列为空；如果出现 ERROR.json 或心跳停更，应先核查原因与实际队列，再按新版本恢复控制，不能绕过核验放行。

## V10 剩余实现

完整工作单见 [V10_REMAINING_WORK.json](V10_REMAINING_WORK.json)。V9 收尾后，先完成既有 BF1 四族的全量结果和分析，再补齐 MLP_VCS_perm8、InfoNCE_host[J_perm1]、MI_native、NMI_native、CoMIR_native、exact-moment RuLSIF、MIND_2D_adapted。每种方法需要实际适配器、针对性数值/几何验证、成本测量、方法附录冻结及完整实验；不能把门禁通过写成科学结论。

随后按既定前置条件激活 EPFL，并在完整方法集及全部选择规则冻结后执行 BF TEST_REPEAT134/112，最后完成可复现结果、成本/失败记账、代码审查和项目记忆。TEST、evaluator_only 的角色限制继续有效。

**自动化范围：**1047062 能自动恢复现有冻结 BF1 作业，并写出 V10_IMPLEMENTATION_READY.json；它不会编写新适配器。工作单中的剩余实现当前仍是待执行工作，不能将排入清单视为已完成。

## 核验与追踪

- [ACTIVATION_VERIFIED.json](ACTIVATION_VERIFIED.json)：激活时全部 8 个 GPU 作业仍在运行，0 个取消，1046917 已 hold，V10 新提交暂停。
- [CONTROL_TESTS.json](CONTROL_TESTS.json)：6 项控制测试通过，覆盖只 hold pending、启动竞争、完整收尾门槛及旧策略不得放行。
- [FREEZE.json](FREEZE.json)：控制代码与工作单冻结；SHA256 `5de4bd678f73bb7324f46508fec428de9d4b144bfe76e3b23e6847f093a51f20`。
- [STATUS.json](STATUS.json)、[LEDGER.json](LEDGER.json)：当前阶段、等待原因、持有及释放记录。
- [V9 动态进度](../v9_full_20261017/FEED_STATUS.json)、[V10 动态进度](../v10_parallel8_20261018/STATUS.json)：实时状态优先于本文激活快照。

本说明不属于已冻结控制源文件，不改变运行中的源码身份。
