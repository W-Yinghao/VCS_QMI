# V10 最多8并行的调度交接

2026-10-18 13:29 UTC实队列核验通过。用户要求“最多八个并行”，延续“V9尾部释放的GPU立即接V10”。V9继续完成原33次全量训练；V10与它共享normal的8张GPU，V10使用冻结要求的L40S。原四任务数组只有四个独立任务，单改成`%8`仍只能并行四个，因此改用冻结runner已有的CAL和FINAL独立入口。

| 作业/阶段 | 当前ID或数量 | 放行条件 |
|---|---|---|
| CPU补位loop | 1046189，RUNNING | 每30秒检查和补位 |
| V10 GPU执行gate | 1045975，PENDING/QOSMaxGRESPerUser | 可用normal L40S；无V9收尾依赖 |
| MF_VCS_perm1 CAL | 1046190 | afterok:1045975 |
| MF_VCS_M0 CAL | 1046191 | afterok:1045975 |
| MF_VCS_M1 CAL | 1046192 | afterok:1045975 |
| MLP_VCS_perm1 CAL | 1046193 | afterok:1045975 |
| 独立FINAL seed | 4族×3seed=12个，自动补交 | 该族CAL成功且完整性核验通过 |
| 旧all-mode数组 | 1045976，CANCELLED，RunTime=0 | 已替代，不可重提 |
| V9原loop / 独立收尾审计 | 1045056 / 1046165 | 照常运行，不限制V10 |

CAL候选仍在各族内部按冻结顺序串行，CAL阶段自身最多4路。每个家族完成CAL后，其seed1001/1002/1003可独立运行，与其他家族CAL或FINAL同时使用最多8张卡。不是等待全部家族CAL结束才开始FINAL。因当前V9占用8张normal GPU，核验时V10尚未运行；并发上限不代表已获得8张L40S。

调度目标为有足够可就绪单元时最多8个pending，V10活跃GPU作业含gate最多16，所有项目normal任务展开数组逐项计数最多29，给原30提交额度留一个位置。CPU job不计GPU；runfill不混入normal额度。初期只有4个CAL可以排队，FINAL需等该族CAL核验，因此不伪造额外实验凑8个pending。运行数由现有normal每用户8GPU配额约束，V9和V10合计共享该额度。

原科学冻结SHA为`a03e88277ba61159d4cd70255d72bf13515b8cac9843465329f6f9bcb15f58a2`，源码commit为`b8f59ea825dc7a2064457d47a628bef09ca4e185`。58个源文件和2个冻结输入核验通过。18个CAL候选、每族8 L40S GPUh、12个FINAL、全量3000步/12个CAL读数、CAL选择规则和数据角色保持原样；TEST134仍HOLD。并发及作业分组由 [调度修订](SCHEDULING_AMENDMENT.json) 明确覆盖，不改原协议字节。每seed独立初始化torch和采样RNG、独立结果目录/锁，最终训练仍绑定同一个已验证CAL选择哈希；拆分产生的启动成本计入FINAL。

loop必须同时看到CAL作业离开队列、batch exit=0、成本账本END/COMPLETE以及完整选择/产物哈希校验通过，才补交对应FINAL。FINAL runner会再次核验gate和选择依赖。独占dispatcher锁与先记录提交意图再调用sbatch防止重复；不明确的提交结果需人工核对队列和账本，不自动重试。确定的提交额度拒绝则稍后补交。失败家族停止后续提交并保留产物；已运行的独立seed保留各自核验记录，家族仍需审查。不会将队列消失或退出0单独解释为实验完成。

权威状态是 [LEDGER.json](LEDGER.json)、[STATUS.json](STATUS.json)、主目录 [LAUNCH.json](../v10_bf1_20261017/LAUNCH.json) 和实队列；不要根据本文初始ID重复提交。若出现`ERROR.json`，先核对失败原因再恢复。`PAUSE_SUBMISSIONS`仅暂停新V10提交，不影响已在跑任务。不要重启旧1044425优先占位watch或恢复已取消的1044420/1044424/1045976。

7项 [控制检查](VALIDATION.json) 通过；[激活记录](ACTIVATION.json) 保存旧数组零运行取消证据，[队列核验](ACTIVATION_VERIFIED.json) 记录新loop、四个CAL、保留gate和V9作业状态。它们不构成科学结果。CPU节点时钟比查询侧约慢106秒，核验的`observed_utc`采用查询侧时钟；不要据此误判loop失联。
