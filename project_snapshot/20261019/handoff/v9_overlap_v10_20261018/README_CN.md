# V9尾部与V10交叠调度

2026-10-18用户明确要求：按“V9尾部释放的GPU立即接V10”调度。该授权取代等待整轮V9收尾后才启动V10的旧顺序。

变更现有作业，不重复提交GPU任务：清除V10 gate1045975对旧CPU收尾1045974的依赖；保留四族正式数组1045976对gate1045975的afterok依赖。四族独立任务并发上限由2提高到4，共享normal的8张GPU总配额。V10仍限定L40S；原协议的每族8 L40S allocated GPUh预算不能静默换成其他硬件。

原PROTOCOL.json的execution.gpu_concurrency=2作为历史基础保留，通过 [SCHEDULING_AMENDMENT.json](SCHEDULING_AMENDMENT.json) 明确覆盖为4。模型、数据角色、训练步数、seed、候选族内顺序、选checkpoint规则及每族预算不变；并发调整会体现在运行记录中，不冒充原封不动的调度复现。

V9原feeder1045056继续所有剩余单位与获准的原配置续跑。新独立CPU审计器接替旧1045974，完成18组/33次结果、产物哈希、三seed/尺度/Oracle/崩塌和成本分析；该审计结果不再作为V10执行条件。审计器由旧冻结代码派生，只有说明、完成状态与报告后续段落改变，核验逻辑不变，见 [AUDITOR_DERIVATION.json](AUDITOR_DERIVATION.json)。旧审计器在gate依赖解除后才取消，所有V9 GPU任务照常运行。

权威状态见 [ACTIVATION.json](ACTIVATION.json)、[STATUS.json](STATUS.json)、V10原LAUNCH账本和实时squeue。若gate仍PENDING且原因为GPU配额/资源，说明正在等待可用L40S；正式数组显示Dependency表示等待自己的GPU执行检查，并非等待V9。控制器保留每步操作日志；出现部分失败需先查ACTION_INTENT/ACTIONS和队列，不盲目重复修改或提交。

旧目录handoff/v9_to_v10_20261018的ACTIVATION/FREEZE作为历史保留，不重启旧启动策略。独立V9审计器的PAUSE_AUTO_START.json现在只暂停审计循环，不能暂停已独立的V10；需要停止V10时按用户当时指令操作现有GPU作业。
