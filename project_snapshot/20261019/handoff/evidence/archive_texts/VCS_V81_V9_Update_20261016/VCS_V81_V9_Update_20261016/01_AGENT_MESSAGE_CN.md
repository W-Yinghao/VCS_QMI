# 给 server：V8.1 / V9 增量审查

请按 SERVER_UPDATE_CN.md 核对当前快照。不要重做 V8.1 全包，不启动 V8.2、V9.1 或 V10。

1. V8 TIFA160 已完成：更新状态和外部汇总；两套人类标签是同一批图像；保留原次级评价、原聚合、各seed和标签用途，不据此调整模型。
2. 推送已被 V8 报告引用的 v8.1/utility-diagnostics 及 D02–D06 数值。我这次读取该 ref 返回404；不能据此断定服务器未做，先给可访问的commit和完成表。epoch0已经更正，不再重训。
3. 先修 N2 真实锁文件和加载接口：HOST_LOCK 当前分号串联字段不符合 YAML；unit 把字符串host送入期望mapping的HostConfig；precheck和unit的键路径不同。统一typed schema，禁止静默默认，用真实锁文件和实际入口测试。
4. precheck 的 AUROC>0.999 / <0.55 不应默认阻断正式块：转成warning，或给出明确授权和范围理由。高AUROC并不意味着噪声加权无用途，低AUROC也不否定NCR/L2RM。不能为通过门槛重训teacher或筛seed。
5. 修复 precheck.spearman 的并列排名，常数=>null+reason；重新输出既有表，不重训模型。
6. 等价复用需完整运行fingerprint，不能只看系数与seed；20% observed-image留出区间暂不称独立样本保证，补配对来源依赖检查。
7. G1/G2、REV10和已批准任务照原范围推进；现有数据不再以许可讨论阻塞。这里只暂停有明确接口错误的新N2提交，不全局停GPU。

收到后先回复 U01–U07 的接受/修正/反驳、源码证据、是否已在本地修复及影响范围，再提交必要小补丁。C1的九个完整适配模型仍需原诊断证据和批准，本件不授权它们自动启动。
