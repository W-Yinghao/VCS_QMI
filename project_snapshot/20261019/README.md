# CS_QMI 研究进展快照：2026-10-19

本版保存V9阶段分析、V10实现及实际执行证据。V9已完成30/33次完整训练；这是进行中快照，最终三种子比较与独立收尾审计尚未完成。

从 [阶段分析](V9_STAGE_REVIEW_CN.md)、[项目记忆](PROJECT_MEMORY_CN.md)、[代码审查](CODE_REVIEW_CN.md) 和 [V10接手说明](V10_TAKEOVER_CN.md) 开始阅读。冻结后续决定的时间优先级保留在原文中；当前执行汇总见 [CURRENT_STATUS.json](CURRENT_STATUS.json)，逐方法所有权见 [V10执行登记](handoff/V10_CURRENT_EXECUTION.json)。

V10原BF1阶段16/16完成；补充方法实际GPU门禁7/7、完整验证6/7、正式CAL家族4/7、最终单元6/15完成。通过验证与完成正式实验是不同状态，比较优势仍需科学分析。父账本中的MI/CoMIR旧失败保留，其当前状态以恢复控制器1047364为准。

本分支代码基于 `ddd3bcb09d75c3d6560ef8f1a5b52fa150e14060`。其它分支及运行版本逐一登记在CURRENT_STATUS中；不能用本分支代码冒充所有历史运行版本。快照保留V9冻结清单中除数据token缓存之外的代码文本和原BF1的58个冻结源文件，以及协议、控制脚本、结果JSON/CSV、CAL/DEV记录和交接证据。实时文件是在SNAPSHOT_MANIFEST所列时间窗口内逐个捕获的，不声称整个目录原子冻结。

原始数据、图像、token缓存、checkpoint二进制、重复训练步日志和数据访问流水保留在服务器。`evaluator_only`目录没有读取或导出；TEST_REPEAT继续HOLD、EPFL未激活。排除规则和实际复制文件的逐项SHA256见 [SNAPSHOT_MANIFEST.json](SNAPSHOT_MANIFEST.json)。已有COMPLETE/产物清单中引用的缺省文件哈希属于原服务器产物，不表示这些大文件也已上传。

原文保留服务器绝对路径以维持来源记录；映射规则见CURRENT_STATUS。指向 `ssl_pilot_*` 的路径属于对应Git工作树，其代码用登记的commit获取。源文件与调度脚本按原字节归档；从GitHub克隆后需先配置数据与工作目录并核验放行条件，不应直接启动这些历史调度器。
