# 09 · 与V8并行的执行契约

## 1. 不串线
| 项目 | V8 | V9 |
|---|---|---|
| 科学问题 | 总体评分分布与选图 | 可靠对应估计、加权学习、有限配准 |
| 输出 | V8既有目录 | 新V9命名空间 |
| 主评价 | GenAI-Rank与保留确认集 | Flickr/COCO retrieval、独立量化、REG |
| 模型选择 | 原V8 frozen规则 | 本包reference角色与任务CAL |
| 共享 | 原始只读图像/固定特征/工具版本可共享 | 哈希与角色审查后只读挂载 |

不修改V8的labels seal、校准规则、checkpoint或正在运行的脚本。V9不得依据V8外部失败例选择自己要报告的子集。

## 2. 作业设计
新worktree建议`v9/reliable-correspondence`。先读该分支是否已存在；不要猜本地路径。由`.env.local`或YAML输入repo_root/data_root/output_root；示例中不给不可验证的服务器绝对路径。

CPU参考测试、有限支持实验、数据查验先行。GPU任务按实验块打包，支持恢复，保存RNG与optimizer状态；恢复后先与未中断小运行核对。V8已经在跑的任务不取消；并发份额按既有授权与实测可用资源安排，未授权大块由owner确认。

每个任务开始锁code hash/config/data/teacher/weights/noise-map/evaluator hash。结果文件和解释分开提交。技术失败重试不能悄悄换loss/数据/学习率；修改注明新version和受影响单元。

## 3. 第一轮交付
V9_00_INTAKE.md：当前heads、环境、V8隔离、已有数据可复用项。
V9_01_SOURCE_AND_BUDGET.md：基线代码/许可、训练/验证独立性、one-epoch时间。
V9_02_PROTOCOL_FILLED.yaml：host真实字段、科学主指标、methods、seeds、calibration roles。
V9_03_BASELINE_FIDELITY.md：NCR/L2RM官方评价与本地差异；ESC/ASL/ReCon可执行性。
V9_N1_RESULTS/：有限支持+自然混合结果，不含测试注入标签泄漏。

定义正确和资源到位后，执行N2-F完整块，不以VCS胜BCE为启动条件。NR与REG的主结论分开。

## 4. 不在本包自动授权范围
付费API、大规模人工标注、重开已关闭SSL/R线、读取V8未开放评价、删除历史产物、上传第三方数据/权重、CC3M大预训练、无权限数据下载、重匹配OT新机制与GREC完整训练。需要时形成单独增量，不能从“V9并行”推断全部都可做。
