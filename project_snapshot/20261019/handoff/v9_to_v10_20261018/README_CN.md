# V9收尾后自动接续V10

用户在2026-10-18进一步要求“也别只排4个任务，V9完成后可以提交V10”。本记录将“预先进入队列”和“开始GPU执行”区分：现在预提交V10，执行仍等待V9全部完成并分析。不会取消、hold或修改V9任务，不增加V9实验。

依赖链为CPU `handoff_v9v10` → `afterok` V10 GPU gate → `afterok` V10四族数组 `0-3%2`。当前增加5个GPU排队任务；四个数组任务各自运行本族冻结的完整CAL与最终seed训练，共18 CAL+12 final。维持原冻结数据、参数、seed、族内顺序与并发上限。

新作业ID以 [ACTIVATION.json](ACTIVATION.json) 为准；旧1044420/1044424已经取消，旧占位watch1044425禁止重启。V10主提交账本仍为 `../v10_bf1_20261017/LAUNCH.json`。容量按数组展开计数，总normal作业不超过29，保留V9续跑空间；提交前检查重复作业和账本，遇到提交结果不明时先保留INTENT记录，禁止盲目重试。

CPU收尾作业每30秒检查V9最新LAUNCH、18组DONE、FEED_FINISHED和实时队列。超时续跑换ID会被动态跟踪；不把旧ID退出或空队列当完成。原wave1暂停与历史作业1042382正常退出记录也需通过。失败或不完整矩阵不会放行；队列查询失败保持等待，不假设任务已结束。

最终核验重新检查V9冻结代码/输入及33个全量单位的身份、40 epoch轨迹、最佳DEV最早并列规则、所有best/last checkpoint和日志哈希，调用原冻结汇总器生成三seed、尺度、Oracle和崩塌比较。收尾报告为 `V9_CLOSEOUT_CN.md`，机器证据为 `CLOSEOUT.json`。训练计时按已完成续跑段累计，并明确与完整allocated GPU时长的区别；会计服务不可用造成的成本缺口如实保留。硬件变化、DEV探索性、Oracle额外真值等限制写入报告。不会读TEST、用V9结果改V10科学选择或重跑模型预测。

V10冻结源码和72个FIT/CAL/DEV缓存在提交前及放行前核对。只有核验与报告生成成功，CPU作业才退出0，Slurm才允许gate开始；gate通过后数组才可运行。gate及数组自身仍执行原冻结身份和GPU门禁检查。

观察入口为 [STATUS.json](STATUS.json)、[PREFLIGHT.json](PREFLIGHT.json)、[VALIDATION.json](VALIDATION.json) 和Slurm队列。若用户后来明确改变优先级，可创建本目录 `PAUSE_AUTO_START.json` 暂停自动放行（或取消本依赖链）；它不改变V9实验。正常运行不需要再次询问用户许可。

本接续程序与科学代码分别冻结。不得修改V9/V10原FROZEN中的文件，也不要启动第二份接续监控。若出现ERROR或SUBMISSION_UNCERTAIN，先核对已提交ID与队列，再修复或以新版本接续。
