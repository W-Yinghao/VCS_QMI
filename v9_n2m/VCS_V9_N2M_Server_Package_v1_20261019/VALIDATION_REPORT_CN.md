# V9 N2M server 包验证报告

日期：2026-10-19。结论：**包内实现、CPU数学与小型SGR执行核对通过，可以交给server接入；实际CUDA和新一轮完整训练待执行。**

## 1. 本次实际执行的验证

| 类别 | 实际结果 | 证据 |
|---|---:|---|
| CPU单元、文件接口及恢复回归 | **119/119 PASS，0 skip** | `validation/LOCAL_VALIDATION.json`、`validation/unittest_output.txt` |
| 真实小型SGR，Torch2.9.0+cpu | **24/24 PASS** | `validation/TORCH_CPU_INTEGRATION.json` |
| 生产行数的合成控制/评分链 | **PASS**：145,000行，U130,500，五行块26,100 | `validation/LOCAL_VALIDATION.json` |
| 原N2F冻结源码/协议文件 | **47个文件SHA匹配** | 原 `FROZEN.json` 和只读检查 |
| 原N2F结果副本 | **30个元数据完整，3个未完整** | `validation/N2F_SNAPSHOT_CENSUS.json` |
| Slurm模板 | Bash语法通过；预退出信号转发/等待行为已用本地子进程核对 | `scripts/*.sbatch`及独立实现复核 |
| 实际CUDA | **NOT_RUN**，当前环境没有CUDA设备 | `validation/CUDA_NOT_RUN.json` |
| 新N2M完整训练 | **NOT_RUN** | 待server按冻结流程执行 |

119项由4项成本、22项汇总/塌陷、22项控制、25项身份/runtime、18项离线评分、17项运行器IO、11项训练补丁检查组成。最后一次全套回归通过。提供真实原执行源码后运行，因此本次没有skip。

源码和元数据核对来自 `87480e2` 所含的真实N2F执行快照。副本没有模型二进制，所以旧30项只能记为metadata complete；这不是说旧训练无效，更不是已在本地验证其checkpoint。server的 `inspect/prepare` 会校验真实best/last SHA并要求33/33完整。

## 2. 真实CPU SGR检查证明了什么

临时安装Torch2.9.0+cpu，并使用原SGR模块进行小型合成训练。CPU环境为Python3.12.14、NumPy2.3.5；它与生产CUDA环境不同，不能复用其PASS来放行服务器训练。

fixture为30个pair、batch8、4epochs，缩小embedding/region等维度，用两轮warmup、后续hard阶段和LR变化覆盖相关路径。完整生产配方仍是batch128、40epochs、原SGR尺寸；本地没有跑这组完整实验。

实际核对包括：

- 原版与补丁版的模型、Adam状态、RNG、best和科学轨迹精确相同，最大张量差为0。
- 连续训练与完整epoch中断后恢复，以上状态同样精确相同。
- 原格式checkpoint的步数恢复在隔离fixture中正确；生产模式则要求新身份绑定，不允许旧checkpoint被新控制采用。
- 已绑定身份的checkpoint可恢复；复制的错误臂或缺少绑定的旧checkpoint被实际拒绝。
- SGR的图像、文本、相似度三个原生部分均有参数更新。
- 实际双向hinge的逐anchor系数，与在float32权重转换后独立计算的float64参考相符；row/column最大绝对误差各为 `6.622738313311061e-08`。
- warmup/hard、LR切换、非满最后一批、真实更新计数和完成状态幂等行为正确。

CPU检查使用的原train SHA为 `cafac01f989b0bbdd88f4d21c51e7475227367d8d4750057897b77832a61302c`；补丁后为 `2b3fb040238d247d46da0bac8ac470e231c5b588ad7be1c7db84a25956532a2f`。检查的是该补丁与实际原代码的执行关系，不是另写的仿真训练器。

## 3. 控制、评分和结果核对

全行数合成检查独立读回了三臂CSV。Constant总质量在该fixture中绝对差为0；Shuffle的完整float64 multiset精确相同，保存26,100项置换；重复生成保持文件身份。评分还检验了score/w保存舍入后的后验重建与恒定分数AUROC=.5。这些是合成实现证据，不是关于VCS效果的结果。

关键失败路径已有测试：重复/缺失/多余ID、oracle字段/路径误入生成器、wrong-seed observed映射、源CSV或lock变更、篡改置换、半写入目录、错误pool大小、非有限数、控制来源尚未核验便读取标签。

完整结果汇总核对四十个epoch、45,320更新、六Recall/rSum/mR、最优checkpoint与更早并列、对应artifact哈希、实验/protocol/权重/runtime身份。单个损坏日志标INVALID，不阻止列出其他单元；未完成当前最好值不进入主终点。

严重塌陷按真实epoch内末200更新定位。旧resume错误计数已在真实N2F完整池日志上复核，未把抽样日志条数当累计更新。此轮不对缩小池的OracleDrop套用完整池1,133步规则。

固定损失评分额外核对CSV、experiment、checkpoint、batch及pair顺序身份。scorer只检查导出身份和重加权数学，不假装重新运行过模型。

## 4. 独立复核后已处理的问题

1. 加强历史VCS复用：除文件存在，还核对原权重、A/U数量、SGR cfg、协议、scale和LR。
2. 拒绝无身份旧科学产物被新臂自动采用；新checkpoint绑定本臂identity和manifest。
3. 前序单元必须属于同一完整实验身份，不能拿另一轮的完成记录解锁顺序。
4. 验证候选完整结果后才发布COMPLETE；晚到信号不会撤回有效完成。
5. 允许派生ESS等1 ULP级诊断舍入，保留原manifest字节；权重身份和配方仍精确匹配。
6. runtime核对实际可导入源码集合，防止未登记模块遮蔽冻结模块。
7. 标签读取顺序、控制来源与固定损失导出身份检查补齐。
8. 对同一Slurm job里的多臂去重计费，未知attempt和缺失allocation字段保留未知。

## 5. 服务器仍需完成的具体动作

按 `01_SERVER_AGENT_PROMPT_CN.md`，使用实际原N2F环境运行同一小型SGR CUDA检查；核对原33项完整产物、生产CSV/token/data身份；`prepare`冻结6–9次实际矩阵；完整训练、逐seed离线评分及固定风险诊断；最后从原始日志和artifact重新汇总。

生产版本为Torch2.9.0+cu128、CUDA12.8、Python3.12.12，GPU/driver/NumPy/cuDNN/线程现场记录并在新三臂保持一致。当前旧记录不足时，auto走9次同环境新训练，不能根据现时probe猜测过去的driver等信息。

原train/dev大文件继承size+mtime身份，旧冻结没有提供完整内容SHA；该限制在准备报告中明确保存。真实数据/模型二进制、CUDA、45,320更新的正式训练均不能由合成或CPU报告替代。
