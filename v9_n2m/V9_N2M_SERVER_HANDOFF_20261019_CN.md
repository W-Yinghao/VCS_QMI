# 给 server agent：执行 V9 N2M 下一轮

请按本包落实 V9 下一轮完整训练与分析。主问题是分清 **A/U 总质量再平衡** 与 **权重对样本的内容关联**；不要仅采用现有 GitHub 报告的比较结论。以锁定权重、完整轨迹和两项预定配对差重新计算。

本包是可执行增量包。先读 `README_CN.md`、`PROTOCOL_CN.md`、`config/protocol.json` 和 `VALIDATION_REPORT_CN.md`，再按下面顺序操作。交付时已经完成 CPU/数学/接口验证；CUDA 与新的完整训练须在服务器实际执行。本交接不表示这些作业已由主会话远程提交。

## 1. 找到正确执行源与当前作业所有权

使用原 `handoff/v9_full_20261017`，其 `FROZEN.json` 应与包内 `provenance/N2F_FROZEN.json` 一致。主会话审阅的 Git 提交是 `87480e2f1dd8da509e85f23002bc9cbb6d19b151`，实际宿主在其中的 `project_snapshot/20261019/handoff/v9_full_20261017/source`。不要用仓库根 `src/v9` 替代。

查询 server 当前 N2F、V10 的真实执行入口、账本、队列和剩余配额。保留现有在跑作业、原产物及失败记录，按当前全局策略共享 normal 最多8 GPU。新包只提供单seed顺序执行，不创建第二个全局 feeder。三个seed可以在配额内独立推进；示例命令不是额外3卡授权。

历史快照为 N2F30/33，缺 BCE SUM902、VCS SUM903、BCE MEAN903。先让原执行器按原配置收完实际剩余项。用 `inspect` 从文件重建33项，不依赖旧表格的 status 字段，也不按中途分数选方法。现有 job ID 和旧预计结束时间必须以服务器当前记录为准。

## 2. 解压、核对与 GPU 接口检查

把包解压至独立目录，建立独立输出 `outputs_v9/n2m_20261019_v1`。例子使用原N2F实际Python环境；如果路径迁移，只替换路径，保留源文件身份、版本和 train/dev 的时间戳/大小。

```bash
N2M_PACKAGE=/absolute/path/VCS_V9_N2M_Server_Package_v1_20261019
N2M_BASE=/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017
N2M_OUT=/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1
N2M_PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python

"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" verify-package
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" inspect --base "$N2M_BASE" --out "$N2M_OUT/inspection"
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" bootstrap --base "$N2M_BASE" --out "$N2M_OUT"
```

`inspect` 在旧单元未完整时返回2并列出缺项；这是旧区块尚未收尾，不是新代码的失败。`bootstrap` 只在新输出复制、核对宿主，并应用已固定的单文件补丁。

按 server 的 Slurm 使用规定运行 NumPy/Torch 验证。给相同机型的一次实际 CUDA allocation 运行：

```bash
sbatch "$N2M_PACKAGE/scripts/verify_gpu.sbatch" \
  "$N2M_PACKAGE" "$N2M_BASE" "$N2M_OUT" "$N2M_PYTHON"
```

默认分区 RTX6000PRO。改用 L40S 等分区时，在 `sbatch --partition=...` 参数中选择，并让后续新三臂使用同一环境。检查原训练与补丁训练、连续与续跑、原生 SGR 参数更新、显式系数和身份拒绝。必须查看实际 `SERVER_VERIFICATION.json` 的 `status=PASS`，不能用 CPU 报告或命令已提交代替。

生产软件固定为 Torch2.9.0+cu128 / CUDA12.8 / Python3.12.12。原 `HOST_LOCK` 的环境注释写着Python3.10，但实际 full_result 记录为3.12.12，应使用实际执行环境。GPU型号、driver、NumPy、cuDNN、线程与精度设置从真实 allocation 采集并锁定。

## 3. 自动冻结6–9次矩阵

当旧 N2F33/33已完成且 CUDA 检查通过：

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" prepare \
  --base "$N2M_BASE" --out "$N2M_OUT" --mode auto
```

`prepare` 将：

1. 核对原source、observed、noise manifest、VCS CSV/lock、vocab、token cache、train/dev身份与旧33项checkpoint。
2. 从每seed原VCS CSV生成Constant-U及一次固定的observed-image五行块置换，读回核对后发布。
3. 将各seed VCS参照决定写入 `reference_decisions`；表格始终九行，新完整训练数量6–9。
4. 写入 `experiment.json`、`EXPERIMENT.sha256`、`UNIT_PLAN.csv`。随后这些身份不能原地改变。

当前旧 hardware 记录不完整，不能凭“还是L40S”认定全部数值环境相同。**没有当时的补充证据时，auto选择9次同环境新训练**，这是既定最多9次预算内的默认处理。无需为这个已规定分支另问主会话。

只有确有原执行期间保存的环境证据，才使用 `--historical-environments /absolute/path/historical_environment_evidence.json`。该JSON按seed映射，每项含 `evidence_path`、`evidence_sha256`、`captured_during_original_job=true`、`original_job_id` 和 `environment`。证据必须来自当时，不能把现在的probe补写成历史。需要补足的字段见 `runtime.py:ENVIRONMENT_KEYS`，包括版本、driver、TF32/确定性/线程和Torch build摘要。若证据矛盾或缺失，使用auto/replay9；不强行reuse6。

## 4. 执行完整40轮与恢复

六个控制必跑；VCS按冻结action复用或新跑。每seed顺序：

| seed | 第1臂 | 第2臂 | 第3臂 |
|---|---|---|---|
| 901 | VCS | Constant-U | Shuffle-VCS |
| 902 | Constant-U | Shuffle-VCS | VCS |
| 903 | Shuffle-VCS | VCS | Constant-U |

按共享配额逐个提交：

```bash
sbatch "$N2M_PACKAGE/scripts/run_seed.sbatch" \
  "$N2M_PACKAGE" "$N2M_OUT/experiment.json" 901 "$N2M_PYTHON"
```

902/903修改seed。每job一张GPU，串行执行该seed剩余新臂。若预算/时限需要单臂作业，在同机型allocation里执行：

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" run \
  --experiment "$N2M_OUT/experiment.json" --unit s901__ConstantU
```

代码核对前序单元的完整实验身份和每臂独占锁。新的控制必须从自身seed初始化训练，不能复制原VCS的epoch4/last checkpoint。运行器拒绝无身份的科学产物，checkpoint也绑定当前臂和manifest。

时限前30分钟模板发SIGUSR1；当前完整epoch验证、落盘后返回75，再提交同一命令恢复。硬超时、OOM或设备错误保留原失败与最近完整epoch。先核对原job实际退出状态，避免重复排队。出现严重塌陷也继续跑到40轮，除非非有限值或基础设施无法继续。

成本按当前观测粗估：9次RTX完整训练约57 allocation内小时，6次L40S控制约68小时，均为既有中位数外推而非承诺。三臂同RTX串行约19小时具有可行性，但仍依赖节点吞吐；预退出和恢复必须启用。实际以本轮attempt及sacct统计为准。

## 5. 不训练模型的评分与固定风险

冻结控制之后，分别对901/902/903执行后验评分。示例：

```bash
N2M_LABELS=/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4/s901/eval_only/pairs.eval_only.json
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" score \
  --experiment "$N2M_OUT/experiment.json" --seed 901 \
  --labels "$N2M_LABELS" --out "$N2M_OUT/analysis/offline/s901"
```

这一步才读取机制标签。指标包括rho原始/投影/实际、后验校准、AUROC/AP、质量与ESS、clean/mismatch保留、残余噪声、同T的oracle-rho敏感性。标签不是语义真值，原rho区间不作覆盖保证。AUROC低或校准差不构成删seed/停训练门槛。

补固定学生诊断（每seed八个batch推理，不新增完整训练）：

```bash
sbatch "$N2M_PACKAGE/scripts/export_losses.sbatch" \
  "$N2M_PACKAGE" "$N2M_OUT/experiment.json" 901 \
  "$N2M_OUT/diagnostics/s901/fixed_losses.csv" "$N2M_PYTHON"
```

实际导出结束后：

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" score-fixed-losses \
  --experiment "$N2M_OUT/experiment.json" --seed 901 --labels "$N2M_LABELS" \
  --loss-csv "$N2M_OUT/diagnostics/s901/fixed_losses.csv" \
  --out "$N2M_OUT/analysis/fixed_loss/s901"
```

固定使用原N2F VCS SUM best checkpoint及规定的epoch5排列前八个batch。三臂和clean oracle复用同一份ell，保留相同观测负候选池；结果只解释同一模型上的加权风险，不代替独立完整训练终点。

## 6. 汇总和科学解读

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" summarize \
  --experiment "$N2M_OUT/experiment.json" --out "$N2M_OUT/analysis/full"
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" accounting \
  --experiment "$N2M_OUT/experiment.json" --out "$N2M_OUT/analysis/cost"
```

先展示三seed三臂best full DEV rSum和两项配对差，再结合posterior/总质量/固定风险/轨迹解释。三对都完整后才报主均差；缺项和失败如实列出。n=3只做描述性配对比较，不给pair-IID置信区间、显著性或等价结论。last/末5轮只诊断，不替换best主终点。

工程投入规则已经固定：两项VCS主差均值均≥3rSum、各至少2/3seed正向、严重塌陷不更多，才继续扩展内容关联可靠度。规则未过不说明等价；可转向解释A/U再平衡。若只优于Constant而没有优于Shuffle，仍需考虑分布/ESS/梯度效应。

固定pair机制标签不进入critic、校准选择或控制生成。teacher与生产critic不重拟合。本轮最多9次完整训练，不增加MEAN/LR/rho/anchor/COCO/REG/Unweighted新网格。常规路径适配、基础设施恢复和包内实现错误先完成修复；若需要改科学配方、数据角色或超出九次预算，向主会话提交具体差异与证据。

## 7. 最终交回主会话的文件

回传新包实际代码commit及package SHA、原N2F最终33项表、`experiment.json`/receipt/`UNIT_PLAN.csv`、CUDA实际报告、controls manifest与置换SHA、每臂identity/run_manifest/full_result、完整train_log和progress、grad窗口、全部attempt和sacct记录、三个离线后验表、三个固定风险诊断、最终summary/两个配对差表。

大图像特征、caption原文和模型二进制可以留server。若只导出元数据，明确写 `checkpoints_not_materialized`；独立汇总不得因此把未核验的checkpoint写成完整主比较。主比较的最终核验在仍持有二进制的server运行，报告保留其artifact SHA。

提交结果时不要只写“VCS有提升”。需要给出这两项主比较的每seed数字、完整性、成本与可支持的机制结论；没有支持的方向也完整交回。
