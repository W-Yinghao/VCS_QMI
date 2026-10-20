# V9 N2M：下一轮 server 包

**本轮只回答一个问题：当前 VCS 的完整训练收益，有多少来自 A/U 总质量再平衡，有多少来自权重与样本内容的对应关系？**

这是接在 N2F 后的可执行增量包，版本 `VCS_V9_N2M_v1_20261019`。新控制为 Constant-U 和 Shuffle-VCS，各跑 901/902/903 三个 seed、完整 40 epoch。VCS 参照只有在原结果完整且原执行环境有充分证据相容时才复用，否则同环境重跑。新增完整训练为 **6–9 次**。

交给 server agent 时，先读 [01_SERVER_AGENT_PROMPT_CN.md](01_SERVER_AGENT_PROMPT_CN.md)。研究规格见 [PROTOCOL_CN.md](PROTOCOL_CN.md)，机器规格为 [config/protocol.json](config/protocol.json)，本地验证状态见 [VALIDATION_REPORT_CN.md](VALIDATION_REPORT_CN.md)。

## 包已实现的内容

| 环节 | 实际入口 | 产物 |
|---|---|---|
| 原 N2F 收尾核对 | `run_n2m.py inspect` | 33 单元 census、真实文件身份、缺失列表 |
| 冻结宿主复制 | `bootstrap` | 独立 runtime、原始/有效源码 SHA、唯一训练遥测补丁 |
| 真实 SGR 接口核对 | `verify-torch` | base/patch、连续/续跑、系数和身份核对 |
| 权重控制及执行冻结 | `prepare` | 3 seed 权重、置换、`experiment.json`、9 行 `UNIT_PLAN.csv` |
| 完整训练与续跑 | `run --seed` / `run --unit` | 每臂 40 epoch、best/last、完整结果、逐 attempt 记录 |
| 后验/总质量分析 | `score` | rho、Brier/logloss、AUROC/AP、校准箱、保留质量、oracle-rho 诊断 |
| 固定学生风险诊断 | `export-fixed-losses` + `score-fixed-losses` | 同一 checkpoint、同一批次、同一负候选池的重加权风险 |
| 主比较 | `summarize` | 两项配对差、六 Recall、末值/塌陷/失败/成本诊断 |
| allocation 成本 | `accounting` | `sacct` 原记录及去重后的 GPU allocation 小时 |

新包通过服务器上的原冻结输入工作，不携带 Flickr30K 特征、caption、原大 checkpoint 或整个 Git 仓库。新增控制、训练、分析代码全部在本包中；原宿主由 `bootstrap` 校验后复制到新输出，只有 `train.py` 的遥测、身份和完整性处理发生变化。

## 原始身份

- 仓库：<https://github.com/W-Yinghao/VCS_QMI>
- 交付依据：`research/v9-v10-progress-20261019` 的 `87480e2f1dd8da509e85f23002bc9cbb6d19b151`。
- 实际原宿主：`project_snapshot/20261019/handoff/v9_full_20261017/source`，其记录的 `source_git` 为 `abd281b855e7159de38d7207b083b62142b0921f`。
- 本包保留原 `FROZEN.json` 和 `GROUPS.json` 作为身份凭据。根目录同名文件不能替代该执行快照。
- 原训练文件 SHA256：`cafac01f989b0bbdd88f4d21c51e7475227367d8d4750057897b77832a61302c`。
- 补丁后训练文件 SHA256：`2b3fb040238d247d46da0bac8ac470e231c5b588ad7be1c7db84a25956532a2f`。

实际数据与环境只会在 server 的 `prepare` 时冻结。本包内 CPU/SYNTHETIC 验证不是新实验结果，也不能作为 CUDA 通过凭据。

## 最短操作路线

先核对下面四个路径。`N2M_PYTHON` 应指向原 N2F 实际使用的环境；历史 `HOST_LOCK` 的 Python3.10 注释已落后于实际结果记录。冻结生产版本为 Torch `2.9.0+cu128`、CUDA `12.8`、Python `3.12.12`。

```bash
N2M_PACKAGE=/absolute/path/VCS_V9_N2M_Server_Package_v1_20261019
N2M_BASE=/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017
N2M_OUT=/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1
N2M_PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python

"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" verify-package
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" inspect --base "$N2M_BASE" --out "$N2M_OUT/inspection"
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" bootstrap --base "$N2M_BASE" --out "$N2M_OUT"
```

`inspect` 会返回实际缺项；`--metadata-only` 只供没有大文件的分析副本使用，不允许放行训练。原 33 项尚未完成时，保留原执行器收尾；本包不会重启旧 wave1，也不会接管原 feeder。

然后，在现有 V9/V10 共享 normal 最多 8 GPU 的配额内，由当前 server agent/执行器提交一次小型 CUDA 检查。模板默认 RTX6000PRO；选择其他分区时，检查和所有新三臂都须用同一机型与软件环境。

```bash
sbatch "$N2M_PACKAGE/scripts/verify_gpu.sbatch" \
  "$N2M_PACKAGE" "$N2M_BASE" "$N2M_OUT" "$N2M_PYTHON"
```

CUDA 检查实际 PASS、原 N2F 33/33 完整后：

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" prepare \
  --base "$N2M_BASE" --out "$N2M_OUT" --mode auto
```

默认 `auto` 不猜测旧 driver/cuDNN/NumPy 等版本。当前 Git 快照的旧 hardware 记录不全，因此如果 server 没有当时保存的补充环境证据，预计会冻结为 **9 次：每 seed 新跑 VCS、Constant-U、Shuffle-VCS**。这在已接受的最多 9 次预算内。若有充分历史证据，可提供 `--historical-environments`，见交接说明；强制 `reuse6` 在证据不足时会拒绝。

`prepare` 输出实际参照决定，科学参数、权重文件和置换从此固定。每 seed 的三臂顺序：901 为 VCS→Constant→Shuffle；902 为 Constant→Shuffle→VCS；903 为 Shuffle→VCS→Constant。

在共享配额允许时，逐个提交下面命令；它不是允许同时额外占用 3 GPU 的独立配额：

```bash
sbatch "$N2M_PACKAGE/scripts/run_seed.sbatch" \
  "$N2M_PACKAGE" "$N2M_OUT/experiment.json" 901 "$N2M_PYTHON"
```

902/903 用对应 seed。每个 job 一张 GPU，按冻结顺序串行跑该 seed 的剩余臂。模板在时限前 30 分钟发出预退出信号，当前 epoch 保存后退出 75；同一命令续跑即可。先查询旧 job 的实际状态与本轮记录，避免重复排队。若只需重启一个臂，在同样的 GPU allocation 中使用 `run --experiment ... --unit s901__ConstantU`。

## 分析与回传

每 seed 先做不训练模型的后验诊断：

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" score \
  --experiment "$N2M_OUT/experiment.json" --seed 901 \
  --labels /projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4/s901/eval_only/pairs.eval_only.json \
  --out "$N2M_OUT/analysis/offline/s901"
```

要看权重在同一学生损失上的作用，用 `scripts/export_losses.sbatch` 导出原 N2F VCS best checkpoint 的八个固定完整批次，再用 `score-fixed-losses` 重加权。该分析不重新训练、不改变负候选池，定义与局限见 [docs/OFFLINE_ANALYSIS_CN.md](docs/OFFLINE_ANALYSIS_CN.md)。

所有 seed 完成或需要中途盘点时：

```bash
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" summarize \
  --experiment "$N2M_OUT/experiment.json" --out "$N2M_OUT/analysis/full"
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" accounting \
  --experiment "$N2M_OUT/experiment.json" --out "$N2M_OUT/analysis/cost"
```

汇总命令允许返回未开始、未完成、失败、未知塌陷和缺失成本。只有三个配对均完整时才生成预定主均差，不能把提前看到的两 seed 均值写成最终结论。旧 N2F 的最终 33 单元表和新 N2M 的 9 行表分别回传。

## 本地复现验证

```bash
cd "$N2M_PACKAGE"
PYTHONPATH=src "$N2M_PYTHON" scripts/validate_local.py \
  --baseline-source "$N2M_BASE/source" --full-shape \
  --out "$N2M_OUT/local_validation"
```

在 server 上，涉及 NumPy/Torch 计算的验证按该机器的 Slurm 使用规则执行。该脚本会明确标注 synthetic；原输入内容与 GPU 正式训练仍需实际 server 核验。完整交付身份由 `PACKAGE_MANIFEST.json` 和压缩包 SHA256 提供。
