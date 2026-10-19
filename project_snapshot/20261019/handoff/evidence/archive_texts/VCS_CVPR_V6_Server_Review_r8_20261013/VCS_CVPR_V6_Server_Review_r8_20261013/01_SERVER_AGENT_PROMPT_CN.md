# 给服务器 Agent 的接入消息 — r8 审查

**交付增量：最后核对到 `3e2fe59`（2026-10-13 18:51:49 UTC）；关键 critic blob 未变，接续状态见 `docs/08_DELIVERY_UPDATE_CN.md`。**

请把本包作为当前提交的**独立审查意见**处理，不要当成必须逐条接受的路线。先核对事实和实现，再决定如何回应。

仓库：`W-Yinghao/VCS_QMI`。本包截点：`54e1715e29dd2caaa6be7ed8e670829310951d39`（2026-10-13 18:36:55 UTC）；上一轮分析截点 `3a5e2f0b19d5d32a3913c3916413bae05753eab2`。新增提交修正 R7-50 integrity 顶层布尔检查并接续 phase F，未修改本次重点审查的 critic 两个模块。[SRC-00, SRC-01]

## 先执行什么

1. 记录当前 HEAD、未提交修改、已运行任务使用的代码哈希，以及 D1-FULL / E1 / E2 / phase F 的实际状态。复用原有日志与队列记录；这不是要求重复提交。
2. 逐条核对 `tables/ISSUE_REGISTER.csv`。每条回复 `接受 / 修改 / 反驳 / 已在新提交解决 / 暂无法核对`，必须给文件、函数、提交或结果支持。
3. 首批做 REV8-00 至 REV8-04：源码与合成验证、mask 四格诊断、CAL 排名校正、同问题同预算与策略 mask 的风险检查。批量 GPU／宿主重跑先给实测预算；本包本身不授权任意规模。
4. 尤其核对：
   - `r7_d1_selectors.py::train`、Qwen `train40` 阶段 B 中 `model.training` / `blocks[*].training`，以及 `requires_grad=False` 后对 mask 的梯度。
   - `soft_hard_gap` / `soft_hard_gap40` 的比较是否同时改变了 Gumbel 噪声。
   - `spearman` 的 ties／常数处理；当前 CAL 选择是否受影响。**先做评估影响分析，不覆盖原冻结选中状态。**
5. 首轮交付 `templates/SERVER_RESPONSE_CN.md`、`templates/INTAKE.json`、逐项证据、诊断结果与缺口清单。没有数据的地方写清原因，不推测。

## 对已有作业的要求

继续完成已经批准的冻结主线，不因本包擅自停机或改变数学／数据／checkpoint 规则。D1-FULL / E2 复用了被审查的模块，需记录影响范围，但不允许直接在主目录“修一下”让前后任务运行不同算法。改进在独立版本中进行。

`54e1715` 已经修复 R7-50 顶层 integrity 检查；将其列为“已修复、运行记录待核对”，不要再次当作未解决 bug。新增训练也不得借用原 phase-F 的批准自动进入官方测试。

## 条件实验，不默认启动

若阶段 B 的随机 critic／部署不一致获得实测支持，优先比较**相同阶段 A critic + 相同 gate 初始化**的旧行为与确定性 critic 行为。首轮只用一个既有宿主、VCS / Logistic ×3 seeds；新的 S3 或第二宿主按问题决定是否另加。

若 Spearman 修正会改变阶段 A 或学习率选择，另开选择敏感性记录：旧模型仍是旧模型，不能静默替换。自然数据的新实验必须有新的开发／确认范围，已读官方测试不再是盲确认。

宿主校正风险头、on-policy mask 训练、视觉覆盖约束都是**条件研究建议**，不与上述最小修正同时堆叠。不要把本包解释成“改完一定超过 JS”或“原负结果全部无效”。

## 验证命令（包内路径）

```bash
python -m unittest discover -s tests -v
python reference/repo_source_audit.py --repo /path/to/VCS_QMI \
  --out /path/out/rev8_source_audit.json
# 可选：合成输入导入生产模块；先阅读脚本。无 checkpoint、无自然数据。
python reference/repo_runtime_probe.py --repo /path/to/VCS_QMI \
  --allow-import --out /path/out/rev8_runtime_probe.json
```

工具不会获取网络数据或写仓库。若 pin／blob 不符，默认退出；后续版本通过 `--allow-newer` 显式记录，不代表审查结论自动适用。

## 报告语言

不要用“理论失败”“修复必胜”“所有旧结果作废”。应写：哪个源码行为被确认、哪个性能解释尚未检验、哪个任务结果支持什么范围。保留 REG 的学习 critic 类收益、r5 归一化收益，以及 D1 的负结果；不给 Logistic／InfoNCE 改名为 VCS。
