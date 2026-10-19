# VCS CVPR 稿件 V6 — 服务器交接材料包

**包标识：`CVPR-MV6-HANDOFF-20261006-r2`**  
**依据：2026-10-06 修订后的 CVPR manuscript V6（Revised）；论文仍为 V6，仅交接包从 r1 更新到 r2。**  
**状态：READY_FOR_INTAKE；新增科学实验协议仍需在服务器核对、补全并冻结。**

## 0. 已接收 r1 时，先看这里

先读 `08_R1_TO_R2_DELTA_CN.md`。本次是论文参考与任务分工的增量同步，不是更换学习目标、重训旧结果或回退服务器。

`reference_v6/` 已换为最新 V6 Revised 源包的原样副本。r1 的接入核对、训练、oracle 与测量记录继续有效；核对未变的输入后直接复用。旧 r1 目录留作历史，在仓库外另解压 r2，不将本包整体覆盖研究仓库。已接入过 r1 的服务器只返回 delta intake，不重复从零盘点。

这轮主要补清两种比较：**固定 encoder 比较 estimator（Table 3）；固定测量流程比较最终 VCS 与外部 SSL encoder（Table 4）。** 后者是明确任务，不再仅写成“VCS encoder 可选加一行”。

## 1. 本包要完成什么

围绕 V6 的同一个研究问题整理执行：把 mixture-reference CS 依赖写成可从样本学习、具有明确误差关系的 estimator，并验证它在视觉表示**学习**和**测量**中的用途。不是重新寻找一条研究路线，也不是仅推进局部可解释性实验。

| 工作 | 在论文中的作用 | 本轮默认动作 |
|---|---|---|
| Estimator / 已知压缩真值 | §3 与 §5.1 的统计和测量证据 | 复用已完成 P86、P118；核对来源、生成表图，缺什么补什么 |
| 完整视觉 SSL | §4.1、§5.2 的主应用 | 整理当前结果与外部对照；衔接已有规模化工程／训练计划 |
| 测量器检查 | §5.3 第一步／Table 3 | 在同一外部模型上补齐 VCS、JS、同目标 kernel、条件 HSIC 对照 |
| SSL 表示分析 | §5.3 第二步／Table 4 | 固定测量流程，复用最终 VCS 与 SimCLR 的同条件检测和预测响应；按库存补外部模型 |
| 真实视觉压缩 | §3.3 的视觉测量应用 | 复用 nested_exact；仅比较实际确定性映射，记录有限拟合误差的影响 |
| 写作集成 | 全文 | 结果按 claim 对齐；不预填新结果、不混合不同 fixture |

依据见 `reference_v6/Paper_Positioning_and_Interpretability_CN.md` §1、§4、§8，以及 `reference_v6/source/main_body.tex` / `supplement.tex`。

## 2. 先读哪些文件

先读 `01_SERVER_AGENT_PROMPT_CN.md`，然后读 `docs/02_PAPER_AND_METHOD_CONTRACT_CN.md`、`docs/03_EVIDENCE_REUSE_CN.md` 和 `docs/04_EXECUTION_PLAN_CN.md`。实现条件审计／压缩时再读 `docs/05_AUDIT_AND_COMPRESSION_SPEC_CN.md`；提交结果时使用 `docs/06_DELIVERY_AND_WRITING_CN.md` 与 `templates/`。

`reference_v6/` 是本次收到的 V6 Revised 源包的逐文件原样副本：正文、完整 PDF、正文与补充 Markdown、中文修订说明、LaTeX、BibTeX、源图、修改差异和证据检查。它不是服务器的新结果输出目录。

## 3. 版本与事实边界

V6 结果快照：

```text
repository: W-Yinghao/VCS_QMI
commit: bc28bf89c94d5331de811a83aa8bcbfb77d7d155
commit_time_utc: 2026-10-06 05:58:03
```

本包不把该快照当成当前服务器 HEAD。服务器先比较快照与当前 checkout，并读取现有预注册、日志和 checkpoint 清单。新结果单独进入 reconciliation 表，不覆盖 V6 原文件。

r1 中列出的仓库来源仍保留为历史入口；r2 只对照已提供的 r1 与 V6 Revised 文件，不重新检查 GitHub HEAD、checkpoint 或作业队列。已报 Table 4 的来源、转换和旧 null 说明随修订稿附入。

修订 V6 的规定标为 **[V6-R]**，未变的原稿规定仍标为 **[V6]**；r1 继承的固定快照信息标为 **[SNAPSHOT]**；交接实现建议标为 **[PROPOSED]**。新增任务编号与记录字段不是论文中的新算法。

## 4. 接入步骤

在当前仓库外解压本包，不覆盖现有 `reports/` 或工作目录。先运行：

```bash
# 以下命令均在材料包根目录执行；只验证文件和小型代数，不训练模型。
python tools/verify_package.py
python tools/check_contract.py
python tools/check_r2_sync.py

# 将 <repo-path> 替换为现有 VCS_QMI checkout 的绝对路径。
# --out 必须位于被检查仓库之外，防止把接入快照写入当前工作树。
python tools/intake_repo.py --repo <repo-path> --out /tmp/vcs_mv6_intake.json
```

`intake_repo.py` 不 fetch/pull、不 checkout/reset、不启动作业；只读取 Git 元数据、已跟踪文件名和证据报告内容摘要。它不验证 checkpoint 是否有效，也不判断作业是否完成。服务器仍需按 `templates/INTAKE_REPORT_TEMPLATE.md` 完成内容核对。

原 V6 检查另需 NumPy、SciPy。为保留参考副本，应将该脚本复制到独立输出目录后执行；具体命令在 `tools/README_CN.md`。r2 实际本地核对记录位于 `checks/`；`checks_r1_history/` 只保留 r1 当时的记录。

## 5. 权限和任务衔接

本次请求是**准备材料包**，不是替用户在此提交训练。接收端可以先完成阅读、只读盘点、协议草稿和小型验证；已有获批任务按其原授权继续，不因 V6 叙事重组而取消、重启或改参。新增 GPU 实验和总预算按项目已有授权流程处理；本包没有替负责人批准任意规模训练。

尤其注意：P132 的 V6 快照只完成 ImageNet 数据、分布式配对和吞吐预检查。报告明确没有训练；后来是否已有新授权／运行，由 intake 核对。不能把“ImageNet 主证据需要补齐”理解为“现在自动提交全部 ImageNet 训练”。

新任务用本包逻辑编号 `MV6-*`；在服务器与主会话协调后映射到尚未占用的 `Pxxx`。不要据此重新占用 P116、P118、P132 或现有 server-v7 编号。

## 6. 接收端第一轮应交付什么

首次接入者完成原 intake；已接入 r1 者只做 delta intake：确认参考稿已更新、Table 3 与 Table 4 的任务分工、Table 4 已有来源与单位、外部 encoder 的真实缺口，以及哪些旧任务无需变化。第一轮交付格式见 `templates/INTAKE_REPORT_TEMPLATE.md`。

证据齐全且授权范围明确的单元可以继续；遇到局部缺失就标出该单元阻塞，不应因此停掉所有独立任务，也不能用另一个方便的实验悄悄替换它。
