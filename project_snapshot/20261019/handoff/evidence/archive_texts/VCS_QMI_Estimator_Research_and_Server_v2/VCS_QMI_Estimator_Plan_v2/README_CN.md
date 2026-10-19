# VCS-QMI 研究与服务器文档 v2

## 两份主要文档

- `VCS_QMI_Estimator_Research_Plan_v2_CN.md/.pdf`：研究主线、直接 CS 对照、完整 SSL、补充意见的处理和来源。
- `VCS_QMI_Estimator_Server_Agent_Spec_v2.md/.pdf`：现有分支接续、数学/实现契约、实验与交付规范。

本版将 **CS estimator 直接比较与完整视觉 SSL** 设为主任务。组合、噪声和增量分析保留为辅助工作；不需要它们先成功，才允许完成原 estimator 的完整 SSL。

## 如何交给服务器

以服务器 Markdown 为实施依据，先做 §1 的现有分支与授权核对，再补直接 CS 对照、复用已完成诊断并提交完整 SSL / S4 的配置与预算。已有批准任务按其授权执行，不因本文自动重启或取消。本文本身不启动任何作业。

P72 的结果由用户补充带入；P84 是本轮另读的已提交报告，已明确区分来源。补充之后的其他实时进展，仍需服务器按实际 HEAD / 工作树 / 作业状态核对。

## 目录

- `inputs/`：原样保存的用户补充。
- `support/checks_v2.py`：本轮新增恒等式与噪声标度的小型核对。
- `support/checks_v2_results.json`：实际本地运行输出。
- `support/build_pdfs.sh`、`support/header.tex`：PDF 构建脚本与样式；不含字体文件。
- `SOURCE_MANIFEST.json`：输入文档 hash、仓库阅读范围和外部文献用途。
- `DELIVERY_CHECKS.json`：PDF 页数、文本与渲染检查记录。
- `PACKAGE_SHA256.json`：包内文件校验值。

## 运行范围

本轮没有神经训练、服务器作业或 GitHub 写入。没有复现补充中未附带的完整 noise_scale_check.py；只对文档新增/限定的数学表达作了独立小型检查。

重新构建 PDF 需要 Pandoc、XeLaTeX 与样式中命名的字体。仅提供脚本，不分发字体。
