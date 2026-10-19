# VCS / CVPR V6：理论对齐服务器包 r3

**标识：`CVPR-MV6-THEORY-20261007-r3`。论文仍为 V6 Revised；r3 是服务器任务修订，不是新稿件版本。**

本包把后续实验重新对齐到合作者原计划：equal-mixture CS、样本二次目标、bounded posterior critic、exact regression gap、估计/优化稳定性。完整 SSL 继续；冻结表示测量检验估计量在真实视觉关系上的重复性和分辨能力。属性检测和同图响应保留为解释案例，不再作为估计精度的替代指标。

## 直接发给服务器的方式

解压到研究仓库以外的目录，把 `01_SERVER_AGENT_PROMPT_CN.md` 发给 agent。先读 `docs/02_THEORY_ALIGNED_SERVER_PLAN_CN.md`，再按需读数学/实现、状态复用与增量说明。`contracts/task_manifest.json` 是任务清单，不是 trainer 配置；包内没有 sbatch、自动训练或删除脚本。

```bash
# 在包根目录运行：只检查文件与小型数学/数据约定，不访问网络或提交作业。
python tools/verify_package.py
python tools/check_reference_math.py --out /tmp/vcs_r3_local_checks.json
```

## 版本与证据

- 本轮核查：`W-Yinghao/VCS_QMI@a6fb342`，提交时间 2026-10-07 20:32:16 UTC。
- 相对会议截点 `8f22e4b`：11 个新提交。完整 SHA 与来源见 `evidence/source_index.json`。
- 未查询实时队列、未复评 checkpoint、未训练神经网络。状态是提交记录时点，不是当前服务器保证。
- 原计划与补充意见原样收录到 `reference/`；V6 Revised 和旧 r2 原包作为只读历史参考。参考文件有各自边界，后续新实验建议不冒充合作者已证明或已批准的内容。

## 阅读优先级

1. r3 接入消息、主任务书与增量说明：本轮新增任务的优先顺序。
2. 服务器已有 frozen protocols：已有作业的参数、统计规则和授权保持不变。
3. 合作者原计划 [O1]、补充意见 [O2]：理论和研究定位的来源。
4. V6 / 旧 r2：已有论文与实施细节；不覆盖服务器新稿或新结果。

若存在冲突，逐项登记并请求负责人决定，不能悄悄修改目标、删除不利结果或中止无关任务。

## 文件目录

| 文件 | 用途 |
|---|---|
| `01_SERVER_AGENT_PROMPT_CN.md` | 直接接入消息 |
| `docs/02_THEORY_ALIGNED_SERVER_PLAN_CN.md` / 根目录同名 PDF | 全文任务顺序、最小实验块、交付和预算 |
| `docs/03_METRICS_AND_IMPLEMENTATION_CN.md` | 概率对象、三类不确定性、梯度和通道约定 |
| `docs/04_SNAPSHOT_AND_REUSE_CN.md` | 最新结果、历史负结果、复用与在跑任务 |
| `docs/05_R2_TO_R3_DELTA_CN.md` | 不重开项目的增量变更 |
| `docs/06_SOURCE_ALIGNMENT_CN.md` | 原计划—补充—新建议逐项对齐 |
| `contracts/`、`templates/` | 任务清单、结果结构、交付模板 |
| `tools/`、`checks/` | CPU 参考实现、实际本地检查记录 |
| `reference/`、`evidence/` | 原材料、来源索引、已读结果转录 |

## 第一轮交付

提交 delta intake、已完成/在跑/缺失表、梯度/每样本读数存量、拟用缓存和磁盘预算，以及每个新单元的真实 runner/config/split/权限。优先复用、重算小统计；新训练只补无法恢复且有授权的缺口。近期有 home quota 事故和提交上限，先确认获批输出目录与可用空间；不得自行删除历史产物。
