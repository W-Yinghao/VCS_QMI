# 07｜来源、实现入口和本次整理范围

## 1. 主依据

`reference_v6/` 31个文件从 V6 Revised ZIP 解出并逐文件原样复制；其内部路径结构不变。完整PDF13页，main+references PDF7页。来源hash和本包文件hash见 `contracts/reference_input.json`、`PACKAGE_MANIFEST.json`。

本包的论文主线／方法／已报表格以V6为依据。2026-09-28原计划与补充意见仅通过V6已吸收部分进入当前任务；不重新激活其中过时的“stable alternative全面占优”“噪声门控”“模型选模”等方向。

r2 核对的是已提供 r1 与修订V6文件，不是新一轮仓库检查。下表保留 r1 已读报告的入口信息；运行前应读取服务器当前文件。

修订新增的 Table4 对应 `reports/P118_nest_results.md`，blob `2ebe8acb02365c9417d6660a251d272344912bbf`；修订稿的原样 `evidence/retention_source_rows.json`、`revision_contract.json` 与 `V6_revision.patch` 提供逐项来源和修改范围。本包新增记录格式是交接组织建议，不是原报告的既有格式。

## 2. r1 同一个固定提交的补查 [SNAPSHOT，历史记录]

基准均为 `bc28bf89c94d5331de811a83aa8bcbfb77d7d155`；只读取以下报告内容，不读取当前default branch、不复评checkpoint。路径相对于仓库，blob_sha是工具返回的文件对象哈希。

| 报告 | 读取范围 | blob_sha | 本包用途 |
|---|---|---|---|
| `reports/P116_V5_SOLVE_REPORT_20261003.md` | 全文 | `58d2a9aab1841c55f8e9082173c83ea2010f82b3` | solver结果与短/长预算区别；不照搬其长预算30×结论 |
| `reports/P116_V5_SOLVE_PREREG_FROZEN_20261003.md` | lines1–65 | `4bab19fa53976f5c5e1e1313113a0fee41887cba` | solver入口、split、lambda/calibration、logit约定、置换 |
| `reports/P118_V5_NEST_PREREG_FROZEN_20261003.md` | lines1–66 | `388b48bb78de4e1b4ce32b6fad6b9c56bd5166eb` | nested入口、oracle/visual fixture谱系、null与读取规则 |
| `reports/P132_V6_IMAGENET_PREFLIGHT_REPORT_20261005.md` | 全文 | `f61a69246fbe80023c0d66176e869c3e1fdf3f7a` | ImageNet入口和“只有预检查、无训练授权”的边界 |

本文件是定位摘记，不冒充这些report的完整原文副本；运行前在实际仓库用 `git show <snapshot>:<path>` 读取。读取范围以本表为准。

## 3. 已由报告定位的入口

| 工作 | 文件／符号 | 证据范围 |
|---|---|---|
| 固定模型solver | `src/vcs_measure/solve.py`、`scripts/p116_solve.py`、`tests/test_p116.py` | P116 prereg直接列明；本轮未读实现源码 |
| 原ridge+tanh | `scripts/precheck_d_tests.py` 中 `closed_form_critic` / `ridge_tanh_calibrated` | P116说明两阶段和shared split接口 |
| 原JS对照 | `scripts/cond_test_t1_ablation.py` 中 `exact_js_critic` | P116说明为数值L-BFGS，不是符号精确解 |
| nested estimator | `src/vcs_measure/nested.py` | P118注明复用且该轮未改变 |
| known-truth压缩 | `scripts/p118_common.py`、`scripts/p118_truth.py` | P118列明generator/oracle和runner |
| 真实压缩 | `scripts/p118_visual.py`、`scripts/p118_aggregate.py`、`tests/test_p118.py` | P118列明；实际HEAD参数待核对 |
| 属性审计 | `scripts/p109_i1_audit.py` | P118 §6.2列明原runner |
| ImageNet preflight | `src/vcs_ssl/imagenet.py`、`scripts/p132_imagenet_preflight.py` | P132报告列明；不是已确认训练CLI |

这些是**报告提供的入口**，不是本包已测试过的可执行调用。接收端需要检查current source、函数签名和已有config。无法确认CLI时先执行允许的 `--help` 或静态读取；不要根据文件名发明 `--dataset`、`--epochs` 参数并提交大任务。

## 4. V6未规定或本次未确认的字段

当前HEAD/Slurm状态、checkpoint物理可用性、追加实验批准记录、实际CPU/GPU配额、第二外部encoder身份及其训练谱系、新条件HSIC实现参数、ImageNet完整训练日程和总预算，均未在本次材料准备中确认。

这些不是需要在聊天里预先猜中的字段；intake应能从仓库和服务器读出的先读取，真正无法确定的再放 `DECISIONS_AND_BUDGET_TEMPLATE.md`。未读取的内容不能被写成完成事实。

## 5. 本地检查覆盖范围

`tools/check_contract.py` 仅用小型确定数组检查数学/配对约定；原V6 replay仅检查已有十个有限分布恒等式与舍入表格统计；`verify_package.py` 检查文件hash。`intake_repo.py` 的 r1 临时toy Git接口测试记录保留在 checks_r1_history，本轮不重称已检查实际server checkout。r2另运行表格语义、参考来源与文件完整性核对，见checks。

它们不覆盖：训练图的PyTorch梯度、DDP归一化、真实transform语义、数据泄漏、真实null calibration、runtime效率或科学结论。服务器仍需运行已有相应测试与冻结实验。
