# R43–R48、R44 布局审计与更正读数（2026-11-04 第三十九版）

接在第三十八版（`R42_RESULTS_20261103_CN.md`）之后。只发布数字表、代码与审计记录，不附解读。不读 TEST。

## 勘误（适用于第三十六–三十八版及本版 R43/R45 的 JOINT 数字）

- R39 起 JOINT 学生向量为 u = l2norm(e − mu[batch])，mu 是父模型在原生评测顺序布局上每 128 行的均值。该布局中第 r 行为图 r//5 与 caption r，图与其真 caption 同批：DEV 上真对共享中心化向量；FIT 上同批按干净的 k//5 分组。真值因此经中心化向量进入 JOINT 训练与评测（`R44_AUDIT.md`、`R44_AUDIT*.json`，零训练、作业 1085476–1085478）。
- 用布局无关中心化（global：每个模态一个 DEV 均值；decoupled：图与 caption 各自随机分批）重评同一检查点后，JOINT 单元低于同样中心化的父 H（例：Flickr60 R39 N-JOINT J 原生 517.87 → global 489.96，父 H global 496.53；全部单元见 `CORRECTED_READOUTS.md`）。第三十六–三十八版报告的 JOINT−FROZEN 增益因此未确立。
- R34 起冻结 critic 单元使用的父模型特征缓存同样由原生 forward_emb 按顺序批中心化得到；其布局无关读数也在 `CORRECTED_READOUTS.md`。
- 原生 IN²R 评测（作者代码 forward_emb，评测 loader 不打乱）本身对每个顺序批中心化：父模型 student-cosine 读出的原生值比布局无关值高约 9–11 rSum（`R44_AUDIT.md`）。这一点影响所有原生 H 数字，包括 DEV 与 TEST 表；评测协议与作者代码相同。
- 原生读数保留不改；更正读数并列给出。

## R44 布局审计（`handoff/r40_scope_present_20261103/R44_AUDIT.md`）

- 触发：R44 ROUTE_PRESENT 在 Flickr60/80 上原生读数 +50/+85（`RESULTS_R44.md` B/C 节，已标注不可使用）。审计四项：参数集与 R39 ALL 相同；无 DEV 张量进入训练图；独立重评复现原生读数；1/p_d 缩放与裁剪不足以解释。原因为上述布局泄漏，ROUTE_PRESENT 使编码器输出塌缩（随机 DEV 项余弦 0.97–0.99）。

## 更正读数（`CORRECTED_READOUTS.md` / `.csv`）

- R37–R46 已完成的 critic 单元（frozen 与 JOINT）共 418 个；每组先在训练 GPU 类型上复现原生日志读数（差 0.0），再给出 global / decoupled 中心化下的两阶段读数、父 H 同口径读数与 student-cosine 读数。代码 `tools/corrected_*.py`，CC 另建审计树 `r47cca/`。

## R43（`RESULTS_R43.md`）

- 66/66：COCO IN²R-native 父模型 × {N-frozen, N-JOINT} × J/BCE/NCE；TRUST+N × {frozen, JOINT} 在 Flickr 80% 与 CC152K。原生读数；JOINT 数字适用上述勘误。代码 `r43o/`、`r43f/`、`r43c/`。

## R44（`RESULTS_R44.md`）

- 39/39：A 16000 步 N-JOINT 训练 seed 2002 对 1001（Flickr60 三父模型）；B 参数块监督分工（ROUTE_PRESENT / ROUTE_RANDOM）；C PRESENT_ALL。B/C 的 ROUTE_PRESENT 数字来自布局泄漏，不可使用。代码 `r44f/`、`r44c/`。

## R45（`RESULTS_R45.md`）

- 6/6：COCO c60/c80 s1001 N-JOINT 16000 步 × J/BCE/NCE。原生读数；适用勘误。代码 `r45o/`。

## R46（`RESULTS_R46.md`）

- A：COCO 80% FIT 池 29k/58k 子集 × {N-frozen, N-JOINT}；B：迭代 N（REBUILT/CONTINUED）。R44 审计后用户取消 JOINT 作业，只列已完成的 13/30 单元（6 个 frozen 全部完成），其余 NA。代码 `r46f/`、`r46o/`；零训练覆盖 `PUBLIC_R46_COVERAGE.json`。

## R47：布局无关中心化重训 JOINT（`RESULTS_R47.md`）

- 30 项：N-JOINT × J/BCE/NCE × {Flickr60 R30-H s1001、Flickr80 IN²R f80 s1001、Flickr0 R30-H f00 s1001、COCO60 IN²R c60 s1001、CC152K R31-H0 s1001} × {G：学生与父缓存用每 split/模态/peer 一个父均值；S：学生用其训练批唯一项均值，父缓存 global}，R39 JOINT 配方 8000 步，DEV 用同口径布局无关读数，H 为各单元 step-0 读数。参照为 `CORRECTED_READOUTS.csv` 中同父 N-frozen 与原生中心化 N-JOINT 的 global 读数。
- 门禁：CPU `PUBLIC_CPU_TEST_R47.json`（一次测试脚本失败记录 `PUBLIC_CPU_TEST_R47_1085587_failed.json`）、GPU smoke `PUBLIC_GPU_SMOKE_R47.json`（step 0 复现布局无关父 H，0 名次差）；冻结 `PUBLIC_PROTOCOL_FROZEN_R47.json`。代码 `r47f/`、`r47o/`、`r47c/`，差异 `R47_DIFF_*.patch`。

## R48：布局无关缓存上重训 N-frozen（`RESULTS_R48.md`）

- 15 项：N-frozen（共享 G 头、cosine 日程、父编码器冻结）× J/BCE/NCE × 与 R47 相同的 5 个父模型，在 R47 G 布局无关父缓存上训练与读出，8000 步；与 R47 G N-JOINT 同初值、同采样流、同缓存与读出。表中另列同父原生中心化 N-frozen 的布局无关重评（`CORRECTED_READOUTS.csv` full_global）。
- 门禁：CPU `PUBLIC_CPU_TEST_R48.json`、GPU smoke `PUBLIC_GPU_SMOKE_R48.json`（step 0 复现布局无关父 H）；冻结 `PUBLIC_PROTOCOL_FROZEN_R48.json`。代码 `r48f/`、`r48o/`、`r48c/`（R47 树字节副本，仅 common.py 末尾改输出目录与冻结文件），配置 `tools/make_configs_r48.py`。

## 其他

- TEST 汇总表与实验清单更新（R33 25/27）。
- 进行中、本版只含代码与配置矩阵：R49（R48 的布局无关 N-frozen 扩到其余 19 个父模型，57 项；`tools/make_configs_r49.py`、`RESOLVED_MATRIX_R49.csv`）；布局无关 TEST 读数（用户授权再读一次 TEST，结果待下一版）。
- 更正读数 418 个单元中 415 个已完成，其余 3 个（R42 Flickr40 s1002 JOINT，需在其训练 GPU 类型上复现）标 PENDING。

## 范围说明

单训练 seed 为主；配对差不给显著性结论；Flickr、COCO 与 CC152K 分开报告。DEV 在开发中被反复使用。设计包与包内参考代码不发布。
