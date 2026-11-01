# 检索 TEST 一次性评测（2026-11-01 第二十四版）

接在同日第二十三版（`R30_UPDATE_20261101d_CN.md`）之后。用户 2026-11-01 明确授权并定为常设：每个已完成单元以其 DEV 选出的 checkpoint 在 TEST 上评测一次，与 DEV 并列报告；不在 TEST 上选择 checkpoint、方法或超参。只发布数字表，不附解读。

## 规则与核对

- 读取 TEST 前先冻结 manifest（各单元现有 model_best 的路径与 sha256；SPS S0 用其完整 DEV 最佳 checkpoint_37，原生 model_best 是按 SPS 自带的 200 图 DEV 子集选的），之后每个读取 TEST 的作业记一行访问日志；增量 manifest 只加入新完成单元。
- 原生 IN²R 协议：两 peer 相似度平均；Flickr30K test（1,000 图 / 5,000 caption）；MS-COCO testall，5-fold 1K 均值（文献主协议）与 5K；CC152K test。
- 每个单元先用同一评测器复算 DEV，与训练日志一致才采信（容差 0.1pp/Recall）：249/249 通过（185 个差 0）。
- 文献行取自 IN²R 论文（arXiv:2606.04061v1，Table 1 与 Table 2；已与 OpenReview 版 PDF 逐项核对），标为「论文报告值，非本处复现」。

## 内容（`handoff/test_eval_20261101/`）

- `TEST_TABLES.md`：Flickr30K 0%/40%/60%、MS-COCO 40%（5-fold 1K 与 5K）、CC152K 的各线各臂逐 seed DEV 与 TEST、三 seed 齐全时的均值 ± 标准差（否则 NA）、三 seed 六项 Recall 均值、按 DEV 均值的排序（标注 seed 数）与 seed1001 配对排序、逐单元表；我们的原生 IN²R 复现行（R25 H0：Flickr 40/60% × 3 seed；R26 H0：COCO 40% / CC152K × 3 seed）单独标出；R27-O/O3 为显式使用真对应的诊断，不是方法。跨方法取 TEST 最大值为事后、偏乐观，表中已注明。
- `PUBLIC_LITERATURE_TABLE.json`：文献值；`PUBLIC_ROOTS.json`：评测覆盖的输出根登记；工具 `tools/build_manifest.py`、`test_eval.py`、`aggregate_test.py`、`test_increment.sh` 与 sbatch。宿主评测代码副本（IN²R 作者源码）不发布。

## 局限

DEV 在开发中被反复使用；这是这些 checkpoint 的第一次 TEST 读取（Flickr TEST 曾在 V9 阶段被其他模型读取过一次）。训练所用 GPU 类型不同；40%/60% 噪声映射由我们生成，与论文的噪声实现不一定相同；不给显著性结论。R30 其余单元完成后增量补测。
