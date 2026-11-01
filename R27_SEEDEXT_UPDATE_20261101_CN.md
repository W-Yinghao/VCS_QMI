# Flickr 种子扩展多 seed 中期表（2026-11-01 第十八版）

接在同日第十七版（`R29_UPDATE_20261101b_CN.md`）之后，按「所有有的结果都push」更新中期结果。只发布数字表，不附解读。全部为 FIT/DEV 开发阶段，未读 TEST。

## 内容

- `handoff/r27_seedext_20261031/SEEDEXT_TABLES.md`、`PUBLIC_SEEDEXT_SUMMARY.json`（`summarize_seedext.py`）：Flickr 40%/60% × seed 1001–1003 的 R27-D（D1–D4）/ O（O1/O2）/ O3 / CS（B-native、C-native）多 seed 表；seed1001 为原单元，1002/1003 为种子扩展（训练代码除种子门槛外逐字节相同），A1 参照取同格同 seed 的 R27 a1_iface。逐 seed 最佳 rSum、已完成 seed 上的均值（标注 n）与逐 seed −A1 配对差及为正的 seed 数。种子扩展 36 项中 25 项已完成，未完成记 NA。R27-O / O3 显式使用真对应，为诊断，不进入主结果表。
- 运行状态：种子扩展其余 11 项运行或排队；全部完成后再发布完整表与逐 query 排名。

## 范围说明

至多三 seed；均值在不同 seed 子集上计算时不可直接互比（以标注的 n 与逐 seed 配对差为准），不给显著性结论。DEV 在开发中被反复使用；TEST 未读。种子扩展的快照与 runner 含包内文件，不发布。
