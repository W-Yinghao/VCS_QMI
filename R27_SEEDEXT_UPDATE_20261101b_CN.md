# Flickr 种子扩展完成（2026-11-01 第二十一版）

接在同日第二十版（`R30_UPDATE_20261101b_CN.md`）之后。分析点：Flickr seed1002/1003 种子扩展（R27-D / O / O3 / CS）已授权部分全部完成。只发布数字表与代码，不附解读。全部为 FIT/DEV 开发阶段，未读 TEST。

## 内容

- `handoff/r27_seedext_20261031/SEEDEXT_TABLES.md`、`PUBLIC_SEEDEXT_SUMMARY.json`（`summarize_seedext.py`）：Flickr 40%/60% × seed 1001–1003 的 D1–D4、O1/O2、O3、B-native/C-native 逐 seed 最佳 full-DEV rSum、已完成 seed 上的均值（标注 n）、逐 seed −A1 配对差与为正的 seed 数。seed1001 为原单元，1002/1003 为种子扩展（训练代码除种子门槛外逐字节相同），A1 参照取同格同 seed 的 R27 a1_iface。扩展 36 项中 35 项完成；唯一未完成的 f60_s1003/bn_cross_t2i 从未开跑，按 R30 复盘包（暂停未启动的旧 D/O2/B/C 扩展）暂停，表中记 NA。R27-O / O3 显式使用真对应，为诊断，不进入主结果表。
- 逐 query 排名导出：`handoff/r27_audit_20261029/audit_ranks_seedext.py`（直接导入未改动的 `audit_ranks.py`，单元列表换成种子扩展输出根与 seed1002/1003 格）与 `sbatch/ranks_seedext.sbatch`（作业 1080490，runfill）；`PUBLIC_SEEDEXT_RANKS_CHECK.json`：35 个单元由保存排名复算六项 Recall，21 个与训练记录差 0，其余差 0.0197–0.0986pp（1–5 个 query；排名在 V100S/A40 上重算，训练在 RTX/L40S），全部在容差内。

## 范围说明

至多三 seed；均值在不同 seed 子集上计算时不可直接互比，以逐 seed 配对差为准；不给显著性结论。DEV 在开发中被反复使用；TEST 未读。种子扩展的快照与 runner 含包内文件，不发布。
