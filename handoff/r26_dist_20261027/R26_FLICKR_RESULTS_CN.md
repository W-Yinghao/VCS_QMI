Publication copy (sample names aliased; numbers and conclusions unchanged).

# R26 Flickr30K 结果：真实候选对应分布 + VCS 二阶学习（2026-10-29）

状态：Flickr 40%/60% × seed 1001–1003 × 五臂（D-VCS / D-BCE / M-VCS / N-VCS / D-sNCE）30 单元全部完成；原生对照复用 R25 H0（同一 IN2R-code、同一共同前缀、同一噪声锁）。全部为 full DEV（1,014 图 / 5,070 caption）开发证据，未读 TEST。数字来源 `analysis/FLICKR_SUMMARY.json`（脚本 `analysis/summarize.py flickr`，逐 run 核对 40 轮 co-train 记录与 DONE.json 最佳值一致）。COCO 40% / CC152K 仍在运行，另行汇总。

## 1. 最佳 full-DEV rSum（三 seed 均值）

| 条件 | H0 原生 | D-VCS | D-BCE | M-VCS | N-VCS | D-sNCE |
|---|---|---|---|---|---|---|
| 40% | 510.49 | 510.25 | 508.87 | 509.75 | 510.70 | **511.24** |
| 60% | **492.75** | 491.82 | 491.78 | 491.07 | 492.16 | 492.20 |

末 5 轮均值：40% H0 509.19 / D-VCS 509.32 / D-BCE 507.65 / M-VCS 509.36 / N-VCS 509.60 / D-sNCE 510.21；60% 491.46 / 490.49 / 490.01 / 488.63 / 489.05 / 490.44。R@1 各臂差别都在 1 分以内。

## 2. 配对比较（逐 seed，最佳 rSum；括号为 >0 的 seed 数）

| 比较 | 40% | 60% |
|---|---|---|
| D-VCS − H0 | −0.24（2/3） | −0.93（0/3） |
| D-VCS − D-BCE | +1.38（3/3） | +0.05（2/3） |
| D-VCS − M-VCS | +0.50（2/3） | +0.75（3/3） |
| D-VCS − N-VCS | −0.45（0/3） | −0.34（0/3） |
| D-VCS − D-sNCE | −0.99（0/3） | −0.37（2/3） |

末 5 轮口径方向基本一致，例外：60% 上 D-VCS 相对 M-VCS/N-VCS 的末 5 轮优势更大（+1.86 / +1.44，均 3/3），即点目标臂后期退化更多。

## 3. 结论（按包 §7 与调整包的判断顺序）

1. **完整系统相对原生强宿主**：没有臂稳定超过 H0。40% 上 D-sNCE +0.75、N-VCS +0.21（各 2/3），60% 上所有臂都低于 H0（−0.55 至 −1.68）。去掉 GraphRefiner、改用真实候选分布这一架构本身没有带来系统级收益；也低于 R25 中保留 refiner 的 V1（VCS-C1）512.32 / 493.84 与 B1（BCE-C1）513.77 / 494.22（R26 用 cosine critic，二者 critic 不同）。
2. **候选分布与二阶结构**：D-VCS 高于未归一化均值点 M-VCS（40% +0.50、60% +0.75），但低于归一化均值点 N-VCS（两条件 0/3）。保留完整分布相对“未归一化均值”有小幅好处，不优于常见的“归一化 prototype”。
3. **VCS 目标特异性**：同图同 critic 下 D-VCS 胜 D-BCE（40% +1.38，3/3；60% 基本持平），但输给同图 soft NCE（D-sNCE）。不能称 VCS 特有优势。
4. 逐 seed 差值都在 ±3 rSum 内（均值差都在 ±1.7 内），三 seed、无逐 query 排名，因此没有图像分组 bootstrap；按设计不设结果门槛，只报告。

## 4. 技术核对与局限

- 30 单元均一次完成（ATTEMPTS=1；6 个 seed1001 单元在原生缺陷修复后从同一前缀重跑，失败尝试已存档，见 `native_bug_rerun.log`）。每 run 40 轮 co-train，第 0 轮是共同原生前缀（各臂逐位相同的 DEV 500.83 等），其中 peer A 因对方队列尚空没有 noisy 损失（原生行为，各臂一致）；`skip_labeled_lt2` 每 run 0–2 次，`skip_small_or_empty` 0–2 次。
- GPU 混合 L40S 与 RTX PRO 6000（每 run 记录在 PEAK_*.json），臂间硬件不完全配对。
- 外系数 c_J 在共同前缀上按 FIT 梯度量级一次标定后冻结（SCALE_flickr.json），critic 为 cosine，不同臂的有效步长只在标定点匹配。
- DEV 已多轮使用；Flickr TEST 旧轮次已读，锁定后只能作重复 benchmark，需用户批准。
- 固定图机制诊断（同图 VCS/BCE/M 梯度、跨 peer 坐标相容性）作业 1071955 尚未运行完成，结果出来后补充。
