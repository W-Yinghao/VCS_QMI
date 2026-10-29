Publication copy (sample names aliased; numbers and conclusions unchanged).

# R26 COCO 40% / CC152K 结果：真实候选对应分布 + VCS 二阶学习（2026-10-29）

状态：阶段 C 24 单元全部完成。COCO 40%（合成错配，与 Flickr 同算法的锁定噪声映射）与 CC152K（真实噪声，不注入）× {原生 H0, D-VCS, D-BCE, M-VCS} × seed 1001–1003。按设计阶段 C 只跑三臂加原生（N-VCS、D-sNCE 未在此两数据集运行）。H0 是 R26 自己在同一共同前缀上跑的原生 IN2R-code。全部为 full DEV 开发证据，未读 TEST。数字来源 `analysis/CROSS_SUMMARY.json`（`analysis/summarize.py cross`，逐 run 核对 40 轮 co-train 记录与 DONE.json 最佳值一致；|差|≤1e-9 记为平局）。

DEV 规模：COCO full DEV 为 1,000 图 / 5,000 caption（NCR coco_precomp 的 dev 划分；设计中“1K 与 5K 同时报”只适用于 TEST，DEV 只有 1K）；CC152K full DEV 为 1,000 图 / 1,000 caption。

## 1. 最佳 full-DEV rSum（三 seed 均值；括号为末 5 轮均值）

| 数据集 | H0 原生 | D-VCS | D-BCE | M-VCS |
|---|---|---|---|---|
| COCO 40% | 542.89（542.35） | **543.81**（543.31） | 543.13（542.41） | 543.61（543.22） |
| CC152K | 389.63（388.40） | **391.50**（390.27） | 389.37（388.45） | 390.50（389.53） |

R@1 均值：COCO i2t H0 85.57 / D-VCS 86.27 / D-BCE 85.80 / M-VCS 86.13，t2i 68.99 / 69.56 / 69.22 / 69.33；CC152K i2t 44.77 / 44.97 / 44.60 / 44.93，t2i 43.33 / 43.33 / 42.77 / 43.13。

## 2. 配对比较（逐 seed，最佳 rSum；括号为 >0 的 seed 数）

| 比较 | COCO 40% | CC152K |
|---|---|---|
| D-VCS − H0 | +0.92（2/3；+1.72, +1.90, −0.86） | **+1.87（3/3；+2.2, +1.4, +2.0）** |
| D-VCS − D-BCE | +0.69（3/3；+0.32, +1.40, +0.34） | +2.13（2/3；+1.7, +5.1, −0.4） |
| D-VCS − M-VCS | +0.20（2/3） | +1.00（2/3） |
| D-BCE − H0 | +0.23（2/3） | −0.27（2/3） |
| M-VCS − H0 | +0.72（2/3） | +0.87（2/3） |

末 5 轮口径方向一致（D-VCS−H0 COCO +0.96、CC152K +1.87 3/3；D-VCS−D-BCE +0.90 3/3、+1.83 2/3）。

## 3. 结论

1. **与 Flickr 不同，在 COCO 与 CC152K 上 D-VCS 是四者中最高的。** 相对原生 H0：COCO +0.92（2/3），CC152K +1.87（3/3，真实噪声数据集上唯一一个三 seed 全正的比较）。幅度小（约 1–2 rSum，R@1 变化 ≤0.7），且只有三 seed、无图像分组 bootstrap。
2. **VCS 相对同图 BCE**：COCO 3/3 为正（+0.69），CC152K 均值 +2.13 但 2/3（seed1002 的 +5.1 主导，seed1003 −0.4）。这是 R25/R26 以来第一次在两个数据集上 D-VCS 都不低于同接口 BCE；但 Flickr 60% 上两者基本持平（+0.05）、R25 中 BCE 反而更高，所以“VCS 优于 BCE”仍依数据集而变，不能写成稳定的 VCS 特有优势。
3. **完整分布 vs 未归一化均值点**：D-VCS 略高于 M-VCS（+0.20 / +1.00，各 2/3），方向与 Flickr 一致但幅度小。
4. 未比较 N-VCS 与 D-sNCE（阶段 C 设计未包含）；Flickr 上这两个对照都不低于 D-VCS，因此本结果不能推出 D-VCS 优于归一化 prototype 或 soft NCE。
5. 与 Flickr 合看：同一架构在 Flickr 上没有超过原生（60% 全低），在 COCO/CC152K 上小幅超过。是否与数据规模（COCO 训练池更大）、真实噪声结构（CC152K）或跨 peer 坐标初期失配（`r26_graph_diag_20261028/ANALYSIS_CN.txt`）有关，现有数据不能区分。

## 4. 技术核对与局限

- 24 单元均一次完成（ATTEMPTS=1）；`skip_labeled_lt2` 每 run 0–2 次；CC152K 部分 run 的 `skip_small_or_empty` 达 11–21 次（真实噪声下 GMM 划分偶有极小 unlabeled 划分，各臂同规则跳过，已计数）。
- GPU 混合 L40S 与 RTX PRO 6000，臂间硬件不完全配对。外系数 c_J 按各数据集 FIT 一次标定后冻结（SCALE_coco.json、SCALE_cc152k.json），cosine critic。
- CC152K 的 TEST 与 COCO 5K TEST 均未读取；锁定后的 TEST 评估需用户批准。
- 成本元数据补齐（2026-10-29）：原生 H0 由 r25_run.py 运行，不写 PEAK 文件，旧表中 H0 的 wall_h_last_attempt=0.0、gpu=null 是缺失而非零成本。现从 RUN_*.json 取 GPU/节点，从 EPOCHS.json 累加 40 轮 co-train 计时（不含数据拷贝与建模）：COCO H0 6.51 / 12.14 / 6.48 h（RTX / L40S / RTX），CC152K 2.56 / 3.98 / 4.10 h（RTX / L40S / L40S）；表中缺失的整作业墙钟记为 None。
