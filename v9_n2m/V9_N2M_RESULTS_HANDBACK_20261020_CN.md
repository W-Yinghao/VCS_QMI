# V9 N2M 结果交回（2026-10-20）

按 `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md`（与 `/home/infres/yinwang/CS_QMI/V9_N2M_SERVER_HANDOFF_20261019_CN.md` 逐字节相同）§6、§7 交回。本文件由 `v9_n2m/handback/n2m_handback.py`（sha256 `0005b4ede20844ea7c964421bf35d7114887689b13c98f268a2b86c3e45348f0`）只读server记录生成，所有数字由该脚本从下列文件计算，引文逐字取自所注文件与行号；生成与核对不提交作业、不训练、不读TEST、不加载模型。冻结实验 `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/experiment.json` sha256 `900dec055d5ce5efc4dc3118bde78b4517ec9e966a1a9c2274c50d125ff0380f`（与 receipt `EXPERIMENT.sha256` 一致），协议 `config/protocol.json` sha256 `de23173fde6f7dc754001114b9c5db05dfa6dc6cd8cf623a17db195833207b12`。

## 0 摘要

- 完整性：9/9 个单元 COMPLETE_VERIFIED（3 seed × VCS/Constant-U/Shuffle-VCS，均为本轮新跑，每个 40 epoch、45320 次更新，每个 seed 的三臂同一作业同一 GPU）。
- best full DEV rSum（VCS / Constant-U / Shuffle-VCS）：s901 467.00 / 443.61 / 434.48；s902 467.00 / 445.68 / 444.50；s903 470.97 / 436.75 / 443.49。
- VCS−Constant-U 均差 +26.31（+23.39, +21.32, +34.22，3/3 为正）；VCS−Shuffle-VCS 均差 +27.50（+32.52, +22.50, +27.48，3/3 为正）；严重塌陷 0/9。
- 工程投入规则：True。这只是资源决定（继续内容关联可靠度扩展），不是显著性、等价或确认；n=3、DEV 复用，仅描述性。
- 机制：保持总质量的 Constant-U、保持总质量与权重分布/ESS 的 Shuffle-VCS 在 3/3 个 seed 都低于 VCS（差 21.32–34.22 与 22.50–32.52）；后验评分中只有 VCS 的分配区分注入错配（AUROC 0.885–0.911，两种控制 ≤ 0.551），固定风险方向一致。错误分配是否比恒定权重更差、再平衡本身相对不加权的作用、相对其他估计器的独有优势、显著性与泛化均未被支持或未检验（§3.5）。
- 成本：已知 GPU allocation 56.81 GPU-h（scontrol 快照重建，sacct 不可用）；CUDA 核对作业 1048422 的 allocation 未知，不计为 0。

## 1 主结果：三seed三臂 best full DEV rSum 与两项配对差

> 先展示三seed三臂best full DEV rSum和两项配对差，再结合posterior/总质量/固定风险/轨迹解释。三对都完整后才报主均差；缺项和失败如实列出。n=3只做描述性配对比较，不给pair-IID置信区间、显著性或等价结论。last/末5轮只诊断，不替换best主终点。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md:134`

| seed | VCS | Constant-U | Shuffle-VCS | VCS−Constant | VCS−Shuffle |
|---:|---:|---:|---:|---:|---:|
| 901 | 467.00 (e37) | 443.61 (e39) | 434.48 (e35) | +23.39 | +32.52 |
| 902 | 467.00 (e35) | 445.68 (e38) | 444.50 (e37) | +21.32 | +22.50 |
| 903 | 470.97 (e39) | 436.75 (e30) | 443.49 (e36) | +34.22 | +27.48 |

括号内为 best epoch；差值为同 seed 的 VCS 减控制。

| 比较 | 完整配对 | 三seed均差 | 最小差 | 最大差 | 正方向seed | 零差 | p值 / 置信区间 |
|---|---:|---:|---:|---:|---:|---:|---|
| VCS-ConstantU | 3/3 | +26.31 | +21.32 | +34.22 | 3/3 | 0 | null（未计算） |
| VCS-ShuffleVCS | 3/3 | +27.50 | +22.50 | +32.52 | 3/3 | 0 | null（未计算） |

三对均完整（9/9 个单元 COMPLETE_VERIFIED），因此报告主均差。summary.json 中两项比较的 `p_value` 与 `confidence_interval` 均为 null；n=3 只作描述性配对比较，本文件不给 pair-IID 置信区间、显著性或等价结论，也不把它当作独立确认（DEV 已在本路线中多次使用）。

**VCS s901 与 s902 的 best rSum 相同（都是 467.0019723865878），这是巧合。** DEV 有 1014 张图像、5070 条 caption：i2t 每项 Recall 为 100·k/1014 = 100·5k/5070，t2i 为 100·m/5070，所以 rSum 只能取 100/5070 的整数倍；两个值都等于 23677×100/5070。两次运行在其他所有字段上都不同：seed、噪声映射与权重文件不同（权重 sha256 `a61fca0f5305…` 与 `e6ad7c8d607e…`），epoch 0 平均损失 17.0118 与 15.6472，best epoch 37 与 35，六项 Recall（i2t R1/R5/R10、t2i R1/R5/R10）69.03/90.73/95.17/51.64/76.47/83.96 与 71.40/91.81/95.46/51.85/75.23/81.24。

同 checkpoint 的六项 Recall 与 mR（best epoch）：

| seed | 臂 | best epoch | i2t R1 | i2t R5 | i2t R10 | t2i R1 | t2i R5 | t2i R10 | mR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s901 | VCS | 37 | 69.03 | 90.73 | 95.17 | 51.64 | 76.47 | 83.96 | 77.83 |
| s901 | Constant-U | 39 | 69.43 | 89.35 | 93.39 | 49.41 | 69.37 | 72.66 | 73.93 |
| s901 | Shuffle-VCS | 35 | 64.69 | 88.86 | 92.90 | 48.19 | 68.40 | 71.44 | 72.41 |
| s902 | VCS | 35 | 71.40 | 91.81 | 95.46 | 51.85 | 75.23 | 81.24 | 77.83 |
| s902 | Constant-U | 38 | 68.15 | 90.34 | 94.67 | 48.62 | 70.73 | 73.18 | 74.28 |
| s902 | Shuffle-VCS | 37 | 68.44 | 89.45 | 93.59 | 48.70 | 70.47 | 73.85 | 74.08 |
| s903 | VCS | 39 | 72.29 | 91.42 | 95.56 | 51.91 | 76.29 | 83.49 | 78.49 |
| s903 | Constant-U | 30 | 66.86 | 87.97 | 92.60 | 47.06 | 68.95 | 73.29 | 72.79 |
| s903 | Shuffle-VCS | 36 | 67.16 | 89.45 | 93.39 | 47.16 | 71.18 | 75.15 | 73.92 |

## 2 完整性

- 矩阵：9 个单元 = 3 seed × 3 臂，`requested_mode=auto`，新完整训练 9 次。三个 VCS 参照都判为新跑（`VCS_action=RUN`）：历史 N2F VCS 的环境与本轮目标不一致或未记录（GPU 期望 NVIDIA L40S、实际 NVIDIA RTX PRO 6000 Blackwell Server Edition；另有 10 项历史环境字段未记录：CUBLAS_WORKSPACE_CONFIG, cudnn_benchmark, cudnn_deterministic, cudnn_tf32, cudnn_version, driver, float32_matmul_precision, num_threads, numpy, torch_build_sha256），因此全部为同环境的新运行，没有历史复用。
- 状态：COMPLETE_VERIFIED 9/9；每个单元 40 个完整 epoch、每 epoch 1133 次更新、共 45320 次（与 checkpoint 的 global_step 一致），恰好一次 attempt（SUCCEEDED），没有续跑，`errors` 与 `missing` 均为空。
- 产物：summary.json 对每个单元记录 train_log.json、run_manifest.json、identity.json、best.pt、last.pt 均为 VERIFIED。本脚本重新核对了 JSON 产物的 sha256、identity 与 experiment/协议/权重身份、best 的选择（最大值、并列取更早）、六项 Recall 与 rSum/mR 的一致性、progress.jsonl 与训练日志逐 epoch 一致，以及 summary.json、unit_results.csv、paired_contrasts.csv、epoch_trajectories.csv 与训练日志重算值一致。模型二进制见 §6（checkpoints_not_materialized）。
- 配对条件：每个 seed 的三臂在同一个作业、同一张 GPU 上按冻结顺序串行（s901：job 1048493，node58；s902：job 1048494，node61；s903：job 1048495，node59）。9 个单元的环境记录除 host/job/时间外与冻结目标环境逐字段相同：NVIDIA RTX PRO 6000 Blackwell Server Edition、driver 615.71.09、Torch 2.9.0+cu128、CUDA 12.8、Python 3.12.12、NumPy 2.5.2、cuDNN 91002、deterministic_algorithms=true、TF32 关闭、cuDNN benchmark 关闭。
- CUDA 核对：`SERVER_VERIFICATION.json` status=PASS（job 1048422，node58），`TORCH_VERIFICATION_cuda.json` status=PASS，scope：synthetic actual-SGR code/resume equivalence; no scientific performance evidence。
- 生产权重未重拟合：三个 seed 的 VCS 臂 CSV、其 lock 与 observed 表都与 N2F `FROZEN.json` 登记的冻结输入 sha256 相同（VCS 臂逐字节复制原 CSV）。
- 数据角色：experiment、summary 与 9 个 identity 的 test_read 均为 false；三个 controls manifest 的 `label_fields_read` 与 `random_permutation_selected_using_labels` 均为 false；离线评分的 `training_label_access` 为 false、`evaluator_label_access` 为 true，即机制标签只在 EVAL_ONLY 评分中读取；固定风险诊断的 `labels_used_only_by_clean_oracle` 为 true。

> 固定pair机制标签不进入critic、校准选择或控制生成。teacher与生产critic不重拟合。本轮最多9次完整训练，不增加MEAN/LR/rho/anchor/COCO/REG/Unweighted新网格。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md:138`

## 3 机制诊断与可支持的结论

### 3.1 总质量、权重分布与 ESS（控制的构造）

| seed | μ = mean(w_U) | U总质量（三臂相同） | 全池分母 v̄ | A占保留质量 | ESS_U：VCS / Constant / Shuffle | Shuffle固定点 | 置换前后权重 r |
|---|---:|---:|---:|---:|---:|---:|---:|
| s901 | 0.611869 | 79848.89 | 0.650682 | 0.1537 | 105233 / 130500 / 105233 | 2 | 0.0571 |
| s902 | 0.606837 | 79192.26 | 0.646153 | 0.1548 | 100854 / 130500 / 100854 | 4 | 0.0582 |
| s903 | 0.623328 | 81344.34 | 0.660995 | 0.1513 | 101416 / 130500 / 101416 | 2 | 0.0620 |

三臂的 U 总质量、全池分母与 A 的质量占比在 float64 舍入内相同（manifest 不变量全部为 true）；Shuffle-VCS 的权重多重集、总质量与 ESS 与 VCS 相同，只改变权重与图像内容的对应。

### 3.2 后验评分（EVAL_ONLY 机制标签）

> 标签含义：clean = 未被注入机制实际错配；不等同于语义兼容性真值。
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/offline/s901/OFFLINE_READOUT_CN.md:3`

| seed | 臂 | Brier | logloss | AUROC (clean=1) | AP | ECE (10箱) | clean平均保留 | mismatch平均保留 | 残余噪声质量占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| s901 | VCS | 0.1359 | 0.4298 | 0.8851 | 0.9118 | 0.0395 | 0.7736 | 0.3693 | 0.2414 |
| s901 | Constant-U | 0.2401 | 0.6733 | 0.5000 | 0.6000 | 0.0119 | 0.6119 | 0.6119 | 0.4000 |
| s901 | Shuffle-VCS | 0.3049 | 0.8860 | 0.5498 | 0.6342 | 0.2406 | 0.6328 | 0.5805 | 0.3795 |
| s902 | VCS | 0.1222 | 0.3923 | 0.9038 | 0.9254 | 0.0363 | 0.7952 | 0.3242 | 0.2137 |
| s902 | Constant-U | 0.2400 | 0.6731 | 0.5000 | 0.6000 | 0.0068 | 0.6068 | 0.6068 | 0.4000 |
| s902 | Shuffle-VCS | 0.3205 | 0.9509 | 0.5493 | 0.6338 | 0.2676 | 0.6300 | 0.5721 | 0.3771 |
| s903 | VCS | 0.1165 | 0.3768 | 0.9112 | 0.9316 | 0.0283 | 0.8195 | 0.3290 | 0.2111 |
| s903 | Constant-U | 0.2405 | 0.6742 | 0.5000 | 0.6000 | 0.0233 | 0.6233 | 0.6233 | 0.4000 |
| s903 | Shuffle-VCS | 0.3229 | 0.9774 | 0.5511 | 0.6346 | 0.2718 | 0.6475 | 0.5870 | 0.3767 |

估计 rho（原始）/实际机制比例：s901 0.3922/0.4000；s902 0.4052/0.4000；s903 0.3708/0.4000。同一 T 换成实际比例的 oracle-rho 诊断（不可部署）：s901 Brier 0.1359、残余噪声 0.2395；s902 Brier 0.1222、残余噪声 0.2149；s903 Brier 0.1162、残余噪声 0.2042。生产权重由锁定 CSV 重建，三个 seed 的 `production_reconstruction.consistent` 均为 true。

### 3.3 固定风险（同一学生、同一 batch、同一负候选）

| seed | batch数 | VCS | Constant-U | Shuffle-VCS | Clean oracle | VCS<Constant | VCS<Shuffle | Clean<VCS | Shuffle<Constant | 固定学生 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s901 | 8 | 0.1688 | 0.2229 | 0.2104 | 0.1079 | 8/8 | 8/8 | 8/8 | 7/8 | 901_1 e39 |
| s902 | 8 | 0.1479 | 0.2205 | 0.2144 | 0.0813 | 8/8 | 8/8 | 8/8 | 5/8 | 902_1 e39 |
| s903 | 8 | 0.1626 | 0.2409 | 0.2221 | 0.1013 | 8/8 | 8/8 | 8/8 | 8/8 | 903_1 e38 |

表中为 `sum(v·ell)/v̄` 除以 batch 大小的均值（越低表示加权后的 hinge 目标越小）。固定学生是原 N2F VCS SUM 的 best checkpoint，其 sha256 与 N2F 结果的 artifact SHA 一致（已核对）。

> Same student, per-anchor losses, batches and negative candidates as declared by the exporter. Reweighting risk diagnostic does not identify the independently trained endpoint or a clean-negative-pool risk.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/fixed_loss/s901/fixed_loss_metrics.json` 字段 `interpretation`

> The scorer reads exported losses without loading the model. Hash binding does not independently prove model execution; no full-training completion or retrained-endpoint claim is made.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/fixed_loss/s901/fixed_loss_metrics.json` 字段 `warnings[0]`

### 3.4 轨迹、梯度窗口与严重塌陷

> 同时报告六Recall、mR、best epoch、epoch4/5、最差hard值、last、最后5轮均值、梯度/clip窗口、严重塌陷与成本。末值不能替换已经固定的主终点。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/PROTOCOL_CN.md:100`

| seed | 臂 | best (epoch) | epoch4 | epoch5 | 最差hard (epoch) | last (e39) | 末5轮均值 | clip比例 | 裁剪前梯度范数 均值/最大 | 窗口样本 | 严重塌陷 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| s901 | VCS | 467.00 (e37) | 371.44 | 400.97 | 400.97 (e5) | 466.29 | 466.19 | 1.00 | 184.0 / 1012.7 | 400 | 否 |
| s901 | Constant-U | 443.61 (e39) | 347.42 | 353.57 | 353.57 (e5) | 443.61 | 430.43 | 1.00 | 158.3 / 1092.8 | 400 | 否 |
| s901 | Shuffle-VCS | 434.48 (e35) | 343.96 | 365.96 | 346.45 (e11) | 416.94 | 428.65 | 1.00 | 170.1 / 1011.1 | 400 | 否 |
| s902 | VCS | 467.00 (e35) | 377.34 | 412.17 | 412.17 (e5) | 458.36 | 459.65 | 1.00 | 169.5 / 1055.8 | 400 | 否 |
| s902 | Constant-U | 445.68 (e38) | 341.36 | 352.21 | 352.21 (e5) | 443.00 | 441.14 | 1.00 | 114.6 / 909.8 | 400 | 否 |
| s902 | Shuffle-VCS | 444.50 (e37) | 329.70 | 352.37 | 352.37 (e5) | 410.83 | 432.73 | 1.00 | 98.5 / 708.2 | 400 | 否 |
| s903 | VCS | 470.97 (e39) | 367.16 | 411.10 | 408.70 (e6) | 470.97 | 470.53 | 1.00 | 169.6 / 1040.7 | 400 | 否 |
| s903 | Constant-U | 436.75 (e30) | 340.93 | 358.50 | 358.50 (e5) | 420.28 | 428.94 | 1.00 | 149.6 / 867.9 | 400 | 否 |
| s903 | Shuffle-VCS | 443.49 (e36) | 335.35 | 338.93 | 282.15 (e7) | 418.15 | 434.80 | 1.00 | 122.2 / 819.7 | 400 | 否 |

- 严重塌陷 0/9。更强的事实：没有任何 hard epoch 的 DEV 低于本臂 epoch4 的50%（hard 阶段最差值与 epoch4 之比最低为 0.841，Shuffle-VCS s903 epoch 7），故严重塌陷定义的第一个条件从未满足。
- VCS 在 warm-up 结束时（epoch 4）已领先：相对 Constant-U +24.02, +35.98, +26.23，相对 Shuffle-VCS +27.48, +47.63, +31.81（seed 901/902/903）。协议规定权重从第 0 次更新起应用、前 5 个 epoch 为 warm-up，所以这一差距在 hard-negative 阶段开始前就已出现。
- 梯度窗口只覆盖 epoch 4 末 200 次与 epoch 5 初 200 次更新（每臂 400 个样本），9 个单元的 clip 比例均为 1.00：窗口内每一步都被裁剪到范数 2，因此该窗口内的更新步长由 clip 决定；它不是全程梯度估计。
- 仅作诊断（不替换 best 主终点）：按 last（e39），VCS−Constant 为 +22.68, +15.36, +50.69，VCS−Shuffle 为 +49.35, +47.53, +52.82；按末5轮均值，VCS−Constant 为 +35.76, +18.51, +41.59，VCS−Shuffle 为 +37.54, +26.92, +35.73。

### 3.5 可支持的机制结论与不支持的方向

**可支持（描述性，n=3，限于本设计与 DEV）：**

1. **不是 A/U 总质量再平衡。** Constant-U 与 VCS 的 U 总质量、全池分母和 A 质量占比相同，但 best rSum 在 3/3 个 seed 低 21.32–34.22。因此 VCS 相对本组控制的终点差不能由“A 的质量占比从 10% 提高到 15.1%–15.5%”这一再平衡解释。
2. **不是权重分布或 ESS 本身。** Shuffle-VCS 保留 VCS 的权重多重集、总质量与 ESS，只打断权重与图像内容的对应，best rSum 在 3/3 个 seed 低 22.50–32.52。
3. **终点差与权重中关于注入错配的逐 pair 信息同时出现。** 只有 VCS 的分配区分了 clean 与 mismatch：AUROC 0.885–0.911（Constant-U 0.500，Shuffle-VCS 0.549–0.551），残余噪声质量占比 0.211–0.241（Constant-U 0.400，Shuffle-VCS 0.377–0.379）。这符合协议预定的解释分支：

> | VCS 同时优于两个控制 | 内容相关分配有用，再看校准或更有信息的输入 |
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/PROTOCOL_CN.md:109`

> 这些是预定的解释分支，不是对未执行结果的预测。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/PROTOCOL_CN.md:114`

4. **固定风险方向一致。** 在同一个固定学生和同一批负候选上，VCS 重加权后的目标在 24/24 个 batch 低于 Constant-U、24/24 个低于 Shuffle-VCS，clean oracle 在 24/24 个 batch 更低：VCS 把质量从高损失 anchor 移开，方向与 clean oracle 相同。它只说明同一模型上的重加权风险。
5. **训练过程。** 9 个单元都没有严重塌陷，hard 阶段也从未跌到 epoch4 的一半以下；VCS 的领先在 epoch 4 已经出现，并保持到 best、last 与末5轮均值（以上均为诊断）。

**不支持或本轮未检验（如实交回）：**

1. **“A/U 再平衡是 N2F 带权收益的主要来源”：不支持**（见上第1点）。本轮没有 Unweighted 臂，再平衡本身相对不加权是否有益不在预定比较内；原 N2F 的 Unweighted 只能作上下文（下表），不能与本轮单元拼成新的预定比较。
2. **“错误分配比恒定权重更差”：不成立为结论。** Shuffle-VCS − Constant-U 的 best 差为 -9.13, -1.18, +6.75（seed 901/902/903），只有 1/3 个 seed 为正，符号不一致；这也不是预定比较。固定风险中 Shuffle 的目标低于 Constant 的 batch 数为 7/8、5/8、8/8（seed 901/902/903），与完整训练终点的排序并不对应。
3. **统计显著性、置信区间、等价性或独立确认：不提供。** n=3 个配对 seed，DEV 已多次使用；工程规则通过不是检验结果。
4. **VCS 相对 BCE/RankMass/RuLSIF 等其他估计器的独有优势：本轮未检验。** N2M 只比较锁定 VCS 权重的分配与两种控制；N2F 收尾中 VCS 专有门槛未通过的结论不变。
5. **校准与终点的剂量关系：无法估计。** 三个 seed 的 VCS Brier 只在 0.1165–0.1359 之间，本轮只有“有信息 vs 无信息”的对比；Constant-U 的 ECE（0.0068–0.0233）低于 VCS（0.0283–0.0395）而终点最低，所以单看 ECE 不能解释本轮的终点差。
6. **固定风险不是重训终点，也不是 clean 负样本池风险**；评分器未加载模型，只做哈希绑定（见上方引文）。
7. **泛化：未检验。** 没有读 TEST，也没有 COCO、其他 rho、anchor 预算、MEAN 尺度或其他学习率；机制标签只表示注入映射，不是语义真值。
8. **跨硬件等价：不主张。** 本轮 VCS 在 RTX PRO 6000 上重跑，与 N2F 的 VCS SUM 相差 -1.32, -0.36, -2.96（N2F 硬件见下表），只说明同配方在不同硬件/作业上的描述性差异。

原 N2F 上下文（只作上下文，来自 N2F 收尾 CLOSEOUT.json；不是本轮预定比较）：

| seed | N2F VCS/SUM | N2F Unweighted_A/SUM | N2M VCS | N2M Constant-U | N2M Shuffle-VCS |
|---|---:|---:|---:|---:|---:|
| s901 | 468.32（L40S node50） | 445.90（RTX PRO 6000 node61） | 467.00 | 443.61 | 434.48 |
| s902 | 467.36（L40S node50） | 432.52（RTX PRO 6000 node58） | 467.00 | 445.68 | 444.50 |
| s903 | 473.93（L40S node52） | 446.67（RTX PRO 6000 node59） | 470.97 | 436.75 | 443.49 |

> 原 Unweighted、RankMass、BCE、RuLSIF、OracleMask/Drop 只在原 N2F 完整表中作为上下文。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/PROTOCOL_CN.md:14`

## 4 工程投入规则：结果与含义

> 工程投入规则已经固定：两项VCS主差均值均≥3rSum、各至少2/3seed正向、严重塌陷不更多，才继续扩展内容关联可靠度。规则未过不说明等价；可转向解释A/U再平衡。若只优于Constant而没有优于Shuffle，仍需考虑分布/ESS/梯度效应。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md:136`

> **工程继续规则**：两项 VCS 主差均值都至少+3rSum，各至少2/3 seed为正，严重塌陷不更多，才继续投入“内容关联的可靠度”扩展。这仅为资源决定，不是显著性检验；未通过也不证明等价。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/PROTOCOL_CN.md:104`

| 比较 | 三seed均差 | 要求 | 正方向 | 要求 | 严重塌陷 VCS vs 控制 | 满足 |
|---|---:|---|---:|---|---|---|
| VCS-ConstantU | +26.31 | ≥ +3 | 3/3 | ≥ 2/3 | 0 vs 0 | 是 |
| VCS-ShuffleVCS | +27.50 | ≥ +3 | 3/3 | ≥ 2/3 | 0 vs 0 | 是 |

规则结果：True（summary.json `investment_rule.pass`，参数取自冻结协议，本脚本复算一致）。按协议，它的含义只有：

- **是资源决定**：满足“继续投入内容关联可靠度扩展”的预设条件；按预定解释分支，下一步看校准或更有信息的输入。
- **不是检验**：不表示显著性、置信区间或独立确认，也不排除任何 run；n=3 与 DEV 复用的限制不变。
- **规则未过的两个分支不适用**：没有出现“未过→转向解释 A/U 再平衡”的情形；VCS 同时优于 Constant 与 Shuffle，所以“只优于 Constant 时仍需考虑分布/ESS/梯度效应”的分支也不适用。
- **它不改变 N2F 的已有结论**：N2F 收尾记录的 VCS 专有门槛（相对 BCE、RankMass、RuLSIF）为 false；本轮问题不同，不据此改写。

> This is a development mechanism experiment on reused DEV, not an independent confirmation set.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/full/summary.json` 字段 `warnings[]`

> Unfinished/invalid units do not contribute their instantaneous best scores to primary endpoints.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/full/summary.json` 字段 `warnings[]`

> Historical reused and new VCS references retain row-level provenance. Inspect matched environment approval recorded by preparation.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/full/summary.json` 字段 `warnings[]`

> Metadata-only copied checkpoints do not qualify as verified complete primary units.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/full/summary.json` 字段 `warnings[]`

> Cost is reported by attempts where available; missing or lost segments remain unknown.
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/full/summary.json` 字段 `warnings[]`

## 5 成本

> sacct / slurmdbd were unreachable for the whole N2M round ("Connection refused" on localhost:6819, rechecked 2026-10-20 09:30 UTC).
>
> —— `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/analysis/cost/SOURCE_NOTE.md:3`

| job | 名称 | 内容 | 状态 | 退出码 | GPU | allocation GPU-h（重建） | 各臂进程时长合计 h |
|---|---|---|---|---|---:|---:|---:|
| 1048493 | n2m_s901 | seed 901 三臂训练（node58） | COMPLETED | 0:0 | 1 | 18.764 | 18.76 |
| 1048494 | n2m_s902 | seed 902 三臂训练（node61） | COMPLETED | 0:0 | 1 | 18.926 | 18.92 |
| 1048495 | n2m_s903 | seed 903 三臂训练（node59） | COMPLETED | 0:0 | 1 | 18.966 | 18.96 |
| 1048496 | n2m_loss901 | 固定损失导出（node59） | COMPLETED | 0:0 | 1 | 0.073 | — |
| 1048497 | n2m_loss902 | 固定损失导出（node59） | COMPLETED | 0:0 | 1 | 0.063 | — |
| 1048498 | n2m_loss903 | 固定损失导出（node59） | COMPLETED | 0:0 | 1 | 0.018 | — |

- 已知 GPU allocation 合计 **56.81 GPU-h**（其中三个 seed 训练作业 56.66）；`complete_terminal_accounting=false`。
- **CUDA 核对作业 1048422 的 allocation 时长未知，不计为 0**：该作业没有 scontrol 快照，sacct 也不可用。其核对进程自记录 2026-10-19T11:08:55.595724+00:00 至 2026-10-19T11:09:05.752695+00:00，这只是进程时间，不是 allocation。
- 9 次训练的 attempt 进程时长合计 56.65 h（每臂 6.14–6.49 h）；同一 seed 的三臂共享一次 allocation，allocation 只计一次。没有历史复用成本。
- CPU 作业（不计 GPU）：有 scontrol 快照的为 1048432 n2m_prepare COMPLETED RunTime 00:00:25；1048435 n2m_ctrldry COMPLETED RunTime 00:00:12；1048499 n2m_score COMPLETED RunTime 00:00:14；1049085 n2m_fixscore CANCELLED RunTime 00:00:00（Reason=Dependency，未运行）；只有 stdout 日志、没有快照（时长未知）的为 1048421 v9n2m_cpucheck、1048423 v9n2m_drvprobe、1048424 v9n2m_drvprobe、1048425 v9n2m_drvprobe、1048433 n2m_ctrldry、1050484 n2m_fixscore。固定风险评分由 1050484 完成（其日志记录三个 seed 均 rc=0，end=2026-10-19T23:08:02Z）。

> 成本按当前观测粗估：9次RTX完整训练约57 allocation内小时，6次L40S控制约68小时，均为既有中位数外推而非承诺。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md:91`

## 6 §7 交回文件清单

> 回传新包实际代码commit及package SHA、原N2F最终33项表、`experiment.json`/receipt/`UNIT_PLAN.csv`、CUDA实际报告、controls manifest与置换SHA、每臂identity/run_manifest/full_result、完整train_log和progress、grad窗口、全部attempt和sacct记录、三个离线后验表、三个固定风险诊断、最终summary/两个配对差表。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md:142`

> 大图像特征、caption原文和模型二进制可以留server。若只导出元数据，明确写 `checkpoints_not_materialized`；独立汇总不得因此把未核验的checkpoint写成完整主比较。主比较的最终核验在仍持有二进制的server运行，报告保留其artifact SHA。
>
> —— `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md:144`

代码与包：包原样提交于 `8fe6f8a92f8685a2f4fee8f0b8acfb74f373089f`（之后未改动，本脚本以 git 核对）；服务器侧脚本最后提交于 `4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab`；包 zip 的 sha256 `cb1da9b4e7d524560327fb8082104131c34d9ebfcfa873d6c880eddda4e21702`；包内 44 个文件与 `PACKAGE_MANIFEST.json` 逐一一致；server 核对记录的 package_code_digest `33ada1faeaf640fa194f1575d49ae1ed781e5864c35be63615a6de4bfb6c421f`，effective runtime digest `7cfe4f867c3607a05f78527ea5635a6112dcebad771a26cc868af81b0f6fc2c0`。

**checkpoints_not_materialized**：本交回只含元数据，21 个模型二进制（18 个 N2M best/last 与 3 个 N2F 参照 checkpoint，共 4.57 GB）留在 server。下表中它们的 sha256 是记录的 artifact SHA；主比较的二进制核验已在仍持有二进制的 server 上由 summarize 完成（summary.json 中 45 项产物为 VERIFIED，含每个单元的 best.pt 与 last.pt），需要时在 server 上运行 `python3 v9_n2m/handback/n2m_handback.py --binaries` 重算。

共 220 个文件；机器可读清单 `v9_n2m/handback/V9_N2M_HANDBACK_FILES_20261020.json`（sha256 `334c6dd4103bb486d7ff6bb54990ec428f8bd4ae395990fb7ad19e8373e272eb`）。位置：repo = 工作树 `/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m`（分支 v9/n2m-20261019），server = 本服务器路径（OUT = `/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1`）。

### 1 新包实际代码commit与package SHA

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `/home/infres/yinwang/CS_QMI/VCS_V9_N2M_Server_Package_v1_20261019.zip` | `cb1da9b4e7d524560327fb8082104131c34d9ebfcfa873d6c880eddda4e21702` | 154569 | server | zip；sha256 与 v9_n2m/PACKAGE_ZIP.sha256 一致；包原样提交于 8fe6f8a92f8685a2f4fee8f0b8acfb74f373089f |
| `v9_n2m/PACKAGE_ZIP.sha256` | `b68fd71fc4de21c5a06caa9b553e0ff0311284d966d4b8b154ac71744c629fd1` | 65 | repo |  |
| `v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/PACKAGE_MANIFEST.json` | `8ea2d1920bda2750775f7f478a73c0d745849d5be277eb6301b6f04a3011d9ad` | 4970 | repo | 包内全部文件与此清单逐一核对一致 |
| `v9_n2m/VCS_V9_N2M_Server_Package_v1_20261019/config/protocol.json` | `de23173fde6f7dc754001114b9c5db05dfa6dc6cd8cf623a17db195833207b12` | 5354 | repo | = experiment.json protocol_sha256 |
| `v9_n2m/V9_N2M_SERVER_HANDOFF_20261019_CN.md` | `c9cb4e5951956c776a0d98946e6b92800c7688757fb51a3ce12163b1a8dcb44e` | 10291 | repo | = 包内 01_SERVER_AGENT_PROMPT_CN.md |
| `v9_n2m/server/README_CN.md` | `a915a45032c855fd5b47e7238bc35e983570c86690e8e0ea13e53b2efdec3716` | 3838 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/cpu_checks.sbatch` | `8d70d189b0a91e4d664633e7b7f087fbccb511a084e3a2e47d2f2b4b38882c15` | 1771 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/driver_probe.sbatch` | `1974456ea1f49a7768f58c178f623022603ba1ec0aa833766186e16656046d5e` | 601 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/env.sh` | `17f3d71e2bde17dfd4df33720c60ba3b739b2ae95e16114831c9f3fd54baec1c` | 761 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/fixscore.sbatch` | `500d6da76c928fd60de7dd7be6797bd36c42d793cc7ce19f6b1c47ed5d349abe` | 1043 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/prepare.sbatch` | `fada5ae363d57fad3302ba29dbfd5f21e9d53cd8447c419a04608a8480c1cf60` | 1226 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/record_slurm.sh` | `811bb6896a6428c45c8a79a991b83ddbad7bf43605dc06339e705c1e1a01ac0c` | 1322 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/score.sbatch` | `56b6dc62c9ef685585a8b3c56c317830d62371f9d877085ba4921a8c7db040dd` | 1250 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |
| `v9_n2m/server/submit_n2m.sh` | `18ce8f69fb6db242b38cc2e3a0cf90f591ba3b08e43b7fa3fc0eb45f71292d3e` | 3524 | repo | 服务器侧脚本，最后提交 4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab |

### 2 原N2F最终33项表

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `/home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018/CLOSEOUT.json` | `0f5ba97134826d14e2bad05d047f48afeb2f4684d316c6da32319af29072c978` | 48690 | server | N2F收尾（CPU作业1046165，PASS） |
| `OUT/N2F_CLOSEOUT.json` | `f9733a22b667455d10b9b209505f2dcf466a8580bf6de767c9733cb2d9c2f8d2` | 41016 | server | prepare时由包重建的33/33 census（含checkpoint核对） |
| `OUT/inspection/N2F_CENSUS.json` | `8da6f86bafd9da041f2e3daf34d9561b97e3ff4014fd483dabd9e78a47b1d70d` | 40213 | server | prepare前的census（32/33） |
| `v9_n2m/n2f_closeout/N2F_FINAL_33_UNITS.csv` | `88b055a2df1a1efe904a15f5c222c64249a234746c37819608ce60b6732dfa44` | 22298 | repo | N2F收尾附录（含分段硬件），提交 c13b249d59e02e52cd08c717aad8fd9750802225 |
| `v9_n2m/n2f_closeout/N2F_FINAL_33_UNITS.json` | `7368c115abd5fdfc65b62050444ba389e4c4053aada8c78d9ed7190a265f48e5` | 238554 | repo | N2F收尾附录（含分段硬件），提交 c13b249d59e02e52cd08c717aad8fd9750802225 |
| `v9_n2m/n2f_closeout/V9_CLOSEOUT_ADDENDUM_CN.md` | `1b74ca86beaa41e83309a8fc74d73faf505ec0178e219c35ad0958de54d6be84` | 40899 | repo | N2F收尾附录（含分段硬件），提交 c13b249d59e02e52cd08c717aad8fd9750802225 |

### 3 experiment.json / receipt / UNIT_PLAN.csv

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/experiment.json` | `900dec055d5ce5efc4dc3118bde78b4517ec9e966a1a9c2274c50d125ff0380f` | 36555 | server |  |
| `OUT/EXPERIMENT.sha256` | `5aa471ad06768124dba77ca012bb3de959c1ac2327fa9d363a62fecbec9b10d4` | 82 | server |  |
| `OUT/UNIT_PLAN.csv` | `50fe23ebd0032a7c67d8a001beecb953b13d9067aaf94296770e4a28befde619` | 1743 | server |  |
| `OUT/runtime/RUNTIME.json` | `d75a2d7b3dea73f69f9fa34fa42b3d5a2572a84d061a3f5094efdfcfab07a132` | 7311 | server |  |
| `OUT/runtime/HOST_LOCK.yaml` | `ea22393b2feeccffef69b33fbc48f3e156a7eb4da1dddf7c8a4edba0b93ec9bf` | 13800 | server |  |
| `OUT/runtime/src/v9/n2/train.py` | `2b3fb040238d247d46da0bac8ac470e231c5b588ad7be1c7db84a25956532a2f` | 17894 | server |  |
| `OUT/inspection/INSPECTION.json` | `d360007f45a0f0053f8965921e28f641d29bf4db62f1cea301c326964b571419` | 51241 | server |  |

### 4 CUDA实际报告（及CPU核对）

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/SERVER_VERIFICATION.json` | `0ae5fb9eb2762a4c23f29464b9aa02dc254c2b92f34f5410649eaa2d74a8aaef` | 1310 | server |  |
| `OUT/TORCH_VERIFICATION_cuda.json` | `b735d1edc8bed71a178829a2832069da91284531d1ba45f8b068132d6a19684f` | 11770 | server |  |
| `OUT/package_checks/unittest_output.txt` | `5335644a33c675b25b77a6167a5b3cc7c3834d956b9275a8f3057e7580747cee` | 18227 | server |  |
| `OUT/package_checks/local_validation/LOCAL_VALIDATION.json` | `e30bbba5bef3ca0d1272f18f2d0960a5f8a5455b4a5bbf0daf13b026b99f2885` | 1115 | server |  |
| `OUT/package_checks/local_validation/unittest_output.txt` | `b452fbcbc3b779ce70bc6ac8372549647ff84d4d771c745002abe01a9396ff55` | 17773 | server |  |

### 5 controls manifest与置换SHA

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/controls/s901/manifest.json` | `f43a609de4c9885c7cb40df1cda5b2d7847e6c42248f7730a0dd3adc030e3993` | 5930 | server |  |
| `OUT/controls/s901/permutation.npy` | `a45b7c9b272bd407f53ce43f3a156dca7f73c1adc4d6e6f13ee3c737bc52e9ce` | 208928 | server | = manifest permutation.sha256 |
| `OUT/controls/s901/VCS.csv` | `a61fca0f53054ba8e294d04cf58de393378e00114683f5354c03cac102d4186d` | 10339979 | server | 逐pair权重，留server |
| `OUT/controls/s901/VCS.csv.lock.json` | `98f4549b53be29d0fcf15da1dae82371b0f6d8962647133c3171aac57313d39a` | 3516 | server |  |
| `OUT/controls/s901/ConstantU.csv` | `64ed08b6a03f374708b836f3d20b36982b82b096952f005d42abc1400b4749b8` | 3684441 | server | 逐pair权重，留server |
| `OUT/controls/s901/ConstantU.csv.lock.json` | `cc6c3bc623021b0481555a594f0c3f609aedc35faae41d8056f69bc8221f21fe` | 1253 | server |  |
| `OUT/controls/s901/ShuffleVCS.csv` | `a740bd1c6c23b0138d67d481dafa0f4ebbe8ec6ab94db4529bd0bbad003d7d9c` | 3530014 | server | 逐pair权重，留server |
| `OUT/controls/s901/ShuffleVCS.csv.lock.json` | `51a8d0a045f99d8de3bfee90d0686189de662d0c410366d1655f550111603790` | 1225 | server |  |
| `OUT/controls/s902/manifest.json` | `3bd95b1d5afa2140f7748f97650a65242d9061a2319b2d2fa2944210a8014b3b` | 5947 | server |  |
| `OUT/controls/s902/permutation.npy` | `aec6bbcec686f5f9aa4b42b5c36a66b1246005f8077f3223e8bf1cd1e731f040` | 208928 | server | = manifest permutation.sha256 |
| `OUT/controls/s902/VCS.csv` | `e6ad7c8d607ee9cdfe19c264a67929ac1e9d758cc9dc167d2abf2fd52d18273d` | 10340044 | server | 逐pair权重，留server |
| `OUT/controls/s902/VCS.csv.lock.json` | `031b2ba6e94611b172e472d62609d85b1e2462b8793881470f9dc0fef80127b9` | 3519 | server |  |
| `OUT/controls/s902/ConstantU.csv` | `b931a296e6ba6ed777eee186c54eca4cc7c32b595a5ec6e5ef1485d847937bc5` | 3684506 | server | 逐pair权重，留server |
| `OUT/controls/s902/ConstantU.csv.lock.json` | `cffd781b3136b511eb22d0227597998a44ec4d83f3967421772d03b92bac1625` | 1278 | server |  |
| `OUT/controls/s902/ShuffleVCS.csv` | `d98b61f352603c245be06b9e7f86518558fa10f577df49eec1fd9f053a491864` | 3523009 | server | 逐pair权重，留server |
| `OUT/controls/s902/ShuffleVCS.csv.lock.json` | `d5b2da0cdcba764756deb5bdad77941383eededb6e119e93261de1425db7439d` | 1227 | server |  |
| `OUT/controls/s903/manifest.json` | `74523482add576098149175fa6474b2ed05685742fc215030afde92ee9e25d0d` | 5948 | server |  |
| `OUT/controls/s903/permutation.npy` | `f9eb10169a8f2af30742d550b7736884a11d6acb2cfba0bf0967bc2a01363d7c` | 208928 | server | = manifest permutation.sha256 |
| `OUT/controls/s903/VCS.csv` | `7e1a09fc196c41ab8b77ccbf84d43f4d03f15956a48136994294aa49b0ad7550` | 10339794 | server | 逐pair权重，留server |
| `OUT/controls/s903/VCS.csv.lock.json` | `8ed76c58b72b2e1a383ab60b79be8a252fed9b76ed37e244c6216206b896b8bf` | 3519 | server |  |
| `OUT/controls/s903/ConstantU.csv` | `e79a3af90bcbfe37213c4be1dcc02d3a07fe6929d69fdf6fe5f70bb6ed5a0423` | 3684256 | server | 逐pair权重，留server |
| `OUT/controls/s903/ConstantU.csv.lock.json` | `a421d47ae83e27b6c0e28fec59eafcd8c45a127a44adee00bb91495c50ed1fc5` | 1278 | server |  |
| `OUT/controls/s903/ShuffleVCS.csv` | `3af5086d80a1832188be8724fc26566753861d1f7e78d8110f330ed053d3ae8f` | 3494784 | server | 逐pair权重，留server |
| `OUT/controls/s903/ShuffleVCS.csv.lock.json` | `6134b394acb6870150287a31559e773b70eeb13511806606109bb18feb6ec500` | 1229 | server |  |

### 6 每臂identity / run_manifest / full_result

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/runs/s901__VCS/identity.json` | `f651bb52d86895d72ce7413c1983670c4a1510f617bd5e2d8de166d015bbb51c` | 964 | server |  |
| `OUT/runs/s901__VCS/run_manifest.json` | `29804703fdad4ce6a1b3918a4be82be89a3d131aab4e7e8d4d0fd23ac3f875ab` | 827 | server |  |
| `OUT/runs/s901__VCS/full_result.json` | `e3fcf49906727f35db2302ee8148188826f530e24bda9e054b4095e0a13da8b9` | 4563 | server |  |
| `OUT/runs/s901__VCS/done.json` | `235bf0c444ca6784a035ff7f37beb5772b95676f626388fceaba4842201400c5` | 265 | server |  |
| `OUT/runs/s901__ConstantU/identity.json` | `41b5a4510f96b6640e5907c8bfd9752a92f8cf2fffdea68da8fa56f44399e192` | 982 | server |  |
| `OUT/runs/s901__ConstantU/run_manifest.json` | `9ece66b481a062a38b52133ced746a3317b28ccb0c3ead969bea945f78b935aa` | 826 | server |  |
| `OUT/runs/s901__ConstantU/full_result.json` | `4f5c4d3b0ebb4eaf4ef97aecbd5b2c24ab328a8a79af1e2d44aabfc0d866bbfb` | 4592 | server |  |
| `OUT/runs/s901__ConstantU/done.json` | `3f6be1ec8925a4fd94d7ac9adcb3658ee3f01119bc6e4759f9fabfc51c67bbbe` | 264 | server |  |
| `OUT/runs/s901__ShuffleVCS/identity.json` | `d0653b284d3cbd928d0b6394d749f486df196b504ac1157c8583da7c97088cfd` | 985 | server |  |
| `OUT/runs/s901__ShuffleVCS/run_manifest.json` | `8573358e094cb42ce38979577c6b36a5a9d4a1143d8c28febc7df0b96a74b6ff` | 827 | server |  |
| `OUT/runs/s901__ShuffleVCS/full_result.json` | `fc50afdb7f1418a1eed3f69e04e1f9f692289f0cc5a6279670470a2e7dfe4c41` | 4594 | server |  |
| `OUT/runs/s901__ShuffleVCS/done.json` | `ead9f2a2a41fff79c968a6810f88f3ba9530dacee92928140eb8adfc17e98dbf` | 264 | server |  |
| `OUT/runs/s902__VCS/identity.json` | `c730a7908510d8f9393309e0307a61f9d47fa1a82ae321ea3b6ffe1abb288705` | 964 | server |  |
| `OUT/runs/s902__VCS/run_manifest.json` | `c7a8299564dad4b2986b54b8803b6123dea0a0e554fe42d6101f8c63992c525c` | 826 | server |  |
| `OUT/runs/s902__VCS/full_result.json` | `5cca853df3b9320e190d6894082899c4e63ac7baed1b3329d096b87bb9b6f0b7` | 4563 | server |  |
| `OUT/runs/s902__VCS/done.json` | `1f75b93d18510b226ca02c4b7275d739a00163d524d18ef40b8c02f44e9e7248` | 265 | server |  |
| `OUT/runs/s902__ConstantU/identity.json` | `ad22ca12d532a95d658857df3427ad7b8c3f1a7fb3892c3514f55f434cd4be7d` | 982 | server |  |
| `OUT/runs/s902__ConstantU/run_manifest.json` | `aeda2233c40ca77ab642c99e6a66c4e362a1f6c914f6fa000797cd8e743f7841` | 827 | server |  |
| `OUT/runs/s902__ConstantU/full_result.json` | `e8b357b4a9cd58b54518bbea2d8c93d1e107ef79b358a93138b59ae8fdc59c2a` | 4592 | server |  |
| `OUT/runs/s902__ConstantU/done.json` | `45812f33a4910743baa990728903535542d2348b0a0237bf65944d56c3aed2e0` | 264 | server |  |
| `OUT/runs/s902__ShuffleVCS/identity.json` | `ce8aa2dda2b7e866abb4268ef4e26806702c7531368a8ddf1b073d81a142e202` | 985 | server |  |
| `OUT/runs/s902__ShuffleVCS/run_manifest.json` | `d55e15093da887fda4636e9ae9a0277afdf74df4bd03d66bb912f25610322cd4` | 827 | server |  |
| `OUT/runs/s902__ShuffleVCS/full_result.json` | `e5b74c93abc70c8be4b24586f59dff2673fe25e585ee5199fa3f1f80117ea965` | 4598 | server |  |
| `OUT/runs/s902__ShuffleVCS/done.json` | `82cfecba4c4fb1fd37d916a9824cbd364eca7f37061ff636f177016097ea8589` | 265 | server |  |
| `OUT/runs/s903__VCS/identity.json` | `b494119eb9042d3db56100af4686e8343ab5784244ab7a86e0c26e5420dab8cb` | 964 | server |  |
| `OUT/runs/s903__VCS/run_manifest.json` | `5f8bbfe3243520e8a0e04616fcb350faf49dcfef2d0d1dad541d62a72e4d19b2` | 828 | server |  |
| `OUT/runs/s903__VCS/full_result.json` | `2dbfb04256527dbfd0d583cb71e9a7100c4b047c250f7546b24a093a53ee3d24` | 4564 | server |  |
| `OUT/runs/s903__VCS/done.json` | `f6ebb903de993555b073a437dd077d160c43b3651f7a7a5755fcc3a5492331fe` | 266 | server |  |
| `OUT/runs/s903__ConstantU/identity.json` | `75ad902d9e1d046315df8dd60882204190a6cd945f2b286dac9f0172d24d28a9` | 982 | server |  |
| `OUT/runs/s903__ConstantU/run_manifest.json` | `5445fde51d3f1bb595de5d4f5fc31132b940eebb1ebab69359a0ee77c02f0c12` | 826 | server |  |
| `OUT/runs/s903__ConstantU/full_result.json` | `e9167703dad6ad7407dd68dbcaefa5bbf1b6c6023e1caa80d58b3ee92bc38dbe` | 4584 | server |  |
| `OUT/runs/s903__ConstantU/done.json` | `d7da5c634a978ca0a522c3c8f6a6c13091c10635d0198e00ce78fe8c0e378be3` | 265 | server |  |
| `OUT/runs/s903__ShuffleVCS/identity.json` | `bdcd004f4a8e5911ed6b3d56a9190bf24c6842e713f71706744bcb43758e1a54` | 985 | server |  |
| `OUT/runs/s903__ShuffleVCS/run_manifest.json` | `69f7d5051e5a83f0b036832a6b7929d1be5515ebfc547ecefa0cedce1533a8b9` | 828 | server |  |
| `OUT/runs/s903__ShuffleVCS/full_result.json` | `cb160dd10802f9bdb064b87a9923fa23ae101deb5f073826e993252ffabf6191` | 4598 | server |  |
| `OUT/runs/s903__ShuffleVCS/done.json` | `bda14db2c428df48078cc7e1e67f86fd8c98c07fd1d78980c2702ea28098f573` | 265 | server |  |

### 7 完整train_log与progress

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/runs/s901__VCS/train_log.json` | `912e2ca18c2ca314b3db10dc34bba1e2fa06b76a47d8003bfdb909ab3969baa3` | 598894 | server |  |
| `OUT/runs/s901__VCS/progress.jsonl` | `52d081e1b31a323c43f36a8fc13572337744996e4d372ace9dddebc6745dabe2` | 7573 | server |  |
| `OUT/runs/s901__ConstantU/train_log.json` | `0dea8b73de6373e2afb87fc204aedcea0ddd284ce48983c95af0e608cf1c48d1` | 604916 | server |  |
| `OUT/runs/s901__ConstantU/progress.jsonl` | `f56d776ae1f49b20a0298ae7aab24b243d701294e46573c5bd341e3ec5bc3ff1` | 7572 | server |  |
| `OUT/runs/s901__ShuffleVCS/train_log.json` | `5b0e7ac27b42f2a4ac185275c98a54c1b9f0f5bd6f437c6101585d2ef32d1390` | 598614 | server |  |
| `OUT/runs/s901__ShuffleVCS/progress.jsonl` | `83b0aa630343f1a8f800b9e385a17586d2080481e7b734a1aacc677619a7d954` | 7580 | server |  |
| `OUT/runs/s902__VCS/train_log.json` | `666604067f256e1747b4fc9df016a82391bdab22ade2d45926aaee8f52cc9bd3` | 598119 | server |  |
| `OUT/runs/s902__VCS/progress.jsonl` | `68945f43d33e43ff4a9c31c9e8801f7a2596dd09511b373c42d4df0cd8206b03` | 7587 | server |  |
| `OUT/runs/s902__ConstantU/train_log.json` | `e2cf9e6ad86447d8cda4b26c449dc41c80b1bb934be0eb588f1c2e9341d6cfe4` | 603002 | server |  |
| `OUT/runs/s902__ConstantU/progress.jsonl` | `eedc0573b42341079342ae6ef46a82dc628993bece215586e69c57f859419f67` | 7580 | server |  |
| `OUT/runs/s902__ShuffleVCS/train_log.json` | `b2068dadd4cd58ff4f3996c80ead2a2159bdbfc44e49718b3a9f55c2ff22fba2` | 597857 | server |  |
| `OUT/runs/s902__ShuffleVCS/progress.jsonl` | `52a5d486660700c4cebdc59a8de2a60a30c3304965822e1521988491f45fa030` | 7570 | server |  |
| `OUT/runs/s903__VCS/train_log.json` | `c032e3011373b9ed233ea607d74f1f76a389a17f3f3ad1b7af7fe12fdc6a6597` | 598231 | server |  |
| `OUT/runs/s903__VCS/progress.jsonl` | `aad60e08a8d9bdaec4d7dc8ab08c68d21b73cdff3aac8555127cf1d2f922b24a` | 7562 | server |  |
| `OUT/runs/s903__ConstantU/train_log.json` | `453bf38d98467ab0dcd9aa2e1ba061a469f5c4665cd8e32a4b76f01004c0f2c7` | 604341 | server |  |
| `OUT/runs/s903__ConstantU/progress.jsonl` | `a34b5725b5366f440f1745576738c6152701693ec02bf69d130686ab7014fb1e` | 7564 | server |  |
| `OUT/runs/s903__ShuffleVCS/train_log.json` | `2832443b2e2093797a746f30139a0ff3ba978adad1b41d921567e5a32021ae88` | 597905 | server |  |
| `OUT/runs/s903__ShuffleVCS/progress.jsonl` | `3fcde3642e76d51b46b486185b57a093efc449395a381a34e025015982fe1255` | 7574 | server |  |

### 8 grad窗口

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/runs/s901__VCS/grad_norms.jsonl` | `81bef2a53a16bb536142f3f1f0a93ea9df9a7267339b6839523a8477fbb5d68f` | 92789 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s901__ConstantU/grad_norms.jsonl` | `e5d05ca317bd330271660535b98dbd97ae05f7a796daf58e64f0dadb6631fb46` | 92817 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s901__ShuffleVCS/grad_norms.jsonl` | `cb87a9ff1647cbc5b1e08521fe38d19ea516ec0f5df60845e91d79470b070ea3` | 92877 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s902__VCS/grad_norms.jsonl` | `1f7541a7a7c01e1b44e2fb3b6d30e21f9c6f671ad77f290fb7aef044e755f982` | 92826 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s902__ConstantU/grad_norms.jsonl` | `9cee4bf5fc30970a4c6c5095ef6bc4c9f2f5a3f589d49059b5cfb47903564b03` | 92911 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s902__ShuffleVCS/grad_norms.jsonl` | `db69cd6b5be421e49e539fc85fa4be9c446f594b2bc1ffc64e1f24c0f30160e5` | 92924 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s903__VCS/grad_norms.jsonl` | `4f6204245577bba380a74a3bb4581b620fb6c87a3a1838aaa10f1b49da9276fc` | 92820 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s903__ConstantU/grad_norms.jsonl` | `58dc576aac54f7ab29672933059510f58c886ab9c86282c862e51a84807e515b` | 92909 | server | epoch4末200与epoch5初200次更新 |
| `OUT/runs/s903__ShuffleVCS/grad_norms.jsonl` | `867fa088238bba22117c6e5741e4f04c6b9ccbe09670dab000a24922e757aed2` | 92894 | server | epoch4末200与epoch5初200次更新 |

### 9 全部attempt与sacct记录

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/runs/s901__VCS/attempts/f97d4c8d-e741-410d-8914-c1b29bd0bc0a.json` | `e67aefe1a1d5e09ae31001fdd0fdd203588170a8fe9cad9bd54e848818efce61` | 1131 | server |  |
| `OUT/runs/s901__ConstantU/attempts/3b2a75d0-c4b0-4012-a4f4-d7dfa43476c5.json` | `ea07f57455fefa335e63df3d59344407c26924c49ca6e319bae55a8e23d9f5ef` | 1137 | server |  |
| `OUT/runs/s901__ShuffleVCS/attempts/ad62ceaa-cb14-4232-92fa-38854299eaf7.json` | `4e7c7a661a3606574f0351fb51891741b9ca063b01c852a52c0e7c6b314ba1b1` | 1137 | server |  |
| `OUT/runs/s902__VCS/attempts/8c4da523-a2a0-4bd3-b2ba-4b9d4a2d8c6f.json` | `60347d0df0dc7b2df95647dca6eeb7dc96ced8261e304cde6a2dac485cd4efd8` | 1131 | server |  |
| `OUT/runs/s902__ConstantU/attempts/374f4cb9-aa3f-40b9-9c27-f72d2553e7eb.json` | `96b3e92981732f718c5fd7e29ee9ae420c939a22134904bfad621131163f854c` | 1137 | server |  |
| `OUT/runs/s902__ShuffleVCS/attempts/140fafa8-f535-40f8-b7b6-dcdec2d84551.json` | `0d0ecd05e713510790263604ac0562fc4709f259f9fc489dd82f5527bdac388b` | 1138 | server |  |
| `OUT/runs/s903__VCS/attempts/ebc4afde-17b3-4299-ac2a-14b15055739a.json` | `0bc1eda8657ecbb490214735a298b0b683901508c3eef7cf7e2f1d781034300a` | 1131 | server |  |
| `OUT/runs/s903__ConstantU/attempts/60454074-f59e-4330-9674-0de632d2489d.json` | `818442d7271b4258c339cd419e712a06eb550aa6dbf881b62db708d8ddb139e1` | 1134 | server |  |
| `OUT/runs/s903__ShuffleVCS/attempts/e8062855-0686-442d-be9b-1eb53e6ec0a6.json` | `f50541e11467c7a7b2548a8684f044842e0692bf3645b041b4f14552896408eb` | 1138 | server |  |
| `OUT/analysis/cost/SOURCE_NOTE.md` | `04615b0d69a7aeef5dbbb8ccdc99a4b05d60a412ac35e79556d7599cedd4fea8` | 764 | server |  |
| `OUT/analysis/cost/sacct_from_scontrol.txt` | `ae101686d56ef31da0f6d3216d3604dcd254817eaa5d22d257ed40af02587a3a` | 1017 | server | sacct不可用；由scontrol快照重建 |
| `OUT/analysis/cost/sacct_raw.txt` | `ae101686d56ef31da0f6d3216d3604dcd254817eaa5d22d257ed40af02587a3a` | 1017 | server | sacct不可用；由scontrol快照重建 |
| `OUT/analysis/cost/allocation_accounting.json` | `c86a964f9e70af88e9b46174563eb71c89668e6feabca3543c97f9b414b657a0` | 5377 | server |  |
| `OUT/analysis/cost/allocation_accounting.csv` | `2b91bac904df16f9d149a37e9c9610d3b2602a6b3796ba4b8133ec739ffb2cb9` | 917 | server |  |
| `OUT/slurm_records/history_1048432.txt` | `a169fc2acfa3f79767d4cb09b18c1729b0a42a2f6395f4fe2fa72ea86881f799` | 4021 | server |  |
| `OUT/slurm_records/history_1048435.txt` | `5dca7acc679802afc2e506d79e9e24d19b36d8cf5b3cb67ff4ee928f52c72f30` | 2792 | server |  |
| `OUT/slurm_records/history_1048493.txt` | `764ed363f7d06f8e3e17009f9e208c0de198fbbadf5e24d2d8ac448a9ed8006d` | 3685386 | server |  |
| `OUT/slurm_records/history_1048494.txt` | `2c6de485e0f43f284eb26575e18863f95728e58fd42196d75bab5a7719a72ccd` | 3700046 | server |  |
| `OUT/slurm_records/history_1048495.txt` | `87f788c4f947a5029998aad1c0f279ca4fe87997c5c1d4f408271454b6cab4e4` | 4078868 | server |  |
| `OUT/slurm_records/history_1048496.txt` | `25ae025baec702b343ac083fc1da65726944f5e0dc8186fff805db9c5153299f` | 17622 | server |  |
| `OUT/slurm_records/history_1048497.txt` | `2cef1d81fdd28a157f6ec8bd48e39dcd841d4239b6e51532d7b19994b26daf2d` | 31177 | server |  |
| `OUT/slurm_records/history_1048498.txt` | `66c6102a515dbd84ac7e7e458eb701095b57824c368607052553beb841e3b16a` | 1518549 | server |  |
| `OUT/slurm_records/history_1048499.txt` | `e6f4b8a493bbfd2307335e773e10a0acd1e4bd17a3370ee71b735f832fbca504` | 2622 | server |  |
| `OUT/slurm_records/history_1049085.txt` | `2ad08ac9d56a496f3d3693b586dc19ce9e9a7f71a718ab77c53818b677f52396` | 3879 | server |  |
| `OUT/slurm_records/scontrol_1048432.txt` | `caa3542a7c7f810c96774308a3fe51d8d2ed54662868299ee9c6913768c83a76` | 1364 | server |  |
| `OUT/slurm_records/scontrol_1048435.txt` | `7f16981bea5fd66832d49cc55c9a16fb389a7fcc47210bfcbd8f1cf9a5e695a6` | 1376 | server |  |
| `OUT/slurm_records/scontrol_1048493.txt` | `de655795dcfc7f4798eb72adcef19579c5f3020ed844acdceee858141b0c12f5` | 1447 | server |  |
| `OUT/slurm_records/scontrol_1048494.txt` | `4f7a002359fe5d2e9b8dd05bff282bcd2d283ed0a0c9e6363c4367b872c20ca5` | 1447 | server |  |
| `OUT/slurm_records/scontrol_1048495.txt` | `4169c296cca7e5ae979ecb99d2cdf876c01700e49e2350e0fd73dae82d76d842` | 1447 | server |  |
| `OUT/slurm_records/scontrol_1048496.txt` | `c2412a0e57d9a3573150fdb8ea5a1ee034f6a82eaa78d2171967a0af147d87b0` | 1461 | server |  |
| `OUT/slurm_records/scontrol_1048497.txt` | `100b9abad2bd00a7ff7ee16307119aa5764a41ab3855b008d153ea0fdc01c6a7` | 1461 | server |  |
| `OUT/slurm_records/scontrol_1048498.txt` | `c6bd7b8c3d4a6b49b3107dcda30f9dd6da2be7f3d251a0973decc77db6339e96` | 1457 | server |  |
| `OUT/slurm_records/scontrol_1048499.txt` | `680e649efa351d1b4777e7dea85c64ceed34cc4bc0b6d8ced0ad279f85df01ea` | 1356 | server |  |
| `OUT/slurm_records/scontrol_1049085.txt` | `b45e7ef3baf6513fbd194438303af203b3ae643788c24a3e8f46b5dd4656778c` | 1319 | server |  |
| `OUT/submissions/SUBMISSIONS.jsonl` | `b0dae84f2ef082caf61764bb0ad8a94b6e028c6532db6e5d246f285dd511214e` | 1596 | server |  |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_ctrldry_1048433.out` | `968ea2f8f604f6a4691b92d8c4b4177578e811d6d1f3fb86d90fc36595c5f722` | 240 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_ctrldry_1048435.out` | `6c2730eb132cb801b914ae4c3ea8d72f77ecd19d8e1695ad1e9776fee707245a` | 3039 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_fixscore_1050484.out` | `575d2f9732554723d99bd79c48498669e9befb0e69d7322a145dda9bec6719fe` | 2236 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_loss901_1048496.out` | `6a38c4f915d48ff2834ab88b28f902a20af573411f6049a3c7bc965c22965afa` | 891 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_loss902_1048497.out` | `da2bc981807e566772ca0149f91d29424d1d47464ebd3cdd98e41a53f5998276` | 891 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_loss903_1048498.out` | `d9e094d4457d5333be033ca9434c89443ffc91e0e0d4bc04f49fb6233b991060` | 891 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_prepare_1048432.out` | `bbfeaf649a95c97c4799552831900611c814a64d4f6b3c3d16849904a1c66d57` | 323 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_s901_1048493.out` | `4ab817611fd2be253be516982b247da1693d147293cf1c35d07baa517b97813a` | 2149 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_s902_1048494.out` | `941fdca80400548aa6f441066c5e91d57e161526377bbabdd0d7274591312a13` | 2151 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_s903_1048495.out` | `e139e534d985e3762b7a67d4b6b00331dd20566dd3708af29602effa9480e719` | 2153 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/n2m_score_1048499.out` | `0090a131ae2d5d39c68d16e512f095b00149001cf06c5ddbc5734cbb41491e84` | 6880 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v9n2m_cpucheck_1048421.out` | `d3054a8f4f57a4c974d4ae31d26230029c2d6c596ab1b97bcab4c847a2c79cbf` | 1726 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v9n2m_drvprobe_1048423.out` | `c35109e4e94fa88b8d42381f0437c0fb7777120f2873bd157d6c527f85b4d5c5` | 261 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v9n2m_drvprobe_1048424.out` | `cde05f88c218a6b2ce95b64f3d3a051fc81c0934ec188298c56e18af368929ad` | 261 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v9n2m_drvprobe_1048425.out` | `e72da3737ef746bdac773aaa616412df02d2cf0e88b82b0c835dcf417c76dcf5` | 261 | server | 作业stdout |
| `/home/infres/yinwang/CS_QMI/slurm_logs/v9n2m_verify_1048422.out` | `dbcbc3aeffb2012fd2c12152f309f71aea4d6ad65598529449186d96b1950594` | 2433 | server | 作业stdout |

### 10 三个离线后验表

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/analysis/offline/s901/offline_metrics.json` | `0b4f78f46dda36bc5465fe3a702f2b7b4e691b9503283cae5bb081b16a33cb79` | 21936 | server |  |
| `OUT/analysis/offline/s901/offline_metrics.csv` | `b3a52eeb9f421ca18221d28cf35c43101a8fea39342b4c4bdbc368531f2068ca` | 1045 | server |  |
| `OUT/analysis/offline/s901/calibration_10bins.csv` | `573d9fd04af2deaae3c8a1bf99f41dc09763a61a8540fc6778b7fe7ca7709f32` | 3549 | server |  |
| `OUT/analysis/offline/s901/OFFLINE_READOUT_CN.md` | `03acf876024cfb2e0bf5c3a020277da0958ae5dade0bad8b817182bcf6dc2541` | 1427 | server |  |
| `OUT/analysis/offline/s902/offline_metrics.json` | `0f5e3d46156f164920ee89a6ec0d7b7eccdb6745c157dc79bc443334c565f031` | 21936 | server |  |
| `OUT/analysis/offline/s902/offline_metrics.csv` | `c91d5ec1c5084e9c610ac01793927e2aa463df6f85324b4507dd985e1aead3bd` | 1051 | server |  |
| `OUT/analysis/offline/s902/calibration_10bins.csv` | `ff3eead967a4938784986d52c1874ae0e7d1628301c5d59230c222193c2a3ff6` | 3542 | server |  |
| `OUT/analysis/offline/s902/OFFLINE_READOUT_CN.md` | `e6b34fd82b0ef4f6e884a8ed187dd69833ce41a79a6ade921a304bfaba90213d` | 1427 | server |  |
| `OUT/analysis/offline/s903/offline_metrics.json` | `2193e4447c21ecf5b2d59e18f0798b4ca91cb506e89126621d650b8b4ed6c1e2` | 21969 | server |  |
| `OUT/analysis/offline/s903/offline_metrics.csv` | `e2e05aba2b58b1b75d1b51c98787174a979c50b8ad4f237fe9cfeb93a707cebd` | 1049 | server |  |
| `OUT/analysis/offline/s903/calibration_10bins.csv` | `ae6b76e6922b405b8dfd973dd9b6e4e301e8be0b4dfe221231830e6d987c3648` | 3585 | server |  |
| `OUT/analysis/offline/s903/OFFLINE_READOUT_CN.md` | `9f6c805a8d537776f13819188ec7d271de84636b9362debb09736e6ac4c77714` | 1426 | server |  |

### 11 三个固定风险诊断

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/analysis/fixed_loss/s901/fixed_loss_metrics.json` | `8caab12d606e0debee6d25f6fc7090a3c92446c3e88e991d44c82f7cad9621cd` | 29698 | server |  |
| `OUT/analysis/fixed_loss/s901/fixed_loss_metrics.csv` | `1dc68a49810d90d45945270c6addf80cb614ab48104e2fe7eef2abace57fa64b` | 2510 | server |  |
| `OUT/diagnostics/s901/fixed_losses.csv` | `0cf879352806c4ea10348e348d8706f2ed47e6e901ae13ba3d8a98405c7da5d6` | 22729 | server | 逐anchor ell，留server |
| `OUT/diagnostics/s901/fixed_losses.csv.meta.json` | `b51d8b9d32caac7cc19c52eac05df03a6ff16e606bcd27e7c84dfe78f2c3b411` | 18868 | server |  |
| `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/results/901_1/VCS_Weight_sum/best.pt` | `2321217686dd849dfd798913e7a8d8876451196f56865715bbc8e7c48e1c6e5a` | 217619173 | server（二进制，checkpoints_not_materialized） | N2F 901_1 VCS SUM best（epoch 39）；模型二进制留server，checkpoints_not_materialized；sha256为N2F artifact SHA |
| `OUT/analysis/fixed_loss/s902/fixed_loss_metrics.json` | `7fe82f110cd5f11828d2ca2449aeeda5ca20b3949a361af45cbcb8f7a129f20f` | 29704 | server |  |
| `OUT/analysis/fixed_loss/s902/fixed_loss_metrics.csv` | `c331bec721300380cc2a7943b5d240dec21d0669cb37bf58d595ac87b6dcc4ef` | 2513 | server |  |
| `OUT/diagnostics/s902/fixed_losses.csv` | `2723ea558642dc8ebcc227dca8a137cbb4bfb017e2ed2a5842a383a3b2d6e2ef` | 22317 | server | 逐anchor ell，留server |
| `OUT/diagnostics/s902/fixed_losses.csv.meta.json` | `ad36427fd40c5200337d50cde50ee10980c95451e6be901c3fd9bc6c9a73b787` | 18871 | server |  |
| `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/results/902_1/VCS_Weight_sum/best.pt` | `356c7537387185db1538ed37709de61b273bbd40802333e05659654638486d77` | 217619237 | server（二进制，checkpoints_not_materialized） | N2F 902_1 VCS SUM best（epoch 39）；模型二进制留server，checkpoints_not_materialized；sha256为N2F artifact SHA |
| `OUT/analysis/fixed_loss/s903/fixed_loss_metrics.json` | `859b955128581e58841bdd6e6cfe6c3b0a8a92705f6f8243a07eb80108a0d993` | 29711 | server |  |
| `OUT/analysis/fixed_loss/s903/fixed_loss_metrics.csv` | `649c92c231ba1ca4acb0ba842b741fae9fe242a754a099fc6728284a7d50611a` | 2510 | server |  |
| `OUT/diagnostics/s903/fixed_losses.csv` | `0800b5953fa32cb35e5581d65b202a72e2c26ede2bdd8cd2a1b5c5e889ee1fde` | 22999 | server | 逐anchor ell，留server |
| `OUT/diagnostics/s903/fixed_losses.csv.meta.json` | `3f137dd78d8b9bcded89f53eeab7930717a0e1af60fdf2e6e9584aa95cec9bfe` | 18881 | server |  |
| `/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017/results/903_1/VCS_Weight_sum/best.pt` | `3c863e9c6e929f6eed7abfb85748a462a601d6622ddeb94358f43ad5b7beaa10` | 217613605 | server（二进制，checkpoints_not_materialized） | N2F 903_1 VCS SUM best（epoch 38）；模型二进制留server，checkpoints_not_materialized；sha256为N2F artifact SHA |

### 12 最终summary与两个配对差表

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/analysis/full/summary.json` | `42aac7b34b5a4f3267a223d66d14207e08e5901ce683d68af76e36ef99993d21` | 513489 | server | 含两项配对差、工程规则与逐单元记录 |
| `OUT/analysis/full/paired_contrasts.csv` | `49386655431dfb767d07f7d124d0a07e82202575792151c937f05e9789a87204` | 954 | server | 两项配对差表：VCS-ConstantU 与 VCS-ShuffleVCS 各3行 |
| `OUT/analysis/full/unit_results.csv` | `ce4297924b1198c086b8d4b864212ec599f0bb24602de1bdad69b42225123e64` | 3253 | server |  |
| `OUT/analysis/full/epoch_trajectories.csv` | `ceafdc2e3dd17433185ce23d94846070e1f3253cd5e7329f9fc9c0de02b1c326` | 92456 | server |  |
| `OUT/analysis/full/RESULTS_CN.md` | `5a95125cc8ab17b62d3548a9b2f96955e6ae1f842155dc346cfe49fe353e9a01` | 1432 | server |  |

### 13 模型二进制（N2M每臂best/last）

| 文件 | sha256 | 字节 | 位置 | 说明 |
|---|---|---:|---|---|
| `OUT/runs/s901__VCS/best.pt` | `25e07993227faf76bb351812c9f5315fbba2721c7195b8a54c0b2143886fc892` | 217642469 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s901__VCS/last.pt` | `146eda13f0a7a55367952c3f4fa02a0dc8b2cec2487cf80f2d6a53c45a6163ac` | 217655525 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s901__ConstantU/best.pt` | `f3981423bf1c094c9ac62fd2a98d42177d0010b5d9237e859bef835c7bf5fbd4` | 217655525 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s901__ConstantU/last.pt` | `33429e634606fee68f4245ba931e00dc99d8161b99c8358475a67405e6ad97e9` | 217655525 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s901__ShuffleVCS/best.pt` | `cec2f4055db1443ffa3acee4d92e93a0096f7b88147d1a5d19337499ca309b55` | 217629541 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s901__ShuffleVCS/last.pt` | `f432aa4df4fcf97ce21af6f53a3d3d0ce08478445666150543215b51d353dffd` | 217655525 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s902__VCS/best.pt` | `6c0a1d0e744e99c2e9ab23551aa34d2808c82d4b62087ecdab854a4a12a207a6` | 217629605 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s902__VCS/last.pt` | `1d8dc05e3bb701dceb64b53025bb4cb5f31983cde04616c92fec398b81736cc8` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s902__ConstantU/best.pt` | `bba4f56834d86c9e31033d682a4759503a0d857aa2061977be7af573b12bb252` | 217649061 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s902__ConstantU/last.pt` | `ef3fb3dc8b0e74d55c55fbc998099bc4dfb49b541c32ea84e0e8be314cda48a3` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s902__ShuffleVCS/best.pt` | `ae5d4845e8ee5af12a107fafcaa4e93177ffbf03aef61285edda5f1744d79779` | 217642533 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s902__ShuffleVCS/last.pt` | `3d6d9f7ea7a70245df372c6ee256884cf672ecda65e5d5df4c06732905f1371c` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s903__VCS/best.pt` | `b52d5b5475872c0f2fd10c5449c5cae209ba57d49c1a683e62f0b1d4bb6e4cd2` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s903__VCS/last.pt` | `07425883e552cabb18a5db005fc58a6344636768f15173bf2e336a60a1beea22` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s903__ConstantU/best.pt` | `5128d3dc44d8b93cd2fae0e0d4dc60f2befb07628c9ccc59dca591782151b937` | 217597157 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s903__ConstantU/last.pt` | `df9a34856293595584453191af79bfa12f43663efb14fae267d9d1e050a43975` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s903__ShuffleVCS/best.pt` | `900f62f450efaaf871932d9b4ea0ee2a3b11af074638efb23bc9ec999015eb87` | 217636133 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |
| `OUT/runs/s903__ShuffleVCS/last.pt` | `dd34da5fd23690f75faac1463c7e22cae61b5ed90bcf7acf6c6c5047dcdf1860` | 217655589 | server（二进制，checkpoints_not_materialized） | 留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED） |

## 7 核验与复现

`python3 v9_n2m/handback/n2m_handback.py --check` 重新核验全部记录并与已写出的文件逐字节比较；`--binaries` 另外重算 21 个 checkpoint 的 sha256（4.57 GB，只在 server 上运行）。核验包括：experiment receipt、协议、CUDA/运行时 receipt 与 N2F census 的哈希链；包文件与 PACKAGE_MANIFEST、zip 与 PACKAGE_ZIP.sha256、相关 git 提交未变；9 个单元的 identity、JSON 产物哈希、40×1133 次更新、best 选择、Recall/rSum/mR、progress、attempt 记录、梯度窗口（由 grad_norms.jsonl 重算）、同 job 配对与环境一致性；summary/unit_results/paired_contrasts/epoch_trajectories 与训练日志重算一致；两项配对差与工程规则复算；controls 的置换与三臂文件哈希、离线评分与固定风险的身份绑定和 CSV/JSON 一致；固定学生 checkpoint 与 N2F artifact SHA 一致；成本行与 scontrol 快照一致、缺失作业确实没有记录。任一不符时脚本拒绝运行，不写文件。
