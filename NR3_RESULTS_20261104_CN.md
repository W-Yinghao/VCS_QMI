# NR3：参考感知的起点校准 × 固定 NCE 前缀后的 J/BCE/NCE 续训（2026-11-04 第四十二版）

接在第四十一版（`R49_R50_NR2_RESULTS_20261104_CN.md`）之后。只发布数字表与代码，不附解读。

## NR3（`handoff/nr3_20261104/`，`RESULTS_NR3.md` 与 `RESULTS_NR3_*.csv`）

- 用户：「查看这个包，准备并提交相关实验」（`VCS_NR2_Review_NR3_Server_20261104`，证据快照 a05c6281）；「NR3结束后把结果push到github」。
- 条件：Flickr00、Flickr60、Flickr80、COCO60、CC152K_RECOVERED_20261104；G 头，seed 1001；SigLIP2 固定 revision、样本级 L2、参考 proposal（0.5 边缘 + 0.5 基础 FIT Top100）、HREF 训练 logit g = s − log w，部署一律用 s。
- D0（只读）：两个固定 FIT 诊断池 CAL-A / READ-B（每方向 32768 个 P 与 32768 个 Q_h）上，NR2 HREF step-8000 J/BCE/NCE 的原始与正仿射校准 HREF-J、g 分位数、|T|>0.99 比例、1−T²、dL/dg，按方向 × P/Q × hard/非 hard 分层；step 0 同一训练批次上 NR2 与 NR3-A 初值的 g 与梯度。人工错配真值只在隔离评价器内使用。
- A（15）：HREF × J/BCE/NCE，初始 a、b 改为在 CAL-A 上对实际 HREF 参考拟合的共同二维 balanced-BCE 校准（L-BFGS-B，a∈[0.1,100]、b∈[−20,20]，校准收据 `PUBLIC_CALIBRATION_RECEIPTS.json`），新头、最后残差层为零，8000 步，其余同 NR2-HREF。
- B（15）：各条件固定 step-8000 HREF-NCE s1001 父 checkpoint（NR2 / NR2C / NR2D）→ 仅用 FIT 的正仿射校准精确折回同一 G（`PUBLIC_PREFIX_LINEAGE.json`：DEV 排名 0 行变化）→ 三份相同副本分别续训 NCE→J、NCE→BCE、NCE→NCE 各 4000 步（新 AdamW，lr 1e−5→1e−6 cosine，sampler seed 101001），总 12000 步；父 checkpoint 只读。
- 30 项均一次完成；每项给出六 Recall、rSum、相对基础模型 / 前缀之差、救回 / 丢失、READ-B 上相同 Q_h 的 HREF-J、曲线与成本。不读 TEST。
- 代码：`code/nr3/`、`sbatch/`。设计包本身（`package/`）、包参考实现副本 `code/nr3/ref_init.py`、改写自未发布 NR2 训练器的 `code/nr3/train_prefix.py` 及其 diff 不发布。

## 范围说明

单训练 seed；配对差不给显著性结论。B 组为 12000 步总预算，不与纯 8000 步结果作等预算比较。Flickr、COCO、CC152K_RECOVERED 分开报告。DEV 在开发中被反复使用。
