# r6 synthesis (R6-06) — what the two natural-data routes established, and the scope of the additional contribution — 2026-10-12

Package `r6/package_r6_20261011` (new applications).
- **R6-00 to R6-04:** intake, research cards, controlled checks, data audit and route decision.
- **R6-05:** REG as the selected route (REG-P1 translation, REG-P2 rigid), plus GAIN as the second route under the owner's "complete every
  task" instruction (AV-MNIST, CREMA-D).
- **Not run (stay cards):** NUISANCE, COMPRESS, AMORTIZE.
- **Discipline:** every unit was frozen before its results; DEV only; no official test read.

## 中文摘要（给负责人）
- **REG（跨模态配准，BF/SHG）**
  - 学习得到的局部关系critic（VCS、logistic、InfoNCE三种目标）构成的配准能量，在平移与±30°刚体两种设定下都比逐图MI/NMI多配准约18–20个百分点（<42 px）。
  - 与CoMIR表示（同一优化器）在<42 px上无可检测差异，在<9 px精度上更好（+6～+11点）。
  - VCS与logistic、InfoNCE无可检测差异：这是"学习critic类"的效应，不是VCS特有的结论。
- **GAIN（按需计算额外模态）**
  - 直接回归实际收益h的策略优于熵和随机。CREMA-D上明显（−0.02 Brier），AV-MNIST上很小。
  - 平方分歧z作为目标不安全：CREMA-D的Brier族中z显著劣于h，与受控验证的警告一致。
  - CREMA-D上真实墙钟成本：25 %预算时约为"总是计算视频"成本的29 %。
  - 两者都是标准风险回归，不是CS特有方法。
- **数据发现：** MultiBench AV-MNIST的6万条音频只有1601个唯一片段（每个重复约37次，标签纯度0.44），原图像分组的划分会把音频"记忆"误当作证据。

## 1. REG — learned local-relation energy for cross-modal registration
| DEV success (%) | translation (P1) < 42 / < 9 px | rigid ±30° (P2) < 42 / < 9 px |
|---|---|---|
| VCS critic | 100.0 / 75.0 | 94.2 / 68.2 |
| logistic critic | 98.9 / 65.8 | 93.1 / 64.9 |
| InfoNCE critic | 95.2 / 57.4 | 94.9 / 59.6 |
| CoMIR features + same optimiser | 90.9 / 63.9 | 90.0 / 62.0 |
| MI / NMI (per image) | 81.9 / 55.6 ; 80.3 / 55.3 | 74.1 / 49.4 ; 75.9 / 50.6 |
| closed-form RFF kernel critic | 57.2 / 36.2 | 42.2 / 24.4 |

**Established (both intervals, seeds and images, exclude 0):**
- VCS − MI: +18.1 (P1) and +20.1 (P2) points at < 42 px.  The same holds against NMI and the kernel.
- VCS − CoMIR at < 9 px: +11.1 (P1) and +6.2 (P2).

**Not established:**
- VCS − CoMIR at < 42 px: +9.1 / +4.2, but one interval reaches 0 in each pilot.
- Any VCS-specific advantage: VCS ≈ logistic ≈ InfoNCE at < 42 px in both pilots (frozen wording: a learned-critic class effect).

**What this adds:**
- Every member of the CS / logistic / InfoNCE critic family yields a registration energy that is more reliable than per-image MI / NMI on a
  hard multimodal pair (BF / SHG).
  - Its failures are small, while MI's are catastrophic (p90 15–21 px vs about 170–270 px).
  - It transfers to rotations it never saw in training.
- The earlier negative (P60: raw-pixel closed-form VCS worse than MI on RGB-NIR) is reproduced here by the closed-form kernel critic.  What works is
  a learned critic, not the closed-form estimator on raw patches.

**Pending:** the CoMIR warp-then-encode sensitivity rows (REG-P2 addendum).

**What a stronger claim would need:**
- A second modality pair: the EPFL RGB-NIR and Zurich data are downloaded.
- The native CoMIR pipeline (SIFT / α-AMD).
- Larger transform families.

## 2. GAIN — on-demand computation of the extra modality
| | AV-MNIST (audio branch; routing quality only) | CREMA-D (video branch; cost-bearing) |
|---|---|---|
| always − never (Brier) | −0.050 / −0.054 | −0.143 / −0.137 |
| gain_h − entropy (budget mean) | −0.0028 [−0.0056, −0.0002] / contains 0 | −0.020 [−0.028, −0.014] / −0.019 [−0.026, −0.011] |
| gain_z − gain_h | contains 0 / contains 0 | +0.0055 [+0.0008, +0.0110] / contains 0 |
| corr(score, realised gain h) | 0.08–0.11 for every policy | gain_h 0.16, gain_z 0.14–0.15, entropy 0.05 |
| cost | microseconds; no claim | gain_h at 25 % budget: Brier 0.564 at 9.7 ms per clip (29 % of always's 33.3 ms) |
(Brier family / CE family; two-way bootstrap over seeds × clip or actor clusters.)

**Established:**
- Regressing the realised improvement h ranks acquisitions better than cheap uncertainty and random.  This holds clearly on CREMA-D, weakly
  on AV-MNIST.
- The squared disagreement z (the evidence-gain proxy) can be strictly worse than h (CREMA-D, Brier family).
- On CREMA-D the routing saves real wall-clock at batch 1, with decoding included.

**Not established:**
- That z is unsafe in every setting.  AV-MNIST and CREMA-D's CE family contain 0.
- Any CS-specific advantage.  Both targets are standard risk regression (frozen wording).

**What this adds:**
- A natural-data confirmation of the controlled warning (R6-02G): the evidence-gain identity h − z = 2 (b − a)·(e − b) holds exactly, but with
  finite models z overstates the gain about 2×.
- A deployable policy should target h.
- A side finding about the AV-MNIST benchmark itself: clip reuse and a noisy pairing.  This matters for anyone using it.

## 3. Scope recommendation for the paper (owner's decision)
- **REG:** a positive application result for the learned-critic family, stated as a class effect.  It does not carry the VCS objective
  specifically.  Candidate text: "a CS-family learned local-relation critic yields a more reliable multimodal registration energy than mutual
  information".
- **GAIN:** a clarifying, partly negative result about the evidence-gain proxy, with a positive deployment demonstration on CREMA-D.
- **Neither result supports a claim that VCS outperforms its logistic / InfoNCE relatives.**  This agrees with the VL findings (R4 / R5), where
  the deficits and gains were properties of the critic class and the normalisation, not of the VCS estimand.

## 4. Records
| unit | protocol | results-only | report |
|---|---|---|---|
| REG-P1 | R6_REG_P1_PROTOCOL_FROZEN_20261011.yaml | f4a8ef8, 26bd7a8 | 6cb7fb8 |
| REG-P2 | R6_REG_P2_PROTOCOL_FROZEN_20261012.yaml | 4120ca3 | 335bce1 |
| GAIN-P1 AV-MNIST | R6_GAIN_P1_PROTOCOL_FROZEN_20261012.yaml | 8abc5aa | 79fb495 |
| GAIN-P1 CREMA-D | R6_GAIN_P1_CREMAD_PROTOCOL_FROZEN_20261012.yaml | 69e283b | 404a966, 29c3665 |
