# P86 — E-line report: the direct CS-estimator comparison on shared pair data (P85) — 2026-09-29

Pre-registration `P85_ESTIMATOR_BENCHMARK_PREREG_FROZEN_20260928.md` (frozen 2026-09-28T18:19:49Z; pilot cost appended 19:43Z); results-only commit 5858785
(`P86_p85_aggregate.{md,json}`, `P86_p85_aggregate_yaml.md`; 81 grid cells + 4 pilot cells, jobs 1013331 and 1013344–1013351, 85/85 cells present).
Reading rules 1–7 applied by their letter, then the descriptive findings.  Same-target rows are read on the selected configuration per cell (SELECT role);
VCS-N below means the K = 8 cyclic-shift construction (the recipe's), with the in-batch construction agreeing within seed sd in every cell; "kernel controls" =
S-KDE (common-risk bandwidth), S-Kernel (RFF, selected m / multiple; Nyström 512 centres), rLS-tanh.  Posterior MSE E_M(T − η)² is the primary quantity;
the trivial estimate T ≡ 0 has posterior MSE = S (0.540 on the pad axis, 0.802 on the N and batch axes).

## 1. Same-target error (rule 1): VCS-N is lower than every kernel control on every axis value
| axis | value | VCS-N | matched JS | S-Kernel RFF (selected) | rLS-tanh | S-KDE | Nyström |
|---|---|---|---|---|---|---|---|
| pad (d_total per side; S = 0.540) | 2 / 10 / 20 / 50 / 100 | 0.004 / 0.026 / 0.055 / 0.169 / 0.354 | 0.004 / 0.023 / 0.053 / 0.183 / 0.509 | 0.008 / 0.181 / 0.353 / 0.507 / 0.540 | 0.008 / 0.161 / 0.324 / 0.508 / 0.541 | 0.090 / 0.399 / 0.481 / 0.527 / 0.537 | 0.025 / 0.407 / 0.493 / 0.541 / 0.541 |
| N (d = 20, S = 0.802) | 256 / 1024 / 4096 / 16384 | 0.364 / 0.146 / 0.059 / 0.029 | 0.323 / 0.122 / 0.052 / 0.026 | 0.508 / 0.273 / 0.185 / 0.149 | 0.491 / 0.179 / 0.163 / 0.104 | 0.624 / 0.607 / 0.602 / 0.598 | 0.676 / 0.550 / 0.533 / 0.532 |
| dependence, gaussian (I nats) | 2 / 4 / 6 / 8 / 10 | 0.066 / 0.059 / 0.049 / 0.035 / 0.023 | 0.061 / 0.052 / 0.041 / 0.029 / 0.018 | 0.175 / 0.185 / 0.173 / 0.155 / 0.134 | 0.095 / 0.163 / 0.131 / 0.102 / 0.076 | 0.441 / 0.602 / 0.666 / 0.693 / 0.703 | 0.404 / 0.533 / 0.578 / 0.576 / 0.559 |
| dependence, cubic | 2 / 4 / 6 / 8 / 10 | 0.206 / 0.222 / 0.195 / 0.169 / 0.124 | 0.197 / 0.205 / 0.186 / 0.158 / 0.127 | 0.423 / 0.528 / 0.551 / 0.544 / 0.521 | 0.392 / 0.507 / 0.512 / 0.492 / 0.456 | 0.507 / 0.681 / 0.746 / 0.779 / 0.782 | 0.548 / 0.764 / 0.861 / 0.899 / 0.913 |
| dependence, xor mixture | 2 / 4 / 6 / 8 / 10 | 0.400 / 0.340 / 0.276 / 0.230 / 0.151 | 0.355 / 0.332 / 0.271 / 0.202 / 0.144 | 0.576 / 0.800 / 0.888 / 0.919 / 0.926 | 0.571 / 0.795 / 0.876 / 0.907 / 0.915 | 0.524 / 0.744 / 0.834 / 0.866 / 0.876 | 0.580 / 0.833 / 0.941 / 0.981 / 0.995 |
| batch, equal updates (B) | 64 / 256 / 1024 | 0.064 / 0.059 / 0.061 | 0.056 / 0.052 / 0.051 | 0.213 / 0.185 / 0.172 | 0.163 (B-free) | 0.602 (B-free) | 0.630 / 0.533 / 0.437 |
| batch, equal exposure (B) | 64 / 256 / 1024 | 0.062 / 0.059 / 0.064 | 0.055 / 0.052 / 0.057 | 0.145 / 0.185 / 0.276 | 0.163 | 0.602 | 0.404 / 0.533 / 0.661 |

Seed sd is ≤ 0.005 for the neural rows away from the trivial regime (≤ 0.05 at d_total ≥ 50 and on the xor setting) and ≤ 0.02 for the kernel rows.
**Verdict (rule 1): "better"** against each of S-KDE, S-Kernel (RFF and Nyström) and rLS-tanh on 5/5 pad values, 4/4 N values, 3/3 + 3/3 batch values and
15/15 staircase cells, every difference exceeding 2 pooled seed sd.  The kernel same-target estimators reach the trivial estimate (posterior MSE ≈ S) at
d_total ≥ 50, at N = 256, and beyond I ≈ 4 nats on the cubic and xor settings, where they are *worse* than trivial (posterior MSE > S: the tanh wrap of an
over-smoothed ratio pushes T away from η).  |J_eval − S| gives the same ordering (aggregate §"native error").

Descriptive, outside the rule: the matched balanced-logistic critic (JS, same network, same pairs) attains a 5–25 % lower posterior MSE than VCS-N in
most cells (e.g. 0.0255 vs 0.0292 at N = 16384; 0.018 vs 0.023 at I = 10 gaussian), within 1–3 seed sd; the two exceptions are d_total = 100, where VCS-N
keeps 0.354 against JS's 0.509 (VCS-N better by 3 seed sd), and the xor cells, where they tie.  The single-shuffle ("product") negative construction is
30–60 % worse than K = 8 shifts or in-batch pairs for both losses.

## 2. Sample efficiency (rule 2; reported, not thresholded)
Slope of log posterior MSE vs log N (N = 256 → 16384, d = 20, I = 4): VCS-N −0.61, JS −0.61, rLS-tanh −0.37, S-Kernel RFF −0.30, Nyström −0.06, S-KDE ≈ 0.
No method's error at N ≤ 4096 is within 2 sd of its N = 16384 value: the neural critics halve their posterior MSE per 4× samples throughout and are still
sample-limited at 16 384 (the P82 reading); the kernel-feature read-outs improve 2–3× slower; the KDE plug-in does not improve with N at d = 20.

## 3. Irrelevant dimensions (rule 3)
Ratio posterior MSE(d_total = 100) / (d_total = 2): VCS-N 82, JS 141, S-Kernel RFF 68 (capped by the trivial ceiling), rLS-tanh 65, S-KDE 6 (already 0.09 at
d_total = 2), Nyström 22.  By the pre-stated thresholds **every method "degrades"** (ratio ≥ 4); the rule does not separate them.  The informative quantity is
the fraction of the posterior variance still recovered, 1 − MSE / S:

| d_total per side | 2 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| VCS-N | 0.99 | 0.95 | 0.90 | 0.69 | 0.35 |
| JS | 0.99 | 0.96 | 0.90 | 0.66 | 0.06 |
| S-Kernel RFF | 0.98 | 0.67 | 0.35 | 0.06 | 0.00 |
| rLS-tanh | 0.98 | 0.70 | 0.40 | 0.06 | 0.00 |
| S-KDE | 0.83 | 0.26 | 0.11 | 0.03 | 0.01 |
| Nyström | 0.95 | 0.25 | 0.09 | 0.00 | 0.00 |

With two signal coordinates per side and N = 4096, the kernel same-target estimators lose two thirds of the signal by 10 padded coordinates and all of it by
50; the neural critics keep 90 % at 20 and two thirds at 50.  At 100 padded coordinates VCS-N is the only estimator still above the trivial estimate.

## 4. Batch (rule 4)
VCS-N, equal updates: B = 64 differs from B = 256 by more than 2 sd (0.064 vs 0.059, worse), B = 1024 does not (0.061).  Equal exposure (512 000 positive
pairs): B = 64 does not differ (0.062), B = 1024 does (0.064, worse: 500 updates).  The batch effect is ≤ 10 % either way, an order of magnitude below the
N effect.  S-Kernel RFF under the same trainer: equal updates 0.213 / 0.185 / 0.172, equal exposure 0.145 / 0.185 / 0.276 — the read-out is optimisation-
limited at 2 000 updates and profits from 8 000 small-batch updates.  The closed-form rows repeat exactly across B (consistency check passed).

## 5. Dependence resolution across targets (rule 5): |Δmean| / pooled bootstrap sd between adjacent levels (P(V_{l+1} > V_l) = 1.00 unless stated)
| estimator | gaussian 2→4 / 4→6 / 6→8 / 8→10 | cubic | xor mixture |
|---|---|---|---|
| VCS-N (cyclic 8) | 88 / 57 / 41 / 16 | 45 / 21 / 12 / 16 | 9.9 / 4.8 / 2.7 / 3.8 |
| JS (cyclic 8) | 63 / 45 / 34 / 20 | 59 / 19 / 8 / 7 | 14 / 5.1 / 3.4 / 3.0 |
| InfoNCE, in-batch (bound log 256 = 5.5 nats) | 118 / 114 / 81 / 52 | 57 / 48 / 40 / 25 | 37 / 36 / 29 / 20 |
| InfoNCE, cyclic 8 (bound log 9 = 2.2 nats) | 80 / 38 / 17 / 17 | 77 / 44 / 22 / 14 | 78 / 39 / 16 / 6.6 |
| DV / MINE, in-batch | 121 / 112 / 66 / 31 | 42 / 28 / 10 / 8 | 16 / 8 / 8 / 3.1 (P 0.99) |
| NWJ, in-batch | 119 / 52 / 38 / 11 | 54 / 35 / 28 / 7 | 18 / 8 / 5 / 1.8 (P 0.93) |
| DV, NWJ with one shuffled negative | fail beyond I = 4 (P 0.45–0.78, ratios < 1) | fail | fail |
| S-Kernel RFF | 84 / 35 / 16 / 11 | 48 / 13 / 7 / 4.5 | 7.1 / 4.3 / 2.3 / 1.3 (P 0.80) |
| rLS-tanh | 47 / 43 / 19 / 12 | 8 / 14 / 8 / 4.5 | 2.5 / 2.1 / 3.2 / 1.2 (P 0.76) |

Every neural estimator with K = 8 or in-batch negatives orders all adjacent levels with probability 1.00 on all three settings.  In-batch InfoNCE resolves
adjacent levels 1.3–6× more finely than VCS-N or JS at every level, including above its log(K + 1) bound (its saturated value still moves monotonically with
tiny variance); VCS-N and JS are within a factor 1.5 of each other everywhere (VCS-N finer at low I on gaussian, JS finer at high I).  The single-negative
MINE / NWJ estimators lose resolution beyond I ≈ 4 nats through rare-event variance, as the owner's oracle table predicts; MINE with a single negative also
produced 1 981 non-finite training steps.  The kernel same-target estimators resolve gaussian steps 1.5–4× more coarsely than VCS-N and fail at the top of
the xor staircase.  Nothing here converts a target into another; the raw errors of MI estimators are not compared with S errors.

## 6. Cost (rule 6; d = 20, N = 4096, B = 256, one H100/A100/L40S GPU)
Fit seconds: VCS-N cyclic 8 ≈ 10–17 s per 2 000 updates (in-batch 6–8 s, single negative 3–5 s); JS the same; InfoNCE / NWJ / DV the same within 10 %;
S-Kernel RFF 1.6–2.9 s, Nyström 1.9–3.6 s; rLS 0.1–0.4 s and S-KDE 0.1 s (closed form) but with 2.3–3.4 GB for the 4 096-feature Gram (rLS) and O(N²)
kernel matrices; CS-K-native 0.02 s / 0.4 GB (O(N²)).  In-batch neural rows hold a B × B pair matrix (2.2–3.8 GB); cyclic-shift rows use 0.13–0.19 GB.

## 7. Own-target failure of the classical plug-in: CS-K-native reads ≈ 0
With the CS-IB bandwidth convention (multiple × FIT median distance), the plug-in kernel CS-QMI at d = 20 (joint 40 dimensions), N = 4096 is 0.30 / 0.015 /
0.001 / 0.0001 for multiples 0.25 / 0.5 / 1 / 2 against its own Lebesgue truth 2.28 (I = 4), and 0.41 / 0.029 / 0.002 / 0.0001 against 6.56 (I = 10); at
d_total = 2 the multiple-1 value is 0.03 against 1.07.  The statistic is monotone in I at the narrowest multiple but recovers ≤ 13 % of its target: at these
dimensions the fixed-measure kernel CS is not a quantitative estimator, and any dependence it reports is bandwidth-dominated.  This is the estimator that
enters P87's Table B as a training objective on 128-dimensional projector outputs; P87 measures whether its *gradient* still trains a useful representation,
which this table does not decide.

## 8. QC and numerical failures
Oracle J on TRUTH within 3 se of S in 85 / 85 cells; RuLSIF identity residual < 1e-9 on TUNE in every cell; no missing cell on any axis; the CS-K-native log
guard never fired on the selected multiple.  Two rows are void and must not be read: (i) **SMILE** (both constructions): values 6e2–4e7 growing with the
learning rate — our τ = 5 clipping is not effective in the implementation (`vcs_estim.estimators`, kind "smile"); a corrected SMILE is an addendum, not a
change to any other row; (ii) **MINE/DV with a single shuffled negative**: 1 981 non-finite training steps (exp overflow), retained only as the failure it is.
The d_total = 100, N = 256 pilot rows are the trivial critic (SELECT kept update 0), recorded as such.

## 9. What can and cannot be written
- Can: on the same target S, the neural VCS critic has a lower posterior MSE than kernel-density, kernel-feature (RFF / Nyström) and RuLSIF-type
  same-target estimators at every dimension, sample size, batch and dependence level tested, with the gap opening from 10 irrelevant coordinates or
  1 024 samples on; the kernel estimators collapse to the trivial estimate where the neural critic still recovers two thirds of the posterior variance.
- Can: the classical plug-in kernel CS with the median-heuristic bandwidth is not quantitative at d = 20.
- Cannot: that VCS is more sample-efficient or more accurate than the matched balanced-logistic (JS) critic — JS is equal or slightly better in posterior MSE
  in all but the 100-padded-coordinate condition (single condition; descriptive).
- Cannot: that VCS resolves dependence levels more finely than in-batch InfoNCE — the opposite holds at every level; VCS ≈ JS.
- SMILE numbers: none until the addendum.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: direct_cs
protocol_id: P85_estimator_benchmark
source_commit: d545e38 (frozen), 2669701 (_peak_mem fix); results 5858785
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural | js | infonce | nwj | dv | smile(void) | kernel_cs_native | kernel_S (S-KDE, RFF, Nystrom) | rLS / rLS-tanh
estimand: S (same-target rows) | native_CS | JS | InfoNCE_B | MI
evaluation_readout: J_common, posterior MSE, native error, ordering probability, fit seconds, peak memory
loss_scale: 1
reference_measure: mixture_equal (S rows) | native_base (CS-K-native)
critic_class: joint MLP 256-256 tanh/native output | RFF-tanh | Nystrom-tanh | KDE ratio | ridge (raw, tanh-wrapped)
gradient_routing: full
n_independent_units: FIT = N per cell; EVAL 32768 blocks; TRUTH 200000; 3 seeds
n_positive_pairs: N per cell; n_negative_pairs: 8N (cyclic) | B(B-1) per batch (in-batch) | N (product)
split_manifest_hash: seeded role streams recorded per cell JSON
status: complete (SMILE rows void; addendum pending)
```

## Addendum (2026-09-30) — SMILE rows after the implementation fix (P85 addendum 2; job 1015177; `P86_smile_fix_aggregate.md`)
The voided SMILE rows (§8) were re-run on all 85 cells with the reference implementation (clipped-DV value, τ = 5; JS training gradient).  All values are
finite now.  Readings under the same rules (cross-target: resolution and native error on the MI scale only):
- **In-batch SMILE** behaves like the other in-batch MI estimators: native error 0.03 → 1.41 nats over d_total 2 → 100 (InfoNCE in-batch 0.02 → 1.40,
  DV in-batch 0.02 → 1.15); 1.85 → 0.09 nats over N 256 → 16 384.  It orders every adjacent staircase level with probability 1.00 on all three settings;
  resolution |Δmean| / sd on the gaussian staircase 61 / 42 / 14 / 7 — between VCS-N (88 / 57 / 41 / 16) at high I and below in-batch InfoNCE (118 / 114 / 81 / 52).
- **Single-negative SMILE** is biased upward by several nats (native error 11.1 at I = 4, d = 20, N = 4 096) and loses resolution at the top of the gaussian and
  xor staircases (P 0.43–0.78) — the known behaviour of a DV partition term with one negative and a τ = 5 clip.
- Nothing in §§1–7 changes: SMILE carries no posterior and enters no same-target comparison.  The §9 statement on dependence resolution becomes: in-batch
  InfoNCE resolves adjacent levels most finely; VCS-N, JS and in-batch SMILE are within a factor ≈ 2 of each other.
