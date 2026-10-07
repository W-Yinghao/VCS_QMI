# P151 (r3 R3-E0) — the estimator evidence re-aggregated into the panels the original plan asks for — report — 2026-10-08

Source: r3 package `CVPR-MV6-THEORY-20261007-r3` §5; original plan [O1] §12.2 and §17.5.  Analysis only (`scripts/p151_e0_aggregate.py`, from the raw
`outputs/P85_estim_benchmark/*.json`, the corrected SMILE files and `outputs/P108_estim/*.json`; 81 full-method cells; registered selections).  Tables:
`reports/P151_E0/tables.md`; machine-readable panels `e_precision.json`, `e_stability.json`, `e_resolution.json`, `axes_N_batch.json`, `p108_j_vs_plugin.json`,
`field_matrix.json`.  Nothing was refitted.

## 1. E-Precision — Gaussian ladder (d 20, N 4096, B 256, U 2000; 3 seeds), posterior MSE on the shared target S
| I | S | VCS in-batch | matched JS in-batch | VCS product | JS product | S-KDE | RFF (sel.) | Nyström | rLS-tanh |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.567 | .065 ± .003 | **.061 ± .005** | .112 | .113 | .441 | .174 | .404 | .095 |
| 4 | 0.803 | .060 ± .001 | **.052 ± .002** | .095 | .089 | .602 | .185 | .533 | .163 |
| 6 | 0.913 | .050 ± .000 | **.041 ± .002** | .069 | .062 | .666 | .173 | .578 | .131 |
| 8 | 0.963 | .037 ± .001 | **.029 ± .001** | .046 | .042 | .693 | .155 | .576 | .101 |
| 10 | 0.985 | .027 ± .005 | **.018 ± .001** | .032 | .027 | .703 | .134 | .558 | .076 |

Native targets (own value, |error to own truth|): in-batch InfoNCE 1.76 / 3.54 / 4.99 / 5.97 / 6.53 for MI 2 / 4 / 6 / 8 / 10 (errors .24 → 3.47; the
EVAL bound is log 1024 = 6.93, EVAL blocks of 1024 — **P86 §5 wrote log 256**); NWJ in-batch 1.76 … 7.72 (.24 → 2.28); DV in-batch 1.77 … 8.33 (.23 → 1.67);
corrected SMILE in-batch 1.67 / 4.27 / 7.80 / 11.35 / 13.39 (overshoots from I 6); product-sampled NWJ / DV diverge at I ≥ 6 (−60.9, −1.5e8; −11.4 … −21.4).
Cubic and xor ladders in `tables.md` (xor: every same-target method ≥ .14 posterior MSE at I 10; kernels fail entirely; VCS / JS seed sd .02–.07).

## 2. E-Stability (Gaussian I 6; seed sd of the value / eval bootstrap SE / SELECT-curve tail sd / non-finite steps / fit s)
VCS in-batch .0014 / .0013 / .005 / 0 / 9.8; JS in-batch .0016 / .0023 / .011 / 0 / 9.5; InfoNCE in-batch .0043 / .0102 / .0095 / 0 / 9.5;
NWJ in-batch .051 / .013 / .098 / 0; DV in-batch .0078 / .013 / .042 / 0; SMILE in-batch .12 / .032 / .13 / 0; product NWJ 109 / 59 / 2e28 / 0;
product DV 3.3 / 2.1 / 3.5 / 0.  Kernels: seed sd .003–.013, no training curve.  Reading: the two balanced-posterior fits (VCS, JS) have the smallest
seed and evaluation spread, but the values live on different scales (S ∈ [0, 1] vs nats), so this panel is **not** a cross-target ranking;
in-batch InfoNCE / DV / NWJ are numerically stable (0 non-finite steps), the product-sampled DV / NWJ are the unstable rows.  Gradient-norm variance
(the §12.2 field) is not in these records → P152.

## 3. E-Resolution (adjacent ladder levels; P(value_{k+1} > value_k) from seed-coupled bootstrap draws; |Δ| / pooled sd)
Gaussian: 1.00 for every in-batch method at every step (standardised gaps 6–94; smallest at 8 → 10: NWJ 6.2, VCS 8.7, SMILE 13.4, DV 23.4, InfoNCE 28.7).
xor: 8 → 10 (ΔS .014): VCS in-batch 0.89 (1.0 sd), product VCS 1.00 (2.8), InfoNCE 1.00 (15.5), NWJ 1.00 (2.6), DV 1.00 (5.3), SMILE 1.00 (6.2).
Reading: on the synthetic ladders every multi-negative method, MI-based or not, resolves adjacent levels; bounded S is **not** uniquely resolving here.
Resolution on a real relation is the open question (P149).

## 4. P108 — J vs plug-in (unchanged caveat)
Plug-in S has smaller |bias| in 70 / 72 and smaller RMSE in 69 / 72 cells; e.g. C1 N 16 384 VCS: J −0.041 / 0.042 vs S_plug −0.015 / 0.020; at N 256 both
≈ −0.55 (fit-limited).  Raw J is a fit-limited lower estimate, not a per-draw lower bound.

## 5. Field matrix → what P152 adds
Present: oracle S / MI, 200-draw bootstraps, SELECT curves, costs, saturation summaries.  Absent (never written): selected weights, per-sample T,
per-update critic gradient norms, output histograms, repeated independent EVAL batches.  P152 records all five on the 3-level ladder.

## 6. Writing items (owner)
W6 InfoNCE bound log 1024 (EVAL) vs log 256 (training batch).  W7 no sentence may read "VCS has lower variance than the MI estimators": the same-target panel
shows JS ≥ VCS on S, and the native-target panel shows the in-batch MI methods ordering the ladder with 0 failures; the defensible statement is that the
bounded quadratic fit has small seed / evaluation spread *and* the best same-target precision together with matched JS, while single-negative DV and
product-sampled NWJ / DV diverge.
