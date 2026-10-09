# VL1-13 — Table A on ReCLIP's own isolation features — report — 2026-10-09

Protocol `VL1_13_RECLIP_FEATURES_FROZEN_20261009.md` (gate: the cache reproduces ReCLIP IPS-only on all 6 442 DEV queries, argmax agreement
99.71 %).  Results-only commit `99ff233` (`VL1_13_table.json`, `VL1_13_reclip_minus_clip.json`, `VL1_13_strata.json`; job 1031677).  N all × 3 seeds;
roles / law / F2r / selection as VL1-10.  The raw row **is** ReCLIP IPS-only (70.81 image-macro, reproduced inside the fitter).

## 1. Results (DEV image-macro Top-1 %, CAL common J × 100)
| route | Top-1 | CAL J | − raw (= ReCLIP IPS), paired |
|---|---|---|---|
| raw cosine = ReCLIP IPS-only (no training) | 70.81 | — | — |
| **VCS (estimator-selected)** | **74.19 ± 0.15** | **10.07** | **+3.38 [+3.01, +3.75]** |
| matched JS (estimator-selected) | 73.17 ± 0.37 | 9.58 | +2.36 [+1.45, +3.27] |
| RFF ridge-tanh | 70.82 ± 0.11 | 6.40 | +0.01 [−0.26, +0.29] |
| candidate softmax (DEV-selected, optimistic) | 75.53 ± 0.24 | — | +4.72 [+4.12, +5.32] |
| *ReCLIP official (IPS + relation heuristics; Table B)* | *73.18* | — | — |

## 2. Frozen readings
1. **Learning value on the best public isolation input:** the VCS critic adds +3.38 Top-1 over ReCLIP IPS-only on exactly the same features
   (interval far above 0); JS adds +2.36.  The kernel route adds nothing.  The CAL-selected VCS critic (74.19) is above ReCLIP official (73.18),
   which adds hand-written spatial-relation parsing tuned on val.  The critic uses the FIT annotations and ReCLIP uses none, so this is a placement,
   not a like-for-like comparison.
2. **VCS − JS: +1.02 Top-1 [+0.08, +1.96] (task-selected +0.83 [+0.17, +1.49]); CAL J +0.49 [−0.07, +1.06].**  This is the first feature set where
   the Top-1 gap exceeds the pre-stated "close" band (|Δ| < 0.5) with an interval excluding 0.  I do not read it as an objective effect:
   both routes picked lr 2e-3, the edge of the grid, so the difference may be optimisation-limited.  It is reported as observed and re-examined by
   VL1-14 (which screens the optimisation).  On the four other feature sets VCS − JS stays within ±0.5.
3. **Estimator panel:** CAL J 10.07 (VCS) is the second highest of the five feature sets (SigLIP 2 11.63 > ReCLIP 10.07 > FG-CLIP v1 9.79 > FG-CLIP 2
   8.76 > CLIP 7.20).  This is the same order as the VCS critic's learned Top-1 on them (76.00 > 74.19 > 73.78 > 72.51 > 68.80), on all five.
4. Strata: VCS − raw positive in both category strata (same-category +3.28, other-category +3.58) and both length strata.  Softmax − VCS +1.27
   (query-weighted), the smallest task-loss margin of all feature sets.
