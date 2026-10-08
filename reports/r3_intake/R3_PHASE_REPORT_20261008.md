# r3 phase report — CVPR-MV6-THEORY-20261007-r3 — 2026-10-08

All r3 measurement units are complete (CPU only; no pretraining).  Intake `R3_DELTA_INTAKE_20261008.md`; analysis `R3_RESULTS_ANALYSIS_20261008.md`.

| r3 id | P | status | report | one-line result |
|---|---|---|---|---|
| R3-E0 | P151 | done | `P151_E0_THEORY_EVIDENCE_REPORT_20261008.md` | on S, matched JS ≥ VCS in precision at every ladder level; in-batch MI methods order the ladder with 0 failures; plug-in beats J in 69–70 / 72 synthetic cells |
| R3-E1 | P152 | done | `P152_R3_E1_REPORT_20261008.md` | bounded objectives + InfoNCE stable, NWJ / DV / SMILE not, on gradient CV, eval variance, histograms; InfoNCE lowest gradient CV; VCS least saturated, no score overflow; 0 non-finite steps in 54 units; fixed budget overfits at I 2 for all |
| R3-V0 / V1 | P149 | done | `P149_R3_TWOVIEW_REPORT_20261008.md` | real two-view relation: all 12 encoder × estimator cells numerically resolvable along the channel; J coordinate-stable to ~1 SE; VCS ≈ JS routes, RFF function-class-limited; S_plugin unusable on real images |
| R3-C0 | P150 | done | `P150_R3_C0_REPORT_20261008.md` | colour 0.1 detected at h (0.60 vs 0.05) with no estimable J for either critic class: P143 is a detection table, P144's J ≤ 0 is not a class limitation |
| R3-A0 | P142 / P143 | reused | `manuscript_v6_handoff/phase_report.md` | unchanged; P143 now read as detection only (P150) |
| R3-S0 | P130 / P138 / P145 add. 1; P147 / P148 | running / closed | `P147…`, `P148…` reports | batch and weight-decay axes closed (no cell promising); confirmations running |

## Theory-to-evidence ledger after r3 (O1 §12 metrics)
| metric | evidence now | status |
|---|---|---|
| estimator mean / sd vs dependence (§12.2, §17.5) | P151 precision panel (same target S) + native-target panel; P152 at I 2 / 6 / 10 | complete on synthetic; the "decisive figure" should plot same-target precision and the P152 stability fields side by side, MI methods on their own targets |
| gradient-norm variance | P152 | complete — InfoNCE ≤ JS ≤ VCS ≪ DV / NWJ (CV); VCS is not uniquely the most stable |
| batch / lr sensitivity | P85 axes (P151 `axes_N_batch.json`) | complete |
| critic-output histograms | P152 (on each method's T coordinate; raw f overflow) | complete — VCS least saturated, no overflow at its registered lr |
| NaN / failure rate | P151 + P152: 0 non-finite steps for every in-batch method; product DV / NWJ (P86) are the failures | complete |
| wall-clock | P85 / P86 fit seconds | complete |
| exact gap, J vs plug-in | P108 (synthetic, plug-in better when calibrated) + **P149 (real images: plug-in inflates under weak signal)** | complete; the paper must state both |
| resolution | synthetic ladders (P151) + **real two-view channel (P149: resolvable for every estimator)** | complete |
| invertible invariance / DPI | P108 E2 (synthetic) + **P149 orthogonal refit (drift ≤ ~1 SE)** + channel ordering (DPI direction respected in all cells) | complete |
| detection vs estimation | **P150** | new caveat for the paper |

## Writing items for the owner (cumulative)
W1–W5 (r2), W6 InfoNCE EVAL bound log 1024, W7 no "lower variance than MI estimators" claim (now backed by P152: InfoNCE has the lowest gradient CV),
**W8** S_plugin is a calibrated-critic quantity: on real images report J (fit-limited lower reading) and treat S_plugin as context (P149),
**W9** detection power ≠ estimable dependence: P143 power 0.60 coexists with J ≈ 0 (P150), **W10** reported estimator values depend on held-out
early stopping at low dependence for every method (P152 §3).

## What remains (my judgement)
No further r3 measurement is needed for the plan's metrics.  Optional, only if the owner wants them in the paper: (i) P149 at the z-site and a second
encoder seed (≈ 1 h CPU per fixture); (ii) P152 at a larger N to see whether the I 2 overfitting persists (cheap).  Neither is submitted.

## Addendum (2026-10-08, P149 add. 1)
z-site (seeds 1–2) and seed-2 encoders at h: resolution holds in 47 / 48 new cells; J coordinate-stable to ~2 SE; the P149 model observation is
seed-stable at h.  **W11**: J at z exceeds J at h by 0.10–0.17 — fitted J is not a data-processing quantity across sites / dimensions; only the
within-site channel ladder supports a DPI-direction reading (`P149_ADDENDUM1_ZSITE_SEED2_REPORT_20261008.md`).
