# P113 — light evaluation batch: A-P3 in P109 / P110, and the P111 mechanism diagnostic — report — 2026-10-03

Pre-registration `P113_LIGHT_EVAL_BATCH_PREREG_FROZEN_20261002.md`; jobs 1018995–1018997 (A100), all exit 0; raw results committed (bfbf2ae), aggregates re-run
(`P109_v4_X_I_results.md`, `P110_v_results.md`).  Evaluation only.

## 1. Mechanism diagnostic (epoch 800, each run's own training augmentation)

| run (seeds) | positive-pair cosine | negative-pair cosine | zero-score threshold −b/a | positives below threshold |
|---|---|---|---|---|
| G2 std (0–2) | 0.907 / 0.907 / 0.907 | 0.004 | 0.50 (fixed) | < 5 % (q05 0.66) |
| G2 strong (0–2) | 0.867 / 0.867 / 0.866 | 0.027 | 0.50 (fixed) | < 5 % (q05 0.55) |
| recipe VCS std (0–2) | 0.974 / 0.973 / 0.974 | 0.772 | 0.90 (learned, a ≈ 24.5) | < 5 % (q05 0.94) |
| recipe VCS strong (0–2) | 0.984 | 0.888 | 0.95 (learned, a ≈ 28.4) | < 5 % (q05 0.96) |
| A-P3 std (0) | 0.888 | 0.003 | 0.50 (fixed) | < 5 % (q05 0.61) |

**Frozen reading: "not supported."**  G2-strong's positive cosine is lower on every seed (0.867 vs 0.907), but the fraction of positives below κ = 0.5 is
< 5 % in both conditions (5th percentile 0.55 vs 0.66), so positives do not fall below the fixed threshold; the hypothesised mechanism is not what happens.
**Descriptive finding (not pre-stated):** the two scorers produce entirely different geometries.  With the learned scale, the representation sits in a
narrow cone — negatives at cosine 0.77 (0.89 under strong augmentation), positives 0.97–0.98, threshold 0.90–0.95.  The fixed (2, −1) scale holds the
threshold at 0.5 and drives negatives to orthogonality (cosine ≈ 0.00–0.03).  Strong augmentation narrows the learned-scale cone further and lowers
G2's positive cosines; why that costs G2 accuracy is not answered here (P112 shows a different fixed κ recovers part of it).

## 2. A-P3 added to P110 (V1 / V2 / V3), seeds 0–2
- **V1 (CIFAR-10 labels 1 / 10 / 100 %):** A-P3 85.09 / 87.74 / 88.93 — on par with G2 (84.97 / 87.79 / 88.67), above SimCLR (83.92 / 87.07 / 88.29) at every
  fraction.  CIFAR-100 frozen transfer (100 %): A-P3 46.81, G2 47.63, recipe 52.70 (SimCLR 38.15, probe caveat of P110 applies).
- **V2 (development corruptions):** A-P3 mCA 63.55 ± 0.15 — same as G2 (63.50) and SimCLR (63.77); U2 remains most tolerant (65.09).
- **V3:** pooled over the now 15 encoders the conclusion is unchanged — no audit rule beats clean-accuracy selection (Δ vs A: VCS −0.43, JS −0.39, HSIC −0.22;
  all CIs below 0).

## 3. A-P3 added to P109 (X1 / I1)
- **X1:** picked measurement J 0.940 (z) / 0.943 (L2 h) vs its own training critic 0.862, linear 89.03 — the same pattern as G2 (0.933 / 0.945 vs 0.859):
  better downstream accuracy at a lower training score, while an independent refit recovers a strong pairing.
- **I1:** all A-P3 level cells ≤ 0.075 (nothing flagged).  Like G2, colour s 0.05 is detected in layer3 but largely removed from h (power 0.10 / 0.06 at
  n 2000); at colour s 0.2 h still detects it (1.00) but the prediction effect is small (Δ true-class probability −0.008, flip rate 0.047; recipe VCS
  0.097) — "retained but largely not used".  Blur σ 1.0 damages class evidence (accuracy 0.87 → 0.65) for A-P3 as for every encoder.
