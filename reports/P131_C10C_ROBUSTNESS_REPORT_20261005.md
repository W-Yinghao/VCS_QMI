# P131 — CIFAR-10-C-style corruption robustness of frozen representations (regenerated locally; reporting only) — report — 2026-10-05

Pre-registration `P131_C10C_ROBUSTNESS_PREREG_FROZEN_20261005.md`; jobs 1021699 (generation, CPU) and 1021700 (evaluation, GPU) exit 0; results-only
commit `092e86a` (`P131_c10c_results.{md,json}`, per-run files `reports/P131/`).  **CIFAR-10-C-style, regenerated locally from official test images
under the owner's reporting-only authorisation; severity parameters transcribed (unverified) → internal comparisons between our encoders only, never
against published CIFAR-10-C numbers.**  12 primary corruptions (15 minus motion_blur, snow, frost) × 5 severities; frozen-h linear head trained on
dev45k; clean reference = development-validation accuracy (recomputed per run = stored value for all 21); clean official-test accuracy never computed.

## 1. Mean corruption accuracy (mCA, primary) and per family

| method | seeds | clean dev-val | **mCA** | drop | noise | blur | weather | digital |
|---|---|---|---|---|---|---|---|---|
| A-P3 | 5 | 89.02 | **70.58** | 18.44 | 51.89 | 70.40 | 83.14 | 78.46 |
| SimCLR | 5 | 88.31 | **70.63** | 17.68 | 52.96 | 70.41 | 81.90 | 78.40 |
| G2 | 5 | 88.69 | 71.31 | 17.38 | 54.38 | 71.27 | 83.03 | 78.17 |
| JS-AP3 | 3 | 88.81 | **71.94** | 16.87 | 54.92 | 71.83 | 83.21 | 79.16 |
| recipe VCS | 3 | 87.01 | 66.23 | 20.78 | 46.34 | 65.25 | 80.24 | 74.88 |

## 2. Paired contrasts (frozen P114 labels)

| contrast | mCA | drop (clean − mCA) | family with a clear label |
|---|---|---|---|
| **A-P3 − SimCLR** (primary, 5 seeds) | −0.04 [−0.80, +0.71] **close** | +0.76 [+0.16, +1.36] clear (A-P3 drops more) | weather +1.24 clear (A-P3 better) |
| **A-P3 − JS-AP3** (primary, 3 seeds) | **−1.36 [−2.65, −0.07] clear (JS more robust)** | +1.57 inconclusive | noise −3.18 clear (JS better) |
| G2 − SimCLR (descriptive) | +0.68 inconclusive | −0.30 close | blur +0.85, weather +1.13 clear |
| A-P3 − recipe VCS (descriptive) | +4.35 clear | −2.33 inconclusive | all four families clear in A-P3's favour |

## 3. Reading
- **A-P3 vs SimCLR: equally robust on average (close).**  A-P3's clean-accuracy lead (+0.72) does not carry over to corrupted inputs: its drop is
  larger (clear), with a weather advantage and a non-significant noise deficit.
- **A-P3 vs JS-AP3: JS is more robust (clear, 3 seeds),** driven by the noise family (−3.18); weather is identical.  Same scorer, pairing and
  gradients — only the loss differs — so on CIFAR-10 the matched-JS loss gives representations whose linear readout degrades less under corruption,
  while clean accuracy is close (P114).  Descriptive, one corruption set with unverified parameters.
- A-P3 is far more robust than the recipe VCS (+4.35, every family), i.e. the A-P3 structure (fixed scorer, all-view tokens, full gradients) improves
  robustness along with clean accuracy.
- Not claimed: comparability with published CIFAR-10-C results; the three omitted corruptions; anything about CIFAR-100 or other readouts.
