# P124 — CIFAR-100 granularity: coarse 20-way, fine 100-way, conditional 5-way (v6 V6-GRANULARITY) — report — 2026-10-03

Pre-registration `P124_V6_GRANULARITY_PREREG_FROZEN_20261003.md`; job 1020312 exit 0; results-only commit `07e5286` (`P124_v6_granularity_results.{md,json}`).
Frozen CIFAR-100 from-scratch SSL encoders (recipe VCS P91, G2 P111, A-P3 P107, SimCLR P91; seeds 0–2), selection split, official test never opened.  Anchor check:
the recomputed fine `recipe_raw` equals the stored training-evaluation linear accuracy within ±0.04 for all 12 encoders.  Labels are used for downstream
evaluation only (never in SSL).  Claim labels per v6 §11.2: accuracies = **observed**; decomposition = **identity**; any "why" = **hypothesis**.

## 1. Primary readout (`recipe_raw`), mean ± sd over 3 seeds

| task | A-P3 | G2 | recipe VCS | SimCLR |
|---|---|---|---|---|
| coarse 20-way | **71.82 ± 0.27** | 70.91 ± 0.35 | 68.77 ± 0.73 | 70.70 ± 0.53 |
| fine 100-way | 60.00 ± 0.25 | 59.93 ± 0.50 | 59.91 ± 0.20 | 58.25 ± 0.34 |
| conditional 5-way (coarse given) | 76.34 ± 0.51 | 75.87 ± 0.40 | **76.61 ± 0.16** | 74.90 ± 0.18 |

## 2. Seed-paired contrasts and the frozen pattern rule

| contrast (primary) | Δ coarse | Δ fine | Δ conditional | **pattern (frozen rule)** |
|---|---|---|---|---|
| A-P3 − recipe VCS | **+3.05 [+0.62, +5.47] clear +** | +0.09 close | −0.27 close | **coarse-specific gain** |
| G2 − recipe VCS | +2.14 [−0.51, +4.79] inconclusive | +0.02 close | −0.74 inconclusive | no pattern (inconclusive at 3 seeds) |
| A-P3 − SimCLR | +1.12 inconclusive | **+1.75 [+1.16, +2.35] clear +** | +1.44 inconclusive | no pattern (inconclusive at 3 seeds) |

**Robustness across readouts (not switched; reported):** A-P3 − recipe Δcoarse is "clear +" under **all five** readouts (+2.57 to +3.05), while Δconditional is close
(recipe_raw, kNN) or negative (raw_std −1.45 clear −; raw_unstd, l2_std inconclusive negative).  G2 − recipe shows the same direction (Δcoarse +1.3…+2.5, clear +
under raw_unstd and kNN; Δconditional −0.7…−1.35, clear − under three readouts).  A-P3 − SimCLR: fine is clear + under the four linear readouts but clear − under
kNN (−1.41) — the same linear/kNN disagreement as P107 layer 2.

## 3. Reading
- **Observed:** on CIFAR-100, the fixed-scale VCS encoders (A-P3, and in the same direction G2) organise the **superclass structure** markedly better than the
  learned-scale recipe (+3 points coarse accuracy), while **within-superclass** discrimination is unchanged or slightly worse.  The two effects cancel in the
  100-way task (Δfine ≈ 0), which is why the CIFAR-10 linear gain did not appear as a CIFAR-100 fine-accuracy gain (P107 layer 2, P111).
- The 100-way head decomposition agrees (identity): A-P3 implied-coarse 73.49 vs recipe 72.76, but fine-given-coarse 81.65 vs 82.33.
- **Hypothesis to test (not a finding):** the fixed scale with negatives pushed toward orthogonality (P113 / P115 geometry) separates coarse groups but compresses
  fine distinctions inside a group; a scorer that preserves within-group geometry — e.g. a different κ, the P126 curvatures, or the per-dataset tuning of v6 §8 —
  would be the test.  P124 does not show which representational property causes it; no information-content claim.
- Not claimed: the official test set, other datasets, CIFAR-10 → CIFAR-100 transfer (P119 separate), "information lost".
