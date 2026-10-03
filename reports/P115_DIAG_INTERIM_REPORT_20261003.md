# P115 diagnostics — interim report (job 1: eight existing standard / strong-augmentation runs) — 2026-10-03

**Interim.**  Job 2 (the four P115 crop-only / jitter-only runs) is pending (orchestrator unit `P115_diag_job2`); this report covers job 1 only.
Pre-registration `P115_DIAG_PREREG_FROZEN_20261003.md` (descriptive only, no pre-written mechanism).  Job 1020003 (CPU), exit 0, 28 checkpoints.
Results: `reports/P115_diag_results.{md,json}` (aggregator `scripts/p115_diag_aggregate.py`); per-run outputs copied to `reports/P115_diag/` (8 JSONs).

**Scope and caveats (as pre-stated).**  One seed (seed 0) per cell; differences are described, not tested.  All quantities are re-evaluations of each
run's own objective on fixed diagnostic batches (2 × B 256 FIT images, the same base images and diagnostic RNG for every run; standard runs share
crop boxes, strong runs share crop boxes).  SimCLR and recipe-VCS runs have no epoch-20 checkpoint, so their earliest point is epoch 100; epoch-20
comparisons exist only for A-P3 and G2.  The epoch-800 fraction of positives below κ is small (≤ 3.7 %), so "positives cross the fixed threshold" is not
the default mechanism, and the final checkpoint does not exclude earlier effects.  Nothing here is a causal statement.

Accuracy change std → strong (3-seed means, context): **falls** for A-P3 (−0.78) and G2 (−1.25); **rises** for SimCLR (+1.17) and recipe VCS (+0.65).
(Seed-0 changes: A-P3 −0.40, G2 −1.80, SimCLR +1.54, recipe +1.36.)

## 1. Changes under strong augmentation that occur in all four methods (do not separate falling from rising)

| epoch 800, std → strong | A-P3 (falls) | G2 (falls) | SimCLR (rises) | recipe VCS (rises) |
|---|---|---|---|---|
| fraction of positive pairs with crop IoU < 0.25 (by design) | 0.101 → 0.299 | 0.101 → 0.299 | 0.101 → 0.299 | 0.101 → 0.299 |
| low-IoU pairs' share of the positive-term encoder gradient | 0.483 → 0.882 | 0.585 → 0.887 | 0.410 → 0.730 | 0.730 → 0.969 |
| IoU < 0.1 pairs' share of that gradient | 0.049 → 0.416 | 0.154 → 0.400 | 0.042 → 0.335 | 0.360 → 0.543 |
| h effective rank (clean held-out images) | 114.9 → 87.8 | 93.8 → 72.8 | 147.9 → 128.6 | 130.7 → 103.3 |
| token-level pull alignment, strong-crop tokens (‖Σ pulls‖/Σ‖pulls‖) | 0.964 → 0.931 | 0.971 → 0.941 | 0.877 → 0.827 | 0.994 → 0.995 |

- Under the strong block the positive-term gradient becomes dominated by low-overlap pairs in **every** method (low-IoU share +0.24 to +0.40 at epoch 800);
  the shift is not specific to the methods that lose accuracy.  The same holds at epochs 100 and 400.
- h effective rank falls by 19–27 under strong augmentation in all four methods.
- Positive alignment: mean positive cosine falls by 0.07 (A-P3), 0.07 (G2) and **0.10 (SimCLR)**, and its 5th percentile by 0.24, 0.23 and **0.42** —
  SimCLR loses the most positive alignment yet gains accuracy, so lower positive cosine per se does not track the accuracy loss.  The recipe's positive
  cosine rises slightly (+0.008).

## 2. Differences that do distinguish the fixed-scorer runs (A-P3, G2) from the learned-scale recipe (both VCS)

| epoch 800 | A-P3 std / strong | G2 std / strong | recipe VCS std / strong |
|---|---|---|---|
| zero-score threshold κ = −b/a | 0.500 / 0.500 (fixed) | 0.500 / 0.500 (fixed) | 0.903 / **0.946** (learned) |
| mean negative cosine | −0.003 / −0.000 | −0.004 / 0.020 | 0.763 / **0.886** |
| fraction of positives below κ | 0.0049 / **0.0365** | 0.0033 / **0.0286** | 0.0013 / 0.0111 |
| IoU < 0.1 group: fraction below κ | 0.129 / **0.256** | 0.045 / **0.187** | 0.045 / 0.096 |
| IoU < 0.1 group: mean T | 0.409 / **0.228** | 0.495 / **0.308** | 0.688 / 0.465 |
| h norm mean | 4.80 / 6.69 | 5.77 / 7.29 | 7.24 / 6.89 |

- With a learned scale the recipe's threshold moves up with strong augmentation (κ 0.903 → 0.946) while its representation sits in a narrower cone
  (negative cosine 0.763 → 0.886); its positives stay above κ (1.1 % below).
- With the fixed κ = 0.5 the negatives stay near orthogonal (|mean cosine| ≤ 0.02 at epoch 800) and more positives fall below κ (2.9–3.7 %); in the
  lowest-overlap group (IoU < 0.1) 19–26 % are below κ and their mean T is 0.23–0.31.  These are the pairs that, under strong augmentation, carry 40–42 %
  of the positive-term encoder gradient for A-P3 and G2.  This is an observed co-occurrence on one seed, not a demonstrated cause.
- h norm grows under strong augmentation for A-P3 (+1.9) and G2 (+1.5) — and for SimCLR (+1.2) — but not for the recipe (−0.35).

## 3. Early training (epoch 20; available for A-P3 and G2 only)
- Negatives start less separated under strong augmentation: mean negative cosine 0.051 (A-P3 strong) vs −0.000 (std); **0.176 (G2 strong) vs 0.013 (G2 std)**,
  decaying to 0.020 by epoch 800 for G2 strong.  G2 detaches the negative side; A-P3 uses the full gradient.
- View-pair positive gradients are almost fully aligned at epoch 20 under strong augmentation (view-pair ‖Σg‖/Σ‖g‖ 0.980 A-P3, 0.966 G2, vs 0.738 / 0.757
  standard); by epoch 100 the strong runs are back in the standard range (0.64–0.72).
- Fraction of positives below κ at epoch 20: 0.101 (A-P3 strong) vs 0.035 (std); 0.052 (G2 strong) vs 0.022 (std).
- No epoch-20 point exists for SimCLR or the recipe, so whether these early differences are specific to the fixed scorer cannot be read from job 1.

## 4. Quantities without a consistent pattern
- P/Q gradient balance (‖∂L_P‖/‖∂L_Q‖ w.r.t. encoder parameters) varies non-monotonically across checkpoints within a run (A-P3 std 12.7 → 14.3 → 14.7 →
  24.0; G2 strong 12.5 → 9.6 → 8.9 → 16.3) and shows no consistent std-vs-strong sign across methods; at z the ratio falls under strong augmentation for the
  three VCS runs (A-P3 8.3 → 5.9, G2 4.1 → 3.8, recipe 2.5 → 2.1) and rises for SimCLR (3.7 → 4.5).  With two diagnostic batches these ratios are noisy.
- View-pair cancellation at epochs ≥ 100: std → strong changes between −0.11 and +0.18 with mixed signs across methods and checkpoints (largest: recipe +0.175 at epoch 800, SimCLR +0.107 at 800, A-P3 −0.110 at 100).

## 5. Summary (descriptive)
Job 1 shows that strong augmentation moves the positive-term gradient onto low-overlap pairs, lowers h's effective rank and (except for the recipe) lowers
positive alignment in all four methods — including SimCLR and the recipe, whose accuracy rises — so none of these alone accounts for the fixed-scorer
runs' loss.  What differs between the fixed-scorer VCS runs and the learned-scale recipe is where the zero-score threshold sits relative to the pairs:
the recipe's learned κ moves with the augmentation inside a narrowing cone, while the fixed κ = 0.5 leaves a growing minority of low-overlap positives at
low T (mean T 0.23–0.31 for IoU < 0.1) that carry a large share of the positive gradient.  Whether this matters causally is not tested here; the P115
crop-only / jitter-only runs (job 2) will show which augmentation component produces these shifts, and P112's κ follow-up seeds test a moved fixed κ.

## Anomalies (none affect the readings)
- The per-checkpoint `epoch` field in the diagnostic outputs is `None` for all runs; epochs were taken from `checkpoint_map` (all mappings recorded).
- The view-pair gradient cosine matrix has diagonal entries 0.992–0.999 instead of 1 and asymmetries ≤ 0.0023, consistent with float32 rounding in a
  Gram matrix over ~11 M encoder parameters; off-diagonal cosines carry the same ≈ 0.007-level error, small relative to the values discussed.
- Cross-view-K runs (G2, recipe) score 1 536 positive pairs per batch, all-view / SimCLR runs 3 072; group fractions and shares are per-run normalised.
