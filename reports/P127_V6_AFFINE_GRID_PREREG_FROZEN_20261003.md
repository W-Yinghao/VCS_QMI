# Pre-registration — P127: v6 §8.2 per-dataset affine grid on the A-P3 structure (CIFAR-10 and CIFAR-100, seed 0) — FROZEN 2026-10-03T21:57:49Z

Owner 2026-10-03: "再提交几个，占满30个的slot限制" — read as approval of the v6 §8 tuning budget package for the VCS family (§8.2 lists it as a separate budget,
not auto-launched).  Grid fixed by v6 §8.2: (a, κ) ∈ {(1.5, .5), (2, .25), (2, .5), (2, .75), (3, .5), (3, .25)} on A-P3's all-view / full-gradient structure;
(2, .5) is the existing A-P3 seed 0 (CIFAR-10 89.06 / 87.30; CIFAR-100 60.20 / 55.92) — 5 new cells per dataset, 10 units.  Each config differs from
that dataset's A-P3 seed-0 config only in cosine_scale_init = a and cosine_bias_init = −aκ (fixed scorer; verified by diff).  P112's K8 / detach grid is
prior information only (v6: not the same training unit).

## Pre-stated reading (within-dataset selection; v6 §8.1 "dataset-selection table", kept apart from the shared-default table)
- Primary: final frozen-h linear (selection split) per dataset; kNN, and on CIFAR-100 the P124 coarse / fine / conditional readouts (evaluation-only addendum),
  reported alongside.  Selection only on the development validation labels.
- **Dataset-selected cell** = highest linear; it replaces (2, .5) as that dataset's tuned VCS cell only if it beats (2, .5) by ≥ 0.30 (CIFAR-10) / ≥ 0.50
  (CIFAR-100) — the P126 thresholds (≈ 2 × A-P3 seed sd); otherwise A-P3 (2, .5) stays.  A replacing cell gets seeds 1–2 (addendum).
- **No tuned-vs-tuned method claim:** matched JS and SimCLR have not received the same tuning budget in this round (v6 §8.2: without equal completed budgets,
  no "fully tuned" superiority statement).  CIFAR-100 question carried from P124: does any κ / a improve the within-superclass (conditional) readout without
  losing the coarse gain? — answered descriptively.
- No early stopping on J / rank / kNN.  Official test closed.
## Compute
10 × 800 epochs; RTX6000PRO / H100 (≈ 4.4 h) or L40S (≈ 8.3 h); normal QOS; node60 allowed.
