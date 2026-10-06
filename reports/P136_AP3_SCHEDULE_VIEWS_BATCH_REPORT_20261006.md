# P136 — VCS (A-P3) schedule / view count / batch size, CIFAR-10 and CIFAR-100, seed 0 — report — 2026-10-06

Pre-registration `P136_AP3_SCHEDULE_VIEWS_BATCH_PREREG_FROZEN_20261005.md`; results-only commit `2729ace` (`P136_results.json`).  ResNet-18, one change
per cell from that dataset's A-P3 seed-0 config; development split (official test closed); frozen-h linear primary, kNN alongside.

## 1. Results

| dataset | cell | linear / kNN | Δ linear | Δ kNN | promising (≥ +0.30 / +0.50) | train time (vs A-P3) |
|---|---|---|---|---|---|---|
| CIFAR-10 | A-P3 (4 views, B 256, 800 ep) | 89.06 / 87.30 | — | — | — | 4.5 h RTX |
| CIFAR-10 | ep1600 | 88.68 / 87.74 | −0.38 | +0.44 | no | 17.4 h L40S |
| CIFAR-10 | v8 | 89.20 / 87.82 | +0.14 | +0.52 | no | 9.3 h RTX |
| CIFAR-10 | b512 (lr kept 1e-3) | 88.58 / 87.28 | −0.48 | −0.02 | no | 8.8 h L40S |
| CIFAR-100 | A-P3 | 60.20 / 55.92 | — | — | — | 7.7 h |
| CIFAR-100 | ep1600 | 59.56 / 56.88 | −0.64 | +0.96 | no | 17.4 h L40S |
| CIFAR-100 | v8 | 59.62 / 57.14 | −0.58 | +1.22 | no | 17.6 h L40S |
| CIFAR-100 | b512 (lr kept 1e-3) | 59.36 / 55.14 | −0.84 | −0.78 | no | 9.0 h L40S |

Train time is the logged training wall time (GPU of the job's start; L40S ≈ 2× slower per epoch than a healthy RTX6000PRO, so it is not a pure
cost comparison).  v8 and ep1600 cost about twice the epoch compute of A-P3 (v8: 56 vs 12 positive view pairs per image).  The trainer does not
log peak memory, so the compute line is wall time only.
ep1600 intermediate (the epoch-800 checkpoint of the same run, cosine over 1600, so the learning rate is still at half its peak): kNN 87.46
(CIFAR-10) / 56.50 (CIFAR-100).  The extra 800 epochs add +0.28 / +0.38 kNN.  No linear probe was run at that checkpoint (the trainer's schedule
evaluates kNN only there), so the linear intermediate is not reported.

## 2. Pre-stated reading
**No cell is promising**, so no seeds-1–2 addendum and no matched-JS addendum.  On both datasets:
- **More compute buys kNN, not linear.** ep1600 and v8 raise kNN (+0.44 / +0.52 CIFAR-10; +0.96 / +1.22 CIFAR-100) while linear stays at or
  below A-P3 (−0.38 to +0.14 CIFAR-10; −0.58 / −0.64 CIFAR-100), at about 2× the compute.
- **b512 at the unchanged lr is worse on both metrics** (−0.48 / −0.84 linear); lr was not scaled with the batch (the disclosed confound), so this
  says nothing about b512 with a tuned lr.
- The A-P3 recipe (4 views, batch 256, 800 epochs) stays the reference.  Single seed: CIFAR-10 seed spread ≈ 0.3, so the CIFAR-10 v8 +0.14 is
  within noise.

## 3. Not claimed
Multi-seed statements; SimCLR or JS at these settings; b512 with a scaled lr; other backbones; the official test set.
