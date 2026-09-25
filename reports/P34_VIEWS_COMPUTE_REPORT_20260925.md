# P34 — four views at equal encoder compute (P33), final

Pre-registration: `P33_VIEWS_COMPUTE_PREREG_FROZEN_20260925.md`.  Table: `P34_views_compute_results_table.md` (summarize job 1009175).
Unit: 4 views, J averaged over the 6 view pairs (K = 8 each), neg-detach cosine base (a0 = 1), **100 epochs** = the encoder compute of the
200-epoch 2-view base (train time 6 081 s vs 6 232 s for the base; 8.96 M image-forwards both).

| configuration | epochs | encoder compute | linear | kNN | h-rank | heldout-J |
|---|---|---|---|---|---|---|
| 2 views (base, 3 seeds) | 200 | 1× | 80.59 ± 0.12 | 74.41 ± 0.19 | 30.4 | 0.928 |
| **4 views, equal compute** | **100** | **1×** | **81.84** | **77.22** | 41.1 | 0.945 |
| 4 views, equal epochs (P29) | 200 | 2× | 84.48 | 81.10 | 57.4 | 0.960 |

## Reading (pre-registered grid: HELPS at both = real gain from more pairs per step; HELPS only at 200 = compute effect)
**HELPS at both**: +1.25 linear / +2.8 kNN at equal compute, +3.9 / +6.7 at equal epochs.  More positive pairs per image per step is a
genuine improvement of the objective's training signal, not a compute artefact; roughly a third of the equal-epoch gain is "free" and
the rest is the extra compute (which the 4-view recipe converts into accuracy at a much better rate than the 2-view recipe did:
2-view 200 → 800 epochs gave +2.4 for 4× compute, P25).  The kNN trajectory of the 100-epoch run tracks the 200-epoch 4-view run
epoch-for-epoch (61.1 / 68.6 / 74.3 / 77.2 vs 61.2 / 69.0 / 74.8 / 78.6 at 10 / 20 / 50 / 100) despite the twice-faster LR decay.

## Consequence
Four views enter the base.  The compute-matched claim for the paper is "+1.25 linear, +2.8 kNN at equal encoder compute (single seed)";
the equal-epoch number is reported alongside with its 2× cost.  P35 (a0 = 5 + 4 views, seeds 0/1/2, 200 epochs) is running.

## Not claimed
Single seed; selection split; no control at 4 views yet (SimCLR / VICReg multi-crop-style variants would need the same treatment before a
comparison, as the owner ruled: controls get the same tuning budget later).
