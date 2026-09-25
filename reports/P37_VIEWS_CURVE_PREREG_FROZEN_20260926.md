# Pre-registration — views at fixed encoder compute on the a0 = 5 base (P37), frozen 2026-09-26 before launch

Trigger: 4 views HELPS at equal compute (P34: 81.84 / kNN 77.22 at 100 epochs vs 80.59 / 74.41 for 2 views at 200 epochs, a0 = 1 base) and at
equal epochs (P29: 84.48 / 81.10).  The number of positive pairs per image per step (1 → 6) is now the strongest lever found; the natural
question is whether it keeps paying at 8 views (28 pairs) and how the gain trades against the compute it costs.  The code path is general
in the number of views (J averaged over all pairs, each with its own K = 8 shifts; negatives still from other UIDs); only the schema was
extended (2, 4, 8), gate re-run.

Base for this wave: cosine critic, K = 8, negative detach, **a0 = 5** (P31: 81.56 ± 0.44 / 76.94 at 200 epochs, 2 views, 1× compute).

| unit | views | epochs | encoder compute | question |
|---|---|---|---|---|
| a5_views4_100ep | 4 | 100 | 1× | the equal-compute 4-view point on the *a0 = 5* base (P34 was on a0 = 1) |
| a5_views8_50ep | 8 | 50 | 1× | equal-compute 8-view point: does the views curve keep rising at fixed compute? |
| a5_views8_100ep | 8 | 100 | 2× | 8 views at the compute of `P35_vcs_a5_views4` (4 views, 200 ep, running) — equal-compute comparison at 2× |

Warm-up stays 10 epochs (a larger fraction of the shorter runs; disclosed).  Memory: 8 views × 256 images per step ≈ 15 GB peak (4 views: 7.6 GB).

## Reading
Fixed-compute curve (1×): 2 views/200 ep = 81.56 (3 seeds) → 4 views/100 ep → 8 views/50 ep; HELPS/HURTS at ±1.0 vs 81.56.
2× curve: 4 views/200 ep (`P35_vcs_a5_views4`, 3 seeds queued) vs 8 views/100 ep.  Secondary: kNN, h-rank, heldout-J, positive-pair
saturation fraction (P29 §4).  Single seed.  sha256 in `configs/HPARAM_L_SHA256.json`; gate + babysitter ids in `job_ids.json`.
