# P40 (interim) — ablations inside the recipe + optimizer re-check (P39): 5 of 8 units final

Pre-registration: `P39_RECIPE_ABLATION_PREREG_FROZEN_20260926.md` (+ addendum).  Table: `P40_recipe_ablation_results_table.md` (summarize
job of babysitter 1009271).  Running: `a5_views4_b64_100ep`, `a5_views4_b128_200ep`, `a5_b128` (batch/steps follow-ups).
Recipe: cosine critic, K = 8, negative detach, a0 = 5; 4-view units at 100 epochs (= compute of the 2-view 200-epoch base).

| unit | change | steps | linear | Δ | kNN | h-rank | (a, b) → thr | sat⁺ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| a5_views4_100ep (P37, comparator) | — | 17.5 k | 81.88 | — | 77.74 | 55.6 | 6.65 / −4.63 → 0.70 | 0.12 | |
| **a5_views4_b128_100ep** | B = 128 | 35.1 k | **83.52** | **+1.64** | 78.30 | 55.6 | 10.14 / −8.16 → 0.80 | 0.06 | HELPS (same compute) |
| a5_views4_k1_100ep | K = 1 | 17.5 k | 81.14 | −0.74 | 75.68 | 45.4 | 6.61 / −4.70 → 0.71 | 0.01 | neutral by rule; kNN −2.1 |
| a5_views4_nodetach_100ep | detach off | 17.5 k | 79.66 | −2.22 | 75.38 | 85.8 | 7.92 / −3.00 → 0.38 | 0.86 | HURTS |
| a5_lr2e-3 (2 v, 200 ep) | lr 2e-3 | 35 k | 81.64 | +0.08 vs 81.56 | 77.50 | 50.6 | | | neutral |
| a5_lr5e-4 (2 v, 200 ep) | lr 5e-4 | 35 k | 80.56 | −1.00 vs 81.56 | 75.76 | 42.0 | | | HURTS (borderline); kNN −1.2 |

## Reading
1. **Optimizer steps are a first-order factor at fixed compute.**  Halving the batch under 4 views (+1.6 linear at identical image-forwards)
   brings the critic to the sharp regime (threshold 0.80, positive saturation 6 %) that B = 256 reaches only after 200 epochs; the
   equal-epoch 4-view run (84.54, 35 k steps, 2× compute) is now only 1.0 above the 1× B = 128 run.  Read with P38 (8 views/50 ep worse than
   4 views/100 ep): once pairs per image ≥ 6, what limits the 1× budget is the number of updates.  The three follow-ups (B = 64; B = 128 at
   200 epochs; B = 128 without views) test how far this goes and whether it needs the views.
2. **Negative detach is necessary under 4 views too** (−2.2; without it 86 % of positives saturate, the threshold falls to 0.38 and h-rank
   inflates to 86 while linear drops — the plain-cosine pattern of P25).
3. **Negatives are a secondary but real ingredient**: K = 1 costs 0.7 linear / 2.1 kNN / 10 rank points.  The recipe's signal is carried
   mostly by the unsaturated positives, but one negative per pair is not enough to shape the metric.
4. **lr 1e-3 stays**: ×2 neutral, ×0.5 −1.0 with kNN −1.2 and rank −4.

## Consequence
Batch size enters the recipe search (P39 addendum, running).  Detach, K = 8 and lr 1e-3 are confirmed as recipe components.

## Not claimed
Single seed throughout; selection split; the B = 128 gain is a single seed at 100 epochs until the follow-ups land.

## Addendum 06:30 UTC — batch / steps follow-ups final
| unit | views | epochs | B | steps | compute | linear | kNN | h-rank | (a, b) → thr | comparator | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a5_views4_b128_100ep (3 seeds) | 4 | 100 | 128 | 35 k | 1× | **83.19 ± 0.40** (83.52 / 82.74 / 83.30) | 78.43 ± 0.11 | 55.8 | 10.1 / −8.2 → 0.80 | 2 v/200 ep 81.56 ± 0.44 | **+1.63** HELPS |
| a5_views4_b64_100ep | 4 | 100 | 64 | 70 k | 1× | 82.78 | 78.56 | 52.2 | | b128/100 ep 83.19 | −0.41 neutral |
| a5_b128 | 2 | 200 | 128 | 70 k | 1× | 82.14 | 76.42 | 42.8 | | 2 v/200 ep 81.56 | +0.58 neutral |
| a5_views4_b128_200ep | 4 | 200 | 128 | 70 k | 2× | 84.74 | 81.16 | 77.1 | 15.2 / −13.1 → 0.86 | 4 v/200 ep 84.54 ± 0.16 | +0.20 neutral |

Reading.  (i) At the 1× budget the recipe is **4 views, B = 128, 100 epochs: 83.19 ± 0.40** — +1.6 over the 2-view base on 3 seeds each.
(ii) The steps effect saturates: 70 k steps (B = 64 at 1×, or B = 128 at 2×) add nothing over 35 k; B = 64 is slightly worse (fewer images
per step for the K = 8 negatives, noisier BN).  (iii) Steps without views are nearly neutral (+0.6 at 70 k steps, 2 views): the update count
matters only once each update carries ≥ 6 pairs per image — the two factors interact rather than add.  (iv) At 2× compute B = 128 and
B = 256 tie (84.74 vs 84.54 ± 0.16); the 2× recipe stays B = 256 (3 seeds in hand).  Remaining P39 units (projector width, wd, K = 127) are
running.
