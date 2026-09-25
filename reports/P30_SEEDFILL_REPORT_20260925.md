# P30 — seed fills for the cosine critic (K = 8 and K = 255): the critic-form and K claims on 3 seeds

Pre-registration: `P28_NEW_CANDIDATES_PREREG_FROZEN_20260925.md` §P29.  Table: `P30_seedfill_results_table.md` (summarize job 1009121).
All numbers: frozen-h linear-val / kNN on the 5k selection split, 200 epochs, seeds 0/1/2 (seed 0 from P18 / P24).

| recipe | n | linear-val (mean ± SD) | seeds | kNN | h-rank | heldout-J |
|---|---|---|---|---|---|---|
| concat-MLP critic, K = 1 (original recipe, P5) | 3 | 74.34 ± 0.46 | 74.84 / 74.24 / 73.94 | 63.93 ± 0.62 | 13.2 | 0.899 |
| concat-MLP critic, K = 8 (P10) | 3 | 76.74 ± 0.78 | 76.12 / 77.62 / 76.48 | 67.09 ± 0.31 | 15.6 | 0.927 |
| cosine critic, K = 8 | 3 | 78.21 ± 0.13 | 78.32 / 78.06 / 78.24 | 73.09 ± 0.39 | 57.7 | 0.967 |
| cosine critic, K = 255 | 3 | 79.09 ± 0.29 | 79.32 / 78.76 / 79.18 | 73.92 ± 0.41 | 60.9 | 0.969 |
| cosine critic, K = 8, negative detach (base) | 3 | 80.59 ± 0.12 | 80.48 / 80.58 / 80.72 | 74.41 ± 0.19 | 30.4 | 0.928 |

## Claims now supported on 3 seeds (Welch t on 3 vs 3; all differences ≫ pooled SD)
1. **Critic form at K = 8: cosine > concat-MLP by +1.5 linear, +6.0 kNN, h-rank 58 vs 16** (78.21 ± 0.13 vs 76.74 ± 0.78).  The kNN
   gap is four times the linear gap: the similarity-type critic changes the metric structure of h far more than its linear readability.
2. **K for the cosine critic: 255 > 8 by +0.9 linear, +0.8 kNN** (79.09 ± 0.29 vs 78.21 ± 0.13; ≈ 3 pooled SD).  Small but consistent,
   as pre-registered ("mild monotone"); it does not survive negative detach (P27: 80.82 vs 80.59).
3. **Negative detach on the cosine critic: +2.4 linear, +1.3 kNN, h-rank halves** (80.59 ± 0.12 vs 78.21 ± 0.13).  The factor with the
   largest single effect also gives the lowest seed variance seen in the project.
4. Seed SD of every cosine recipe is ≤ 0.3, vs 0.5–0.8 for the MLP recipes: the similarity-type critic also makes training more reproducible.

## Cumulative, 200 epochs, seed means
74.34 (original) → 76.74 (K = 8) → 78.21 (cosine critic) → 80.59 (negative detach) → 81.5–81.9 (a0 = 5, single seed, P31 confirming):
+6.3 so far, of which +2.4 K, +1.5 critic form, +2.4 negative detach, ≈ +1.2 initial scale.  SimCLR matched: 86.09 ± 0.38.

## Not claimed
Selection split; 200 epochs; the a0 = 5 step is a single seed until P31 lands; no comparison with controls at equal tuning budget yet.
