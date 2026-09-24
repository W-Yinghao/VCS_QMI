# P11 — VCS-QMI single-factor K study report (K ∈ {1, 8, 64}; seeds 0,1,2; 200 epochs)

Status: **COMPLETE** (P10 runs 2026-09-24 19:32–23:14 UTC; 6/6 COMPLETED, no failures, no resumes).  Neutral table:
`reports/P11_kstudy200_results_table.md`.  Pre-registration: `reports/P10_KSTUDY_PREREG_FROZEN_20260924.md`.
K = 1 comparator: the P5 VCS runs (same seeds, same initial weights, same 200-epoch schedule).

## A. What changed
Only `pairing.k` (number of distinct nonzero cyclic shifts per step; negatives per step = K·B = 256 / 2,048 / 16,384, averaged as one
distribution).  Config policy relaxed from `K == 1` to `1 ≤ K ≤ B−1` (spec §6.1 interface); tests added; everything else identical to P5.
The critic hold-out diagnostic uses each run's own K.

## B. Results (selection set, frozen `h`, fixed epoch-200 checkpoint; mean ± sample SD over 3 seeds)

| K | linear-val (%) | kNN (%) | h eff-rank | z eff-rank | heldout-J | train J_raw ep200 | step s (A100) |
|---|---|---|---|---|---|---|---|
| 1 (P5) | 74.34 ± 0.46 | 63.93 ± 0.62 | 13.18 ± 0.30 | 7.1 | 0.899 ± 0.002 | 0.902–0.907 | 0.159–0.177 |
| 8 | 76.74 ± 0.78 | 67.09 ± 0.31 | 15.57 ± 0.18 | — | 0.927 ± 0.001 | — | ≈ 0.18 |
| 64 | 76.80 ± 0.31 | 67.53 ± 0.61 | 15.93 ± 0.18 | — | 0.932 ± 0.000 | — | ≈ 0.18 |

Per seed (linear-val, K=1 → 8 → 64): seed 0 74.84 → 76.12 → 76.46; seed 1 74.24 → 77.62 → 76.86; seed 2 73.94 → 76.48 → 77.08.
Per seed (kNN): 64.64 → 67.44 → 67.86; 63.64 → 66.90 → 66.82; 63.50 → 66.92 → 67.90.
Epoch-0 references are identical to P5 (same `initial.pt` per seed).  Peak memory 3.69 GB at K = 64 (vs 3.69 GB at K = 1 on A100).

## C. Reading (per the frozen grid)
- **K = 1 → 8 helps in every seed**: +1.3 / +3.4 / +2.5 linear points (mean +2.4, > the 0.46 K=1 seed SD in all seeds) and +2.8 / +3.3 / +3.4
  kNN points.  This satisfies the pre-committed "effect" criterion.
- **K = 8 → 64 gives no further change**: 76.74 vs 76.80 linear (difference within seed SD), kNN +0.4, rank +0.4; the effect is
  **not monotone beyond K = 8** in this range, i.e. it saturates.
- **Concentration hypothesis: only weakly supported.**  h effective rank rose from 13.2 to 15.6–15.9 and `p_raw` still has ≈ 7–8 dominant
  directions; the controls sit at 76–90.  Enlarging the per-step negative pool by 64× moved the rank by ≈ 20 %, so the low-dimensional
  concentration is **not mainly a negative-sample-count effect** under this objective/critic.
- `heldout_J` increased with K (0.899 → 0.927 → 0.932) together with train `J_raw`; as pre-stated this is a finite-sample pairing effect and
  not evidence about the population objective; train and held-out J remain within ≈ 0.01 in every run.
- Cost: the K·B critic forward is negligible (step time and memory unchanged within measurement noise on the A100).

## D. Consequence for the tuning plan
K = 8 is the current best single change (+2.4 linear points at no cost) and K ≥ 8 can be treated as saturated for the next rounds; any
combination with other factors should use K = 8 (cheaper) unless a later factor interacts with the negative pool.  No control has been run
at K > 1 and none is needed: K is VCS-specific.  These are selection-set development results; the final comparison must still give the
controls an equal tuning budget.

**Waiting for the owner.**  (The A-group screen P12 and the 800-epoch runs P8 continue in parallel.)
