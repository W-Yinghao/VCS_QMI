# P36 — the a₀ = 5 base with 4 views: final report (2026-09-28; supersedes `P36_A5_BASE_REPORT_interim_20260926.md`)

Pre-registration `P35_A5_BASE_PREREG_FROZEN_20260925.md` (+ addenda).  Table `P36_a5_base_results_table.md` / `.json` (last results-only commit b7fbc2b; all rows
COMPLETED).  Selection split, frozen-h linear probe / kNN.  The interim report's 200-epoch reading is unchanged; this report adds the 400/800-epoch cells that were
pending there and the official test values that P68 later attached to the same checkpoints.

## 1. Cells (mean ± sd over seeds 0/1/2 where three exist)
| cell | epochs | views | selection linear | kNN | held-out J | h eff-rank | train s (seed 0) | test linear (P68) |
|---|---|---|---|---|---|---|---|---|
| a5_views4 (2×) | 200 | 4 | **84.54 ± 0.16** (84.50 / 84.40 / 84.72) | 81.40 ± 0.22 | 0.964 | 76 | 13 682 | 84.52 ± 0.18 |
| a5_800ep (4×) | 800 | 2 | **85.30 ± 0.21** (85.54 / 85.18 / 85.18) | 82.33 ± 0.25 | 0.964 | 86 | 22 455 | 85.28 ± 0.40 |
| a5_views4_400ep (4×) | 400 | 4 | 85.90 (seed 0) | 83.58 | 0.970 | 105 | 24 183 | 85.63 |
| a5_views4_800ep (8×) | 800 | 4 | **87.01 ± 0.53** (86.42 / 87.16 / 87.44) | 85.46 ± 0.12 | 0.974 | 134 | 48 508 | 86.65 ± 0.26 |

## 2. Reading (pre-committed in P35: report the recipe at each horizon; no control comparison inside this unit)
1. The recipe gains ≈ +1.2 linear per compute doubling on this split (84.54 → 85.95 [8v/200ep, P38] → 87.01) and the gain is carried by views more than by epochs
   at 4× (4v/400ep 85.90 ≈ 8v/200ep 85.95 > 2v/800ep 85.30).
2. Seed spread grows with the horizon (0.16 at 200 ep, 0.53 at 800 ep); the 800-epoch seeds are ordered by their kNN at epoch 600 (85.2 / 85.4 / 85.5), not by anything
   in the configuration.
3. Held-out J rises with the horizon (0.964 → 0.974) while the threshold of the learned critic stays at cos* ≈ 0.80–0.91; positive saturation reaches 61 % at 4v/800ep
   (P84), so "positives stay unsaturated" holds for the 200-epoch two-view runs only.
4. The 8× cell is the paper's VCS number; its ceiling neighbours (P44: 4v/1600ep 87.50, 8v/800ep 87.14, strong-aug 87.78, all single seed) are within +0.8 of it.
   The equal-budget controls are in P41 (tuned SimCLR 88.32 ± 0.30, VICReg 87.11 ± 0.51 at 8×) and the test-set values in P68.

## 3. Consequence
Recipe frozen as in the technical note §3; no further objective-side tuning (owner, 2026-09-26; S3).  Strong augmentation on the long schedule (+0.8, one seed) is the
only fixed-compute lever found and is the subject of the S4 unit (P89, method × augmentation with 3 seeds each).
