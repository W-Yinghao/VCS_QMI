# P9 — VCS-QMI 800-epoch runs (K = 1, seeds 0,1,2) report

Status: **COMPLETE** (P8 runs 2026-09-24 19:15 → 2026-09-25 02:24 UTC; 3/3 COMPLETED, no failures, no resumes).
Neutral table: `reports/P9_long800_results_table.md`.  Pre-registration: `reports/P8_LONG800_PREREG_FROZEN_20260924.md`.
Reference: the 200-epoch VCS runs (P5, same seeds and initial weights).  **No control was run at 800 epochs; no VCS-vs-control statement
is made from this stage.**

## A. Setup
Recipe identical to P5 except `epochs=800` (warm-up 10 epochs, cosine to 1 % over T = 140,000 steps; LR 8.7e-4 at epoch 200, 5.2e-4 at
400, 1.6e-4 at 600, 1.0e-5 at 800), checkpoints 100/200/400/600/800, monitor at 0/20/50/100/200/400/600/800.  GPUs: H100 (seed 0,
3.6 h), RTX PRO 6000 (seed 1, 5.1 h), A100-PCIE (seed 2, 6.9 h).  Code unchanged since `76cebd6` in `src/`, `reference/`.

## B. Results (selection set, frozen `h`, epoch-800 checkpoint, pilot probe protocol)

| | 200 epochs (P5) | 800 epochs (P8) | Δ |
|---|---|---|---|
| linear-val (%) | 74.34 ± 0.46 | **78.93 ± 0.40** | +4.6 (per seed +4.40 / +4.24 / +5.12) |
| kNN (%) | 63.93 ± 0.62 | 71.52 ± 0.14 | +7.6 |
| h effective rank | 13.2 ± 0.3 | 20.5 ± 0.4 | +7.4 |
| dominant `p_raw` directions | 7 (all seeds) | 10 (seed 0) | +3 |
| heldout-J | 0.899 ± 0.002 | 0.947 ± 0.001 | +0.048 (train J_raw 0.952–0.955; gap ≤ 0.008) |
| saturation |t| > 0.95 (pos / neg, held-out) | 0.76 / 0.85 | 0.84–0.93 / 0.93–0.95 | |

Per seed at 800: linear 79.24 / 78.48 / 79.06; kNN 71.36 / 71.60 / 71.60; h-rank 20.1 / 20.7 / 20.8; J 0.948 / 0.946 / 0.947.

Trajectories (seed means): kNN 37.0 (0) → 48.7 (20) → 56.3 (50) → 60.9 (100) → 64.8 (200) → 68.7 (400) → 71.0 (600) → 71.5 (800);
heldout-J 0 → 0.71 → 0.80 → 0.85 → 0.895 → 0.925 → 0.942 → 0.947.
Increments per 200 epochs after epoch 200: kNN +3.9, +2.3, +0.5; J +0.030, +0.017, +0.005.

## C. Reading (per the frozen grid)
- **Continued improvement: yes, in every seed** (+4.2 to +5.1 linear points over the same seed at 200 epochs, ≈ 10× the seed SD).  The
  encoder was far from converged at 200 epochs under this recipe.
- **Concentration: persistent but slowly relaxing.**  h-rank 13 → 20.5 and 7 → 10 dominant `p_raw` directions with 4× the budget; still
  far from the 200-epoch controls (76–90).  Pre-committed clause: "persistent" (rank < 20 was the threshold; observed 20.1–20.8 sits at the
  boundary → reported as *slowly relaxing, not resolving*).
- **J vs transfer decoupled at the end**: between epochs 600 and 800 heldout-J moved +0.005 and kNN +0.5 while the LR fell from 1.6e-4 to
  1e-5; between 400 and 600 (LR 5e-4 → 1.6e-4) kNN still moved +2.3.  The final plateau is therefore at least partly schedule-driven;
  the P14 `const` / `floor0.1` runs test this directly.
- **Saturation grows with training** (positives 76 % → 84–93 % of held-out pairs with |t| > 0.95): as J approaches its bound the fraction of
  pairs still carrying gradient shrinks.  Together with the K and critic-capacity results (P11/P13: negatives and critic size are not
  the limiter), this supports the working hypothesis that the bounded quadratic objective is nearly satisfied by a low-dimensional code and
  supplies little further learning signal — a property to bring to the theory side, not something to patch with regularizers.
- Seed variance stays small (0.40 linear, 0.14 kNN), consistent with the owner's decision to explore hyper-parameters single-seed.
- Cost: 12,995 s (H100) / 18,272 s (RTX PRO 6000) / 24,890 s (A100-PCIE) per 800 epochs; memory unchanged.

## D. Consequence
The "best VCS" search should assume ≥ 800 epochs (or a schedule that keeps the LR higher for longer) when combining the factors that
helped (K = 8: +2.4 at 200 epochs; critic LR ×3–×10: kNN/rank gains).  Whether these gains persist at 800 epochs is untested.

**Waiting for the owner.**
