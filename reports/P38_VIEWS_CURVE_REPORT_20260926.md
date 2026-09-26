# P38 — views at fixed encoder compute on the a0 = 5 base (P37), final

Pre-registration: `P37_VIEWS_CURVE_PREREG_FROZEN_20260926.md`.  Table: `P38_views_curve_results_table.md`.  Base: cosine critic, K = 8,
negative detach, a0 = 5.  Compute unit 1× = 200 epochs of 2 views = 8.96 M image-forwards.

| views | epochs | compute | linear | kNN | h-rank | z-rank | (a, b) at end → thr | sat⁺ / sat⁻ | train s |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 200 | 1× | 81.56 ± 0.44 (3 seeds, P32) | 76.94 | 45.6 | 20 | 10.1 / −8.2 → 0.82 | 0.00 / 0.67 | ≈ 6 200 |
| 4 | 100 | 1× | **81.88** | **77.74** | 55.6 | 31 | 6.65 / −4.63 → 0.70 | 0.12 / 0.65 | 6 847 |
| 8 | 50 | 1× | 80.60 | 76.08 | 59.0 | 58 | 4.55 / −2.10 → 0.46 | 0.48 / 0.55 | 6 172 |
| 4 | 200 | 2× | **84.54 ± 0.16** (3 seeds, P36) | **81.40** | 75.7 | 30 | 10.8 / −8.7 → 0.80 | 0.25–0.30 / 0.76 | ≈ 13 400 |
| 8 | 100 | 2× | 83.40 | 80.44 | 74.3 | 42 | 6.70 / −4.45 → 0.66 | 0.36 / 0.72 | 12 229 |

## Reading
1. **The views curve peaks at 4 at both compute budgets** (1×: 81.56 → 81.88 → 80.60; 2×: 84.54 → 83.40).  8 views buy 28 pairs per
   image per step but only half the optimizer steps of 4 views at the same compute, and the shorter runs leave the critic less sharp
   (threshold 0.46–0.66 vs 0.70–0.82) with far more saturated positives (36–48 % vs 0–30 %).  Per pre-registered rule vs the 2-view base:
   4 views/100 ep neutral (+0.3), 8 views/50 ep HURTS (−1.0); at 2×, 8 views/100 ep is 1.1 below 4 views/200 ep.
2. **Rank keeps rising with views and with compute** (46 → 56 → 59 at 1×; 76 → 74 at 2×), kNN follows compute more than views.  The
   representation gets richer with more pairs, but the linear endpoint at fixed compute is governed by the number of optimizer steps once
   the pair count is ≥ 6 per image.
3. On the a0 = 5 base the equal-compute 4-view gain is +0.3 (on the a0 = 1 base it was +1.25, P34): the two early-training factors overlap
   (P36 §2).  The recipe's 4-view gain is therefore mostly an *equal-epoch* (2× compute) gain: +3.0 on 3 seeds.

## Consequence
4 views stays in the recipe; 8 views is dropped.  Remaining views-related question for the paper: the same 4-view treatment for the
controls before comparison (owner's rule).

## Not claimed
Single seed at 8 views and at 4 views/100 ep; warm-up (10 epochs) is a larger fraction of the 50-epoch run; selection split.

## Addendum 2026-09-26 13:20 UTC — 8 views with the steps of the 4-view winner (owner-requested units; stage table regenerates when the 200-ep unit lands)
| unit | views | epochs | B | steps | compute | linear | kNN | h-rank | (a, b) → thr | sat⁺ |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 views / 200 ep / B 256 (3 seeds, P36) | 4 | 200 | 256 | 35 k | 2× | 84.54 ± 0.16 | 81.40 | 76 | 10.8 / −8.7 → 0.80 | 0.25–0.30 |
| 8 views / 100 ep / B 256 (P38) | 8 | 100 | 256 | 17.5 k | 2× | 83.40 | 80.44 | 74 | 6.7 / −4.5 → 0.66 | 0.36 |
| **8 views / 100 ep / B 128** (new) | 8 | 100 | 128 | 35 k | 2× | **84.60** | **81.62** | 78 | 10.6 / −8.5 → 0.80 | 0.35 |
| 8 views / 200 ep / B 256 (running) | 8 | 200 | 256 | 35 k | 4× | | | | | |

Reading: with the update count restored to 35 k, 8 views ties 4 views at the same compute (84.60 vs 84.54 ± 0.16; kNN +0.2; rank +2) and the
critic ends at the same sharpness (threshold 0.80).  The P38 8-view deficit was therefore a steps effect; **beyond 6 pairs per image, more
pairs per step neither help nor hurt at fixed compute** — the objective's signal per update saturates around 4 views.  The 8-view kNN curve
is steeper early (79.3 at epoch 50 vs 74.9 for 4 views/200 ep at epoch 50), which is what the long 8-view chain (P43) tests: whether that
head start survives 800 epochs.

## Addendum 2026-09-26 17:00 UTC — 8 views / 200 ep / B 256 (4× compute, 35 k steps): 86.28
`P37_vcs_a5_views8_200ep_seed0`: **linear 86.28 / kNN 83.66 / h-rank 103.7** (z-rank 30, heldout-J 0.972).  Stage table `P38_views_curve_results_table.md` regenerated.

| views | epochs | B | steps | compute | linear | kNN | h-rank |
|---|---|---|---|---|---|---|---|
| 2 | 200 | 256 | 35 k | 1× | 81.56 ± 0.44 | 76.94 | 46 |
| 4 | 200 | 256 | 35 k | 2× | 84.54 ± 0.16 | 81.40 | 76 |
| 8 | 100 | 128 | 35 k | 2× | 84.60 | 81.62 | 78 |
| **8** | **200** | **256** | **35 k** | **4×** | **86.28** | **83.66** | **104** |
| 4 | 400 | 256 | 70 k | 4× | 85.90 | 83.58 | 105 |
| 2 | 800 | 256 | 140 k | 4× | 85.30 ± 0.21 | 82.33 | 86 |
| 4 | 800 | 256 | 140 k | 8× | 86.42 | 85.60 | 136 |

Reading: at a fixed 200-epoch schedule the views curve is 81.56 → 84.54 → 86.28 (+3.0, +1.7) — more pairs per image do keep paying when the
step count is held at 35 k and the batch at 256; at the 4× budget, 8 views × 200 ep beats 4 views × 400 ep (+0.4 / +0.1 kNN) and 2 views
× 800 ep (+1.0 / +1.3).  The P38 conclusion "peaks at 4" was a fixed-compute statement at 1×–2×, where 8 views could only be bought with
fewer steps or a smaller batch; with the owner's compute-unconstrained goal the right reading is: **views scale as well as epochs, and
better than epochs at equal compute from 4× on**.  This is the second-best absolute number (0.14 below the 8× 4-view run at half its
compute).  Follow-ups (P43 addendum): 8 views × 400 ep (8×, vs 86.42) and 16 views × 200 ep (8×), plus the running 8 views × 800 ep (16×).
