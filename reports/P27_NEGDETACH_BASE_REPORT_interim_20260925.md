# P27 (interim) — wave G on the neg-detach cosine base: 9 of 10 units final

Pre-registration: `P26_NEGDETACH_BASE_PREREG_FROZEN_20260925.md`.  Table: `P27_negdetach_base_results_table_interim.md` (summarize job 1009162,
2026-09-25 18:18 UTC).  `base_800ep` is at epoch ≈230/800 and is reported when it finishes (final P27 table is dependency-gated on it).
All numbers: frozen-h linear-val on the 5k selection split, 200 epochs, seed 0 unless stated.

## Base anchor (a0 = 1, learnable; cosine critic, K = 8, negative partner detached)
| seed | linear | kNN | h-rank | heldout-J |
|---|---|---|---|---|
| 0 | 80.48 | 74.62 | 30.9 | 0.929 |
| 1 | 80.58 | 74.26 | 29.6 | 0.927 |
| 2 | 80.72 | 74.34 | 30.6 | 0.927 |
| mean ± SD | **80.59 ± 0.12** | 74.41 ± 0.19 | 30.4 | 0.928 |

The base is very stable across seeds (SD 0.12 vs 0.46 for the K=1 MLP recipe).  Deltas below are against 80.59; the pre-registered rule is
HELPS > +1.0, HURTS < −1.0.

## Units
| unit | linear | Δ | kNN | h-rank | p_raw rank | learned a / b at ep 200 | verdict |
|---|---|---|---|---|---|---|---|
| a5_learn (N2, a0 = 5) | **81.90** | **+1.31** | **77.00** | 45.0 | 20.1 | 10.08 / −8.22 | HELPS |
| a5_fixed (N3, a ≡ 5) | 81.54 | +0.95 | **77.00** | 46.9 | 20.9 | 5 / −3.33 | neutral by rule; ties N2 |
| k255 | 80.82 | +0.23 | 75.58 | 33.8 | 15.0 | 10.01 / −8.24 | neutral |
| b0_calib (N1) | 80.12 | −0.47 | 73.04 | 22.8 | — | — | neutral (kNN −1.4, rank −8) |
| proj_outBN (N4) | 78.52 | −2.07 | 72.68 | 4.5 (top eig 56 %) | 2.0 | 7.06 / −5.24 | HURTS |
| shared_metric (N6) | 75.28 | −5.31 | 69.28 | 1.3 (top eig 95 %) | 1.0 | 6.28 / −4.40 | HURTS, collapse-suspected |
| interact_only (N5) | stopped ep 30 | — | 20.5 | 1.0 | — | — | collapsed (J ≈ 0), cancelled to free the GPU |
| base_800ep | running | | 75.98 @ ep 200 | | | | pending |

## Reading (against the pre-registered questions)
1. **N2/N3 — initial sharpness is the one positive factor, and it is an early-trajectory effect.** a0 = 5 gives +1.3 (≈ 10 base SDs, single
   seed) and +2.6 kNN with a 50 % higher h-rank.  Learnable and frozen scale tie (81.90 vs 81.54, identical kNN) although they end at very
   different (a, b): the learnable run drifts to the same endpoint as the base (a ≈ 10, b ≈ −8.2, threshold cos* ≈ 0.82), the frozen run stays
   at threshold 0.67.  In the learnable run `a` first *falls* from 5 to 3.3 (epoch 20) and is back at 5.5 by epoch 50, where the base also
   reaches 5.  So the two runs differ from the base only during the first ≈50 epochs, yet kNN separates from epoch 100 and never closes
   (73.5 vs 71.8 at 100; 77.0 vs 74.6 at 200).  Per the pre-registration N3 ≈ N2 ⇒ late scale freedom is not needed; the sweep of a0 (P31:
   2, 10, 20, plus seeds 1/2 of a0 = 5) decides the new base.
2. **N1 — the initial common offset is not the issue.** Calibrating b0 so that the initial scores are centred (initial J −0.26 instead of
   −0.57) is neutral on linear and slightly negative on kNN and rank.
3. **N4 — affine-free BN on the projector output hurts (−2.1)** and concentrates 56 % of h-variance in one direction (p_raw effective rank 2).
   Combined with the `cos_on_h` failure (P25: pre-normalization scale collapses without control), the scale freedom of the cosine critic
   (synthesis §4.5) is real but neither removing it (BN) nor leaving it free is a fix; the BN version makes the low-rank solution easier.
4. **N6 — a learnable shared metric degenerates.** W (128×128, init identity) is used to make the pair problem trivial: p_raw effective rank
   1.0, h-rank 1.3 with 95 % of variance in one direction, kNN −5.  This is the cleanest instance so far of synthesis §4.3: **any extra
   freedom given to the critic is spent on collapsing the code, not on enriching h.**  (Per-layer probe: the projector already keeps only
   ≈10–14 effective dimensions with the plain cosine critic.)
5. **N5 — MLP-type critics + negative detach collapse** (interact-only here, concat-MLP in P20: 44.5).  Negative detach is only usable with
   similarity-type critics; the mechanism claim in synthesis §3.2 stands.
6. **K = 255 no longer adds on top of negative detach** (+0.23; it gave +1.0 without detach).  The two factors do not stack: both seem to act on
   the same thing (how much repulsion reaches the encoder), and detach saturates it.

## Consequences for the search
- New base candidate: cosine critic, K = 8, negative detach, **a0 = 5** (learnable or fixed — tie).  Confirmation and the a0 curve are in P31.
- Dropped: output BN, learnable shared metric, interaction-only MLP, bias calibration, K = 255 with detach.
- Still open (P28, running): monotone spline score, diagonal metric, 4 views, linear-only projector, no projector (BN on h).

## Not claimed
Single seed for every unit except the base; selection split; the a0 effect is a single seed until P31 lands; effective ranks are variance
based and a rank of 1–4 with kNN ≈ 70 means one dominant direction, not a dead representation.
