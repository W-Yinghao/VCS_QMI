# P36 (interim) — the a0 = 5 base with 4 views: seeds 0/1/2 final; 800-epoch runs pending

Pre-registration: `P35_A5_BASE_PREREG_FROZEN_20260925.md` (+ addenda).  Table: `P36_a5_base_results_table_interim.md` (summarize job 1009325).
`a5_800ep` (epoch ≈ 560/800) and `a5_views4_800ep` (≈ 180/800) are reported when final.

## Recipe at 200 epochs: cosine critic tanh(a⟨z1,z2⟩+b), K = 8, negative partner detached, a0 = 5, **4 views** (J averaged over the 6 pairs)
| seed | linear | kNN | h-rank | z-rank | heldout-J | (a, b) at 200 → threshold | sat⁺ / sat⁻ |
|---|---|---|---|---|---|---|---|
| 0 | 84.50 | 81.40 | 76.0 | 30.0 | 0.964 | 10.83 / −8.70 → 0.80 | 0.25 / 0.76 |
| 1 | 84.40 | 81.18 | 76.1 | 29.9 | 0.964 | 10.82 / −8.69 → 0.80 | 0.28 / 0.76 |
| 2 | 84.72 | 81.62 | 75.1 | 29.6 | 0.964 | 10.84 / −8.71 → 0.80 | 0.30 / 0.77 |
| **mean ± SD** | **84.54 ± 0.16** | **81.40 ± 0.22** | 75.7 | 29.8 | 0.964 | | |

Reference points (200 epochs, same split and probe): SimCLR matched 86.09 ± 0.38 / kNN 83.98 / rank 90; VICReg matched-128 85.46 ± 0.26 /
81.92 / 76; original VCS recipe 74.34 ± 0.46 / 63.93 / 13.  Encoder compute of this recipe: 2× the 2-view runs (train time ≈ 13 400 s vs
≈ 6 200 s).

## Reading
1. **The 200-epoch recipe is 84.54 ± 0.16, i.e. 1.55 below SimCLR and 0.9 below VICReg, with kNN 81.4 (SimCLR 84.0, VICReg 81.9) and an
   h effective rank of 76 (VICReg 76, SimCLR 90).**  From the original recipe that is +10.2 linear / +17.5 kNN.  The three seeds agree to
   0.16 — the tightest spread in the project — and the learned critics are numerically identical across seeds (a 10.8, b −8.7).
2. **a0 = 5 and 4 views do not add.**  4 views on a0 = 1: 84.48 (P29, single seed); on a0 = 5: 84.54 ± 0.16.  At equal compute: 4 views/100 ep
   on a0 = 1 gave +1.25 over its base, on a0 = 5 +0.3 (P37: 81.88 vs 81.56).  Both factors act early in training (P32 §2) and 4 views
   already provides the effect; the initial scale only lifts h-rank (57 → 76).  For the paper the recipe can be stated with either a0 = 1
   or a0 = 5; a0 = 5 is kept because it costs nothing and is what the seeds were run with.
3. **The views curve at fixed compute peaks at 4** (P37 so far: 8 views/50 ep 80.60 < 4 views/100 ep 81.88 ≈ 2 views/200 ep 81.56); the 8-view
   100-epoch (2×) point is pending.  Positive-pair saturation rises with views (0 % at 2 views → 25–30 % at 4 views, P29 §4).
4. **K = 1 costs 0.7 linear / 2.1 kNN under this recipe** (P39, pending table): negatives remain a necessary but secondary ingredient.

## Consequence
The 200-epoch VCS recipe for the paper is fixed at: cosine critic, K = 8, negative detach, a0 = 5, 4 views — **84.54 ± 0.16 / kNN 81.40**.
Pending: its 800-epoch value (a5_views4_800ep), the 2-view 800-epoch a0 = 5 value, the P37/P39 ablations, and — per the owner's rule —
controls re-tuned with an equal budget (in particular a 4-view / multi-crop SimCLR and VICReg) before any comparison is written up.

## Not claimed
Selection split; controls not yet tuned or given 4 views; compute is 2× the 2-view controls at equal epochs.

## Addendum 04:10 UTC — a5_800ep final
`P35_vcs_a5_800ep_seed0` (a0 = 5, 2 views, 800 epochs): **85.54 / kNN 82.62 / h-rank 88.2** — +0.9 linear / +1.2 kNN over the a0 = 1
800-epoch run (84.64 / 81.44) and +4.0 over its own 200-epoch value (81.56).  At 4× the compute of a 200-epoch 2-view run it passes VICReg
(85.46, 200 ep) and is 0.55 below SimCLR (86.09, 200 ep); h-rank 88 now equals SimCLR's 90.  `a5_views4_800ep` (running, epoch ≈ 300/800)
is the last long run.

## Addendum 2026-09-26 12:50 UTC — a5_views4_800ep seed 0 final: the recipe passes the untuned controls
`P35_vcs_a5_views4_800ep_seed0` (cosine critic, K = 8, negative detach, a0 = 5, 4 views, B = 256, 800 epochs; 8× the 2-view 200-epoch compute):
**linear 86.42 / kNN 85.60 / h-rank 135.9** (z-rank 30; heldout-J 0.974; critic a 24.4 / b −22.1 → threshold 0.90; positive saturation 61 %).
kNN curve: 68.4 / 74.5 / 78.8 / 81.9 / 84.2 / 85.2 / 85.6 at epochs 20 / 50 / 100 / 200 / 400 / 600 / 800 — still rising by 0.4 per 200 epochs at the end.

| | linear | kNN | h-rank |
|---|---|---|---|
| VCS recipe, 4 views, 800 ep (this run, 1 seed) | **86.42** | **85.60** | 136 |
| VCS recipe, 2 views, 800 ep (P35, 1 seed) | 85.54 | 82.62 | 88 |
| SimCLR matched, 2 views, 200 ep (P5, 3 seeds; untuned) | 86.09 ± 0.38 | 83.98 | 90 |
| VICReg matched, 2 views, 200 ep (P5, 3 seeds; untuned) | 85.46 ± 0.26 | 81.92 | 76 |
| original VCS recipe, 200 ep (P5, 3 seeds) | 74.34 ± 0.46 | 63.93 | 13 |

Reading: first VCS run above both controls on both metrics (+0.3 linear, +1.6 kNN over SimCLR) — with the caveats that the controls are
untuned, at 200 epochs and 2 views (owner's equal-budget rule: P41 draft), and this is one seed (seeds 1/2 running, ≈ 10 h).  h-rank 136
is above SimCLR's 90; the critic keeps sharpening to a 0.90 threshold; kNN had not saturated at 800 epochs, which is the question the
1600-epoch chain (P43) answers.  `P35_vcs_a5_800ep_seed1` (2 views, 800 ep) finished training at kNN 82.20 (seed 0: 82.62); linear pending.
