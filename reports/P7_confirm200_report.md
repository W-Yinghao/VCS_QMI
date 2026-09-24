# P7 — CIFAR-10 200-epoch confirmation report (VCS-QMI, SimCLR-matched, VICReg-matched-128; seeds 0,1,2)

Status: **COMPLETE (P5 launched 2026-09-24 16:16 UTC, last run finished 18:51 UTC, P6 table 18:51 UTC); awaiting owner review.**
Neutral table: `reports/P6_confirm200_results_table.md`.  Pre-registration: `reports/P5_CONFIRM200_PREREG_FROZEN_20260924.md`.
Pilot report (20 epochs): `reports/P4_ssl_pilot_report.md`.  Contract: `VCS_QMI_SSL_Server_Agent_Spec_v1.md`.

## A. Implementation check (delta vs the pilot)

Code (`src/`, `reference/`) identical to the pilot commit `76cebd6`; `summarize.py` only gained a generic final-epoch lookup and
seed aggregation.  Configs derived from the frozen pilot configs by the six overrides listed in `configs/CONFIRM200_SHA256.json`
(`stage`, `seed`, `epochs=200`, `warmup_epochs=10`, checkpoint epochs, kNN epochs); T = 35,000 steps, W = 1,750; everything else unchanged
(B=256, AdamW 1e-3, wd 1e-4/0, cosine → 1 %, FP32, TF32 off, K=1, reference critic/losses, split `c35d7cd3…`).
Per seed, encoder/projector initial weights are bit-identical across the three methods (seed 0: `bdc3a4e3…`, seed 1: `94c2cb88…`,
seed 2: `4385db04…`); seed 0 is the pilot's initialization.  Runs were submitted in parallel (owner request), one GPU each; the
scheduler mixed GPU types (RTX PRO 6000 Blackwell for vcs s0, simclr s0, vicreg s2; A100-PCIE-40GB otherwise), so wall-clock costs are
compared only within a GPU type.  `code_dirty=True` in 8 run manifests refers to the untracked `reports/job_ids.json` at launch time;
`src/` matched commit `13559b9` (= `7e68404` for vicreg s2) for every run.

## B. Tests
No new gates; the P0–P2 gates of the pilot apply to this unchanged code.  All 9 runs COMPLETED (200/200 epochs, 35,000 steps,
8,960,000 base-image presentations each); no non-finite value, no OOM, no `COLLAPSE_SUSPECTED`, in-training evaluation left the
training RNG untouched at every one of the 7 monitored epochs of every run; every run has both `initial.pt` and `epoch_200.pt` evaluations.

## C. Run table (development selection set, 5,000 images; **no test accuracy**)

Per-method mean ± sample SD over seeds {0,1,2} (frozen `h`, fixed epoch-200 checkpoint, pilot probe protocol):

| method | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h eff-rank | heldout-J |
|---|---|---|---|---|---|---|
| **vcs_qmi** | **74.34 ± 0.46** | 42.19 ± 0.74 | +32.15 ± 1.08 | 63.93 ± 0.62 | 13.18 ± 0.30 | 0.899 ± 0.002 |
| simclr_matched | 86.09 ± 0.38 | 42.19 ± 0.74 | +43.90 ± 0.71 | 83.98 ± 0.24 | 89.96 ± 0.83 | — |
| vicreg_matched_128 | 85.46 ± 0.26 | 42.12 ± 0.75 | +43.34 ± 0.66 | 81.92 ± 0.40 | 75.66 ± 0.37 | — |

Per run:

| run | linear-val ep0 → 200 | kNN ep0 → 200 | h-rank ep0 → 200 | z-rank 200 | heldout-J 200 (±sd, 4 rep.) | train J_raw ep200 | GPU | train s |
|---|---|---|---|---|---|---|---|---|
| P5_vcs_seed0 | 41.94 → 74.84 | 36.54 → 64.64 | 2.94 → 12.87 | 7.14 | 0.8975 ± 0.0047 | 0.9021 | RTX PRO 6000 | 4560 |
| P5_vcs_seed1 | 41.60 → 74.24 | 37.42 → 63.64 | 3.22 → 13.46 | 7.02 | 0.8973 ± 0.0046 | 0.9058 | A100 | 5715 |
| P5_vcs_seed2 | 43.02 → 73.94 | 37.08 → 63.50 | 3.22 → 13.22 | 7.14 | 0.9013 ± 0.0077 | 0.9068 | A100 | 5634 |
| P5_simclr_seed0 | 41.94 → 86.44 | 36.54 → 84.02 | 2.94 → 90.86 | 76.8 | — | NT-Xent 2.138 | RTX PRO 6000 | 4506 |
| P5_simclr_seed1 | 41.60 → 85.68 | 37.42 → 83.72 | 3.22 → 89.80 | 76.6 | — | 2.140 | A100 | 6281 |
| P5_simclr_seed2 | 43.02 → 86.14 | 37.08 → 84.20 | 3.22 → 89.22 | 77.2 | — | 2.140 | A100 | 6229 |
| P5_vicreg_seed0 | 41.78 → 85.64 | 36.58 → 81.46 | 2.94 → 75.26 | 77.3 | — | inv .154 var .019 cov 2.00 | A100 | 6135 |
| P5_vicreg_seed1 | 41.60 → 85.16 | 37.42 → 82.18 | 3.22 → 75.97 | 77.3 | — | .153 / .019 / 2.01 | A100 | 6259 |
| P5_vicreg_seed2 | 42.98 → 85.58 | 37.08 → 82.12 | 3.22 → 75.75 | 77.9 | — | .153 / .018 / 1.99 | RTX PRO 6000 | 4533 |

kNN trajectories (epochs 0/10/20/50/100/150/200), seed means: VCS 37.0 / 42.3 / 48.4 / 55.7 / 61.0 / 63.4 / 63.9;
SimCLR 37.0 / 61.8 / 69.0 / 76.3 / 81.5 / 83.4 / 84.0; VICReg 37.0 / 59.5 / 68.1 / 75.8 / 79.8 / 81.5 / 81.9.
VCS `heldout_J` (seed means): 0 / 0.605 / 0.708 / 0.803 / 0.862 / 0.888 / 0.899.

Pilot comparison (descriptive only; different warm-up/horizon): VCS seed 0 at 20 epochs was 56.26 % linear / 41.08 % kNN / h-rank 5.8 /
J 0.68; at 200 epochs 74.84 % / 64.64 % / 12.9 / 0.90.  Controls moved 75.0 → 86.1 % and 75.6 → 85.5 %.

Cost (same GPU type): on the RTX PRO 6000, VCS 0.1295 s/step (1977 images/s) vs SimCLR 0.1279 (2001) vs VICReg 0.1287 (1989) → critic
overhead ≈ 1 %; on the A100-PCIE, VCS 0.159–0.161 s/step vs controls 0.173–0.177 (different nodes; not a controlled comparison).
Peak allocated memory 2.79 GB (RTX) / 3.68 GB (A100) for all methods.  Total ≈ 14 GPU-hours for the nine runs.

## D. Observations (3 seeds, 200 epochs; per the frozen reading grid)

**Encoder improvement (all 3 seeds positive → claim allowed).**  VCS frozen `h` improved from 42.2 ± 0.7 % to 74.3 ± 0.5 % linear-val
(Δ = +32.2 ± 1.1) and kNN from 37.0 to 63.9 ± 0.6 %, with seed SD < 1 point.  Under the identical matched recipe the controls reached
86.1 ± 0.4 % (SimCLR) and 85.5 ± 0.3 % (VICReg-128).  The VCS–control gap (≈ 11–12 linear points, ≈ 18–20 kNN points) is ~20× the seed
SD; it is a gap under this matched short recipe, not a verdict on the objective (no tuning was done for any method, K = 1 only).

**Critic vs features.**  `heldout_J` rose to 0.899 ± 0.002 and tracks training `J_raw` (0.902–0.907) within ≤ 0.009 in every seed:
no train/held-out gap in the critic objective at any monitored epoch.  Pair discrimination on unseen images therefore approaches the
bound while feature quality rises much more slowly ("pair discrimination without proportional transfer" persists at 200 epochs).
Scores are heavily saturated at the end: |t| > 0.95 for ≈ 76 % of positive and ≈ 85 % of negative held-out pairs.

**Dimensional concentration persists (pre-committed clause: rank < 10 / ≤ 5 directions not met literally, but the pattern holds).**
`h` effective rank 13.2 ± 0.3 versus 90.0 (SimCLR) and 75.7 (VICReg); `z_l2` effective rank 7.1 versus ≈ 77 for both controls.  In all
three VCS seeds `p_raw` has exactly **7** dominant eigen-directions followed by a cliff (≈ 550–590 → 36–90), up from 4 at 20 epochs.
Reading: the concentration is not a short-horizon artefact; it grows slowly with training (4 → 7 directions, h-rank 5.8 → 13) and does
not approach the controls within 200 epochs.  The hypothesis remains that with K = 1 and B = 256 the ordered-concat tanh critic separates
the positive pairs from the single shifted pairing using a low-dimensional projector subspace, leaving weak pressure to spread
information; K, critic capacity and B were not varied and no regularizer was added, so this is untested.

**Trajectory shape.**  VCS kNN is still rising at epoch 200 (+0.3 to +0.9 between 150 and 200 across seeds) while J is essentially flat
(0.888 → 0.899); the controls are also still rising slightly (+0.4 to +0.7).  Longer training might move VCS further, but the slope is small.

**Controls.**  Normal and tight: SimCLR-matched 86.1 ± 0.4 %, VICReg-matched-128 85.5 ± 0.3 %, smooth probe curves (±0.3 points over the
probe run), VICReg variance term ≈ 0.019 (no dimensional collapse), NT-Xent 2.14.  These are component-matched runs, not tuned native baselines.

**Probe protocol.**  As in the pilot, the un-normalized VCS `h` (norm ≈ 27 vs 8–10 for the controls) makes the SGD-0.1 probe noisy in its
first half (VCS curves swing 59–71 % before settling at 74 %); controls are flat.  The final-probe-epoch rule was applied uniformly.  A
common change (e.g. feature standardization for all methods) would be a protocol decision for the owner, not a VCS-specific fix.

**Reproducibility.**  Seed SD of the primary endpoint is 0.26–0.46 points for all methods, well above the ≈ 0.2-point hardware/float
floor measured in the pilot; GPU run-to-run nondeterminism (P2b) is therefore a minor part of the seed variance.

## E. Next step (one prioritized proposal; nothing is started)

The confirmation answers the pilot's two open questions: (1) the encoder does improve under the full schedule, in every seed, with small
seed variance; (2) the low-dimensional concentration of the VCS representation is persistent and is the dominant structural difference
from the controls.  The most informative single-factor next round is therefore **K (number of nonzero cyclic shifts per step)**: the
pairing reference already supports K > 1 with the same averaging rule (spec §6.1), it changes only the negative pool the critic sees,
and it directly tests the hypothesis behind the 7-direction concentration.  Proposal: VCS only, seeds {0,1,2}, 200 epochs, K ∈ {8, 64}
as two separately named configs (`K` is the only change), ≈ 9–10 GPU-hours in parallel, reading rule frozen before launch (h-rank,
number of dominant `p_raw` directions, linear-val, heldout-J vs K).  Alternative single factors, if the owner prefers: critic capacity, or
the common probe standardization question above.  Not proposed: regularizers, EMA, queues, CIFAR-100/ImageNet, test-set evaluation.

**Waiting for the owner's decision.**
