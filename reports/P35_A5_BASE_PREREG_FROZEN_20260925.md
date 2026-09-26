# Pre-registration — first units on the a0 = 5 base (P35), frozen 2026-09-25 before launch

Trigger: P31 seeds confirm the initial cosine scale.  a0 = 5 (learnable), cosine critic, K = 8, negative detach:
**81.90 / 81.06 / 81.72** (mean 81.56 ± 0.44), kNN 77.00 / 76.98 / 76.84 (76.94 ± 0.09), h-rank 45–46 — vs the a0 = 1 base
80.59 ± 0.12 / 74.41 ± 0.19 / 30.4.  Linear +1.0 (≈ 3.7 pooled SD), kNN +2.5, rank +50 %.  This is the new base
(`cifar10_hpG_a5_learn_vcs_seed0.yaml`).  Queued now because a GPU slot is free (owner: do not idle GPUs); the a0 sweep (2, 10, 20) is
still running and may move the base's a0 later.

| unit | change | question |
|---|---|---|
| a5_views4 | 4 views (J over the 6 view pairs, K = 8 each), 200 epochs | do the two positive factors stack?  `P28_vcs_views4_seed0` (a0 = 1) is at kNN 80.4 at epoch 150 (+6.1 over its base) while still running — read together with its final and with the equal-compute P33 unit |
| a5_800ep | 800 epochs (checkpoints 100/200/400/600/800; kNN 0/20/50/100/200/400/600/800; linear at 800) | best-achievable estimate on the new base; compared with `P26_vcs_base_800ep` (a0 = 1, running) and `P24_vcs_cos_800ep` (80.74) |

## Reading
a5_views4: final linear vs 81.56 (HELPS > +1.0 / HURTS < −1.0), plus the equal-epoch comparison with `P28_vcs_views4_seed0` (a0 = 1,
4 views): the difference isolates a0 = 5 under 4 views.  a5_800ep: vs `P26_vcs_base_800ep` at 800; the kNN curve at 200/400/600/800.
Single seed.  No code change (configs only; last gate 1009177 on commit 9a1a3c0+).  sha256: `configs/HPARAM_K_SHA256.json`.

## Addendum 21:40 UTC (before any a5_views4 result) — seeds 1/2 of a5_views4
`P28_vcs_views4_seed0` (a0 = 1, 4 views) finished at **84.48 / kNN 81.10 / h-rank 57.4** (+3.9 over the a0 = 1 base, the largest single
factor so far; at 2× encoder compute per epoch — the equal-compute P33 unit is at epoch 58/100).  `P35_vcs_a5_views4_seed0` is at epoch 22
with kNN +7.6 over the a0 = 5 base at the same epoch.  Because the base gets seeds and a0 = 5 + 4 views is the expected next base, seeds 1/2
of `a5_views4` are queued now (same config, seed only) rather than after seed 0 finishes (≈ 3 h), so the quota is not left idle.
Reading: mean ± SD over 3 seeds vs 81.56 (a0 = 5, 2 views) and vs 84.48 (a0 = 1, 4 views, single seed).  Configs in `HPARAM_K_SHA256.json`.

## Addendum 2026-09-26 — a5_views4_800ep (queued before any a5_views4 result; a GPU slot was free)
`P26_vcs_base_800ep_seed0` (a0 = 1, 2 views, 800 ep) finished at **84.64 / kNN 81.44 / h-rank 61**: the detach gain grows with the schedule
(+2.2 at 200 ep → +3.9 over plain cosine at 800 ep) and kNN was still rising at 800.  4 views at 200 ep gives 84.48 at half that compute.
Unit `a5_views4_800ep`: a0 = 5, 4 views, 800 epochs (checkpoints 100/200/400/600/800; kNN 0/20/50/100/200/400/600/800; linear at 800;
≈ 4× the compute of the 4-view 200-epoch run, ≈ 14 h).  Question: best-achievable estimate of the full recipe, and whether the schedule gain
and the views gain stack (read against 84.64 at equal wall-clock/epochs × 2 compute, and against `P35_vcs_a5_800ep`, 2 views).
Single seed; HELPS/HURTS not applied (no same-compute base); reported as an absolute number with its compute.  Config sha in `HPARAM_K_SHA256.json`.

## Addendum 2026-09-26 06:25 UTC — seeds 1/2 of a5_800ep
`P35_vcs_a5_800ep_seed0` (a0 = 5, 2 views, 800 epochs) = 85.54 / kNN 82.62 / h-rank 88 — the recipe's long-schedule headline at 2-view
cost.  Two GPU slots are idle and the 200-epoch numbers are all on 3 seeds; the 800-epoch 2-view number gets seeds 1/2 now (≈ 6.2 h each) so
it can be quoted as mean ± SD.  Reading: mean ± SD; compared with `P26_vcs_base_800ep` (a0 = 1, 84.64, single seed) and the 200-epoch
controls.  Configs in `HPARAM_K_SHA256.json`.
