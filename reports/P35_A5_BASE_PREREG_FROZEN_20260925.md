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
