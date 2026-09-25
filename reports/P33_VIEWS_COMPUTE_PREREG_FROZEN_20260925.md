# Pre-registration — equal-compute control for 4 views (P33), frozen 2026-09-25 before launch

Trigger: `P28_vcs_views4_seed0` (4 views, J averaged over the 6 view pairs, K = 8 each, neg-detach cosine base) is +7…+8 kNN over the
base at equal *epochs* (61.2 vs 53.9 at epoch 10; 69.0 vs 60.9 at epoch 20) while still running.  A 4-view step costs two encoder
forwards per image, so at equal epochs it has twice the encoder compute of the base.  The P28 pre-registration said views4 is also compared
at equal compute; this unit provides that comparison.

| unit | change vs `cifar10_hpH_views4_vcs_seed0.yaml` | question |
|---|---|---|
| views4_100ep | 100 epochs (warm-up 10 unchanged, cosine to 1 %); checkpoints 20/50/100; kNN 0/10/20/50/100; linear at 100 | at the encoder compute of the 200-epoch base (8.96 M image-forwards), does 4-view training beat 80.59 (base, 3 seeds)? |

## Reading
Primary: final linear-val vs the 3-seed base mean 80.59 (HELPS > +1.0 / HURTS < −1.0).  Read together with `P28_vcs_views4_seed0` at
200 epochs (equal epochs, double compute): HELPS at both = more positive pairs per step is a real gain; HELPS only at 200 = a compute effect.
Secondary: kNN, h-rank, heldout-J.  Single seed.  No code change (configs only); sha256 in `configs/HPARAM_J_SHA256.json`.
