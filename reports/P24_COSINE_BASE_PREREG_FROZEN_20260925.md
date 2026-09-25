# Pre-registration — single factors on the new cosine-critic base (P24–P25), frozen 2026-09-25 before launch

P19 established that the critic's function form was the limiter: cosine critic tanh(a·⟨z1,z2⟩+b) with K=8 gives 78.32 % / kNN 72.76 /
h-rank 57.7 (`P18_vcs_crit_cosine_seed0`).  This wave re-bases the still-open single factors on that configuration.  The plan's content
(J, mixture reference, product-of-marginals cyclic-shift negatives, tanh output) is unchanged; f_θ = a·⟨z1,z2⟩+b is a permitted critic class.

## Units (seed 0, 200 epochs unless stated; base `configs/cifar10_hpC_crit_cosine_vcs_seed0.yaml`: cosine critic, K=8, critic lr ×1, scale init 1)
| unit | change | question |
|---|---|---|
| cos_k64 / cos_k255 | K = 64 / 255 | with a similarity critic, do more negatives now matter (SimCLR-like regime)? |
| cos_clr10 | critic lr ×10 | do the two critic scalars need to move faster? |
| cos_scale10 | initial a = 10 (`model.critic.cosine_scale_init`) | temperature-like initial sharpness |
| cos_on_h | critic reads L2-normalized h (512-d) | is the projector still useful with a similarity critic? |
| cos_negdetach | shifted partner detached in negatives | gradient routing |
| cos_ema0.99 / cos_ema0.996 / cos_stopgrad / cos_sg_pred / cos_ema0.99_pred | target-branch variants (re-based from P22) | BYOL/SimSiam-style wiring with the VCS objective |
| cos_proj_depth3 | 3-layer projector | |
| cos_800ep | 800 epochs (checkpoints 100/200/400/600/800) | best-achievable estimate on the new base; compared with P8/P16 800-epoch runs |

Configs and sha256: `configs/HPARAM_F_COSINE_SHA256.json`.  The seven P20/P22 units that had already started on the concat-MLP base are left to
finish and are reported as such; their two pending units and summarize jobs were cancelled as superseded.

## Reading
Primary: final linear-val vs 78.32 (HELPS > +1.0 / HURTS < −1.0 / neutral); secondary: kNN, h-rank, heldout-J, saturation, learned scale a
and bias b (logged in the critic state), collapse flag.  cos_800ep is compared with `P8_vcs800_seed0` (79.24) and `P16_vcs_k8_clr10_800ep_seed0`.

## Not claimed
Single seed; selection set; controls untouched.
