# Pre-registration — wave G on the neg-detach cosine base (P26–P27), frozen 2026-09-25 before launch

Owner: submit the independent tasks now, do not idle the GPUs.  Base = current best 200-epoch VCS run `P24_vcs_cos_negdetach_seed0`
(cosine critic tanh(a⟨z1,z2⟩+b), K=8, negative partner detached, critic lr ×1, a0=1, b0=0): **80.48 % / kNN 74.62 / h-rank 30.9 /
heldout-J 0.929** (standardized probe 79.98).  Candidates N1–N6 come from the external review
(`CS_QMI/VCS_QMI_SSL_Bottleneck_Review_20260925.md`, §3–§5, §8.C); the rest from this session.  The plan's content (J, mixture reference,
product-of-marginals cyclic-shift negatives, tanh output) is untouched.

| unit | change | single question |
|---|---|---|
| base_seed1 / base_seed2 | seeds 1, 2 of the base | seed anchor for the new base (owner allowed single-seed exploration; the *base* gets seeds) |
| b0_calib (N1) | `cosine_bias_calibrate=true`: b0 = −a0·μ, μ = ½(mean s_pos + mean s_neg) on the first 512 fit UIDs (fixed RNG, train-mode BN, buffers restored, record saved) | initial common offset (initial J ≈ −tanh²(1) = −0.58 suggests all pairs start in a narrow cone) |
| a5_learn (N2) | a0 = 5, learnable | initial sharpness |
| a5_fixed (N3) | a = 5 fixed (excluded from the optimizer), b learnable; same a0/b0 as N2 | late-training scale freedom (the base learned a ≈ 9.8, b ≈ −8.1 by epoch 200) |
| proj_outBN (N4) | last projector Linear(bias=False) + BatchNorm1d(128, affine=False) before L2 | output-coordinate scale control (cos_on_h showed the cosine critic leaves the pre-normalization scale unconstrained) |
| interact_only (N5) | MLP([z1⊙z2, \|z1−z2\|]) 256→256→1, no raw z channel | is the raw single-view channel what the concat MLP exploited? |
| shared_metric (N6) | tanh(a⟨norm(Wz1), norm(Wz2)⟩+b), W 128×128 init identity | does a shared learnable metric help beyond plain cosine? W spectrum logged via checkpoint |
| k255 | K = 255 | combine the two positive factors (K=255 gave +1.0 on the cosine base) |
| base_800ep | 800 epochs (checkpoints 100/200/400/600/800) | best-achievable estimate; compared with `P24_vcs_cos_800ep_seed0` and `P8_vcs800_seed0` |

Configs and sha256: `configs/HPARAM_G_NEGDETACH_SHA256.json`.  Code (commit recorded in `job_ids.json`): fixed scale, bias calibration,
affine-free output BN, `InteractOnlyCritic`, `SharedMetricCritic`; hold-out diagnostic follows the training wiring.  Controls stay locked.

## Reading
Primary: final linear-val vs 80.48 (HELPS > +1.0 / HURTS < −1.0 / neutral); seeds 1/2 give the base's SD (compare with 0.46 at K=1 MLP).
Secondary: kNN, h-rank, heldout-J, learned a/b trajectories (`cos_scale`, `cos_bias` in steps.jsonl), collapse flag, standardized probe.
N2/N3 are read as a pair (same a0/b0): if N3 ≥ N2, late scale freedom is not needed.  N4 is read together with ‖h‖ / ‖p_raw‖.

## Not claimed
Selection set; controls untouched; single seed except the base.
