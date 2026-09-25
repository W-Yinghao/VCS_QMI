# Watcher status — 2026-09-25 10:01:22 UTC

Stages: P16_vcs_hparamB_set, P18_vcs_critic_variants, P20_vcs_ssl_wiring, P22_vcs_target_branch.  Baseline for deltas: same-seed K=1 200-epoch run (P5).  Signal rule (pre-registered): HELPS if final linear-val > baseline + 1.0, HURTS if < baseline − 1.0, else neutral.

## P16_vcs_hparamB_set

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P16_vcs_k8_clr10_seed0 | K=8, critic_lr_multiplier=10.0 | COMPLETED | 200/200 | — | 68.40 (200) | 3.76 | 0.9397 | 21.44 | 77.44 | HELPS (+2.60) | — | 1008237 |
| P16_vcs_k8_clr3_seed0 | K=8, critic_lr_multiplier=3.0 | COMPLETED | 200/200 | — | 67.32 (200) | 2.68 | 0.9310 | 17.42 | 76.80 | HELPS (+1.96) | — | 1008236 |
| P16_vcs_k8_aug_strong_seed0 | K=8, crop_scale_min=0.08, color_jitter=[0.8, 0.8, 0.8, 0.2], gaussian_blur_p=0.5 | COMPLETED | 200/200 | — | 68.96 (200) | 4.32 | 0.8212 | 18.36 | 76.70 | HELPS (+1.86) | — | 1008241 |
| P16_vcs_aug_cj08_seed0 | color_jitter=[0.8, 0.8, 0.8, 0.2] | COMPLETED | 200/200 | — | 65.38 (200) | 0.74 | 0.8844 | 15.54 | 75.14 | neutral (+0.30) | — | 1008227 |
| P16_vcs_wd1e-5_seed0 | matrix_weight_decay=1e-05 | COMPLETED | 200/200 | — | 63.60 (200) | -1.04 | 0.8956 | 13.12 | 74.86 | neutral (+0.02) | — | 1008234 |
| P16_vcs_b512_seed0 | batch_size_images=512 | COMPLETED | 200/200 | — | 63.60 (200) | -1.04 | 0.8973 | 12.52 | 74.34 | neutral (-0.50) | — | 1008232 |
| P16_vcs_aug_crop008_seed0 | crop_scale_min=0.08 | COMPLETED | 200/200 | — | 63.12 (200) | -1.52 | 0.8269 | 13.80 | 74.04 | neutral (-0.80) | — | 1008226 |
| P16_vcs_wd5e-4_seed0 | matrix_weight_decay=0.0005 | COMPLETED | 200/200 | — | 63.38 (200) | -1.26 | 0.8980 | 13.74 | 73.84 | HURTS (-1.00) | — | 1008235 |
| P16_vcs_aug_blur05_seed0 | gaussian_blur_p=0.5 | COMPLETED | 200/200 | — | 64.90 (200) | 0.26 | 0.8928 | 14.21 | 73.64 | HURTS (-1.20) | — | 1008228 |
| P16_vcs_aug_strong_seed0 | crop_scale_min=0.08, color_jitter=[0.8, 0.8, 0.8, 0.2], gaussian_blur_p=0.5 | COMPLETED | 200/200 | — | 65.32 (200) | 0.68 | 0.7873 | 14.78 | 73.36 | HURTS (-1.48) | — | 1008229 |
| P16_vcs_b128_seed0 | batch_size_images=128 | COMPLETED | 200/200 | — | 63.68 (200) | -0.96 | 0.9036 | 14.03 | 73.22 | HURTS (-1.62) | — | 1008231 |
| P16_vcs_b1024_seed0 | batch_size_images=1024 | COMPLETED | 200/200 | — | 63.30 (200) | -1.34 | 0.8935 | 12.01 | 73.00 | HURTS (-1.84) | — | 1008233 |
| P16_vcs_aug_weak_seed0 | crop_scale_min=0.5, color_jitter=[0.2, 0.2, 0.2, 0.05] | COMPLETED | 200/200 | — | 55.20 (200) | -9.44 | 0.9746 | 12.42 | 67.78 | HURTS (-7.06) | — | 1008230 |
| P16_vcs_k8_clr10_800ep_seed0 | K=8, critic_lr_multiplier=10.0 | RUNNING | 456/800 | 182 | 72.14 (400) | — | 0.9497 | 31.67 | — | — | — | 1008239 |

kNN trajectories:

- P16_vcs_aug_blur05_seed0: ep0: 36.6, ep10: 41.4, ep20: 47.8, ep50: 56.4, ep100: 61.9, ep150: 64.1, ep200: 64.9
- P16_vcs_aug_cj08_seed0: ep0: 36.6, ep10: 44.3, ep20: 49.6, ep50: 58.0, ep100: 62.9, ep150: 64.8, ep200: 65.4
- P16_vcs_aug_crop008_seed0: ep0: 36.6, ep10: 41.7, ep20: 46.4, ep50: 53.8, ep100: 60.8, ep150: 62.4, ep200: 63.1
- P16_vcs_aug_strong_seed0: ep0: 36.6, ep10: 40.2, ep20: 47.6, ep50: 56.8, ep100: 62.1, ep150: 65.1, ep200: 65.3
- P16_vcs_aug_weak_seed0: ep0: 36.6, ep10: 39.9, ep20: 43.4, ep50: 48.8, ep100: 51.7, ep150: 54.4, ep200: 55.2
- P16_vcs_b1024_seed0: ep0: 36.6, ep10: 39.2, ep20: 44.1, ep50: 52.3, ep100: 59.8, ep150: 62.6, ep200: 63.3
- P16_vcs_b128_seed0: ep0: 36.6, ep10: 42.3, ep20: 48.5, ep50: 56.8, ep100: 61.0, ep150: 63.3, ep200: 63.7
- P16_vcs_b512_seed0: ep0: 36.6, ep10: 41.2, ep20: 47.5, ep50: 55.6, ep100: 61.1, ep150: 63.4, ep200: 63.6
- P16_vcs_k8_aug_strong_seed0: ep0: 36.6, ep10: 45.0, ep20: 53.2, ep50: 60.8, ep100: 65.9, ep150: 68.5, ep200: 69.0
- P16_vcs_k8_clr10_800ep_seed0: ep0: 36.6, ep20: 57.4, ep50: 64.1, ep100: 67.0, ep200: 70.2, ep400: 72.1
- P16_vcs_k8_clr10_seed0: ep0: 36.6, ep10: 51.8, ep20: 56.9, ep50: 64.0, ep100: 66.7, ep150: 68.4, ep200: 68.4
- P16_vcs_k8_clr3_seed0: ep0: 36.6, ep10: 47.7, ep20: 53.6, ep50: 63.5, ep100: 66.0, ep150: 67.1, ep200: 67.3
- P16_vcs_wd1e-5_seed0: ep0: 36.6, ep10: 42.1, ep20: 48.0, ep50: 56.6, ep100: 61.5, ep150: 63.5, ep200: 63.6
- P16_vcs_wd5e-4_seed0: ep0: 36.6, ep10: 41.4, ep20: 47.8, ep50: 56.4, ep100: 60.8, ep150: 63.4, ep200: 63.4

## P18_vcs_critic_variants

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P18_vcs_crit_cosine_seed0 | K=8, critic_input=cosine | COMPLETED | 200/200 | — | 72.76 (200) | 5.32 | 0.9676 | 57.74 | 78.32 | HELPS (+2.20) | — | 1008403 |
| P18_vcs_crit_interact_seed0 | K=8, critic_input=concat_interact | COMPLETED | 200/200 | — | 70.74 (200) | 3.30 | 0.9623 | 39.28 | 77.34 | HELPS (+1.22) | — | 1008401 |
| P18_vcs_pair_sym_seed0 | K=8, pair_symmetric=True | COMPLETED | 200/200 | — | 67.60 (200) | 0.16 | 0.9307 | 17.00 | 77.08 | neutral (+0.96) | — | 1008404 |
| P18_vcs_crit_steps2_seed0 | K=8, critic_steps=2 | COMPLETED | 200/200 | — | 67.54 (200) | 0.10 | 0.9312 | 18.06 | 76.76 | neutral (+0.64) | — | 1008405 |
| P18_vcs_crit_bilinear_seed0 | K=8, critic_input=bilinear_concat | COMPLETED | 200/200 | — | 66.76 (200) | -0.68 | 0.9306 | 16.68 | 76.18 | neutral (+0.06) | — | 1008402 |
| P18_vcs_crit_steps5_seed0 | K=8, critic_steps=5 | COMPLETED | 200/200 | — | 68.80 (200) | 1.36 | 0.9371 | 21.08 | 75.62 | neutral (-0.50) | — | 1008406 |

kNN trajectories:

- P18_vcs_crit_bilinear_seed0: ep0: 36.6, ep10: 47.0, ep20: 53.0, ep50: 61.2, ep100: 65.0, ep150: 66.6, ep200: 66.8
- P18_vcs_crit_cosine_seed0: ep0: 36.6, ep10: 56.5, ep20: 64.0, ep50: 69.2, ep100: 71.7, ep150: 72.8, ep200: 72.8
- P18_vcs_crit_interact_seed0: ep0: 36.6, ep10: 56.1, ep20: 61.4, ep50: 67.0, ep100: 69.7, ep150: 69.8, ep200: 70.7
- P18_vcs_crit_steps2_seed0: ep0: 36.6, ep10: 49.9, ep20: 54.9, ep50: 61.7, ep100: 65.0, ep150: 67.2, ep200: 67.5
- P18_vcs_crit_steps5_seed0: ep0: 36.6, ep10: 50.8, ep20: 56.1, ep50: 62.9, ep100: 65.9, ep150: 68.4, ep200: 68.8
- P18_vcs_pair_sym_seed0: ep0: 36.6, ep10: 46.9, ep20: 53.8, ep50: 61.7, ep100: 65.0, ep150: 66.9, ep200: 67.6

## P20_vcs_ssl_wiring

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P20_vcs_crit_on_h_seed0 | K=8, critic_lr_multiplier=10.0, critic_feature_source=h_l2 | RUNNING | 30/200 | 80 | 56.18 (20) | — | 0.7923 | 13.81 | — | — | — | 1008560 |
| P20_vcs_crit_steps5_seed0 | K=8, critic_lr_multiplier=10.0, critic_steps=5 | RUNNING | 17/200 | 105 | 52.00 (10) | — | 0.7056 | 15.50 | — | — | — | 1008562 |
| P20_vcs_neg_detach_seed0 | K=8, critic_lr_multiplier=10.0, negative_detach=True | RUNNING | 23/200 | 83 | 38.02 (20) | — | 0.0000 | 1.06 | — | — | COLLAPSE_SUSPECTED | 1008561 |

kNN trajectories:

- P20_vcs_crit_on_h_seed0: ep0: 36.6, ep10: 50.4, ep20: 56.2
- P20_vcs_crit_steps5_seed0: ep0: 36.6, ep10: 52.0
- P20_vcs_neg_detach_seed0: ep0: 36.6, ep10: 37.5, ep20: 38.0

## P22_vcs_target_branch

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P22_vcs_ema0.996_seed0 | K=8, critic_lr_multiplier=10.0, target_branch=ema_0.996 | RUNNING | 5/200 | 128 | 36.58 (0) | 0.00 | -0.0000 | 2.94 | — | — | — | 1008574 |
| P22_vcs_ema0.99_seed0 | K=8, critic_lr_multiplier=10.0, target_branch=ema_0.99 | RUNNING | 8/200 | 126 | 36.58 (0) | 0.00 | -0.0000 | 2.94 | — | — | — | 1008573 |
| P22_vcs_sg_pred_seed0 | K=8, critic_lr_multiplier=10.0, target_branch=stopgrad, predictor=True | RUNNING | 4/200 | 104 | 36.58 (0) | 0.00 | -0.0000 | 2.94 | — | — | — | 1008577 |
| P22_vcs_stopgrad_seed0 | K=8, critic_lr_multiplier=10.0, target_branch=stopgrad | RUNNING | 6/200 | 102 | 36.58 (0) | 0.00 | -0.0000 | 2.94 | — | — | — | 1008576 |

---
event_key: done=19 mon_evals=170 flags=1
