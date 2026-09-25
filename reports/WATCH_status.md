# Watcher status — 2026-09-25 09:25:53 UTC

Stages: P16_vcs_hparamB_set, P18_vcs_critic_variants, P20_vcs_ssl_wiring.  Baseline for deltas: same-seed K=1 200-epoch run (P5).  Signal rule (pre-registered): HELPS if final linear-val > baseline + 1.0, HURTS if < baseline − 1.0, else neutral.

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
| P16_vcs_k8_clr10_800ep_seed0 | K=8, critic_lr_multiplier=10.0 | RUNNING | 392/800 | 216 | 70.16 (200) | 5.52 | 0.9291 | 26.11 | — | — | — | 1008239 |

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
- P16_vcs_k8_clr10_800ep_seed0: ep0: 36.6, ep20: 57.4, ep50: 64.1, ep100: 67.0, ep200: 70.2
- P16_vcs_k8_clr10_seed0: ep0: 36.6, ep10: 51.8, ep20: 56.9, ep50: 64.0, ep100: 66.7, ep150: 68.4, ep200: 68.4
- P16_vcs_k8_clr3_seed0: ep0: 36.6, ep10: 47.7, ep20: 53.6, ep50: 63.5, ep100: 66.0, ep150: 67.1, ep200: 67.3
- P16_vcs_wd1e-5_seed0: ep0: 36.6, ep10: 42.1, ep20: 48.0, ep50: 56.6, ep100: 61.5, ep150: 63.5, ep200: 63.6
- P16_vcs_wd5e-4_seed0: ep0: 36.6, ep10: 41.4, ep20: 47.8, ep50: 56.4, ep100: 60.8, ep150: 63.4, ep200: 63.4

## P18_vcs_critic_variants

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P18_vcs_crit_bilinear_seed0 | K=8, critic_input=bilinear_concat | RUNNING | 143/200 | 30 | 65.04 (100) | 0.28 | 0.9019 | 15.89 | — | — | — | 1008402 |
| P18_vcs_crit_cosine_seed0 | K=8, critic_input=cosine | RUNNING | 144/200 | 29 | 71.72 (100) | 6.96 | 0.9479 | 53.36 | — | — | — | 1008403 |
| P18_vcs_crit_interact_seed0 | K=8, critic_input=concat_interact | RUNNING | 144/200 | 29 | 69.70 (100) | 4.94 | 0.9419 | 37.79 | — | — | — | 1008401 |
| P18_vcs_crit_steps2_seed0 | K=8, critic_steps=2 | RUNNING | 155/200 | 22 | 67.22 (150) | 0.40 | 0.9250 | 17.49 | — | — | — | 1008405 |
| P18_vcs_crit_steps5_seed0 | K=8, critic_steps=5 | RUNNING | 147/200 | 27 | 65.92 (100) | 1.16 | 0.8985 | 20.56 | — | — | — | 1008406 |
| P18_vcs_pair_sym_seed0 | K=8, pair_symmetric=True | RUNNING | 153/200 | 23 | 66.88 (150) | 0.06 | 0.9244 | 16.29 | — | — | — | 1008404 |

kNN trajectories:

- P18_vcs_crit_bilinear_seed0: ep0: 36.6, ep10: 47.0, ep20: 53.0, ep50: 61.2, ep100: 65.0
- P18_vcs_crit_cosine_seed0: ep0: 36.6, ep10: 56.5, ep20: 64.0, ep50: 69.2, ep100: 71.7
- P18_vcs_crit_interact_seed0: ep0: 36.6, ep10: 56.1, ep20: 61.4, ep50: 67.0, ep100: 69.7
- P18_vcs_crit_steps2_seed0: ep0: 36.6, ep10: 49.9, ep20: 54.9, ep50: 61.7, ep100: 65.0, ep150: 67.2
- P18_vcs_crit_steps5_seed0: ep0: 36.6, ep10: 50.8, ep20: 56.1, ep50: 62.9, ep100: 65.9
- P18_vcs_pair_sym_seed0: ep0: 36.6, ep10: 46.9, ep20: 53.8, ep50: 61.7, ep100: 65.0, ep150: 66.9

---
event_key: done=13 mon_evals=141 flags=0
