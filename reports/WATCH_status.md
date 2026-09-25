# Watcher status — 2026-09-25 02:27:17 UTC

Stages: P12_vcs_hparamA, P14_vcs_hparamB_lr, P8_long800.  Baseline for deltas: same-seed K=1 200-epoch run (P5).  Signal rule (pre-registered): HELPS if final linear-val > baseline + 1.0, HURTS if < baseline − 1.0, else neutral.

## P12_vcs_hparamA

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P12_vcs_clr10_seed0 | critic_lr_multiplier=10.0 | COMPLETED | 200/200 | — | 66.56 (200) | 1.92 | 0.9176 | 17.70 | 75.08 | neutral (+0.24) | — | 1007980 |
| P12_vcs_cw2048_seed0 | critic_hidden_dims=[2048, 2048] | COMPLETED | 200/200 | — | 63.84 (200) | -0.80 | 0.9099 | 15.15 | 74.80 | neutral (-0.04) | — | 1007976 |
| P12_vcs_po256_seed0 | projector_output_dim=256 | COMPLETED | 200/200 | — | 63.98 (200) | -0.66 | 0.9042 | 13.53 | 74.52 | neutral (-0.32) | — | 1007979 |
| P12_vcs_cw256_seed0 | critic_hidden_dims=[256, 256] | COMPLETED | 200/200 | — | 64.30 (200) | -0.34 | 0.8962 | 12.51 | 74.40 | neutral (-0.44) | — | 1007977 |
| P12_vcs_cw1024_seed0 | critic_hidden_dims=[1024, 1024] | COMPLETED | 200/200 | — | 63.80 (200) | -0.84 | 0.9068 | 14.63 | 74.14 | neutral (-0.70) | — | 1007985 |
| P12_vcs_ph2048_seed0 | projector_hidden_dim=2048 | COMPLETED | 200/200 | — | 64.06 (200) | -0.58 | 0.8976 | 12.98 | 74.04 | neutral (-0.80) | — | 1007987 |
| P12_vcs_cd3_seed0 | critic_hidden_dims=[512, 512, 512] | COMPLETED | 200/200 | — | 63.70 (200) | -0.94 | 0.9033 | 14.27 | 74.00 | neutral (-0.84) | — | 1007983 |
| P12_vcs_po512_seed0 | projector_output_dim=512 | COMPLETED | 200/200 | — | 64.04 (200) | -0.60 | 0.9046 | 13.44 | 73.62 | HURTS (-1.22) | — | 1007978 |
| P12_vcs_cd1_seed0 | critic_hidden_dims=[512] | COMPLETED | 200/200 | — | 62.52 (200) | -2.12 | 0.8911 | 11.76 | 73.32 | HURTS (-1.52) | — | 1007984 |
| P12_vcs_cw128_seed0 | critic_hidden_dims=[128, 128] | COMPLETED | 200/200 | — | 62.86 (200) | -1.78 | 0.8923 | 11.88 | 73.16 | HURTS (-1.68) | — | 1007986 |
| P12_vcs_clr0.1_seed0 | critic_lr_multiplier=0.1 | COMPLETED | 200/200 | — | 62.66 (200) | -1.98 | 0.8882 | 11.44 | 72.86 | HURTS (-1.98) | — | 1007981 |
| P12_vcs_clr0.3_seed0 | critic_lr_multiplier=0.3 | RUNNING | 143/200 | 30 | 58.36 (100) | -2.52 | 0.8455 | 10.82 | — | — | — | 1007990 |
| P12_vcs_clr3_seed0 | critic_lr_multiplier=3.0 | RUNNING | 149/200 | 27 | 62.90 (100) | 2.02 | 0.8658 | 14.27 | — | — | — | 1007989 |
| P12_vcs_cwd1e-3_seed0 | critic_weight_decay=0.001 | RUNNING | 137/200 | 33 | 60.60 (100) | -0.28 | 0.8608 | 12.26 | — | — | — | 1008008 |
| P12_vcs_cwd1e-4_seed0 | critic_weight_decay=0.0001 | RUNNING | 137/200 | 34 | 60.84 (100) | -0.04 | 0.8595 | 11.59 | — | — | — | 1008006 |
| P12_vcs_gain0.01_seed0 | critic_last_gain=0.01 | RUNNING | 4/200 | 101 | 36.58 (0) | 0.04 | 0.0000 | 2.94 | — | — | — | 1008010 |
| P12_vcs_gain1.0_seed0 | critic_last_gain=1.0 | RUNNING | 20/200 | 93 | 50.74 (20) | 3.32 | 0.7254 | 10.51 | — | — | — | 1008009 |
| P12_vcs_ph1024_seed0 | projector_hidden_dim=1024 | RUNNING | 170/200 | 16 | 63.16 (150) | -0.62 | 0.8882 | 12.66 | — | — | — | 1007988 |
| P12_vcs_po64_seed0 | projector_output_dim=64 | RUNNING | 141/200 | 31 | 60.46 (100) | -0.42 | 0.8606 | 12.29 | — | — | — | 1008005 |

kNN trajectories:

- P12_vcs_cd1_seed0: ep0: 36.6, ep10: 37.6, ep20: 41.6, ep50: 52.6, ep100: 59.4, ep150: 62.3, ep200: 62.5
- P12_vcs_cd3_seed0: ep0: 36.6, ep10: 43.3, ep20: 49.6, ep50: 56.1, ep100: 61.6, ep150: 63.7, ep200: 63.7
- P12_vcs_clr0.1_seed0: ep0: 36.6, ep10: 37.2, ep20: 40.4, ep50: 52.9, ep100: 59.6, ep150: 61.5, ep200: 62.7
- P12_vcs_clr0.3_seed0: ep0: 36.6, ep10: 37.5, ep20: 42.5, ep50: 51.8, ep100: 58.4
- P12_vcs_clr10_seed0: ep0: 36.6, ep10: 48.4, ep20: 52.9, ep50: 60.3, ep100: 64.1, ep150: 66.0, ep200: 66.6
- P12_vcs_clr3_seed0: ep0: 36.6, ep10: 45.0, ep20: 51.1, ep50: 56.4, ep100: 62.9
- P12_vcs_cw1024_seed0: ep0: 36.6, ep10: 44.1, ep20: 49.6, ep50: 57.5, ep100: 61.8, ep150: 63.6, ep200: 63.8
- P12_vcs_cw128_seed0: ep0: 36.6, ep10: 38.7, ep20: 44.9, ep50: 55.0, ep100: 60.0, ep150: 63.0, ep200: 62.9
- P12_vcs_cw2048_seed0: ep0: 36.6, ep10: 44.7, ep20: 50.0, ep50: 56.1, ep100: 62.4, ep150: 63.9, ep200: 63.8
- P12_vcs_cw256_seed0: ep0: 36.6, ep10: 41.5, ep20: 46.7, ep50: 54.0, ep100: 60.1, ep150: 63.7, ep200: 64.3
- P12_vcs_cwd1e-3_seed0: ep0: 36.6, ep10: 41.8, ep20: 47.7, ep50: 56.3, ep100: 60.6
- P12_vcs_cwd1e-4_seed0: ep0: 36.6, ep10: 42.5, ep20: 47.7, ep50: 55.6, ep100: 60.8
- P12_vcs_gain1.0_seed0: ep0: 36.6, ep10: 44.9, ep20: 50.7
- P12_vcs_ph1024_seed0: ep0: 36.6, ep10: 42.0, ep20: 47.6, ep50: 56.8, ep100: 60.7, ep150: 63.2
- P12_vcs_ph2048_seed0: ep0: 36.6, ep10: 42.1, ep20: 47.2, ep50: 54.9, ep100: 59.9, ep150: 63.7, ep200: 64.1
- P12_vcs_po256_seed0: ep0: 36.6, ep10: 41.8, ep20: 47.6, ep50: 55.7, ep100: 61.2, ep150: 64.2, ep200: 64.0
- P12_vcs_po512_seed0: ep0: 36.6, ep10: 42.8, ep20: 45.4, ep50: 55.4, ep100: 61.7, ep150: 63.5, ep200: 64.0
- P12_vcs_po64_seed0: ep0: 36.6, ep10: 42.2, ep20: 47.6, ep50: 56.5, ep100: 60.5

## P8_long800

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P8_vcs800_seed0 | baseline recipe | COMPLETED | 800/800 | — | 71.36 (800) | — | 0.9479 | 20.13 | 79.24 | HELPS (+4.40) | — | 1007959 |
| P8_vcs800_seed2 | baseline recipe | COMPLETED | 800/800 | — | 71.60 (800) | — | 0.9469 | 20.80 | 79.06 | HELPS (+5.12) | — | 1007961 |
| P8_vcs800_seed1 | baseline recipe | COMPLETED | 800/800 | — | 71.60 (800) | — | 0.9457 | 20.69 | 78.48 | HELPS (+4.24) | — | 1007960 |

kNN trajectories:

- P8_vcs800_seed0: ep0: 36.6, ep20: 47.4, ep50: 55.5, ep100: 60.6, ep200: 65.3, ep400: 69.0, ep600: 71.2, ep800: 71.4
- P8_vcs800_seed1: ep0: 37.4, ep20: 49.6, ep50: 56.1, ep100: 60.6, ep200: 64.5, ep400: 68.3, ep600: 70.6, ep800: 71.6
- P8_vcs800_seed2: ep0: 37.1, ep20: 49.1, ep50: 57.3, ep100: 61.5, ep200: 64.7, ep400: 68.7, ep600: 71.2, ep800: 71.6

---
event_key: done=14 mon_evals=150 flags=0
