# Watcher status — 2026-09-25 00:12:44 UTC

Stages: P12_vcs_hparamA, P8_long800.  Baseline for deltas: same-seed K=1 200-epoch run (P5).  Signal rule (pre-registered): HELPS if final linear-val > baseline + 1.0, HURTS if < baseline − 1.0, else neutral.

## P12_vcs_hparamA

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P12_vcs_cw2048_seed0 | critic_hidden_dims=[2048, 2048] | COMPLETED | 200/200 | — | 63.84 (200) | -0.80 | 0.9099 | 15.15 | 74.80 | neutral (-0.04) | — | 1007976 |
| P12_vcs_po256_seed0 | projector_output_dim=256 | COMPLETED | 200/200 | — | 63.98 (200) | -0.66 | 0.9042 | 13.53 | 74.52 | neutral (-0.32) | — | 1007979 |
| P12_vcs_cw256_seed0 | critic_hidden_dims=[256, 256] | COMPLETED | 200/200 | — | 64.30 (200) | -0.34 | 0.8962 | 12.51 | 74.40 | neutral (-0.44) | — | 1007977 |
| P12_vcs_po512_seed0 | projector_output_dim=512 | COMPLETED | 200/200 | — | 64.04 (200) | -0.60 | 0.9046 | 13.44 | 73.62 | HURTS (-1.22) | — | 1007978 |
| P12_vcs_cd1_seed0 | critic_hidden_dims=[512] | RUNNING | 97/200 | 55 | 52.56 (50) | -2.04 | 0.7836 | 9.17 | — | — | — | 1007984 |
| P12_vcs_cd3_seed0 | critic_hidden_dims=[512, 512, 512] | RUNNING | 99/200 | 53 | 56.10 (50) | 1.50 | 0.8036 | 11.48 | — | — | — | 1007983 |
| P12_vcs_clr0.1_seed0 | critic_lr_multiplier=0.1 | RUNNING | 105/200 | 50 | 59.64 (100) | -1.24 | 0.8419 | 10.03 | — | — | — | 1007981 |
| P12_vcs_clr10_seed0 | critic_lr_multiplier=10.0 | RUNNING | 131/200 | 37 | 64.06 (100) | 3.18 | 0.8794 | 16.39 | — | — | — | 1007980 |
| P12_vcs_cw1024_seed0 | critic_hidden_dims=[1024, 1024] | RUNNING | 97/200 | 55 | 57.52 (50) | 2.92 | 0.8200 | 11.60 | — | — | — | 1007985 |
| P12_vcs_cw128_seed0 | critic_hidden_dims=[128, 128] | RUNNING | 97/200 | 54 | 55.02 (50) | 0.42 | 0.7833 | 8.96 | — | — | — | 1007986 |

kNN trajectories:

- P12_vcs_cd1_seed0: ep0: 36.6, ep10: 37.6, ep20: 41.6, ep50: 52.6
- P12_vcs_cd3_seed0: ep0: 36.6, ep10: 43.3, ep20: 49.6, ep50: 56.1
- P12_vcs_clr0.1_seed0: ep0: 36.6, ep10: 37.2, ep20: 40.4, ep50: 52.9, ep100: 59.6
- P12_vcs_clr10_seed0: ep0: 36.6, ep10: 48.4, ep20: 52.9, ep50: 60.3, ep100: 64.1
- P12_vcs_cw1024_seed0: ep0: 36.6, ep10: 44.1, ep20: 49.6, ep50: 57.5
- P12_vcs_cw128_seed0: ep0: 36.6, ep10: 38.7, ep20: 44.9, ep50: 55.0
- P12_vcs_cw2048_seed0: ep0: 36.6, ep10: 44.7, ep20: 50.0, ep50: 56.1, ep100: 62.4, ep150: 63.9, ep200: 63.8
- P12_vcs_cw256_seed0: ep0: 36.6, ep10: 41.5, ep20: 46.7, ep50: 54.0, ep100: 60.1, ep150: 63.7, ep200: 64.3
- P12_vcs_po256_seed0: ep0: 36.6, ep10: 41.8, ep20: 47.6, ep50: 55.7, ep100: 61.2, ep150: 64.2, ep200: 64.0
- P12_vcs_po512_seed0: ep0: 36.6, ep10: 42.8, ep20: 45.4, ep50: 55.4, ep100: 61.7, ep150: 63.5, ep200: 64.0

## P8_long800

| run | changed | status | epoch | ETA (min) | last kNN (ep) | Δ kNN vs base @same ep | heldout-J | h-rank | final linear | signal | flags | job |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P8_vcs800_seed0 | baseline recipe | COMPLETED | 800/800 | — | 71.36 (800) | — | 0.9479 | 20.13 | 79.24 | HELPS (+4.40) | — | 1007959 |
| P8_vcs800_seed1 | baseline recipe | RUNNING | 769/800 | 12 | 70.58 (600) | — | 0.9406 | 20.64 | — | — | — | 1007960 |
| P8_vcs800_seed2 | baseline recipe | RUNNING | 556/800 | 127 | 68.72 (400) | — | 0.9222 | 19.45 | — | — | — | 1007961 |

kNN trajectories:

- P8_vcs800_seed0: ep0: 36.6, ep20: 47.4, ep50: 55.5, ep100: 60.6, ep200: 65.3, ep400: 69.0, ep600: 71.2, ep800: 71.4
- P8_vcs800_seed1: ep0: 37.4, ep20: 49.6, ep50: 56.1, ep100: 60.6, ep200: 64.5, ep400: 68.3, ep600: 70.6
- P8_vcs800_seed2: ep0: 37.1, ep20: 49.1, ep50: 57.3, ep100: 61.5, ep200: 64.7, ep400: 68.7

---
event_key: done=5 mon_evals=80 flags=0
