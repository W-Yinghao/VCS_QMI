# P41_control_tuning — neutral results table (observed values only)

Generated 2026-09-28T13:36:28Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | simclr_matched | null | 0 | 800/800 | 88.38 | 87.12 | null | 149.07 | 12774 | 3644/4330 | COMPLETED |
| P41_simclr_800ep_seed1 | simclr_matched | null | 1 | 800/800 | 88.28 | 86.94 | null | 146.71 | 7908 | 2788/3518 | COMPLETED |
| P41_simclr_800ep_seed2 | simclr_matched | null | 2 | 800/800 | 88.38 | 87.72 | null | 149.29 | 7922 | 2788/3518 | COMPLETED |
| P41_simclr_b128_seed0 | simclr_matched | null | 0 | 200/200 | 86.36 | 83.96 | null | 92.13 | 3363 | 2215/3646 | COMPLETED |
| P41_simclr_tau0.1_seed0 | simclr_matched | null | 0 | 200/200 | 84.30 | 80.16 | null | 79.46 | 1989 | 2788/3518 | COMPLETED |
| P41_simclr_tau0.5_seed0 | simclr_matched | null | 0 | 200/200 | 86.36 | 83.16 | null | 57.95 | 1981 | 2788/3518 | COMPLETED |
| P41_simclr_views4_800ep_seed0 | simclr_matched | null | 0 | 800/800 | 88.20 | 87.20 | null | 158.27 | 15810 | 5269/8140 | COMPLETED |
| P41_simclr_views4_800ep_seed1 | simclr_matched | null | 1 | 800/800 | 88.10 | 87.76 | null | 154.70 | 15783 | 5269/8140 | COMPLETED |
| P41_simclr_views4_800ep_seed2 | simclr_matched | null | 2 | 800/800 | 88.66 | 87.98 | null | 161.32 | 15813 | 5269/8140 | COMPLETED |
| P41_simclr_views4_b128_100ep_seed0 | simclr_matched | null | 0 | 100/100 | 86.42 | 84.50 | null | 87.14 | 4551 | 2790/3518 | COMPLETED |
| P41_simclr_views4_b128_100ep_seed1 | simclr_matched | null | 1 | 100/100 | 87.02 | 84.38 | null | 84.48 | 2010 | 2790/3518 | COMPLETED |
| P41_simclr_views4_b128_100ep_seed2 | simclr_matched | null | 2 | 100/100 | 86.48 | 84.46 | null | 86.30 | 4550 | 2790/3518 | COMPLETED |
| P41_simclr_views4_seed0 | simclr_matched | null | 0 | 200/200 | 87.48 | 86.12 | null | 116.15 | 3961 | 5269/8140 | COMPLETED |
| P41_simclr_views4_seed1 | simclr_matched | null | 1 | 200/200 | 88.30 | 86.84 | null | 114.06 | 6869 | 7066/10148 | COMPLETED |
| P41_simclr_views4_seed2 | simclr_matched | null | 2 | 200/200 | 87.74 | 86.60 | null | 113.26 | 6812 | 7066/10148 | COMPLETED |
| P41_vicreg_800ep_seed0 | vicreg_matched_128 | null | 0 | 800/800 | 86.70 | 84.36 | null | 110.49 | 7997 | 2788/3516 | COMPLETED |
| P41_vicreg_800ep_seed1 | vicreg_matched_128 | null | 1 | 800/800 | 87.20 | 84.72 | null | 111.11 | 8009 | 2788/3516 | COMPLETED |
| P41_vicreg_800ep_seed2 | vicreg_matched_128 | null | 2 | 800/800 | 87.94 | 84.52 | null | 109.09 | 18138 | 2788/3516 | COMPLETED |
| P41_vicreg_b128_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.52 | 82.84 | null | 74.59 | 1935 | 1548/2224 | COMPLETED |
| P41_vicreg_cov0.1_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 83.36 | 78.08 | null | 24.55 | 1997 | 2788/3516 | COMPLETED |
| P41_vicreg_views4_800ep_seed0 | vicreg_matched_128 | null | 0 | 800/800 | 87.50 | 84.38 | null | 105.20 | 16442 | 5269/8138 | COMPLETED |
| P41_vicreg_views4_800ep_seed1 | vicreg_matched_128 | null | 1 | 800/800 | 86.54 | 84.46 | null | 110.01 | 26210 | 5268/8026 | COMPLETED |
| P41_vicreg_views4_800ep_seed2 | vicreg_matched_128 | null | 2 | 800/800 | 87.30 | 84.54 | null | 103.41 | 26241 | 5268/8026 | COMPLETED |
| P41_vicreg_views4_b128_100ep_seed0 | vicreg_matched_128 | null | 0 | 100/100 | 87.24 | 83.72 | null | 74.28 | 3363 | 3645/4330 | COMPLETED |
| P41_vicreg_views4_b128_100ep_seed1 | vicreg_matched_128 | null | 1 | 100/100 | 87.06 | 83.96 | null | 74.73 | 2157 | 2790/3518 | COMPLETED |
| P41_vicreg_views4_b128_100ep_seed2 | vicreg_matched_128 | null | 2 | 100/100 | 87.06 | 84.06 | null | 73.95 | 2151 | 2790/3518 | COMPLETED |
| P41_vicreg_views4_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.64 | 84.14 | null | 98.43 | 8873 | 5269/8138 | COMPLETED |
| P41_vicreg_views4_seed1 | vicreg_matched_128 | null | 1 | 200/200 | 87.06 | 84.00 | null | 100.12 | 4106 | 5269/8138 | COMPLETED |
| P41_vicreg_views4_seed2 | vicreg_matched_128 | null | 2 | 200/200 | 86.48 | 84.66 | null | 97.90 | 4095 | 5269/8138 | COMPLETED |
| P41_vicreg_w10_10_1_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.28 | 83.54 | null | 107.15 | 2004 | 2788/3516 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_b128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 200 |
| P41_simclr_tau0.1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_simclr_tau0.5_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_simclr_views4_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_views4_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_views4_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_views4_b128_100ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_simclr_views4_b128_100ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_simclr_views4_b128_100ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_simclr_views4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_simclr_views4_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_simclr_views4_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_b128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 200 |
| P41_vicreg_cov0.1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_views4_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_views4_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_views4_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_views4_b128_100ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_vicreg_views4_b128_100ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_vicreg_views4_b128_100ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_vicreg_views4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_views4_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_views4_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_w10_10_1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | 41.98 | 88.38 | 46.40 | 36.58 | 87.12 | 50.54 | 2.94 | 149.07 | null |
| P41_simclr_800ep_seed1 | 41.68 | 88.28 | 46.60 | 37.40 | 86.94 | 49.54 | 3.22 | 146.71 | null |
| P41_simclr_800ep_seed2 | 42.98 | 88.38 | 45.40 | 37.08 | 87.72 | 50.64 | 3.22 | 149.29 | null |
| P41_simclr_b128_seed0 | 41.98 | 86.36 | 44.38 | 36.58 | 83.96 | 47.38 | 2.94 | 92.13 | null |
| P41_simclr_tau0.1_seed0 | 41.94 | 84.30 | 42.36 | 36.54 | 80.16 | 43.62 | 2.94 | 79.46 | null |
| P41_simclr_tau0.5_seed0 | 41.94 | 86.36 | 44.42 | 36.54 | 83.16 | 46.62 | 2.94 | 57.95 | null |
| P41_simclr_views4_800ep_seed0 | 41.94 | 88.20 | 46.26 | 36.54 | 87.20 | 50.66 | 2.94 | 158.27 | null |
| P41_simclr_views4_800ep_seed1 | 41.68 | 88.10 | 46.42 | 37.40 | 87.76 | 50.36 | 3.22 | 154.70 | null |
| P41_simclr_views4_800ep_seed2 | 42.98 | 88.66 | 45.68 | 37.08 | 87.98 | 50.90 | 3.22 | 161.32 | null |
| P41_simclr_views4_b128_100ep_seed0 | 41.94 | 86.42 | 44.48 | 36.54 | 84.50 | 47.96 | 2.94 | 87.14 | null |
| P41_simclr_views4_b128_100ep_seed1 | 41.68 | 87.02 | 45.34 | 37.40 | 84.38 | 46.98 | 3.22 | 84.48 | null |
| P41_simclr_views4_b128_100ep_seed2 | 42.98 | 86.48 | 43.50 | 37.08 | 84.46 | 47.38 | 3.22 | 86.30 | null |
| P41_simclr_views4_seed0 | 41.94 | 87.48 | 45.54 | 36.54 | 86.12 | 49.58 | 2.94 | 116.15 | null |
| P41_simclr_views4_seed1 | 41.72 | 88.30 | 46.58 | 37.42 | 86.84 | 49.42 | 3.22 | 114.06 | null |
| P41_simclr_views4_seed2 | 42.98 | 87.74 | 44.76 | 37.06 | 86.60 | 49.54 | 3.22 | 113.26 | null |
| P41_vicreg_800ep_seed0 | 41.94 | 86.70 | 44.76 | 36.54 | 84.36 | 47.82 | 2.94 | 110.49 | null |
| P41_vicreg_800ep_seed1 | 41.68 | 87.20 | 45.52 | 37.40 | 84.72 | 47.32 | 3.22 | 111.11 | null |
| P41_vicreg_800ep_seed2 | 42.98 | 87.94 | 44.96 | 37.08 | 84.52 | 47.44 | 3.22 | 109.09 | null |
| P41_vicreg_b128_seed0 | 41.94 | 86.52 | 44.58 | 36.54 | 82.84 | 46.30 | 2.94 | 74.59 | null |
| P41_vicreg_cov0.1_seed0 | 41.94 | 83.36 | 41.42 | 36.54 | 78.08 | 41.54 | 2.94 | 24.55 | null |
| P41_vicreg_views4_800ep_seed0 | 41.94 | 87.50 | 45.56 | 36.54 | 84.38 | 47.84 | 2.94 | 105.20 | null |
| P41_vicreg_views4_800ep_seed1 | 41.68 | 86.54 | 44.86 | 37.40 | 84.46 | 47.06 | 3.22 | 110.01 | null |
| P41_vicreg_views4_800ep_seed2 | 42.98 | 87.30 | 44.32 | 37.08 | 84.54 | 47.46 | 3.22 | 103.41 | null |
| P41_vicreg_views4_b128_100ep_seed0 | 41.98 | 87.24 | 45.26 | 36.58 | 83.72 | 47.14 | 2.94 | 74.28 | null |
| P41_vicreg_views4_b128_100ep_seed1 | 41.68 | 87.06 | 45.38 | 37.40 | 83.96 | 46.56 | 3.22 | 74.73 | null |
| P41_vicreg_views4_b128_100ep_seed2 | 42.98 | 87.06 | 44.08 | 37.08 | 84.06 | 46.98 | 3.22 | 73.95 | null |
| P41_vicreg_views4_seed0 | 41.94 | 86.64 | 44.70 | 36.54 | 84.14 | 47.60 | 2.94 | 98.43 | null |
| P41_vicreg_views4_seed1 | 41.68 | 87.06 | 45.38 | 37.40 | 84.00 | 46.60 | 3.22 | 100.12 | null |
| P41_vicreg_views4_seed2 | 42.98 | 86.48 | 43.50 | 37.08 | 84.66 | 47.58 | 3.22 | 97.90 | null |
| P41_vicreg_w10_10_1_seed0 | 41.94 | 86.28 | 44.34 | 36.54 | 83.54 | 47.00 | 2.94 | 107.15 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P41_simclr_800ep_seed0: ep0: 36.58, ep20: 68.24, ep50: 77.18, ep100: 80.82, ep200: 84.90, ep400: 86.38, ep600: 86.98, ep800: 87.12
- P41_simclr_800ep_seed1: ep0: 37.40, ep20: 68.98, ep50: 76.26, ep100: 81.42, ep200: 84.72, ep400: 86.40, ep600: 87.16, ep800: 86.94
- P41_simclr_800ep_seed2: ep0: 37.08, ep20: 69.10, ep50: 76.34, ep100: 81.70, ep200: 84.16, ep400: 86.68, ep600: 87.24, ep800: 87.72
- P41_simclr_b128_seed0: ep0: 36.58, ep10: 63.44, ep20: 69.94, ep50: 77.76, ep100: 81.88, ep150: 83.52, ep200: 83.96
- P41_simclr_tau0.1_seed0: ep0: 36.54, ep10: 62.12, ep20: 67.48, ep50: 73.04, ep100: 77.62, ep150: 79.70, ep200: 80.16
- P41_simclr_tau0.5_seed0: ep0: 36.54, ep10: 58.88, ep20: 67.70, ep50: 75.50, ep100: 81.12, ep150: 82.96, ep200: 83.16
- P41_simclr_views4_800ep_seed0: ep0: 36.54, ep20: 73.80, ep50: 81.84, ep100: 85.10, ep200: 86.68, ep400: 87.58, ep600: 87.38, ep800: 87.20
- P41_simclr_views4_800ep_seed1: ep0: 37.40, ep20: 73.98, ep50: 80.56, ep100: 84.96, ep200: 86.72, ep400: 87.18, ep600: 87.74, ep800: 87.76
- P41_simclr_views4_800ep_seed2: ep0: 37.08, ep20: 73.30, ep50: 81.12, ep100: 84.56, ep200: 87.28, ep400: 87.68, ep600: 87.54, ep800: 87.98
- P41_simclr_views4_b128_100ep_seed0: ep0: 36.54, ep10: 69.00, ep20: 74.70, ep50: 82.00, ep100: 84.50
- P41_simclr_views4_b128_100ep_seed1: ep0: 37.40, ep10: 68.86, ep20: 74.66, ep50: 82.04, ep100: 84.38
- P41_simclr_views4_b128_100ep_seed2: ep0: 37.08, ep10: 68.50, ep20: 74.78, ep50: 82.06, ep100: 84.46
- P41_simclr_views4_seed0: ep0: 36.54, ep10: 66.12, ep20: 74.00, ep50: 81.02, ep100: 84.72, ep150: 86.20, ep200: 86.12
- P41_simclr_views4_seed1: ep0: 37.42, ep10: 67.00, ep20: 73.70, ep50: 80.92, ep100: 84.90, ep150: 85.84, ep200: 86.84
- P41_simclr_views4_seed2: ep0: 37.06, ep10: 67.20, ep20: 73.68, ep50: 81.44, ep100: 85.44, ep150: 86.08, ep200: 86.60
- P41_vicreg_800ep_seed0: ep0: 36.54, ep20: 67.64, ep50: 76.16, ep100: 79.24, ep200: 82.32, ep400: 84.14, ep600: 84.32, ep800: 84.36
- P41_vicreg_800ep_seed1: ep0: 37.40, ep20: 67.20, ep50: 75.94, ep100: 79.66, ep200: 83.20, ep400: 84.68, ep600: 84.74, ep800: 84.72
- P41_vicreg_800ep_seed2: ep0: 37.08, ep20: 68.72, ep50: 76.26, ep100: 79.64, ep200: 83.10, ep400: 84.14, ep600: 84.66, ep800: 84.52
- P41_vicreg_b128_seed0: ep0: 36.54, ep10: 61.74, ep20: 69.88, ep50: 77.06, ep100: 81.06, ep150: 82.84, ep200: 82.84
- P41_vicreg_cov0.1_seed0: ep0: 36.54, ep10: 51.78, ep20: 60.48, ep50: 70.68, ep100: 75.80, ep150: 77.68, ep200: 78.08
- P41_vicreg_views4_800ep_seed0: ep0: 36.54, ep20: 74.84, ep50: 79.76, ep100: 83.46, ep200: 84.52, ep400: 84.68, ep600: 84.42, ep800: 84.38
- P41_vicreg_views4_800ep_seed1: ep0: 37.40, ep20: 73.78, ep50: 79.66, ep100: 82.66, ep200: 84.28, ep400: 84.74, ep600: 84.38, ep800: 84.46
- P41_vicreg_views4_800ep_seed2: ep0: 37.08, ep20: 73.74, ep50: 80.14, ep100: 83.84, ep200: 84.74, ep400: 84.72, ep600: 84.54, ep800: 84.54
- P41_vicreg_views4_b128_100ep_seed0: ep0: 36.58, ep10: 68.66, ep20: 75.34, ep50: 81.92, ep100: 83.72
- P41_vicreg_views4_b128_100ep_seed1: ep0: 37.40, ep10: 69.18, ep20: 76.08, ep50: 81.60, ep100: 83.96
- P41_vicreg_views4_b128_100ep_seed2: ep0: 37.08, ep10: 69.48, ep20: 75.62, ep50: 81.96, ep100: 84.06
- P41_vicreg_views4_seed0: ep0: 36.54, ep10: 64.70, ep20: 74.30, ep50: 80.14, ep100: 83.18, ep150: 83.90, ep200: 84.14
- P41_vicreg_views4_seed1: ep0: 37.40, ep10: 65.78, ep20: 74.38, ep50: 80.12, ep100: 83.08, ep150: 84.02, ep200: 84.00
- P41_vicreg_views4_seed2: ep0: 37.08, ep10: 67.54, ep20: 73.64, ep50: 80.56, ep100: 83.72, ep150: 84.44, ep200: 84.66
- P41_vicreg_w10_10_1_seed0: ep0: 36.54, ep10: 61.82, ep20: 68.76, ep50: 77.60, ep100: 81.54, ep150: 83.38, ep200: 83.54

## Held-out J trajectory (VCS only)


## Final-epoch training objective values (epoch means)

- P41_simclr_800ep_seed0: J_raw null, R_binary null, nt_xent 1.9396, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_800ep_seed1: J_raw null, R_binary null, nt_xent 1.9384, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_800ep_seed2: J_raw null, R_binary null, nt_xent 1.9386, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_b128_seed0: J_raw null, R_binary null, nt_xent 1.5510, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_tau0.1_seed0: J_raw null, R_binary null, nt_xent 0.3712, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_tau0.5_seed0: J_raw null, R_binary null, nt_xent 4.4735, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_800ep_seed0: J_raw null, R_binary null, nt_xent 1.8630, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_800ep_seed1: J_raw null, R_binary null, nt_xent 1.8652, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_800ep_seed2: J_raw null, R_binary null, nt_xent 1.8662, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_b128_100ep_seed0: J_raw null, R_binary null, nt_xent 1.5437, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_b128_100ep_seed1: J_raw null, R_binary null, nt_xent 1.5456, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_b128_100ep_seed2: J_raw null, R_binary null, nt_xent 1.5447, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_seed0: J_raw null, R_binary null, nt_xent 2.0265, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_seed1: J_raw null, R_binary null, nt_xent 2.0288, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_seed2: J_raw null, R_binary null, nt_xent 2.0257, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_vicreg_800ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.09391757002898625, 'vicreg_variance': 0.008313221570902637, 'vicreg_covariance': 1.3883616542816162}
- P41_vicreg_800ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.09376617708376475, 'vicreg_variance': 0.008220109379451189, 'vicreg_covariance': 1.4046891328266689}
- P41_vicreg_800ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.09512636308159148, 'vicreg_variance': 0.008887588315244232, 'vicreg_covariance': 1.4035253374917167}
- P41_vicreg_b128_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.15739956285878803, 'vicreg_variance': 0.03675391739420062, 'vicreg_covariance': 2.713556081820757}
- P41_vicreg_cov0.1_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.08672738528677396, 'vicreg_variance': 0.0053946533865694484, 'vicreg_covariance': 16.27707865033831}
- P41_vicreg_views4_800ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.06431664843644415, 'vicreg_variance': 0.005204583865789963, 'vicreg_covariance': 1.2178863293784006}
- P41_vicreg_views4_800ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.06417840038027083, 'vicreg_variance': 0.005266339316565011, 'vicreg_covariance': 1.2313447945458549}
- P41_vicreg_views4_800ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.06374934058104242, 'vicreg_variance': 0.005413842369203589, 'vicreg_covariance': 1.233451282296862}
- P41_vicreg_views4_b128_100ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.14858319550922452, 'vicreg_variance': 0.035261114653295435, 'vicreg_covariance': 2.7362947593047746}
- P41_vicreg_views4_b128_100ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.15035699063811206, 'vicreg_variance': 0.03481885710428668, 'vicreg_covariance': 2.721456541295065}
- P41_vicreg_views4_b128_100ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.14987704183301356, 'vicreg_variance': 0.034857185942219024, 'vicreg_covariance': 2.7400623719576758}
- P41_vicreg_views4_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11799566720213209, 'vicreg_variance': 0.012244413729224886, 'vicreg_covariance': 1.6777392305646623}
- P41_vicreg_views4_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11904662758111954, 'vicreg_variance': 0.012262377412989736, 'vicreg_covariance': 1.6708863639831544}
- P41_vicreg_views4_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11866631541933333, 'vicreg_variance': 0.01197244039470596, 'vicreg_covariance': 1.6643838814326695}
- P41_vicreg_w10_10_1_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.18148469941956658, 'vicreg_variance': 0.024802822140710695, 'vicreg_covariance': 0.84139078106199}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | 0.0899 | 2849 | 5698 | 12774 | 35 | 13 | 35840000 | None | 1011789@nodesumo01 |
| P41_simclr_800ep_seed1 | 0.0557 | 4600 | 9199 | 7908 | 18 | 7 | 35840000 | None | 1012226@node58 |
| P41_simclr_800ep_seed2 | 0.0558 | 4591 | 9181 | 7922 | 18 | 7 | 35840000 | None | 1012227@node58 |
| P41_simclr_b128_seed0 | 0.0474 | 2699 | 5397 | 3363 | 30 | 13 | 8985600 | None | 1011788@nodesumo01 |
| P41_simclr_tau0.1_seed0 | 0.0558 | 4590 | 9179 | 1989 | 16 | 7 | 8960000 | None | 1011784@node58 |
| P41_simclr_tau0.5_seed0 | 0.0556 | 4600 | 9200 | 1981 | 16 | 7 | 8960000 | None | 1011785@node58 |
| P41_simclr_views4_800ep_seed0 | 0.1115 | 2297 | 4593 | 15810 | 18 | 7 | 35840000 | None | 1012268@node58 |
| P41_simclr_views4_800ep_seed1 | 0.1113 | 2300 | 4600 | 15783 | 18 | 7 | 35840000 | None | 1012269@node58 |
| P41_simclr_views4_800ep_seed2 | 0.1115 | 2296 | 4593 | 15813 | 18 | 7 | 35840000 | None | 1012270@node59 |
| P41_simclr_views4_b128_100ep_seed0 | 0.1291 | 991 | 1983 | 4551 | 22 | 10 | 4492800 | None | 1011787@node60 |
| P41_simclr_views4_b128_100ep_seed1 | 0.0567 | 2257 | 4514 | 2010 | 11 | 7 | 4492800 | None | 1012222@node59 |
| P41_simclr_views4_b128_100ep_seed2 | 0.1292 | 991 | 1982 | 4550 | 22 | 9 | 4492800 | None | 1012223@node60 |
| P41_simclr_views4_seed0 | 0.1116 | 2294 | 4589 | 3961 | 16 | 7 | 8960000 | None | 1011786@node59 |
| P41_simclr_views4_seed1 | 0.1936 | 1322 | 2644 | 6869 | 30 | 13 | 8960000 | None | 1012224@nodesumo01 |
| P41_simclr_views4_seed2 | 0.1920 | 1333 | 2667 | 6812 | 31 | 13 | 8960000 | None | 1012225@nodesumo01 |
| P41_vicreg_800ep_seed0 | 0.0563 | 4550 | 9099 | 7997 | 18 | 13 | 35840000 | None | 1011887@node58 |
| P41_vicreg_800ep_seed1 | 0.0564 | 4541 | 9082 | 8009 | 18 | 7 | 35840000 | None | 1012232@node59 |
| P41_vicreg_800ep_seed2 | 0.1288 | 1988 | 3976 | 18138 | 36 | 15 | 35840000 | None | 1012233@node61 |
| P41_vicreg_b128_seed0 | 0.0273 | 4695 | 9390 | 1935 | 16 | 7 | 8985600 | None | 1011796@node59 |
| P41_vicreg_cov0.1_seed0 | 0.0562 | 4554 | 9109 | 1997 | 15 | 7 | 8960000 | None | 1011790@node58 |
| P41_vicreg_views4_800ep_seed0 | 0.1160 | 2208 | 4415 | 16442 | 18 | 7 | 35840000 | None | 1012271@node59 |
| P41_vicreg_views4_800ep_seed1 | 0.1158 | 2210 | 4420 | 26210 | 31 | 10 | 35840000 | None | 1012737@node59 |
| P41_vicreg_views4_800ep_seed2 | 0.1161 | 2206 | 4411 | 26241 | 31 | 7 | 35840000 | None | 1012738@node59 |
| P41_vicreg_views4_b128_100ep_seed0 | 0.0950 | 1347 | 2694 | 3363 | 22 | 13 | 4492800 | None | 1011795@nodesumo01 |
| P41_vicreg_views4_b128_100ep_seed1 | 0.0610 | 2099 | 4197 | 2157 | 11 | 7 | 4492800 | None | 1012228@node59 |
| P41_vicreg_views4_b128_100ep_seed2 | 0.0608 | 2104 | 4208 | 2151 | 11 | 7 | 4492800 | None | 1012229@node59 |
| P41_vicreg_views4_seed0 | 0.2521 | 1015 | 2031 | 8873 | 31 | 9 | 8960000 | None | 1011794@node60 |
| P41_vicreg_views4_seed1 | 0.1158 | 2211 | 4421 | 4106 | 16 | 7 | 8960000 | None | 1012230@node59 |
| P41_vicreg_views4_seed2 | 0.1155 | 2217 | 4435 | 4095 | 18 | 7 | 8960000 | None | 1012231@node61 |
| P41_vicreg_w10_10_1_seed0 | 0.0564 | 4538 | 9077 | 2004 | 16 | 7 | 8960000 | None | 1011791@node58 |

## Provenance

- P41_simclr_800ep_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `cc3ed318fa8ebb7b6d52aec55a0c9d6833e715b83fda66c38f3973363240a5a2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_800ep_seed1: commit `be4febcd1a056e00dedc87f7d50a8fc6cf74c937` dirty=True, config `cb1db8f032a479f179a19e981f1ed33629a7e8be09de449a4940aadbd9c14ad5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_simclr_800ep_seed2: commit `be4febcd1a056e00dedc87f7d50a8fc6cf74c937` dirty=True, config `f38dd199a9de00a052d12ada8365437eab63eebaaf01e286bfc2aee495f8bfa3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_simclr_b128_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `24b5304c1f5a405d8eaee08ab8d521e56bb854b17b07708d71020193a26a25ad`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_tau0.1_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `122e0a3d8fe4ed56d011772bbc96b09d4dcef9c30c3c50eddff86f88f5953c60`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_tau0.5_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `f7b57c3de650df57e53d3b147c76b82950f94425493a6696e68ed9dcb3483b29`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_views4_800ep_seed0: commit `1b85af029ec6fd60548b8e1d150d47bd4235836c` dirty=True, config `af1eb8ed0835259998d234a61095873dd76ca117408c7da39661f748b8a0603c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_views4_800ep_seed1: commit `1b85af029ec6fd60548b8e1d150d47bd4235836c` dirty=True, config `b6bae7edf12dff7d21dc376ed7b1ab65d3c2aefceb601d436df8a3d4792a2fd2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_simclr_views4_800ep_seed2: commit `1b85af029ec6fd60548b8e1d150d47bd4235836c` dirty=True, config `8b3e394108dbee85bc6a2401e46ff00872c70928ac596d25b9c3dfc66311af9d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_simclr_views4_b128_100ep_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `21fc8b01df0394b73af9497c4259beede2c6517ae2966417b11aa83d3ba1fc60`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_views4_b128_100ep_seed1: commit `be4febcd1a056e00dedc87f7d50a8fc6cf74c937` dirty=True, config `06ed26cb234f8d50d24f592652d4a640cf0bc179eba7d2393262c88bee7e09e8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_simclr_views4_b128_100ep_seed2: commit `be4febcd1a056e00dedc87f7d50a8fc6cf74c937` dirty=True, config `0c1b845f0c6eef35e3a8049778e4d733631d568e61eb58c7eeabb54f4fd02f4a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_simclr_views4_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `a4fc31ffbc5c982ca2fd650a40d141f8f30cf115d466b2c7ac0000be89c72187`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_views4_seed1: commit `be4febcd1a056e00dedc87f7d50a8fc6cf74c937` dirty=True, config `65f40a62872a87c992fd76b7b438c30b984be0901fab0237adfb0aa7142c63b9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_simclr_views4_seed2: commit `be4febcd1a056e00dedc87f7d50a8fc6cf74c937` dirty=True, config `d9aa124bbdfe6be1c4d96433324f21fd63cc43220a878c2aeb06570cab7b07d7`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_vicreg_800ep_seed0: commit `cff7290659d49a09291fd018dea26373bdcbd78f` dirty=True, config `6a574d430404e3d83fe6146a2514ce7beff9aff0a979a7f8f9a40aa18fe38488`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_800ep_seed1: commit `0cb9f6ef94db4240e08ee4b90da192afeffe2dd3` dirty=True, config `2fe1edc9d33afab5e693339d11eaca8d33efbcf4a1b6978437fe5c97989f7d67`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_vicreg_800ep_seed2: commit `0cb9f6ef94db4240e08ee4b90da192afeffe2dd3` dirty=True, config `b0d04b28e8089cb382b61bbdde9c5ca30085fe540b3c16c400ee974e530c7836`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_vicreg_b128_seed0: commit `92b5ba66f99982f6f4c6fcca7bc70cca41d0f9fd` dirty=True, config `74f32458f4ccbd75bf27c2ee771175dcac834e49ff40e58d96235a99fb1b2a64`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_cov0.1_seed0: commit `19703e87fe363f86092bf9aac0d188ee0d87143a` dirty=True, config `d9981a902d96453c93bf9080c143577b821058d7b8b119f602fa039fc810e3da`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_views4_800ep_seed0: commit `1b85af029ec6fd60548b8e1d150d47bd4235836c` dirty=True, config `12261d2e6ee5efc569661f7d1f144ce712937981a6a3363961cc0a3220f4b442`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_views4_800ep_seed1: commit `7da410d80f486180e9b817c22ba09dafad48ca70` dirty=False, config `7ab2dbb55d3f41c70841a3da258bd402bf3716d0c143a9822c23304252181827`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_vicreg_views4_800ep_seed2: commit `7da410d80f486180e9b817c22ba09dafad48ca70` dirty=False, config `04ad79e04b231f5a2a1a19778a1d9c4cdbbfcb4851918359a712d66109502b17`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_vicreg_views4_b128_100ep_seed0: commit `92b5ba66f99982f6f4c6fcca7bc70cca41d0f9fd` dirty=True, config `c9585528aa122ce2f2f88edc5f5a1e0b7b65336d15bf1f0eb592a9d220807ce3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_views4_b128_100ep_seed1: commit `0cb9f6ef94db4240e08ee4b90da192afeffe2dd3` dirty=True, config `82d2ef69800746d4ffb25f9cab2b9e5045e47ee6778cb1b6414e5c7ea2764090`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_vicreg_views4_b128_100ep_seed2: commit `0cb9f6ef94db4240e08ee4b90da192afeffe2dd3` dirty=True, config `55d50c9cfb6238e765228b97b0a82ad221e776540b8062000a05301a1046396f`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_vicreg_views4_seed0: commit `19703e87fe363f86092bf9aac0d188ee0d87143a` dirty=True, config `562d650a2e58b7a22d232a60623d3b6ca9702935624cc5a95a33b2dfb612a321`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_views4_seed1: commit `0cb9f6ef94db4240e08ee4b90da192afeffe2dd3` dirty=True, config `6be2981d7b3c3b93216d513888ee46cd31d36d6c13f85e656efc1b97ac1cb119`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P41_vicreg_views4_seed2: commit `0cb9f6ef94db4240e08ee4b90da192afeffe2dd3` dirty=True, config `357d03356f556afd2f55bf66146bf53005d0886bdb1f5d7b0a26d61413465e6a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P41_vicreg_w10_10_1_seed0: commit `19703e87fe363f86092bf9aac0d188ee0d87143a` dirty=True, config `406f5e9f6cdfdc1497a80e72270f12900a070c611c9fd114781ecba5f73f50f3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 15 | 87.36 ± 1.20 | 42.16 ± 0.53 | 45.21 ± 1.23 | 85.66 ± 2.18 | 116.69 ± 34.30 | null |
| vicreg_matched_128 | 15 | 86.73 ± 1.03 | 42.15 ± 0.53 | 44.57 ± 1.05 | 83.73 ± 1.64 | 91.67 ± 23.59 | null |

## Coverage checks

- P41_simclr_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_b128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_tau0.1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_tau0.5_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_views4_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_simclr_views4_b128_100ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_simclr_views4_b128_100ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_simclr_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_views4_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_views4_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_b128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_cov0.1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_views4_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_vicreg_views4_b128_100ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_vicreg_views4_b128_100ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_vicreg_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_views4_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_views4_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_w10_10_1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
