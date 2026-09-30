# P91_cifar100 — neutral results table (observed values only)

Generated 2026-09-30T10:33:00Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | simclr_matched | null | 0 | 200/200 | 59.74 | 54.70 | null | 141.23 | 9583 | 5268/8028 | COMPLETED |
| P91_c100_simclr_views4_200ep_seed1 | simclr_matched | null | 1 | 200/200 | 59.30 | 54.02 | null | 141.31 | 4974 | 5268/8028 | COMPLETED |
| P91_c100_simclr_views4_200ep_seed2 | simclr_matched | null | 2 | 200/200 | 59.22 | 54.42 | null | 140.14 | 4966 | 5268/8028 | COMPLETED |
| P91_c100_simclr_views4_800ep_seed0 | simclr_matched | null | 0 | 800/800 | 58.66 | 57.38 | null | 193.33 | 15768 | 5269/8140 | COMPLETED |
| P91_c100_simclr_views4_800ep_seed1 | simclr_matched | null | 1 | 800/800 | 58.02 | 56.78 | null | 196.62 | 15822 | 5269/8140 | COMPLETED |
| P91_c100_simclr_views4_800ep_seed2 | simclr_matched | null | 2 | 800/800 | 58.08 | 57.46 | null | 187.59 | 15800 | 5269/8140 | COMPLETED |
| P91_c100_vcs_a5_views4_200ep_seed0 | vcs_qmi | 8 | 0 | 200/200 | 57.74 | 48.44 | 0.9614±0.0012 | 79.93 | 13445 | 7568/12672 | COMPLETED |
| P91_c100_vcs_a5_views4_200ep_seed1 | vcs_qmi | 8 | 1 | 200/200 | 58.16 | 48.52 | 0.9617±0.0006 | 80.33 | 11555 | 5268/8028 | COMPLETED |
| P91_c100_vcs_a5_views4_200ep_seed2 | vcs_qmi | 8 | 2 | 200/200 | 57.98 | 49.06 | 0.9614±0.0010 | 79.58 | 10678 | 5268/8028 | COMPLETED |
| P91_c100_vcs_a5_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 60.00 | 55.42 | 0.9698±0.0006 | 153.04 | 16145 | 5269/8140 | COMPLETED |
| P91_c100_vcs_a5_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 59.68 | 54.56 | 0.9693±0.0012 | 151.55 | 16108 | 5269/8140 | COMPLETED |
| P91_c100_vcs_a5_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 60.04 | 54.86 | 0.9689±0.0012 | 150.02 | 16125 | 5269/8140 | COMPLETED |
| P91_c100_vicreg_views4_200ep_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 56.80 | 48.72 | null | 112.16 | 8880 | 5269/8138 | COMPLETED |
| P91_c100_vicreg_views4_200ep_seed1 | vicreg_matched_128 | null | 1 | 200/200 | 56.50 | 49.02 | null | 113.17 | 6970 | 7066/10146 | COMPLETED |
| P91_c100_vicreg_views4_200ep_seed2 | vicreg_matched_128 | null | 2 | 200/200 | 55.98 | 47.94 | null | 111.26 | 6824 | 7066/10146 | COMPLETED |
| P91_c100_vicreg_views4_800ep_seed0 | vicreg_matched_128 | null | 0 | 800/800 | 56.28 | 47.98 | null | 115.22 | 16415 | 5269/8138 | COMPLETED |
| P91_c100_vicreg_views4_800ep_seed1 | vicreg_matched_128 | null | 1 | 800/800 | 55.78 | 48.72 | null | 115.14 | 16377 | 5269/8138 | COMPLETED |
| P91_c100_vicreg_views4_800ep_seed2 | vicreg_matched_128 | null | 2 | 800/800 | 55.98 | 48.44 | null | 114.10 | 35497 | 5269/8138 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_simclr_views4_200ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_simclr_views4_200ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_simclr_views4_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P91_c100_simclr_views4_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P91_c100_simclr_views4_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P91_c100_vcs_a5_views4_200ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P91_c100_vcs_a5_views4_200ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P91_c100_vcs_a5_views4_200ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P91_c100_vcs_a5_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P91_c100_vcs_a5_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P91_c100_vcs_a5_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P91_c100_vicreg_views4_200ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_vicreg_views4_200ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_vicreg_views4_200ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_vicreg_views4_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P91_c100_vicreg_views4_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P91_c100_vicreg_views4_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | 20.12 | 59.74 | 39.62 | 14.00 | 54.70 | 40.70 | 2.74 | 141.23 | null |
| P91_c100_simclr_views4_200ep_seed1 | 19.34 | 59.30 | 39.96 | 14.72 | 54.02 | 39.30 | 2.89 | 141.31 | null |
| P91_c100_simclr_views4_200ep_seed2 | 20.04 | 59.22 | 39.18 | 13.78 | 54.42 | 40.64 | 2.92 | 140.14 | null |
| P91_c100_simclr_views4_800ep_seed0 | 20.12 | 58.66 | 38.54 | 14.00 | 57.38 | 43.38 | 2.74 | 193.33 | null |
| P91_c100_simclr_views4_800ep_seed1 | 19.34 | 58.02 | 38.68 | 14.72 | 56.78 | 42.06 | 2.89 | 196.62 | null |
| P91_c100_simclr_views4_800ep_seed2 | 20.04 | 58.08 | 38.04 | 13.78 | 57.46 | 43.68 | 2.92 | 187.59 | null |
| P91_c100_vcs_a5_views4_200ep_seed0 | 20.12 | 57.74 | 37.62 | 14.02 | 48.44 | 34.42 | 2.74 | 79.93 | -0.9998 |
| P91_c100_vcs_a5_views4_200ep_seed1 | 19.34 | 58.16 | 38.82 | 14.72 | 48.52 | 33.80 | 2.89 | 80.33 | -0.9998 |
| P91_c100_vcs_a5_views4_200ep_seed2 | 20.04 | 57.98 | 37.94 | 13.78 | 49.06 | 35.28 | 2.92 | 79.58 | -0.9998 |
| P91_c100_vcs_a5_views4_800ep_seed0 | 20.12 | 60.00 | 39.88 | 14.00 | 55.42 | 41.42 | 2.74 | 153.04 | -0.9998 |
| P91_c100_vcs_a5_views4_800ep_seed1 | 19.34 | 59.68 | 40.34 | 14.72 | 54.56 | 39.84 | 2.89 | 151.55 | -0.9998 |
| P91_c100_vcs_a5_views4_800ep_seed2 | 20.04 | 60.04 | 40.00 | 13.78 | 54.86 | 41.08 | 2.92 | 150.02 | -0.9998 |
| P91_c100_vicreg_views4_200ep_seed0 | 20.12 | 56.80 | 36.68 | 14.00 | 48.72 | 34.72 | 2.74 | 112.16 | null |
| P91_c100_vicreg_views4_200ep_seed1 | 19.34 | 56.50 | 37.16 | 14.72 | 49.02 | 34.30 | 2.89 | 113.17 | null |
| P91_c100_vicreg_views4_200ep_seed2 | 20.08 | 55.98 | 35.90 | 13.82 | 47.94 | 34.12 | 2.92 | 111.26 | null |
| P91_c100_vicreg_views4_800ep_seed0 | 20.12 | 56.28 | 36.16 | 14.00 | 47.98 | 33.98 | 2.74 | 115.22 | null |
| P91_c100_vicreg_views4_800ep_seed1 | 19.34 | 55.78 | 36.44 | 14.72 | 48.72 | 34.00 | 2.89 | 115.14 | null |
| P91_c100_vicreg_views4_800ep_seed2 | 20.04 | 55.98 | 35.94 | 13.78 | 48.44 | 34.66 | 2.92 | 114.10 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P91_c100_simclr_views4_200ep_seed0: ep0: 14.02, ep10: 33.02, ep20: 38.04, ep50: 47.32, ep100: 52.46, ep150: 54.02, ep200: 54.70
- P91_c100_simclr_views4_200ep_seed1: ep0: 14.72, ep10: 32.74, ep20: 38.38, ep50: 46.76, ep100: 52.46, ep150: 54.08, ep200: 54.02
- P91_c100_simclr_views4_200ep_seed2: ep0: 13.82, ep10: 32.68, ep20: 39.02, ep50: 46.94, ep100: 52.60, ep150: 53.94, ep200: 54.42
- P91_c100_simclr_views4_800ep_seed0: ep0: 14.00, ep20: 38.12, ep50: 47.40, ep100: 52.70, ep200: 55.40, ep400: 57.46, ep600: 57.18, ep800: 57.38
- P91_c100_simclr_views4_800ep_seed1: ep0: 14.72, ep20: 38.62, ep50: 46.74, ep100: 52.56, ep200: 55.78, ep400: 56.68, ep600: 56.80, ep800: 56.78
- P91_c100_simclr_views4_800ep_seed2: ep0: 13.78, ep20: 38.76, ep50: 47.08, ep100: 51.86, ep200: 56.10, ep400: 56.94, ep600: 57.24, ep800: 57.46
- P91_c100_vcs_a5_views4_200ep_seed0: ep0: 14.02, ep10: 26.48, ep20: 34.20, ep50: 39.90, ep100: 45.40, ep150: 47.94, ep200: 48.44
- P91_c100_vcs_a5_views4_200ep_seed1: ep0: 14.72, ep10: 26.34, ep20: 34.10, ep50: 41.48, ep100: 45.18, ep150: 47.96, ep200: 48.52
- P91_c100_vcs_a5_views4_200ep_seed2: ep0: 13.82, ep10: 26.48, ep20: 35.02, ep50: 40.70, ep100: 45.64, ep150: 48.04, ep200: 49.06
- P91_c100_vcs_a5_views4_800ep_seed0: ep0: 14.00, ep20: 34.24, ep50: 40.42, ep100: 45.86, ep200: 50.14, ep400: 53.52, ep600: 55.22, ep800: 55.42
- P91_c100_vcs_a5_views4_800ep_seed1: ep0: 14.72, ep20: 34.76, ep50: 41.48, ep100: 45.78, ep200: 50.36, ep400: 54.04, ep600: 54.22, ep800: 54.56
- P91_c100_vcs_a5_views4_800ep_seed2: ep0: 13.78, ep20: 35.08, ep50: 40.74, ep100: 45.76, ep200: 51.10, ep400: 53.18, ep600: 54.32, ep800: 54.86
- P91_c100_vicreg_views4_200ep_seed0: ep0: 14.00, ep10: 27.54, ep20: 34.34, ep50: 43.28, ep100: 47.50, ep150: 48.22, ep200: 48.72
- P91_c100_vicreg_views4_200ep_seed1: ep0: 14.72, ep10: 27.66, ep20: 33.54, ep50: 43.06, ep100: 46.96, ep150: 48.84, ep200: 49.02
- P91_c100_vicreg_views4_200ep_seed2: ep0: 13.82, ep10: 27.38, ep20: 34.80, ep50: 43.42, ep100: 46.94, ep150: 47.78, ep200: 47.94
- P91_c100_vicreg_views4_800ep_seed0: ep0: 14.00, ep20: 34.40, ep50: 43.08, ep100: 47.08, ep200: 49.76, ep400: 48.98, ep600: 48.50, ep800: 47.98
- P91_c100_vicreg_views4_800ep_seed1: ep0: 14.72, ep20: 34.16, ep50: 43.04, ep100: 47.72, ep200: 49.44, ep400: 49.40, ep600: 48.54, ep800: 48.72
- P91_c100_vicreg_views4_800ep_seed2: ep0: 13.78, ep20: 35.22, ep50: 42.40, ep100: 47.24, ep200: 49.52, ep400: 48.98, ep600: 48.42, ep800: 48.44

## Held-out J trajectory (VCS only)

- P91_c100_vcs_a5_views4_200ep_seed0: ep0: -0.9998, ep10: 0.7279, ep20: 0.8713, ep50: 0.9183, ep100: 0.9476, ep150: 0.9579, ep200: 0.9614
- P91_c100_vcs_a5_views4_200ep_seed1: ep0: -0.9998, ep10: 0.7421, ep20: 0.8622, ep50: 0.9200, ep100: 0.9449, ep150: 0.9576, ep200: 0.9617
- P91_c100_vcs_a5_views4_200ep_seed2: ep0: -0.9998, ep10: 0.7005, ep20: 0.8751, ep50: 0.9189, ep100: 0.9469, ep150: 0.9575, ep200: 0.9614
- P91_c100_vcs_a5_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8715, ep50: 0.9147, ep100: 0.9420, ep200: 0.9549, ep400: 0.9643, ep600: 0.9683, ep800: 0.9698
- P91_c100_vcs_a5_views4_800ep_seed1: ep0: -0.9998, ep20: 0.8693, ep50: 0.9226, ep100: 0.9386, ep200: 0.9542, ep400: 0.9647, ep600: 0.9679, ep800: 0.9693
- P91_c100_vcs_a5_views4_800ep_seed2: ep0: -0.9998, ep20: 0.8625, ep50: 0.9220, ep100: 0.9426, ep200: 0.9535, ep400: 0.9644, ep600: 0.9677, ep800: 0.9689

## Final-epoch training objective values (epoch means)

- P91_c100_simclr_views4_200ep_seed0: J_raw null, R_binary null, nt_xent 2.0719, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_200ep_seed1: J_raw null, R_binary null, nt_xent 2.0742, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_200ep_seed2: J_raw null, R_binary null, nt_xent 2.0731, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_800ep_seed0: J_raw null, R_binary null, nt_xent 1.8903, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_800ep_seed1: J_raw null, R_binary null, nt_xent 1.8939, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_800ep_seed2: J_raw null, R_binary null, nt_xent 1.8938, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_200ep_seed0: J_raw 0.9703, R_binary 0.0297, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_200ep_seed1: J_raw 0.9700, R_binary 0.0300, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_200ep_seed2: J_raw 0.9701, R_binary 0.0299, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_800ep_seed0: J_raw 0.9854, R_binary 0.0146, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_800ep_seed1: J_raw 0.9853, R_binary 0.0147, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_800ep_seed2: J_raw 0.9856, R_binary 0.0144, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vicreg_views4_200ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11261215107781547, 'vicreg_variance': 0.013424802695933198, 'vicreg_covariance': 1.9636033739362444}
- P91_c100_vicreg_views4_200ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11382313042879105, 'vicreg_variance': 0.013560276909598282, 'vicreg_covariance': 1.9750301817485265}
- P91_c100_vicreg_views4_200ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11541412885699953, 'vicreg_variance': 0.013514666063045817, 'vicreg_covariance': 2.0068757779257638}
- P91_c100_vicreg_views4_800ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.052985574922391344, 'vicreg_variance': 0.005546277594819133, 'vicreg_covariance': 1.5828962298801967}
- P91_c100_vicreg_views4_800ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.05244227564760617, 'vicreg_variance': 0.005905134853507792, 'vicreg_covariance': 1.579573516845703}
- P91_c100_vicreg_views4_800ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.05280358912689345, 'vicreg_variance': 0.005996583841874131, 'vicreg_covariance': 1.5755351093837193}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | 0.2471 | 1036 | 2072 | 9583 | 41 | 10 | 8960000 | None | 1013676@node61 |
| P91_c100_simclr_views4_200ep_seed1 | 0.1115 | 2296 | 4593 | 4974 | 33 | 7 | 8960000 | None | 1013677@node59 |
| P91_c100_simclr_views4_200ep_seed2 | 0.1113 | 2301 | 4601 | 4966 | 34 | 7 | 8960000 | None | 1013678@node61 |
| P91_c100_simclr_views4_800ep_seed0 | 0.1111 | 2303 | 4607 | 15768 | 18 | 7 | 35840000 | None | 1014243@node61 |
| P91_c100_simclr_views4_800ep_seed1 | 0.1116 | 2295 | 4590 | 15822 | 18 | 7 | 35840000 | None | 1014244@node59 |
| P91_c100_simclr_views4_800ep_seed2 | 0.1114 | 2298 | 4596 | 15800 | 18 | 7 | 35840000 | None | 1014378@node58 |
| P91_c100_vcs_a5_views4_200ep_seed0 | 0.3801 | 674 | 1347 | 13445 | 146 | 36 | 8960000 | 2 | 1013431@node54 |
| P91_c100_vcs_a5_views4_200ep_seed1 | 0.2490 | 1028 | 2056 | 11555 | 123 | 17 | 8960000 | 2 | 1013675@node61 |
| P91_c100_vcs_a5_views4_200ep_seed2 | 0.2495 | 1026 | 2053 | 10678 | 111 | 17 | 8960000 | 2 | 1013674@node61 |
| P91_c100_vcs_a5_views4_800ep_seed0 | 0.1139 | 2249 | 4497 | 16145 | 50 | 11 | 35840000 | 2 | 1014239@node59 |
| P91_c100_vcs_a5_views4_800ep_seed1 | 0.1136 | 2254 | 4508 | 16108 | 52 | 11 | 35840000 | 2 | 1014240@node58 |
| P91_c100_vcs_a5_views4_800ep_seed2 | 0.1137 | 2251 | 4503 | 16125 | 51 | 12 | 35840000 | 2 | 1014241@node59 |
| P91_c100_vicreg_views4_200ep_seed0 | 0.2521 | 1015 | 2031 | 8880 | 33 | 16 | 8960000 | None | 1013608@node61 |
| P91_c100_vicreg_views4_200ep_seed1 | 0.1966 | 1302 | 2605 | 6970 | 30 | 13 | 8960000 | None | 1013613@node53 |
| P91_c100_vicreg_views4_200ep_seed2 | 0.1925 | 1330 | 2660 | 6824 | 29 | 13 | 8960000 | None | 1013623@node53 |
| P91_c100_vicreg_views4_800ep_seed0 | 0.1158 | 2210 | 4421 | 16415 | 18 | 7 | 35840000 | None | 1014387@node59 |
| P91_c100_vicreg_views4_800ep_seed1 | 0.1155 | 2217 | 4435 | 16377 | 18 | 7 | 35840000 | None | 1014423@node61 |
| P91_c100_vicreg_views4_800ep_seed2 | 0.2521 | 1015 | 2031 | 35497 | 36 | 10 | 35840000 | None | 1014449@node61 |

## Provenance

- P91_c100_simclr_views4_200ep_seed0: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `5f16e1fb4c570a3f76ab1671662482c0fad51c5bb20d9028ee9fa1767aef65cc`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_simclr_views4_200ep_seed1: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `bc06897ca9600316cebb6b56f27dea3defbce78fc875aab645028e933c9821ea`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_simclr_views4_200ep_seed2: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `4d349abd525ea662a0ab2c4f58633fd05727fa68bd11cc2669e570388dae14f5`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_simclr_views4_800ep_seed0: commit `ca3c3082a43f2853f6b76422eb9f5da0fa5048c1` dirty=True, config `07da1fcaa37c000d8dad93a649add4889a64e55195826b923d8e235a57e3b603`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_simclr_views4_800ep_seed1: commit `8a6ccae580504e6da3405a74a13f8c41daa5816e` dirty=True, config `c8f23f5a3565349c476e1064752f50bc5caf7f256be55d674140655caae08600`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_simclr_views4_800ep_seed2: commit `8a6ccae580504e6da3405a74a13f8c41daa5816e` dirty=True, config `dd3d8a58af64c3be4dd796e6c416f6083c6ed05924f5c700df5f197e26df1334`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_vcs_a5_views4_200ep_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `3446b88e90c7eb90a5e10808ce7b09d0b81bc76dc83745fb811d9196b8c15a04`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_vcs_a5_views4_200ep_seed1: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `47f081f63fd6a303407087e7ba42d72fa1d8f14f38bd18340114259cf9f8d038`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_vcs_a5_views4_200ep_seed2: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `bc5ed3d400460e565dec8826c4c6fd181cca288aecca4709801d249d618e1233`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_vcs_a5_views4_800ep_seed0: commit `ca3c3082a43f2853f6b76422eb9f5da0fa5048c1` dirty=True, config `0b2b7a84fd2bce827e8472859cfb4aea212c4f3defd7d3d46db6cf155ba9b894`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_vcs_a5_views4_800ep_seed1: commit `ca3c3082a43f2853f6b76422eb9f5da0fa5048c1` dirty=True, config `1d651d82d0d3adbfc7014c5cbb2425beafee581b203270ea0dd2bc3fe15bb0ca`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_vcs_a5_views4_800ep_seed2: commit `ca3c3082a43f2853f6b76422eb9f5da0fa5048c1` dirty=True, config `2cee3ea45c3c80b983b9e227fc59675275fb89b7b1011571e0d1aa3702d58f09`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_vicreg_views4_200ep_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `04f8b0d3298e312239306ce98d2f7d6e6ea0e9edea283b4b67a39fda679957a4`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_vicreg_views4_200ep_seed1: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `9e20c965712cbe6ca512258187c9211b3a9011f91fc1c3f7dce764b472e0e461`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_vicreg_views4_200ep_seed2: commit `db99964976a8743f7a186803b90c958096bdff55` dirty=True, config `801494e160b572fef66ab92578561848eb19aff2ddbf358ff8ec06bc5a7e2bd6`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_vicreg_views4_800ep_seed0: commit `8a6ccae580504e6da3405a74a13f8c41daa5816e` dirty=True, config `d944a57ff508e158414e8a0af1fcc826221146d33b709ab36fab6155a2c049e3`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_vicreg_views4_800ep_seed1: commit `8a6ccae580504e6da3405a74a13f8c41daa5816e` dirty=True, config `1d1554089896c3feb5df00ea4870a2ab82f3a209007d4e560dac96a98e2b66f9`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_vicreg_views4_800ep_seed2: commit `8a6ccae580504e6da3405a74a13f8c41daa5816e` dirty=True, config `c3c6321ff5def018c767b3a962ddaa19593e66466d3a1de5939f1a21f85ab703`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 6 | 58.84 ± 0.70 | 19.83 ± 0.38 | 39.00 ± 0.72 | 55.79 ± 1.58 | 166.70 ± 28.42 | null |
| vcs_qmi K=8 | 6 | 58.93 ± 1.08 | 19.83 ± 0.38 | 39.10 ± 1.15 | 51.81 ± 3.45 | 115.74 ± 39.22 | 0.97 ± 0.00 |
| vicreg_matched_128 | 6 | 56.22 ± 0.38 | 19.84 ± 0.39 | 36.38 ± 0.48 | 48.47 ± 0.44 | 113.51 ± 1.61 | null |

## Coverage checks

- P91_c100_simclr_views4_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_simclr_views4_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_simclr_views4_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_simclr_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_simclr_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_simclr_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
