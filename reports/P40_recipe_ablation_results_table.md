# P39_vcs_recipe_ablation — neutral results table (observed values only)

Generated 2026-09-26T08:37:53Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | vcs_qmi | 8 | 0 | 200/200 | 82.14 | 76.42 | 0.9394±0.0017 | 42.78 | 6091 | 2076/3482 | COMPLETED |
| P39_vcs_a5_lr2e-3_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.64 | 77.50 | 0.9424±0.0015 | 50.55 | 5560 | 3683/4842 | COMPLETED |
| P39_vcs_a5_lr5e-4_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.56 | 75.76 | 0.9389±0.0016 | 41.99 | 5613 | 3683/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_hid2048_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.82 | 78.52 | 0.9549±0.0009 | 60.14 | 5803 | 3695/4840 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_k127_seed0 | vcs_qmi | 127 | 0 | 100/100 | 82.86 | 78.92 | 0.9544±0.0011 | 57.77 | 5812 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_out512_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.88 | 78.26 | 0.9538±0.0011 | 56.57 | 5772 | 3684/4846 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 83.52 | 78.30 | 0.9535±0.0011 | 55.62 | 6563 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_seed1 | vcs_qmi | 8 | 1 | 100/100 | 82.74 | 78.48 | 0.9534±0.0013 | 55.51 | 5798 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_seed2 | vcs_qmi | 8 | 2 | 100/100 | 83.30 | 78.50 | 0.9534±0.0014 | 56.26 | 5799 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_warm5_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.00 | 78.50 | 0.9528±0.0010 | 54.11 | 5795 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.54 | 78.12 | 0.9534±0.0009 | 56.97 | 5793 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.96 | 78.28 | 0.9529±0.0012 | 54.78 | 5791 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_200ep_seed0 | vcs_qmi | 8 | 0 | 200/200 | 84.74 | 81.16 | 0.9644±0.0012 | 77.09 | 11582 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b64_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.78 | 78.56 | 0.9522±0.0014 | 52.20 | 6491 | 2077/3482 | COMPLETED |
| P39_vcs_a5_views4_k1_100ep_seed0 | vcs_qmi | 1 | 0 | 100/100 | 81.14 | 75.68 | 0.9418±0.0015 | 45.40 | 6105 | 7568/12670 | COMPLETED |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 79.66 | 75.38 | 0.9750±0.0011 | 85.79 | 6852 | 7568/12674 | COMPLETED |
| P39_vcs_views4_b128_100ep_a1_seed0 | vcs_qmi | 8 | 0 | 100/100 | 82.12 | 77.38 | 0.9446±0.0013 | 38.35 | 5777 | 3682/4842 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 200 |
| P39_vcs_a5_lr2e-3_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.002 | 200 |
| P39_vcs_a5_lr5e-4_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.0005 | 200 |
| P39_vcs_a5_views4_b128_100ep_hid2048_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 2048 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_k127_seed0 | 127 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_out512_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 512 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_warm5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_200ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 200 |
| P39_vcs_a5_views4_b64_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 64 | 0.001 | 100 |
| P39_vcs_a5_views4_k1_100ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |
| P39_vcs_views4_b128_100ep_a1_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | 41.78 | 82.14 | 40.36 | 36.58 | 76.42 | 39.84 | 2.94 | 42.78 | -0.9998 |
| P39_vcs_a5_lr2e-3_seed0 | 41.78 | 81.64 | 39.86 | 36.58 | 77.50 | 40.92 | 2.94 | 50.55 | -0.9998 |
| P39_vcs_a5_lr5e-4_seed0 | 41.78 | 80.56 | 38.78 | 36.58 | 75.76 | 39.18 | 2.94 | 41.99 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_hid2048_seed0 | 41.78 | 82.82 | 41.04 | 36.58 | 78.52 | 41.94 | 2.94 | 60.14 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_k127_seed0 | 41.78 | 82.86 | 41.08 | 36.58 | 78.92 | 42.34 | 2.94 | 57.77 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_out512_seed0 | 41.78 | 82.88 | 41.10 | 36.58 | 78.26 | 41.68 | 2.94 | 56.57 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_seed0 | 41.78 | 83.52 | 41.74 | 36.58 | 78.30 | 41.72 | 2.94 | 55.62 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_seed1 | 41.60 | 82.74 | 41.14 | 37.42 | 78.48 | 41.06 | 3.22 | 55.51 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_seed2 | 43.02 | 83.30 | 40.28 | 37.08 | 78.50 | 41.42 | 3.22 | 56.26 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_warm5_seed0 | 41.78 | 82.00 | 40.22 | 36.58 | 78.50 | 41.92 | 2.94 | 54.11 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0 | 41.78 | 82.54 | 40.76 | 36.58 | 78.12 | 41.54 | 2.94 | 56.97 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0 | 41.78 | 82.96 | 41.18 | 36.58 | 78.28 | 41.70 | 2.94 | 54.78 | -0.9998 |
| P39_vcs_a5_views4_b128_200ep_seed0 | 41.78 | 84.74 | 42.96 | 36.58 | 81.16 | 44.58 | 2.94 | 77.09 | -0.9998 |
| P39_vcs_a5_views4_b64_100ep_seed0 | 41.78 | 82.78 | 41.00 | 36.58 | 78.56 | 41.98 | 2.94 | 52.20 | -0.9998 |
| P39_vcs_a5_views4_k1_100ep_seed0 | 41.78 | 81.14 | 39.36 | 36.58 | 75.68 | 39.10 | 2.94 | 45.40 | -0.9998 |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | 41.78 | 79.66 | 37.88 | 36.58 | 75.38 | 38.80 | 2.94 | 85.79 | -0.9998 |
| P39_vcs_views4_b128_100ep_a1_seed0 | 41.78 | 82.12 | 40.34 | 36.58 | 77.38 | 40.80 | 2.94 | 38.35 | -0.5686 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P39_vcs_a5_b128_seed0: ep0: 36.58, ep10: 53.98, ep20: 63.74, ep50: 70.28, ep100: 73.94, ep150: 76.04, ep200: 76.42
- P39_vcs_a5_lr2e-3_seed0: ep0: 36.58, ep10: 54.26, ep20: 62.26, ep50: 69.66, ep100: 74.40, ep150: 77.22, ep200: 77.50
- P39_vcs_a5_lr5e-4_seed0: ep0: 36.58, ep10: 47.90, ep20: 56.52, ep50: 68.36, ep100: 72.76, ep150: 75.18, ep200: 75.76
- P39_vcs_a5_views4_b128_100ep_hid2048_seed0: ep0: 36.58, ep10: 61.56, ep20: 70.84, ep50: 76.02, ep100: 78.52
- P39_vcs_a5_views4_b128_100ep_k127_seed0: ep0: 36.58, ep10: 62.26, ep20: 70.40, ep50: 75.92, ep100: 78.92
- P39_vcs_a5_views4_b128_100ep_out512_seed0: ep0: 36.58, ep10: 62.64, ep20: 70.78, ep50: 75.86, ep100: 78.26
- P39_vcs_a5_views4_b128_100ep_seed0: ep0: 36.58, ep10: 61.82, ep20: 70.32, ep50: 75.78, ep100: 78.30
- P39_vcs_a5_views4_b128_100ep_seed1: ep0: 37.42, ep10: 62.32, ep20: 69.74, ep50: 76.42, ep100: 78.48
- P39_vcs_a5_views4_b128_100ep_seed2: ep0: 37.08, ep10: 61.48, ep20: 68.64, ep50: 76.20, ep100: 78.50
- P39_vcs_a5_views4_b128_100ep_warm5_seed0: ep0: 36.58, ep10: 64.76, ep20: 71.38, ep50: 75.98, ep100: 78.50
- P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0: ep0: 36.58, ep10: 61.68, ep20: 70.26, ep50: 76.28, ep100: 78.12
- P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0: ep0: 36.58, ep10: 62.80, ep20: 70.42, ep50: 75.70, ep100: 78.28
- P39_vcs_a5_views4_b128_200ep_seed0: ep0: 36.58, ep10: 61.14, ep20: 69.90, ep50: 75.32, ep100: 79.08, ep150: 81.02, ep200: 81.16
- P39_vcs_a5_views4_b64_100ep_seed0: ep0: 36.58, ep10: 64.02, ep20: 71.18, ep50: 76.26, ep100: 78.56
- P39_vcs_a5_views4_k1_100ep_seed0: ep0: 36.58, ep10: 54.00, ep20: 64.70, ep50: 72.80, ep100: 75.68
- P39_vcs_a5_views4_nodetach_100ep_seed0: ep0: 36.58, ep10: 58.56, ep20: 67.24, ep50: 72.82, ep100: 75.38
- P39_vcs_views4_b128_100ep_a1_seed0: ep0: 36.58, ep10: 63.50, ep20: 69.26, ep50: 74.46, ep100: 77.38

## Held-out J trajectory (VCS only)

- P39_vcs_a5_b128_seed0: ep0: -0.9998, ep10: 0.7479, ep20: 0.8090, ep50: 0.8701, ep100: 0.9138, ep150: 0.9331, ep200: 0.9394
- P39_vcs_a5_lr2e-3_seed0: ep0: -0.9998, ep10: 0.7049, ep20: 0.7988, ep50: 0.8693, ep100: 0.9150, ep150: 0.9348, ep200: 0.9424
- P39_vcs_a5_lr5e-4_seed0: ep0: -0.9998, ep10: 0.5000, ep20: 0.7746, ep50: 0.8648, ep100: 0.9138, ep150: 0.9323, ep200: 0.9389
- P39_vcs_a5_views4_b128_100ep_hid2048_seed0: ep0: -0.9998, ep10: 0.7921, ep20: 0.8843, ep50: 0.9270, ep100: 0.9549
- P39_vcs_a5_views4_b128_100ep_k127_seed0: ep0: -0.9998, ep10: 0.8162, ep20: 0.8848, ep50: 0.9284, ep100: 0.9544
- P39_vcs_a5_views4_b128_100ep_out512_seed0: ep0: -0.9998, ep10: 0.8278, ep20: 0.8900, ep50: 0.9283, ep100: 0.9538
- P39_vcs_a5_views4_b128_100ep_seed0: ep0: -0.9998, ep10: 0.7937, ep20: 0.8825, ep50: 0.9247, ep100: 0.9535
- P39_vcs_a5_views4_b128_100ep_seed1: ep0: -0.9998, ep10: 0.8101, ep20: 0.8731, ep50: 0.9295, ep100: 0.9534
- P39_vcs_a5_views4_b128_100ep_seed2: ep0: -0.9998, ep10: 0.7663, ep20: 0.8674, ep50: 0.9284, ep100: 0.9534
- P39_vcs_a5_views4_b128_100ep_warm5_seed0: ep0: -0.9998, ep10: 0.8354, ep20: 0.8833, ep50: 0.9266, ep100: 0.9528
- P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0: ep0: -0.9998, ep10: 0.7960, ep20: 0.8814, ep50: 0.9264, ep100: 0.9534
- P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0: ep0: -0.9998, ep10: 0.7971, ep20: 0.8755, ep50: 0.9242, ep100: 0.9529
- P39_vcs_a5_views4_b128_200ep_seed0: ep0: -0.9998, ep10: 0.8085, ep20: 0.8794, ep50: 0.9234, ep100: 0.9492, ep150: 0.9604, ep200: 0.9644
- P39_vcs_a5_views4_b64_100ep_seed0: ep0: -0.9998, ep10: 0.8094, ep20: 0.8803, ep50: 0.9307, ep100: 0.9522
- P39_vcs_a5_views4_k1_100ep_seed0: ep0: -0.9998, ep10: 0.6888, ep20: 0.8362, ep50: 0.9043, ep100: 0.9418
- P39_vcs_a5_views4_nodetach_100ep_seed0: ep0: -0.9998, ep10: 0.7615, ep20: 0.8859, ep50: 0.9492, ep100: 0.9750
- P39_vcs_views4_b128_100ep_a1_seed0: ep0: -0.5686, ep10: 0.7483, ep20: 0.8774, ep50: 0.9211, ep100: 0.9446

## Final-epoch training objective values (epoch means)

- P39_vcs_a5_b128_seed0: J_raw 0.9440, R_binary 0.0560, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_lr2e-3_seed0: J_raw 0.9472, R_binary 0.0528, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_lr5e-4_seed0: J_raw 0.9430, R_binary 0.0570, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_hid2048_seed0: J_raw 0.9607, R_binary 0.0393, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_k127_seed0: J_raw 0.9602, R_binary 0.0398, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_out512_seed0: J_raw 0.9595, R_binary 0.0405, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_seed0: J_raw 0.9591, R_binary 0.0409, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_seed1: J_raw 0.9589, R_binary 0.0411, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_seed2: J_raw 0.9594, R_binary 0.0406, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_warm5_seed0: J_raw 0.9585, R_binary 0.0415, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0: J_raw 0.9590, R_binary 0.0410, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0: J_raw 0.9587, R_binary 0.0413, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_200ep_seed0: J_raw 0.9711, R_binary 0.0289, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b64_100ep_seed0: J_raw 0.9585, R_binary 0.0415, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_k1_100ep_seed0: J_raw 0.9470, R_binary 0.0530, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_nodetach_100ep_seed0: J_raw 0.9785, R_binary 0.0215, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_views4_b128_100ep_a1_seed0: J_raw 0.9504, R_binary 0.0496, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | 0.0862 | 1485 | 2970 | 6091 | 125 | 29 | 8985600 | 2 | 1009342@nodeaudible01 |
| P39_vcs_a5_lr2e-3_seed0 | 0.1570 | 1631 | 3262 | 5560 | 125 | 28 | 8960000 | 2 | 1009275@nodeaudible01 |
| P39_vcs_a5_lr5e-4_seed0 | 0.1584 | 1616 | 3233 | 5613 | 134 | 30 | 8960000 | 2 | 1009276@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_hid2048_seed0 | 0.1643 | 779 | 1558 | 5803 | 90 | 28 | 4492800 | 2 | 1009356@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_k127_seed0 | 0.1644 | 778 | 1557 | 5812 | 96 | 29 | 4492800 | 2 | 1009361@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_out512_seed0 | 0.1635 | 783 | 1566 | 5772 | 92 | 29 | 4492800 | 2 | 1009355@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_seed0 | 0.1857 | 689 | 1379 | 6563 | 112 | 38 | 4492800 | 2 | 1009274@node01 |
| P39_vcs_a5_views4_b128_100ep_seed1 | 0.1642 | 780 | 1559 | 5798 | 90 | 29 | 4492800 | 2 | 1009350@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_seed2 | 0.1641 | 780 | 1560 | 5799 | 95 | 30 | 4492800 | 2 | 1009351@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_warm5_seed0 | 0.1641 | 780 | 1560 | 5795 | 90 | 28 | 4492800 | 2 | 1009369@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0 | 0.1641 | 780 | 1560 | 5793 | 91 | 29 | 4492800 | 2 | 1009360@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0 | 0.1640 | 780 | 1561 | 5791 | 90 | 28 | 4492800 | 2 | 1009359@nodeaudible01 |
| P39_vcs_a5_views4_b128_200ep_seed0 | 0.1639 | 781 | 1561 | 11582 | 129 | 28 | 8985600 | 2 | 1009341@nodeaudible01 |
| P39_vcs_a5_views4_b64_100ep_seed0 | 0.0920 | 696 | 1391 | 6491 | 91 | 28 | 4499200 | 2 | 1009340@nodeaudible01 |
| P39_vcs_a5_views4_k1_100ep_seed0 | 0.3454 | 741 | 1482 | 6105 | 93 | 31 | 4480000 | 2 | 1009272@nodeaudible01 |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | 0.3869 | 662 | 1323 | 6852 | 114 | 37 | 4480000 | 2 | 1009273@node06 |
| P39_vcs_views4_b128_100ep_a1_seed0 | 0.1636 | 782 | 1565 | 5777 | 92 | 29 | 4492800 | 2 | 1009368@nodeaudible01 |

## Provenance

- P39_vcs_a5_b128_seed0: commit `8626b9834cd3c677071cf2f8914dd8daeb49b009` dirty=True, config `da792c7d66978c2b12473fc2db68c2dbd51c9aebf700b394c87378bd8d725e71`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_lr2e-3_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `56e68e9af470149201fe6534849de401df4ba159f15251fed4144ca191e240a4`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_lr5e-4_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `009a3de7ce0e83ef3cd568a1ca0d33605a563c998ff192b66d81933fad1ee0e6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_hid2048_seed0: commit `23b79d9ce43f013708ea8826af5993b33c19b441` dirty=False, config `b367116da3012e75f823eacd947fe172e199c08193e1c5c098569a889f180f24`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_k127_seed0: commit `a0c861a07c3f86c07dce99db865431b50c45c446` dirty=True, config `83cf0bedb15ee81dff733adb701596ac58ea8933c1197adf2a9f79326add9eac`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_out512_seed0: commit `23b79d9ce43f013708ea8826af5993b33c19b441` dirty=False, config `47fb2d09deedcc6e86174f6d1104d580de4f8b23e394f54b2a5582f33df7d303`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `94ada458c11336bc97c00843ebc88c08315a19cc7f9a369053a5b409a3875c80`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_seed1: commit `0853b9b643ddf40495961f54910b24c5cd9b9cc8` dirty=True, config `6533529b73eecd57f1607d618e57112c1b7b80fc585863f5d41d04d52a89cbc9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P39_vcs_a5_views4_b128_100ep_seed2: commit `0853b9b643ddf40495961f54910b24c5cd9b9cc8` dirty=True, config `ef09b43c099c7072434d37c640410bcaf70c0d317875186228b67e8c812bcc6e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P39_vcs_a5_views4_b128_100ep_warm5_seed0: commit `6b975c7eadad32e48b9d8eff635380536a391f9b` dirty=True, config `3923f9f10ec57ccf4499780ee6e9942524aedc81cd09cea64e66474a9f07b7fb`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0: commit `a0c861a07c3f86c07dce99db865431b50c45c446` dirty=True, config `fc771efa95db5e214580bbe9366a275d049c5c0f1f2c4ea9a78c853e6f8d55cf`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0: commit `a0c861a07c3f86c07dce99db865431b50c45c446` dirty=True, config `0bb2e2c5bd6a1a5118a57e0bb390dddfab8172ab6920815aa879efe2ebb5addc`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_200ep_seed0: commit `8626b9834cd3c677071cf2f8914dd8daeb49b009` dirty=False, config `97cb173c1c0a6bc41a85056786b85e3a12bce46b82eed0d0c9475c3be796d463`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b64_100ep_seed0: commit `8626b9834cd3c677071cf2f8914dd8daeb49b009` dirty=False, config `2f81635f2a94542e6399e017d811115d10a4cd3ed791f12f8fb52d57752eeeff`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_k1_100ep_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `ff6c645b31653aecc9178a9289a62854b5eca80c4af4cd0a2f760432a74e3161`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_nodetach_100ep_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `23e2210d0978099a3adedc735cf6628c7cb3e475a87f9772f474c8f2622439f8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_views4_b128_100ep_a1_seed0: commit `6b975c7eadad32e48b9d8eff635380536a391f9b` dirty=True, config `b2788d07ee5e97d361cd88a2c620a5f67ea8ad7cd3f98f68f36407092e234a9d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=1 | 1 | 81.14 | 41.78 | 39.36 | 75.68 | 45.40 | 0.94 |
| vcs_qmi K=127 | 1 | 82.86 | 41.78 | 41.08 | 78.92 | 57.77 | 0.95 |
| vcs_qmi K=8 | 15 | 82.43 ± 1.20 | 41.85 ± 0.33 | 40.58 ± 1.19 | 77.94 ± 1.38 | 55.91 ± 12.20 | 0.95 ± 0.01 |

## Coverage checks

- P39_vcs_a5_b128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P39_vcs_a5_lr2e-3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P39_vcs_a5_lr5e-4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_hid2048_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_k127_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_out512_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_warm5_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_wd5e-4_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_wd5e-5_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b64_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_k1_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_nodetach_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_views4_b128_100ep_a1_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
