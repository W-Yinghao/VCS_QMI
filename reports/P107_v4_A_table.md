# P107_v4_A_batch — neutral results table (observed values only)

Generated 2026-10-03T08:20:46Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.54 | 86.80 | 0.8499±0.0019 | 92.97 | 15998 | 5269/8140 | COMPLETED |
| P107_AL2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.32 | 83.88 | 0.9759±0.0014 | 93.14 | 16089 | 5269/8140 | COMPLETED |
| P107_AP1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.82 | 87.30 | 0.8612±0.0026 | 119.10 | 16119 | 5269/8142 | COMPLETED |
| P107_AP2F_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.90 | 85.08 | 0.9684±0.0012 | 100.61 | 16150 | 5269/8154 | COMPLETED |
| P107_AP2F_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 86.74 | 84.88 | 0.9685±0.0016 | 104.36 | 16089 | 5269/8154 | COMPLETED |
| P107_AP2F_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.28 | 85.22 | 0.9688±0.0007 | 101.27 | 16135 | 5269/8154 | COMPLETED |
| P107_AP2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.98 | 86.60 | 0.8612±0.0014 | 98.97 | 27839 | 7066/10154 | COMPLETED |
| P107_AP2_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 88.18 | 86.78 | 0.8610±0.0016 | 99.98 | 16036 | 5269/8146 | COMPLETED |
| P107_AP2_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 88.04 | 86.82 | 0.8620±0.0023 | 101.63 | 16111 | 5269/8146 | COMPLETED |
| P107_AP3F_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 83.20 | 79.38 | 0.9880±0.0011 | 123.60 | 35271 | 5269/8154 | COMPLETED |
| P107_AP3F_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 82.62 | 79.28 | 0.9884±0.0009 | 124.42 | 16178 | 5269/8154 | COMPLETED |
| P107_AP3F_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 82.94 | 79.52 | 0.9883±0.0007 | 123.43 | 35227 | 5269/8154 | COMPLETED |
| P107_AP3_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 89.06 | 87.30 | 0.8604±0.0029 | 121.12 | 16053 | 5269/8146 | COMPLETED |
| P107_AP3_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 88.96 | 87.48 | 0.8607±0.0030 | 116.88 | 16099 | 5269/8146 | COMPLETED |
| P107_AP3_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 89.06 | 87.36 | 0.8608±0.0031 | 119.25 | 16100 | 5269/8146 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AL2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP2F_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP2F_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP2F_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP2_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP2_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3F_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP3F_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP3F_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP3_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | 41.94 | 88.54 | 46.60 | 36.54 | 86.80 | 50.26 | 2.94 | 92.97 | -0.5569 |
| P107_AL2_views4_800ep_seed0 | 41.94 | 86.32 | 44.38 | 36.54 | 83.88 | 47.34 | 2.94 | 93.14 | -0.5569 |
| P107_AP1_views4_800ep_seed0 | 41.94 | 88.82 | 46.88 | 36.54 | 87.30 | 50.76 | 2.94 | 119.10 | -0.5569 |
| P107_AP2F_views4_800ep_seed0 | 41.94 | 86.90 | 44.96 | 36.54 | 85.08 | 48.54 | 2.94 | 100.61 | -0.5569 |
| P107_AP2F_views4_800ep_seed1 | 41.68 | 86.74 | 45.06 | 37.40 | 84.88 | 47.48 | 3.22 | 104.36 | -0.5511 |
| P107_AP2F_views4_800ep_seed2 | 42.98 | 86.28 | 43.30 | 37.08 | 85.22 | 48.14 | 3.22 | 101.27 | -0.5471 |
| P107_AP2_views4_800ep_seed0 | 41.98 | 88.98 | 47.00 | 36.58 | 86.60 | 50.02 | 2.94 | 98.97 | -0.5569 |
| P107_AP2_views4_800ep_seed1 | 41.68 | 88.18 | 46.50 | 37.40 | 86.78 | 49.38 | 3.22 | 99.98 | -0.5511 |
| P107_AP2_views4_800ep_seed2 | 42.98 | 88.04 | 45.06 | 37.08 | 86.82 | 49.74 | 3.22 | 101.63 | -0.5471 |
| P107_AP3F_views4_800ep_seed0 | 41.94 | 83.20 | 41.26 | 36.54 | 79.38 | 42.84 | 2.94 | 123.60 | -0.5569 |
| P107_AP3F_views4_800ep_seed1 | 41.68 | 82.62 | 40.94 | 37.40 | 79.28 | 41.88 | 3.22 | 124.42 | -0.5511 |
| P107_AP3F_views4_800ep_seed2 | 42.98 | 82.94 | 39.96 | 37.08 | 79.52 | 42.44 | 3.22 | 123.43 | -0.5471 |
| P107_AP3_views4_800ep_seed0 | 41.94 | 89.06 | 47.12 | 36.54 | 87.30 | 50.76 | 2.94 | 121.12 | -0.5569 |
| P107_AP3_views4_800ep_seed1 | 41.68 | 88.96 | 47.28 | 37.40 | 87.48 | 50.08 | 3.22 | 116.88 | -0.5511 |
| P107_AP3_views4_800ep_seed2 | 42.98 | 89.06 | 46.08 | 37.08 | 87.36 | 50.28 | 3.22 | 119.25 | -0.5471 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P107_AL1_views4_800ep_seed0: ep0: 36.54, ep20: 67.10, ep50: 75.84, ep100: 81.28, ep200: 83.82, ep400: 85.78, ep600: 86.54, ep800: 86.80
- P107_AL2_views4_800ep_seed0: ep0: 36.54, ep20: 66.86, ep50: 72.58, ep100: 76.64, ep200: 79.96, ep400: 82.96, ep600: 83.60, ep800: 83.88
- P107_AP1_views4_800ep_seed0: ep0: 36.54, ep20: 69.80, ep50: 77.24, ep100: 81.30, ep200: 84.58, ep400: 86.62, ep600: 87.24, ep800: 87.30
- P107_AP2F_views4_800ep_seed0: ep0: 36.54, ep20: 67.10, ep50: 73.44, ep100: 78.06, ep200: 81.34, ep400: 84.04, ep600: 84.64, ep800: 85.08
- P107_AP2F_views4_800ep_seed1: ep0: 37.40, ep20: 67.32, ep50: 74.02, ep100: 77.72, ep200: 81.36, ep400: 83.78, ep600: 84.88, ep800: 84.88
- P107_AP2F_views4_800ep_seed2: ep0: 37.08, ep20: 66.64, ep50: 73.38, ep100: 77.42, ep200: 81.18, ep400: 83.58, ep600: 84.46, ep800: 85.22
- P107_AP2_views4_800ep_seed0: ep0: 36.58, ep20: 68.34, ep50: 75.94, ep100: 80.60, ep200: 83.86, ep400: 86.22, ep600: 86.48, ep800: 86.60
- P107_AP2_views4_800ep_seed1: ep0: 37.40, ep20: 68.76, ep50: 75.66, ep100: 80.84, ep200: 83.82, ep400: 86.08, ep600: 86.80, ep800: 86.78
- P107_AP2_views4_800ep_seed2: ep0: 37.08, ep20: 67.86, ep50: 75.90, ep100: 81.50, ep200: 84.58, ep400: 85.88, ep600: 86.36, ep800: 86.82
- P107_AP3F_views4_800ep_seed0: ep0: 36.54, ep20: 69.58, ep50: 73.08, ep100: 75.14, ep200: 76.58, ep400: 78.20, ep600: 79.00, ep800: 79.38
- P107_AP3F_views4_800ep_seed1: ep0: 37.40, ep20: 69.20, ep50: 72.30, ep100: 75.92, ep200: 77.58, ep400: 78.26, ep600: 79.40, ep800: 79.28
- P107_AP3F_views4_800ep_seed2: ep0: 37.08, ep20: 69.60, ep50: 73.14, ep100: 74.82, ep200: 76.82, ep400: 78.28, ep600: 79.04, ep800: 79.52
- P107_AP3_views4_800ep_seed0: ep0: 36.54, ep20: 70.54, ep50: 77.50, ep100: 81.78, ep200: 84.58, ep400: 86.52, ep600: 86.92, ep800: 87.30
- P107_AP3_views4_800ep_seed1: ep0: 37.40, ep20: 69.96, ep50: 76.80, ep100: 81.50, ep200: 84.76, ep400: 86.62, ep600: 87.30, ep800: 87.48
- P107_AP3_views4_800ep_seed2: ep0: 37.08, ep20: 70.20, ep50: 78.06, ep100: 82.40, ep200: 84.98, ep400: 86.40, ep600: 87.04, ep800: 87.36

## Held-out J trajectory (VCS only)

- P107_AL1_views4_800ep_seed0: ep0: -0.5569, ep20: 0.7651, ep50: 0.8076, ep100: 0.8327, ep200: 0.8442, ep400: 0.8497, ep600: 0.8497, ep800: 0.8499
- P107_AL2_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8539, ep50: 0.9111, ep100: 0.9414, ep200: 0.9599, ep400: 0.9702, ep600: 0.9747, ep800: 0.9759
- P107_AP1_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8024, ep50: 0.8314, ep100: 0.8518, ep200: 0.8614, ep400: 0.8621, ep600: 0.8617, ep800: 0.8612
- P107_AP2F_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8561, ep50: 0.9023, ep100: 0.9321, ep200: 0.9506, ep400: 0.9618, ep600: 0.9665, ep800: 0.9684
- P107_AP2F_views4_800ep_seed1: ep0: -0.5511, ep20: 0.8361, ep50: 0.9032, ep100: 0.9299, ep200: 0.9505, ep400: 0.9619, ep600: 0.9670, ep800: 0.9685
- P107_AP2F_views4_800ep_seed2: ep0: -0.5471, ep20: 0.8327, ep50: 0.8972, ep100: 0.9269, ep200: 0.9490, ep400: 0.9626, ep600: 0.9675, ep800: 0.9688
- P107_AP2_views4_800ep_seed0: ep0: -0.5569, ep20: 0.7791, ep50: 0.8188, ep100: 0.8421, ep200: 0.8533, ep400: 0.8599, ep600: 0.8608, ep800: 0.8612
- P107_AP2_views4_800ep_seed1: ep0: -0.5511, ep20: 0.7699, ep50: 0.8181, ep100: 0.8400, ep200: 0.8524, ep400: 0.8580, ep600: 0.8606, ep800: 0.8610
- P107_AP2_views4_800ep_seed2: ep0: -0.5471, ep20: 0.7717, ep50: 0.8145, ep100: 0.8378, ep200: 0.8540, ep400: 0.8599, ep600: 0.8615, ep800: 0.8620
- P107_AP3F_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8985, ep50: 0.9475, ep100: 0.9669, ep200: 0.9782, ep400: 0.9839, ep600: 0.9872, ep800: 0.9880
- P107_AP3F_views4_800ep_seed1: ep0: -0.5511, ep20: 0.8861, ep50: 0.9475, ep100: 0.9682, ep200: 0.9790, ep400: 0.9849, ep600: 0.9878, ep800: 0.9884
- P107_AP3F_views4_800ep_seed2: ep0: -0.5471, ep20: 0.8861, ep50: 0.9393, ep100: 0.9637, ep200: 0.9777, ep400: 0.9847, ep600: 0.9878, ep800: 0.9883
- P107_AP3_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8028, ep50: 0.8333, ep100: 0.8528, ep200: 0.8605, ep400: 0.8613, ep600: 0.8611, ep800: 0.8604
- P107_AP3_views4_800ep_seed1: ep0: -0.5511, ep20: 0.8034, ep50: 0.8347, ep100: 0.8514, ep200: 0.8592, ep400: 0.8613, ep600: 0.8606, ep800: 0.8607
- P107_AP3_views4_800ep_seed2: ep0: -0.5471, ep20: 0.7893, ep50: 0.8337, ep100: 0.8505, ep200: 0.8597, ep400: 0.8621, ep600: 0.8613, ep800: 0.8608

## Final-epoch training objective values (epoch means)

- P107_AL1_views4_800ep_seed0: J_raw 0.8873, R_binary 0.1127, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AL2_views4_800ep_seed0: J_raw 0.9865, R_binary 0.0135, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP1_views4_800ep_seed0: J_raw 0.9030, R_binary 0.0970, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2F_views4_800ep_seed0: J_raw 0.9822, R_binary 0.0178, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2F_views4_800ep_seed1: J_raw 0.9824, R_binary 0.0176, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2F_views4_800ep_seed2: J_raw 0.9820, R_binary 0.0180, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2_views4_800ep_seed0: J_raw 0.8948, R_binary 0.1052, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2_views4_800ep_seed1: J_raw 0.8947, R_binary 0.1053, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2_views4_800ep_seed2: J_raw 0.8947, R_binary 0.1053, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3F_views4_800ep_seed0: J_raw 0.9939, R_binary 0.0061, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3F_views4_800ep_seed1: J_raw 0.9938, R_binary 0.0062, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3F_views4_800ep_seed2: J_raw 0.9937, R_binary 0.0063, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_views4_800ep_seed0: J_raw 0.9034, R_binary 0.0966, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_views4_800ep_seed1: J_raw 0.9032, R_binary 0.0968, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_views4_800ep_seed2: J_raw 0.9026, R_binary 0.0974, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | 0.1127 | 2271 | 4542 | 15998 | 56 | 12 | 35840000 | None | 1017350@node58 |
| P107_AL2_views4_800ep_seed0 | 0.1134 | 2257 | 4514 | 16089 | 51 | 17 | 35840000 | 2 | 1017352@node59 |
| P107_AP1_views4_800ep_seed0 | 0.1137 | 2252 | 4504 | 16119 | 52 | 11 | 35840000 | None | 1017353@node59 |
| P107_AP2F_views4_800ep_seed0 | 0.1139 | 2248 | 4497 | 16150 | 53 | 11 | 35840000 | 2 | 1017932@node59 |
| P107_AP2F_views4_800ep_seed1 | 0.1134 | 2257 | 4513 | 16089 | 53 | 12 | 35840000 | 2 | 1017933@node61 |
| P107_AP2F_views4_800ep_seed2 | 0.1138 | 2250 | 4500 | 16135 | 51 | 12 | 35840000 | 2 | 1017934@node58 |
| P107_AP2_views4_800ep_seed0 | 0.1960 | 1306 | 2612 | 27839 | 101 | 21 | 35840000 | None | 1017354@nodesumo01 |
| P107_AP2_views4_800ep_seed1 | 0.1130 | 2265 | 4529 | 16036 | 52 | 11 | 35840000 | None | 1017928@node61 |
| P107_AP2_views4_800ep_seed2 | 0.1136 | 2253 | 4506 | 16111 | 52 | 11 | 35840000 | None | 1017929@node58 |
| P107_AP3F_views4_800ep_seed0 | 0.2505 | 1022 | 2044 | 35271 | 85 | 16 | 35840000 | 2 | 1017935@node61 |
| P107_AP3F_views4_800ep_seed1 | 0.1140 | 2245 | 4490 | 16178 | 56 | 12 | 35840000 | 2 | 1017936@node58 |
| P107_AP3F_views4_800ep_seed2 | 0.2502 | 1023 | 2046 | 35227 | 85 | 15 | 35840000 | 2 | 1017937@node60 |
| P107_AP3_views4_800ep_seed0 | 0.1132 | 2262 | 4524 | 16053 | 52 | 12 | 35840000 | None | 1017356@node61 |
| P107_AP3_views4_800ep_seed1 | 0.1134 | 2257 | 4514 | 16099 | 56 | 12 | 35840000 | None | 1017930@node58 |
| P107_AP3_views4_800ep_seed2 | 0.1135 | 2255 | 4510 | 16100 | 51 | 11 | 35840000 | None | 1017931@node59 |

## Provenance

- P107_AL1_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `4aa524331e6d6dcc1ca168709dee3954585631c505dc541943e3a65016f9bac9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AL2_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `c1308c3a95978a9e9b7f1cfea620b8fb0c18190b8313b6af032ef1ce2b463b2a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP1_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `11b5a305c98f71d4db2e5b524a1ec093c1668c6f759031a08f27bf14d55bee82`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP2F_views4_800ep_seed0: commit `3df5fa3fcbb4e32db24dbcb9fa1dcdd462ff9fea` dirty=True, config `65c11981add226e1227000fd72a44acf57c15560fa01c71dc55e3d321fe76583`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP2F_views4_800ep_seed1: commit `3df5fa3fcbb4e32db24dbcb9fa1dcdd462ff9fea` dirty=True, config `e95c4a70b065c81211999b6f2fe7c569b3d644cc93d2543adef160bbf9b006d3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P107_AP2F_views4_800ep_seed2: commit `3df5fa3fcbb4e32db24dbcb9fa1dcdd462ff9fea` dirty=True, config `5a386a2c51cfc480ba8dab30396febb92d6945787309cb321cd4649a7c7aad1e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P107_AP2_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `ebf62c1d8c4d5b7c87adae4d6c19b47d4f894eb9b7221dea7bf21b4401c2c718`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP2_views4_800ep_seed1: commit `60f5cdfc665864846253471939063c33df7c99bb` dirty=True, config `b4d1e9d7e7a08b29af3f7de68891356d081a1997c6276cd5fdfca3b8ba720393`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P107_AP2_views4_800ep_seed2: commit `60f5cdfc665864846253471939063c33df7c99bb` dirty=True, config `1aa95a446d710c4476094815a6d531a09ddfa1287fa01bdaefa4ec141fe04353`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P107_AP3F_views4_800ep_seed0: commit `3df5fa3fcbb4e32db24dbcb9fa1dcdd462ff9fea` dirty=True, config `dac434e55ae59a0b0c4e2336a0d997dcf87e5cc776b9e4359c2b1b7876ec1206`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP3F_views4_800ep_seed1: commit `3df5fa3fcbb4e32db24dbcb9fa1dcdd462ff9fea` dirty=True, config `71771bb98d87de6f9acbdfbd319d9755f0c5de6d6b9da463debc6d6cd149fa58`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P107_AP3F_views4_800ep_seed2: commit `5e6aa86b6787535b7d708a2036e5855bc5d7d4ae` dirty=True, config `fa109930ac76fc26ab5fba1c17a39f9f1017003781104fdd0ef8688cc2faa1d0`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P107_AP3_views4_800ep_seed0: commit `f29a6b261d66696034ff8b1b86d158a1c8480ac8` dirty=True, config `2f3e24e0c5d7386fa68c223ad8991752e16776378bbe66882554e0a0a5c60140`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP3_views4_800ep_seed1: commit `a468c6bba0cffbecc3d5d40e2674be4c8291d734` dirty=True, config `efe0d32ff7fe603f8a56b970a5efd4a5467dee978cb3478915c44afa6f0842c4`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P107_AP3_views4_800ep_seed2: commit `3df5fa3fcbb4e32db24dbcb9fa1dcdd462ff9fea` dirty=True, config `ed3a7c0fc08c4d01868a0411737a77eb346554fd3a99f4d8e6e0735f46f94e91`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 15 | 86.98 ± 2.32 | 42.15 ± 0.53 | 44.83 ± 2.42 | 84.91 ± 3.05 | 109.38 ± 11.86 | 0.91 ± 0.06 |

## Coverage checks

- P107_AL1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AL2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2F_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2F_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2F_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3F_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3F_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3F_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
