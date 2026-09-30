# P95_v1_in_ssl — neutral results table (observed values only)

Generated 2026-09-30T22:59:42Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P95_dictionary_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.56 | 83.20 | 0.9631±0.0014 | 73.44 | 18233 | 5281/8168 | COMPLETED |
| P95_dictionary_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 84.60 | 82.64 | 0.9570±0.0019 | 1.96 | 18347 | 5281/8168 | COMPLETED |
| P95_dictionary_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.20 | 83.68 | 0.9649±0.0017 | 100.52 | 18310 | 5281/8168 | COMPLETED |
| P95_js_matched_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.20 | 84.60 | 0.9822±0.0011 | 137.40 | 27777 | 7066/10148 | COMPLETED |
| P95_js_matched_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 86.62 | 85.06 | 0.9824±0.0012 | 134.51 | 27799 | 7066/10148 | COMPLETED |
| P95_js_matched_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.68 | 85.00 | 0.9822±0.0014 | 135.93 | 16066 | 5269/8140 | COMPLETED |
| P95_noise_tau0.1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.52 | 85.42 | 0.9760±0.0008 | 131.28 | 16398 | 5269/8140 | COMPLETED |
| P95_noise_tau0.3_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.32 | 84.92 | 0.9794±0.0018 | 121.76 | 16325 | 5269/8140 | COMPLETED |
| P95_noise_tau0.3_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 87.12 | 84.98 | 0.9785±0.0010 | 123.20 | 16365 | 5269/8140 | COMPLETED |
| P95_noise_tau0.3_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 87.02 | 84.82 | 0.9795±0.0011 | 123.30 | 16349 | 5269/8140 | COMPLETED |
| P95_refresh_R200_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 85.94 | 84.38 | 0.9771±0.0010 | 130.04 | 16089 | 5269/8140 | COMPLETED |
| P95_refresh_R200_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 85.78 | 84.46 | 0.9854±0.0010 | 137.77 | 16103 | 5269/8140 | COMPLETED |
| P95_refresh_R200_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.42 | 84.46 | 0.9829±0.0015 | 132.00 | 16135 | 5269/8140 | COMPLETED |
| P95_refresh_R50_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 84.96 | 82.14 | 0.9879±0.0006 | 126.64 | 16091 | 5269/8140 | COMPLETED |
| P95_residual_lam0.25_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.02 | 80.90 | 0.9464±0.0021 | 5.54 | 28842 | 7071/10156 | COMPLETED |
| P95_residual_lam0.25_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 86.06 | 81.26 | 0.9470±0.0021 | 9.75 | 17120 | 5274/8148 | COMPLETED |
| P95_residual_lam0.25_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 86.32 | 80.86 | 0.9479±0.0022 | 22.20 | 36279 | 5274/8148 | COMPLETED |
| P95_residual_lam0.5_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 85.74 | 83.54 | 0.4823±0.0014 | 89.23 | 17138 | 5274/8148 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P95_dictionary_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | DictionarySimplexCritic | 256 | 0.001 | 800 |
| P95_dictionary_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | DictionarySimplexCritic | 256 | 0.001 | 800 |
| P95_dictionary_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | DictionarySimplexCritic | 256 | 0.001 | 800 |
| P95_js_matched_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_js_matched_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_js_matched_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_noise_tau0.1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | NoisyCosineCritic | 256 | 0.001 | 800 |
| P95_noise_tau0.3_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | NoisyCosineCritic | 256 | 0.001 | 800 |
| P95_noise_tau0.3_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | NoisyCosineCritic | 256 | 0.001 | 800 |
| P95_noise_tau0.3_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | NoisyCosineCritic | 256 | 0.001 | 800 |
| P95_refresh_R200_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_refresh_R200_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_refresh_R200_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_refresh_R50_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P95_residual_lam0.25_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | ResidualCosineMLPCritic | 256 | 0.001 | 800 |
| P95_residual_lam0.25_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | ResidualCosineMLPCritic | 256 | 0.001 | 800 |
| P95_residual_lam0.25_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | ResidualCosineMLPCritic | 256 | 0.001 | 800 |
| P95_residual_lam0.5_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | ResidualCosineMLPCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P95_dictionary_views4_800ep_seed0 | 41.94 | 86.56 | 44.62 | 36.54 | 83.20 | 46.66 | 2.94 | 73.44 | -0.1110 |
| P95_dictionary_views4_800ep_seed1 | 41.68 | 84.60 | 42.92 | 37.40 | 82.64 | 45.24 | 3.22 | 1.96 | -0.1057 |
| P95_dictionary_views4_800ep_seed2 | 42.98 | 86.20 | 43.22 | 37.08 | 83.68 | 46.60 | 3.22 | 100.52 | -0.1118 |
| P95_js_matched_views4_800ep_seed0 | 41.98 | 86.20 | 44.22 | 36.58 | 84.60 | 48.02 | 2.94 | 137.40 | -0.9998 |
| P95_js_matched_views4_800ep_seed1 | 41.72 | 86.62 | 44.90 | 37.42 | 85.06 | 47.64 | 3.22 | 134.51 | -0.9998 |
| P95_js_matched_views4_800ep_seed2 | 42.98 | 86.68 | 43.70 | 37.08 | 85.00 | 47.92 | 3.22 | 135.93 | -0.9998 |
| P95_noise_tau0.1_views4_800ep_seed0 | 41.94 | 86.52 | 44.58 | 36.54 | 85.42 | 48.88 | 2.94 | 131.28 | -0.9998 |
| P95_noise_tau0.3_views4_800ep_seed0 | 41.94 | 87.32 | 45.38 | 36.54 | 84.92 | 48.38 | 2.94 | 121.76 | -0.9998 |
| P95_noise_tau0.3_views4_800ep_seed1 | 41.68 | 87.12 | 45.44 | 37.40 | 84.98 | 47.58 | 3.22 | 123.20 | -0.9998 |
| P95_noise_tau0.3_views4_800ep_seed2 | 42.98 | 87.02 | 44.04 | 37.08 | 84.82 | 47.74 | 3.22 | 123.30 | -0.9998 |
| P95_refresh_R200_views4_800ep_seed0 | 41.94 | 85.94 | 44.00 | 36.54 | 84.38 | 47.84 | 2.94 | 130.04 | -0.9998 |
| P95_refresh_R200_views4_800ep_seed1 | 41.68 | 85.78 | 44.10 | 37.40 | 84.46 | 47.06 | 3.22 | 137.77 | -0.9998 |
| P95_refresh_R200_views4_800ep_seed2 | 42.98 | 86.42 | 43.44 | 37.08 | 84.46 | 47.38 | 3.22 | 132.00 | -0.9998 |
| P95_refresh_R50_views4_800ep_seed0 | 41.94 | 84.96 | 43.02 | 36.54 | 82.14 | 45.60 | 2.94 | 126.64 | -0.9998 |
| P95_residual_lam0.25_views4_800ep_seed0 | 41.98 | 86.02 | 44.04 | 36.58 | 80.90 | 44.32 | 2.94 | 5.54 | -0.5612 |
| P95_residual_lam0.25_views4_800ep_seed1 | 41.68 | 86.06 | 44.38 | 37.40 | 81.26 | 43.86 | 3.22 | 9.75 | -0.5619 |
| P95_residual_lam0.25_views4_800ep_seed2 | 42.98 | 86.32 | 43.34 | 37.08 | 80.86 | 43.78 | 3.22 | 22.20 | -0.5602 |
| P95_residual_lam0.5_views4_800ep_seed0 | 41.94 | 85.74 | 43.80 | 36.54 | 83.54 | 47.00 | 2.94 | 89.23 | -0.2484 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P95_dictionary_views4_800ep_seed0: ep0: 36.54, ep20: 47.94, ep50: 48.02, ep100: 71.98, ep200: 77.66, ep400: 82.36, ep600: 82.98, ep800: 83.20
- P95_dictionary_views4_800ep_seed1: ep0: 37.40, ep20: 46.10, ep50: 38.70, ep100: 46.94, ep200: 74.16, ep400: 80.62, ep600: 82.18, ep800: 82.64
- P95_dictionary_views4_800ep_seed2: ep0: 37.08, ep20: 48.32, ep50: 56.22, ep100: 71.64, ep200: 78.40, ep400: 82.18, ep600: 83.92, ep800: 83.68
- P95_js_matched_views4_800ep_seed0: ep0: 36.58, ep20: 68.42, ep50: 73.82, ep100: 77.58, ep200: 81.38, ep400: 83.88, ep600: 84.50, ep800: 84.60
- P95_js_matched_views4_800ep_seed1: ep0: 37.42, ep20: 67.78, ep50: 73.44, ep100: 78.58, ep200: 81.26, ep400: 83.64, ep600: 84.74, ep800: 85.06
- P95_js_matched_views4_800ep_seed2: ep0: 37.08, ep20: 67.98, ep50: 74.48, ep100: 77.98, ep200: 81.38, ep400: 83.26, ep600: 84.68, ep800: 85.00
- P95_noise_tau0.1_views4_800ep_seed0: ep0: 36.54, ep20: 68.56, ep50: 74.34, ep100: 78.76, ep200: 81.78, ep400: 84.28, ep600: 85.30, ep800: 85.42
- P95_noise_tau0.3_views4_800ep_seed0: ep0: 36.54, ep20: 68.22, ep50: 74.86, ep100: 78.60, ep200: 81.48, ep400: 83.62, ep600: 84.70, ep800: 84.92
- P95_noise_tau0.3_views4_800ep_seed1: ep0: 37.40, ep20: 67.76, ep50: 74.82, ep100: 78.60, ep200: 81.64, ep400: 83.96, ep600: 85.04, ep800: 84.98
- P95_noise_tau0.3_views4_800ep_seed2: ep0: 37.08, ep20: 67.88, ep50: 75.00, ep100: 78.78, ep200: 81.94, ep400: 83.74, ep600: 84.82, ep800: 84.82
- P95_refresh_R200_views4_800ep_seed0: ep0: 36.54, ep20: 68.38, ep50: 74.26, ep100: 78.36, ep200: 82.02, ep400: 84.24, ep600: 85.14, ep800: 84.38
- P95_refresh_R200_views4_800ep_seed1: ep0: 37.40, ep20: 68.72, ep50: 74.26, ep100: 78.54, ep200: 81.66, ep400: 84.38, ep600: 85.28, ep800: 84.46
- P95_refresh_R200_views4_800ep_seed2: ep0: 37.08, ep20: 67.94, ep50: 74.80, ep100: 79.34, ep200: 82.66, ep400: 84.40, ep600: 85.30, ep800: 84.46
- P95_refresh_R50_views4_800ep_seed0: ep0: 36.54, ep20: 68.56, ep50: 74.26, ep100: 78.30, ep200: 81.76, ep400: 81.42, ep600: 81.56, ep800: 82.14
- P95_residual_lam0.25_views4_800ep_seed0: ep0: 36.58, ep20: 66.36, ep50: 70.60, ep100: 75.28, ep200: 77.26, ep400: 80.02, ep600: 80.74, ep800: 80.90
- P95_residual_lam0.25_views4_800ep_seed1: ep0: 37.40, ep20: 66.34, ep50: 70.70, ep100: 74.86, ep200: 77.28, ep400: 80.08, ep600: 80.68, ep800: 81.26
- P95_residual_lam0.25_views4_800ep_seed2: ep0: 37.08, ep20: 65.74, ep50: 71.74, ep100: 75.08, ep200: 77.80, ep400: 79.96, ep600: 80.56, ep800: 80.86
- P95_residual_lam0.5_views4_800ep_seed0: ep0: 36.54, ep20: 49.86, ep50: 68.34, ep100: 74.82, ep200: 78.64, ep400: 82.04, ep600: 83.32, ep800: 83.54

## Held-out J trajectory (VCS only)

- P95_dictionary_views4_800ep_seed0: ep0: -0.1110, ep20: -0.0001, ep50: 0.5285, ep100: 0.8874, ep200: 0.9351, ep400: 0.9539, ep600: 0.9610, ep800: 0.9631
- P95_dictionary_views4_800ep_seed1: ep0: -0.1057, ep20: 0.2827, ep50: 0.3294, ep100: 0.0001, ep200: 0.9043, ep400: 0.9447, ep600: 0.9543, ep800: 0.9570
- P95_dictionary_views4_800ep_seed2: ep0: -0.1118, ep20: 0.0000, ep50: 0.7178, ep100: 0.8942, ep200: 0.9382, ep400: 0.9568, ep600: 0.9630, ep800: 0.9649
- P95_js_matched_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8802, ep50: 0.9354, ep100: 0.9571, ep200: 0.9708, ep400: 0.9780, ep600: 0.9813, ep800: 0.9822
- P95_js_matched_views4_800ep_seed1: ep0: -0.9998, ep20: 0.8743, ep50: 0.9395, ep100: 0.9554, ep200: 0.9708, ep400: 0.9783, ep600: 0.9813, ep800: 0.9824
- P95_js_matched_views4_800ep_seed2: ep0: -0.9998, ep20: 0.8499, ep50: 0.9317, ep100: 0.9588, ep200: 0.9695, ep400: 0.9774, ep600: 0.9812, ep800: 0.9822
- P95_noise_tau0.1_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8800, ep50: 0.9136, ep100: 0.9409, ep200: 0.9602, ep400: 0.9699, ep600: 0.9745, ep800: 0.9760
- P95_noise_tau0.3_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8773, ep50: 0.9116, ep100: 0.9480, ep200: 0.9633, ep400: 0.9742, ep600: 0.9781, ep800: 0.9794
- P95_noise_tau0.3_views4_800ep_seed1: ep0: -0.9998, ep20: 0.8659, ep50: 0.9222, ep100: 0.9424, ep200: 0.9641, ep400: 0.9731, ep600: 0.9774, ep800: 0.9785
- P95_noise_tau0.3_views4_800ep_seed2: ep0: -0.9998, ep20: 0.8561, ep50: 0.9150, ep100: 0.9476, ep200: 0.9652, ep400: 0.9750, ep600: 0.9785, ep800: 0.9795
- P95_refresh_R200_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8722, ep50: 0.9179, ep100: 0.9419, ep200: 0.9744, ep400: 0.9777, ep600: 0.9801, ep800: 0.9771
- P95_refresh_R200_views4_800ep_seed1: ep0: -0.9998, ep20: 0.8683, ep50: 0.9218, ep100: 0.9414, ep200: 0.9727, ep400: 0.9781, ep600: 0.9794, ep800: 0.9854
- P95_refresh_R200_views4_800ep_seed2: ep0: -0.9998, ep20: 0.8506, ep50: 0.9110, ep100: 0.9395, ep200: 0.9734, ep400: 0.9795, ep600: 0.9807, ep800: 0.9829
- P95_refresh_R50_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8779, ep50: 0.9469, ep100: 0.9662, ep200: 0.9754, ep400: 0.9822, ep600: 0.9861, ep800: 0.9879
- P95_residual_lam0.25_views4_800ep_seed0: ep0: -0.5612, ep20: 0.7470, ep50: 0.7929, ep100: 0.8703, ep200: 0.9065, ep400: 0.9323, ep600: 0.9437, ep800: 0.9464
- P95_residual_lam0.25_views4_800ep_seed1: ep0: -0.5619, ep20: 0.7486, ep50: 0.8053, ep100: 0.8710, ep200: 0.9027, ep400: 0.9333, ep600: 0.9435, ep800: 0.9470
- P95_residual_lam0.25_views4_800ep_seed2: ep0: -0.5602, ep20: 0.7411, ep50: 0.8062, ep100: 0.8720, ep200: 0.9094, ep400: 0.9358, ep600: 0.9453, ep800: 0.9479
- P95_residual_lam0.5_views4_800ep_seed0: ep0: -0.2484, ep20: 0.3292, ep50: 0.4282, ep100: 0.4525, ep200: 0.4690, ep400: 0.4772, ep600: 0.4812, ep800: 0.4823

## Final-epoch training objective values (epoch means)

- P95_dictionary_views4_800ep_seed0: J_raw 0.9754, R_binary 0.0246, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_dictionary_views4_800ep_seed1: J_raw 0.9695, R_binary 0.0305, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_dictionary_views4_800ep_seed2: J_raw 0.9772, R_binary 0.0228, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_js_matched_views4_800ep_seed0: J_raw 0.9915, R_binary 0.0085, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_js_matched_views4_800ep_seed1: J_raw 0.9913, R_binary 0.0087, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_js_matched_views4_800ep_seed2: J_raw 0.9914, R_binary 0.0086, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_noise_tau0.1_views4_800ep_seed0: J_raw 0.9844, R_binary 0.0156, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_noise_tau0.3_views4_800ep_seed0: J_raw 0.9777, R_binary 0.0223, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_noise_tau0.3_views4_800ep_seed1: J_raw 0.9774, R_binary 0.0226, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_noise_tau0.3_views4_800ep_seed2: J_raw 0.9775, R_binary 0.0225, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_refresh_R200_views4_800ep_seed0: J_raw 0.9884, R_binary 0.0116, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_refresh_R200_views4_800ep_seed1: J_raw 0.9936, R_binary 0.0064, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_refresh_R200_views4_800ep_seed2: J_raw 0.9921, R_binary 0.0079, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_refresh_R50_views4_800ep_seed0: J_raw 0.9934, R_binary 0.0066, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_residual_lam0.25_views4_800ep_seed0: J_raw 0.9591, R_binary 0.0409, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_residual_lam0.25_views4_800ep_seed1: J_raw 0.9597, R_binary 0.0403, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_residual_lam0.25_views4_800ep_seed2: J_raw 0.9606, R_binary 0.0394, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P95_residual_lam0.5_views4_800ep_seed0: J_raw 0.4882, R_binary 0.5118, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P95_dictionary_views4_800ep_seed0 | 0.1288 | 1988 | 3976 | 18233 | 52 | 12 | 35840000 | 805895 | 1014630@node59 |
| P95_dictionary_views4_800ep_seed1 | 0.1296 | 1976 | 3951 | 18347 | 50 | 11 | 35840000 | 805895 | 1015159@node58 |
| P95_dictionary_views4_800ep_seed2 | 0.1293 | 1979 | 3959 | 18310 | 52 | 12 | 35840000 | 805895 | 1015160@node59 |
| P95_js_matched_views4_800ep_seed0 | 0.1951 | 1312 | 2624 | 27777 | 104 | 22 | 35840000 | 2 | 1014739@nodesumo01 |
| P95_js_matched_views4_800ep_seed1 | 0.1952 | 1311 | 2623 | 27799 | 102 | 21 | 35840000 | 2 | 1015163@nodesumo01 |
| P95_js_matched_views4_800ep_seed2 | 0.1133 | 2260 | 4520 | 16066 | 51 | 14 | 35840000 | 2 | 1015164@node59 |
| P95_noise_tau0.1_views4_800ep_seed0 | 0.1156 | 2214 | 4427 | 16398 | 51 | 11 | 35840000 | 2 | 1014631@node58 |
| P95_noise_tau0.3_views4_800ep_seed0 | 0.1151 | 2224 | 4448 | 16325 | 53 | 11 | 35840000 | 2 | 1014673@node61 |
| P95_noise_tau0.3_views4_800ep_seed1 | 0.1154 | 2218 | 4436 | 16365 | 53 | 11 | 35840000 | 2 | 1015161@node59 |
| P95_noise_tau0.3_views4_800ep_seed2 | 0.1153 | 2220 | 4441 | 16349 | 53 | 12 | 35840000 | 2 | 1015162@node58 |
| P95_refresh_R200_views4_800ep_seed0 | 0.1135 | 2256 | 4512 | 16089 | 51 | 11 | 35840000 | 2 | 1014693@node58 |
| P95_refresh_R200_views4_800ep_seed1 | 0.1135 | 2255 | 4509 | 16103 | 54 | 15 | 35840000 | 2 | 1015182@node58 |
| P95_refresh_R200_views4_800ep_seed2 | 0.1138 | 2250 | 4499 | 16135 | 51 | 11 | 35840000 | 2 | 1015183@node59 |
| P95_refresh_R50_views4_800ep_seed0 | 0.1135 | 2256 | 4511 | 16091 | 51 | 11 | 35840000 | 2 | 1014692@node59 |
| P95_residual_lam0.25_views4_800ep_seed0 | 0.2027 | 1263 | 2526 | 28842 | 104 | 22 | 35840000 | 394755 | 1014621@nodesumo01 |
| P95_residual_lam0.25_views4_800ep_seed1 | 0.1208 | 2119 | 4237 | 17120 | 52 | 11 | 35840000 | 394755 | 1015157@node61 |
| P95_residual_lam0.25_views4_800ep_seed2 | 0.2576 | 994 | 1987 | 36279 | 86 | 22 | 35840000 | 394755 | 1015158@node61 |
| P95_residual_lam0.5_views4_800ep_seed0 | 0.1209 | 2117 | 4234 | 17138 | 51 | 11 | 35840000 | 394755 | 1014622@node58 |

## Provenance

- P95_dictionary_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `5171ab179cb00091b9a282f2418273e33612cbe6690a1cec7a1b84069d02bc1f`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_dictionary_views4_800ep_seed1: commit `43c87cc5d98e572f7d5e2bb377760b886c18e9de` dirty=True, config `467d2229450549c6184defcbfa04cbeea7ae1d88d3bcc57fa8b78c8abcdf2470`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P95_dictionary_views4_800ep_seed2: commit `43c87cc5d98e572f7d5e2bb377760b886c18e9de` dirty=True, config `a2150c90ef09382bb5ee67650fd63cc8173bdb65ab36ba28b12aeb4296b999be`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P95_js_matched_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `6041249972e85b176fdac1c152d7e55f5c788a4850d7d9512658b82fa859af64`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_js_matched_views4_800ep_seed1: commit `764a5bb91a1dff3ce23cc00a4f19f79f1faa75c8` dirty=True, config `a1d816fa47060cd99c1263e26115d09f1e38600f538a7a29acf7a544b31726c0`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P95_js_matched_views4_800ep_seed2: commit `764a5bb91a1dff3ce23cc00a4f19f79f1faa75c8` dirty=True, config `2a45ebec58dae33f6d9609270105e22f0b708a910c4823e7f295a9a37582d459`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P95_noise_tau0.1_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `1b4b9f605867248d53598edeef0d9f7d95105e3d044d0714d9ebcfe851096e00`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_noise_tau0.3_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `33ef4c2e8fe931b2bd3963cc1c6fde2e7150f3148e4d18faa2aee843237ee339`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_noise_tau0.3_views4_800ep_seed1: commit `82d2ef834741d5e36423f5216f7cf200f8f5908f` dirty=True, config `4433aad0da0ecea2d44f6c17aeef182ddc208cc11fd1250bcb8128006eb84f9b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P95_noise_tau0.3_views4_800ep_seed2: commit `9a1d32268a6b35edd811f2af6b3e309aa6bc9497` dirty=True, config `614d6ef1bd0b6323c5fb24470e44abe8c824ff9943b1142bf0ba86be932df0f4`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P95_refresh_R200_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `d54c9c7e7aafb904a32c20bbba409b97c4ddee92e4ed672db3ca5d7dde8da049`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_refresh_R200_views4_800ep_seed1: commit `764a5bb91a1dff3ce23cc00a4f19f79f1faa75c8` dirty=True, config `815d17be2b71770b97ae63ed04470b55ee9f1a3209082c31c08b042d4198374d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P95_refresh_R200_views4_800ep_seed2: commit `764a5bb91a1dff3ce23cc00a4f19f79f1faa75c8` dirty=True, config `fed8e61506395b3011b017fea17aeb4d52c28f451334372a6e03b915566f5bdf`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P95_refresh_R50_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `3de410c71c1899189f1c31a9653dd5fadd13c4bca877bbf77e55b7e5a4e44995`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_residual_lam0.25_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `ec408bd05a277fe31af725ba7ebe2ccef2a06c35ca557553d13361d138391fba`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P95_residual_lam0.25_views4_800ep_seed1: commit `43c87cc5d98e572f7d5e2bb377760b886c18e9de` dirty=True, config `aaf4d11659e8cf73b771a1d22fe33dcb98eeea45fd3f9c07689069933de11b1d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P95_residual_lam0.25_views4_800ep_seed2: commit `43c87cc5d98e572f7d5e2bb377760b886c18e9de` dirty=True, config `a1ee6fe79481860e4b118f777e0072a8f7014d5920853f8d14cea3ba5239a766`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P95_residual_lam0.5_views4_800ep_seed0: commit `e607a277af78d45e50daba756b967b1c3b9b6c20` dirty=True, config `5a07c9039aeeb0da22634e967facc56edbac76cc8423ecbf7b1a763c3ea9a4c3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 18 | 86.23 ± 0.69 | 42.16 ± 0.53 | 44.06 ± 0.74 | 83.68 ± 1.51 | 96.47 ± 50.76 | 0.94 ± 0.12 |

## Coverage checks

- P95_dictionary_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_dictionary_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_dictionary_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_js_matched_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_js_matched_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_js_matched_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_noise_tau0.1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_noise_tau0.3_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_noise_tau0.3_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_noise_tau0.3_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_refresh_R200_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_refresh_R200_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_refresh_R200_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_refresh_R50_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_residual_lam0.25_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_residual_lam0.25_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_residual_lam0.25_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P95_residual_lam0.5_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
