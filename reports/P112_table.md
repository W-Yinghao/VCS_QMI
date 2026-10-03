# P112_fixed_scale_sensitivity — neutral results table (observed values only)

Generated 2026-10-03T08:20:22Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P112_c100_a2_k0.25_seed0 | vcs_qmi | 8 | 0 | 800/800 | 59.84 | 55.64 | 0.8006±0.0013 | 168.27 | 16020 | 5269/8140 | COMPLETED |
| P112_c100_a2_k0.75_seed0 | vcs_qmi | 8 | 0 | 800/800 | 58.34 | 51.24 | 0.7052±0.0022 | 104.99 | 16057 | 5269/8140 | COMPLETED |
| P112_c100_a3_k0.5_seed0 | vcs_qmi | 8 | 0 | 800/800 | 59.66 | 54.24 | 0.9251±0.0022 | 130.49 | 34990 | 5269/8140 | COMPLETED |
| P112_std_a1.5_k0.5_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.06 | 86.64 | 0.7779±0.0016 | 91.29 | 16066 | 5269/8140 | COMPLETED |
| P112_std_a2_k0.25_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.72 | 86.88 | 0.8060±0.0013 | 133.43 | 16039 | 5269/8140 | COMPLETED |
| P112_std_a2_k0.75_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.64 | 86.82 | 0.7279±0.0013 | 74.60 | 16074 | 5269/8140 | COMPLETED |
| P112_std_a3_k0.5_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.48 | 86.34 | 0.9335±0.0021 | 106.59 | 16106 | 5269/8140 | COMPLETED |
| P112_strong_a2_k0.25_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.88 | 84.56 | 0.7584±0.0010 | 89.28 | 16056 | 5269/8140 | COMPLETED |
| P112_strong_a2_k0.75_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.22 | 85.74 | 0.6479±0.0012 | 49.40 | 16079 | 5269/8140 | COMPLETED |
| P112_strong_a2_k0.75_seed1 | vcs_qmi | 8 | 1 | 35/800 | null | null | null | null | null | null/null | RUNNING |
| P112_strong_a2_k0.75_seed2 | vcs_qmi | 8 | 2 | 3/800 | null | null | null | null | null | null/null | RUNNING |
| P112_strong_a3_k0.5_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.92 | 84.64 | 0.8595±0.0015 | 92.44 | 16113 | 5269/8140 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P112_c100_a2_k0.25_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_c100_a2_k0.75_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_c100_a3_k0.5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_std_a1.5_k0.5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_std_a2_k0.25_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_std_a2_k0.75_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_std_a3_k0.5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_strong_a2_k0.25_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_strong_a2_k0.75_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_strong_a2_k0.75_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_strong_a2_k0.75_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P112_strong_a3_k0.5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P112_c100_a2_k0.25_seed0 | 20.12 | 59.84 | 39.72 | 14.00 | 55.64 | 41.64 | 2.74 | 168.27 | -0.8068 |
| P112_c100_a2_k0.75_seed0 | 20.12 | 58.34 | 38.22 | 14.00 | 51.24 | 37.24 | 2.74 | 104.99 | -0.1828 |
| P112_c100_a3_k0.5_seed0 | 20.12 | 59.66 | 39.54 | 14.00 | 54.24 | 40.24 | 2.74 | 130.49 | -0.8001 |
| P112_std_a1.5_k0.5_seed0 | 41.94 | 88.06 | 46.12 | 36.54 | 86.64 | 50.10 | 2.94 | 91.29 | -0.3824 |
| P112_std_a2_k0.25_seed0 | 41.94 | 88.72 | 46.78 | 36.54 | 86.88 | 50.34 | 2.94 | 133.43 | -0.8076 |
| P112_std_a2_k0.75_seed0 | 41.94 | 88.64 | 46.70 | 36.54 | 86.82 | 50.28 | 2.94 | 74.60 | -0.1851 |
| P112_std_a3_k0.5_seed0 | 41.94 | 88.48 | 46.54 | 36.54 | 86.34 | 49.80 | 2.94 | 106.59 | -0.8015 |
| P112_strong_a2_k0.25_seed0 | 41.94 | 87.88 | 45.94 | 36.54 | 84.56 | 48.02 | 2.94 | 89.28 | -0.8051 |
| P112_strong_a2_k0.75_seed0 | 41.94 | 88.22 | 46.28 | 36.54 | 85.74 | 49.20 | 2.94 | 49.40 | -0.1827 |
| P112_strong_a2_k0.75_seed1 | null | null | null | null | null | null | null | null | null |
| P112_strong_a2_k0.75_seed2 | null | null | null | null | null | null | null | null | null |
| P112_strong_a3_k0.5_seed0 | 41.94 | 86.92 | 44.98 | 36.54 | 84.64 | 48.10 | 2.94 | 92.44 | -0.7973 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P112_c100_a2_k0.25_seed0: ep0: 14.00, ep20: 32.76, ep50: 39.64, ep100: 45.92, ep200: 51.14, ep400: 54.18, ep600: 55.26, ep800: 55.64
- P112_c100_a2_k0.75_seed0: ep0: 14.00, ep20: 23.56, ep50: 31.18, ep100: 38.68, ep200: 45.92, ep400: 49.94, ep600: 50.90, ep800: 51.24
- P112_c100_a3_k0.5_seed0: ep0: 14.00, ep20: 31.16, ep50: 37.54, ep100: 43.98, ep200: 48.76, ep400: 53.40, ep600: 53.94, ep800: 54.24
- P112_std_a1.5_k0.5_seed0: ep0: 36.54, ep20: 67.10, ep50: 75.96, ep100: 81.16, ep200: 83.72, ep400: 85.80, ep600: 86.38, ep800: 86.64
- P112_std_a2_k0.25_seed0: ep0: 36.54, ep20: 69.34, ep50: 75.82, ep100: 80.02, ep200: 83.30, ep400: 85.48, ep600: 86.62, ep800: 86.88
- P112_std_a2_k0.75_seed0: ep0: 36.54, ep20: 65.68, ep50: 76.36, ep100: 81.52, ep200: 84.50, ep400: 86.74, ep600: 86.56, ep800: 86.82
- P112_std_a3_k0.5_seed0: ep0: 36.54, ep20: 68.42, ep50: 74.22, ep100: 78.72, ep200: 82.84, ep400: 85.74, ep600: 86.44, ep800: 86.34
- P112_strong_a2_k0.25_seed0: ep0: 36.54, ep20: 67.80, ep50: 73.24, ep100: 77.56, ep200: 81.14, ep400: 83.42, ep600: 84.48, ep800: 84.56
- P112_strong_a2_k0.75_seed0: ep0: 36.54, ep20: 61.48, ep50: 71.08, ep100: 77.22, ep200: 81.92, ep400: 84.70, ep600: 85.34, ep800: 85.74
- P112_strong_a2_k0.75_seed1: ep0: 37.40, ep20: 62.98
- P112_strong_a2_k0.75_seed2: ep0: 37.08
- P112_strong_a3_k0.5_seed0: ep0: 36.54, ep20: 67.56, ep50: 73.76, ep100: 77.52, ep200: 80.52, ep400: 83.56, ep600: 84.22, ep800: 84.64

## Held-out J trajectory (VCS only)

- P112_c100_a2_k0.25_seed0: ep0: -0.8068, ep20: 0.7396, ep50: 0.7553, ep100: 0.7838, ep200: 0.7947, ep400: 0.7994, ep600: 0.8004, ep800: 0.8006
- P112_c100_a2_k0.75_seed0: ep0: -0.1828, ep20: 0.6268, ep50: 0.6562, ep100: 0.6864, ep200: 0.7039, ep400: 0.7074, ep600: 0.7056, ep800: 0.7052
- P112_c100_a3_k0.5_seed0: ep0: -0.8001, ep20: 0.8341, ep50: 0.8747, ep100: 0.9047, ep200: 0.9153, ep400: 0.9223, ep600: 0.9245, ep800: 0.9251
- P112_std_a1.5_k0.5_seed0: ep0: -0.3824, ep20: 0.7038, ep50: 0.7407, ep100: 0.7644, ep200: 0.7727, ep400: 0.7778, ep600: 0.7778, ep800: 0.7779
- P112_std_a2_k0.25_seed0: ep0: -0.8076, ep20: 0.7371, ep50: 0.7688, ep100: 0.7873, ep200: 0.7981, ep400: 0.8045, ep600: 0.8057, ep800: 0.8060
- P112_std_a2_k0.75_seed0: ep0: -0.1851, ep20: 0.6437, ep50: 0.6804, ep100: 0.7099, ep200: 0.7228, ep400: 0.7290, ep600: 0.7289, ep800: 0.7279
- P112_std_a3_k0.5_seed0: ep0: -0.8015, ep20: 0.8490, ep50: 0.8826, ep100: 0.9079, ep200: 0.9214, ep400: 0.9300, ep600: 0.9326, ep800: 0.9335
- P112_strong_a2_k0.25_seed0: ep0: -0.8051, ep20: 0.6342, ep50: 0.6807, ep100: 0.7106, ep200: 0.7289, ep400: 0.7470, ep600: 0.7562, ep800: 0.7584
- P112_strong_a2_k0.75_seed0: ep0: -0.1827, ep20: 0.4868, ep50: 0.5500, ep100: 0.5853, ep200: 0.6119, ep400: 0.6321, ep600: 0.6438, ep800: 0.6479
- P112_strong_a2_k0.75_seed1: ep0: -0.1740, ep20: 0.5079
- P112_strong_a2_k0.75_seed2: ep0: -0.1699
- P112_strong_a3_k0.5_seed0: ep0: -0.7973, ep20: 0.6714, ep50: 0.7569, ep100: 0.7965, ep200: 0.8208, ep400: 0.8432, ep600: 0.8562, ep800: 0.8595

## Final-epoch training objective values (epoch means)

- P112_c100_a2_k0.25_seed0: J_raw 0.8253, R_binary 0.1747, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_c100_a2_k0.75_seed0: J_raw 0.7764, R_binary 0.2236, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_c100_a3_k0.5_seed0: J_raw 0.9523, R_binary 0.0477, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_std_a1.5_k0.5_seed0: J_raw 0.8166, R_binary 0.1834, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_std_a2_k0.25_seed0: J_raw 0.8261, R_binary 0.1739, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_std_a2_k0.75_seed0: J_raw 0.7807, R_binary 0.2193, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_std_a3_k0.5_seed0: J_raw 0.9552, R_binary 0.0448, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_strong_a2_k0.25_seed0: J_raw 0.7734, R_binary 0.2266, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_strong_a2_k0.75_seed0: J_raw 0.6754, R_binary 0.3246, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_strong_a2_k0.75_seed1: J_raw 0.5418, R_binary 0.4582, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_strong_a2_k0.75_seed2: J_raw 0.1828, R_binary 0.8172, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P112_strong_a3_k0.5_seed0: J_raw 0.8771, R_binary 0.1229, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P112_c100_a2_k0.25_seed0 | 0.1129 | 2267 | 4533 | 16020 | 53 | 11 | 35840000 | None | 1018310@node61 |
| P112_c100_a2_k0.75_seed0 | 0.1132 | 2262 | 4524 | 16057 | 51 | 12 | 35840000 | None | 1018311@node58 |
| P112_c100_a3_k0.5_seed0 | 0.2485 | 1030 | 2060 | 34990 | 83 | 16 | 35840000 | None | 1018312@node60 |
| P112_std_a1.5_k0.5_seed0 | 0.1133 | 2260 | 4519 | 16066 | 50 | 12 | 35840000 | None | 1018303@node59 |
| P112_std_a2_k0.25_seed0 | 0.1131 | 2264 | 4528 | 16039 | 52 | 11 | 35840000 | None | 1018305@node61 |
| P112_std_a2_k0.75_seed0 | 0.1134 | 2258 | 4517 | 16074 | 51 | 11 | 35840000 | None | 1018306@node58 |
| P112_std_a3_k0.5_seed0 | 0.1135 | 2256 | 4512 | 16106 | 58 | 12 | 35840000 | None | 1018304@node59 |
| P112_strong_a2_k0.25_seed0 | 0.1132 | 2262 | 4524 | 16056 | 55 | 11 | 35840000 | None | 1018307@node58 |
| P112_strong_a2_k0.75_seed0 | 0.1134 | 2258 | 4515 | 16079 | 50 | 11 | 35840000 | None | 1018308@node59 |
| P112_strong_a2_k0.75_seed1 | null | null | null | null | null | null | None | None | 1019690@node58 |
| P112_strong_a2_k0.75_seed2 | null | null | null | null | null | null | None | None | 1019691@node60 |
| P112_strong_a3_k0.5_seed0 | 0.1135 | 2255 | 4509 | 16113 | 57 | 13 | 35840000 | None | 1018309@node59 |

## Provenance

- P112_c100_a2_k0.25_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `91b6d0e2af8cf70107dd1ef42e3d3ce0848d86f9d16350165b6cca66550ce751`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_c100_a2_k0.75_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `115852310f79fc37dfd4f8a03d8621e2b0f6cd8f580a1772d000ac66e291bb30`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_c100_a3_k0.5_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `4e2f7c0efb5d853a3dbb06230579cfb9d01704464ac0dc1c9c80a6db69d58aa9`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_std_a1.5_k0.5_seed0: commit `4bf73bada7ac6714140c38b1e0c63d25d644270c` dirty=True, config `3bba301523a78c81fde4497073960c53c698776bd44bcd21398f5f0fb344ec44`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_std_a2_k0.25_seed0: commit `c339c7bdce4afad4929aa3f2b4a53633ceeeba0d` dirty=True, config `7644240b5f7820fb8fab26097da9ce637123b0079b8cfc51581753bbf837c17e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_std_a2_k0.75_seed0: commit `c339c7bdce4afad4929aa3f2b4a53633ceeeba0d` dirty=True, config `4f0c96b9bf21e04135fd542b6779e2ef2649c4252e0d68dfbae8b6a5a8b661bb`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_std_a3_k0.5_seed0: commit `c339c7bdce4afad4929aa3f2b4a53633ceeeba0d` dirty=True, config `1df127df107058865cb032f1b01a9b661b8a327de964612c614572a0a452052a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_strong_a2_k0.25_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `acc51b327fc7a642d297b9aac38636c11c86c54ec87a7380ce6f70ad8b984d25`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_strong_a2_k0.75_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `ce918a6a2dc5ac7b05f0c94710aa7c8af416aaa8d08e48bcf381b8c32a9abb05`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P112_strong_a2_k0.75_seed1: commit `93dcad5b368045b0015375ce32b304a8a699edcc` dirty=True, config `71f5d5666047a6ecd3d40c4ee3222d8929eb8ca68765ce97e3c74cbfc6a30d2b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P112_strong_a2_k0.75_seed2: commit `bfbf2ae0b61c0f1e6020f40faefd0bed1be90b0a` dirty=True, config `f46bd65a42a3950307b8d0156db0a363f148d48ee90568a5868b470513c3f00e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P112_strong_a3_k0.5_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `7657fd1629b219f5d20ae049cd0e5250bf86a0472cebfc7bb00f9c1000a28fbb`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: False
- identical split across runs: False

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 10 | 79.48 ± 13.95 | 35.39 ± 10.54 | 44.08 ± 3.46 | 76.27 ± 15.63 | 104.08 ± 33.44 | 0.79 ± 0.09 |

## Coverage checks

- P112_c100_a2_k0.25_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_c100_a2_k0.75_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_c100_a3_k0.5_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_std_a1.5_k0.5_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_std_a2_k0.25_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_std_a2_k0.75_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_std_a3_k0.5_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_strong_a2_k0.25_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_strong_a2_k0.75_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P112_strong_a2_k0.75_seed1: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P112_strong_a2_k0.75_seed2: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P112_strong_a3_k0.5_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
