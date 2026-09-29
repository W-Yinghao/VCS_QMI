# P87_s_cs_dev — neutral results table (observed values only)

Generated 2026-09-29T08:36:36Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_1x_seed0 | cs_kernel_native | null | 0 | 100/100 | 67.58 | 58.76 | null | 57.77 | 5920 | 3682/4840 | COMPLETED |
| P87_kcs_bw1_1x_seed0 | cs_kernel_native | null | 0 | 100/100 | 62.84 | 49.68 | null | 1.90 | 5968 | 3682/4840 | COMPLETED COLLAPSE_SUSPECTED |
| P87_kcs_bw2_1x_seed0 | cs_kernel_native | null | 0 | 100/100 | 36.54 | 32.48 | null | 1.47 | 5969 | 3682/4840 | COMPLETED COLLAPSE_SUSPECTED |
| P87_skernel_m1024_bw0.5_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 78.06 | 70.86 | 0.8722±0.0027 | 8.67 | 6007 | 3683/4856 | COMPLETED |
| P87_skernel_m1024_bw1_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 80.56 | 70.16 | 0.8291±0.0038 | 16.14 | 6765 | 3683/4856 | COMPLETED |
| P87_skernel_m1024_bw2_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 76.62 | 64.14 | 0.6565±0.0051 | 6.44 | 5968 | 3683/4856 | COMPLETED |
| P87_skernel_m256_bw0.5_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 77.46 | 70.42 | 0.8835±0.0021 | 2.31 | 5978 | 3682/4854 | COMPLETED |
| P87_skernel_m256_bw1_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 79.94 | 69.98 | 0.8299±0.0038 | 22.91 | 6689 | 3682/4854 | COMPLETED |
| P87_skernel_m256_bw2_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 61.40 | 47.74 | 0.4059±0.0026 | 2.79 | 5985 | 3682/4854 | COMPLETED |
| P87_skernel_m4096_bw0.5_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 80.76 | 71.76 | 0.8807±0.0033 | 19.44 | 6032 | 3687/4862 | COMPLETED |
| P87_skernel_m4096_bw1_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 79.06 | 67.80 | 0.7220±0.0066 | 7.93 | 6012 | 3687/4862 | COMPLETED |
| P87_skernel_m4096_bw2_1x_seed0 | vcs_qmi | 8 | 0 | 100/100 | 78.18 | 67.24 | 0.6677±0.0053 | 7.01 | 5978 | 3687/4862 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_1x_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P87_kcs_bw1_1x_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P87_kcs_bw2_1x_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P87_skernel_m1024_bw0.5_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m1024_bw1_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m1024_bw2_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m256_bw0.5_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m256_bw1_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m256_bw2_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m4096_bw0.5_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m4096_bw1_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |
| P87_skernel_m4096_bw2_1x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 128 | 0.001 | 100 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_1x_seed0 | 41.78 | 67.58 | 25.80 | 36.58 | 58.76 | 22.18 | 2.94 | 57.77 | null |
| P87_kcs_bw1_1x_seed0 | 41.78 | 62.84 | 21.06 | 36.58 | 49.68 | 13.10 | 2.94 | 1.90 | null |
| P87_kcs_bw2_1x_seed0 | 41.78 | 36.54 | -5.24 | 36.58 | 32.48 | -4.10 | 2.94 | 1.47 | null |
| P87_skernel_m1024_bw0.5_1x_seed0 | 41.78 | 78.06 | 36.28 | 36.58 | 70.86 | 34.28 | 2.94 | 8.67 | -0.0000 |
| P87_skernel_m1024_bw1_1x_seed0 | 41.78 | 80.56 | 38.78 | 36.58 | 70.16 | 33.58 | 2.94 | 16.14 | -0.0000 |
| P87_skernel_m1024_bw2_1x_seed0 | 41.78 | 76.62 | 34.84 | 36.58 | 64.14 | 27.56 | 2.94 | 6.44 | -0.0000 |
| P87_skernel_m256_bw0.5_1x_seed0 | 41.78 | 77.46 | 35.68 | 36.58 | 70.42 | 33.84 | 2.94 | 2.31 | -0.0000 |
| P87_skernel_m256_bw1_1x_seed0 | 41.78 | 79.94 | 38.16 | 36.58 | 69.98 | 33.40 | 2.94 | 22.91 | -0.0000 |
| P87_skernel_m256_bw2_1x_seed0 | 41.78 | 61.40 | 19.62 | 36.58 | 47.74 | 11.16 | 2.94 | 2.79 | -0.0000 |
| P87_skernel_m4096_bw0.5_1x_seed0 | 41.78 | 80.76 | 38.98 | 36.58 | 71.76 | 35.18 | 2.94 | 19.44 | -0.0000 |
| P87_skernel_m4096_bw1_1x_seed0 | 41.78 | 79.06 | 37.28 | 36.58 | 67.80 | 31.22 | 2.94 | 7.93 | -0.0000 |
| P87_skernel_m4096_bw2_1x_seed0 | 41.78 | 78.18 | 36.40 | 36.58 | 67.24 | 30.66 | 2.94 | 7.01 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P87_kcs_bw0.5_1x_seed0: ep0: 36.58, ep10: 40.66, ep20: 43.94, ep50: 56.48, ep100: 58.76
- P87_kcs_bw1_1x_seed0: ep0: 36.58, ep10: 40.24, ep20: 40.84, ep50: 48.04, ep100: 49.68
- P87_kcs_bw2_1x_seed0: ep0: 36.58, ep10: 30.64, ep20: 30.10, ep50: 29.56, ep100: 32.48
- P87_skernel_m1024_bw0.5_1x_seed0: ep0: 36.58, ep10: 46.40, ep20: 55.18, ep50: 66.70, ep100: 70.86
- P87_skernel_m1024_bw1_1x_seed0: ep0: 36.58, ep10: 52.38, ep20: 60.96, ep50: 67.66, ep100: 70.16
- P87_skernel_m1024_bw2_1x_seed0: ep0: 36.58, ep10: 43.78, ep20: 51.10, ep50: 61.08, ep100: 64.14
- P87_skernel_m256_bw0.5_1x_seed0: ep0: 36.58, ep10: 46.22, ep20: 58.28, ep50: 67.64, ep100: 70.42
- P87_skernel_m256_bw1_1x_seed0: ep0: 36.58, ep10: 50.56, ep20: 59.70, ep50: 67.54, ep100: 69.98
- P87_skernel_m256_bw2_1x_seed0: ep0: 36.58, ep10: 24.70, ep20: 26.34, ep50: 28.34, ep100: 47.74
- P87_skernel_m4096_bw0.5_1x_seed0: ep0: 36.58, ep10: 48.12, ep20: 62.18, ep50: 68.50, ep100: 71.76
- P87_skernel_m4096_bw1_1x_seed0: ep0: 36.58, ep10: 47.10, ep20: 56.24, ep50: 64.92, ep100: 67.80
- P87_skernel_m4096_bw2_1x_seed0: ep0: 36.58, ep10: 47.40, ep20: 56.10, ep50: 63.10, ep100: 67.24

## Held-out J trajectory (VCS only)

- P87_skernel_m1024_bw0.5_1x_seed0: ep0: -0.0000, ep10: 0.6145, ep20: 0.6222, ep50: 0.8259, ep100: 0.8722
- P87_skernel_m1024_bw1_1x_seed0: ep0: -0.0000, ep10: 0.5735, ep20: 0.6972, ep50: 0.7920, ep100: 0.8291
- P87_skernel_m1024_bw2_1x_seed0: ep0: -0.0000, ep10: 0.3547, ep20: 0.4080, ep50: 0.6116, ep100: 0.6565
- P87_skernel_m256_bw0.5_1x_seed0: ep0: -0.0000, ep10: 0.6292, ep20: 0.7484, ep50: 0.8356, ep100: 0.8835
- P87_skernel_m256_bw1_1x_seed0: ep0: -0.0000, ep10: 0.5752, ep20: 0.7015, ep50: 0.7922, ep100: 0.8299
- P87_skernel_m256_bw2_1x_seed0: ep0: -0.0000, ep10: -0.0000, ep20: -0.0000, ep50: -0.0000, ep100: 0.4059
- P87_skernel_m4096_bw0.5_1x_seed0: ep0: -0.0000, ep10: 0.6482, ep20: 0.7810, ep50: 0.8524, ep100: 0.8807
- P87_skernel_m4096_bw1_1x_seed0: ep0: -0.0000, ep10: 0.4779, ep20: 0.6139, ep50: 0.6849, ep100: 0.7220
- P87_skernel_m4096_bw2_1x_seed0: ep0: -0.0000, ep10: 0.3806, ep20: 0.5424, ep50: 0.6298, ep100: 0.6677

## Final-epoch training objective values (epoch means)

- P87_kcs_bw0.5_1x_seed0: J_raw null, R_binary null, nt_xent null, vicreg None
- P87_kcs_bw1_1x_seed0: J_raw null, R_binary null, nt_xent null, vicreg None
- P87_kcs_bw2_1x_seed0: J_raw null, R_binary null, nt_xent null, vicreg None
- P87_skernel_m1024_bw0.5_1x_seed0: J_raw 0.8813, R_binary 0.1187, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m1024_bw1_1x_seed0: J_raw 0.8379, R_binary 0.1621, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m1024_bw2_1x_seed0: J_raw 0.6645, R_binary 0.3355, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m256_bw0.5_1x_seed0: J_raw 0.8915, R_binary 0.1085, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m256_bw1_1x_seed0: J_raw 0.8407, R_binary 0.1593, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m256_bw2_1x_seed0: J_raw 0.4088, R_binary 0.5912, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m4096_bw0.5_1x_seed0: J_raw 0.8887, R_binary 0.1113, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m4096_bw1_1x_seed0: J_raw 0.7316, R_binary 0.2684, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m4096_bw2_1x_seed0: J_raw 0.6764, R_binary 0.3236, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_1x_seed0 | 0.1677 | 763 | 1526 | 5920 | 35 | 17 | 4492800 | None | 1013432@nodeaudible01 |
| P87_kcs_bw1_1x_seed0 | 0.1689 | 758 | 1516 | 5968 | 35 | 18 | 4492800 | None | 1013477@nodeaudible01 |
| P87_kcs_bw2_1x_seed0 | 0.1690 | 757 | 1515 | 5969 | 35 | 17 | 4492800 | None | 1013544@nodeaudible01 |
| P87_skernel_m1024_bw0.5_1x_seed0 | 0.1700 | 753 | 1506 | 6007 | 96 | 30 | 4492800 | 1025 | 1013416@nodeaudible01 |
| P87_skernel_m1024_bw1_1x_seed0 | 0.1916 | 668 | 1336 | 6765 | 108 | 36 | 4492800 | 1025 | 1013417@node03 |
| P87_skernel_m1024_bw2_1x_seed0 | 0.1690 | 757 | 1514 | 5968 | 91 | 28 | 4492800 | 1025 | 1013418@nodeaudible01 |
| P87_skernel_m256_bw0.5_1x_seed0 | 0.1693 | 756 | 1513 | 5978 | 95 | 29 | 4492800 | 257 | 1013413@nodeaudible01 |
| P87_skernel_m256_bw1_1x_seed0 | 0.1893 | 676 | 1352 | 6689 | 111 | 37 | 4492800 | 257 | 1013414@node01 |
| P87_skernel_m256_bw2_1x_seed0 | 0.1693 | 756 | 1512 | 5985 | 95 | 30 | 4492800 | 257 | 1013415@nodeaudible01 |
| P87_skernel_m4096_bw0.5_1x_seed0 | 0.1709 | 749 | 1498 | 6032 | 96 | 29 | 4492800 | 4097 | 1013419@nodeaudible01 |
| P87_skernel_m4096_bw1_1x_seed0 | 0.1703 | 752 | 1504 | 6012 | 97 | 30 | 4492800 | 4097 | 1013424@nodeaudible01 |
| P87_skernel_m4096_bw2_1x_seed0 | 0.1694 | 756 | 1511 | 5978 | 92 | 29 | 4492800 | 4097 | 1013430@nodeaudible01 |

## Provenance

- P87_kcs_bw0.5_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `a02c454a2efd3e092cd0bcb71fda66803507db7f035d0308238e8ef283bccda8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_kcs_bw1_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `02c95ea425dbc2bd17cd625018262560fc7a9de521e6d7029683d4ae41dc8f3b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_kcs_bw2_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `e8ceeeed76a3ede5a1ad138cad1ed6702f2fc404d4decff7e45a7d4fe65a846c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m1024_bw0.5_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `2a67829b0686d66f0165bc820297c8c6de0c0c113b90806ad5c048db672601f3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m1024_bw1_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `704ddbd857f06542951920272bf75c706a87c61ad7d401d8ea756e6c237fa58d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m1024_bw2_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `e392527a258dcc3c7e9b161afddf49737247d8d6fcac59259e2a033665df27c2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m256_bw0.5_1x_seed0: commit `b5673a29cd25879f4ffcca23bf326e28936c3c35` dirty=True, config `041cc7f262ec074f99eee43d71137db039364dc846343c1e9d1921a2d42a04f6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m256_bw1_1x_seed0: commit `b5673a29cd25879f4ffcca23bf326e28936c3c35` dirty=True, config `777c5889153ed5c3c2169546faa192699e3368d7e07030fc56c9beba580ebd0b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m256_bw2_1x_seed0: commit `b5673a29cd25879f4ffcca23bf326e28936c3c35` dirty=True, config `12dd4bd2e8dad6c0f0dab0b03a55823e56ff77e7f56e8cc55f826f06a1993297`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m4096_bw0.5_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `e79ca37b2062308067e503a936163d78c3e1d99040374763fe8f685e80b8dbdc`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m4096_bw1_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `dce150de3d281cb71cb4c8e3c11562d0cf745b4eafac88deb9adbe3a52f39dde`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m4096_bw2_1x_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `d3a0f73b75643e004efb00f043a8e00531458098ddfa72fb602c3d909f6800c6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| cs_kernel_native | 3 | 55.65 ± 16.72 | 41.78 ± 0.00 | 13.87 ± 16.72 | 46.97 ± 13.35 | 20.38 ± 32.38 | null |
| vcs_qmi K=8 | 9 | 76.89 ± 5.98 | 41.78 ± 0.00 | 35.11 ± 5.98 | 66.68 ± 7.48 | 10.40 ± 7.34 | 0.75 ± 0.16 |

## Coverage checks

- P87_kcs_bw0.5_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_kcs_bw1_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_kcs_bw2_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m1024_bw0.5_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m1024_bw1_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m1024_bw2_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m256_bw0.5_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m256_bw1_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m256_bw2_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m4096_bw0.5_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m4096_bw1_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P87_skernel_m4096_bw2_1x_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
