# P61_batchsweep — neutral results table (observed values only)

Generated 2026-09-27T20:16:00Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P61_simclr_B128_seed0 | simclr_matched | null | 0 | 200/200 | 85.24 | 82.98 | null | 74.64 | 1893 | 1548/2222 | COMPLETED |
| P61_simclr_B256_seed0 | simclr_matched | null | 0 | 200/200 | 86.32 | 83.80 | null | 92.15 | 4505 | 2788/3518 | COMPLETED |
| P61_simclr_B32_lr2x_seed0 | simclr_matched | null | 0 | 200/200 | 84.48 | 80.70 | null | 47.58 | 6361 | 1383/2224 | COMPLETED |
| P61_simclr_B32_seed0 | simclr_matched | null | 0 | 200/200 | 84.64 | 79.64 | null | 41.17 | 2652 | 1383/2224 | COMPLETED |
| P61_simclr_B64_seed0 | simclr_matched | null | 0 | 200/200 | 85.12 | 80.82 | null | 56.52 | 2059 | 1385/2222 | COMPLETED |
| P61_vcs_B128_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.06 | 76.78 | 0.9400±0.0013 | 40.35 | 3421 | 2215/3648 | COMPLETED |
| P61_vcs_B256_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.82 | 76.84 | 0.9408±0.0015 | 44.77 | 1998 | 2788/3518 | COMPLETED |
| P61_vcs_B32_lr2x_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.26 | 74.36 | 0.9307±0.0019 | 28.79 | 2809 | 1383/2224 | COMPLETED |
| P61_vcs_B32_seed0 | vcs_qmi | 8 | 0 | 200/200 | 79.42 | 74.60 | 0.9308±0.0014 | 29.62 | 5139 | 1537/2486 | COMPLETED |
| P61_vcs_B64_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.96 | 74.96 | 0.9350±0.0015 | 35.11 | 3629 | 1538/2486 | COMPLETED |
| P61_vicreg_B128_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.14 | 82.60 | null | 67.40 | 1942 | 1548/2224 | COMPLETED |
| P61_vicreg_B256_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 85.12 | 81.86 | null | 75.47 | 2002 | 2788/3516 | COMPLETED |
| P61_vicreg_B32_lr2x_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 85.68 | 81.70 | null | 40.45 | 2788 | 1383/2224 | COMPLETED |
| P61_vicreg_B32_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 85.06 | 80.86 | null | 36.72 | 2791 | 1383/2224 | COMPLETED |
| P61_vicreg_B64_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 85.90 | 81.86 | null | 52.10 | 2151 | 1385/2224 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P61_simclr_B128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.0005 | 200 |
| P61_simclr_B256_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P61_simclr_B32_lr2x_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 32 | 0.00025 | 200 |
| P61_simclr_B32_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 32 | 0.000125 | 200 |
| P61_simclr_B64_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 64 | 0.00025 | 200 |
| P61_vcs_B128_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.0005 | 200 |
| P61_vcs_B256_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P61_vcs_B32_lr2x_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 32 | 0.00025 | 200 |
| P61_vcs_B32_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 32 | 0.000125 | 200 |
| P61_vcs_B64_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 64 | 0.00025 | 200 |
| P61_vicreg_B128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.0005 | 200 |
| P61_vicreg_B256_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P61_vicreg_B32_lr2x_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 32 | 0.00025 | 200 |
| P61_vicreg_B32_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 32 | 0.000125 | 200 |
| P61_vicreg_B64_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 64 | 0.00025 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P61_simclr_B128_seed0 | 41.94 | 85.24 | 43.30 | 36.54 | 82.98 | 46.44 | 2.94 | 74.64 | null |
| P61_simclr_B256_seed0 | 41.94 | 86.32 | 44.38 | 36.54 | 83.80 | 47.26 | 2.94 | 92.15 | null |
| P61_simclr_B32_lr2x_seed0 | 41.94 | 84.48 | 42.54 | 36.54 | 80.70 | 44.16 | 2.94 | 47.58 | null |
| P61_simclr_B32_seed0 | 41.94 | 84.64 | 42.70 | 36.54 | 79.64 | 43.10 | 2.94 | 41.17 | null |
| P61_simclr_B64_seed0 | 41.94 | 85.12 | 43.18 | 36.54 | 80.82 | 44.28 | 2.94 | 56.52 | null |
| P61_vcs_B128_seed0 | 41.98 | 81.06 | 39.08 | 36.58 | 76.78 | 40.20 | 2.94 | 40.35 | -0.9998 |
| P61_vcs_B256_seed0 | 41.94 | 81.82 | 39.88 | 36.54 | 76.84 | 40.30 | 2.94 | 44.77 | -0.9998 |
| P61_vcs_B32_lr2x_seed0 | 41.94 | 80.26 | 38.32 | 36.54 | 74.36 | 37.82 | 2.94 | 28.79 | -0.9998 |
| P61_vcs_B32_seed0 | 41.98 | 79.42 | 37.44 | 36.58 | 74.60 | 38.02 | 2.94 | 29.62 | -0.9998 |
| P61_vcs_B64_seed0 | 41.98 | 80.96 | 38.98 | 36.58 | 74.96 | 38.38 | 2.94 | 35.11 | -0.9998 |
| P61_vicreg_B128_seed0 | 41.94 | 86.14 | 44.20 | 36.54 | 82.60 | 46.06 | 2.94 | 67.40 | null |
| P61_vicreg_B256_seed0 | 41.94 | 85.12 | 43.18 | 36.54 | 81.86 | 45.32 | 2.94 | 75.47 | null |
| P61_vicreg_B32_lr2x_seed0 | 41.94 | 85.68 | 43.74 | 36.54 | 81.70 | 45.16 | 2.94 | 40.45 | null |
| P61_vicreg_B32_seed0 | 41.94 | 85.06 | 43.12 | 36.54 | 80.86 | 44.32 | 2.94 | 36.72 | null |
| P61_vicreg_B64_seed0 | 41.94 | 85.90 | 43.96 | 36.54 | 81.86 | 45.32 | 2.94 | 52.10 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P61_simclr_B128_seed0: ep0: 36.54, ep10: 61.32, ep20: 69.02, ep50: 76.86, ep100: 81.16, ep150: 82.44, ep200: 82.98
- P61_simclr_B256_seed0: ep0: 36.54, ep10: 62.14, ep20: 69.16, ep50: 76.40, ep100: 81.26, ep150: 83.60, ep200: 83.80
- P61_simclr_B32_lr2x_seed0: ep0: 36.54, ep10: 61.54, ep20: 69.14, ep50: 74.36, ep100: 77.96, ep150: 80.76, ep200: 80.70
- P61_simclr_B32_seed0: ep0: 36.54, ep10: 58.54, ep20: 67.08, ep50: 73.24, ep100: 77.28, ep150: 79.34, ep200: 79.64
- P61_simclr_B64_seed0: ep0: 36.54, ep10: 60.48, ep20: 68.60, ep50: 75.02, ep100: 78.62, ep150: 80.66, ep200: 80.82
- P61_vcs_B128_seed0: ep0: 36.58, ep10: 50.90, ep20: 62.28, ep50: 69.60, ep100: 74.24, ep150: 75.94, ep200: 76.78
- P61_vcs_B256_seed0: ep0: 36.54, ep10: 50.26, ep20: 61.54, ep50: 69.18, ep100: 73.94, ep150: 76.28, ep200: 76.84
- P61_vcs_B32_lr2x_seed0: ep0: 36.54, ep10: 55.74, ep20: 63.46, ep50: 69.48, ep100: 72.52, ep150: 74.00, ep200: 74.36
- P61_vcs_B32_seed0: ep0: 36.58, ep10: 51.78, ep20: 61.58, ep50: 68.72, ep100: 72.44, ep150: 73.98, ep200: 74.60
- P61_vcs_B64_seed0: ep0: 36.58, ep10: 51.26, ep20: 62.46, ep50: 69.82, ep100: 72.82, ep150: 74.70, ep200: 74.96
- P61_vicreg_B128_seed0: ep0: 36.54, ep10: 59.06, ep20: 69.26, ep50: 76.32, ep100: 81.14, ep150: 82.20, ep200: 82.60
- P61_vicreg_B256_seed0: ep0: 36.54, ep10: 59.42, ep20: 67.70, ep50: 76.08, ep100: 79.64, ep150: 81.66, ep200: 81.86
- P61_vicreg_B32_lr2x_seed0: ep0: 36.54, ep10: 59.90, ep20: 68.94, ep50: 76.00, ep100: 80.30, ep150: 81.56, ep200: 81.70
- P61_vicreg_B32_seed0: ep0: 36.54, ep10: 54.88, ep20: 66.28, ep50: 74.14, ep100: 78.86, ep150: 80.54, ep200: 80.86
- P61_vicreg_B64_seed0: ep0: 36.54, ep10: 57.56, ep20: 68.76, ep50: 75.80, ep100: 80.06, ep150: 81.82, ep200: 81.86

## Held-out J trajectory (VCS only)

- P61_vcs_B128_seed0: ep0: -0.9998, ep10: 0.6314, ep20: 0.8103, ep50: 0.8696, ep100: 0.9147, ep150: 0.9336, ep200: 0.9400
- P61_vcs_B256_seed0: ep0: -0.9998, ep10: 0.6354, ep20: 0.7986, ep50: 0.8638, ep100: 0.9127, ep150: 0.9338, ep200: 0.9408
- P61_vcs_B32_lr2x_seed0: ep0: -0.9998, ep10: 0.7122, ep20: 0.8123, ep50: 0.8792, ep100: 0.9085, ep150: 0.9226, ep200: 0.9307
- P61_vcs_B32_seed0: ep0: -0.9998, ep10: 0.6554, ep20: 0.8090, ep50: 0.8778, ep100: 0.9075, ep150: 0.9229, ep200: 0.9308
- P61_vcs_B64_seed0: ep0: -0.9998, ep10: 0.6540, ep20: 0.8148, ep50: 0.8796, ep100: 0.9145, ep150: 0.9293, ep200: 0.9350

## Final-epoch training objective values (epoch means)

- P61_simclr_B128_seed0: J_raw null, R_binary null, nt_xent 1.5801, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_simclr_B256_seed0: J_raw null, R_binary null, nt_xent 2.1374, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_simclr_B32_lr2x_seed0: J_raw null, R_binary null, nt_xent 0.6904, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_simclr_B32_seed0: J_raw null, R_binary null, nt_xent 0.7127, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_simclr_B64_seed0: J_raw null, R_binary null, nt_xent 1.1024, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_vcs_B128_seed0: J_raw 0.9444, R_binary 0.0556, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_vcs_B256_seed0: J_raw 0.9453, R_binary 0.0547, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_vcs_B32_lr2x_seed0: J_raw 0.9393, R_binary 0.0607, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_vcs_B32_seed0: J_raw 0.9391, R_binary 0.0609, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_vcs_B64_seed0: J_raw 0.9408, R_binary 0.0592, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P61_vicreg_B128_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.1596808231666557, 'vicreg_variance': 0.04011451491709279, 'vicreg_covariance': 2.8400133708942987}
- P61_vicreg_B256_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.15383938725505555, 'vicreg_variance': 0.01845858662522265, 'vicreg_covariance': 2.01395001411438}
- P61_vicreg_B32_lr2x_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.10012630631042324, 'vicreg_variance': 0.2531695456202983, 'vicreg_covariance': 3.4122340262020976}
- P61_vicreg_B32_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.1007991405720604, 'vicreg_variance': 0.2663650509091902, 'vicreg_covariance': 3.323378506459688}
- P61_vicreg_B64_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.14307935958778537, 'vicreg_variance': 0.116845092862039, 'vicreg_covariance': 3.6188113940389535}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P61_simclr_B128_seed0 | 0.0267 | 4800 | 9601 | 1893 | 16 | 7 | 8985600 | None | 1011672@node58 |
| P61_simclr_B256_seed0 | 0.1279 | 2001 | 4002 | 4505 | 31 | 9 | 8960000 | None | 1011732@node60 |
| P61_simclr_B32_lr2x_seed0 | 0.0225 | 1422 | 2843 | 6361 | 31 | 9 | 8998400 | None | 1011392@node60 |
| P61_simclr_B32_seed0 | 0.0093 | 3434 | 6867 | 2652 | 16 | 8 | 8998400 | None | 1011329@node58 |
| P61_simclr_B64_seed0 | 0.0145 | 4416 | 8832 | 2059 | 16 | 7 | 8998400 | None | 1011574@node59 |
| P61_vcs_B128_seed0 | 0.0483 | 2651 | 5301 | 3421 | 82 | 20 | 8985600 | 2 | 1011632@nodesumo01 |
| P61_vcs_B256_seed0 | 0.0563 | 4550 | 9099 | 1998 | 46 | 11 | 8960000 | 2 | 1011731@node58 |
| P61_vcs_B32_lr2x_seed0 | 0.0099 | 3237 | 6475 | 2809 | 45 | 12 | 8998400 | 2 | 1011328@node58 |
| P61_vcs_B32_seed0 | 0.0181 | 1766 | 3532 | 5139 | 83 | 20 | 8998400 | 2 | 1011315@nodesumo01 |
| P61_vcs_B64_seed0 | 0.0256 | 2503 | 5006 | 3629 | 82 | 21 | 8998400 | 2 | 1011511@nodesumo01 |
| P61_vicreg_B128_seed0 | 0.0274 | 4678 | 9356 | 1942 | 16 | 7 | 8985600 | None | 1011681@node58 |
| P61_vicreg_B256_seed0 | 0.0564 | 4541 | 9082 | 2002 | 16 | 7 | 8960000 | None | 1011737@node58 |
| P61_vicreg_B32_lr2x_seed0 | 0.0098 | 3264 | 6528 | 2788 | 16 | 7 | 8998400 | None | 1011423@node58 |
| P61_vicreg_B32_seed0 | 0.0098 | 3266 | 6532 | 2791 | 23 | 7 | 8998400 | None | 1011422@node59 |
| P61_vicreg_B64_seed0 | 0.0151 | 4227 | 8455 | 2151 | 15 | 7 | 8998400 | None | 1011579@node59 |

## Provenance

- P61_simclr_B128_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `a32b0d09cdd82376730590df90fb264c69ba635d5d6c5973a4fcc42ce810af8c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_simclr_B256_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `04229fb8ddbe58af35f9bc72f68f3bf85deec003fd619779330bacfdddd159b8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_simclr_B32_lr2x_seed0: commit `fb271e53e56a732db40846bb20e596c5feac682b` dirty=True, config `9da370bff148b2ccf8465735b6496ecd314723d24bd9fab56f80feb196ae93bc`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_simclr_B32_seed0: commit `fb271e53e56a732db40846bb20e596c5feac682b` dirty=True, config `add791f43f00baccbd1c2c2af39c929d9a35fb0b8b5aea64c79b6b6c34fa1153`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_simclr_B64_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `6f976fb46c4b05fdb2f58d4b80d6ca0a938ce576c6be016148aadd3ed9c1ede1`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vcs_B128_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `23f275db0e175da4f06c999cda4d879b4e40274df80290cdd28de765600fb551`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vcs_B256_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `78779bd7254810d6e83a9a375923ad3361e28652ce1587916082bb2908993918`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vcs_B32_lr2x_seed0: commit `fb271e53e56a732db40846bb20e596c5feac682b` dirty=True, config `355d479748f26ae4ec4185c8e8603083871a5020983e1ce4c2899c32702be00c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vcs_B32_seed0: commit `fb271e53e56a732db40846bb20e596c5feac682b` dirty=True, config `e4d737f826dc5665b6bcaa7895bdd177b0a89cf0b91b56066a4e07e7fd9df4b9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vcs_B64_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `21caffd78ae87523ffe59d41e5c391efbff88045ead12b9e1eb350a452571e7f`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vicreg_B128_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `458ccb097182539fcf7da49aa04b819704e3af92974329f5c844ca0a892c26d1`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vicreg_B256_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `428884ec927145965756422cbe06dd11e00378c3fe3220968eca8bfa12c5bb5e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vicreg_B32_lr2x_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `b51a90229daa648a6756e5803b37c923e659a9c02fcbad0ff6e7a0bbd42cb799`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vicreg_B32_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `7459e8fc889f96473c512bf7591cd8562be2bf0263c1902975e1a0345b450b63`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P61_vicreg_B64_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `06a68c36f214bc4f1ac4421e40ec68b5dd07753dc413a03c7d3fc9c2c6cf2e7b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 5 | 85.16 ± 0.72 | 41.94 ± 0.00 | 43.22 ± 0.72 | 81.59 ± 1.73 | 62.41 ± 20.86 | null |
| vcs_qmi K=8 | 5 | 80.70 ± 0.91 | 41.96 ± 0.02 | 38.74 ± 0.91 | 75.51 ± 1.21 | 35.73 ± 6.87 | 0.94 ± 0.00 |
| vicreg_matched_128 | 5 | 85.58 ± 0.48 | 41.94 ± 0.00 | 43.64 ± 0.48 | 81.78 ± 0.62 | 54.43 ± 16.77 | null |

## Coverage checks

- P61_simclr_B128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_simclr_B256_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_simclr_B32_lr2x_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_simclr_B32_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_simclr_B64_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vcs_B128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vcs_B256_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vcs_B32_lr2x_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vcs_B32_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vcs_B64_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vicreg_B128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vicreg_B256_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vicreg_B32_lr2x_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vicreg_B32_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P61_vicreg_B64_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
