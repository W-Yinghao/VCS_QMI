# P10_kstudy200 — neutral results table (observed values only)

Generated 2026-09-24T23:20:10Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P10_vcs_k64_seed0 | vcs_qmi | 64 | 0 | 200/200 | 76.46 | 67.86 | 0.9324±0.0022 | 15.92 | 6258 | 3688/4848 | COMPLETED |
| P10_vcs_k64_seed1 | vcs_qmi | 64 | 1 | 200/200 | 76.86 | 66.82 | 0.9322±0.0021 | 15.75 | 6469 | 3688/4848 | COMPLETED |
| P10_vcs_k64_seed2 | vcs_qmi | 64 | 2 | 200/200 | 77.08 | 67.90 | 0.9324±0.0018 | 16.11 | 6295 | 3688/4848 | COMPLETED |
| P10_vcs_k8_seed0 | vcs_qmi | 8 | 0 | 200/200 | 76.12 | 67.44 | 0.9261±0.0012 | 15.59 | 6394 | 3687/4848 | COMPLETED |
| P10_vcs_k8_seed1 | vcs_qmi | 8 | 1 | 200/200 | 77.62 | 66.90 | 0.9286±0.0030 | 15.73 | 6373 | 3687/4848 | COMPLETED |
| P10_vcs_k8_seed2 | vcs_qmi | 8 | 2 | 200/200 | 76.48 | 66.92 | 0.9263±0.0024 | 15.37 | 6356 | 3687/4848 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P10_vcs_k64_seed0 | 64 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P10_vcs_k64_seed1 | 64 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P10_vcs_k64_seed2 | 64 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P10_vcs_k8_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P10_vcs_k8_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P10_vcs_k8_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P10_vcs_k64_seed0 | 41.78 | 76.46 | 34.68 | 36.58 | 67.86 | 31.28 | 2.94 | 15.92 | -0.0000 |
| P10_vcs_k64_seed1 | 41.60 | 76.86 | 35.26 | 37.42 | 66.82 | 29.40 | 3.22 | 15.75 | -0.0000 |
| P10_vcs_k64_seed2 | 43.02 | 77.08 | 34.06 | 37.08 | 67.90 | 30.82 | 3.22 | 16.11 | -0.0000 |
| P10_vcs_k8_seed0 | 41.78 | 76.12 | 34.34 | 36.58 | 67.44 | 30.86 | 2.94 | 15.59 | -0.0000 |
| P10_vcs_k8_seed1 | 41.60 | 77.62 | 36.02 | 37.42 | 66.90 | 29.48 | 3.22 | 15.73 | -0.0000 |
| P10_vcs_k8_seed2 | 43.02 | 76.48 | 33.46 | 37.08 | 66.92 | 29.84 | 3.22 | 15.37 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P10_vcs_k64_seed0: ep0: 36.58, ep10: 46.84, ep20: 54.06, ep50: 60.68, ep100: 65.28, ep150: 67.28, ep200: 67.86
- P10_vcs_k64_seed1: ep0: 37.42, ep10: 47.32, ep20: 53.20, ep50: 60.90, ep100: 64.90, ep150: 66.44, ep200: 66.82
- P10_vcs_k64_seed2: ep0: 37.08, ep10: 50.24, ep20: 54.98, ep50: 61.82, ep100: 65.08, ep150: 67.16, ep200: 67.90
- P10_vcs_k8_seed0: ep0: 36.58, ep10: 45.58, ep20: 52.44, ep50: 60.30, ep100: 64.76, ep150: 66.82, ep200: 67.44
- P10_vcs_k8_seed1: ep0: 37.42, ep10: 47.10, ep20: 54.34, ep50: 60.88, ep100: 64.66, ep150: 66.72, ep200: 66.90
- P10_vcs_k8_seed2: ep0: 37.08, ep10: 45.78, ep20: 53.38, ep50: 60.58, ep100: 64.36, ep150: 66.18, ep200: 66.92

## Held-out J trajectory (VCS only)

- P10_vcs_k64_seed0: ep0: -0.0000, ep10: 0.6802, ep20: 0.7657, ep50: 0.8525, ep100: 0.9044, ep150: 0.9262, ep200: 0.9324
- P10_vcs_k64_seed1: ep0: -0.0000, ep10: 0.6825, ep20: 0.7709, ep50: 0.8576, ep100: 0.8988, ep150: 0.9249, ep200: 0.9322
- P10_vcs_k64_seed2: ep0: -0.0000, ep10: 0.7090, ep20: 0.7904, ep50: 0.8578, ep100: 0.9041, ep150: 0.9268, ep200: 0.9324
- P10_vcs_k8_seed0: ep0: -0.0000, ep10: 0.6781, ep20: 0.7384, ep50: 0.8500, ep100: 0.8959, ep150: 0.9176, ep200: 0.9261
- P10_vcs_k8_seed1: ep0: -0.0000, ep10: 0.7051, ep20: 0.7896, ep50: 0.8550, ep100: 0.8986, ep150: 0.9204, ep200: 0.9286
- P10_vcs_k8_seed2: ep0: -0.0000, ep10: 0.6669, ep20: 0.7809, ep50: 0.8525, ep100: 0.8951, ep150: 0.9193, ep200: 0.9263

## Final-epoch training objective values (epoch means)

- P10_vcs_k64_seed0: J_raw 0.9351, R_binary 0.0649, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P10_vcs_k64_seed1: J_raw 0.9373, R_binary 0.0627, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P10_vcs_k64_seed2: J_raw 0.9369, R_binary 0.0631, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P10_vcs_k8_seed0: J_raw 0.9292, R_binary 0.0708, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P10_vcs_k8_seed1: J_raw 0.9346, R_binary 0.0654, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P10_vcs_k8_seed2: J_raw 0.9302, R_binary 0.0698, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P10_vcs_k64_seed0 | 0.1767 | 1449 | 2898 | 6258 | 150 | 35 | 8960000 | 394753 | 1007969@node04 |
| P10_vcs_k64_seed1 | 0.1827 | 1401 | 2802 | 6469 | 153 | 36 | 8960000 | 394753 | 1007970@node04 |
| P10_vcs_k64_seed2 | 0.1777 | 1441 | 2882 | 6295 | 144 | 35 | 8960000 | 394753 | 1007971@node54 |
| P10_vcs_k8_seed0 | 0.1806 | 1418 | 2836 | 6394 | 149 | 36 | 8960000 | 394753 | 1007966@node01 |
| P10_vcs_k8_seed1 | 0.1800 | 1423 | 2845 | 6373 | 153 | 36 | 8960000 | 394753 | 1007967@node02 |
| P10_vcs_k8_seed2 | 0.1794 | 1427 | 2853 | 6356 | 151 | 36 | 8960000 | 394753 | 1007968@node02 |

## Provenance

- P10_vcs_k64_seed0: commit `b13b684e253c96a38c14efabba3c79a37bc6a27e` dirty=True, config `cc0a04dcd055b6a4fc4e5469ecb2d10bc3ebc896c67177e2d033867a0b87cca8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P10_vcs_k64_seed1: commit `b13b684e253c96a38c14efabba3c79a37bc6a27e` dirty=True, config `3f7366bd9e307404853dccf3120c227aa9875a54c61543839e2a76665db6acd7`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P10_vcs_k64_seed2: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `7a2e191dc3c13f3af0fff20f523255b779f0074cbf6116b2a04a4f162de52872`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P10_vcs_k8_seed0: commit `b13b684e253c96a38c14efabba3c79a37bc6a27e` dirty=True, config `812c0f288726a30486dac0e2b72556fff628178f3b23c0f0781004b3d5a345ce`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P10_vcs_k8_seed1: commit `b13b684e253c96a38c14efabba3c79a37bc6a27e` dirty=True, config `06e32708aab62597fb7bb1a24d36554eb22fc53aab6b5ae153ce4a3412d1729a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P10_vcs_k8_seed2: commit `b13b684e253c96a38c14efabba3c79a37bc6a27e` dirty=True, config `5d5bc4d42101f217bffbe9a4a61be1142844962124367ff618f042cad52dc7d1`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=64 | 3 | 76.80 ± 0.31 | 42.13 ± 0.77 | 34.67 ± 0.60 | 67.53 ± 0.61 | 15.93 ± 0.18 | 0.93 ± 0.00 |
| vcs_qmi K=8 | 3 | 76.74 ± 0.78 | 42.13 ± 0.77 | 34.61 ± 1.30 | 67.09 ± 0.31 | 15.57 ± 0.18 | 0.93 ± 0.00 |

## Coverage checks

- P10_vcs_k64_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P10_vcs_k64_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P10_vcs_k64_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P10_vcs_k8_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P10_vcs_k8_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P10_vcs_k8_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
