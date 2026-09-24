# P5_confirm200 — neutral results table (observed values only)

Generated 2026-09-24T18:51:31Z. Failed / stopped runs are listed, never dropped.

| run | method | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | simclr_matched | 0 | 200/200 | 86.44 | 84.02 | null | 90.86 | 4506 | 2787/3516 | COMPLETED |
| P5_simclr_seed1 | simclr_matched | 1 | 200/200 | 85.68 | 83.72 | null | 89.80 | 6281 | 3683/4840 | COMPLETED |
| P5_simclr_seed2 | simclr_matched | 2 | 200/200 | 86.14 | 84.20 | null | 89.22 | 6229 | 3683/4840 | COMPLETED |
| P5_vcs_seed0 | vcs_qmi | 0 | 200/200 | 74.84 | 64.64 | 0.8975±0.0047 | 12.87 | 4560 | 2793/3524 | COMPLETED |
| P5_vcs_seed1 | vcs_qmi | 1 | 200/200 | 74.24 | 63.64 | 0.8973±0.0046 | 13.46 | 5715 | 3687/4848 | COMPLETED |
| P5_vcs_seed2 | vcs_qmi | 2 | 200/200 | 73.94 | 63.50 | 0.9013±0.0077 | 13.22 | 5634 | 3687/4848 | COMPLETED |
| P5_vicreg_seed0 | vicreg_matched_128 | 0 | 200/200 | 85.64 | 81.46 | null | 75.26 | 6135 | 3683/4840 | COMPLETED |
| P5_vicreg_seed1 | vicreg_matched_128 | 1 | 200/200 | 85.16 | 82.18 | null | 75.97 | 6259 | 3683/4840 | COMPLETED |
| P5_vicreg_seed2 | vicreg_matched_128 | 2 | 200/200 | 85.58 | 82.12 | null | 75.75 | 4533 | 2787/3516 | COMPLETED |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | 41.94 | 86.44 | 44.50 | 36.54 | 84.02 | 47.48 | 2.94 | 90.86 | null |
| P5_simclr_seed1 | 41.60 | 85.68 | 44.08 | 37.42 | 83.72 | 46.30 | 3.22 | 89.80 | null |
| P5_simclr_seed2 | 43.02 | 86.14 | 43.12 | 37.08 | 84.20 | 47.12 | 3.22 | 89.22 | null |
| P5_vcs_seed0 | 41.94 | 74.84 | 32.90 | 36.54 | 64.64 | 28.10 | 2.94 | 12.87 | -0.0000 |
| P5_vcs_seed1 | 41.60 | 74.24 | 32.64 | 37.42 | 63.64 | 26.22 | 3.22 | 13.46 | -0.0000 |
| P5_vcs_seed2 | 43.02 | 73.94 | 30.92 | 37.08 | 63.50 | 26.42 | 3.22 | 13.22 | -0.0000 |
| P5_vicreg_seed0 | 41.78 | 85.64 | 43.86 | 36.58 | 81.46 | 44.88 | 2.94 | 75.26 | null |
| P5_vicreg_seed1 | 41.60 | 85.16 | 43.56 | 37.42 | 82.18 | 44.76 | 3.22 | 75.97 | null |
| P5_vicreg_seed2 | 42.98 | 85.58 | 42.60 | 37.08 | 82.12 | 45.04 | 3.22 | 75.75 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P5_simclr_seed0: ep0: 36.54, ep10: 61.68, ep20: 68.58, ep50: 76.40, ep100: 81.44, ep150: 83.48, ep200: 84.02
- P5_simclr_seed1: ep0: 37.42, ep10: 61.52, ep20: 68.60, ep50: 76.12, ep100: 81.30, ep150: 83.02, ep200: 83.72
- P5_simclr_seed2: ep0: 37.08, ep10: 62.12, ep20: 69.74, ep50: 76.52, ep100: 81.78, ep150: 83.80, ep200: 84.20
- P5_vcs_seed0: ep0: 36.54, ep10: 42.02, ep20: 47.42, ep50: 54.60, ep100: 60.88, ep150: 63.78, ep200: 64.64
- P5_vcs_seed1: ep0: 37.42, ep10: 43.44, ep20: 48.50, ep50: 55.70, ep100: 60.54, ep150: 63.34, ep200: 63.64
- P5_vcs_seed2: ep0: 37.08, ep10: 41.34, ep20: 49.26, ep50: 56.76, ep100: 61.54, ep150: 63.18, ep200: 63.50
- P5_vicreg_seed0: ep0: 36.58, ep10: 60.52, ep20: 68.06, ep50: 75.74, ep100: 79.24, ep150: 81.00, ep200: 81.46
- P5_vicreg_seed1: ep0: 37.42, ep10: 59.26, ep20: 67.08, ep50: 75.34, ep100: 80.42, ep150: 81.72, ep200: 82.18
- P5_vicreg_seed2: ep0: 37.08, ep10: 58.60, ep20: 69.16, ep50: 76.18, ep100: 79.78, ep150: 81.70, ep200: 82.12

## Held-out J trajectory (VCS only)

- P5_vcs_seed0: ep0: -0.0000, ep10: 0.6055, ep20: 0.7001, ep50: 0.8034, ep100: 0.8584, ep150: 0.8874, ep200: 0.8975
- P5_vcs_seed1: ep0: -0.0000, ep10: 0.6008, ep20: 0.7135, ep50: 0.8034, ep100: 0.8614, ep150: 0.8877, ep200: 0.8973
- P5_vcs_seed2: ep0: -0.0000, ep10: 0.6087, ep20: 0.7111, ep50: 0.8007, ep100: 0.8658, ep150: 0.8899, ep200: 0.9013

## Final-epoch training objective values (epoch means)

- P5_simclr_seed0: J_raw null, R_binary null, nt_xent 2.1375, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P5_simclr_seed1: J_raw null, R_binary null, nt_xent 2.1395, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P5_simclr_seed2: J_raw null, R_binary null, nt_xent 2.1397, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P5_vcs_seed0: J_raw 0.9021, R_binary 0.0979, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P5_vcs_seed1: J_raw 0.9058, R_binary 0.0942, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P5_vcs_seed2: J_raw 0.9068, R_binary 0.0932, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P5_vicreg_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.15438217009816851, 'vicreg_variance': 0.01867345490359834, 'vicreg_covariance': 2.00276171207428}
- P5_vicreg_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.1526246239883559, 'vicreg_variance': 0.018582143022545745, 'vicreg_covariance': 2.0121028300694057}
- P5_vicreg_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.15336868924754007, 'vicreg_variance': 0.018496667247797763, 'vicreg_covariance': 1.9862159872055054}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | 0.1279 | 2001 | 4002 | 4506 | 31 | 9 | 8960000 | None | 1007795@node60 |
| P5_simclr_seed1 | 0.1773 | 1444 | 2887 | 6281 | 59 | 22 | 8960000 | None | 1007809@node01 |
| P5_simclr_seed2 | 0.1758 | 1456 | 2912 | 6229 | 58 | 22 | 8960000 | None | 1007810@node02 |
| P5_vcs_seed0 | 0.1295 | 1977 | 3954 | 4560 | 73 | 15 | 8960000 | 394753 | 1007792@node60 |
| P5_vcs_seed1 | 0.1611 | 1589 | 3177 | 5715 | 144 | 29 | 8960000 | 394753 | 1007807@nodeaudible01 |
| P5_vcs_seed2 | 0.1592 | 1608 | 3217 | 5634 | 126 | 28 | 8960000 | 394753 | 1007808@nodeaudible01 |
| P5_vicreg_seed0 | 0.1732 | 1478 | 2956 | 6135 | 56 | 22 | 8960000 | None | 1007798@node56 |
| P5_vicreg_seed1 | 0.1767 | 1449 | 2897 | 6259 | 58 | 22 | 8960000 | None | 1007811@node02 |
| P5_vicreg_seed2 | 0.1287 | 1989 | 3978 | 4533 | 31 | 9 | 8960000 | None | 1007812@node60 |

## Provenance

- P5_simclr_seed0: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `af3ee84e1256dd366624ea94f2124768b8c97263f0cb760374599271d68ee694`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P5_simclr_seed1: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `3e5bb8b951af97a357edf4a1958e8022f0803e9c48851840875cba19692b8f5a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P5_simclr_seed2: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `363945c3147540fb8c813c130fe2c6587da1717d6589fa70d2da5a27e9f35949`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P5_vcs_seed0: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `94f56c989a30c90f27e4df468f2d476cedc02633e73802ac82cda4f7a6a6b7b3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P5_vcs_seed1: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `7288fb286861982dd848b022e1d48099f94cd952d8b22ffb1ab88a7e599988c5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P5_vcs_seed2: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `7449332e555b7497802d3f49f031236b87caf22709ad7980a39a2f793e28a618`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P5_vicreg_seed0: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `5d1463cf8e1697f3c846dd3084c8b7eb45f08040c091444d4a0b1f8204950823`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P5_vicreg_seed1: commit `13559b9c74009ce8398ebdcbc557cb7f52d6863b` dirty=True, config `9d869699d585c1fb98d7bcd402296565184a2682e065e22ebf32648a7c549733`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P5_vicreg_seed2: commit `7e68404a8a5b88687467e466e61da3292548a60f` dirty=False, config `e4a268d88d85120e8994f3a42ee15b98aad72d375f5a20fe7c78f46155b08c8e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 3 | 86.09 ± 0.38 | 42.19 ± 0.74 | 43.90 ± 0.71 | 83.98 ± 0.24 | 89.96 ± 0.83 | null |
| vcs_qmi | 3 | 74.34 ± 0.46 | 42.19 ± 0.74 | 32.15 ± 1.08 | 63.93 ± 0.62 | 13.18 ± 0.30 | 0.90 ± 0.00 |
| vicreg_matched_128 | 3 | 85.46 ± 0.26 | 42.12 ± 0.75 | 43.34 ± 0.66 | 81.92 ± 0.40 | 75.66 ± 0.37 | null |

## Coverage checks

- P5_simclr_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_simclr_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_simclr_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_vcs_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_vcs_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_vcs_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_vicreg_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_vicreg_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P5_vicreg_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
