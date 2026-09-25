# P31_vcs_scale_init — neutral results table (observed values only)

Generated 2026-09-25T22:26:48Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P31_vcs_a10_learn_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.66 | 76.30 | 0.9390±0.0013 | 44.98 | 6248 | 3683/4842 | COMPLETED |
| P31_vcs_a20_learn_seed0 | vcs_qmi | 8 | 0 | 200/200 | 78.52 | 73.98 | 0.9291±0.0017 | 36.49 | 6341 | 3683/4842 | COMPLETED |
| P31_vcs_a2_learn_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.84 | 76.42 | 0.9384±0.0016 | 40.19 | 6109 | 3683/4842 | COMPLETED |
| P31_vcs_a5_learn_seed1 | vcs_qmi | 8 | 1 | 200/200 | 81.06 | 76.98 | 0.9410±0.0017 | 45.53 | 6325 | 3683/4842 | COMPLETED |
| P31_vcs_a5_learn_seed2 | vcs_qmi | 8 | 2 | 200/200 | 81.72 | 76.84 | 0.9427±0.0012 | 46.33 | 6332 | 3683/4842 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P31_vcs_a10_learn_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P31_vcs_a20_learn_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P31_vcs_a2_learn_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P31_vcs_a5_learn_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P31_vcs_a5_learn_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P31_vcs_a10_learn_seed0 | 41.78 | 80.66 | 38.88 | 36.58 | 76.30 | 39.72 | 2.94 | 44.98 | -1.0000 |
| P31_vcs_a20_learn_seed0 | 41.78 | 78.52 | 36.74 | 36.58 | 73.98 | 37.40 | 2.94 | 36.49 | -1.0000 |
| P31_vcs_a2_learn_seed0 | 41.78 | 81.84 | 40.06 | 36.58 | 76.42 | 39.84 | 2.94 | 40.19 | -0.9245 |
| P31_vcs_a5_learn_seed1 | 41.60 | 81.06 | 39.46 | 37.42 | 76.98 | 39.56 | 3.22 | 45.53 | -0.9998 |
| P31_vcs_a5_learn_seed2 | 43.02 | 81.72 | 38.70 | 37.08 | 76.84 | 39.76 | 3.22 | 46.33 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P31_vcs_a10_learn_seed0: ep0: 36.58, ep10: 46.46, ep20: 55.36, ep50: 67.58, ep100: 73.18, ep150: 75.32, ep200: 76.30
- P31_vcs_a20_learn_seed0: ep0: 36.58, ep10: 40.94, ep20: 45.42, ep50: 59.80, ep100: 70.36, ep150: 73.50, ep200: 73.98
- P31_vcs_a2_learn_seed0: ep0: 36.58, ep10: 54.42, ep20: 62.00, ep50: 69.06, ep100: 73.30, ep150: 76.22, ep200: 76.42
- P31_vcs_a5_learn_seed1: ep0: 37.42, ep10: 50.00, ep20: 61.06, ep50: 68.88, ep100: 73.34, ep150: 76.34, ep200: 76.98
- P31_vcs_a5_learn_seed2: ep0: 37.08, ep10: 51.52, ep20: 61.62, ep50: 69.56, ep100: 73.74, ep150: 76.76, ep200: 76.84

## Held-out J trajectory (VCS only)

- P31_vcs_a10_learn_seed0: ep0: -1.0000, ep10: 0.3933, ep20: 0.7452, ep50: 0.8524, ep100: 0.9105, ep150: 0.9316, ep200: 0.9390
- P31_vcs_a20_learn_seed0: ep0: -1.0000, ep10: 0.1801, ep20: 0.4711, ep50: 0.8540, ep100: 0.8944, ep150: 0.9207, ep200: 0.9291
- P31_vcs_a2_learn_seed0: ep0: -0.9245, ep10: 0.6282, ep20: 0.8056, ep50: 0.8726, ep100: 0.9136, ep150: 0.9318, ep200: 0.9384
- P31_vcs_a5_learn_seed1: ep0: -0.9998, ep10: 0.5824, ep20: 0.8006, ep50: 0.8714, ep100: 0.9150, ep150: 0.9322, ep200: 0.9410
- P31_vcs_a5_learn_seed2: ep0: -0.9998, ep10: 0.6421, ep20: 0.8081, ep50: 0.8701, ep100: 0.9105, ep150: 0.9359, ep200: 0.9427

## Final-epoch training objective values (epoch means)

- P31_vcs_a10_learn_seed0: J_raw 0.9439, R_binary 0.0561, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P31_vcs_a20_learn_seed0: J_raw 0.9333, R_binary 0.0667, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P31_vcs_a2_learn_seed0: J_raw 0.9433, R_binary 0.0567, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P31_vcs_a5_learn_seed1: J_raw 0.9468, R_binary 0.0532, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P31_vcs_a5_learn_seed2: J_raw 0.9473, R_binary 0.0527, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P31_vcs_a10_learn_seed0 | 0.1764 | 1451 | 2902 | 6248 | 149 | 35 | 8960000 | 2 | 1009129@node05 |
| P31_vcs_a20_learn_seed0 | 0.1786 | 1433 | 2866 | 6341 | 159 | 37 | 8960000 | 2 | 1009131@node06 |
| P31_vcs_a2_learn_seed0 | 0.1724 | 1485 | 2970 | 6109 | 153 | 35 | 8960000 | 2 | 1009128@node05 |
| P31_vcs_a5_learn_seed1 | 0.1786 | 1433 | 2866 | 6325 | 152 | 35 | 8960000 | 2 | 1009126@node01 |
| P31_vcs_a5_learn_seed2 | 0.1784 | 1435 | 2870 | 6332 | 156 | 37 | 8960000 | 2 | 1009127@node01 |

## Provenance

- P31_vcs_a10_learn_seed0: commit `d9cbfd5d003279ffff79d19ac198c7a2e320898a` dirty=True, config `3d6b4c866986ebb9a49456d4e66262a99a4cc2b3739b21ccf841b1a832713bd6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P31_vcs_a20_learn_seed0: commit `96235df871ab962089c62854cb8ca5a4d6853fa0` dirty=False, config `0f8c7b5a56937509e308d5c357838284fcc5bfa48e71c11402351b9371c64f77`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P31_vcs_a2_learn_seed0: commit `d9cbfd5d003279ffff79d19ac198c7a2e320898a` dirty=False, config `3728c3925febdaf19e7f1013174cfc915a95cc3f3e8b76a53652fe462ce153a3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P31_vcs_a5_learn_seed1: commit `82e0bab93b706283a97fb2aff97379d0fb8977e5` dirty=True, config `004a9eebecaf350bd4a311c64ae2889f83ad9974e4a4aec5740463da9e92b0bd`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P31_vcs_a5_learn_seed2: commit `ca3a8de15b3bd5bd0aaaaa61baec124a6ca47286` dirty=False, config `eff8ebde5ec8f43614d9359527d9d6d85ceaa51a60d3924142538d38acb4e2ad`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 5 | 80.76 ± 1.34 | 41.99 ± 0.58 | 38.77 ± 1.25 | 76.10 ± 1.22 | 42.71 ± 4.22 | 0.94 ± 0.01 |

## Coverage checks

- P31_vcs_a10_learn_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P31_vcs_a20_learn_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P31_vcs_a2_learn_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P31_vcs_a5_learn_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P31_vcs_a5_learn_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
