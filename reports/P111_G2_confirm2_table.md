# P111_G2_confirm2 — neutral results table (observed values only)

Generated 2026-10-02T14:29:06Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P111_G2_augstrong_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.86 | 84.90 | 0.7855±0.0009 | 75.49 | 16025 | 5269/8140 | COMPLETED |
| P111_G2_augstrong_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 87.70 | 85.30 | 0.7853±0.0011 | 73.67 | 16089 | 5269/8140 | COMPLETED |
| P111_G2_augstrong_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 87.80 | 85.52 | 0.7878±0.0009 | 72.55 | 16098 | 5269/8140 | COMPLETED |
| P111_G2_c100_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 59.46 | 54.12 | 0.8484±0.0023 | 129.99 | 16102 | 5269/8140 | COMPLETED |
| P111_G2_c100_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 59.88 | 54.26 | 0.8484±0.0020 | 129.12 | 27787 | 7066/10148 | COMPLETED |
| P111_G2_c100_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 60.46 | 54.00 | 0.8494±0.0022 | 126.95 | 16079 | 5269/8140 | COMPLETED |
| P111_G2_views4_800ep_seed3 | vcs_qmi | 8 | 3 | 800/800 | 88.72 | 87.04 | 0.8616±0.0027 | 97.40 | 16097 | 5269/8140 | COMPLETED |
| P111_G2_views4_800ep_seed4 | vcs_qmi | 8 | 4 | 800/800 | 88.62 | 86.98 | 0.8617±0.0023 | 99.67 | 16081 | 5269/8140 | COMPLETED |
| P111_simclr_views4_800ep_seed3 | simclr_matched | null | 3 | 800/800 | 88.28 | 88.36 | null | 166.07 | 15817 | 5269/8140 | COMPLETED |
| P111_simclr_views4_800ep_seed4 | simclr_matched | null | 4 | 800/800 | 88.30 | 88.00 | null | 159.45 | 34793 | 5269/8140 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P111_G2_augstrong_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_augstrong_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_augstrong_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_c100_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_c100_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_c100_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_views4_800ep_seed3 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_G2_views4_800ep_seed4 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P111_simclr_views4_800ep_seed3 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P111_simclr_views4_800ep_seed4 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P111_G2_augstrong_views4_800ep_seed0 | 41.94 | 86.86 | 44.92 | 36.54 | 84.90 | 48.36 | 2.94 | 75.49 | -0.5526 |
| P111_G2_augstrong_views4_800ep_seed1 | 41.68 | 87.70 | 46.02 | 37.40 | 85.30 | 47.90 | 3.22 | 73.67 | -0.5437 |
| P111_G2_augstrong_views4_800ep_seed2 | 42.98 | 87.80 | 44.82 | 37.08 | 85.52 | 48.44 | 3.22 | 72.55 | -0.5406 |
| P111_G2_c100_views4_800ep_seed0 | 20.12 | 59.46 | 39.34 | 14.00 | 54.12 | 40.12 | 2.74 | 129.99 | -0.5552 |
| P111_G2_c100_views4_800ep_seed1 | 19.34 | 59.88 | 40.54 | 14.72 | 54.26 | 39.54 | 2.89 | 129.12 | -0.5486 |
| P111_G2_c100_views4_800ep_seed2 | 20.04 | 60.46 | 40.42 | 13.78 | 54.00 | 40.22 | 2.92 | 126.95 | -0.5447 |
| P111_G2_views4_800ep_seed3 | 41.86 | 88.72 | 46.86 | 37.16 | 87.04 | 49.88 | 2.98 | 97.40 | -0.5588 |
| P111_G2_views4_800ep_seed4 | 42.88 | 88.62 | 45.74 | 36.56 | 86.98 | 50.42 | 2.60 | 99.67 | -0.5540 |
| P111_simclr_views4_800ep_seed3 | 41.86 | 88.28 | 46.42 | 37.16 | 88.36 | 51.20 | 2.98 | 166.07 | null |
| P111_simclr_views4_800ep_seed4 | 42.88 | 88.30 | 45.42 | 36.56 | 88.00 | 51.44 | 2.60 | 159.45 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P111_G2_augstrong_views4_800ep_seed0: ep0: 36.54, ep20: 66.46, ep50: 72.00, ep100: 77.60, ep200: 81.18, ep400: 84.12, ep600: 84.38, ep800: 84.90
- P111_G2_augstrong_views4_800ep_seed1: ep0: 37.40, ep20: 65.48, ep50: 72.80, ep100: 77.70, ep200: 81.70, ep400: 84.14, ep600: 84.74, ep800: 85.30
- P111_G2_augstrong_views4_800ep_seed2: ep0: 37.08, ep20: 65.92, ep50: 73.50, ep100: 77.86, ep200: 81.58, ep400: 84.26, ep600: 85.14, ep800: 85.52
- P111_G2_c100_views4_800ep_seed0: ep0: 14.00, ep20: 25.00, ep50: 34.34, ep100: 43.26, ep200: 49.46, ep400: 52.92, ep600: 54.00, ep800: 54.12
- P111_G2_c100_views4_800ep_seed1: ep0: 14.72, ep20: 24.78, ep50: 35.08, ep100: 43.24, ep200: 48.74, ep400: 52.96, ep600: 54.00, ep800: 54.26
- P111_G2_c100_views4_800ep_seed2: ep0: 13.78, ep20: 25.62, ep50: 34.88, ep100: 42.48, ep200: 48.86, ep400: 52.74, ep600: 53.92, ep800: 54.00
- P111_G2_views4_800ep_seed3: ep0: 37.16, ep20: 67.36, ep50: 75.70, ep100: 80.92, ep200: 83.76, ep400: 86.24, ep600: 86.98, ep800: 87.04
- P111_G2_views4_800ep_seed4: ep0: 36.56, ep20: 67.56, ep50: 76.12, ep100: 80.26, ep200: 84.16, ep400: 85.98, ep600: 86.64, ep800: 86.98
- P111_simclr_views4_800ep_seed3: ep0: 37.16, ep20: 73.14, ep50: 81.06, ep100: 85.38, ep200: 86.92, ep400: 87.84, ep600: 88.34, ep800: 88.36
- P111_simclr_views4_800ep_seed4: ep0: 36.56, ep20: 73.22, ep50: 80.94, ep100: 84.56, ep200: 86.60, ep400: 87.80, ep600: 87.68, ep800: 88.00

## Held-out J trajectory (VCS only)

- P111_G2_augstrong_views4_800ep_seed0: ep0: -0.5526, ep20: 0.6171, ep50: 0.6754, ep100: 0.7228, ep200: 0.7423, ep400: 0.7718, ep600: 0.7823, ep800: 0.7855
- P111_G2_augstrong_views4_800ep_seed1: ep0: -0.5437, ep20: 0.6150, ep50: 0.6744, ep100: 0.7272, ep200: 0.7526, ep400: 0.7701, ep600: 0.7814, ep800: 0.7853
- P111_G2_augstrong_views4_800ep_seed2: ep0: -0.5406, ep20: 0.6231, ep50: 0.6873, ep100: 0.7257, ep200: 0.7556, ep400: 0.7740, ep600: 0.7844, ep800: 0.7878
- P111_G2_c100_views4_800ep_seed0: ep0: -0.5552, ep20: 0.7533, ep50: 0.7997, ep100: 0.8313, ep200: 0.8430, ep400: 0.8478, ep600: 0.8486, ep800: 0.8484
- P111_G2_c100_views4_800ep_seed1: ep0: -0.5486, ep20: 0.7476, ep50: 0.8001, ep100: 0.8280, ep200: 0.8444, ep400: 0.8482, ep600: 0.8483, ep800: 0.8484
- P111_G2_c100_views4_800ep_seed2: ep0: -0.5447, ep20: 0.7552, ep50: 0.8030, ep100: 0.8298, ep200: 0.8445, ep400: 0.8489, ep600: 0.8494, ep800: 0.8494
- P111_G2_views4_800ep_seed3: ep0: -0.5588, ep20: 0.7659, ep50: 0.8160, ep100: 0.8404, ep200: 0.8534, ep400: 0.8582, ep600: 0.8611, ep800: 0.8616
- P111_G2_views4_800ep_seed4: ep0: -0.5540, ep20: 0.7730, ep50: 0.8153, ep100: 0.8382, ep200: 0.8530, ep400: 0.8591, ep600: 0.8611, ep800: 0.8617

## Final-epoch training objective values (epoch means)

- P111_G2_augstrong_views4_800ep_seed0: J_raw 0.8084, R_binary 0.1916, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_augstrong_views4_800ep_seed1: J_raw 0.8085, R_binary 0.1915, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_augstrong_views4_800ep_seed2: J_raw 0.8106, R_binary 0.1894, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_c100_views4_800ep_seed0: J_raw 0.8912, R_binary 0.1088, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_c100_views4_800ep_seed1: J_raw 0.8908, R_binary 0.1092, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_c100_views4_800ep_seed2: J_raw 0.8911, R_binary 0.1089, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_views4_800ep_seed3: J_raw 0.8941, R_binary 0.1059, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_G2_views4_800ep_seed4: J_raw 0.8942, R_binary 0.1058, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_simclr_views4_800ep_seed3: J_raw null, R_binary null, nt_xent 1.8670, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P111_simclr_views4_800ep_seed4: J_raw null, R_binary null, nt_xent 1.8665, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P111_G2_augstrong_views4_800ep_seed0 | 0.1130 | 2265 | 4531 | 16025 | 51 | 11 | 35840000 | None | 1017771@node61 |
| P111_G2_augstrong_views4_800ep_seed1 | 0.1134 | 2257 | 4514 | 16089 | 52 | 11 | 35840000 | None | 1017772@node58 |
| P111_G2_augstrong_views4_800ep_seed2 | 0.1135 | 2256 | 4513 | 16098 | 57 | 12 | 35840000 | None | 1017773@node58 |
| P111_G2_c100_views4_800ep_seed0 | 0.1136 | 2254 | 4509 | 16102 | 52 | 11 | 35840000 | None | 1017774@node59 |
| P111_G2_c100_views4_800ep_seed1 | 0.1956 | 1309 | 2617 | 27787 | 102 | 21 | 35840000 | None | 1017775@nodesumo01 |
| P111_G2_c100_views4_800ep_seed2 | 0.1133 | 2259 | 4519 | 16079 | 53 | 11 | 35840000 | None | 1017776@node59 |
| P111_G2_views4_800ep_seed3 | 0.1134 | 2257 | 4513 | 16097 | 54 | 11 | 35840000 | None | 1017767@node58 |
| P111_G2_views4_800ep_seed4 | 0.1133 | 2259 | 4517 | 16081 | 53 | 12 | 35840000 | None | 1017769@node59 |
| P111_simclr_views4_800ep_seed3 | 0.1115 | 2295 | 4590 | 15817 | 20 | 8 | 35840000 | None | 1017768@node59 |
| P111_simclr_views4_800ep_seed4 | 0.2471 | 1036 | 2072 | 34793 | 36 | 9 | 35840000 | None | 1017770@node61 |

## Provenance

- P111_G2_augstrong_views4_800ep_seed0: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `ad93a5bdf6f71d460f59d6d6b3f05955baec7e11bd7b71b409711f6d836a96a3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P111_G2_augstrong_views4_800ep_seed1: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `e1433f6211536e7c3d44a9fde2984e2ea005b06f1de3ca72646c1086a10a28a5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P111_G2_augstrong_views4_800ep_seed2: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `438c4a09629911a9736bdc92c6972ba5002dab684c90aa9528587e2ce138c75a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P111_G2_c100_views4_800ep_seed0: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `2eb7f84645ac558264cce37189f6d597553f2b6997818e604631cb6050b56c4f`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P111_G2_c100_views4_800ep_seed1: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `c56f06d727bd0c2c4b8a5f81f266c07ed94bd1db750feb308ef61598a6241c5a`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P111_G2_c100_views4_800ep_seed2: commit `60f5cdfc665864846253471939063c33df7c99bb` dirty=True, config `4442c539422d7c31e646fbfd2141c2113b7935360f2fd0c177dcc1429a06dd6d`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P111_G2_views4_800ep_seed3: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `542090aff55d76118f897661d492bbd80ff2ecc851208dceca7e7fc0ee9dfa67`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `fecfb95c91407f043cc32395ba397ef6008bd47ffaba4c8fd7465385aca33342`
- P111_G2_views4_800ep_seed4: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `ca632f708f8a63e1c0015351dd084e8749a24c6d59c68bc5fe951c400ef86ba6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `d0e3a8dd8745d301009fa90da9468814b4d11cc4fec5f2df49ca63f73d65dace`
- P111_simclr_views4_800ep_seed3: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `e47952e1dde50d4b50067e0087d3aa7a53909f6ca2572722e98dd9bcb82aef45`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `fecfb95c91407f043cc32395ba397ef6008bd47ffaba4c8fd7465385aca33342`
- P111_simclr_views4_800ep_seed4: commit `2c7a2f21fcdd27f0596d218e30f2f4946896a4ce` dirty=True, config `0d3f5f0a37b7b2d96d8ea3a1a5ab493ed1fbed36b0404f1a8670b1436c8b69e8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `d0e3a8dd8745d301009fa90da9468814b4d11cc4fec5f2df49ca63f73d65dace`

- identical encoder init across runs: False
- identical split across runs: False

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 2 | 88.29 ± 0.01 | 42.37 ± 0.72 | 45.92 ± 0.71 | 88.18 ± 0.25 | 162.76 ± 4.68 | null |
| vcs_qmi K=8 | 8 | 77.44 ± 14.51 | 33.85 ± 11.62 | 43.58 ± 2.97 | 74.02 ± 16.49 | 100.60 ± 25.43 | 0.83 ± 0.04 |

## Coverage checks

- P111_G2_augstrong_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_augstrong_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_augstrong_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_c100_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_c100_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_c100_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_views4_800ep_seed3: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_G2_views4_800ep_seed4: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_simclr_views4_800ep_seed3: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P111_simclr_views4_800ep_seed4: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
