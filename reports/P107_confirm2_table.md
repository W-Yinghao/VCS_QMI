# P107_confirm2 — neutral results table (observed values only)

Generated 2026-10-03T14:03:35Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.66 | 86.20 | 0.7965±0.0012 | 91.02 | 16103 | 5269/8146 | COMPLETED |
| P107_AP3_augstrong_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 88.26 | 85.96 | 0.7961±0.0014 | 87.91 | 16053 | 5269/8146 | COMPLETED |
| P107_AP3_augstrong_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 87.82 | 86.18 | 0.7971±0.0014 | 90.93 | 16082 | 5269/8146 | COMPLETED |
| P107_AP3_c100_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 60.20 | 55.92 | 0.8462±0.0032 | 155.53 | 27793 | 7066/10154 | COMPLETED |
| P107_AP3_c100_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 60.08 | 55.96 | 0.8455±0.0018 | 157.71 | 27190 | 7066/10154 | COMPLETED |
| P107_AP3_c100_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 59.72 | 55.52 | 0.8463±0.0024 | 157.66 | 27685 | 7066/10154 | COMPLETED |
| P107_AP3_views4_800ep_seed3 | vcs_qmi | 8 | 3 | 800/800 | 89.00 | 87.46 | 0.8606±0.0024 | 120.37 | 35935 | 5269/8146 | COMPLETED |
| P107_AP3_views4_800ep_seed4 | vcs_qmi | 8 | 4 | 800/800 | 89.04 | 87.44 | 0.8597±0.0037 | 116.65 | 35169 | 5269/8146 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_augstrong_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_augstrong_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_c100_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_c100_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_c100_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_views4_800ep_seed3 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_views4_800ep_seed4 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 | 41.94 | 88.66 | 46.72 | 36.54 | 86.20 | 49.66 | 2.94 | 91.02 | -0.5526 |
| P107_AP3_augstrong_views4_800ep_seed1 | 41.68 | 88.26 | 46.58 | 37.40 | 85.96 | 48.56 | 3.22 | 87.91 | -0.5437 |
| P107_AP3_augstrong_views4_800ep_seed2 | 42.98 | 87.82 | 44.84 | 37.08 | 86.18 | 49.10 | 3.22 | 90.93 | -0.5406 |
| P107_AP3_c100_views4_800ep_seed0 | 20.12 | 60.20 | 40.08 | 14.02 | 55.92 | 41.90 | 2.74 | 155.53 | -0.5552 |
| P107_AP3_c100_views4_800ep_seed1 | 19.34 | 60.08 | 40.74 | 14.72 | 55.96 | 41.24 | 2.89 | 157.71 | -0.5486 |
| P107_AP3_c100_views4_800ep_seed2 | 20.08 | 59.72 | 39.64 | 13.82 | 55.52 | 41.70 | 2.92 | 157.66 | -0.5447 |
| P107_AP3_views4_800ep_seed3 | 41.86 | 89.00 | 47.14 | 37.16 | 87.46 | 50.30 | 2.98 | 120.37 | -0.5588 |
| P107_AP3_views4_800ep_seed4 | 42.88 | 89.04 | 46.16 | 36.56 | 87.44 | 50.88 | 2.60 | 116.65 | -0.5540 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P107_AP3_augstrong_views4_800ep_seed0: ep0: 36.54, ep20: 69.00, ep50: 75.50, ep100: 79.68, ep200: 83.08, ep400: 85.56, ep600: 86.20, ep800: 86.20
- P107_AP3_augstrong_views4_800ep_seed1: ep0: 37.40, ep20: 69.58, ep50: 75.76, ep100: 80.14, ep200: 82.56, ep400: 85.16, ep600: 85.80, ep800: 85.96
- P107_AP3_augstrong_views4_800ep_seed2: ep0: 37.08, ep20: 68.38, ep50: 75.58, ep100: 80.14, ep200: 82.42, ep400: 84.86, ep600: 85.96, ep800: 86.18
- P107_AP3_c100_views4_800ep_seed0: ep0: 14.02, ep20: 32.26, ep50: 40.34, ep100: 47.76, ep200: 52.86, ep400: 54.68, ep600: 55.62, ep800: 55.92
- P107_AP3_c100_views4_800ep_seed1: ep0: 14.72, ep20: 33.16, ep50: 41.14, ep100: 47.70, ep200: 52.28, ep400: 55.20, ep600: 55.26, ep800: 55.96
- P107_AP3_c100_views4_800ep_seed2: ep0: 13.82, ep20: 32.66, ep50: 40.58, ep100: 47.74, ep200: 51.70, ep400: 54.54, ep600: 55.14, ep800: 55.52
- P107_AP3_views4_800ep_seed3: ep0: 37.16, ep20: 69.80, ep50: 77.12, ep100: 81.64, ep200: 85.22, ep400: 86.98, ep600: 87.36, ep800: 87.46
- P107_AP3_views4_800ep_seed4: ep0: 36.56, ep20: 70.06, ep50: 78.12, ep100: 81.78, ep200: 84.94, ep400: 86.74, ep600: 87.30, ep800: 87.44

## Held-out J trajectory (VCS only)

- P107_AP3_augstrong_views4_800ep_seed0: ep0: -0.5526, ep20: 0.6507, ep50: 0.7179, ep100: 0.7485, ep200: 0.7680, ep400: 0.7836, ep600: 0.7937, ep800: 0.7965
- P107_AP3_augstrong_views4_800ep_seed1: ep0: -0.5437, ep20: 0.6691, ep50: 0.7168, ep100: 0.7465, ep200: 0.7735, ep400: 0.7853, ep600: 0.7929, ep800: 0.7961
- P107_AP3_augstrong_views4_800ep_seed2: ep0: -0.5406, ep20: 0.6650, ep50: 0.7199, ep100: 0.7479, ep200: 0.7730, ep400: 0.7870, ep600: 0.7945, ep800: 0.7971
- P107_AP3_c100_views4_800ep_seed0: ep0: -0.5552, ep20: 0.7910, ep50: 0.8223, ep100: 0.8443, ep200: 0.8506, ep400: 0.8493, ep600: 0.8465, ep800: 0.8462
- P107_AP3_c100_views4_800ep_seed1: ep0: -0.5486, ep20: 0.7910, ep50: 0.8220, ep100: 0.8422, ep200: 0.8502, ep400: 0.8498, ep600: 0.8456, ep800: 0.8455
- P107_AP3_c100_views4_800ep_seed2: ep0: -0.5447, ep20: 0.7882, ep50: 0.8253, ep100: 0.8437, ep200: 0.8503, ep400: 0.8480, ep600: 0.8464, ep800: 0.8463
- P107_AP3_views4_800ep_seed3: ep0: -0.5588, ep20: 0.7974, ep50: 0.8309, ep100: 0.8511, ep200: 0.8595, ep400: 0.8602, ep600: 0.8605, ep800: 0.8606
- P107_AP3_views4_800ep_seed4: ep0: -0.5540, ep20: 0.7964, ep50: 0.8319, ep100: 0.8505, ep200: 0.8604, ep400: 0.8607, ep600: 0.8596, ep800: 0.8597

## Final-epoch training objective values (epoch means)

- P107_AP3_augstrong_views4_800ep_seed0: J_raw 0.8243, R_binary 0.1757, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_augstrong_views4_800ep_seed1: J_raw 0.8242, R_binary 0.1758, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_augstrong_views4_800ep_seed2: J_raw 0.8261, R_binary 0.1739, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_c100_views4_800ep_seed0: J_raw 0.9006, R_binary 0.0994, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_c100_views4_800ep_seed1: J_raw 0.9003, R_binary 0.0997, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_c100_views4_800ep_seed2: J_raw 0.9005, R_binary 0.0995, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_views4_800ep_seed3: J_raw 0.9029, R_binary 0.0971, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_views4_800ep_seed4: J_raw 0.9032, R_binary 0.0968, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 | 0.1134 | 2257 | 4513 | 16103 | 57 | 12 | 35840000 | None | 1018765@node59 |
| P107_AP3_augstrong_views4_800ep_seed1 | 0.1132 | 2261 | 4523 | 16053 | 51 | 12 | 35840000 | None | 1018766@node58 |
| P107_AP3_augstrong_views4_800ep_seed2 | 0.1134 | 2258 | 4516 | 16082 | 57 | 12 | 35840000 | None | 1018767@node59 |
| P107_AP3_c100_views4_800ep_seed0 | 0.1959 | 1307 | 2614 | 27793 | 91 | 20 | 35840000 | None | 1018768@node53 |
| P107_AP3_c100_views4_800ep_seed1 | 0.1916 | 1336 | 2673 | 27190 | 90 | 20 | 35840000 | None | 1018769@node53 |
| P107_AP3_c100_views4_800ep_seed2 | 0.1951 | 1312 | 2624 | 27685 | 90 | 20 | 35840000 | None | 1018770@node53 |
| P107_AP3_views4_800ep_seed3 | 0.2553 | 1003 | 2006 | 35935 | 86 | 16 | 35840000 | None | 1018763@node61 |
| P107_AP3_views4_800ep_seed4 | 0.2498 | 1025 | 2049 | 35169 | 83 | 16 | 35840000 | None | 1018764@node60 |

## Provenance

- P107_AP3_augstrong_views4_800ep_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `0381a7bc8364607ecf71475bf4b1625b533df312f3dcb86865bb84bc24ee4834`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP3_augstrong_views4_800ep_seed1: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `ec218e96d2c6fc4d7f492e84c3d2791c934fa971d258d5765d4ab6e730809b8a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P107_AP3_augstrong_views4_800ep_seed2: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `2ac01da2813d82c29b19812acb1d3bb61dd2e568fc1bf16285cc298cc2d4b71c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P107_AP3_c100_views4_800ep_seed0: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `b0ecf86f425782e9a6a47f4aceba688c649a701b89709eb33f800e98de0cec20`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP3_c100_views4_800ep_seed1: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `b3601b2eb95ea46b4d6db7ec678823bd4cb385b24a45f22a274f964387baac3f`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P107_AP3_c100_views4_800ep_seed2: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `638106e69888e3dc78b7c20fcf8c9fad730b40ae8f0c558ec13ceaa35295e5c9`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P107_AP3_views4_800ep_seed3: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `c3c4e0b736aa5ff8343cebfb8e032188a16eb123f5a0aff0e2af3b1857358e03`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `fecfb95c91407f043cc32395ba397ef6008bd47ffaba4c8fd7465385aca33342`
- P107_AP3_views4_800ep_seed4: commit `c3fd5df0b5e13dac3dde8ba3f1654c2f517fcfbb` dirty=True, config `2c1517bc88c0a379ce5bdf0fc6468d35315538015319c9f66f8d1b73a47c3794`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `d0e3a8dd8745d301009fa90da9468814b4d11cc4fec5f2df49ca63f73d65dace`

- identical encoder init across runs: False
- identical split across runs: False

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 8 | 77.85 ± 14.78 | 33.86 ± 11.62 | 43.99 ± 3.26 | 75.08 ± 15.98 | 122.22 ± 31.14 | 0.83 ± 0.03 |

## Coverage checks

- P107_AP3_augstrong_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_augstrong_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_augstrong_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_c100_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_c100_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_c100_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_views4_800ep_seed3: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_views4_800ep_seed4: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
