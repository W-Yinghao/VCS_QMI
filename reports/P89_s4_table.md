# P89_s4_aug_interaction — neutral results table (observed values only)

Generated 2026-09-29T16:56:35Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P89_simclr_views4_800ep_augstrong_seed0 | simclr_matched | null | 0 | 800/800 | 89.74 | 88.86 | null | 134.39 | 32900 | 7063/9772 | COMPLETED |
| P89_simclr_views4_800ep_augstrong_seed1 | simclr_matched | null | 1 | 800/800 | 89.08 | 88.68 | null | 133.83 | 30445 | 7063/9772 | COMPLETED |
| P89_simclr_views4_800ep_augstrong_seed2 | simclr_matched | null | 2 | 800/800 | 89.66 | 88.60 | null | 139.21 | 15818 | 5269/8140 | COMPLETED |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | vcs_qmi | 8 | 1 | 800/800 | 87.54 | 85.42 | 0.8761±0.0004 | 104.87 | 16142 | 5269/8140 | COMPLETED |
| P89_vcs_a5_views4_800ep_augstrong_seed2 | vcs_qmi | 8 | 2 | 800/800 | 87.66 | 84.84 | 0.8768±0.0014 | 102.13 | 16076 | 5269/8140 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P89_simclr_views4_800ep_augstrong_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P89_simclr_views4_800ep_augstrong_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P89_simclr_views4_800ep_augstrong_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P89_vcs_a5_views4_800ep_augstrong_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P89_simclr_views4_800ep_augstrong_seed0 | 41.98 | 89.74 | 47.76 | 36.58 | 88.86 | 52.28 | 2.94 | 134.39 | null |
| P89_simclr_views4_800ep_augstrong_seed1 | 41.72 | 89.08 | 47.36 | 37.42 | 88.68 | 51.26 | 3.22 | 133.83 | null |
| P89_simclr_views4_800ep_augstrong_seed2 | 42.98 | 89.66 | 46.68 | 37.08 | 88.60 | 51.52 | 3.22 | 139.21 | null |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 41.68 | 87.54 | 45.86 | 37.40 | 85.42 | 48.02 | 3.22 | 104.87 | -0.9997 |
| P89_vcs_a5_views4_800ep_augstrong_seed2 | 42.98 | 87.66 | 44.68 | 37.08 | 84.84 | 47.76 | 3.22 | 102.13 | -0.9997 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P89_simclr_views4_800ep_augstrong_seed0: ep0: 36.58, ep20: 73.48, ep50: 80.00, ep100: 84.24, ep200: 86.92, ep400: 88.12, ep600: 88.58, ep800: 88.86
- P89_simclr_views4_800ep_augstrong_seed1: ep0: 37.42, ep20: 73.48, ep50: 80.76, ep100: 84.40, ep200: 86.22, ep400: 88.08, ep600: 88.38, ep800: 88.68
- P89_simclr_views4_800ep_augstrong_seed2: ep0: 37.08, ep20: 71.66, ep50: 80.40, ep100: 84.48, ep200: 86.72, ep400: 88.56, ep600: 88.46, ep800: 88.60
- P89_vcs_a5_views4_800ep_augstrong_seed1: ep0: 37.40, ep20: 68.30, ep50: 75.88, ep100: 79.78, ep200: 82.30, ep400: 84.14, ep600: 85.08, ep800: 85.42
- P89_vcs_a5_views4_800ep_augstrong_seed2: ep0: 37.08, ep20: 66.82, ep50: 75.30, ep100: 79.18, ep200: 81.96, ep400: 83.46, ep600: 84.58, ep800: 84.84

## Held-out J trajectory (VCS only)

- P89_vcs_a5_views4_800ep_augstrong_seed1: ep0: -0.9997, ep20: 0.6843, ep50: 0.7431, ep100: 0.7901, ep200: 0.8296, ep400: 0.8560, ep600: 0.8705, ep800: 0.8761
- P89_vcs_a5_views4_800ep_augstrong_seed2: ep0: -0.9997, ep20: 0.6827, ep50: 0.7402, ep100: 0.7982, ep200: 0.8246, ep400: 0.8591, ep600: 0.8716, ep800: 0.8768

## Final-epoch training objective values (epoch means)

- P89_simclr_views4_800ep_augstrong_seed0: J_raw null, R_binary null, nt_xent 2.3814, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P89_simclr_views4_800ep_augstrong_seed1: J_raw null, R_binary null, nt_xent 2.3837, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P89_simclr_views4_800ep_augstrong_seed2: J_raw null, R_binary null, nt_xent 2.3753, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P89_vcs_a5_views4_800ep_augstrong_seed1: J_raw 0.8973, R_binary 0.1027, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P89_vcs_a5_views4_800ep_augstrong_seed2: J_raw 0.8981, R_binary 0.1019, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P89_simclr_views4_800ep_augstrong_seed0 | 0.1951 | 1312 | 2625 | 32900 | 48 | 13 | 35840000 | None | 1013661@node53 |
| P89_simclr_views4_800ep_augstrong_seed1 | 0.1907 | 1342 | 2685 | 30445 | 45 | 13 | 35840000 | None | 1013662@node53 |
| P89_simclr_views4_800ep_augstrong_seed2 | 0.1115 | 2296 | 4592 | 15818 | 20 | 7 | 35840000 | None | 1013578@node59 |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 0.1138 | 2249 | 4497 | 16142 | 55 | 11 | 35840000 | 2 | 1013594@node59 |
| P89_vcs_a5_views4_800ep_augstrong_seed2 | 0.1134 | 2258 | 4516 | 16076 | 55 | 11 | 35840000 | 2 | 1013598@node61 |

## Provenance

- P89_simclr_views4_800ep_augstrong_seed0: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `9f3534fa7527048a5bc01b79f29a0d040687342f8f4371e58bd2a03967bc6390`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P89_simclr_views4_800ep_augstrong_seed1: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `7873c9405ecb1da11aa86b773b5d27f562ce9ba844c8f963b2334bfdd32dfc2b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P89_simclr_views4_800ep_augstrong_seed2: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `c7932b6b8766f6b1ab78e70dab3fbeb4cbf32a5e9ce1f169ae79a041cd723d99`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P89_vcs_a5_views4_800ep_augstrong_seed1: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `7b837489c64104498d3d04e410dbe6603a26d8d09c0a84fada414afc409ccf05`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P89_vcs_a5_views4_800ep_augstrong_seed2: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `c28f56cedf03cdcbd6b7923196eb51863a867b06234dcb90a724b9713413fa94`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 3 | 89.49 ± 0.36 | 42.23 ± 0.67 | 47.27 ± 0.55 | 88.71 ± 0.13 | 135.81 ± 2.96 | null |
| vcs_qmi K=8 | 2 | 87.60 ± 0.08 | 42.33 ± 0.92 | 45.27 ± 0.83 | 85.13 ± 0.41 | 103.50 ± 1.93 | 0.88 ± 0.00 |

## Coverage checks

- P89_simclr_views4_800ep_augstrong_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P89_simclr_views4_800ep_augstrong_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P89_simclr_views4_800ep_augstrong_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P89_vcs_a5_views4_800ep_augstrong_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P89_vcs_a5_views4_800ep_augstrong_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
