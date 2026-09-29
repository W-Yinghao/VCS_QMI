# P91_cifar100 — neutral results table (observed values only)

Generated 2026-09-29T14:18:03Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | simclr_matched | null | 0 | 200/200 | 59.74 | 54.70 | null | 141.23 | 9583 | 5268/8028 | COMPLETED |
| P91_c100_simclr_views4_200ep_seed1 | simclr_matched | null | 1 | 200/200 | 59.30 | 54.02 | null | 141.31 | 4974 | 5268/8028 | COMPLETED |
| P91_c100_simclr_views4_200ep_seed2 | simclr_matched | null | 2 | 200/200 | 59.22 | 54.42 | null | 140.14 | 4966 | 5268/8028 | COMPLETED |
| P91_c100_vcs_a5_views4_200ep_seed0 | vcs_qmi | 8 | 0 | 200/200 | 57.74 | 48.44 | 0.9614±0.0012 | 79.93 | 13445 | 7568/12672 | COMPLETED |
| P91_c100_vcs_a5_views4_200ep_seed1 | vcs_qmi | 8 | 1 | 200/200 | 58.16 | 48.52 | 0.9617±0.0006 | 80.33 | 11555 | 5268/8028 | COMPLETED |
| P91_c100_vcs_a5_views4_200ep_seed2 | vcs_qmi | 8 | 2 | 200/200 | 57.98 | 49.06 | 0.9614±0.0010 | 79.58 | 10678 | 5268/8028 | COMPLETED |
| P91_c100_vicreg_views4_200ep_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 56.80 | 48.72 | null | 112.16 | 8880 | 5269/8138 | COMPLETED |
| P91_c100_vicreg_views4_200ep_seed1 | vicreg_matched_128 | null | 1 | 200/200 | 56.50 | 49.02 | null | 113.17 | 6970 | 7066/10146 | COMPLETED |
| P91_c100_vicreg_views4_200ep_seed2 | vicreg_matched_128 | null | 2 | 200/200 | 55.98 | 47.94 | null | 111.26 | 6824 | 7066/10146 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_simclr_views4_200ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_simclr_views4_200ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_vcs_a5_views4_200ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P91_c100_vcs_a5_views4_200ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P91_c100_vcs_a5_views4_200ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P91_c100_vicreg_views4_200ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_vicreg_views4_200ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P91_c100_vicreg_views4_200ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | 20.12 | 59.74 | 39.62 | 14.00 | 54.70 | 40.70 | 2.74 | 141.23 | null |
| P91_c100_simclr_views4_200ep_seed1 | 19.34 | 59.30 | 39.96 | 14.72 | 54.02 | 39.30 | 2.89 | 141.31 | null |
| P91_c100_simclr_views4_200ep_seed2 | 20.04 | 59.22 | 39.18 | 13.78 | 54.42 | 40.64 | 2.92 | 140.14 | null |
| P91_c100_vcs_a5_views4_200ep_seed0 | 20.12 | 57.74 | 37.62 | 14.02 | 48.44 | 34.42 | 2.74 | 79.93 | -0.9998 |
| P91_c100_vcs_a5_views4_200ep_seed1 | 19.34 | 58.16 | 38.82 | 14.72 | 48.52 | 33.80 | 2.89 | 80.33 | -0.9998 |
| P91_c100_vcs_a5_views4_200ep_seed2 | 20.04 | 57.98 | 37.94 | 13.78 | 49.06 | 35.28 | 2.92 | 79.58 | -0.9998 |
| P91_c100_vicreg_views4_200ep_seed0 | 20.12 | 56.80 | 36.68 | 14.00 | 48.72 | 34.72 | 2.74 | 112.16 | null |
| P91_c100_vicreg_views4_200ep_seed1 | 19.34 | 56.50 | 37.16 | 14.72 | 49.02 | 34.30 | 2.89 | 113.17 | null |
| P91_c100_vicreg_views4_200ep_seed2 | 20.08 | 55.98 | 35.90 | 13.82 | 47.94 | 34.12 | 2.92 | 111.26 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P91_c100_simclr_views4_200ep_seed0: ep0: 14.02, ep10: 33.02, ep20: 38.04, ep50: 47.32, ep100: 52.46, ep150: 54.02, ep200: 54.70
- P91_c100_simclr_views4_200ep_seed1: ep0: 14.72, ep10: 32.74, ep20: 38.38, ep50: 46.76, ep100: 52.46, ep150: 54.08, ep200: 54.02
- P91_c100_simclr_views4_200ep_seed2: ep0: 13.82, ep10: 32.68, ep20: 39.02, ep50: 46.94, ep100: 52.60, ep150: 53.94, ep200: 54.42
- P91_c100_vcs_a5_views4_200ep_seed0: ep0: 14.02, ep10: 26.48, ep20: 34.20, ep50: 39.90, ep100: 45.40, ep150: 47.94, ep200: 48.44
- P91_c100_vcs_a5_views4_200ep_seed1: ep0: 14.72, ep10: 26.34, ep20: 34.10, ep50: 41.48, ep100: 45.18, ep150: 47.96, ep200: 48.52
- P91_c100_vcs_a5_views4_200ep_seed2: ep0: 13.82, ep10: 26.48, ep20: 35.02, ep50: 40.70, ep100: 45.64, ep150: 48.04, ep200: 49.06
- P91_c100_vicreg_views4_200ep_seed0: ep0: 14.00, ep10: 27.54, ep20: 34.34, ep50: 43.28, ep100: 47.50, ep150: 48.22, ep200: 48.72
- P91_c100_vicreg_views4_200ep_seed1: ep0: 14.72, ep10: 27.66, ep20: 33.54, ep50: 43.06, ep100: 46.96, ep150: 48.84, ep200: 49.02
- P91_c100_vicreg_views4_200ep_seed2: ep0: 13.82, ep10: 27.38, ep20: 34.80, ep50: 43.42, ep100: 46.94, ep150: 47.78, ep200: 47.94

## Held-out J trajectory (VCS only)

- P91_c100_vcs_a5_views4_200ep_seed0: ep0: -0.9998, ep10: 0.7279, ep20: 0.8713, ep50: 0.9183, ep100: 0.9476, ep150: 0.9579, ep200: 0.9614
- P91_c100_vcs_a5_views4_200ep_seed1: ep0: -0.9998, ep10: 0.7421, ep20: 0.8622, ep50: 0.9200, ep100: 0.9449, ep150: 0.9576, ep200: 0.9617
- P91_c100_vcs_a5_views4_200ep_seed2: ep0: -0.9998, ep10: 0.7005, ep20: 0.8751, ep50: 0.9189, ep100: 0.9469, ep150: 0.9575, ep200: 0.9614

## Final-epoch training objective values (epoch means)

- P91_c100_simclr_views4_200ep_seed0: J_raw null, R_binary null, nt_xent 2.0719, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_200ep_seed1: J_raw null, R_binary null, nt_xent 2.0742, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_simclr_views4_200ep_seed2: J_raw null, R_binary null, nt_xent 2.0731, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_200ep_seed0: J_raw 0.9703, R_binary 0.0297, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_200ep_seed1: J_raw 0.9700, R_binary 0.0300, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vcs_a5_views4_200ep_seed2: J_raw 0.9701, R_binary 0.0299, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P91_c100_vicreg_views4_200ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11261215107781547, 'vicreg_variance': 0.013424802695933198, 'vicreg_covariance': 1.9636033739362444}
- P91_c100_vicreg_views4_200ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11382313042879105, 'vicreg_variance': 0.013560276909598282, 'vicreg_covariance': 1.9750301817485265}
- P91_c100_vicreg_views4_200ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11541412885699953, 'vicreg_variance': 0.013514666063045817, 'vicreg_covariance': 2.0068757779257638}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P91_c100_simclr_views4_200ep_seed0 | 0.2471 | 1036 | 2072 | 9583 | 41 | 10 | 8960000 | None | 1013676@node61 |
| P91_c100_simclr_views4_200ep_seed1 | 0.1115 | 2296 | 4593 | 4974 | 33 | 7 | 8960000 | None | 1013677@node59 |
| P91_c100_simclr_views4_200ep_seed2 | 0.1113 | 2301 | 4601 | 4966 | 34 | 7 | 8960000 | None | 1013678@node61 |
| P91_c100_vcs_a5_views4_200ep_seed0 | 0.3801 | 674 | 1347 | 13445 | 146 | 36 | 8960000 | 2 | 1013431@node54 |
| P91_c100_vcs_a5_views4_200ep_seed1 | 0.2490 | 1028 | 2056 | 11555 | 123 | 17 | 8960000 | 2 | 1013675@node61 |
| P91_c100_vcs_a5_views4_200ep_seed2 | 0.2495 | 1026 | 2053 | 10678 | 111 | 17 | 8960000 | 2 | 1013674@node61 |
| P91_c100_vicreg_views4_200ep_seed0 | 0.2521 | 1015 | 2031 | 8880 | 33 | 16 | 8960000 | None | 1013608@node61 |
| P91_c100_vicreg_views4_200ep_seed1 | 0.1966 | 1302 | 2605 | 6970 | 30 | 13 | 8960000 | None | 1013613@node53 |
| P91_c100_vicreg_views4_200ep_seed2 | 0.1925 | 1330 | 2660 | 6824 | 29 | 13 | 8960000 | None | 1013623@node53 |

## Provenance

- P91_c100_simclr_views4_200ep_seed0: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `5f16e1fb4c570a3f76ab1671662482c0fad51c5bb20d9028ee9fa1767aef65cc`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_simclr_views4_200ep_seed1: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `bc06897ca9600316cebb6b56f27dea3defbce78fc875aab645028e933c9821ea`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_simclr_views4_200ep_seed2: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `4d349abd525ea662a0ab2c4f58633fd05727fa68bd11cc2669e570388dae14f5`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_vcs_a5_views4_200ep_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `3446b88e90c7eb90a5e10808ce7b09d0b81bc76dc83745fb811d9196b8c15a04`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_vcs_a5_views4_200ep_seed1: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `47f081f63fd6a303407087e7ba42d72fa1d8f14f38bd18340114259cf9f8d038`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_vcs_a5_views4_200ep_seed2: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `bc5ed3d400460e565dec8826c4c6fd181cca288aecca4709801d249d618e1233`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P91_c100_vicreg_views4_200ep_seed0: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `04f8b0d3298e312239306ce98d2f7d6e6ea0e9edea283b4b67a39fda679957a4`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P91_c100_vicreg_views4_200ep_seed1: commit `16eb8b33f2e2c7c675ea20ab10ef0184ae041292` dirty=True, config `9e20c965712cbe6ca512258187c9211b3a9011f91fc1c3f7dce764b472e0e461`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P91_c100_vicreg_views4_200ep_seed2: commit `db99964976a8743f7a186803b90c958096bdff55` dirty=True, config `801494e160b572fef66ab92578561848eb19aff2ddbf358ff8ec06bc5a7e2bd6`, split `96dbd6d427807a0ef94fbcbbff9277bd4c8ebca3f5f7bb44e3246c0a45ad5430`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 3 | 59.42 ± 0.28 | 19.83 ± 0.43 | 39.59 ± 0.39 | 54.38 ± 0.34 | 140.89 ± 0.66 | null |
| vcs_qmi K=8 | 3 | 57.96 ± 0.21 | 19.83 ± 0.43 | 38.13 ± 0.62 | 48.67 ± 0.34 | 79.95 ± 0.38 | 0.96 ± 0.00 |
| vicreg_matched_128 | 3 | 56.43 ± 0.41 | 19.85 ± 0.44 | 36.58 ± 0.64 | 48.56 ± 0.56 | 112.20 ± 0.95 | null |

## Coverage checks

- P91_c100_simclr_views4_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_simclr_views4_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_simclr_views4_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vcs_a5_views4_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P91_c100_vicreg_views4_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
