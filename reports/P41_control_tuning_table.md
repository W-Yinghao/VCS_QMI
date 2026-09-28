# P41_control_tuning — neutral results table (observed values only)

Generated 2026-09-28T01:31:14Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | simclr_matched | null | 0 | 800/800 | 88.38 | 87.12 | null | 149.07 | 12774 | 3644/4330 | COMPLETED |
| P41_simclr_b128_seed0 | simclr_matched | null | 0 | 200/200 | 86.36 | 83.96 | null | 92.13 | 3363 | 2215/3646 | COMPLETED |
| P41_simclr_tau0.1_seed0 | simclr_matched | null | 0 | 200/200 | 84.30 | 80.16 | null | 79.46 | 1989 | 2788/3518 | COMPLETED |
| P41_simclr_tau0.5_seed0 | simclr_matched | null | 0 | 200/200 | 86.36 | 83.16 | null | 57.95 | 1981 | 2788/3518 | COMPLETED |
| P41_simclr_views4_b128_100ep_seed0 | simclr_matched | null | 0 | 100/100 | 86.42 | 84.50 | null | 87.14 | 4551 | 2790/3518 | COMPLETED |
| P41_simclr_views4_seed0 | simclr_matched | null | 0 | 200/200 | 87.48 | 86.12 | null | 116.15 | 3961 | 5269/8140 | COMPLETED |
| P41_vicreg_800ep_seed0 | vicreg_matched_128 | null | 0 | 800/800 | 86.70 | 84.36 | null | 110.49 | 7997 | 2788/3516 | COMPLETED |
| P41_vicreg_b128_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.52 | 82.84 | null | 74.59 | 1935 | 1548/2224 | COMPLETED |
| P41_vicreg_cov0.1_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 83.36 | 78.08 | null | 24.55 | 1997 | 2788/3516 | COMPLETED |
| P41_vicreg_views4_b128_100ep_seed0 | vicreg_matched_128 | null | 0 | 100/100 | 87.24 | 83.72 | null | 74.28 | 3363 | 3645/4330 | COMPLETED |
| P41_vicreg_views4_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.64 | 84.14 | null | 98.43 | 8873 | 5269/8138 | COMPLETED |
| P41_vicreg_w10_10_1_seed0 | vicreg_matched_128 | null | 0 | 200/200 | 86.28 | 83.54 | null | 107.15 | 2004 | 2788/3516 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_simclr_b128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 200 |
| P41_simclr_tau0.1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_simclr_tau0.5_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_simclr_views4_b128_100ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_simclr_views4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P41_vicreg_b128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 200 |
| P41_vicreg_cov0.1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_views4_b128_100ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 128 | 0.001 | 100 |
| P41_vicreg_views4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |
| P41_vicreg_w10_10_1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | 41.98 | 88.38 | 46.40 | 36.58 | 87.12 | 50.54 | 2.94 | 149.07 | null |
| P41_simclr_b128_seed0 | 41.98 | 86.36 | 44.38 | 36.58 | 83.96 | 47.38 | 2.94 | 92.13 | null |
| P41_simclr_tau0.1_seed0 | 41.94 | 84.30 | 42.36 | 36.54 | 80.16 | 43.62 | 2.94 | 79.46 | null |
| P41_simclr_tau0.5_seed0 | 41.94 | 86.36 | 44.42 | 36.54 | 83.16 | 46.62 | 2.94 | 57.95 | null |
| P41_simclr_views4_b128_100ep_seed0 | 41.94 | 86.42 | 44.48 | 36.54 | 84.50 | 47.96 | 2.94 | 87.14 | null |
| P41_simclr_views4_seed0 | 41.94 | 87.48 | 45.54 | 36.54 | 86.12 | 49.58 | 2.94 | 116.15 | null |
| P41_vicreg_800ep_seed0 | 41.94 | 86.70 | 44.76 | 36.54 | 84.36 | 47.82 | 2.94 | 110.49 | null |
| P41_vicreg_b128_seed0 | 41.94 | 86.52 | 44.58 | 36.54 | 82.84 | 46.30 | 2.94 | 74.59 | null |
| P41_vicreg_cov0.1_seed0 | 41.94 | 83.36 | 41.42 | 36.54 | 78.08 | 41.54 | 2.94 | 24.55 | null |
| P41_vicreg_views4_b128_100ep_seed0 | 41.98 | 87.24 | 45.26 | 36.58 | 83.72 | 47.14 | 2.94 | 74.28 | null |
| P41_vicreg_views4_seed0 | 41.94 | 86.64 | 44.70 | 36.54 | 84.14 | 47.60 | 2.94 | 98.43 | null |
| P41_vicreg_w10_10_1_seed0 | 41.94 | 86.28 | 44.34 | 36.54 | 83.54 | 47.00 | 2.94 | 107.15 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P41_simclr_800ep_seed0: ep0: 36.58, ep20: 68.24, ep50: 77.18, ep100: 80.82, ep200: 84.90, ep400: 86.38, ep600: 86.98, ep800: 87.12
- P41_simclr_b128_seed0: ep0: 36.58, ep10: 63.44, ep20: 69.94, ep50: 77.76, ep100: 81.88, ep150: 83.52, ep200: 83.96
- P41_simclr_tau0.1_seed0: ep0: 36.54, ep10: 62.12, ep20: 67.48, ep50: 73.04, ep100: 77.62, ep150: 79.70, ep200: 80.16
- P41_simclr_tau0.5_seed0: ep0: 36.54, ep10: 58.88, ep20: 67.70, ep50: 75.50, ep100: 81.12, ep150: 82.96, ep200: 83.16
- P41_simclr_views4_b128_100ep_seed0: ep0: 36.54, ep10: 69.00, ep20: 74.70, ep50: 82.00, ep100: 84.50
- P41_simclr_views4_seed0: ep0: 36.54, ep10: 66.12, ep20: 74.00, ep50: 81.02, ep100: 84.72, ep150: 86.20, ep200: 86.12
- P41_vicreg_800ep_seed0: ep0: 36.54, ep20: 67.64, ep50: 76.16, ep100: 79.24, ep200: 82.32, ep400: 84.14, ep600: 84.32, ep800: 84.36
- P41_vicreg_b128_seed0: ep0: 36.54, ep10: 61.74, ep20: 69.88, ep50: 77.06, ep100: 81.06, ep150: 82.84, ep200: 82.84
- P41_vicreg_cov0.1_seed0: ep0: 36.54, ep10: 51.78, ep20: 60.48, ep50: 70.68, ep100: 75.80, ep150: 77.68, ep200: 78.08
- P41_vicreg_views4_b128_100ep_seed0: ep0: 36.58, ep10: 68.66, ep20: 75.34, ep50: 81.92, ep100: 83.72
- P41_vicreg_views4_seed0: ep0: 36.54, ep10: 64.70, ep20: 74.30, ep50: 80.14, ep100: 83.18, ep150: 83.90, ep200: 84.14
- P41_vicreg_w10_10_1_seed0: ep0: 36.54, ep10: 61.82, ep20: 68.76, ep50: 77.60, ep100: 81.54, ep150: 83.38, ep200: 83.54

## Held-out J trajectory (VCS only)


## Final-epoch training objective values (epoch means)

- P41_simclr_800ep_seed0: J_raw null, R_binary null, nt_xent 1.9396, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_b128_seed0: J_raw null, R_binary null, nt_xent 1.5510, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_tau0.1_seed0: J_raw null, R_binary null, nt_xent 0.3712, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_tau0.5_seed0: J_raw null, R_binary null, nt_xent 4.4735, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_b128_100ep_seed0: J_raw null, R_binary null, nt_xent 1.5437, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_simclr_views4_seed0: J_raw null, R_binary null, nt_xent 2.0265, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P41_vicreg_800ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.09391757002898625, 'vicreg_variance': 0.008313221570902637, 'vicreg_covariance': 1.3883616542816162}
- P41_vicreg_b128_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.15739956285878803, 'vicreg_variance': 0.03675391739420062, 'vicreg_covariance': 2.713556081820757}
- P41_vicreg_cov0.1_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.08672738528677396, 'vicreg_variance': 0.0053946533865694484, 'vicreg_covariance': 16.27707865033831}
- P41_vicreg_views4_b128_100ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.14858319550922452, 'vicreg_variance': 0.035261114653295435, 'vicreg_covariance': 2.7362947593047746}
- P41_vicreg_views4_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.11799566720213209, 'vicreg_variance': 0.012244413729224886, 'vicreg_covariance': 1.6777392305646623}
- P41_vicreg_w10_10_1_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.18148469941956658, 'vicreg_variance': 0.024802822140710695, 'vicreg_covariance': 0.84139078106199}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P41_simclr_800ep_seed0 | 0.0899 | 2849 | 5698 | 12774 | 35 | 13 | 35840000 | None | 1011789@nodesumo01 |
| P41_simclr_b128_seed0 | 0.0474 | 2699 | 5397 | 3363 | 30 | 13 | 8985600 | None | 1011788@nodesumo01 |
| P41_simclr_tau0.1_seed0 | 0.0558 | 4590 | 9179 | 1989 | 16 | 7 | 8960000 | None | 1011784@node58 |
| P41_simclr_tau0.5_seed0 | 0.0556 | 4600 | 9200 | 1981 | 16 | 7 | 8960000 | None | 1011785@node58 |
| P41_simclr_views4_b128_100ep_seed0 | 0.1291 | 991 | 1983 | 4551 | 22 | 10 | 4492800 | None | 1011787@node60 |
| P41_simclr_views4_seed0 | 0.1116 | 2294 | 4589 | 3961 | 16 | 7 | 8960000 | None | 1011786@node59 |
| P41_vicreg_800ep_seed0 | 0.0563 | 4550 | 9099 | 7997 | 18 | 13 | 35840000 | None | 1011887@node58 |
| P41_vicreg_b128_seed0 | 0.0273 | 4695 | 9390 | 1935 | 16 | 7 | 8985600 | None | 1011796@node59 |
| P41_vicreg_cov0.1_seed0 | 0.0562 | 4554 | 9109 | 1997 | 15 | 7 | 8960000 | None | 1011790@node58 |
| P41_vicreg_views4_b128_100ep_seed0 | 0.0950 | 1347 | 2694 | 3363 | 22 | 13 | 4492800 | None | 1011795@nodesumo01 |
| P41_vicreg_views4_seed0 | 0.2521 | 1015 | 2031 | 8873 | 31 | 9 | 8960000 | None | 1011794@node60 |
| P41_vicreg_w10_10_1_seed0 | 0.0564 | 4538 | 9077 | 2004 | 16 | 7 | 8960000 | None | 1011791@node58 |

## Provenance

- P41_simclr_800ep_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `cc3ed318fa8ebb7b6d52aec55a0c9d6833e715b83fda66c38f3973363240a5a2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_b128_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `24b5304c1f5a405d8eaee08ab8d521e56bb854b17b07708d71020193a26a25ad`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_tau0.1_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `122e0a3d8fe4ed56d011772bbc96b09d4dcef9c30c3c50eddff86f88f5953c60`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_tau0.5_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `f7b57c3de650df57e53d3b147c76b82950f94425493a6696e68ed9dcb3483b29`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_views4_b128_100ep_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `21fc8b01df0394b73af9497c4259beede2c6517ae2966417b11aa83d3ba1fc60`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_simclr_views4_seed0: commit `f0538462d339531804f8acc5053ea7c734124b5e` dirty=True, config `a4fc31ffbc5c982ca2fd650a40d141f8f30cf115d466b2c7ac0000be89c72187`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_800ep_seed0: commit `cff7290659d49a09291fd018dea26373bdcbd78f` dirty=True, config `6a574d430404e3d83fe6146a2514ce7beff9aff0a979a7f8f9a40aa18fe38488`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_b128_seed0: commit `92b5ba66f99982f6f4c6fcca7bc70cca41d0f9fd` dirty=True, config `74f32458f4ccbd75bf27c2ee771175dcac834e49ff40e58d96235a99fb1b2a64`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_cov0.1_seed0: commit `19703e87fe363f86092bf9aac0d188ee0d87143a` dirty=True, config `d9981a902d96453c93bf9080c143577b821058d7b8b119f602fa039fc810e3da`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_views4_b128_100ep_seed0: commit `92b5ba66f99982f6f4c6fcca7bc70cca41d0f9fd` dirty=True, config `c9585528aa122ce2f2f88edc5f5a1e0b7b65336d15bf1f0eb592a9d220807ce3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_views4_seed0: commit `19703e87fe363f86092bf9aac0d188ee0d87143a` dirty=True, config `562d650a2e58b7a22d232a60623d3b6ca9702935624cc5a95a33b2dfb612a321`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P41_vicreg_w10_10_1_seed0: commit `19703e87fe363f86092bf9aac0d188ee0d87143a` dirty=True, config `406f5e9f6cdfdc1497a80e72270f12900a070c611c9fd114781ecba5f73f50f3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 6 | 86.55 ± 1.37 | 41.95 ± 0.02 | 44.60 ± 1.36 | 84.17 ± 2.44 | 96.98 ± 31.72 | null |
| vicreg_matched_128 | 6 | 86.12 ± 1.39 | 41.95 ± 0.02 | 44.18 ± 1.38 | 82.78 ± 2.36 | 81.59 ± 32.03 | null |

## Coverage checks

- P41_simclr_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_simclr_b128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_tau0.1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_tau0.5_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_simclr_views4_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_simclr_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P41_vicreg_b128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_cov0.1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_views4_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P41_vicreg_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P41_vicreg_w10_10_1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
