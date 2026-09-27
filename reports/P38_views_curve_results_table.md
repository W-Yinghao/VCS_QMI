# P37_vcs_views_curve — neutral results table (observed values only)

Generated 2026-09-27T06:30:53Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P37_vcs_a5_views4_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 81.88 | 77.74 | 0.9513±0.0011 | 55.59 | 6847 | 7568/12672 | COMPLETED |
| P37_vcs_a5_views8_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 83.40 | 80.44 | 0.9649±0.0012 | 74.34 | 12229 | 14898/25962 | COMPLETED |
| P37_vcs_a5_views8_200ep_seed0 | vcs_qmi | 8 | 0 | 200/200 | 86.28 | 83.66 | 0.9721±0.0016 | 103.70 | 27674 | 14898/25962 | COMPLETED |
| P37_vcs_a5_views8_200ep_seed1 | vcs_qmi | 8 | 1 | 200/200 | 85.58 | 83.04 | 0.9716±0.0015 | 103.07 | 24417 | 14898/25962 | COMPLETED |
| P37_vcs_a5_views8_200ep_seed2 | vcs_qmi | 8 | 2 | 200/200 | 85.98 | 83.72 | 0.9720±0.0011 | 102.92 | 27382 | 14898/25962 | COMPLETED |
| P37_vcs_a5_views8_50ep_seed0 | vcs_qmi | 8 | 0 | 50/50 | 80.60 | 76.08 | 0.9535±0.0009 | 59.01 | 6172 | 14898/25962 | COMPLETED |
| P37_vcs_a5_views8_b128_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 84.60 | 81.62 | 0.9661±0.0010 | 78.31 | 14652 | 7572/12674 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P37_vcs_a5_views4_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |
| P37_vcs_a5_views8_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |
| P37_vcs_a5_views8_200ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P37_vcs_a5_views8_200ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P37_vcs_a5_views8_200ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P37_vcs_a5_views8_50ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 50 |
| P37_vcs_a5_views8_b128_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P37_vcs_a5_views4_100ep_seed0 | 41.78 | 81.88 | 40.10 | 36.58 | 77.74 | 41.16 | 2.94 | 55.59 | -0.9998 |
| P37_vcs_a5_views8_100ep_seed0 | 41.78 | 83.40 | 41.62 | 36.58 | 80.44 | 43.86 | 2.94 | 74.34 | -0.9998 |
| P37_vcs_a5_views8_200ep_seed0 | 41.78 | 86.28 | 44.50 | 36.58 | 83.66 | 47.08 | 2.94 | 103.70 | -0.9998 |
| P37_vcs_a5_views8_200ep_seed1 | 41.60 | 85.58 | 43.98 | 37.42 | 83.04 | 45.62 | 3.22 | 103.07 | -0.9998 |
| P37_vcs_a5_views8_200ep_seed2 | 43.02 | 85.98 | 42.96 | 37.08 | 83.72 | 46.64 | 3.22 | 102.92 | -0.9998 |
| P37_vcs_a5_views8_50ep_seed0 | 41.78 | 80.60 | 38.82 | 36.58 | 76.08 | 39.50 | 2.94 | 59.01 | -0.9998 |
| P37_vcs_a5_views8_b128_100ep_seed0 | 41.78 | 84.60 | 42.82 | 36.58 | 81.62 | 45.04 | 2.94 | 78.31 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P37_vcs_a5_views4_100ep_seed0: ep0: 36.58, ep10: 58.16, ep20: 67.90, ep50: 74.00, ep100: 77.74
- P37_vcs_a5_views8_100ep_seed0: ep0: 36.58, ep10: 62.32, ep20: 71.68, ep50: 78.06, ep100: 80.44
- P37_vcs_a5_views8_200ep_seed0: ep0: 36.58, ep10: 62.40, ep20: 71.16, ep50: 77.76, ep100: 81.62, ep150: 83.28, ep200: 83.66
- P37_vcs_a5_views8_200ep_seed1: ep0: 37.42, ep10: 61.72, ep20: 71.26, ep50: 76.86, ep100: 81.42, ep150: 82.58, ep200: 83.04
- P37_vcs_a5_views8_200ep_seed2: ep0: 37.08, ep10: 62.16, ep20: 71.50, ep50: 76.96, ep100: 81.34, ep150: 83.48, ep200: 83.72
- P37_vcs_a5_views8_50ep_seed0: ep0: 36.58, ep10: 62.10, ep20: 71.22, ep50: 76.08
- P37_vcs_a5_views8_b128_100ep_seed0: ep0: 36.58, ep10: 66.18, ep20: 73.48, ep50: 79.32, ep100: 81.62

## Held-out J trajectory (VCS only)

- P37_vcs_a5_views4_100ep_seed0: ep0: -0.9998, ep10: 0.6779, ep20: 0.8818, ep50: 0.9210, ep100: 0.9513
- P37_vcs_a5_views8_100ep_seed0: ep0: -0.9998, ep10: 0.7678, ep20: 0.9026, ep50: 0.9454, ep100: 0.9649
- P37_vcs_a5_views8_200ep_seed0: ep0: -0.9998, ep10: 0.7549, ep20: 0.9038, ep50: 0.9401, ep100: 0.9618, ep150: 0.9696, ep200: 0.9721
- P37_vcs_a5_views8_200ep_seed1: ep0: -0.9998, ep10: 0.7840, ep20: 0.9012, ep50: 0.9402, ep100: 0.9604, ep150: 0.9692, ep200: 0.9716
- P37_vcs_a5_views8_200ep_seed2: ep0: -0.9998, ep10: 0.7889, ep20: 0.9011, ep50: 0.9396, ep100: 0.9609, ep150: 0.9693, ep200: 0.9720
- P37_vcs_a5_views8_50ep_seed0: ep0: -0.9998, ep10: 0.7675, ep20: 0.9023, ep50: 0.9535
- P37_vcs_a5_views8_b128_100ep_seed0: ep0: -0.9998, ep10: 0.8357, ep20: 0.9133, ep50: 0.9479, ep100: 0.9661

## Final-epoch training objective values (epoch means)

- P37_vcs_a5_views4_100ep_seed0: J_raw 0.9563, R_binary 0.0437, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P37_vcs_a5_views8_100ep_seed0: J_raw 0.9715, R_binary 0.0285, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P37_vcs_a5_views8_200ep_seed0: J_raw 0.9809, R_binary 0.0191, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P37_vcs_a5_views8_200ep_seed1: J_raw 0.9812, R_binary 0.0188, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P37_vcs_a5_views8_200ep_seed2: J_raw 0.9811, R_binary 0.0189, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P37_vcs_a5_views8_50ep_seed0: J_raw 0.9584, R_binary 0.0416, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P37_vcs_a5_views8_b128_100ep_seed0: J_raw 0.9735, R_binary 0.0265, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P37_vcs_a5_views4_100ep_seed0 | 0.3865 | 662 | 1325 | 6847 | 115 | 37 | 4480000 | 2 | 1009239@node06 |
| P37_vcs_a5_views8_100ep_seed0 | 0.6925 | 370 | 739 | 12229 | 90 | 29 | 4480000 | 2 | 1009241@nodeaudible01 |
| P37_vcs_a5_views8_200ep_seed0 | 0.7830 | 327 | 654 | 27674 | 153 | 36 | 8960000 | 2 | 1009475@node02 |
| P37_vcs_a5_views8_200ep_seed1 | 0.6907 | 371 | 741 | 24417 | 133 | 29 | 8960000 | 2 | 1010233@nodeaudible01 |
| P37_vcs_a5_views8_200ep_seed2 | 0.7744 | 331 | 661 | 27382 | 150 | 35 | 8960000 | 2 | 1010234@node54 |
| P37_vcs_a5_views8_50ep_seed0 | 0.6995 | 366 | 732 | 6172 | 74 | 29 | 2240000 | 2 | 1009240@nodeaudible01 |
| P37_vcs_a5_views8_b128_100ep_seed0 | 0.4152 | 308 | 617 | 14652 | 106 | 35 | 4492800 | 2 | 1009476@node02 |

## Provenance

- P37_vcs_a5_views4_100ep_seed0: commit `acaf5017851b93777dc8029a61beb87ede071321` dirty=True, config `3f9781fc31feaca8cc9398311c3aa7716b9aa11f9bf41d793a0013089e820f2e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P37_vcs_a5_views8_100ep_seed0: commit `45badb3d47986f6d0048e5644aab758cf68dab83` dirty=True, config `bb8f237e5166a760b3494d545bf39a2ce8cc0f4ab159b824ade6d8b114897540`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P37_vcs_a5_views8_200ep_seed0: commit `cddd18ed4778eee3c4945c9e12d5bc8d7ce86b33` dirty=False, config `a87ca0f24b0f149ebd2f5f3edc9d0ffed2c43a024ce61e227f50fdec61466b51`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P37_vcs_a5_views8_200ep_seed1: commit `12de250b921e3ba75e9de27d5c566d5e4b67a520` dirty=False, config `77566fc1857760f3c11dc05213debf53f046ba308b1b298792a8d96da3594080`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P37_vcs_a5_views8_200ep_seed2: commit `12de250b921e3ba75e9de27d5c566d5e4b67a520` dirty=True, config `5b49557c0b00b27f15113ecfbccd4f6851c50408399d6cf88463ec3d69c34311`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P37_vcs_a5_views8_50ep_seed0: commit `acaf5017851b93777dc8029a61beb87ede071321` dirty=True, config `ba8474fdf6d2d92ad06a35943f193487fa54b242761e4638f9c2b96bc01bad98`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P37_vcs_a5_views8_b128_100ep_seed0: commit `cddd18ed4778eee3c4945c9e12d5bc8d7ce86b33` dirty=False, config `8471d9ce03f678a4e69314b3b010bbbb01238ab32aa6bbb0f62413c83e17cf1e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 7 | 84.05 ± 2.17 | 41.93 ± 0.48 | 42.11 ± 2.06 | 80.90 ± 3.00 | 82.42 ± 21.01 | 0.96 ± 0.01 |

## Coverage checks

- P37_vcs_a5_views4_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P37_vcs_a5_views8_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P37_vcs_a5_views8_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P37_vcs_a5_views8_200ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P37_vcs_a5_views8_200ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P37_vcs_a5_views8_50ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_050.json), status COMPLETED, failure None
- P37_vcs_a5_views8_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
