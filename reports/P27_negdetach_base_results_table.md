# P26_vcs_negdetach_base — neutral results table (observed values only)

Generated 2026-09-25T22:54:16Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P26_vcs_a5_fixed_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.54 | 77.00 | 0.9309±0.0015 | 46.86 | 5607 | 3683/4842 | COMPLETED |
| P26_vcs_a5_learn_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.90 | 77.00 | 0.9415±0.0018 | 44.99 | 6227 | 3683/4842 | COMPLETED |
| P26_vcs_b0_calib_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.12 | 73.04 | 0.9114±0.0021 | 22.76 | 5581 | 3683/4842 | COMPLETED |
| P26_vcs_base_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 84.64 | 81.44 | 0.9563±0.0012 | 61.10 | 22251 | 3683/4842 | COMPLETED |
| P26_vcs_base_seed1_seed1 | vcs_qmi | 8 | 1 | 200/200 | 80.58 | 74.26 | 0.9274±0.0020 | 29.59 | 6206 | 3683/4842 | COMPLETED |
| P26_vcs_base_seed2_seed2 | vcs_qmi | 8 | 2 | 200/200 | 80.72 | 74.34 | 0.9269±0.0018 | 30.62 | 5657 | 3683/4842 | COMPLETED |
| P26_vcs_interact_only_seed0 | vcs_qmi | 8 | 0 | 30/200 | null | null | null | null | 934 | 3684/4846 | STOPPED_BUDGET COLLAPSE_SUSPECTED |
| P26_vcs_k255_seed0 | vcs_qmi | 255 | 0 | 200/200 | 80.82 | 75.58 | 0.9348±0.0015 | 33.81 | 6260 | 3683/4844 | COMPLETED |
| P26_vcs_proj_outBN_seed0 | vcs_qmi | 8 | 0 | 200/200 | 78.52 | 72.68 | 0.9158±0.0027 | 4.49 | 6230 | 3682/4840 | COMPLETED |
| P26_vcs_shared_metric_seed0 | vcs_qmi | 8 | 0 | 200/200 | 75.28 | 69.28 | 0.9062±0.0028 | 1.32 | 5733 | 3683/4846 | COMPLETED COLLAPSE_SUSPECTED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P26_vcs_a5_fixed_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_a5_learn_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_b0_calib_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_base_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P26_vcs_base_seed1_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_base_seed2_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_interact_only_seed0 | 8 | [256, 256] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | InteractOnlyCritic | 256 | 0.001 | 200 |
| P26_vcs_k255_seed0 | 255 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_proj_outBN_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P26_vcs_shared_metric_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | SharedMetricCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P26_vcs_a5_fixed_seed0 | 41.78 | 81.54 | 39.76 | 36.58 | 77.00 | 40.42 | 2.94 | 46.86 | -0.9998 |
| P26_vcs_a5_learn_seed0 | 41.78 | 81.90 | 40.12 | 36.58 | 77.00 | 40.42 | 2.94 | 44.99 | -0.9998 |
| P26_vcs_b0_calib_seed0 | 41.78 | 80.12 | 38.34 | 36.58 | 73.04 | 36.46 | 2.94 | 22.76 | -0.2557 |
| P26_vcs_base_800ep_seed0 | 41.78 | 84.64 | 42.86 | 36.58 | 81.44 | 44.86 | 2.94 | 61.10 | -0.5686 |
| P26_vcs_base_seed1_seed1 | 41.60 | 80.58 | 38.98 | 37.42 | 74.26 | 36.84 | 3.22 | 29.59 | -0.5658 |
| P26_vcs_base_seed2_seed2 | 43.02 | 80.72 | 37.70 | 37.08 | 74.34 | 37.26 | 3.22 | 30.62 | -0.5639 |
| P26_vcs_interact_only_seed0 | null | null | null | null | null | null | null | null | null |
| P26_vcs_k255_seed0 | 41.78 | 80.82 | 39.04 | 36.58 | 75.58 | 39.00 | 2.94 | 33.81 | -0.5686 |
| P26_vcs_proj_outBN_seed0 | 41.78 | 78.52 | 36.74 | 36.58 | 72.68 | 36.10 | 2.94 | 4.49 | -0.5701 |
| P26_vcs_shared_metric_seed0 | 41.78 | 75.28 | 33.50 | 36.58 | 69.28 | 32.70 | 2.94 | 1.32 | -0.5686 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P26_vcs_a5_fixed_seed0: ep0: 36.58, ep10: 49.92, ep20: 58.34, ep50: 68.64, ep100: 74.18, ep150: 76.54, ep200: 77.00
- P26_vcs_a5_learn_seed0: ep0: 36.58, ep10: 51.06, ep20: 61.08, ep50: 68.74, ep100: 73.50, ep150: 76.46, ep200: 77.00
- P26_vcs_b0_calib_seed0: ep0: 36.58, ep10: 48.50, ep20: 54.90, ep50: 65.64, ep100: 70.52, ep150: 72.34, ep200: 73.04
- P26_vcs_base_800ep_seed0: ep0: 36.58, ep20: 60.68, ep50: 68.54, ep100: 71.20, ep200: 75.98, ep400: 79.36, ep600: 80.96, ep800: 81.44
- P26_vcs_base_seed1_seed1: ep0: 37.42, ep10: 51.88, ep20: 61.26, ep50: 67.48, ep100: 71.50, ep150: 73.76, ep200: 74.26
- P26_vcs_base_seed2_seed2: ep0: 37.08, ep10: 52.86, ep20: 61.22, ep50: 67.92, ep100: 71.94, ep150: 74.28, ep200: 74.34
- P26_vcs_interact_only_seed0: ep0: 36.58, ep10: 32.34, ep20: 20.54
- P26_vcs_k255_seed0: ep0: 36.58, ep10: 55.62, ep20: 61.38, ep50: 69.34, ep100: 72.88, ep150: 75.10, ep200: 75.58
- P26_vcs_proj_outBN_seed0: ep0: 36.58, ep10: 55.52, ep20: 61.38, ep50: 66.84, ep100: 70.78, ep150: 72.96, ep200: 72.68
- P26_vcs_shared_metric_seed0: ep0: 36.58, ep10: 52.62, ep20: 59.96, ep50: 66.34, ep100: 68.78, ep150: 70.10, ep200: 69.28

## Held-out J trajectory (VCS only)

- P26_vcs_a5_fixed_seed0: ep0: -0.9998, ep10: 0.5720, ep20: 0.7751, ep50: 0.8485, ep100: 0.9020, ep150: 0.9241, ep200: 0.9309
- P26_vcs_a5_learn_seed0: ep0: -0.9998, ep10: 0.5969, ep20: 0.7883, ep50: 0.8661, ep100: 0.9141, ep150: 0.9353, ep200: 0.9415
- P26_vcs_b0_calib_seed0: ep0: -0.2557, ep10: 0.6013, ep20: 0.7232, ep50: 0.8321, ep100: 0.8814, ep150: 0.9033, ep200: 0.9114
- P26_vcs_base_800ep_seed0: ep0: -0.5686, ep20: 0.7776, ep50: 0.8552, ep100: 0.8942, ep200: 0.9243, ep400: 0.9430, ep600: 0.9535, ep800: 0.9563
- P26_vcs_base_seed1_seed1: ep0: -0.5658, ep10: 0.6426, ep20: 0.7749, ep50: 0.8623, ep100: 0.9033, ep150: 0.9186, ep200: 0.9274
- P26_vcs_base_seed2_seed2: ep0: -0.5639, ep10: 0.6320, ep20: 0.7749, ep50: 0.8640, ep100: 0.8960, ep150: 0.9205, ep200: 0.9269
- P26_vcs_interact_only_seed0: ep0: 0.0001, ep10: 0.0347, ep20: -0.0205
- P26_vcs_k255_seed0: ep0: -0.5686, ep10: 0.6357, ep20: 0.7810, ep50: 0.8678, ep100: 0.9099, ep150: 0.9283, ep200: 0.9348
- P26_vcs_proj_outBN_seed0: ep0: -0.5701, ep10: 0.6538, ep20: 0.7796, ep50: 0.8747, ep100: 0.8975, ep150: 0.9117, ep200: 0.9158
- P26_vcs_shared_metric_seed0: ep0: -0.5686, ep10: 0.6019, ep20: 0.7583, ep50: 0.8443, ep100: 0.8734, ep150: 0.8941, ep200: 0.9062

## Final-epoch training objective values (epoch means)

- P26_vcs_a5_fixed_seed0: J_raw 0.9362, R_binary 0.0638, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_a5_learn_seed0: J_raw 0.9462, R_binary 0.0538, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_b0_calib_seed0: J_raw 0.9159, R_binary 0.0841, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_base_800ep_seed0: J_raw 0.9645, R_binary 0.0355, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_base_seed1_seed1: J_raw 0.9329, R_binary 0.0671, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_base_seed2_seed2: J_raw 0.9315, R_binary 0.0685, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_interact_only_seed0: J_raw -0.0116, R_binary 1.0116, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_k255_seed0: J_raw 0.9395, R_binary 0.0605, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_proj_outBN_seed0: J_raw 0.9194, R_binary 0.0806, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P26_vcs_shared_metric_seed0: J_raw 0.9070, R_binary 0.0930, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P26_vcs_a5_fixed_seed0 | 0.1581 | 1619 | 3238 | 5607 | 138 | 30 | 8960000 | 2 | 1009001@nodeaudible01 |
| P26_vcs_a5_learn_seed0 | 0.1758 | 1456 | 2913 | 6227 | 147 | 37 | 8960000 | 2 | 1009000@node07 |
| P26_vcs_b0_calib_seed0 | 0.1575 | 1625 | 3251 | 5581 | 135 | 30 | 8960000 | 2 | 1008999@nodeaudible01 |
| P26_vcs_base_800ep_seed0 | 0.1569 | 1632 | 3264 | 22251 | 157 | 29 | 35840000 | 2 | 1009006@nodeaudible01 |
| P26_vcs_base_seed1_seed1 | 0.1751 | 1462 | 2923 | 6206 | 148 | 35 | 8960000 | 2 | 1008997@node54 |
| P26_vcs_base_seed2_seed2 | 0.1598 | 1602 | 3205 | 5657 | 132 | 30 | 8960000 | 2 | 1008998@nodeaudible01 |
| P26_vcs_interact_only_seed0 | 0.1758 | 1457 | 2913 | 934 | 61 | null | 1358080 | 131841 | 1009003@node54 |
| P26_vcs_k255_seed0 | 0.1767 | 1449 | 2898 | 6260 | 145 | 35 | 8960000 | 2 | 1009005@node54 |
| P26_vcs_proj_outBN_seed0 | 0.1758 | 1456 | 2912 | 6230 | 150 | 36 | 8960000 | 2 | 1009002@node56 |
| P26_vcs_shared_metric_seed0 | 0.1619 | 1581 | 3163 | 5733 | 140 | 30 | 8960000 | 16386 | 1009004@nodeaudible01 |

## Provenance

- P26_vcs_a5_fixed_seed0: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `3cb5183c22373c9cf1ac17917763cc0ce56f3002e3a48a4686b47bdb06df8474`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_a5_learn_seed0: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `e2bc01de281bec7ff2bd05c1c0635479887475973911abc7911d81d1efde7b67`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_b0_calib_seed0: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `e82adf98e17e88c1c204d3b35651947d83d2a73bec7f132e34820816fb0b49e6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_base_800ep_seed0: commit `2592e1081fcf4a2546123b7a9c9a7f296e312eb6` dirty=True, config `f7ef4af556507fb950eab1fcaad4cc26a52148a7a2e445794c79770f8dcb7cad`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_base_seed1_seed1: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `b48f5b9dd8b96c076feac7bc95181f72c12a23a3ef46038a81e9345f90a91610`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P26_vcs_base_seed2_seed2: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `f1c7836ecef5b73180c57842f993053f807c98be744c5713c25a4f14f873993d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P26_vcs_interact_only_seed0: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `d51d3e4fe3418af05176e396398c681eef71dab0ac2fdcab323eb24862957d35`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_k255_seed0: commit `2592e1081fcf4a2546123b7a9c9a7f296e312eb6` dirty=True, config `f36b1cd6e831d87b1beb81f2935c0e5bd860cc021164cbab3db6186c0aa21456`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_proj_outBN_seed0: commit `e25c739429fafbf7f266e32f1b66dd2d4106fb42` dirty=True, config `6508f5c627527854f7b088f613281000053177983e94fb08832e707ba4c57b5a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P26_vcs_shared_metric_seed0: commit `2592e1081fcf4a2546123b7a9c9a7f296e312eb6` dirty=True, config `72136089d1b80e652cb18466802f56789fdcf48dad4538c3115f261f9d8387bf`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=255 | 1 | 80.82 | 41.78 | 39.04 | 75.58 | 33.81 | 0.93 |
| vcs_qmi K=8 | 8 | 80.41 ± 2.71 | 41.91 ± 0.45 | 38.50 ± 2.73 | 74.88 ± 3.63 | 30.22 ± 20.71 | 0.93 ± 0.02 |

## Coverage checks

- P26_vcs_a5_fixed_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_a5_learn_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_b0_calib_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_base_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P26_vcs_base_seed1_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_base_seed2_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_interact_only_seed0: epoch0 eval False, final eval False (None), status STOPPED_BUDGET, failure stopped: signal 15; last completed epoch 30 is in last.pt; mid-epoch progress discarded
- P26_vcs_k255_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_proj_outBN_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P26_vcs_shared_metric_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
