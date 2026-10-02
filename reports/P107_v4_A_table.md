# P107_v4_A_batch — neutral results table (observed values only)

Generated 2026-10-02T05:13:12Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.54 | 86.80 | 0.8499±0.0019 | 92.97 | 15998 | 5269/8140 | COMPLETED |
| P107_AL2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.32 | 83.88 | 0.9759±0.0014 | 93.14 | 16089 | 5269/8140 | COMPLETED |
| P107_AP1_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.82 | 87.30 | 0.8612±0.0026 | 119.10 | 16119 | 5269/8142 | COMPLETED |
| P107_AP2_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 88.98 | 86.60 | 0.8612±0.0014 | 98.97 | 27839 | 7066/10154 | COMPLETED |
| P107_AP3_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 89.06 | 87.30 | 0.8604±0.0029 | 121.12 | 16053 | 5269/8146 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AL2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P107_AP1_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP2_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |
| P107_AP3_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | FixedCosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | 41.94 | 88.54 | 46.60 | 36.54 | 86.80 | 50.26 | 2.94 | 92.97 | -0.5569 |
| P107_AL2_views4_800ep_seed0 | 41.94 | 86.32 | 44.38 | 36.54 | 83.88 | 47.34 | 2.94 | 93.14 | -0.5569 |
| P107_AP1_views4_800ep_seed0 | 41.94 | 88.82 | 46.88 | 36.54 | 87.30 | 50.76 | 2.94 | 119.10 | -0.5569 |
| P107_AP2_views4_800ep_seed0 | 41.98 | 88.98 | 47.00 | 36.58 | 86.60 | 50.02 | 2.94 | 98.97 | -0.5569 |
| P107_AP3_views4_800ep_seed0 | 41.94 | 89.06 | 47.12 | 36.54 | 87.30 | 50.76 | 2.94 | 121.12 | -0.5569 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P107_AL1_views4_800ep_seed0: ep0: 36.54, ep20: 67.10, ep50: 75.84, ep100: 81.28, ep200: 83.82, ep400: 85.78, ep600: 86.54, ep800: 86.80
- P107_AL2_views4_800ep_seed0: ep0: 36.54, ep20: 66.86, ep50: 72.58, ep100: 76.64, ep200: 79.96, ep400: 82.96, ep600: 83.60, ep800: 83.88
- P107_AP1_views4_800ep_seed0: ep0: 36.54, ep20: 69.80, ep50: 77.24, ep100: 81.30, ep200: 84.58, ep400: 86.62, ep600: 87.24, ep800: 87.30
- P107_AP2_views4_800ep_seed0: ep0: 36.58, ep20: 68.34, ep50: 75.94, ep100: 80.60, ep200: 83.86, ep400: 86.22, ep600: 86.48, ep800: 86.60
- P107_AP3_views4_800ep_seed0: ep0: 36.54, ep20: 70.54, ep50: 77.50, ep100: 81.78, ep200: 84.58, ep400: 86.52, ep600: 86.92, ep800: 87.30

## Held-out J trajectory (VCS only)

- P107_AL1_views4_800ep_seed0: ep0: -0.5569, ep20: 0.7651, ep50: 0.8076, ep100: 0.8327, ep200: 0.8442, ep400: 0.8497, ep600: 0.8497, ep800: 0.8499
- P107_AL2_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8539, ep50: 0.9111, ep100: 0.9414, ep200: 0.9599, ep400: 0.9702, ep600: 0.9747, ep800: 0.9759
- P107_AP1_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8024, ep50: 0.8314, ep100: 0.8518, ep200: 0.8614, ep400: 0.8621, ep600: 0.8617, ep800: 0.8612
- P107_AP2_views4_800ep_seed0: ep0: -0.5569, ep20: 0.7791, ep50: 0.8188, ep100: 0.8421, ep200: 0.8533, ep400: 0.8599, ep600: 0.8608, ep800: 0.8612
- P107_AP3_views4_800ep_seed0: ep0: -0.5569, ep20: 0.8028, ep50: 0.8333, ep100: 0.8528, ep200: 0.8605, ep400: 0.8613, ep600: 0.8611, ep800: 0.8604

## Final-epoch training objective values (epoch means)

- P107_AL1_views4_800ep_seed0: J_raw 0.8873, R_binary 0.1127, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AL2_views4_800ep_seed0: J_raw 0.9865, R_binary 0.0135, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP1_views4_800ep_seed0: J_raw 0.9030, R_binary 0.0970, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP2_views4_800ep_seed0: J_raw 0.8948, R_binary 0.1052, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P107_AP3_views4_800ep_seed0: J_raw 0.9034, R_binary 0.0966, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P107_AL1_views4_800ep_seed0 | 0.1127 | 2271 | 4542 | 15998 | 56 | 12 | 35840000 | None | 1017350@node58 |
| P107_AL2_views4_800ep_seed0 | 0.1134 | 2257 | 4514 | 16089 | 51 | 17 | 35840000 | 2 | 1017352@node59 |
| P107_AP1_views4_800ep_seed0 | 0.1137 | 2252 | 4504 | 16119 | 52 | 11 | 35840000 | None | 1017353@node59 |
| P107_AP2_views4_800ep_seed0 | 0.1960 | 1306 | 2612 | 27839 | 101 | 21 | 35840000 | None | 1017354@nodesumo01 |
| P107_AP3_views4_800ep_seed0 | 0.1132 | 2262 | 4524 | 16053 | 52 | 12 | 35840000 | None | 1017356@node61 |

## Provenance

- P107_AL1_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `4aa524331e6d6dcc1ca168709dee3954585631c505dc541943e3a65016f9bac9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AL2_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `c1308c3a95978a9e9b7f1cfea620b8fb0c18190b8313b6af032ef1ce2b463b2a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP1_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `11b5a305c98f71d4db2e5b524a1ec093c1668c6f759031a08f27bf14d55bee82`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP2_views4_800ep_seed0: commit `7e7498ddd134400d305114cb6eda20d58c2387f7` dirty=True, config `ebf62c1d8c4d5b7c87adae4d6c19b47d4f894eb9b7221dea7bf21b4401c2c718`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P107_AP3_views4_800ep_seed0: commit `f29a6b261d66696034ff8b1b86d158a1c8480ac8` dirty=True, config `2f3e24e0c5d7386fa68c223ad8991752e16776378bbe66882554e0a0a5c60140`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 5 | 88.34 ± 1.15 | 41.95 ± 0.02 | 46.40 ± 1.14 | 86.38 ± 1.43 | 105.06 ± 13.97 | 0.88 ± 0.05 |

## Coverage checks

- P107_AL1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AL2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP1_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP2_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P107_AP3_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
