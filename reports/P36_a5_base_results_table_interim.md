# P35_vcs_a5_base — neutral results table (observed values only)

Generated 2026-09-26T02:06:10Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | vcs_qmi | 8 | 0 | 561/800 | null | null | null | null | null | null/null | RUNNING |
| P35_vcs_a5_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 178/800 | null | null | null | null | null | null/null | RUNNING |
| P35_vcs_a5_views4_seed0 | vcs_qmi | 8 | 0 | 200/200 | 84.50 | 81.40 | 0.9640±0.0011 | 75.96 | 13682 | 7568/12672 | COMPLETED |
| P35_vcs_a5_views4_seed1 | vcs_qmi | 8 | 1 | 200/200 | 84.40 | 81.18 | 0.9636±0.0009 | 76.10 | 13208 | 7568/12672 | COMPLETED |
| P35_vcs_a5_views4_seed2 | vcs_qmi | 8 | 2 | 200/200 | 84.72 | 81.62 | 0.9636±0.0011 | 75.07 | 13536 | 7568/12672 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_views4_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P35_vcs_a5_views4_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P35_vcs_a5_views4_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | null | null | null | null | null | null | null | null | null |
| P35_vcs_a5_views4_800ep_seed0 | null | null | null | null | null | null | null | null | null |
| P35_vcs_a5_views4_seed0 | 41.78 | 84.50 | 42.72 | 36.58 | 81.40 | 44.82 | 2.94 | 75.96 | -0.9998 |
| P35_vcs_a5_views4_seed1 | 41.60 | 84.40 | 42.80 | 37.42 | 81.18 | 43.76 | 3.22 | 76.10 | -0.9998 |
| P35_vcs_a5_views4_seed2 | 43.02 | 84.72 | 41.70 | 37.08 | 81.62 | 44.54 | 3.22 | 75.07 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P35_vcs_a5_800ep_seed0: ep0: 36.58, ep20: 60.84, ep50: 68.82, ep100: 73.42, ep200: 77.78, ep400: 80.68
- P35_vcs_a5_views4_800ep_seed0: ep0: 36.58, ep20: 68.36, ep50: 74.46, ep100: 78.78
- P35_vcs_a5_views4_seed0: ep0: 36.58, ep10: 58.32, ep20: 68.66, ep50: 74.94, ep100: 77.98, ep150: 80.64, ep200: 81.40
- P35_vcs_a5_views4_seed1: ep0: 37.42, ep10: 56.74, ep20: 67.78, ep50: 74.24, ep100: 78.60, ep150: 80.78, ep200: 81.18
- P35_vcs_a5_views4_seed2: ep0: 37.08, ep10: 57.82, ep20: 68.40, ep50: 74.80, ep100: 79.04, ep150: 81.06, ep200: 81.62

## Held-out J trajectory (VCS only)

- P35_vcs_a5_800ep_seed0: ep0: -0.9998, ep20: 0.8000, ep50: 0.8602, ep100: 0.9076, ep200: 0.9357, ep400: 0.9514
- P35_vcs_a5_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8803, ep50: 0.9148, ep100: 0.9418
- P35_vcs_a5_views4_seed0: ep0: -0.9998, ep10: 0.7061, ep20: 0.8745, ep50: 0.9169, ep100: 0.9450, ep150: 0.9601, ep200: 0.9640
- P35_vcs_a5_views4_seed1: ep0: -0.9998, ep10: 0.7034, ep20: 0.8660, ep50: 0.9179, ep100: 0.9450, ep150: 0.9596, ep200: 0.9636
- P35_vcs_a5_views4_seed2: ep0: -0.9998, ep10: 0.7215, ep20: 0.8364, ep50: 0.9173, ep100: 0.9469, ep150: 0.9605, ep200: 0.9636

## Final-epoch training objective values (epoch means)

- P35_vcs_a5_800ep_seed0: J_raw 0.9686, R_binary 0.0314, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_800ep_seed0: J_raw 0.9647, R_binary 0.0353, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_seed0: J_raw 0.9705, R_binary 0.0295, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_seed1: J_raw 0.9704, R_binary 0.0296, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_seed2: J_raw 0.9707, R_binary 0.0293, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | null | null | null | null | null | null | None | 2 | 1009210@nodeaudible01 |
| P35_vcs_a5_views4_800ep_seed0 | null | null | null | null | null | null | None | 2 | 1009268@nodeaudible01 |
| P35_vcs_a5_views4_seed0 | 0.3861 | 663 | 1326 | 13682 | 160 | 37 | 8960000 | 2 | 1009209@node01 |
| P35_vcs_a5_views4_seed1 | 0.3733 | 686 | 1372 | 13208 | 153 | 35 | 8960000 | 2 | 1009219@node05 |
| P35_vcs_a5_views4_seed2 | 0.3827 | 669 | 1338 | 13536 | 149 | 35 | 8960000 | 2 | 1009220@node05 |

## Provenance

- P35_vcs_a5_800ep_seed0: commit `a9ef673a17008f760f55639e4a1c4d07860cc4a6` dirty=True, config `2a0c3e94673a1fe6993817a3179d98f16b39c6c5cd3633d21e9b54419813f723`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_views4_800ep_seed0: commit `5ae36125529aa158b237f7d8c15238170e83e34c` dirty=True, config `951d163871a466d2abdb86f2abf0fa98282b6ef5a4f763a3077393046ea81060`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_views4_seed0: commit `a9ef673a17008f760f55639e4a1c4d07860cc4a6` dirty=False, config `64e637f951a5dd1a592d1d8f7097cfae974501c127800a2aa2c0d4d8732e62f8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_views4_seed1: commit `a4c31661ecac6cb0cf6ebf707c4f89cfcbef9702` dirty=False, config `282530e5462851288ad57c1e05c438bbf6eeb118a45486fb4f37b654590a630a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P35_vcs_a5_views4_seed2: commit `a4c31661ecac6cb0cf6ebf707c4f89cfcbef9702` dirty=False, config `15807ebbc442d080bd9f4eefdfde519f0a58765610a77a7fed690099ff81a8e8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 3 | 84.54 ± 0.16 | 42.13 ± 0.77 | 42.41 ± 0.61 | 81.40 ± 0.22 | 75.71 ± 0.56 | 0.96 ± 0.00 |

## Coverage checks

- P35_vcs_a5_800ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P35_vcs_a5_views4_800ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P35_vcs_a5_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P35_vcs_a5_views4_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P35_vcs_a5_views4_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
