# P35_vcs_a5_base — neutral results table (observed values only)

Generated 2026-09-26T13:37:33Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 85.54 | 82.62 | 0.9635±0.0013 | 88.18 | 22455 | 3683/4842 | COMPLETED |
| P35_vcs_a5_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 85.18 | 82.20 | 0.9632±0.0011 | 85.88 | 22307 | 3683/4842 | COMPLETED |
| P35_vcs_a5_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 85.18 | 82.16 | 0.9639±0.0014 | 84.15 | 25269 | 3683/4842 | COMPLETED |
| P35_vcs_a5_views4_400ep_seed0 | vcs_qmi | 8 | 0 | 356/400 | null | null | null | null | null | null/null | RUNNING |
| P35_vcs_a5_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 86.42 | 85.60 | 0.9743±0.0015 | 135.89 | 48508 | 7568/12672 | COMPLETED |
| P35_vcs_a5_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 326/800 | null | null | null | null | null | null/null | RUNNING |
| P35_vcs_a5_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 319/800 | null | null | null | null | null | null/null | RUNNING |
| P35_vcs_a5_views4_seed0 | vcs_qmi | 8 | 0 | 200/200 | 84.50 | 81.40 | 0.9640±0.0011 | 75.96 | 13682 | 7568/12672 | COMPLETED |
| P35_vcs_a5_views4_seed1 | vcs_qmi | 8 | 1 | 200/200 | 84.40 | 81.18 | 0.9636±0.0009 | 76.10 | 13208 | 7568/12672 | COMPLETED |
| P35_vcs_a5_views4_seed2 | vcs_qmi | 8 | 2 | 200/200 | 84.72 | 81.62 | 0.9636±0.0011 | 75.07 | 13536 | 7568/12672 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_views4_400ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 400 |
| P35_vcs_a5_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P35_vcs_a5_views4_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P35_vcs_a5_views4_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P35_vcs_a5_views4_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | 41.78 | 85.54 | 43.76 | 36.58 | 82.62 | 46.04 | 2.94 | 88.18 | -0.9998 |
| P35_vcs_a5_800ep_seed1 | 41.60 | 85.18 | 43.58 | 37.42 | 82.20 | 44.78 | 3.22 | 85.88 | -0.9998 |
| P35_vcs_a5_800ep_seed2 | 43.02 | 85.18 | 42.16 | 37.08 | 82.16 | 45.08 | 3.22 | 84.15 | -0.9998 |
| P35_vcs_a5_views4_400ep_seed0 | null | null | null | null | null | null | null | null | null |
| P35_vcs_a5_views4_800ep_seed0 | 41.78 | 86.42 | 44.64 | 36.58 | 85.60 | 49.02 | 2.94 | 135.89 | -0.9998 |
| P35_vcs_a5_views4_800ep_seed1 | null | null | null | null | null | null | null | null | null |
| P35_vcs_a5_views4_800ep_seed2 | null | null | null | null | null | null | null | null | null |
| P35_vcs_a5_views4_seed0 | 41.78 | 84.50 | 42.72 | 36.58 | 81.40 | 44.82 | 2.94 | 75.96 | -0.9998 |
| P35_vcs_a5_views4_seed1 | 41.60 | 84.40 | 42.80 | 37.42 | 81.18 | 43.76 | 3.22 | 76.10 | -0.9998 |
| P35_vcs_a5_views4_seed2 | 43.02 | 84.72 | 41.70 | 37.08 | 81.62 | 44.54 | 3.22 | 75.07 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P35_vcs_a5_800ep_seed0: ep0: 36.58, ep20: 60.84, ep50: 68.82, ep100: 73.42, ep200: 77.78, ep400: 80.68, ep600: 82.10, ep800: 82.62
- P35_vcs_a5_800ep_seed1: ep0: 37.42, ep20: 61.06, ep50: 68.00, ep100: 72.76, ep200: 78.08, ep400: 80.56, ep600: 81.52, ep800: 82.20
- P35_vcs_a5_800ep_seed2: ep0: 37.08, ep20: 61.22, ep50: 69.06, ep100: 73.72, ep200: 76.96, ep400: 80.74, ep600: 82.02, ep800: 82.16
- P35_vcs_a5_views4_400ep_seed0: ep0: 36.58, ep20: 67.96, ep50: 74.20, ep100: 78.32, ep200: 81.64, ep300: 82.94
- P35_vcs_a5_views4_800ep_seed0: ep0: 36.58, ep20: 68.36, ep50: 74.46, ep100: 78.78, ep200: 81.92, ep400: 84.16, ep600: 85.16, ep800: 85.60
- P35_vcs_a5_views4_800ep_seed1: ep0: 37.42, ep20: 68.14, ep50: 74.14, ep100: 79.14, ep200: 81.88
- P35_vcs_a5_views4_800ep_seed2: ep0: 37.08, ep20: 67.42, ep50: 73.86, ep100: 79.10, ep200: 82.10
- P35_vcs_a5_views4_seed0: ep0: 36.58, ep10: 58.32, ep20: 68.66, ep50: 74.94, ep100: 77.98, ep150: 80.64, ep200: 81.40
- P35_vcs_a5_views4_seed1: ep0: 37.42, ep10: 56.74, ep20: 67.78, ep50: 74.24, ep100: 78.60, ep150: 80.78, ep200: 81.18
- P35_vcs_a5_views4_seed2: ep0: 37.08, ep10: 57.82, ep20: 68.40, ep50: 74.80, ep100: 79.04, ep150: 81.06, ep200: 81.62

## Held-out J trajectory (VCS only)

- P35_vcs_a5_800ep_seed0: ep0: -0.9998, ep20: 0.8000, ep50: 0.8602, ep100: 0.9076, ep200: 0.9357, ep400: 0.9514, ep600: 0.9611, ep800: 0.9635
- P35_vcs_a5_800ep_seed1: ep0: -0.9998, ep20: 0.8113, ep50: 0.8724, ep100: 0.9088, ep200: 0.9339, ep400: 0.9523, ep600: 0.9606, ep800: 0.9632
- P35_vcs_a5_800ep_seed2: ep0: -0.9998, ep20: 0.7959, ep50: 0.8618, ep100: 0.9036, ep200: 0.9317, ep400: 0.9517, ep600: 0.9609, ep800: 0.9639
- P35_vcs_a5_views4_400ep_seed0: ep0: -0.9998, ep20: 0.8809, ep50: 0.9146, ep100: 0.9429, ep200: 0.9602, ep300: 0.9683
- P35_vcs_a5_views4_800ep_seed0: ep0: -0.9998, ep20: 0.8803, ep50: 0.9148, ep100: 0.9418, ep200: 0.9589, ep400: 0.9686, ep600: 0.9728, ep800: 0.9743
- P35_vcs_a5_views4_800ep_seed1: ep0: -0.9998, ep20: 0.8606, ep50: 0.9202, ep100: 0.9400, ep200: 0.9575
- P35_vcs_a5_views4_800ep_seed2: ep0: -0.9998, ep20: 0.8590, ep50: 0.9128, ep100: 0.9420, ep200: 0.9583
- P35_vcs_a5_views4_seed0: ep0: -0.9998, ep10: 0.7061, ep20: 0.8745, ep50: 0.9169, ep100: 0.9450, ep150: 0.9601, ep200: 0.9640
- P35_vcs_a5_views4_seed1: ep0: -0.9998, ep10: 0.7034, ep20: 0.8660, ep50: 0.9179, ep100: 0.9450, ep150: 0.9596, ep200: 0.9636
- P35_vcs_a5_views4_seed2: ep0: -0.9998, ep10: 0.7215, ep20: 0.8364, ep50: 0.9173, ep100: 0.9469, ep150: 0.9605, ep200: 0.9636

## Final-epoch training objective values (epoch means)

- P35_vcs_a5_800ep_seed0: J_raw 0.9725, R_binary 0.0275, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_800ep_seed1: J_raw 0.9723, R_binary 0.0277, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_800ep_seed2: J_raw 0.9721, R_binary 0.0279, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_400ep_seed0: J_raw 0.9793, R_binary 0.0207, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_800ep_seed0: J_raw 0.9861, R_binary 0.0139, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_800ep_seed1: J_raw 0.9753, R_binary 0.0247, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_800ep_seed2: J_raw 0.9749, R_binary 0.0251, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_seed0: J_raw 0.9705, R_binary 0.0295, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_seed1: J_raw 0.9704, R_binary 0.0296, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P35_vcs_a5_views4_seed2: J_raw 0.9707, R_binary 0.0293, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_800ep_seed0 | 0.1584 | 1616 | 3233 | 22455 | 165 | 29 | 35840000 | 2 | 1009210@nodeaudible01 |
| P35_vcs_a5_800ep_seed1 | 0.1573 | 1627 | 3255 | 22307 | 148 | 29 | 35840000 | 2 | 1009364@nodeaudible01 |
| P35_vcs_a5_800ep_seed2 | 0.1781 | 1437 | 2874 | 25269 | 184 | 36 | 35840000 | 2 | 1009365@node06 |
| P35_vcs_a5_views4_400ep_seed0 | null | null | null | null | null | null | None | 2 | 1009376@nodeaudible01 |
| P35_vcs_a5_views4_800ep_seed0 | 0.3425 | 747 | 1495 | 48508 | 156 | 30 | 35840000 | 2 | 1009268@nodeaudible01 |
| P35_vcs_a5_views4_800ep_seed1 | null | null | null | null | null | null | None | 2 | 1009374@node05 |
| P35_vcs_a5_views4_800ep_seed2 | null | null | null | null | null | null | None | 2 | 1009375@node05 |
| P35_vcs_a5_views4_seed0 | 0.3861 | 663 | 1326 | 13682 | 160 | 37 | 8960000 | 2 | 1009209@node01 |
| P35_vcs_a5_views4_seed1 | 0.3733 | 686 | 1372 | 13208 | 153 | 35 | 8960000 | 2 | 1009219@node05 |
| P35_vcs_a5_views4_seed2 | 0.3827 | 669 | 1338 | 13536 | 149 | 35 | 8960000 | 2 | 1009220@node05 |

## Provenance

- P35_vcs_a5_800ep_seed0: commit `a9ef673a17008f760f55639e4a1c4d07860cc4a6` dirty=True, config `2a0c3e94673a1fe6993817a3179d98f16b39c6c5cd3633d21e9b54419813f723`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_800ep_seed1: commit `6641797cbebef22315f4e997b5121448d2c0bd6f` dirty=True, config `162ec9e4e0eb108a254175c64a708989f5cc5b3fc4b4fe209a8532dcd0bf7790`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P35_vcs_a5_800ep_seed2: commit `6641797cbebef22315f4e997b5121448d2c0bd6f` dirty=True, config `7524920e22b9bab1ad8668b44ee19fbcfb650ba9d6dfc5d983664108815b3421`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P35_vcs_a5_views4_400ep_seed0: commit `4e77d3800790f085c39282e66129f0f34da582b4` dirty=True, config `62058851cf37b5fab96aa2062f1047ac52bfa8a9995b1873aec53cea82d24c24`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_views4_800ep_seed0: commit `5ae36125529aa158b237f7d8c15238170e83e34c` dirty=True, config `951d163871a466d2abdb86f2abf0fa98282b6ef5a4f763a3077393046ea81060`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_views4_800ep_seed1: commit `4e77d3800790f085c39282e66129f0f34da582b4` dirty=True, config `1e1d6b856fe93e9ce2e6f450628892dbcd5f51bd9866ed50d9783a413d2590b8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P35_vcs_a5_views4_800ep_seed2: commit `4e77d3800790f085c39282e66129f0f34da582b4` dirty=True, config `26fe0625ba71b9c4f12528de90861e85cf93e9592c926a371a2955cfc80bc44e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P35_vcs_a5_views4_seed0: commit `a9ef673a17008f760f55639e4a1c4d07860cc4a6` dirty=False, config `64e637f951a5dd1a592d1d8f7097cfae974501c127800a2aa2c0d4d8732e62f8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P35_vcs_a5_views4_seed1: commit `a4c31661ecac6cb0cf6ebf707c4f89cfcbef9702` dirty=False, config `282530e5462851288ad57c1e05c438bbf6eeb118a45486fb4f37b654590a630a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P35_vcs_a5_views4_seed2: commit `a4c31661ecac6cb0cf6ebf707c4f89cfcbef9702` dirty=False, config `15807ebbc442d080bd9f4eefdfde519f0a58765610a77a7fed690099ff81a8e8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 7 | 85.13 ± 0.70 | 42.08 ± 0.65 | 43.05 ± 1.01 | 82.40 ± 1.50 | 88.75 ± 21.46 | 0.97 ± 0.00 |

## Coverage checks

- P35_vcs_a5_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P35_vcs_a5_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P35_vcs_a5_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P35_vcs_a5_views4_400ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P35_vcs_a5_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P35_vcs_a5_views4_800ep_seed1: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P35_vcs_a5_views4_800ep_seed2: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P35_vcs_a5_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P35_vcs_a5_views4_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P35_vcs_a5_views4_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
