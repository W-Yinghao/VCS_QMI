# P39_vcs_recipe_ablation — neutral results table (observed values only)

Generated 2026-09-26T03:39:36Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | vcs_qmi | 8 | 0 | 31/200 | null | null | null | null | null | null/null | RUNNING |
| P39_vcs_a5_lr2e-3_seed0 | vcs_qmi | 8 | 0 | 200/200 | 81.64 | 77.50 | 0.9424±0.0015 | 50.55 | 5560 | 3683/4842 | COMPLETED |
| P39_vcs_a5_lr5e-4_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.56 | 75.76 | 0.9389±0.0016 | 41.99 | 5613 | 3683/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 83.52 | 78.30 | 0.9535±0.0011 | 55.62 | 6563 | 3682/4842 | COMPLETED |
| P39_vcs_a5_views4_b128_200ep_seed0 | vcs_qmi | 8 | 0 | 40/200 | null | null | null | null | null | null/null | RUNNING |
| P39_vcs_a5_views4_b64_100ep_seed0 | vcs_qmi | 8 | 0 | 36/100 | null | null | null | null | null | null/null | RUNNING |
| P39_vcs_a5_views4_k1_100ep_seed0 | vcs_qmi | 1 | 0 | 100/100 | 81.14 | 75.68 | 0.9418±0.0015 | 45.40 | 6105 | 7568/12670 | COMPLETED |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 79.66 | 75.38 | 0.9750±0.0011 | 85.79 | 6852 | 7568/12674 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 200 |
| P39_vcs_a5_lr2e-3_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.002 | 200 |
| P39_vcs_a5_lr5e-4_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.0005 | 200 |
| P39_vcs_a5_views4_b128_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 100 |
| P39_vcs_a5_views4_b128_200ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 200 |
| P39_vcs_a5_views4_b64_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 64 | 0.001 | 100 |
| P39_vcs_a5_views4_k1_100ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | null | null | null | null | null | null | null | null | null |
| P39_vcs_a5_lr2e-3_seed0 | 41.78 | 81.64 | 39.86 | 36.58 | 77.50 | 40.92 | 2.94 | 50.55 | -0.9998 |
| P39_vcs_a5_lr5e-4_seed0 | 41.78 | 80.56 | 38.78 | 36.58 | 75.76 | 39.18 | 2.94 | 41.99 | -0.9998 |
| P39_vcs_a5_views4_b128_100ep_seed0 | 41.78 | 83.52 | 41.74 | 36.58 | 78.30 | 41.72 | 2.94 | 55.62 | -0.9998 |
| P39_vcs_a5_views4_b128_200ep_seed0 | null | null | null | null | null | null | null | null | null |
| P39_vcs_a5_views4_b64_100ep_seed0 | null | null | null | null | null | null | null | null | null |
| P39_vcs_a5_views4_k1_100ep_seed0 | 41.78 | 81.14 | 39.36 | 36.58 | 75.68 | 39.10 | 2.94 | 45.40 | -0.9998 |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | 41.78 | 79.66 | 37.88 | 36.58 | 75.38 | 38.80 | 2.94 | 85.79 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P39_vcs_a5_b128_seed0: ep0: 36.58, ep10: 53.98, ep20: 63.74
- P39_vcs_a5_lr2e-3_seed0: ep0: 36.58, ep10: 54.26, ep20: 62.26, ep50: 69.66, ep100: 74.40, ep150: 77.22, ep200: 77.50
- P39_vcs_a5_lr5e-4_seed0: ep0: 36.58, ep10: 47.90, ep20: 56.52, ep50: 68.36, ep100: 72.76, ep150: 75.18, ep200: 75.76
- P39_vcs_a5_views4_b128_100ep_seed0: ep0: 36.58, ep10: 61.82, ep20: 70.32, ep50: 75.78, ep100: 78.30
- P39_vcs_a5_views4_b128_200ep_seed0: ep0: 36.58, ep10: 61.14, ep20: 69.90
- P39_vcs_a5_views4_b64_100ep_seed0: ep0: 36.58, ep10: 64.02, ep20: 71.18
- P39_vcs_a5_views4_k1_100ep_seed0: ep0: 36.58, ep10: 54.00, ep20: 64.70, ep50: 72.80, ep100: 75.68
- P39_vcs_a5_views4_nodetach_100ep_seed0: ep0: 36.58, ep10: 58.56, ep20: 67.24, ep50: 72.82, ep100: 75.38

## Held-out J trajectory (VCS only)

- P39_vcs_a5_b128_seed0: ep0: -0.9998, ep10: 0.7479, ep20: 0.8090
- P39_vcs_a5_lr2e-3_seed0: ep0: -0.9998, ep10: 0.7049, ep20: 0.7988, ep50: 0.8693, ep100: 0.9150, ep150: 0.9348, ep200: 0.9424
- P39_vcs_a5_lr5e-4_seed0: ep0: -0.9998, ep10: 0.5000, ep20: 0.7746, ep50: 0.8648, ep100: 0.9138, ep150: 0.9323, ep200: 0.9389
- P39_vcs_a5_views4_b128_100ep_seed0: ep0: -0.9998, ep10: 0.7937, ep20: 0.8825, ep50: 0.9247, ep100: 0.9535
- P39_vcs_a5_views4_b128_200ep_seed0: ep0: -0.9998, ep10: 0.8085, ep20: 0.8794
- P39_vcs_a5_views4_b64_100ep_seed0: ep0: -0.9998, ep10: 0.8094, ep20: 0.8803
- P39_vcs_a5_views4_k1_100ep_seed0: ep0: -0.9998, ep10: 0.6888, ep20: 0.8362, ep50: 0.9043, ep100: 0.9418
- P39_vcs_a5_views4_nodetach_100ep_seed0: ep0: -0.9998, ep10: 0.7615, ep20: 0.8859, ep50: 0.9492, ep100: 0.9750

## Final-epoch training objective values (epoch means)

- P39_vcs_a5_b128_seed0: J_raw 0.8491, R_binary 0.1509, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_lr2e-3_seed0: J_raw 0.9472, R_binary 0.0528, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_lr5e-4_seed0: J_raw 0.9430, R_binary 0.0570, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_100ep_seed0: J_raw 0.9591, R_binary 0.0409, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b128_200ep_seed0: J_raw 0.9201, R_binary 0.0799, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_b64_100ep_seed0: J_raw 0.9226, R_binary 0.0774, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_k1_100ep_seed0: J_raw 0.9470, R_binary 0.0530, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P39_vcs_a5_views4_nodetach_100ep_seed0: J_raw 0.9785, R_binary 0.0215, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P39_vcs_a5_b128_seed0 | null | null | null | null | null | null | None | 2 | 1009342@nodeaudible01 |
| P39_vcs_a5_lr2e-3_seed0 | 0.1570 | 1631 | 3262 | 5560 | 125 | 28 | 8960000 | 2 | 1009275@nodeaudible01 |
| P39_vcs_a5_lr5e-4_seed0 | 0.1584 | 1616 | 3233 | 5613 | 134 | 30 | 8960000 | 2 | 1009276@nodeaudible01 |
| P39_vcs_a5_views4_b128_100ep_seed0 | 0.1857 | 689 | 1379 | 6563 | 112 | 38 | 4492800 | 2 | 1009274@node01 |
| P39_vcs_a5_views4_b128_200ep_seed0 | null | null | null | null | null | null | None | 2 | 1009341@nodeaudible01 |
| P39_vcs_a5_views4_b64_100ep_seed0 | null | null | null | null | null | null | None | 2 | 1009340@nodeaudible01 |
| P39_vcs_a5_views4_k1_100ep_seed0 | 0.3454 | 741 | 1482 | 6105 | 93 | 31 | 4480000 | 2 | 1009272@nodeaudible01 |
| P39_vcs_a5_views4_nodetach_100ep_seed0 | 0.3869 | 662 | 1323 | 6852 | 114 | 37 | 4480000 | 2 | 1009273@node06 |

## Provenance

- P39_vcs_a5_b128_seed0: commit `8626b9834cd3c677071cf2f8914dd8daeb49b009` dirty=True, config `da792c7d66978c2b12473fc2db68c2dbd51c9aebf700b394c87378bd8d725e71`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_lr2e-3_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `56e68e9af470149201fe6534849de401df4ba159f15251fed4144ca191e240a4`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_lr5e-4_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `009a3de7ce0e83ef3cd568a1ca0d33605a563c998ff192b66d81933fad1ee0e6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_100ep_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `94ada458c11336bc97c00843ebc88c08315a19cc7f9a369053a5b409a3875c80`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b128_200ep_seed0: commit `8626b9834cd3c677071cf2f8914dd8daeb49b009` dirty=False, config `97cb173c1c0a6bc41a85056786b85e3a12bce46b82eed0d0c9475c3be796d463`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_b64_100ep_seed0: commit `8626b9834cd3c677071cf2f8914dd8daeb49b009` dirty=False, config `2f81635f2a94542e6399e017d811115d10a4cd3ed791f12f8fb52d57752eeeff`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_k1_100ep_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `ff6c645b31653aecc9178a9289a62854b5eca80c4af4cd0a2f760432a74e3161`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P39_vcs_a5_views4_nodetach_100ep_seed0: commit `5a7674875692fa5f98c37fa3c8e4a1cf7a1eb633` dirty=True, config `23e2210d0978099a3adedc735cf6628c7cb3e475a87f9772f474c8f2622439f8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=1 | 1 | 81.14 | 41.78 | 39.36 | 75.68 | 45.40 | 0.94 |
| vcs_qmi K=8 | 4 | 81.34 ± 1.66 | 41.78 ± 0.00 | 39.56 ± 1.66 | 76.73 ± 1.39 | 58.49 ± 19.05 | 0.95 ± 0.02 |

## Coverage checks

- P39_vcs_a5_b128_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P39_vcs_a5_lr2e-3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P39_vcs_a5_lr5e-4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_b128_200ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P39_vcs_a5_views4_b64_100ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P39_vcs_a5_views4_k1_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
- P39_vcs_a5_views4_nodetach_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
