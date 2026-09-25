# P8_long800 — neutral results table (observed values only)

Generated 2026-09-25T02:24:20Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P8_vcs800_seed0 | vcs_qmi | 1 | 0 | 800/800 | 79.24 | 71.36 | 0.9479±0.0032 | 20.13 | 12995 | 3648/4336 | COMPLETED |
| P8_vcs800_seed1 | vcs_qmi | 1 | 1 | 800/800 | 78.48 | 71.60 | 0.9457±0.0037 | 20.69 | 18272 | 2793/3524 | COMPLETED |
| P8_vcs800_seed2 | vcs_qmi | 1 | 2 | 800/800 | 79.06 | 71.60 | 0.9469±0.0043 | 20.80 | 24890 | 3687/4848 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P8_vcs800_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P8_vcs800_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P8_vcs800_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P8_vcs800_seed0 | 41.98 | 79.24 | 37.26 | 36.58 | 71.36 | 34.78 | 2.94 | 20.13 | -0.0000 |
| P8_vcs800_seed1 | 41.68 | 78.48 | 36.80 | 37.40 | 71.60 | 34.20 | 3.22 | 20.69 | -0.0000 |
| P8_vcs800_seed2 | 43.02 | 79.06 | 36.04 | 37.08 | 71.60 | 34.52 | 3.22 | 20.80 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P8_vcs800_seed0: ep0: 36.58, ep20: 47.44, ep50: 55.52, ep100: 60.62, ep200: 65.26, ep400: 68.96, ep600: 71.20, ep800: 71.36
- P8_vcs800_seed1: ep0: 37.40, ep20: 49.64, ep50: 56.06, ep100: 60.58, ep200: 64.48, ep400: 68.28, ep600: 70.58, ep800: 71.60
- P8_vcs800_seed2: ep0: 37.08, ep20: 49.06, ep50: 57.28, ep100: 61.50, ep200: 64.68, ep400: 68.72, ep600: 71.22, ep800: 71.60

## Held-out J trajectory (VCS only)

- P8_vcs800_seed0: ep0: -0.0000, ep20: 0.7017, ep50: 0.7952, ep100: 0.8458, ep200: 0.8974, ep400: 0.9257, ep600: 0.9428, ep800: 0.9479
- P8_vcs800_seed1: ep0: -0.0000, ep20: 0.7158, ep50: 0.8049, ep100: 0.8503, ep200: 0.8919, ep400: 0.9262, ep600: 0.9406, ep800: 0.9457
- P8_vcs800_seed2: ep0: -0.0000, ep20: 0.7178, ep50: 0.8035, ep100: 0.8524, ep200: 0.8952, ep400: 0.9222, ep600: 0.9424, ep800: 0.9469

## Final-epoch training objective values (epoch means)

- P8_vcs800_seed0: J_raw 0.9519, R_binary 0.0481, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P8_vcs800_seed1: J_raw 0.9522, R_binary 0.0478, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P8_vcs800_seed2: J_raw 0.9538, R_binary 0.0462, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P8_vcs800_seed0 | 0.0914 | 2800 | 5601 | 12995 | 92 | 20 | 35840000 | 394753 | 1007959@node53 |
| P8_vcs800_seed1 | 0.1297 | 1973 | 3947 | 18272 | 84 | 15 | 35840000 | 394753 | 1007960@node60 |
| P8_vcs800_seed2 | 0.1757 | 1457 | 2915 | 24890 | 171 | 35 | 35840000 | 394753 | 1007961@node56 |

## Provenance

- P8_vcs800_seed0: commit `61ffcaadaa7dc2ca5f3c426ddb29a7c5806dfc9c` dirty=True, config `512cf0644095569e56bde0e9fcea6af878f8f1ddcd14d39c25d15de3127cbe9b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P8_vcs800_seed1: commit `61ffcaadaa7dc2ca5f3c426ddb29a7c5806dfc9c` dirty=True, config `820440dbb113fe0113717b0da276976ab4cf68db70d2e299619829f8cec8f6ef`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P8_vcs800_seed2: commit `61ffcaadaa7dc2ca5f3c426ddb29a7c5806dfc9c` dirty=True, config `6447d22bd2d8c8c775a9d2a694b8fbd4cbe0f62f3286f07e5be76cb775cc1855`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=1 | 3 | 78.93 ± 0.40 | 42.23 ± 0.70 | 36.70 ± 0.62 | 71.52 ± 0.14 | 20.54 ± 0.36 | 0.95 ± 0.00 |

## Coverage checks

- P8_vcs800_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P8_vcs800_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P8_vcs800_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
