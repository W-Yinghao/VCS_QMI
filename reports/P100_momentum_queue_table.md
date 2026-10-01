# P100_momentum_queue — neutral results table (observed values only)

Generated 2026-10-01T14:47:35Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P100_simclr_mq_views4_800ep_seed0 | simclr_matched | null | 0 | 800/800 | 78.08 | 80.06 | null | 1.00 | 16820 | 5400/8184 | COMPLETED COLLAPSE_SUSPECTED |
| P100_simclr_mq_views4_800ep_seed1 | simclr_matched | null | 1 | 800/800 | 78.98 | 80.54 | null | 1.00 | 16768 | 5400/8184 | COMPLETED COLLAPSE_SUSPECTED |
| P100_simclr_mq_views4_800ep_seed2 | simclr_matched | null | 2 | 800/800 | 76.68 | 79.94 | null | 1.00 | 16813 | 5400/8184 | COMPLETED COLLAPSE_SUSPECTED |
| P100_vcs_mq_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 11.30 | 27.36 | -0.6413±0.0063 | 1.00 | 17016 | 5400/8184 | COMPLETED COLLAPSE_SUSPECTED |
| P100_vcs_mq_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 13.12 | 27.54 | -0.7902±0.0028 | 1.04 | 16972 | 5400/8184 | COMPLETED COLLAPSE_SUSPECTED |
| P100_vcs_mq_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 13.32 | 29.52 | -0.8032±0.0016 | 1.04 | 37445 | 5400/8184 | COMPLETED COLLAPSE_SUSPECTED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P100_simclr_mq_views4_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P100_simclr_mq_views4_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P100_simclr_mq_views4_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P100_vcs_mq_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P100_vcs_mq_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P100_vcs_mq_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P100_simclr_mq_views4_800ep_seed0 | 41.94 | 78.08 | 36.14 | 36.54 | 80.06 | 43.52 | 2.94 | 1.00 | null |
| P100_simclr_mq_views4_800ep_seed1 | 41.68 | 78.98 | 37.30 | 37.40 | 80.54 | 43.14 | 3.22 | 1.00 | null |
| P100_simclr_mq_views4_800ep_seed2 | 42.98 | 76.68 | 33.70 | 37.08 | 79.94 | 42.86 | 3.22 | 1.00 | null |
| P100_vcs_mq_views4_800ep_seed0 | 41.94 | 11.30 | -30.64 | 36.54 | 27.36 | -9.18 | 2.94 | 1.00 | -0.9998 |
| P100_vcs_mq_views4_800ep_seed1 | 41.68 | 13.12 | -28.56 | 37.40 | 27.54 | -9.86 | 3.22 | 1.04 | -0.9998 |
| P100_vcs_mq_views4_800ep_seed2 | 42.98 | 13.32 | -29.66 | 37.08 | 29.52 | -7.56 | 3.22 | 1.04 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P100_simclr_mq_views4_800ep_seed0: ep0: 36.54, ep20: 66.44, ep50: 72.24, ep100: 75.88, ep200: 77.20, ep400: 79.12, ep600: 78.94, ep800: 80.06
- P100_simclr_mq_views4_800ep_seed1: ep0: 37.40, ep20: 66.90, ep50: 72.96, ep100: 75.84, ep200: 77.92, ep400: 80.06, ep600: 79.94, ep800: 80.54
- P100_simclr_mq_views4_800ep_seed2: ep0: 37.08, ep20: 67.78, ep50: 72.78, ep100: 76.10, ep200: 78.60, ep400: 78.62, ep600: 79.84, ep800: 79.96
- P100_vcs_mq_views4_800ep_seed0: ep0: 36.54, ep20: 25.46, ep50: 29.14, ep100: 24.94, ep200: 26.22, ep400: 29.50, ep600: 27.56, ep800: 27.42
- P100_vcs_mq_views4_800ep_seed1: ep0: 37.40, ep20: 28.30, ep50: 27.44, ep100: 23.76, ep200: 27.08, ep400: 27.20, ep600: 26.36, ep800: 27.44
- P100_vcs_mq_views4_800ep_seed2: ep0: 37.08, ep20: 31.62, ep50: 28.78, ep100: 25.90, ep200: 28.20, ep400: 29.14, ep600: 30.60, ep800: 29.52

## Held-out J trajectory (VCS only)

- P100_vcs_mq_views4_800ep_seed0: ep0: -0.9998, ep20: -0.9511, ep50: -0.9214, ep100: -0.8961, ep200: -0.8998, ep400: -0.9387, ep600: -0.8976, ep800: -0.6413
- P100_vcs_mq_views4_800ep_seed1: ep0: -0.9998, ep20: -0.6142, ep50: -0.9173, ep100: -0.9505, ep200: -0.9362, ep400: -0.9422, ep600: -0.9353, ep800: -0.7902
- P100_vcs_mq_views4_800ep_seed2: ep0: -0.9998, ep20: -0.9978, ep50: -0.9488, ep100: -0.9453, ep200: -0.9383, ep400: -0.9574, ep600: -0.9330, ep800: -0.8032

## Final-epoch training objective values (epoch means)

- P100_simclr_mq_views4_800ep_seed0: J_raw null, R_binary null, nt_xent 4.6658, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P100_simclr_mq_views4_800ep_seed1: J_raw null, R_binary null, nt_xent 4.5825, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P100_simclr_mq_views4_800ep_seed2: J_raw null, R_binary null, nt_xent 4.4551, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P100_vcs_mq_views4_800ep_seed0: J_raw 0.8100, R_binary 0.1900, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P100_vcs_mq_views4_800ep_seed1: J_raw 0.7970, R_binary 0.2030, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P100_vcs_mq_views4_800ep_seed2: J_raw 0.9100, R_binary 0.0900, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P100_simclr_mq_views4_800ep_seed0 | 0.1187 | 2158 | 4315 | 16820 | 19 | 7 | 35840000 | None | 1015935@node58 |
| P100_simclr_mq_views4_800ep_seed1 | 0.1183 | 2164 | 4328 | 16768 | 18 | 7 | 35840000 | None | 1015936@node61 |
| P100_simclr_mq_views4_800ep_seed2 | 0.1185 | 2160 | 4320 | 16813 | 19 | 12 | 35840000 | None | 1015937@node58 |
| P100_vcs_mq_views4_800ep_seed0 | 0.1201 | 2132 | 4264 | 17016 | 54 | 11 | 35840000 | 2 | 1015932@node59 |
| P100_vcs_mq_views4_800ep_seed1 | 0.1197 | 2138 | 4276 | 16972 | 54 | 12 | 35840000 | 2 | 1015933@node61 |
| P100_vcs_mq_views4_800ep_seed2 | 0.2660 | 962 | 1925 | 37445 | 85 | 16 | 35840000 | 2 | 1015934@node61 |

## Provenance

- P100_simclr_mq_views4_800ep_seed0: commit `82ab9d3d3c85c6414a7a9822e7ede4d919d32f7c` dirty=True, config `1784a4c90ef9e04faba21f78b7990fa20d599bd0d13fac0e82b2f712f1b8a4da`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P100_simclr_mq_views4_800ep_seed1: commit `cfa03d05d116c32e9c332ccefa0849cad7434a22` dirty=True, config `edbf699cb52794c28e48a62242a1a26179b3ec68f20b13652fb9b07ef8fe2032`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P100_simclr_mq_views4_800ep_seed2: commit `7ef1717e9b65698bbe8d94d084c7fe8c0d101271` dirty=True, config `b67f8e75aa8e66479aec26e685757cd5993eaf0b872154879f98b19c565030c3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P100_vcs_mq_views4_800ep_seed0: commit `5a02b77a74f44fd9548170dea11e8420ff04573d` dirty=True, config `60b9910a662450703a159f292ffd498d01ea0310a3f2a7d295a989b41df4dae8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P100_vcs_mq_views4_800ep_seed1: commit `a8177b944ba3fdc1680c835911832433a26c6a76` dirty=True, config `7dcb12990c35df45fe7dce0e187d602f712b38c5580516ef73cc200811cafb81`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P100_vcs_mq_views4_800ep_seed2: commit `6e709ba2364cfd426966d2fe7ccf8e93fd971184` dirty=True, config `bf9d51c74feac20512a85d0d66a14c918c235e77ebbd8d984e92cd30aee9ae21`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 3 | 77.91 ± 1.16 | 42.20 ± 0.69 | 35.71 ± 1.84 | 80.18 ± 0.32 | 1.00 ± 0.00 | null |
| vcs_qmi K=8 | 3 | 12.58 ± 1.11 | 42.20 ± 0.69 | -29.62 ± 1.04 | 28.14 ± 1.20 | 1.03 ± 0.02 | -0.74 ± 0.09 |

## Coverage checks

- P100_simclr_mq_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P100_simclr_mq_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P100_simclr_mq_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P100_vcs_mq_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P100_vcs_mq_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P100_vcs_mq_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
