# P87_s_cs — neutral results table (observed values only)

Generated 2026-09-30T00:43:02Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_views4_800ep_seed0 | cs_kernel_native | null | 0 | 800/800 | 78.88 | 74.04 | null | 7.08 | 35371 | 5269/8138 | COMPLETED |
| P87_kcs_bw0.5_views4_800ep_seed1 | cs_kernel_native | null | 1 | 800/800 | 79.20 | 72.62 | null | 10.09 | 28234 | 7066/10146 | COMPLETED |
| P87_kcs_bw0.5_views4_800ep_seed2 | cs_kernel_native | null | 2 | 800/800 | 77.60 | 68.60 | null | 8.44 | 27566 | 7066/10146 | COMPLETED |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 84.74 | 79.60 | 0.9144±0.0025 | 11.06 | 16516 | 5392/8158 | COMPLETED |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | vcs_qmi | 8 | 1 | 800/800 | 84.94 | 80.86 | 0.9270±0.0019 | 37.57 | 16503 | 5392/8158 | COMPLETED |
| P87_skernel_m4096_bw0.5_views4_800ep_seed2 | vcs_qmi | 8 | 2 | 800/800 | 85.04 | 80.66 | 0.9242±0.0015 | 38.66 | 16445 | 5392/8158 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_views4_800ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P87_kcs_bw0.5_views4_800ep_seed1 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P87_kcs_bw0.5_views4_800ep_seed2 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 800 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 256 | 0.001 | 800 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 256 | 0.001 | 800 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | RFFTanhCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_views4_800ep_seed0 | 41.94 | 78.88 | 36.94 | 36.54 | 74.04 | 37.50 | 2.94 | 7.08 | null |
| P87_kcs_bw0.5_views4_800ep_seed1 | 41.72 | 79.20 | 37.48 | 37.42 | 72.62 | 35.20 | 3.22 | 10.09 | null |
| P87_kcs_bw0.5_views4_800ep_seed2 | 42.98 | 77.60 | 34.62 | 37.06 | 68.60 | 31.54 | 3.22 | 8.44 | null |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | 41.94 | 84.74 | 42.80 | 36.54 | 79.60 | 43.06 | 2.94 | 11.06 | -0.0000 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | 41.68 | 84.94 | 43.26 | 37.40 | 80.86 | 43.46 | 3.22 | 37.57 | 0.0000 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed2 | 42.98 | 85.04 | 42.06 | 37.08 | 80.66 | 43.58 | 3.22 | 38.66 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P87_kcs_bw0.5_views4_800ep_seed0: ep0: 36.54, ep20: 39.20, ep50: 44.64, ep100: 51.34, ep200: 67.34, ep400: 72.10, ep600: 73.64, ep800: 74.04
- P87_kcs_bw0.5_views4_800ep_seed1: ep0: 37.42, ep20: 41.56, ep50: 45.70, ep100: 53.44, ep200: 68.58, ep400: 72.26, ep600: 72.34, ep800: 72.62
- P87_kcs_bw0.5_views4_800ep_seed2: ep0: 37.06, ep20: 42.08, ep50: 45.42, ep100: 60.20, ep200: 67.94, ep400: 69.56, ep600: 68.64, ep800: 68.60
- P87_skernel_m4096_bw0.5_views4_800ep_seed0: ep0: 36.54, ep20: 60.16, ep50: 67.52, ep100: 75.00, ep200: 77.72, ep400: 79.42, ep600: 79.44, ep800: 79.60
- P87_skernel_m4096_bw0.5_views4_800ep_seed1: ep0: 37.40, ep20: 61.38, ep50: 67.60, ep100: 74.52, ep200: 77.06, ep400: 80.22, ep600: 80.66, ep800: 80.86
- P87_skernel_m4096_bw0.5_views4_800ep_seed2: ep0: 37.08, ep20: 61.14, ep50: 67.76, ep100: 74.58, ep200: 77.72, ep400: 79.86, ep600: 80.58, ep800: 80.66

## Held-out J trajectory (VCS only)

- P87_skernel_m4096_bw0.5_views4_800ep_seed0: ep0: -0.0000, ep20: 0.7525, ep50: 0.8348, ep100: 0.8794, ep200: 0.8938, ep400: 0.9106, ep600: 0.9118, ep800: 0.9144
- P87_skernel_m4096_bw0.5_views4_800ep_seed1: ep0: 0.0000, ep20: 0.7649, ep50: 0.8367, ep100: 0.8713, ep200: 0.8921, ep400: 0.9140, ep600: 0.9248, ep800: 0.9270
- P87_skernel_m4096_bw0.5_views4_800ep_seed2: ep0: -0.0000, ep20: 0.7507, ep50: 0.8322, ep100: 0.8737, ep200: 0.8964, ep400: 0.9132, ep600: 0.9211, ep800: 0.9242

## Final-epoch training objective values (epoch means)

- P87_kcs_bw0.5_views4_800ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg None
- P87_kcs_bw0.5_views4_800ep_seed1: J_raw null, R_binary null, nt_xent null, vicreg None
- P87_kcs_bw0.5_views4_800ep_seed2: J_raw null, R_binary null, nt_xent null, vicreg None
- P87_skernel_m4096_bw0.5_views4_800ep_seed0: J_raw 0.9358, R_binary 0.0642, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m4096_bw0.5_views4_800ep_seed1: J_raw 0.9488, R_binary 0.0512, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P87_skernel_m4096_bw0.5_views4_800ep_seed2: J_raw 0.9453, R_binary 0.0547, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P87_kcs_bw0.5_views4_800ep_seed0 | 0.2512 | 1019 | 2038 | 35371 | 35 | 9 | 35840000 | None | 1013704@node61 |
| P87_kcs_bw0.5_views4_800ep_seed1 | 0.1990 | 1287 | 2573 | 28234 | 34 | 13 | 35840000 | None | 1013705@node53 |
| P87_kcs_bw0.5_views4_800ep_seed2 | 0.1942 | 1318 | 2636 | 27566 | 33 | 13 | 35840000 | None | 1013706@node53 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | 0.1165 | 2198 | 4395 | 16516 | 52 | 11 | 35840000 | 4097 | 1013701@node59 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | 0.1164 | 2199 | 4398 | 16503 | 52 | 11 | 35840000 | 4097 | 1013702@node59 |
| P87_skernel_m4096_bw0.5_views4_800ep_seed2 | 0.1160 | 2207 | 4415 | 16445 | 54 | 12 | 35840000 | 4097 | 1013703@node61 |

## Provenance

- P87_kcs_bw0.5_views4_800ep_seed0: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `55f7d1751cd9fe3c58828d951d69e449418a3378ed012538683882cb0dc5055f`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_kcs_bw0.5_views4_800ep_seed1: commit `478cc62c50be29e3d295a6b45a3a2493ee72f47f` dirty=True, config `7b6897e991abbe494adaa927996902c7270065c11e37ca1a2a87bbbe5af5d0b5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P87_kcs_bw0.5_views4_800ep_seed2: commit `478cc62c50be29e3d295a6b45a3a2493ee72f47f` dirty=True, config `d59401aee906dfec4a643288da501ac03e113fc5a3f6c44711fa37b4ed846e31`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P87_skernel_m4096_bw0.5_views4_800ep_seed0: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `c2591d62f34f1b59bafebb717e0af02e04aee22111e00781e94751fd0f7c9371`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P87_skernel_m4096_bw0.5_views4_800ep_seed1: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `2765b8a7e60685d633ba2d523e1b3b8b6837164385bb5d4efc25d49355375ac6`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P87_skernel_m4096_bw0.5_views4_800ep_seed2: commit `f54a7a2f7d6025202612dae2ee771a5a4c274b53` dirty=True, config `92c97365e4784b405122ca17c3350267c3cd3beeff2a0a00cdb308eadb56fcfd`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| cs_kernel_native | 3 | 78.56 ± 0.85 | 42.21 ± 0.67 | 36.35 ± 1.52 | 71.75 ± 2.82 | 8.54 ± 1.51 | null |
| vcs_qmi K=8 | 3 | 84.91 ± 0.15 | 42.20 ± 0.69 | 42.71 ± 0.61 | 80.37 ± 0.68 | 29.10 ± 15.63 | 0.92 ± 0.01 |

## Coverage checks

- P87_kcs_bw0.5_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P87_kcs_bw0.5_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P87_kcs_bw0.5_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P87_skernel_m4096_bw0.5_views4_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P87_skernel_m4096_bw0.5_views4_800ep_seed1: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P87_skernel_m4096_bw0.5_views4_800ep_seed2: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
