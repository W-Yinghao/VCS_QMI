# P63_queue — neutral results table (observed values only)

Generated 2026-09-27T20:16:03Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P63_simclr_B16_queue4096_seed0 | simclr_matched | null | 0 | 200/200 | 10.58 | 10.66 | null | 1.26 | 4409 | 1387/2226 | COMPLETED COLLAPSE_SUSPECTED |
| P63_simclr_B32_queue4096_seed0 | simclr_matched | null | 0 | 200/200 | 10.74 | 10.60 | null | 2.24 | 6386 | 1385/2224 | COMPLETED |
| P63_vcs_B16_queue4096_seed0 | vcs_qmi | 8 | 0 | 200/200 | 28.72 | 34.00 | -0.5228±0.0015 | 1.41 | 8566 | 1541/2486 | COMPLETED COLLAPSE_SUSPECTED |
| P63_vcs_B32_queue4096_seed0 | vcs_qmi | 8 | 0 | 200/200 | 22.12 | 22.26 | -0.5506±0.0039 | 2.62 | 3166 | 1385/2224 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P63_simclr_B16_queue4096_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 16 | 6.25e-05 | 200 |
| P63_simclr_B32_queue4096_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 32 | 0.000125 | 200 |
| P63_vcs_B16_queue4096_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 16 | 6.25e-05 | 200 |
| P63_vcs_B32_queue4096_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 32 | 0.000125 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P63_simclr_B16_queue4096_seed0 | 41.94 | 10.58 | -31.36 | 36.54 | 10.66 | -25.88 | 2.94 | 1.26 | null |
| P63_simclr_B32_queue4096_seed0 | 41.94 | 10.74 | -31.20 | 36.54 | 10.60 | -25.94 | 2.94 | 2.24 | null |
| P63_vcs_B16_queue4096_seed0 | 41.98 | 28.72 | -13.26 | 36.58 | 34.00 | -2.58 | 2.94 | 1.41 | -0.9998 |
| P63_vcs_B32_queue4096_seed0 | 41.94 | 22.12 | -19.82 | 36.54 | 22.26 | -14.28 | 2.94 | 2.62 | -0.9998 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P63_simclr_B16_queue4096_seed0: ep0: 36.54, ep10: 35.52, ep20: 34.44, ep50: 30.72, ep100: 11.14, ep150: 10.98, ep200: 10.56
- P63_simclr_B32_queue4096_seed0: ep0: 36.54, ep10: 33.06, ep20: 30.90, ep50: 26.64, ep100: 18.24, ep150: 11.34, ep200: 10.54
- P63_vcs_B16_queue4096_seed0: ep0: 36.58, ep10: 41.78, ep20: 43.46, ep50: 36.44, ep100: 36.62, ep150: 34.76, ep200: 33.96
- P63_vcs_B32_queue4096_seed0: ep0: 36.54, ep10: 32.14, ep20: 38.98, ep50: 31.76, ep100: 22.16, ep150: 22.94, ep200: 22.40

## Held-out J trajectory (VCS only)

- P63_vcs_B16_queue4096_seed0: ep0: -0.9998, ep10: -0.2131, ep20: -0.3249, ep50: -0.0933, ep100: -0.2809, ep150: -0.5366, ep200: -0.5228
- P63_vcs_B32_queue4096_seed0: ep0: -0.9998, ep10: -0.5288, ep20: -0.4735, ep50: -0.3657, ep100: -0.6028, ep150: -0.6157, ep200: -0.5506

## Final-epoch training objective values (epoch means)

- P63_simclr_B16_queue4096_seed0: J_raw null, R_binary null, nt_xent 3.9257, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P63_simclr_B32_queue4096_seed0: J_raw null, R_binary null, nt_xent 4.0656, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P63_vcs_B16_queue4096_seed0: J_raw 0.9669, R_binary 0.0331, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P63_vcs_B32_queue4096_seed0: J_raw 0.9646, R_binary 0.0354, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P63_simclr_B16_queue4096_seed0 | 0.0077 | 2071 | 4141 | 4409 | 20 | 7 | 8998400 | None | 1011314@node59 |
| P63_simclr_B32_queue4096_seed0 | 0.0226 | 1416 | 2831 | 6386 | 31 | 9 | 8998400 | None | 1011510@node60 |
| P63_vcs_B16_queue4096_seed0 | 0.0151 | 1060 | 2121 | 8566 | 84 | 20 | 8998400 | 2 | 1011131@nodesumo01 |
| P63_vcs_B32_queue4096_seed0 | 0.0111 | 2871 | 5742 | 3166 | 46 | 11 | 8998400 | 2 | 1011424@node58 |

## Provenance

- P63_simclr_B16_queue4096_seed0: commit `fb271e53e56a732db40846bb20e596c5feac682b` dirty=True, config `f27cc8bf38a185a955e9865e1bdc53dea1c2d8ea76f7c5761648f90d37543799`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P63_simclr_B32_queue4096_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `e1d51a07db66b5f57f1aef56797e0de57345f91e25035a7e6a64a896db9c1a10`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P63_vcs_B16_queue4096_seed0: commit `8cd6125593196e960f665e0e8a80f4bc11c7b08d` dirty=True, config `1c0cccd1acbcdc34d538e2139d30e79bd97db54a5fe621bfaa47e6a12209b6f5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P63_vcs_B32_queue4096_seed0: commit `062ae89b058ef50bbcdf9fdc4870544e8482c40c` dirty=True, config `f2620fb9d431e49eed4265ded037d4a3f129dbc6668a829d205776803578e331`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 2 | 10.66 ± 0.11 | 41.94 ± 0.00 | -31.28 ± 0.11 | 10.63 ± 0.04 | 1.75 ± 0.69 | null |
| vcs_qmi K=8 | 2 | 25.42 ± 4.67 | 41.96 ± 0.03 | -16.54 ± 4.64 | 28.13 ± 8.30 | 2.01 ± 0.85 | -0.54 ± 0.02 |

## Coverage checks

- P63_simclr_B16_queue4096_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P63_simclr_B32_queue4096_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P63_vcs_B16_queue4096_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P63_vcs_B32_queue4096_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
