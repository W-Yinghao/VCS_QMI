# P3_pilot — neutral results table (observed values only)

Generated 2026-09-24T16:08:48Z. Failed / stopped runs are listed, never dropped.

| run | method | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|
| P3_simclr_seed0 | simclr_matched | 0 | 20/20 | 74.98 | 67.90 | null | 29.13 | 452 | 2787/3516 | COMPLETED |
| P3_vcs_seed0 | vcs_qmi | 0 | 20/20 | 56.26 | 41.08 | 0.6766±0.0046 | 5.79 | 628 | 3687/4848 | COMPLETED |
| P3_vicreg_seed0 | vicreg_matched_128 | 0 | 20/20 | 75.58 | 68.20 | null | 24.81 | 453 | 2787/3516 | COMPLETED |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val ep20 (%) | Δ linear | kNN ep0 (%) | kNN ep20 (%) | Δ kNN | h-rank ep0 | h-rank ep20 | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P3_simclr_seed0 | 41.94 | 74.98 | 33.04 | 36.54 | 67.90 | 31.36 | 2.94 | 29.13 | null |
| P3_vcs_seed0 | 41.78 | 56.26 | 14.48 | 36.58 | 41.08 | 4.50 | 2.94 | 5.79 | -0.0000 |
| P3_vicreg_seed0 | 41.94 | 75.58 | 33.64 | 36.54 | 68.20 | 31.66 | 2.94 | 24.81 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P3_simclr_seed0: ep0: 36.54, ep5: 58.24, ep10: 64.80, ep20: 67.90
- P3_vcs_seed0: ep0: 36.58, ep5: 35.64, ep10: 38.28, ep20: 41.08
- P3_vicreg_seed0: ep0: 36.54, ep5: 54.48, ep10: 63.40, ep20: 68.20

## Held-out J trajectory (VCS only)

- P3_vcs_seed0: ep0: -0.0000, ep5: 0.4578, ep10: 0.6094, ep20: 0.6766

## Final-epoch training objective values (epoch means)

- P3_simclr_seed0: J_raw null, R_binary null, nt_xent 2.6169, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P3_vcs_seed0: J_raw 0.6781, R_binary 0.3219, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P3_vicreg_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.18399963762078966, 'vicreg_variance': 0.17193299455302102, 'vicreg_covariance': 2.845895106451852}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | ep20 eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P3_simclr_seed0 | 0.1280 | 2000 | 4001 | 452 | 18 | 9 | 896000 | None | 1007758@node60 |
| P3_vcs_seed0 | 0.1772 | 1445 | 2889 | 628 | 82 | 35 | 896000 | 394753 | 1007757@node54 |
| P3_vicreg_seed0 | 0.1286 | 1990 | 3980 | 453 | 18 | 9 | 896000 | None | 1007759@node60 |

## Provenance

- P3_simclr_seed0: commit `163a91223ef4ebb89d4edae1ce17402a934d54c4` dirty=True, config `de51efd8a0f9dcc81628f1c2a6f9dc7c62675ff2bf50c0bc63b163ba4e153d5c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P3_vcs_seed0: commit `57ac0c83026bf15fb2ac8518f6bc576e5a6c1417` dirty=True, config `cfe89416b54381ff870379c9d6cb734f1111c8fbe6d23ffb6f57426af6c8407a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P3_vicreg_seed0: commit `163a91223ef4ebb89d4edae1ce17402a934d54c4` dirty=True, config `0cff83f7f9480a23331af29359476e8887f424ad2b10254766803f0f5b44b4ec`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Coverage checks

- P3_simclr_seed0: epoch0 eval True, final eval True, status COMPLETED, failure None
- P3_vcs_seed0: epoch0 eval True, final eval True, status COMPLETED, failure None
- P3_vicreg_seed0: epoch0 eval True, final eval True, status COMPLETED, failure None
