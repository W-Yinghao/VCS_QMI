# P33_vcs_views_compute — neutral results table (observed values only)

Generated 2026-09-25T22:20:13Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P33_vcs_views4_100ep_seed0 | vcs_qmi | 8 | 0 | 100/100 | 81.84 | 77.22 | 0.9451±0.0013 | 41.11 | 6081 | 7568/12672 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P33_vcs_views4_100ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 100 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P33_vcs_views4_100ep_seed0 | 41.78 | 81.84 | 40.06 | 36.58 | 77.22 | 40.64 | 2.94 | 41.11 | -0.5686 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P33_vcs_views4_100ep_seed0: ep0: 36.58, ep10: 61.06, ep20: 68.56, ep50: 74.26, ep100: 77.22

## Held-out J trajectory (VCS only)

- P33_vcs_views4_100ep_seed0: ep0: -0.5686, ep10: 0.6717, ep20: 0.8546, ep50: 0.9182, ep100: 0.9451

## Final-epoch training objective values (epoch means)

- P33_vcs_views4_100ep_seed0: J_raw 0.9500, R_binary 0.0500, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P33_vcs_views4_100ep_seed0 | 0.3435 | 745 | 1491 | 6081 | 97 | 30 | 4480000 | 2 | 1009174@nodeaudible01 |

## Provenance

- P33_vcs_views4_100ep_seed0: commit `96235df871ab962089c62854cb8ca5a4d6853fa0` dirty=False, config `9a3ea218a1a5ee33a744627207c0fed5f5bef92c9c6341b7d5b97c89d1924aea`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 1 | 81.84 | 41.78 | 40.06 | 77.22 | 41.11 | 0.95 |

## Coverage checks

- P33_vcs_views4_100ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_100.json), status COMPLETED, failure None
