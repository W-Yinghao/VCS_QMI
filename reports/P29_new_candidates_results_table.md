# P28_vcs_new_candidates — neutral results table (observed values only)

Generated 2026-09-25T21:35:18Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P28_vcs_diag_metric_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.56 | 74.02 | 0.9253±0.0017 | 27.47 | 5657 | 3683/4844 | COMPLETED |
| P28_vcs_mono_spline_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.46 | 73.90 | 0.9190±0.0017 | 24.98 | 6219 | 3683/4842 | COMPLETED |
| P28_vcs_proj_bnonly_seed0 | vcs_qmi | 8 | 0 | 82/200 | null | null | null | null | null | null/null | RUNNING |
| P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925 | vcs_qmi | 8 | 0 | 0/200 | null | null | null | null | 0 | 3559/4826 | FAILED_INFRA |
| P28_vcs_proj_linear_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.02 | 75.74 | 0.9327±0.0017 | 30.49 | 6310 | 3680/4836 | COMPLETED |
| P28_vcs_views4_seed0 | vcs_qmi | 8 | 0 | 200/200 | 84.48 | 81.10 | 0.9597±0.0013 | 57.42 | 12238 | 7568/12672 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P28_vcs_diag_metric_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | DiagMetricCritic | 256 | 0.001 | 200 |
| P28_vcs_mono_spline_seed0 | 8 | [8, 8] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | MonoSplineCritic | 256 | 0.001 | 200 |
| P28_vcs_proj_bnonly_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 512 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 512 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P28_vcs_proj_linear_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P28_vcs_views4_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P28_vcs_diag_metric_seed0 | 41.78 | 80.56 | 38.78 | 36.58 | 74.02 | 37.44 | 2.94 | 27.47 | -0.5686 |
| P28_vcs_mono_spline_seed0 | 41.78 | 80.46 | 38.68 | 36.58 | 73.90 | 37.32 | 2.94 | 24.98 | -0.5701 |
| P28_vcs_proj_bnonly_seed0 | null | null | null | null | null | null | null | null | null |
| P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925 | null | null | null | null | null | null | null | null | null |
| P28_vcs_proj_linear_seed0 | 41.78 | 80.02 | 38.24 | 36.58 | 75.74 | 39.16 | 2.94 | 30.49 | -0.5704 |
| P28_vcs_views4_seed0 | 41.78 | 84.48 | 42.70 | 36.58 | 81.10 | 44.52 | 2.94 | 57.42 | -0.5686 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P28_vcs_diag_metric_seed0: ep0: 36.58, ep10: 53.60, ep20: 61.02, ep50: 67.82, ep100: 71.60, ep150: 73.56, ep200: 74.02
- P28_vcs_mono_spline_seed0: ep0: 36.58, ep10: 53.52, ep20: 59.68, ep50: 66.76, ep100: 71.78, ep150: 73.60, ep200: 73.90
- P28_vcs_proj_bnonly_seed0: ep0: 36.58, ep10: 56.28, ep20: 62.68, ep50: 69.58
- P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925: ep0: 36.58
- P28_vcs_proj_linear_seed0: ep0: 36.58, ep10: 55.18, ep20: 62.50, ep50: 69.20, ep100: 73.64, ep150: 75.50, ep200: 75.74
- P28_vcs_views4_seed0: ep0: 36.58, ep10: 61.18, ep20: 68.96, ep50: 74.80, ep100: 78.56, ep150: 80.40, ep200: 81.10

## Held-out J trajectory (VCS only)

- P28_vcs_diag_metric_seed0: ep0: -0.5686, ep10: 0.6217, ep20: 0.7788, ep50: 0.8541, ep100: 0.8993, ep150: 0.9172, ep200: 0.9253
- P28_vcs_mono_spline_seed0: ep0: -0.5701, ep10: 0.4818, ep20: 0.6787, ep50: 0.8396, ep100: 0.8894, ep150: 0.9117, ep200: 0.9190
- P28_vcs_proj_bnonly_seed0: ep0: -0.5696, ep10: 0.6634, ep20: 0.7973, ep50: 0.8971
- P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925: ep0: -0.5696
- P28_vcs_proj_linear_seed0: ep0: -0.5704, ep10: 0.6415, ep20: 0.7798, ep50: 0.8675, ep100: 0.9090, ep150: 0.9259, ep200: 0.9327
- P28_vcs_views4_seed0: ep0: -0.5686, ep10: 0.6496, ep20: 0.8447, ep50: 0.9141, ep100: 0.9421, ep150: 0.9554, ep200: 0.9597

## Final-epoch training objective values (epoch means)

- P28_vcs_diag_metric_seed0: J_raw 0.9291, R_binary 0.0709, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P28_vcs_mono_spline_seed0: J_raw 0.9232, R_binary 0.0768, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P28_vcs_proj_bnonly_seed0: J_raw 0.9238, R_binary 0.0762, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925: J_raw null, R_binary null, nt_xent null, vicreg None
- P28_vcs_proj_linear_seed0: J_raw 0.9374, R_binary 0.0626, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P28_vcs_views4_seed0: J_raw 0.9660, R_binary 0.0340, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P28_vcs_diag_metric_seed0 | 0.1594 | 1606 | 3213 | 5657 | 136 | 29 | 8960000 | 130 | 1009115@nodeaudible01 |
| P28_vcs_mono_spline_seed0 | 0.1756 | 1458 | 2916 | 6219 | 150 | 35 | 8960000 | 9 | 1009114@node55 |
| P28_vcs_proj_bnonly_seed0 | null | null | null | null | null | null | None | 2 | 1009178@node01 |
| P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925 | null | null | null | 0 | 21 | null | 0 | 2 | 1009122@node54 |
| P28_vcs_proj_linear_seed0 | 0.1778 | 1440 | 2879 | 6310 | 160 | 37 | 8960000 | 2 | 1009119@node06 |
| P28_vcs_views4_seed0 | 0.3462 | 740 | 1479 | 12238 | 135 | 31 | 8960000 | 2 | 1009117@nodeaudible01 |

## Provenance

- P28_vcs_diag_metric_seed0: commit `4cb554f21216bfc3e42873a45b8361c0bf4048da` dirty=True, config `6b7f25f74602d35a8a7e22bf826695495fff3db053ae864aab55654689f24c00`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P28_vcs_mono_spline_seed0: commit `7950a76d3199390d3dcf151c2716db0f650cf830` dirty=False, config `33bde565921d07c96d3394aecc10d392dfd99865877569c02d588b64d34806d5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P28_vcs_proj_bnonly_seed0: commit `96235df871ab962089c62854cb8ca5a4d6853fa0` dirty=True, config `8b9b15bff13e3c362fe791370a585c338e36deb2340e784f6f720b86bcb7bfaa`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925: commit `82e0bab93b706283a97fb2aff97379d0fb8977e5` dirty=True, config `8b9b15bff13e3c362fe791370a585c338e36deb2340e784f6f720b86bcb7bfaa`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P28_vcs_proj_linear_seed0: commit `82e0bab93b706283a97fb2aff97379d0fb8977e5` dirty=True, config `9c463b5eb780d4732234f17387a73c29f4c45008abfd75eab08791f67501b62e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P28_vcs_views4_seed0: commit `aa880996f65d62b258fe8e5704429418f1a46968` dirty=True, config `ff952e5ca48cf7b202e1e8db00703b4c3efd0a1a9b331c575bbc1feccc2e27eb`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 4 | 81.38 ± 2.08 | 41.78 ± 0.00 | 39.60 ± 2.08 | 76.19 ± 3.38 | 35.09 ± 15.05 | 0.93 ± 0.02 |

## Coverage checks

- P28_vcs_diag_metric_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P28_vcs_mono_spline_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P28_vcs_proj_bnonly_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P28_vcs_proj_bnonly_seed0_FAILED_attempt1_20260925: epoch0 eval False, final eval False (None), status FAILED_INFRA, failure RuntimeError: first-step gradient check failed: projector gradient norm None
- P28_vcs_proj_linear_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P28_vcs_views4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
