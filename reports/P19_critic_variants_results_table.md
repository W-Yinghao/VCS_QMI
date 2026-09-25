# P18_vcs_critic_variants — neutral results table (observed values only)

Generated 2026-09-25T09:58:29Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P18_vcs_crit_bilinear_seed0 | vcs_qmi | 8 | 0 | 200/200 | 76.18 | 66.76 | 0.9306±0.0023 | 16.68 | 6299 | 3687/4850 | COMPLETED |
| P18_vcs_crit_cosine_seed0 | vcs_qmi | 8 | 0 | 200/200 | 78.32 | 72.76 | 0.9676±0.0010 | 57.74 | 6236 | 3683/4842 | COMPLETED |
| P18_vcs_crit_interact_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.34 | 70.74 | 0.9623±0.0010 | 39.28 | 6234 | 3689/4854 | COMPLETED |
| P18_vcs_crit_steps2_seed0 | vcs_qmi | 8 | 0 | 200/200 | 76.76 | 67.54 | 0.9312±0.0023 | 18.06 | 5754 | 3687/4848 | COMPLETED |
| P18_vcs_crit_steps5_seed0 | vcs_qmi | 8 | 0 | 200/200 | 75.62 | 68.80 | 0.9371±0.0026 | 21.08 | 6134 | 3687/4848 | COMPLETED |
| P18_vcs_pair_sym_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.08 | 67.60 | 0.9307±0.0025 | 17.00 | 5845 | 3688/4848 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P18_vcs_crit_bilinear_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | BilinearConcatCritic | 256 | 0.001 | 200 |
| P18_vcs_crit_cosine_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P18_vcs_crit_interact_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | InteractCritic | 256 | 0.001 | 200 |
| P18_vcs_crit_steps2_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P18_vcs_crit_steps5_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P18_vcs_pair_sym_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P18_vcs_crit_bilinear_seed0 | 41.78 | 76.18 | 34.40 | 36.58 | 66.76 | 30.18 | 2.94 | 16.68 | -0.0000 |
| P18_vcs_crit_cosine_seed0 | 41.78 | 78.32 | 36.54 | 36.58 | 72.76 | 36.18 | 2.94 | 57.74 | -0.5686 |
| P18_vcs_crit_interact_seed0 | 41.78 | 77.34 | 35.56 | 36.58 | 70.74 | 34.16 | 2.94 | 39.28 | -0.0000 |
| P18_vcs_crit_steps2_seed0 | 41.78 | 76.76 | 34.98 | 36.58 | 67.54 | 30.96 | 2.94 | 18.06 | -0.0000 |
| P18_vcs_crit_steps5_seed0 | 41.78 | 75.62 | 33.84 | 36.58 | 68.80 | 32.22 | 2.94 | 21.08 | -0.0000 |
| P18_vcs_pair_sym_seed0 | 41.78 | 77.08 | 35.30 | 36.58 | 67.60 | 31.02 | 2.94 | 17.00 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P18_vcs_crit_bilinear_seed0: ep0: 36.58, ep10: 47.02, ep20: 53.02, ep50: 61.24, ep100: 65.04, ep150: 66.62, ep200: 66.76
- P18_vcs_crit_cosine_seed0: ep0: 36.58, ep10: 56.54, ep20: 64.00, ep50: 69.16, ep100: 71.72, ep150: 72.84, ep200: 72.76
- P18_vcs_crit_interact_seed0: ep0: 36.58, ep10: 56.08, ep20: 61.40, ep50: 66.96, ep100: 69.70, ep150: 69.82, ep200: 70.74
- P18_vcs_crit_steps2_seed0: ep0: 36.58, ep10: 49.94, ep20: 54.90, ep50: 61.72, ep100: 65.02, ep150: 67.22, ep200: 67.54
- P18_vcs_crit_steps5_seed0: ep0: 36.58, ep10: 50.82, ep20: 56.12, ep50: 62.86, ep100: 65.92, ep150: 68.38, ep200: 68.80
- P18_vcs_pair_sym_seed0: ep0: 36.58, ep10: 46.88, ep20: 53.84, ep50: 61.74, ep100: 64.96, ep150: 66.88, ep200: 67.60

## Held-out J trajectory (VCS only)

- P18_vcs_crit_bilinear_seed0: ep0: -0.0000, ep10: 0.6825, ep20: 0.7709, ep50: 0.8616, ep100: 0.9019, ep150: 0.9239, ep200: 0.9306
- P18_vcs_crit_cosine_seed0: ep0: -0.5686, ep10: 0.6801, ep20: 0.8241, ep50: 0.9132, ep100: 0.9479, ep150: 0.9636, ep200: 0.9676
- P18_vcs_crit_interact_seed0: ep0: -0.0000, ep10: 0.8015, ep20: 0.8653, ep50: 0.9114, ep100: 0.9419, ep150: 0.9576, ep200: 0.9623
- P18_vcs_crit_steps2_seed0: ep0: -0.0000, ep10: 0.7031, ep20: 0.7374, ep50: 0.8600, ep100: 0.9034, ep150: 0.9250, ep200: 0.9312
- P18_vcs_crit_steps5_seed0: ep0: -0.0000, ep10: 0.7102, ep20: 0.7784, ep50: 0.8604, ep100: 0.8985, ep150: 0.9299, ep200: 0.9371
- P18_vcs_pair_sym_seed0: ep0: -0.0000, ep10: 0.7001, ep20: 0.7610, ep50: 0.8575, ep100: 0.9055, ep150: 0.9244, ep200: 0.9307

## Final-epoch training objective values (epoch means)

- P18_vcs_crit_bilinear_seed0: J_raw 0.9349, R_binary 0.0651, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P18_vcs_crit_cosine_seed0: J_raw 0.9706, R_binary 0.0294, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P18_vcs_crit_interact_seed0: J_raw 0.9672, R_binary 0.0328, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P18_vcs_crit_steps2_seed0: J_raw 0.9344, R_binary 0.0656, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P18_vcs_crit_steps5_seed0: J_raw 0.9403, R_binary 0.0597, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P18_vcs_pair_sym_seed0: J_raw 0.9364, R_binary 0.0636, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P18_vcs_crit_bilinear_seed0 | 0.1778 | 1440 | 2880 | 6299 | 146 | 35 | 8960000 | 411137 | 1008402@node56 |
| P18_vcs_crit_cosine_seed0 | 0.1760 | 1454 | 2908 | 6236 | 144 | 36 | 8960000 | 2 | 1008403@node56 |
| P18_vcs_crit_interact_seed0 | 0.1759 | 1455 | 2910 | 6234 | 147 | 35 | 8960000 | 525825 | 1008401@node56 |
| P18_vcs_crit_steps2_seed0 | 0.1623 | 1577 | 3154 | 5754 | 137 | 30 | 8960000 | 394753 | 1008405@nodeaudible01 |
| P18_vcs_crit_steps5_seed0 | 0.1733 | 1477 | 2954 | 6134 | 133 | 29 | 8960000 | 394753 | 1008406@nodeaudible01 |
| P18_vcs_pair_sym_seed0 | 0.1650 | 1551 | 3103 | 5845 | 134 | 31 | 8960000 | 394753 | 1008404@nodeaudible01 |

## Provenance

- P18_vcs_crit_bilinear_seed0: commit `3a044e4c49cd0d105d365a2404d0b52462a7a12b` dirty=True, config `2b7cb1bb49a3dc8cfb4f4f8811775aaaaadb543451e3c035eb5942bddb1b800b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P18_vcs_crit_cosine_seed0: commit `3a044e4c49cd0d105d365a2404d0b52462a7a12b` dirty=True, config `0c3ce18263cc6dd95099b048d1aa96dc1533d165e3d8dc516e72933e5f7e649d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P18_vcs_crit_interact_seed0: commit `3a044e4c49cd0d105d365a2404d0b52462a7a12b` dirty=True, config `2dfedd2c9c3b5a221cce89f04b9465d53b76ad5fcf58cd11820ac766c03bcf22`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P18_vcs_crit_steps2_seed0: commit `3a044e4c49cd0d105d365a2404d0b52462a7a12b` dirty=True, config `ae00c5f3f8918be8a3a59b2ff3a6aa805b17ba321c2623bf5a3e8d454ae1a2a2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P18_vcs_crit_steps5_seed0: commit `3a044e4c49cd0d105d365a2404d0b52462a7a12b` dirty=True, config `8fd7bd5ae9cacab4471397f27c17161d2bbd72ff5d6a306de036d69b11a2fffe`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P18_vcs_pair_sym_seed0: commit `3a044e4c49cd0d105d365a2404d0b52462a7a12b` dirty=True, config `3aae1c77d475fc8baad84c628c638bde1e82b48568fb3fcd7ff1db5e2a31804a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 6 | 76.88 ± 0.94 | 41.78 ± 0.00 | 35.10 ± 0.94 | 69.03 ± 2.29 | 28.31 ± 16.78 | 0.94 ± 0.02 |

## Coverage checks

- P18_vcs_crit_bilinear_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P18_vcs_crit_cosine_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P18_vcs_crit_interact_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P18_vcs_crit_steps2_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P18_vcs_crit_steps5_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P18_vcs_pair_sym_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
