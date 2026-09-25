# P14_vcs_hparamB_lr — neutral results table (observed values only)

Generated 2026-09-25T05:30:27Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P14_vcs_const_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.88 | 64.40 | 0.8852±0.0045 | 15.01 | 6256 | 3687/4848 | COMPLETED |
| P14_vcs_floor0.1_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.20 | 63.82 | 0.8966±0.0055 | 13.72 | 2039 | 2793/3524 | COMPLETED |
| P14_vcs_lr1e-2_seed0 | vcs_qmi | 1 | 0 | 200/200 | 71.38 | 63.54 | 0.9033±0.0052 | 13.68 | 3222 | 3648/4336 | COMPLETED |
| P14_vcs_lr3e-3_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.74 | 64.24 | 0.9048±0.0060 | 14.60 | 3177 | 3648/4336 | COMPLETED |
| P14_vcs_lr3e-4_seed0 | vcs_qmi | 1 | 0 | 200/200 | 72.80 | 63.32 | 0.9024±0.0035 | 14.52 | 2035 | 2793/3524 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P14_vcs_const_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P14_vcs_floor0.1_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P14_vcs_lr1e-2_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.01 | 200 |
| P14_vcs_lr3e-3_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.003 | 200 |
| P14_vcs_lr3e-4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.0003 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P14_vcs_const_seed0 | 41.78 | 74.88 | 33.10 | 36.58 | 64.40 | 27.82 | 2.94 | 15.01 | -0.0000 |
| P14_vcs_floor0.1_seed0 | 41.94 | 74.20 | 32.26 | 36.54 | 63.82 | 27.28 | 2.94 | 13.72 | -0.0000 |
| P14_vcs_lr1e-2_seed0 | 41.98 | 71.38 | 29.40 | 36.58 | 63.54 | 26.96 | 2.94 | 13.68 | -0.0000 |
| P14_vcs_lr3e-3_seed0 | 41.98 | 73.74 | 31.76 | 36.58 | 64.24 | 27.66 | 2.94 | 14.60 | -0.0000 |
| P14_vcs_lr3e-4_seed0 | 41.94 | 72.80 | 30.86 | 36.54 | 63.32 | 26.78 | 2.94 | 14.52 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P14_vcs_const_seed0: ep0: 36.58, ep10: 42.68, ep20: 46.96, ep50: 55.86, ep100: 60.32, ep150: 62.72, ep200: 64.40
- P14_vcs_floor0.1_seed0: ep0: 36.54, ep10: 43.60, ep20: 47.36, ep50: 55.98, ep100: 60.92, ep150: 63.10, ep200: 63.82
- P14_vcs_lr1e-2_seed0: ep0: 36.58, ep10: 36.78, ep20: 44.34, ep50: 56.80, ep100: 59.72, ep150: 63.24, ep200: 63.54
- P14_vcs_lr3e-3_seed0: ep0: 36.58, ep10: 42.44, ep20: 47.10, ep50: 56.60, ep100: 61.06, ep150: 63.98, ep200: 64.24
- P14_vcs_lr3e-4_seed0: ep0: 36.54, ep10: 40.98, ep20: 48.16, ep50: 55.36, ep100: 60.82, ep150: 62.50, ep200: 63.32

## Held-out J trajectory (VCS only)

- P14_vcs_const_seed0: ep0: -0.0000, ep10: 0.6223, ep20: 0.7122, ep50: 0.8075, ep100: 0.8497, ep150: 0.8756, ep200: 0.8852
- P14_vcs_floor0.1_seed0: ep0: -0.0000, ep10: 0.6229, ep20: 0.7207, ep50: 0.8017, ep100: 0.8604, ep150: 0.8854, ep200: 0.8966
- P14_vcs_lr1e-2_seed0: ep0: -0.0000, ep10: 0.5199, ep20: 0.6631, ep50: 0.8044, ep100: 0.8573, ep150: 0.8941, ep200: 0.9033
- P14_vcs_lr3e-3_seed0: ep0: -0.0000, ep10: 0.6034, ep20: 0.6871, ep50: 0.8090, ep100: 0.8657, ep150: 0.8963, ep200: 0.9048
- P14_vcs_lr3e-4_seed0: ep0: -0.0000, ep10: 0.5845, ep20: 0.7099, ep50: 0.8103, ep100: 0.8612, ep150: 0.8920, ep200: 0.9024

## Final-epoch training objective values (epoch means)

- P14_vcs_const_seed0: J_raw 0.8949, R_binary 0.1051, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P14_vcs_floor0.1_seed0: J_raw 0.9028, R_binary 0.0972, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P14_vcs_lr1e-2_seed0: J_raw 0.9087, R_binary 0.0913, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P14_vcs_lr3e-3_seed0: J_raw 0.9104, R_binary 0.0896, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P14_vcs_lr3e-4_seed0: J_raw 0.9070, R_binary 0.0930, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P14_vcs_const_seed0 | 0.1766 | 1450 | 2900 | 6256 | 146 | 35 | 8960000 | 394753 | 1008192@node56 |
| P14_vcs_floor0.1_seed0 | 0.0574 | 4463 | 8926 | 2039 | 45 | 11 | 8960000 | 394753 | 1008191@node59 |
| P14_vcs_lr1e-2_seed0 | 0.0907 | 2822 | 5644 | 3222 | 79 | 20 | 8960000 | 394753 | 1008189@node53 |
| P14_vcs_lr3e-3_seed0 | 0.0895 | 2862 | 5723 | 3177 | 79 | 20 | 8960000 | 394753 | 1008188@node53 |
| P14_vcs_lr3e-4_seed0 | 0.0573 | 4470 | 8939 | 2035 | 47 | 11 | 8960000 | 394753 | 1008190@node59 |

## Provenance

- P14_vcs_const_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `cb2f08792e358d70737214570d981f0f68ff13706879b99789bed02dd05cf73b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P14_vcs_floor0.1_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `2740319c21b83ad64b44570673d82731d5419e1b92be4b00afbd71462f443235`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P14_vcs_lr1e-2_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `ddbed532f101e759f471c6443b0ff7becb0e5185c9857fa3747fca0399d0eb2c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P14_vcs_lr3e-3_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `a83c974211ee440e715a91a6a49045085055e93d3c4d30417faa9c27359aa010`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P14_vcs_lr3e-4_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `2ce83cf2928096185691637fd6017a30ef7b1b45c2695b082e9be7f2f025f763`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=1 | 5 | 73.40 ± 1.36 | 41.92 ± 0.08 | 31.48 ± 1.42 | 63.86 ± 0.46 | 14.31 ± 0.59 | 0.90 ± 0.01 |

## Coverage checks

- P14_vcs_const_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P14_vcs_floor0.1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P14_vcs_lr1e-2_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P14_vcs_lr3e-3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P14_vcs_lr3e-4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
