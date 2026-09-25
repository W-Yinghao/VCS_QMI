# P29_vcs_seedfill — neutral results table (observed values only)

Generated 2026-09-25T20:23:38Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P29_vcs_cosK255_seed1 | vcs_qmi | 255 | 1 | 200/200 | 78.76 | 74.00 | 0.9687±0.0007 | 59.53 | 6260 | 3683/4844 | COMPLETED |
| P29_vcs_cosK255_seed2 | vcs_qmi | 255 | 2 | 200/200 | 79.18 | 74.28 | 0.9690±0.0012 | 61.94 | 5725 | 3683/4844 | COMPLETED |
| P29_vcs_cosK8_seed1 | vcs_qmi | 8 | 1 | 200/200 | 78.06 | 73.00 | 0.9671±0.0015 | 57.05 | 6205 | 3683/4842 | COMPLETED |
| P29_vcs_cosK8_seed2 | vcs_qmi | 8 | 2 | 200/200 | 78.24 | 73.52 | 0.9664±0.0008 | 58.45 | 6228 | 3683/4842 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P29_vcs_cosK255_seed1 | 255 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P29_vcs_cosK255_seed2 | 255 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P29_vcs_cosK8_seed1 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P29_vcs_cosK8_seed2 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P29_vcs_cosK255_seed1 | 41.60 | 78.76 | 37.16 | 37.42 | 74.00 | 36.58 | 3.22 | 59.53 | -0.5658 |
| P29_vcs_cosK255_seed2 | 43.02 | 79.18 | 36.16 | 37.08 | 74.28 | 37.20 | 3.22 | 61.94 | -0.5638 |
| P29_vcs_cosK8_seed1 | 41.60 | 78.06 | 36.46 | 37.42 | 73.00 | 35.58 | 3.22 | 57.05 | -0.5658 |
| P29_vcs_cosK8_seed2 | 43.02 | 78.24 | 35.22 | 37.08 | 73.52 | 36.44 | 3.22 | 58.45 | -0.5639 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P29_vcs_cosK255_seed1: ep0: 37.42, ep10: 57.00, ep20: 64.76, ep50: 68.98, ep100: 71.74, ep150: 73.42, ep200: 74.00
- P29_vcs_cosK255_seed2: ep0: 37.08, ep10: 57.52, ep20: 65.46, ep50: 69.86, ep100: 72.20, ep150: 73.62, ep200: 74.28
- P29_vcs_cosK8_seed1: ep0: 37.42, ep10: 54.60, ep20: 63.66, ep50: 68.30, ep100: 71.08, ep150: 72.74, ep200: 73.00
- P29_vcs_cosK8_seed2: ep0: 37.08, ep10: 55.82, ep20: 64.12, ep50: 68.48, ep100: 71.44, ep150: 73.28, ep200: 73.52

## Held-out J trajectory (VCS only)

- P29_vcs_cosK255_seed1: ep0: -0.5658, ep10: 0.6721, ep20: 0.8321, ep50: 0.9157, ep100: 0.9508, ep150: 0.9647, ep200: 0.9687
- P29_vcs_cosK255_seed2: ep0: -0.5638, ep10: 0.6915, ep20: 0.8222, ep50: 0.9187, ep100: 0.9512, ep150: 0.9646, ep200: 0.9690
- P29_vcs_cosK8_seed1: ep0: -0.5658, ep10: 0.6615, ep20: 0.8304, ep50: 0.9103, ep100: 0.9490, ep150: 0.9628, ep200: 0.9671
- P29_vcs_cosK8_seed2: ep0: -0.5639, ep10: 0.6666, ep20: 0.8208, ep50: 0.9133, ep100: 0.9483, ep150: 0.9619, ep200: 0.9664

## Final-epoch training objective values (epoch means)

- P29_vcs_cosK255_seed1: J_raw 0.9732, R_binary 0.0268, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P29_vcs_cosK255_seed2: J_raw 0.9727, R_binary 0.0273, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P29_vcs_cosK8_seed1: J_raw 0.9711, R_binary 0.0289, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P29_vcs_cosK8_seed2: J_raw 0.9695, R_binary 0.0305, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P29_vcs_cosK255_seed1 | 0.1767 | 1449 | 2898 | 6260 | 149 | 35 | 8960000 | 2 | 1009118@node54 |
| P29_vcs_cosK255_seed2 | 0.1615 | 1586 | 3171 | 5725 | 136 | 30 | 8960000 | 2 | 1009120@nodeaudible01 |
| P29_vcs_cosK8_seed1 | 0.1751 | 1462 | 2924 | 6205 | 145 | 35 | 8960000 | 2 | 1009113@node54 |
| P29_vcs_cosK8_seed2 | 0.1758 | 1456 | 2913 | 6228 | 147 | 35 | 8960000 | 2 | 1009116@node54 |

## Provenance

- P29_vcs_cosK255_seed1: commit `aa880996f65d62b258fe8e5704429418f1a46968` dirty=True, config `265b3c691afc12036950db0722fb142cf3aa5a3cd6201d88f1949817e756bd5f`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P29_vcs_cosK255_seed2: commit `82e0bab93b706283a97fb2aff97379d0fb8977e5` dirty=True, config `f9023ac924f394e484f643fad74c5c2cc38584fffe4d52a280605dc217090c35`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`
- P29_vcs_cosK8_seed1: commit `7950a76d3199390d3dcf151c2716db0f650cf830` dirty=False, config `f675afb5be73646b14af09e6a39a38d4f88532544baa6c966bd40ad391325803`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `94c2cb882a7ebf80a65d48352d007cf5fd956df2f37cd9a10980d83b11f4a51a`
- P29_vcs_cosK8_seed2: commit `aa880996f65d62b258fe8e5704429418f1a46968` dirty=True, config `45b78b32e387df6e66371df037f0c87636848c787331fed76f8bf078381753f2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `4385db04f2c25cafec319bb8e1dc5725215e7eebf11ab2727a37174e73c8c9d5`

- identical encoder init across runs: False
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=255 | 2 | 78.97 ± 0.30 | 42.31 ± 1.00 | 36.66 ± 0.71 | 74.14 ± 0.20 | 60.74 ± 1.71 | 0.97 ± 0.00 |
| vcs_qmi K=8 | 2 | 78.15 ± 0.13 | 42.31 ± 1.00 | 35.84 ± 0.88 | 73.26 ± 0.37 | 57.75 ± 0.99 | 0.97 ± 0.00 |

## Coverage checks

- P29_vcs_cosK255_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P29_vcs_cosK255_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P29_vcs_cosK8_seed1: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P29_vcs_cosK8_seed2: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
