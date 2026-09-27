# P43_vcs_ceiling — neutral results table (observed values only)

Generated 2026-09-27T01:29:35Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P43_vcs_a5_views16_200ep_seed0 | vcs_qmi | 8 | 0 | 200/200 | 86.74 | 84.48 | 0.9745±0.0010 | 125.82 | 29465 | 20143/26200 | COMPLETED |
| P43_vcs_a5_views4_1600ep_seed0 | vcs_qmi | 8 | 0 | 1231/1600 | null | null | null | null | 5516 | 7568/12672 | RUNNING |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | vcs_qmi | 8 | 0 | 800/800 | 87.78 | 85.58 | 0.8772±0.0007 | 107.27 | 36544 | 5268/8028 | COMPLETED |
| P43_vcs_a5_views4_b128_800ep_seed0 | vcs_qmi | 8 | 0 | 753/800 | null | null | null | null | 30768 | 3682/4842 | RUNNING |
| P43_vcs_a5_views8_400ep_seed0 | vcs_qmi | 8 | 0 | 400/400 | 86.76 | 84.94 | 0.9752±0.0015 | 136.99 | 28413 | 10226/15558 | COMPLETED |
| P43_vcs_a5_views8_800ep_seed0 | vcs_qmi | 8 | 0 | 614/800 | null | null | null | null | 33761 | 10226/15558 | RUNNING |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P43_vcs_a5_views16_200ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P43_vcs_a5_views4_1600ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 1600 |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P43_vcs_a5_views4_b128_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 128 | 0.001 | 800 |
| P43_vcs_a5_views8_400ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 400 |
| P43_vcs_a5_views8_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P43_vcs_a5_views16_200ep_seed0 | 41.94 | 86.74 | 44.80 | 36.54 | 84.48 | 47.94 | 2.94 | 125.82 | -0.9998 |
| P43_vcs_a5_views4_1600ep_seed0 | null | null | null | null | null | null | null | null | null |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 41.94 | 87.78 | 45.84 | 36.54 | 85.58 | 49.04 | 2.94 | 107.27 | -0.9998 |
| P43_vcs_a5_views4_b128_800ep_seed0 | null | null | null | null | null | null | null | null | null |
| P43_vcs_a5_views8_400ep_seed0 | 41.94 | 86.76 | 44.82 | 36.54 | 84.94 | 48.40 | 2.94 | 136.99 | -0.9998 |
| P43_vcs_a5_views8_800ep_seed0 | null | null | null | null | null | null | null | null | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P43_vcs_a5_views16_200ep_seed0: ep0: 36.58, ep10: 64.96, ep20: 73.70, ep50: 79.76, ep100: 83.38, ep150: 84.68, ep200: 84.48
- P43_vcs_a5_views4_1600ep_seed0: ep0: 36.58, ep50: 73.78, ep100: 78.18, ep200: 81.56, ep400: 84.70, ep800: 85.48, ep1200: 85.50
- P43_vcs_a5_views4_800ep_augstrong_seed0: ep0: 36.58, ep20: 68.02, ep50: 75.68, ep100: 79.12, ep200: 82.32, ep400: 84.50, ep600: 85.26, ep800: 85.58
- P43_vcs_a5_views4_b128_800ep_seed0: ep0: 36.58, ep20: 70.38, ep50: 75.82, ep100: 79.34, ep200: 81.84, ep400: 83.58, ep600: 83.94
- P43_vcs_a5_views8_400ep_seed0: ep0: 36.58, ep20: 71.38, ep50: 77.32, ep100: 80.94, ep200: 84.22, ep300: 85.08, ep400: 84.94
- P43_vcs_a5_views8_800ep_seed0: ep0: 36.58, ep20: 71.46, ep50: 76.96, ep100: 81.48, ep200: 84.14, ep400: 85.74, ep600: 86.18

## Held-out J trajectory (VCS only)

- P43_vcs_a5_views16_200ep_seed0: ep0: -0.9998, ep10: 0.7949, ep20: 0.9159, ep50: 0.9540, ep100: 0.9675, ep150: 0.9726, ep200: 0.9745
- P43_vcs_a5_views4_1600ep_seed0: ep0: -0.9998, ep50: 0.9156, ep100: 0.9429, ep200: 0.9575, ep400: 0.9665, ep800: 0.9716, ep1200: 0.9742
- P43_vcs_a5_views4_800ep_augstrong_seed0: ep0: -0.9998, ep20: 0.6843, ep50: 0.7445, ep100: 0.7945, ep200: 0.8321, ep400: 0.8512, ep600: 0.8734, ep800: 0.8772
- P43_vcs_a5_views4_b128_800ep_seed0: ep0: -0.9998, ep20: 0.8838, ep50: 0.9204, ep100: 0.9460, ep200: 0.9592, ep400: 0.9671, ep600: 0.9719
- P43_vcs_a5_views8_400ep_seed0: ep0: -0.9998, ep20: 0.9007, ep50: 0.9406, ep100: 0.9584, ep200: 0.9691, ep300: 0.9738, ep400: 0.9752
- P43_vcs_a5_views8_800ep_seed0: ep0: -0.9998, ep20: 0.9030, ep50: 0.9382, ep100: 0.9578, ep200: 0.9671, ep400: 0.9720, ep600: 0.9751

## Final-epoch training objective values (epoch means)

- P43_vcs_a5_views16_200ep_seed0: J_raw 0.9865, R_binary 0.0135, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P43_vcs_a5_views4_1600ep_seed0: J_raw 0.9893, R_binary 0.0107, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P43_vcs_a5_views4_800ep_augstrong_seed0: J_raw 0.8976, R_binary 0.1024, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P43_vcs_a5_views4_b128_800ep_seed0: J_raw 0.9852, R_binary 0.0148, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P43_vcs_a5_views8_400ep_seed0: J_raw 0.9872, R_binary 0.0128, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P43_vcs_a5_views8_800ep_seed0: J_raw 0.9900, R_binary 0.0100, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P43_vcs_a5_views16_200ep_seed0 | 0.5234 | 489 | 978 | 29465 | 91 | 11 | 8960000 | 2 | 1010201@node58 |
| P43_vcs_a5_views4_1600ep_seed0 | 0.3427 | 747 | 1494 | 5516 | 39 | null | 4115968 | 2 | 1009757@nodesumo01 |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 0.1139 | 2248 | 4497 | 36544 | 144 | 16 | 35840000 | 2 | 1010205@node59 |
| P43_vcs_a5_views4_b128_800ep_seed0 | 0.1832 | 699 | 1397 | 30768 | 129 | null | 21383296 | 2 | 1010207@node60 |
| P43_vcs_a5_views8_400ep_seed0 | 0.2456 | 1042 | 2085 | 28413 | 91 | 11 | 17920000 | 2 | 1010203@node59 |
| P43_vcs_a5_views8_800ep_seed0 | 0.5338 | 480 | 959 | 33761 | 85 | null | 20116224 | 2 | 1010214@nodesumo01 |

## Provenance

- P43_vcs_a5_views16_200ep_seed0: commit `bc3961c42ad40b47f9208187f63c3f74d58ef059` dirty=True, config `80da25b5d7014f0fcb33d6bf8395feca2b427dfa7202ec93eac48af4a26c8ae5`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P43_vcs_a5_views4_1600ep_seed0: commit `f076f45eba700854430dea1d01b11f2ab8f26b3f` dirty=True, config `af4ca38b72359beee6175d36b7c75dc8ff343e89d24d36e0e39b23c6de2b000d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P43_vcs_a5_views4_800ep_augstrong_seed0: commit `d8df85ddf706c43affd81aa6ce34aeaa09886dd5` dirty=False, config `599f3e1ccc2cd500c7f9cd3a024c9775d0465450a1633a135741b42a43dc9672`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P43_vcs_a5_views4_b128_800ep_seed0: commit `d8df85ddf706c43affd81aa6ce34aeaa09886dd5` dirty=False, config `fb8957fd246caf542b6abef493a1ae8ea496c1c06a70ee119eab31392aa635aa`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P43_vcs_a5_views8_400ep_seed0: commit `d8df85ddf706c43affd81aa6ce34aeaa09886dd5` dirty=False, config `2a6037add8612a6c95fa162052b73ba47744475246b846c449e5383268e6afb3`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P43_vcs_a5_views8_800ep_seed0: commit `236931d1bc7c227177b566db290823f01dccaf96` dirty=False, config `d0f9b5fd324ada5ddd69f8f4539830c778c0a3bac7dd3dec5659c387c95819e2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=8 | 3 | 87.09 ± 0.59 | 41.94 ± 0.00 | 45.15 ± 0.59 | 85.00 ± 0.55 | 123.36 ± 15.01 | 0.94 ± 0.06 |

## Coverage checks

- P43_vcs_a5_views16_200ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P43_vcs_a5_views4_1600ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P43_vcs_a5_views4_800ep_augstrong_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P43_vcs_a5_views4_b128_800ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
- P43_vcs_a5_views8_400ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_400.json), status COMPLETED, failure None
- P43_vcs_a5_views8_800ep_seed0: epoch0 eval False, final eval False (None), status RUNNING, failure None
