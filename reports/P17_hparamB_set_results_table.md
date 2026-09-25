# P16_vcs_hparamB_set — neutral results table (observed values only)

Generated 2026-09-25T13:09:58Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P16_vcs_aug_blur05_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.64 | 64.90 | 0.8928±0.0061 | 14.21 | 6423 | 3687/4848 | COMPLETED |
| P16_vcs_aug_cj08_seed0 | vcs_qmi | 1 | 0 | 200/200 | 75.14 | 65.38 | 0.8844±0.0052 | 15.54 | 6396 | 3687/4848 | COMPLETED |
| P16_vcs_aug_crop008_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.04 | 63.12 | 0.8269±0.0020 | 13.80 | 5745 | 3687/4848 | COMPLETED |
| P16_vcs_aug_strong_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.36 | 65.32 | 0.7873±0.0069 | 14.78 | 6437 | 3687/4848 | COMPLETED |
| P16_vcs_aug_weak_seed0 | vcs_qmi | 1 | 0 | 200/200 | 67.78 | 55.20 | 0.9746±0.0020 | 12.42 | 5669 | 3687/4848 | COMPLETED |
| P16_vcs_b1024_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.00 | 63.30 | 0.8935±0.0053 | 12.01 | 5966 | 14894/25948 | COMPLETED |
| P16_vcs_b128_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.22 | 63.68 | 0.9036±0.0053 | 14.03 | 6882 | 2081/3488 | COMPLETED |
| P16_vcs_b512_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.34 | 63.60 | 0.8973±0.0045 | 12.52 | 6689 | 7570/12678 | COMPLETED |
| P16_vcs_k8_aug_strong_seed0 | vcs_qmi | 8 | 0 | 200/200 | 76.70 | 68.96 | 0.8212±0.0036 | 18.36 | 6266 | 3687/4848 | COMPLETED |
| P16_vcs_k8_clr10_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 79.56 | 74.28 | 0.9637±0.0013 | 34.76 | 25457 | 3687/4848 | COMPLETED |
| P16_vcs_k8_clr10_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.44 | 68.40 | 0.9397±0.0016 | 21.44 | 6407 | 3687/4848 | COMPLETED |
| P16_vcs_k8_clr3_seed0 | vcs_qmi | 8 | 0 | 200/200 | 76.80 | 67.32 | 0.9310±0.0017 | 17.42 | 6322 | 3687/4848 | COMPLETED |
| P16_vcs_wd1e-5_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.86 | 63.60 | 0.8956±0.0047 | 13.12 | 6258 | 3687/4848 | COMPLETED |
| P16_vcs_wd5e-4_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.84 | 63.38 | 0.8980±0.0049 | 13.74 | 5668 | 3687/4848 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P16_vcs_aug_blur05_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_aug_cj08_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_aug_crop008_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_aug_strong_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_aug_weak_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_b1024_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 1024 | 0.001 | 200 |
| P16_vcs_b128_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 128 | 0.001 | 200 |
| P16_vcs_b512_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 512 | 0.001 | 200 |
| P16_vcs_k8_aug_strong_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_k8_clr10_800ep_seed0 | 8 | [512, 512] | 0.1 | 10.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 800 |
| P16_vcs_k8_clr10_seed0 | 8 | [512, 512] | 0.1 | 10.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_k8_clr3_seed0 | 8 | [512, 512] | 0.1 | 3.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_wd1e-5_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P16_vcs_wd5e-4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P16_vcs_aug_blur05_seed0 | 41.78 | 73.64 | 31.86 | 36.58 | 64.90 | 28.32 | 2.94 | 14.21 | -0.0000 |
| P16_vcs_aug_cj08_seed0 | 41.78 | 75.14 | 33.36 | 36.58 | 65.38 | 28.80 | 2.94 | 15.54 | -0.0000 |
| P16_vcs_aug_crop008_seed0 | 41.78 | 74.04 | 32.26 | 36.58 | 63.12 | 26.54 | 2.94 | 13.80 | -0.0000 |
| P16_vcs_aug_strong_seed0 | 41.78 | 73.36 | 31.58 | 36.58 | 65.32 | 28.74 | 2.94 | 14.78 | -0.0000 |
| P16_vcs_aug_weak_seed0 | 41.78 | 67.78 | 26.00 | 36.58 | 55.20 | 18.62 | 2.94 | 12.42 | -0.0000 |
| P16_vcs_b1024_seed0 | 41.78 | 73.00 | 31.22 | 36.58 | 63.30 | 26.72 | 2.94 | 12.01 | -0.0000 |
| P16_vcs_b128_seed0 | 41.78 | 73.22 | 31.44 | 36.58 | 63.68 | 27.10 | 2.94 | 14.03 | -0.0000 |
| P16_vcs_b512_seed0 | 41.78 | 74.34 | 32.56 | 36.58 | 63.60 | 27.02 | 2.94 | 12.52 | -0.0000 |
| P16_vcs_k8_aug_strong_seed0 | 41.78 | 76.70 | 34.92 | 36.58 | 68.96 | 32.38 | 2.94 | 18.36 | -0.0000 |
| P16_vcs_k8_clr10_800ep_seed0 | 41.78 | 79.56 | 37.78 | 36.58 | 74.28 | 37.70 | 2.94 | 34.76 | -0.0000 |
| P16_vcs_k8_clr10_seed0 | 41.78 | 77.44 | 35.66 | 36.58 | 68.40 | 31.82 | 2.94 | 21.44 | -0.0000 |
| P16_vcs_k8_clr3_seed0 | 41.78 | 76.80 | 35.02 | 36.58 | 67.32 | 30.74 | 2.94 | 17.42 | -0.0000 |
| P16_vcs_wd1e-5_seed0 | 41.78 | 74.86 | 33.08 | 36.58 | 63.60 | 27.02 | 2.94 | 13.12 | -0.0000 |
| P16_vcs_wd5e-4_seed0 | 41.78 | 73.84 | 32.06 | 36.58 | 63.38 | 26.80 | 2.94 | 13.74 | -0.0000 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P16_vcs_aug_blur05_seed0: ep0: 36.58, ep10: 41.44, ep20: 47.78, ep50: 56.40, ep100: 61.86, ep150: 64.12, ep200: 64.90
- P16_vcs_aug_cj08_seed0: ep0: 36.58, ep10: 44.28, ep20: 49.60, ep50: 57.98, ep100: 62.86, ep150: 64.76, ep200: 65.38
- P16_vcs_aug_crop008_seed0: ep0: 36.58, ep10: 41.68, ep20: 46.38, ep50: 53.76, ep100: 60.76, ep150: 62.36, ep200: 63.12
- P16_vcs_aug_strong_seed0: ep0: 36.58, ep10: 40.20, ep20: 47.62, ep50: 56.80, ep100: 62.06, ep150: 65.10, ep200: 65.32
- P16_vcs_aug_weak_seed0: ep0: 36.58, ep10: 39.86, ep20: 43.44, ep50: 48.76, ep100: 51.74, ep150: 54.42, ep200: 55.20
- P16_vcs_b1024_seed0: ep0: 36.58, ep10: 39.24, ep20: 44.10, ep50: 52.30, ep100: 59.78, ep150: 62.58, ep200: 63.30
- P16_vcs_b128_seed0: ep0: 36.58, ep10: 42.30, ep20: 48.48, ep50: 56.78, ep100: 60.96, ep150: 63.30, ep200: 63.68
- P16_vcs_b512_seed0: ep0: 36.58, ep10: 41.22, ep20: 47.54, ep50: 55.58, ep100: 61.08, ep150: 63.42, ep200: 63.60
- P16_vcs_k8_aug_strong_seed0: ep0: 36.58, ep10: 44.98, ep20: 53.22, ep50: 60.84, ep100: 65.86, ep150: 68.48, ep200: 68.96
- P16_vcs_k8_clr10_800ep_seed0: ep0: 36.58, ep20: 57.40, ep50: 64.10, ep100: 67.04, ep200: 70.16, ep400: 72.14, ep600: 74.00, ep800: 74.28
- P16_vcs_k8_clr10_seed0: ep0: 36.58, ep10: 51.78, ep20: 56.86, ep50: 63.96, ep100: 66.70, ep150: 68.38, ep200: 68.40
- P16_vcs_k8_clr3_seed0: ep0: 36.58, ep10: 47.66, ep20: 53.58, ep50: 63.48, ep100: 66.04, ep150: 67.06, ep200: 67.32
- P16_vcs_wd1e-5_seed0: ep0: 36.58, ep10: 42.12, ep20: 48.04, ep50: 56.64, ep100: 61.48, ep150: 63.54, ep200: 63.60
- P16_vcs_wd5e-4_seed0: ep0: 36.58, ep10: 41.38, ep20: 47.78, ep50: 56.42, ep100: 60.80, ep150: 63.42, ep200: 63.38

## Held-out J trajectory (VCS only)

- P16_vcs_aug_blur05_seed0: ep0: -0.0000, ep10: 0.5521, ep20: 0.6783, ep50: 0.7927, ep100: 0.8517, ep150: 0.8829, ep200: 0.8928
- P16_vcs_aug_cj08_seed0: ep0: -0.0000, ep10: 0.5599, ep20: 0.6827, ep50: 0.7605, ep100: 0.8412, ep150: 0.8730, ep200: 0.8844
- P16_vcs_aug_crop008_seed0: ep0: -0.0000, ep10: 0.5110, ep20: 0.5905, ep50: 0.7046, ep100: 0.7773, ep150: 0.8153, ep200: 0.8269
- P16_vcs_aug_strong_seed0: ep0: -0.0000, ep10: 0.3541, ep20: 0.4727, ep50: 0.6397, ep100: 0.7213, ep150: 0.7758, ep200: 0.7873
- P16_vcs_aug_weak_seed0: ep0: -0.0000, ep10: 0.8032, ep20: 0.8824, ep50: 0.9312, ep100: 0.9531, ep150: 0.9692, ep200: 0.9746
- P16_vcs_b1024_seed0: ep0: -0.0000, ep10: 0.5251, ep20: 0.6718, ep50: 0.7767, ep100: 0.8517, ep150: 0.8825, ep200: 0.8935
- P16_vcs_b128_seed0: ep0: -0.0000, ep10: 0.5716, ep20: 0.7105, ep50: 0.8142, ep100: 0.8641, ep150: 0.8947, ep200: 0.9036
- P16_vcs_b512_seed0: ep0: -0.0000, ep10: 0.6080, ep20: 0.7051, ep50: 0.8105, ep100: 0.8600, ep150: 0.8865, ep200: 0.8973
- P16_vcs_k8_aug_strong_seed0: ep0: -0.0000, ep10: 0.4681, ep20: 0.5938, ep50: 0.6974, ep100: 0.7719, ep150: 0.8071, ep200: 0.8212
- P16_vcs_k8_clr10_800ep_seed0: ep0: -0.0000, ep20: 0.7921, ep50: 0.8742, ep100: 0.9030, ep200: 0.9291, ep400: 0.9497, ep600: 0.9598, ep800: 0.9637
- P16_vcs_k8_clr10_seed0: ep0: -0.0000, ep10: 0.7466, ep20: 0.8089, ep50: 0.8721, ep100: 0.9119, ep150: 0.9320, ep200: 0.9397
- P16_vcs_k8_clr3_seed0: ep0: -0.0000, ep10: 0.6954, ep20: 0.7787, ep50: 0.8564, ep100: 0.9033, ep150: 0.9226, ep200: 0.9310
- P16_vcs_wd1e-5_seed0: ep0: -0.0000, ep10: 0.6063, ep20: 0.7118, ep50: 0.7989, ep100: 0.8587, ep150: 0.8871, ep200: 0.8956
- P16_vcs_wd5e-4_seed0: ep0: -0.0000, ep10: 0.6075, ep20: 0.7077, ep50: 0.8046, ep100: 0.8626, ep150: 0.8884, ep200: 0.8980

## Final-epoch training objective values (epoch means)

- P16_vcs_aug_blur05_seed0: J_raw 0.9013, R_binary 0.0987, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_aug_cj08_seed0: J_raw 0.8885, R_binary 0.1115, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_aug_crop008_seed0: J_raw 0.8304, R_binary 0.1696, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_aug_strong_seed0: J_raw 0.7847, R_binary 0.2153, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_aug_weak_seed0: J_raw 0.9768, R_binary 0.0232, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_b1024_seed0: J_raw 0.9022, R_binary 0.0978, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_b128_seed0: J_raw 0.9100, R_binary 0.0900, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_b512_seed0: J_raw 0.9064, R_binary 0.0936, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_k8_aug_strong_seed0: J_raw 0.8209, R_binary 0.1791, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_k8_clr10_800ep_seed0: J_raw 0.9690, R_binary 0.0310, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_k8_clr10_seed0: J_raw 0.9422, R_binary 0.0578, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_k8_clr3_seed0: J_raw 0.9350, R_binary 0.0650, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_wd1e-5_seed0: J_raw 0.8999, R_binary 0.1001, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P16_vcs_wd5e-4_seed0: J_raw 0.9015, R_binary 0.0985, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P16_vcs_aug_blur05_seed0 | 0.1806 | 1417 | 2835 | 6423 | 173 | 39 | 8960000 | 394753 | 1008228@node02 |
| P16_vcs_aug_cj08_seed0 | 0.1806 | 1418 | 2835 | 6396 | 152 | 36 | 8960000 | 394753 | 1008227@node01 |
| P16_vcs_aug_crop008_seed0 | 0.1618 | 1582 | 3164 | 5745 | 140 | 30 | 8960000 | 394753 | 1008226@nodeaudible01 |
| P16_vcs_aug_strong_seed0 | 0.1811 | 1413 | 2827 | 6437 | 167 | 38 | 8960000 | 394753 | 1008229@node03 |
| P16_vcs_aug_weak_seed0 | 0.1599 | 1601 | 3201 | 5669 | 138 | 30 | 8960000 | 394753 | 1008230@nodeaudible01 |
| P16_vcs_b1024_seed0 | 0.6629 | 1545 | 3090 | 5966 | 140 | 30 | 8806400 | 394753 | 1008233@nodeaudible01 |
| P16_vcs_b128_seed0 | 0.0974 | 1315 | 2630 | 6882 | 147 | 35 | 8985600 | 394753 | 1008231@node56 |
| P16_vcs_b512_seed0 | 0.3766 | 1360 | 2719 | 6689 | 151 | 35 | 8908800 | 394753 | 1008232@node54 |
| P16_vcs_k8_aug_strong_seed0 | 0.1762 | 1453 | 2906 | 6266 | 163 | 37 | 8960000 | 394753 | 1008241@node56 |
| P16_vcs_k8_clr10_800ep_seed0 | 0.1797 | 1424 | 2849 | 25457 | 177 | 36 | 35840000 | 394753 | 1008239@node02 |
| P16_vcs_k8_clr10_seed0 | 0.1809 | 1415 | 2830 | 6407 | 150 | 36 | 8960000 | 394753 | 1008237@node01 |
| P16_vcs_k8_clr3_seed0 | 0.1785 | 1434 | 2869 | 6322 | 145 | 35 | 8960000 | 394753 | 1008236@node56 |
| P16_vcs_wd1e-5_seed0 | 0.1766 | 1450 | 2899 | 6258 | 148 | 35 | 8960000 | 394753 | 1008234@node56 |
| P16_vcs_wd5e-4_seed0 | 0.1598 | 1602 | 3204 | 5668 | 134 | 29 | 8960000 | 394753 | 1008235@nodeaudible01 |

## Provenance

- P16_vcs_aug_blur05_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `c3d17539b8b86acff883978529c55f8ac69d1503ed5ac1df532d3421f0e2c671`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_aug_cj08_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `d46e66604dbd8e83862cc651fe065869600347b4a2c341961319da3ee2411649`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_aug_crop008_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `19849e7036ebe36f3c6245e98ae9566126f0bdb11af078f8a5ae68f2259c24ef`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_aug_strong_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `3557f11caa3677440e0396998fdaec05a3302a67e1c0f569f32d043347b42d65`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_aug_weak_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `2994a28e7860b284c94bbb46d9f5684a4fc159456af7066332137e02e87550e2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_b1024_seed0: commit `b02a784027fdcef4a04e2a3a9bcf5d7ae664d1f3` dirty=True, config `a5f1de626933b4f42837a89091824144fe2bd0479ee2f3d6ad15676b26169e16`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_b128_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `6ef52ee9e37d57b4f1cece49364fa302c22b627b8238715a05881d972d14f92c`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_b512_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `3f632a7dcf22a13dfc7d0e69eb1b624c4e052bd55033dfebe7f116b17423f847`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_k8_aug_strong_seed0: commit `68a5041b60e861a7091d1e92fba83bace937d219` dirty=True, config `ac02516f3215c7f28bf1390aef712b73605fb5d9571188710f2594fc1e0194d2`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_k8_clr10_800ep_seed0: commit `68a5041b60e861a7091d1e92fba83bace937d219` dirty=True, config `874a2edbf98e198e550284878b295ce3eca884979716e7c7e0df54ab19b4be88`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_k8_clr10_seed0: commit `68a5041b60e861a7091d1e92fba83bace937d219` dirty=True, config `f3e6a5e1da28f91b79d34b9880fbb00117686206f266e759843c59b53b199833`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_k8_clr3_seed0: commit `68a5041b60e861a7091d1e92fba83bace937d219` dirty=True, config `c78eb867e73be3eee2aa29fb08490bbf46812e4f5e8e5a357fc3d611abb13358`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_wd1e-5_seed0: commit `b02a784027fdcef4a04e2a3a9bcf5d7ae664d1f3` dirty=True, config `a847ace8cb6913c4b204ff7edc848b420fc2e2b3467263b205f71a7766750b9b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P16_vcs_wd5e-4_seed0: commit `68a5041b60e861a7091d1e92fba83bace937d219` dirty=True, config `6ab8dae0975f97bc209a659ef28c0a7ff1bd573172c8fdfb21a83ce15304794d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=1 | 10 | 73.32 ± 2.07 | 41.78 ± 0.00 | 31.54 ± 2.07 | 63.15 ± 2.92 | 13.62 ± 1.11 | 0.89 ± 0.05 |
| vcs_qmi K=8 | 4 | 77.62 ± 1.33 | 41.78 ± 0.00 | 35.85 ± 1.33 | 69.74 ± 3.10 | 23.00 ± 8.03 | 0.91 ± 0.06 |

## Coverage checks

- P16_vcs_aug_blur05_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_aug_cj08_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_aug_crop008_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_aug_strong_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_aug_weak_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_b1024_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_b128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_b512_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_k8_aug_strong_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_k8_clr10_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P16_vcs_k8_clr10_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_k8_clr3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_wd1e-5_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P16_vcs_wd5e-4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
