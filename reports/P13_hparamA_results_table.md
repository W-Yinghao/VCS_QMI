# P12_vcs_hparamA — neutral results table (observed values only)

Generated 2026-09-25T04:36:55Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P12_vcs_cd1_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.32 | 62.52 | 0.8911±0.0052 | 11.76 | 6361 | 3684/4844 | COMPLETED |
| P12_vcs_cd3_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.00 | 63.70 | 0.9033±0.0045 | 14.27 | 6275 | 3690/4854 | COMPLETED |
| P12_vcs_clr0.1_seed0 | vcs_qmi | 1 | 0 | 200/200 | 72.86 | 62.66 | 0.8882±0.0061 | 11.44 | 6278 | 3687/4848 | COMPLETED |
| P12_vcs_clr0.3_seed0 | vcs_qmi | 1 | 0 | 200/200 | 72.80 | 61.68 | 0.8890±0.0072 | 11.37 | 6242 | 3687/4848 | COMPLETED |
| P12_vcs_clr10_seed0 | vcs_qmi | 1 | 0 | 200/200 | 75.08 | 66.56 | 0.9176±0.0045 | 17.70 | 6401 | 3687/4848 | COMPLETED |
| P12_vcs_clr3_seed0 | vcs_qmi | 1 | 0 | 200/200 | 75.06 | 64.08 | 0.9065±0.0034 | 14.88 | 6281 | 3687/4848 | COMPLETED |
| P12_vcs_cw1024_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.14 | 63.80 | 0.9068±0.0048 | 14.63 | 6360 | 3699/4846 | COMPLETED |
| P12_vcs_cw128_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.16 | 62.86 | 0.8923±0.0053 | 11.88 | 6254 | 3683/4842 | COMPLETED |
| P12_vcs_cw2048_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.80 | 63.84 | 0.9099±0.0042 | 15.15 | 6363 | 3736/4874 | COMPLETED |
| P12_vcs_cw256_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.40 | 64.30 | 0.8962±0.0052 | 12.51 | 6413 | 3684/4844 | COMPLETED |
| P12_vcs_cwd1e-3_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.36 | 63.48 | 0.8975±0.0055 | 13.65 | 6367 | 3687/4848 | COMPLETED |
| P12_vcs_cwd1e-4_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.34 | 63.68 | 0.8987±0.0045 | 13.04 | 6398 | 3687/4848 | COMPLETED |
| P12_vcs_gain0.01_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.44 | 65.04 | 0.9019±0.0037 | 13.02 | 6225 | 3687/4848 | COMPLETED |
| P12_vcs_gain1.0_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.06 | 63.34 | 0.9074±0.0047 | 14.30 | 6216 | 3687/4848 | COMPLETED |
| P12_vcs_k255_seed0 | vcs_qmi | 255 | 0 | 200/200 | 76.22 | 67.62 | 0.9325±0.0021 | 16.68 | 3437 | 3648/4338 | COMPLETED |
| P12_vcs_ph1024_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.46 | 64.04 | 0.8996±0.0042 | 13.10 | 6391 | 3689/4844 | COMPLETED |
| P12_vcs_ph2048_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.04 | 64.06 | 0.8976±0.0041 | 12.98 | 6217 | 3700/4846 | COMPLETED |
| P12_vcs_po256_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.52 | 63.98 | 0.9042±0.0050 | 13.53 | 6355 | 3689/4854 | COMPLETED |
| P12_vcs_po512_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.62 | 64.04 | 0.9046±0.0049 | 13.44 | 6385 | 3694/4852 | COMPLETED |
| P12_vcs_po64_seed0 | vcs_qmi | 1 | 0 | 200/200 | 73.78 | 63.08 | 0.8957±0.0041 | 13.18 | 6255 | 3686/4846 | COMPLETED |
| P12_vcs_rawp_seed0 | vcs_qmi | 1 | 0 | 200/200 | 74.78 | 64.22 | 0.9067±0.0049 | 14.85 | 6303 | 3687/4848 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P12_vcs_cd1_seed0 | 1 | [512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCriticMLP | 256 | 0.001 | 200 |
| P12_vcs_cd3_seed0 | 1 | [512, 512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCriticMLP | 256 | 0.001 | 200 |
| P12_vcs_clr0.1_seed0 | 1 | [512, 512] | 0.1 | 0.1 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_clr0.3_seed0 | 1 | [512, 512] | 0.1 | 0.3 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_clr10_seed0 | 1 | [512, 512] | 0.1 | 10.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_clr3_seed0 | 1 | [512, 512] | 0.1 | 3.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_cw1024_seed0 | 1 | [1024, 1024] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_cw128_seed0 | 1 | [128, 128] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_cw2048_seed0 | 1 | [2048, 2048] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_cw256_seed0 | 1 | [256, 256] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_cwd1e-3_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.001 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_cwd1e-4_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0001 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_gain0.01_seed0 | 1 | [512, 512] | 0.01 | 1.0 | 0.0 | 512 | 128 | l2 | PairCriticMLP | 256 | 0.001 | 200 |
| P12_vcs_gain1.0_seed0 | 1 | [512, 512] | 1.0 | 1.0 | 0.0 | 512 | 128 | l2 | PairCriticMLP | 256 | 0.001 | 200 |
| P12_vcs_k255_seed0 | 255 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_ph1024_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 1024 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_ph2048_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 2048 | 128 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_po256_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 256 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_po512_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 512 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_po64_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 64 | l2 | PairCritic | 256 | 0.001 | 200 |
| P12_vcs_rawp_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | none | PairCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P12_vcs_cd1_seed0 | 41.78 | 73.32 | 31.54 | 36.58 | 62.52 | 25.94 | 2.94 | 11.76 | -0.0000 |
| P12_vcs_cd3_seed0 | 41.78 | 74.00 | 32.22 | 36.58 | 63.70 | 27.12 | 2.94 | 14.27 | 0.0000 |
| P12_vcs_clr0.1_seed0 | 41.78 | 72.86 | 31.08 | 36.58 | 62.66 | 26.08 | 2.94 | 11.44 | -0.0000 |
| P12_vcs_clr0.3_seed0 | 41.78 | 72.80 | 31.02 | 36.58 | 61.68 | 25.10 | 2.94 | 11.37 | -0.0000 |
| P12_vcs_clr10_seed0 | 41.78 | 75.08 | 33.30 | 36.58 | 66.56 | 29.98 | 2.94 | 17.70 | -0.0000 |
| P12_vcs_clr3_seed0 | 41.78 | 75.06 | 33.28 | 36.58 | 64.08 | 27.50 | 2.94 | 14.88 | -0.0000 |
| P12_vcs_cw1024_seed0 | 41.78 | 74.14 | 32.36 | 36.58 | 63.80 | 27.22 | 2.94 | 14.63 | 0.0000 |
| P12_vcs_cw128_seed0 | 41.78 | 73.16 | 31.38 | 36.58 | 62.86 | 26.28 | 2.94 | 11.88 | -0.0000 |
| P12_vcs_cw2048_seed0 | 41.78 | 74.80 | 33.02 | 36.58 | 63.84 | 27.26 | 2.94 | 15.15 | -0.0000 |
| P12_vcs_cw256_seed0 | 41.78 | 74.40 | 32.62 | 36.58 | 64.30 | 27.72 | 2.94 | 12.51 | -0.0000 |
| P12_vcs_cwd1e-3_seed0 | 41.78 | 74.36 | 32.58 | 36.58 | 63.48 | 26.90 | 2.94 | 13.65 | -0.0000 |
| P12_vcs_cwd1e-4_seed0 | 41.78 | 73.34 | 31.56 | 36.58 | 63.68 | 27.10 | 2.94 | 13.04 | -0.0000 |
| P12_vcs_gain0.01_seed0 | 41.78 | 74.44 | 32.66 | 36.58 | 65.04 | 28.46 | 2.94 | 13.02 | 0.0000 |
| P12_vcs_gain1.0_seed0 | 41.78 | 74.06 | 32.28 | 36.58 | 63.34 | 26.76 | 2.94 | 14.30 | -0.0010 |
| P12_vcs_k255_seed0 | 41.98 | 76.22 | 34.24 | 36.58 | 67.62 | 31.04 | 2.94 | 16.68 | -0.0000 |
| P12_vcs_ph1024_seed0 | 41.78 | 74.46 | 32.68 | 36.58 | 64.04 | 27.46 | 2.94 | 13.10 | 0.0000 |
| P12_vcs_ph2048_seed0 | 41.78 | 74.04 | 32.26 | 36.58 | 64.06 | 27.48 | 2.94 | 12.98 | -0.0000 |
| P12_vcs_po256_seed0 | 41.78 | 74.52 | 32.74 | 36.58 | 63.98 | 27.40 | 2.94 | 13.53 | -0.0000 |
| P12_vcs_po512_seed0 | 41.78 | 73.62 | 31.84 | 36.58 | 64.04 | 27.46 | 2.94 | 13.44 | -0.0000 |
| P12_vcs_po64_seed0 | 41.78 | 73.78 | 32.00 | 36.58 | 63.08 | 26.50 | 2.94 | 13.18 | 0.0000 |
| P12_vcs_rawp_seed0 | 41.78 | 74.78 | 33.00 | 36.58 | 64.22 | 27.64 | 2.94 | 14.85 | -0.0001 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P12_vcs_cd1_seed0: ep0: 36.58, ep10: 37.60, ep20: 41.64, ep50: 52.56, ep100: 59.38, ep150: 62.34, ep200: 62.52
- P12_vcs_cd3_seed0: ep0: 36.58, ep10: 43.26, ep20: 49.62, ep50: 56.10, ep100: 61.64, ep150: 63.68, ep200: 63.70
- P12_vcs_clr0.1_seed0: ep0: 36.58, ep10: 37.16, ep20: 40.36, ep50: 52.92, ep100: 59.64, ep150: 61.52, ep200: 62.66
- P12_vcs_clr0.3_seed0: ep0: 36.58, ep10: 37.52, ep20: 42.48, ep50: 51.80, ep100: 58.36, ep150: 60.80, ep200: 61.68
- P12_vcs_clr10_seed0: ep0: 36.58, ep10: 48.36, ep20: 52.92, ep50: 60.26, ep100: 64.06, ep150: 65.98, ep200: 66.56
- P12_vcs_clr3_seed0: ep0: 36.58, ep10: 44.98, ep20: 51.10, ep50: 56.44, ep100: 62.90, ep150: 64.04, ep200: 64.08
- P12_vcs_cw1024_seed0: ep0: 36.58, ep10: 44.14, ep20: 49.64, ep50: 57.52, ep100: 61.82, ep150: 63.64, ep200: 63.80
- P12_vcs_cw128_seed0: ep0: 36.58, ep10: 38.72, ep20: 44.88, ep50: 55.02, ep100: 60.00, ep150: 63.00, ep200: 62.86
- P12_vcs_cw2048_seed0: ep0: 36.58, ep10: 44.68, ep20: 50.00, ep50: 56.06, ep100: 62.44, ep150: 63.94, ep200: 63.84
- P12_vcs_cw256_seed0: ep0: 36.58, ep10: 41.48, ep20: 46.66, ep50: 53.96, ep100: 60.14, ep150: 63.70, ep200: 64.30
- P12_vcs_cwd1e-3_seed0: ep0: 36.58, ep10: 41.76, ep20: 47.72, ep50: 56.32, ep100: 60.60, ep150: 63.68, ep200: 63.48
- P12_vcs_cwd1e-4_seed0: ep0: 36.58, ep10: 42.50, ep20: 47.70, ep50: 55.58, ep100: 60.84, ep150: 63.68, ep200: 63.68
- P12_vcs_gain0.01_seed0: ep0: 36.58, ep10: 41.62, ep20: 47.06, ep50: 56.72, ep100: 62.16, ep150: 64.04, ep200: 65.04
- P12_vcs_gain1.0_seed0: ep0: 36.58, ep10: 44.94, ep20: 50.74, ep50: 56.64, ep100: 61.42, ep150: 62.92, ep200: 63.34
- P12_vcs_k255_seed0: ep0: 36.58, ep10: 47.38, ep20: 53.96, ep50: 61.44, ep100: 64.78, ep150: 67.34, ep200: 67.62
- P12_vcs_ph1024_seed0: ep0: 36.58, ep10: 41.98, ep20: 47.62, ep50: 56.78, ep100: 60.72, ep150: 63.16, ep200: 64.04
- P12_vcs_ph2048_seed0: ep0: 36.58, ep10: 42.14, ep20: 47.22, ep50: 54.94, ep100: 59.86, ep150: 63.66, ep200: 64.06
- P12_vcs_po256_seed0: ep0: 36.58, ep10: 41.82, ep20: 47.62, ep50: 55.74, ep100: 61.22, ep150: 64.22, ep200: 63.98
- P12_vcs_po512_seed0: ep0: 36.58, ep10: 42.76, ep20: 45.38, ep50: 55.38, ep100: 61.72, ep150: 63.50, ep200: 64.04
- P12_vcs_po64_seed0: ep0: 36.58, ep10: 42.20, ep20: 47.62, ep50: 56.46, ep100: 60.46, ep150: 63.04, ep200: 63.08
- P12_vcs_rawp_seed0: ep0: 36.58, ep10: 42.42, ep20: 49.64, ep50: 57.06, ep100: 61.18, ep150: 63.64, ep200: 64.22

## Held-out J trajectory (VCS only)

- P12_vcs_cd1_seed0: ep0: -0.0000, ep10: 0.5622, ep20: 0.6426, ep50: 0.7836, ep100: 0.8445, ep150: 0.8814, ep200: 0.8911
- P12_vcs_cd3_seed0: ep0: 0.0000, ep10: 0.5800, ep20: 0.7243, ep50: 0.8036, ep100: 0.8645, ep150: 0.8933, ep200: 0.9033
- P12_vcs_clr0.1_seed0: ep0: -0.0000, ep10: 0.4784, ep20: 0.6231, ep50: 0.7744, ep100: 0.8419, ep150: 0.8773, ep200: 0.8882
- P12_vcs_clr0.3_seed0: ep0: -0.0000, ep10: 0.5641, ep20: 0.6703, ep50: 0.7723, ep100: 0.8455, ep150: 0.8794, ep200: 0.8890
- P12_vcs_clr10_seed0: ep0: -0.0000, ep10: 0.6692, ep20: 0.7377, ep50: 0.8347, ep100: 0.8794, ep150: 0.9090, ep200: 0.9176
- P12_vcs_clr3_seed0: ep0: -0.0000, ep10: 0.6450, ep20: 0.7410, ep50: 0.8099, ep100: 0.8658, ep150: 0.8976, ep200: 0.9065
- P12_vcs_cw1024_seed0: ep0: 0.0000, ep10: 0.6367, ep20: 0.7387, ep50: 0.8200, ep100: 0.8701, ep150: 0.8980, ep200: 0.9068
- P12_vcs_cw128_seed0: ep0: -0.0000, ep10: 0.5511, ep20: 0.6842, ep50: 0.7833, ep100: 0.8484, ep150: 0.8847, ep200: 0.8923
- P12_vcs_cw2048_seed0: ep0: -0.0000, ep10: 0.6549, ep20: 0.7271, ep50: 0.8113, ep100: 0.8713, ep150: 0.9000, ep200: 0.9099
- P12_vcs_cw256_seed0: ep0: -0.0000, ep10: 0.5944, ep20: 0.6883, ep50: 0.7906, ep100: 0.8624, ep150: 0.8862, ep200: 0.8962
- P12_vcs_cwd1e-3_seed0: ep0: -0.0000, ep10: 0.6207, ep20: 0.7129, ep50: 0.7946, ep100: 0.8608, ep150: 0.8879, ep200: 0.8975
- P12_vcs_cwd1e-4_seed0: ep0: -0.0000, ep10: 0.6169, ep20: 0.6879, ep50: 0.8015, ep100: 0.8595, ep150: 0.8896, ep200: 0.8987
- P12_vcs_gain0.01_seed0: ep0: 0.0000, ep10: 0.5929, ep20: 0.7143, ep50: 0.8017, ep100: 0.8638, ep150: 0.8912, ep200: 0.9019
- P12_vcs_gain1.0_seed0: ep0: -0.0010, ep10: 0.6382, ep20: 0.7254, ep50: 0.8098, ep100: 0.8666, ep150: 0.8973, ep200: 0.9074
- P12_vcs_k255_seed0: ep0: -0.0000, ep10: 0.6769, ep20: 0.7755, ep50: 0.8458, ep100: 0.9004, ep150: 0.9253, ep200: 0.9325
- P12_vcs_ph1024_seed0: ep0: 0.0000, ep10: 0.6131, ep20: 0.7114, ep50: 0.8043, ep100: 0.8596, ep150: 0.8882, ep200: 0.8996
- P12_vcs_ph2048_seed0: ep0: -0.0000, ep10: 0.6249, ep20: 0.7016, ep50: 0.7979, ep100: 0.8537, ep150: 0.8876, ep200: 0.8976
- P12_vcs_po256_seed0: ep0: -0.0000, ep10: 0.6194, ep20: 0.6970, ep50: 0.7916, ep100: 0.8634, ep150: 0.8923, ep200: 0.9042
- P12_vcs_po512_seed0: ep0: -0.0000, ep10: 0.6046, ep20: 0.6952, ep50: 0.7967, ep100: 0.8605, ep150: 0.8950, ep200: 0.9046
- P12_vcs_po64_seed0: ep0: 0.0000, ep10: 0.6060, ep20: 0.7151, ep50: 0.8075, ep100: 0.8606, ep150: 0.8874, ep200: 0.8957
- P12_vcs_rawp_seed0: ep0: -0.0001, ep10: 0.6272, ep20: 0.7289, ep50: 0.8213, ep100: 0.8710, ep150: 0.8981, ep200: 0.9067

## Final-epoch training objective values (epoch means)

- P12_vcs_cd1_seed0: J_raw 0.8942, R_binary 0.1058, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cd3_seed0: J_raw 0.9077, R_binary 0.0923, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_clr0.1_seed0: J_raw 0.8911, R_binary 0.1089, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_clr0.3_seed0: J_raw 0.8937, R_binary 0.1063, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_clr10_seed0: J_raw 0.9216, R_binary 0.0784, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_clr3_seed0: J_raw 0.9125, R_binary 0.0875, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cw1024_seed0: J_raw 0.9120, R_binary 0.0880, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cw128_seed0: J_raw 0.8967, R_binary 0.1033, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cw2048_seed0: J_raw 0.9138, R_binary 0.0862, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cw256_seed0: J_raw 0.8999, R_binary 0.1001, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cwd1e-3_seed0: J_raw 0.9015, R_binary 0.0985, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_cwd1e-4_seed0: J_raw 0.9017, R_binary 0.0983, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_gain0.01_seed0: J_raw 0.9090, R_binary 0.0910, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_gain1.0_seed0: J_raw 0.9104, R_binary 0.0896, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_k255_seed0: J_raw 0.9364, R_binary 0.0636, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_ph1024_seed0: J_raw 0.9046, R_binary 0.0954, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_ph2048_seed0: J_raw 0.9019, R_binary 0.0981, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_po256_seed0: J_raw 0.9081, R_binary 0.0919, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_po512_seed0: J_raw 0.9097, R_binary 0.0903, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_po64_seed0: J_raw 0.8999, R_binary 0.1001, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P12_vcs_rawp_seed0: J_raw 0.9118, R_binary 0.0882, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P12_vcs_cd1_seed0 | 0.1796 | 1426 | 2851 | 6361 | 152 | 36 | 8960000 | 132097 | 1007984@node01 |
| P12_vcs_cd3_seed0 | 0.1771 | 1445 | 2891 | 6275 | 146 | 35 | 8960000 | 657409 | 1007983@node54 |
| P12_vcs_clr0.1_seed0 | 0.1772 | 1445 | 2889 | 6278 | 146 | 35 | 8960000 | 394753 | 1007981@node54 |
| P12_vcs_clr0.3_seed0 | 0.1762 | 1453 | 2906 | 6242 | 146 | 35 | 8960000 | 394753 | 1007990@node54 |
| P12_vcs_clr10_seed0 | 0.1807 | 1416 | 2833 | 6401 | 153 | 36 | 8960000 | 394753 | 1007980@node03 |
| P12_vcs_clr3_seed0 | 0.1773 | 1444 | 2888 | 6281 | 148 | 34 | 8960000 | 394753 | 1007989@node54 |
| P12_vcs_cw1024_seed0 | 0.1795 | 1427 | 2853 | 6360 | 155 | 37 | 8960000 | 1313793 | 1007985@node02 |
| P12_vcs_cw128_seed0 | 0.1765 | 1450 | 2900 | 6254 | 146 | 35 | 8960000 | 49537 | 1007986@node56 |
| P12_vcs_cw2048_seed0 | 0.1796 | 1425 | 2851 | 6363 | 155 | 36 | 8960000 | 4724737 | 1007976@node02 |
| P12_vcs_cw256_seed0 | 0.1811 | 1413 | 2827 | 6413 | 152 | 36 | 8960000 | 131841 | 1007977@node01 |
| P12_vcs_cwd1e-3_seed0 | 0.1796 | 1425 | 2851 | 6367 | 156 | 36 | 8960000 | 394753 | 1008008@node02 |
| P12_vcs_cwd1e-4_seed0 | 0.1807 | 1417 | 2834 | 6398 | 153 | 36 | 8960000 | 394753 | 1008006@node01 |
| P12_vcs_gain0.01_seed0 | 0.1757 | 1457 | 2915 | 6225 | 148 | 35 | 8960000 | 394753 | 1008010@node56 |
| P12_vcs_gain1.0_seed0 | 0.1754 | 1460 | 2919 | 6216 | 148 | 35 | 8960000 | 394753 | 1008009@node04 |
| P12_vcs_k255_seed0 | 0.0969 | 2643 | 5286 | 3437 | 80 | 20 | 8960000 | 394753 | 1008173@node53 |
| P12_vcs_ph1024_seed0 | 0.1804 | 1419 | 2837 | 6391 | 154 | 36 | 8960000 | 394753 | 1007988@node03 |
| P12_vcs_ph2048_seed0 | 0.1755 | 1459 | 2917 | 6217 | 149 | 35 | 8960000 | 394753 | 1007987@node04 |
| P12_vcs_po256_seed0 | 0.1794 | 1427 | 2854 | 6355 | 153 | 36 | 8960000 | 525825 | 1007979@node02 |
| P12_vcs_po512_seed0 | 0.1803 | 1420 | 2840 | 6385 | 151 | 36 | 8960000 | 787969 | 1007978@node01 |
| P12_vcs_po64_seed0 | 0.1765 | 1450 | 2901 | 6255 | 147 | 35 | 8960000 | 329217 | 1008005@node56 |
| P12_vcs_rawp_seed0 | 0.1779 | 1439 | 2877 | 6303 | 146 | 35 | 8960000 | 394753 | 1008097@node56 |

## Provenance

- P12_vcs_cd1_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `06dd0215168e03cfd4a13c2acceb9ec5029ae8d47c59d4c91c1c98f40875f23d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cd3_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `daf64a43a111bc9e5941a002976e999437425e2e66d81d31b38172a8d8dfc41b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_clr0.1_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `6203728145ab6a3f1db3bedd3001ecd6bedacc13a2519388e4e30bc230fba58b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_clr0.3_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `42010e4611cc2c2033fa995963a7bde7b734486ea8907b50bfcbe18fa782456e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_clr10_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `2b302df612d79fd4c25e0bfbe5edb5f5adfd4e59c77b081c37f68f10939082a9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_clr3_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `933014edcb8d5694dd2dc39ac4535aeca6365c7c82843ec7a7b44a2c79467a6d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cw1024_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `4c540a55c1197793d821499fad106000522a65a2251ba69d35b4ab06283b6e3b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cw128_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `c66f21a12c4c9e78dda473b8694c9c3a394720f4ccccb5f31d021aea717433ba`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cw2048_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `724e07c1bc125fbf5856e1798f376543924bdcfe6192c96d9c2bc147bef719df`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cw256_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `0d01eb44ac8d606df7adcd080392fdbf316a8d459729a5afd664c1f14747fd36`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cwd1e-3_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `73bfa219b2f68200c953128ea01737f223832a22b4d201f06208073bc0e6967d`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_cwd1e-4_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `da270986e5fb43ea48f65958f5e871489a677b8dc6bd4c694c2cf59bf8c06643`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_gain0.01_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `0862a99818966432b90e623a440cacd590bf7edb7e5ca2c663c534e72c202569`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_gain1.0_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `2c379c4579114347a9ca8dd6daa2d53af65c0687b40c3823cba8db690f3ad9b8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_k255_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `c61299a6bd4c925e8e3e3f4c8e4b782d1e5381b2ab25d414a6722dd4f1f596d1`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_ph1024_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `6c652f2cdf49b6faadb7c3e716ae2edaa8c4b0674dec38ae8be153f77d92e58a`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_ph2048_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `bfbaa2046c0d46b70c0607e82c50de41904cb4e445185c966e2a0d8106c55687`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_po256_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `27486039ce5eeea6bde40eb25223190683621720976a62feaaa3e7d5c1184b37`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_po512_seed0: commit `0c495990218fcaba56e5ba36a3deb74b9b66091c` dirty=True, config `fea927ebfb44d0c485280069da639b964e7b8ce28a930de5dded6be065fc2834`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_po64_seed0: commit `7488dca8bc338379d238780305e13c12207b3852` dirty=True, config `06b5a2209a40a95e5f10df3f1942d637e8f5a781b66a8670b8bb2903f2b28103`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P12_vcs_rawp_seed0: commit `7ea49c458a979c8c97d71f1d6bc0e019ac74f4c6` dirty=True, config `4c7da97cc038b9edc249fb5cf64e526bcb50ea67803a511465e2c502a6479276`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=1 | 20 | 74.05 ± 0.69 | 41.78 ± 0.00 | 32.27 ± 0.69 | 63.75 ± 1.00 | 13.53 ± 1.50 | 0.90 ± 0.01 |
| vcs_qmi K=255 | 1 | 76.22 | 41.98 | 34.24 | 67.62 | 16.68 | 0.93 |

## Coverage checks

- P12_vcs_cd1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cd3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_clr0.1_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_clr0.3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_clr10_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_clr3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cw1024_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cw128_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cw2048_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cw256_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cwd1e-3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_cwd1e-4_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_gain0.01_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_gain1.0_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_k255_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_ph1024_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_ph2048_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_po256_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_po512_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_po64_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P12_vcs_rawp_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
