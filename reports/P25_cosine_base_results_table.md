# P24_vcs_cosine_base — neutral results table (observed values only)

Generated 2026-09-25T20:49:39Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P24_vcs_cos_800ep_seed0 | vcs_qmi | 8 | 0 | 800/800 | 80.74 | 76.48 | 0.9823±0.0004 | 89.13 | 24739 | 3683/4842 | COMPLETED |
| P24_vcs_cos_clr10_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.26 | 71.74 | 0.9651±0.0016 | 53.35 | 6343 | 3683/4842 | COMPLETED |
| P24_vcs_cos_ema0.996_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.98 | 72.62 | 0.9657±0.0015 | 57.41 | 7832 | 4075/4932 | COMPLETED |
| P24_vcs_cos_ema0.99_pred_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.88 | 71.62 | 0.9513±0.0018 | 34.49 | 8578 | 4077/4934 | COMPLETED |
| P24_vcs_cos_ema0.99_seed0 | vcs_qmi | 8 | 0 | 200/200 | 78.80 | 72.70 | 0.9664±0.0013 | 59.16 | 7795 | 4075/4932 | COMPLETED |
| P24_vcs_cos_k255_seed0 | vcs_qmi | 255 | 0 | 200/200 | 79.32 | 73.48 | 0.9692±0.0008 | 61.31 | 5679 | 3683/4844 | COMPLETED |
| P24_vcs_cos_k64_seed0 | vcs_qmi | 64 | 0 | 200/200 | 78.74 | 73.56 | 0.9690±0.0012 | 59.42 | 5610 | 3683/4842 | COMPLETED |
| P24_vcs_cos_negdetach_seed0 | vcs_qmi | 8 | 0 | 200/200 | 80.48 | 74.62 | 0.9287±0.0019 | 30.87 | 6232 | 3683/4842 | COMPLETED |
| P24_vcs_cos_on_h_seed0 | vcs_qmi | 8 | 0 | 200/200 | 72.72 | 76.48 | 0.9679±0.0008 | 235.41 | 6243 | 3649/4838 | COMPLETED |
| P24_vcs_cos_proj_depth3_seed0 | vcs_qmi | 8 | 0 | 200/200 | 78.68 | 72.42 | 0.9612±0.0016 | 44.89 | 6260 | 3686/4848 | COMPLETED |
| P24_vcs_cos_scale10_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.56 | 71.28 | 0.9663±0.0017 | 65.67 | 6241 | 3683/4842 | COMPLETED |
| P24_vcs_cos_sg_pred_seed0 | vcs_qmi | 8 | 0 | 200/200 | 77.30 | 70.76 | 0.9445±0.0024 | 29.06 | 5659 | 3684/4846 | COMPLETED |
| P24_vcs_cos_stopgrad_seed0 | vcs_qmi | 8 | 0 | 200/200 | 78.74 | 73.20 | 0.9661±0.0015 | 58.14 | 6230 | 3683/4842 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P24_vcs_cos_800ep_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 800 |
| P24_vcs_cos_clr10_seed0 | 8 | [512, 512] | 0.1 | 10.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_ema0.996_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_ema0.99_pred_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_ema0.99_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_k255_seed0 | 255 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_k64_seed0 | 64 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_negdetach_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_on_h_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_proj_depth3_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_scale10_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_sg_pred_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |
| P24_vcs_cos_stopgrad_seed0 | 8 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | CosineCritic | 256 | 0.001 | 200 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P24_vcs_cos_800ep_seed0 | 41.78 | 80.74 | 38.96 | 36.58 | 76.48 | 39.90 | 2.94 | 89.13 | -0.5686 |
| P24_vcs_cos_clr10_seed0 | 41.78 | 77.26 | 35.48 | 36.58 | 71.74 | 35.16 | 2.94 | 53.35 | -0.5686 |
| P24_vcs_cos_ema0.996_seed0 | 41.78 | 77.98 | 36.20 | 36.58 | 72.62 | 36.04 | 2.94 | 57.41 | -0.5686 |
| P24_vcs_cos_ema0.99_pred_seed0 | 41.78 | 77.88 | 36.10 | 36.58 | 71.62 | 35.04 | 2.94 | 34.49 | -0.0018 |
| P24_vcs_cos_ema0.99_seed0 | 41.78 | 78.80 | 37.02 | 36.58 | 72.70 | 36.12 | 2.94 | 59.16 | -0.5686 |
| P24_vcs_cos_k255_seed0 | 41.78 | 79.32 | 37.54 | 36.58 | 73.48 | 36.90 | 2.94 | 61.31 | -0.5686 |
| P24_vcs_cos_k64_seed0 | 41.78 | 78.74 | 36.96 | 36.58 | 73.56 | 36.98 | 2.94 | 59.42 | -0.5686 |
| P24_vcs_cos_negdetach_seed0 | 41.78 | 80.48 | 38.70 | 36.58 | 74.62 | 38.04 | 2.94 | 30.87 | -0.5686 |
| P24_vcs_cos_on_h_seed0 | 41.78 | 72.72 | 30.94 | 36.58 | 76.48 | 39.90 | 2.94 | 235.41 | -0.5696 |
| P24_vcs_cos_proj_depth3_seed0 | 41.78 | 78.68 | 36.90 | 36.58 | 72.42 | 35.84 | 2.94 | 44.89 | -0.5621 |
| P24_vcs_cos_scale10_seed0 | 41.78 | 77.56 | 35.78 | 36.58 | 71.28 | 34.70 | 2.94 | 65.67 | -1.0000 |
| P24_vcs_cos_sg_pred_seed0 | 41.78 | 77.30 | 35.52 | 36.58 | 70.76 | 34.18 | 2.94 | 29.06 | -0.0018 |
| P24_vcs_cos_stopgrad_seed0 | 41.78 | 78.74 | 36.96 | 36.58 | 73.20 | 36.62 | 2.94 | 58.14 | -0.5686 |

## kNN trajectory (in-training monitor, selection top-1 %)

- P24_vcs_cos_800ep_seed0: ep0: 36.58, ep20: 64.50, ep50: 69.08, ep100: 71.20, ep200: 72.78, ep400: 74.92, ep600: 75.74, ep800: 76.48
- P24_vcs_cos_clr10_seed0: ep0: 36.58, ep10: 56.80, ep20: 62.28, ep50: 66.74, ep100: 70.32, ep150: 71.30, ep200: 71.74
- P24_vcs_cos_ema0.996_seed0: ep0: 36.58, ep10: 51.82, ep20: 59.60, ep50: 67.74, ep100: 70.96, ep150: 72.60, ep200: 72.62
- P24_vcs_cos_ema0.99_pred_seed0: ep0: 36.58, ep10: 53.80, ep20: 60.50, ep50: 67.86, ep100: 70.68, ep150: 71.40, ep200: 71.62
- P24_vcs_cos_ema0.99_seed0: ep0: 36.58, ep10: 53.04, ep20: 61.96, ep50: 67.98, ep100: 71.22, ep150: 72.50, ep200: 72.70
- P24_vcs_cos_k255_seed0: ep0: 36.58, ep10: 57.64, ep20: 64.24, ep50: 70.04, ep100: 72.16, ep150: 72.96, ep200: 73.48
- P24_vcs_cos_k64_seed0: ep0: 36.58, ep10: 58.10, ep20: 64.58, ep50: 69.44, ep100: 71.92, ep150: 73.18, ep200: 73.56
- P24_vcs_cos_negdetach_seed0: ep0: 36.58, ep10: 53.94, ep20: 60.88, ep50: 68.14, ep100: 71.84, ep150: 74.26, ep200: 74.62
- P24_vcs_cos_on_h_seed0: ep0: 36.58, ep10: 52.50, ep20: 62.06, ep50: 69.70, ep100: 73.32, ep150: 75.80, ep200: 76.48
- P24_vcs_cos_proj_depth3_seed0: ep0: 36.58, ep10: 55.36, ep20: 62.56, ep50: 68.32, ep100: 70.74, ep150: 71.84, ep200: 72.42
- P24_vcs_cos_scale10_seed0: ep0: 36.58, ep10: 46.46, ep20: 57.10, ep50: 66.14, ep100: 70.14, ep150: 71.08, ep200: 71.28
- P24_vcs_cos_sg_pred_seed0: ep0: 36.58, ep10: 55.38, ep20: 62.92, ep50: 65.90, ep100: 69.68, ep150: 70.66, ep200: 70.76
- P24_vcs_cos_stopgrad_seed0: ep0: 36.58, ep10: 56.40, ep20: 63.32, ep50: 68.24, ep100: 71.90, ep150: 73.02, ep200: 73.20

## Held-out J trajectory (VCS only)

- P24_vcs_cos_800ep_seed0: ep0: -0.5686, ep20: 0.8309, ep50: 0.9117, ep100: 0.9415, ep200: 0.9606, ep400: 0.9741, ep600: 0.9806, ep800: 0.9823
- P24_vcs_cos_clr10_seed0: ep0: -0.5686, ep10: 0.7794, ep20: 0.8570, ep50: 0.9156, ep100: 0.9451, ep150: 0.9611, ep200: 0.9651
- P24_vcs_cos_ema0.996_seed0: ep0: -0.5686, ep10: 0.6622, ep20: 0.8244, ep50: 0.9103, ep100: 0.9443, ep150: 0.9609, ep200: 0.9660
- P24_vcs_cos_ema0.99_pred_seed0: ep0: -0.5686, ep10: 0.4293, ep20: 0.3726, ep50: 0.0546, ep100: -0.4263, ep150: -0.5755, ep200: -0.5787
- P24_vcs_cos_ema0.99_seed0: ep0: -0.5686, ep10: 0.6857, ep20: 0.8290, ep50: 0.9136, ep100: 0.9477, ep150: 0.9624, ep200: 0.9666
- P24_vcs_cos_k255_seed0: ep0: -0.5686, ep10: 0.6748, ep20: 0.8348, ep50: 0.9176, ep100: 0.9520, ep150: 0.9651, ep200: 0.9692
- P24_vcs_cos_k64_seed0: ep0: -0.5686, ep10: 0.6624, ep20: 0.8313, ep50: 0.9135, ep100: 0.9506, ep150: 0.9649, ep200: 0.9690
- P24_vcs_cos_negdetach_seed0: ep0: -0.5686, ep10: 0.6178, ep20: 0.7745, ep50: 0.8574, ep100: 0.9013, ep150: 0.9214, ep200: 0.9287
- P24_vcs_cos_on_h_seed0: ep0: -0.5696, ep10: 0.5939, ep20: 0.8056, ep50: 0.9112, ep100: 0.9472, ep150: 0.9634, ep200: 0.9679
- P24_vcs_cos_proj_depth3_seed0: ep0: -0.5621, ep10: 0.6552, ep20: 0.8017, ep50: 0.8996, ep100: 0.9389, ep150: 0.9564, ep200: 0.9612
- P24_vcs_cos_scale10_seed0: ep0: -1.0000, ep10: 0.5551, ep20: 0.7837, ep50: 0.8988, ep100: 0.9452, ep150: 0.9619, ep200: 0.9663
- P24_vcs_cos_sg_pred_seed0: ep0: -0.5686, ep10: 0.3999, ep20: 0.3538, ep50: 0.0677, ep100: -0.3601, ep150: -0.5117, ep200: -0.5130
- P24_vcs_cos_stopgrad_seed0: ep0: -0.5686, ep10: 0.6794, ep20: 0.8216, ep50: 0.9140, ep100: 0.9462, ep150: 0.9624, ep200: 0.9664

## Final-epoch training objective values (epoch means)

- P24_vcs_cos_800ep_seed0: J_raw 0.9857, R_binary 0.0143, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_clr10_seed0: J_raw 0.9688, R_binary 0.0312, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_ema0.996_seed0: J_raw 0.9695, R_binary 0.0305, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_ema0.99_pred_seed0: J_raw 0.9564, R_binary 0.0436, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_ema0.99_seed0: J_raw 0.9700, R_binary 0.0300, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_k255_seed0: J_raw 0.9723, R_binary 0.0277, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_k64_seed0: J_raw 0.9717, R_binary 0.0283, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_negdetach_seed0: J_raw 0.9329, R_binary 0.0671, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_on_h_seed0: J_raw 0.9691, R_binary 0.0309, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_proj_depth3_seed0: J_raw 0.9654, R_binary 0.0346, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_scale10_seed0: J_raw 0.9703, R_binary 0.0297, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_sg_pred_seed0: J_raw 0.9500, R_binary 0.0500, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P24_vcs_cos_stopgrad_seed0: J_raw 0.9694, R_binary 0.0306, nt_xent null, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P24_vcs_cos_800ep_seed0 | 0.1746 | 1466 | 2933 | 24739 | 172 | 36 | 35840000 | 2 | 1008617@node56 |
| P24_vcs_cos_clr10_seed0 | 0.1788 | 1432 | 2864 | 6343 | 157 | 37 | 8960000 | 2 | 1008605@node01 |
| P24_vcs_cos_ema0.996_seed0 | 0.2217 | 1155 | 2309 | 7832 | 154 | 36 | 8960000 | 2 | 1008612@nodeaudible01 |
| P24_vcs_cos_ema0.99_pred_seed0 | 0.2430 | 1054 | 2107 | 8578 | 148 | 40 | 8960000 | 2 | 1008615@node56 |
| P24_vcs_cos_ema0.99_seed0 | 0.2207 | 1160 | 2320 | 7795 | 134 | 34 | 8960000 | 2 | 1008611@nodeaudible01 |
| P24_vcs_cos_k255_seed0 | 0.1602 | 1598 | 3195 | 5679 | 136 | 29 | 8960000 | 2 | 1008603@nodeaudible01 |
| P24_vcs_cos_k64_seed0 | 0.1582 | 1618 | 3236 | 5610 | 136 | 29 | 8960000 | 2 | 1008602@nodeaudible01 |
| P24_vcs_cos_negdetach_seed0 | 0.1758 | 1456 | 2912 | 6232 | 146 | 35 | 8960000 | 2 | 1008610@node56 |
| P24_vcs_cos_on_h_seed0 | 0.1762 | 1453 | 2906 | 6243 | 155 | 36 | 8960000 | 2 | 1008609@node02 |
| P24_vcs_cos_proj_depth3_seed0 | 0.1766 | 1450 | 2899 | 6260 | 145 | 36 | 8960000 | 2 | 1008616@node54 |
| P24_vcs_cos_scale10_seed0 | 0.1762 | 1453 | 2905 | 6241 | 148 | 35 | 8960000 | 2 | 1008608@node54 |
| P24_vcs_cos_sg_pred_seed0 | 0.1597 | 1603 | 3205 | 5659 | 137 | 29 | 8960000 | 2 | 1008614@nodeaudible01 |
| P24_vcs_cos_stopgrad_seed0 | 0.1759 | 1456 | 2911 | 6230 | 150 | 36 | 8960000 | 2 | 1008613@node55 |

## Provenance

- P24_vcs_cos_800ep_seed0: commit `398c17f6b1057c939aa93bc5ead77b7c762f65d9` dirty=False, config `af928807d419d701a10086c8c31d64b2c352d28dfd9599f220267c5f78c0a889`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_clr10_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `026c629731892cd401f944a8e80cb22cba45c76198a48bd7c86682fbd8cb7cef`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_ema0.996_seed0: commit `315c369d3a7297ac5cef2a83feb5d5b6dd546053` dirty=True, config `a3da61bb0727995cec8709345767ea18ab26a1e2e861e2603a5fd6eb1e6e0524`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_ema0.99_pred_seed0: commit `398c17f6b1057c939aa93bc5ead77b7c762f65d9` dirty=False, config `990b7b8eb29c3b29de8556bfd2797751d1cfa1b15dbb2804b7148997a623e7c8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_ema0.99_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `d6cc1110473fd3ea53ffffc2b25290f81fd9a79385faf6b78d881c60fa85c968`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_k255_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `2341969e73c5c3630b61681ef748680870ad6258871800f64df747d828941fbd`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_k64_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `ccced882b95a4ca062431e307b5ef69b6a6ff727cecdc41df64a1c8bdc274589`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_negdetach_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `859e17e98f3eca13d6d14ad6ce6ce141aebeaba8fdf47f5333274dd2df62ea3b`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_on_h_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `cf1a5be6df87054ad084b837a68b30d106fc2b81ea8a27342adb349338d92c92`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_proj_depth3_seed0: commit `398c17f6b1057c939aa93bc5ead77b7c762f65d9` dirty=False, config `66eb3bb1fbca7f42d475d2c3ed37a1515d33cd6e10094f9d1678e249d08fc1cd`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_scale10_seed0: commit `c397469cd5834abebe4c22c483dd69c0faba820c` dirty=True, config `aa59fb768756ca066cad4b6ac1c80e6237e6b45a51e96334318bd07320c4581e`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_sg_pred_seed0: commit `315c369d3a7297ac5cef2a83feb5d5b6dd546053` dirty=True, config `da77e4e69e65f5810f29dbda60c92e00415730d115d5c70f52a62db600408bc9`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P24_vcs_cos_stopgrad_seed0: commit `315c369d3a7297ac5cef2a83feb5d5b6dd546053` dirty=True, config `9e708fe55da2e8924bb2429eff39db7644e60b12d9a9d6bb484be90a7a7722e8`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| vcs_qmi K=255 | 1 | 79.32 | 41.78 | 37.54 | 73.48 | 61.31 | 0.97 |
| vcs_qmi K=64 | 1 | 78.74 | 41.78 | 36.96 | 73.56 | 59.42 | 0.97 |
| vcs_qmi K=8 | 11 | 78.01 ± 2.11 | 41.78 ± 0.00 | 36.23 ± 2.11 | 73.08 ± 1.97 | 68.87 ± 57.90 | 0.96 ± 0.01 |

## Coverage checks

- P24_vcs_cos_800ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_800.json), status COMPLETED, failure None
- P24_vcs_cos_clr10_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_ema0.996_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_ema0.99_pred_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_ema0.99_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_k255_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_k64_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_negdetach_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_on_h_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_proj_depth3_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_scale10_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_sg_pred_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
- P24_vcs_cos_stopgrad_seed0: epoch0 eval True, final eval True (evaluation_epoch_200.json), status COMPLETED, failure None
