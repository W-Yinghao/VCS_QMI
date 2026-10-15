# V8-05 confirmation — GenAI-Bench non-Rank prompts (generated from confirm.json, 2026-10-15T17:34:22Z)

Images scored 5100 over 850 prompts; dropped (missing in some method) 0; without rating 0.  Bootstrap: 5000 prompt-cluster resamples, seed 2718.

## primary_spearman

| contrast | metric | Δ [95 %] | p | p Holm | ours / comp |
|---|---|---|---|---|---|
| U-CS − B0_FGA_released_O | spearman | +0.003 [-0.008, +0.014] | 0.6056 | 1.0000 | 0.575 / 0.572 |
| U-CS − B1_VQAScore | spearman | -0.008 [-0.041, +0.024] | 0.6280 | 1.0000 | 0.575 / 0.583 |
| U-CS − B4T_TIFA | spearman | +0.312 [+0.267, +0.358] | 0.0002 | 0.0008 | 0.575 / 0.262 |
| U-CS − B4D_DSG | spearman | +0.193 [+0.156, +0.229] | 0.0002 | 0.0008 | 0.575 / 0.381 |

## primary_pairwise

| contrast | metric | Δ [95 %] | p | p Holm | ours / comp |
|---|---|---|---|---|---|
| U-CS − B0_FGA_released_O | pairwise | -0.001 [-0.010, +0.007] | 0.7680 | 0.7680 | 0.726 / 0.728 |
| U-CS − B1_VQAScore | pairwise | +0.044 [+0.025, +0.063] | 0.0002 | 0.0008 | 0.726 / 0.682 |
| U-CS − B4T_TIFA | pairwise | +0.146 [+0.129, +0.163] | 0.0002 | 0.0008 | 0.726 / 0.581 |
| U-CS − B4D_DSG | pairwise | +0.143 [+0.126, +0.160] | 0.0002 | 0.0008 | 0.726 / 0.583 |

## attribution

| contrast | metric | Δ [95 %] | p | p Holm | ours / comp |
|---|---|---|---|---|---|
| U-CS − U-BCE | spearman | -0.016 [-0.021, -0.011] | 0.0002 | 0.0002 | 0.575 / 0.590 |
| U-CS − U-BCE | pairwise | -0.001 [-0.005, +0.003] | 0.5880 | 0.5880 | 0.726 / 0.728 |

## same_data_descriptive

| contrast | metric | Δ [95 %] | p | p Holm | ours / comp |
|---|---|---|---|---|---|
| U-CS − U-RPS | spearman | +0.011 [+0.005, +0.018] | 0.0012 | nan | 0.575 / 0.563 |
| U-CS − U-RPS | pairwise | +0.001 [-0.004, +0.006] | 0.6140 | nan | 0.726 / 0.725 |
| U-CS − U-CE | spearman | +0.008 [+0.000, +0.016] | 0.0372 | nan | 0.575 / 0.566 |
| U-CS − U-CE | pairwise | +0.000 [-0.005, +0.005] | 0.9696 | nan | 0.726 / 0.726 |
| U-CS − U-MSE | spearman | +0.006 [+0.001, +0.012] | 0.0328 | nan | 0.575 / 0.568 |
| U-CS − U-MSE | pairwise | +0.003 [-0.002, +0.008] | 0.2360 | nan | 0.726 / 0.724 |
| U-CS − U-L1 | spearman | +0.004 [-0.002, +0.009] | 0.2024 | nan | 0.575 / 0.571 |
| U-CS − U-L1 | pairwise | +0.002 [-0.003, +0.007] | 0.5104 | nan | 0.726 / 0.725 |
| U-CS − U-RANK | spearman | +0.052 [+0.036, +0.067] | 0.0002 | nan | 0.575 / 0.523 |
| U-CS − U-RANK | pairwise | -0.002 [-0.009, +0.005] | 0.5636 | nan | 0.726 / 0.729 |
| U-CS − U-LIST | spearman | +0.016 [+0.006, +0.026] | 0.0008 | nan | 0.575 / 0.559 |
| U-CS − U-LIST | pairwise | -0.002 [-0.008, +0.005] | 0.6040 | nan | 0.726 / 0.728 |

## Levels (pooled Spearman / pairwise accuracy / Spearman per generator)

| method | Spearman | pairwise | DALLE_3 | DeepFloyd_I_XL_v1 | Midjourney_6 | SDXL_2_1 | SDXL_Base | SDXL_Turbo |
|---|---|---|---|---|---|---|---|---|
| U-CS | 0.575 | 0.726 | 0.438 | 0.512 | 0.521 | 0.603 | 0.587 | 0.564 |
| U-BCE | 0.590 | 0.728 | 0.458 | 0.531 | 0.537 | 0.618 | 0.605 | 0.585 |
| U-RPS | 0.563 | 0.725 | 0.405 | 0.502 | 0.500 | 0.609 | 0.582 | 0.561 |
| U-CE | 0.566 | 0.726 | 0.409 | 0.506 | 0.503 | 0.609 | 0.583 | 0.567 |
| U-MSE | 0.568 | 0.724 | 0.416 | 0.508 | 0.504 | 0.606 | 0.589 | 0.567 |
| U-L1 | 0.571 | 0.725 | 0.426 | 0.517 | 0.512 | 0.604 | 0.585 | 0.568 |
| U-RANK | 0.523 | 0.729 | 0.387 | 0.475 | 0.451 | 0.552 | 0.535 | 0.502 |
| U-LIST | 0.559 | 0.728 | 0.409 | 0.502 | 0.493 | 0.591 | 0.572 | 0.557 |
| B0_FGA_released_O | 0.572 | 0.728 | 0.422 | 0.512 | 0.518 | 0.615 | 0.591 | 0.568 |
| B1_VQAScore | 0.583 | 0.682 | 0.547 | 0.547 | 0.574 | 0.536 | 0.567 | 0.544 |
| B4T_TIFA | 0.262 | 0.581 | 0.216 | 0.237 | 0.264 | 0.298 | 0.256 | 0.241 |
| B4D_DSG | 0.381 | 0.583 | 0.379 | 0.341 | 0.326 | 0.399 | 0.382 | 0.358 |

## Per seed (V8 arms)

| unit | Spearman | pairwise |
|---|---|---|
| U-CS-s0 | 0.585 | 0.728 |
| U-CS-s1 | 0.526 | 0.715 |
| U-CS-s2 | 0.580 | 0.727 |
| U-BCE-s0 | 0.587 | 0.724 |
| U-BCE-s1 | 0.575 | 0.724 |
| U-BCE-s2 | 0.590 | 0.729 |
| U-RPS-s0 | 0.571 | 0.726 |
| U-RPS-s1 | 0.531 | 0.711 |
| U-RPS-s2 | 0.564 | 0.724 |
| U-CE-s0 | 0.574 | 0.729 |
| U-CE-s1 | 0.565 | 0.730 |
| U-CE-s2 | 0.542 | 0.719 |
| U-MSE-s0 | 0.574 | 0.731 |
| U-MSE-s1 | 0.526 | 0.711 |
| U-MSE-s2 | 0.577 | 0.728 |
| U-L1-s0 | 0.568 | 0.724 |
| U-L1-s1 | 0.524 | 0.708 |
| U-L1-s2 | 0.582 | 0.723 |
| U-RANK-s0 | 0.516 | 0.730 |
| U-RANK-s1 | 0.514 | 0.722 |
| U-RANK-s2 | 0.518 | 0.730 |
| U-LIST-s0 | 0.548 | 0.724 |
| U-LIST-s1 | 0.549 | 0.727 |
| U-LIST-s2 | 0.558 | 0.732 |
