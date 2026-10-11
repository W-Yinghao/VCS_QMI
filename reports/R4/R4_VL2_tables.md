# R4-VL2 — detector-score reference ν for VCS / JS: results

Gates (step-0 given-box DEV within 0.3 of the frozen raw value): 12 / 12 pass.  Missing: none.

## refcocog

| arm | DEV Acc@0.5 |
|---|---|
| VCS(ν), corrected rank (primary) | 73.81 ± 0.27 |
| JS(ν), corrected rank | 73.68 ± 0.21 |
| VCS(ν), plain cosine | 67.12 ± 0.49 |
| JS(ν), plain cosine | 66.23 ± 0.33 |
| VCS(uniform) — R4-VL1 | 71.73 ± 0.21 |
| JS(uniform) — R4-VL1 | 71.36 ± 0.27 |
| softmax — R4-VL1 | 73.58 ± 0.36 |
| VCS(uniform) + log s (no training) | 74.03 ± 0.07 |
| JS(uniform) + log s (no training) | 73.59 ± 0.21 |
| softmax + log s (no training) | 74.32 ± 0.18 |

| ν-trained secondary | VCS(ν) | JS(ν) |
|---|---|---|
| Acc@0.75 | 58.76 ± 0.22 | 58.63 ± 0.49 |
| J_recal (ν laws) | 51.02 ± 0.15 | 51.32 ± 0.39 |
| margin (×100 cos) | 4.98 ± 0.17 | 4.57 ± 0.78 |
| given-box Top-1 | 82.81 ± 0.24 | 82.46 ± 0.29 |
| top-K (K on CAL) | 73.60 ± 0.31 | 73.49 ± 0.29 |
| selected epoch / K | [5, 6, 5] / [32, 32, 32] | [4, 3, 6] / [32, 32, 32] |

| contrast (paired by seed) | value |
|---|---|
| C1_vcs_nu_minus_softmax | +0.23 [-1.34, +1.80] |
| C2_vcs_nu_minus_vcs_uniform | +2.08 [+0.91, +3.24] |
| C3a_vcs_nu_minus_softmax_fused | -0.51 [-1.52, +0.50] |
| C3b_vcs_nu_minus_vcs_uniform_fused | -0.22 [-0.83, +0.39] |
| C4_vcs_nu_minus_js_nu | +0.13 [-0.30, +0.57] |
| js_nu_minus_js_uniform | +2.32 [+1.21, +3.43] |
| fusion_softmax | +0.74 [-0.04, +1.52] |
| fusion_vcs_uniform | +2.30 [+1.74, +2.85] |
| fusion_js_uniform | +2.24 [+1.53, +2.94] |

Plain-rank reproduction from the score files (|diff| max): 0.24 points.

## refcocoplus

| arm | DEV Acc@0.5 |
|---|---|
| VCS(ν), corrected rank (primary) | 75.49 ± 0.12 |
| JS(ν), corrected rank | 75.57 ± 0.12 |
| VCS(ν), plain cosine | 67.94 ± 1.02 |
| JS(ν), plain cosine | 67.62 ± 1.05 |
| VCS(uniform) — R4-VL1 | 73.03 ± 0.34 |
| JS(uniform) — R4-VL1 | 72.86 ± 0.16 |
| softmax — R4-VL1 | 76.25 ± 0.45 |
| VCS(uniform) + log s (no training) | 75.09 ± 0.52 |
| JS(uniform) + log s (no training) | 74.92 ± 0.50 |
| softmax + log s (no training) | 76.86 ± 0.34 |

| ν-trained secondary | VCS(ν) | JS(ν) |
|---|---|---|
| Acc@0.75 | 62.42 ± 0.40 | 62.40 ± 0.24 |
| J_recal (ν laws) | 54.82 ± 0.81 | 55.43 ± 0.15 |
| margin (×100 cos) | 4.65 ± 1.18 | 4.14 ± 0.43 |
| given-box Top-1 | 83.49 ± 0.30 | 83.10 ± 0.35 |
| top-K (K on CAL) | 75.37 ± 0.10 | 75.43 ± 0.09 |
| selected epoch / K | [7, 3, 3] / [32, 32, 32] | [3, 3, 4] / [32, 32, 32] |

| contrast (paired by seed) | value |
|---|---|
| C1_vcs_nu_minus_softmax | -0.76 [-1.68, +0.17] |
| C2_vcs_nu_minus_vcs_uniform | +2.47 [+1.54, +3.40] |
| C3a_vcs_nu_minus_softmax_fused | -1.36 [-2.10, -0.63] |
| C3b_vcs_nu_minus_vcs_uniform_fused | +0.40 [-0.85, +1.65] |
| C4_vcs_nu_minus_js_nu | -0.08 [-0.21, +0.05] |
| js_nu_minus_js_uniform | +2.72 [+2.03, +3.40] |
| fusion_softmax | +0.61 [+0.26, +0.96] |
| fusion_vcs_uniform | +2.06 [+1.41, +2.72] |
| fusion_js_uniform | +2.07 [+0.47, +3.66] |

Plain-rank reproduction from the score files (|diff| max): 0.08 points.

Pooled over the two datasets (descriptive): C1_vcs_nu_minus_softmax -0.26; C2_vcs_nu_minus_vcs_uniform +2.27; C3a_vcs_nu_minus_softmax_fused -0.94; C3b_vcs_nu_minus_vcs_uniform_fused +0.09
