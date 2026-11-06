Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR6 results (numbers only; DEV, fixed step 8000; no TEST)

Protocol CR6_SOURCE_REPLICATE_AND_STUDENT_SCOPE_V1; frozen sha bec67c4ad18d6bd835c78335ef6b3c225f9cbbf3e62f9bdc88a632800f59dac1

## Group GPUs

- A2_Flickr60_t1002: {'gpu_name': 'Tesla V100S-PCIE-32GB', 'driver': '580.126.18'}
- A2_Flickr60_t1003: {'gpu_name': 'Tesla V100S-PCIE-32GB', 'driver': '580.126.18'}
- B_CC152K_RECOVERED_20261104_t1002: {'gpu_name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'driver': '615.71.10'}
- B_Flickr60_t1002: {'gpu_name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'driver': '615.71.10'}
- A1: {'gpu_name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'driver': '615.71.10'}

## A: source replicate (Flickr60; new = src2002 graphs, old = CR4 graphs re-verified/re-run in the same group)

| student | contrast | new source | old source | new−old |
|---|---|---|---|---|
| 1002 | J-BCE | -0.75 | -0.06 | -0.69 |
| 1002 | J-J_MASS_OBS | -0.30 | +0.36 | -0.65 |
| 1002 | J-OBS | -0.28 | +0.10 | -0.37 |
| 1002 | J-BASE_KEEP_TOP1 | +0.00 | +0.37 | -0.37 |
| 1002 | J-NCE_KEEP_TOP1 | +0.00 | +0.37 | -0.37 |
| 1002 | J_KEEP_TOP1 new-old source | 560.63 | 561.01 | -0.37 |
| 1002 | BCE_KEEP_TOP1 new-old source | 561.38 | 561.07 | +0.32 |
| 1002 | J_MASS_OBS new-old source | 560.93 | 560.65 | +0.28 |
| 1003 | J-BCE | -0.51 | +0.24 | -0.75 |
| 1003 | J-J_MASS_OBS | -0.22 | -0.04 | -0.18 |
| 1003 | J-OBS | -0.55 | -0.18 | -0.37 |
| 1003 | J-BASE_KEEP_TOP1 | -0.37 | +0.00 | -0.37 |
| 1003 | J-NCE_KEEP_TOP1 | +0.28 | +0.65 | -0.37 |
| 1003 | J_KEEP_TOP1 new-old source | 560.77 | 561.14 | -0.37 |
| 1003 | BCE_KEEP_TOP1 new-old source | 561.28 | 560.91 | +0.37 |
| 1003 | J_MASS_OBS new-old source | 560.99 | 561.18 | -0.20 |

## B: scope effect D_adapt = TAIL − FROZEN (same graph, same group)

| condition | graph | TAIL | FROZEN | D_adapt | TAIL student-cos full | FROZEN student-cos full |
|---|---|---|---|---|---|---|
| Flickr60 | OBS | 566.49 | 560.91 | +5.58 | 567.42 | 558.44 |
| Flickr60 | J_MASS_OBS | 566.45 | 560.65 | +5.80 | 567.73 | 558.44 |
| Flickr60 | BASE_KEEP_TOP1 | 565.17 | 560.63 | +4.54 | 566.25 | 558.44 |
| Flickr60 | J_KEEP_TOP1 | 565.54 | 561.01 | +4.54 | 565.98 | 558.44 |
| Flickr60 | BCE_KEEP_TOP1 | 565.27 | 561.07 | +4.20 | 566.09 | 558.44 |
| Flickr60 | NCE_KEEP_TOP1 | 565.33 | 560.63 | +4.69 | 566.02 | 558.44 |
| CC152K_RECOVERED_20261104 | OBS | 537.23 | 531.75 | +5.48 | 530.30 | 518.61 |
| CC152K_RECOVERED_20261104 | J_MASS_OBS | 537.66 | 531.60 | +6.06 | 530.30 | 518.61 |
| CC152K_RECOVERED_20261104 | BASE_KEEP_TOP1 | 536.80 | 531.46 | +5.34 | 529.29 | 518.61 |
| CC152K_RECOVERED_20261104 | J_KEEP_TOP1 | 537.66 | 532.32 | +5.34 | 529.29 | 518.61 |
| CC152K_RECOVERED_20261104 | BCE_KEEP_TOP1 | 536.22 | 532.76 | +3.46 | 529.44 | 518.61 |
| CC152K_RECOVERED_20261104 | NCE_KEEP_TOP1 | 537.23 | 532.18 | +5.05 | 529.44 | 518.61 |

## B: scope × source interaction I = D_source(TAIL) − D_source(FROZEN)

| condition | contrast | D_source TAIL | D_source FROZEN | I |
|---|---|---|---|---|
| Flickr60 | J-OBS | -0.95 | +0.10 | -1.05 |
| Flickr60 | J-J_MASS_OBS | -0.91 | +0.36 | -1.26 |
| Flickr60 | J-BASE_KEEP_TOP1 | +0.37 | +0.37 | -0.00 |
| Flickr60 | J-BCE_KEEP_TOP1 | +0.28 | -0.06 | +0.34 |
| Flickr60 | J-NCE_KEEP_TOP1 | +0.22 | +0.37 | -0.16 |
| CC152K_RECOVERED_20261104 | J-OBS | +0.43 | +0.58 | -0.14 |
| CC152K_RECOVERED_20261104 | J-J_MASS_OBS | +0.00 | +0.72 | -0.72 |
| CC152K_RECOVERED_20261104 | J-BASE_KEEP_TOP1 | +0.87 | +0.87 | -0.00 |
| CC152K_RECOVERED_20261104 | J-BCE_KEEP_TOP1 | +1.44 | -0.43 | +1.88 |
| CC152K_RECOVERED_20261104 | J-NCE_KEEP_TOP1 | +0.43 | +0.14 | +0.29 |

## Grouped resampling (image groups; fixed predictions; descriptive)

| contrast | student | diff | 95% interval |
|---|---|---|---|
| J_KEEP_TOP1 src2002 - src1001 | 1002 | -0.37 | [-1.34, +0.57] |
| BCE_KEEP_TOP1 src2002 - src1001 | 1002 | +0.32 | [-0.41, +1.10] |
| J_MASS_OBS src2002 - src1001 | 1002 | +0.28 | [-0.63, +1.18] |
| J_KEEP_TOP1 src2002 - src1001 | 1003 | -0.37 | [-1.05, +0.26] |
| BCE_KEEP_TOP1 src2002 - src1001 | 1003 | +0.37 | [-0.41, +1.16] |
| J_MASS_OBS src2002 - src1001 | 1003 | -0.20 | [-0.93, +0.55] |
| Flickr60 OBS TAIL-FROZEN | 1002 | +5.58 | [+3.89, +7.42] |
| Flickr60 J_MASS_OBS TAIL-FROZEN | 1002 | +5.80 | [+4.04, +7.67] |
| Flickr60 BASE_KEEP_TOP1 TAIL-FROZEN | 1002 | +4.54 | [+3.04, +6.09] |
| Flickr60 J_KEEP_TOP1 TAIL-FROZEN | 1002 | +4.54 | [+2.94, +6.19] |
| Flickr60 BCE_KEEP_TOP1 TAIL-FROZEN | 1002 | +4.20 | [+2.82, +5.64] |
| Flickr60 NCE_KEEP_TOP1 TAIL-FROZEN | 1002 | +4.69 | [+3.10, +6.21] |
| CC152K_RECOVERED_20261104 OBS TAIL-FROZEN | 1002 | +5.48 | [+2.74, +8.37] |
| CC152K_RECOVERED_20261104 J_MASS_OBS TAIL-FROZEN | 1002 | +6.06 | [+3.17, +9.24] |
| CC152K_RECOVERED_20261104 BASE_KEEP_TOP1 TAIL-FROZEN | 1002 | +5.34 | [+2.45, +8.23] |
| CC152K_RECOVERED_20261104 J_KEEP_TOP1 TAIL-FROZEN | 1002 | +5.34 | [+2.45, +8.51] |
| CC152K_RECOVERED_20261104 BCE_KEEP_TOP1 TAIL-FROZEN | 1002 | +3.46 | [+0.58, +6.35] |
| CC152K_RECOVERED_20261104 NCE_KEEP_TOP1 TAIL-FROZEN | 1002 | +5.05 | [+2.16, +7.94] |

Files: TASK_ENDPOINTS_CR6.csv, TASK_CURVES_CR6.csv, A_SOURCE_CONTRASTS.csv, B_SCOPE_EFFECTS.csv, B_SCOPE_X_SOURCE_INTERACTION.csv, GROUPED_RESAMPLING_CR6.csv,
SOURCE_VS_SOURCE_DIAGNOSTICS.json, CR4_CR5_CR5P_PC_SIX_RECALLS.csv, CR4_CR5_CR5P_PC_GROUPED_J_DIFFS.csv, COSTS_CR6.csv.
Source and student layers are reported separately; 2 sources x 2 students are not four independent replicates. Intervals: fixed DEV/predictions only.
