# P157 (SEP-2) aggregate

Effects: (e_VCS − e_JS) / mean e_JS of the cell (negative = VCS more accurate); paired t over 8 seeds.

## Selection: selected — outcome: **CONFIRMED (persists at 4x N)** (64 cells)

| test | mean [95 % t] | p | p Holm | cells | complete | reading |
|---|---|---|---|---|---|---|
| Q1_N4096 | -0.2613 [-0.2700, -0.2526] | 2.97e-11 | 2.97e-11 | 32 | True | < 0 (VCS more accurate) |
| Q2_N16384 | -0.2749 [-0.2792, -0.2705] | 1.54e-13 | 4.63e-13 | 32 | True | < 0 (VCS more accurate) |
| Q3_cos2_bothN | -0.2027 [-0.2066, -0.1988] | 6.5e-13 | 1.3e-12 | 32 | True | < 0 (VCS more accurate) |

### Cells (absolute Δe, Holm over all cells)

| class | type | ε | d | I | N | e_JS | e_VCS | Δe [95 % t] | p Holm | reading |
|---|---|---|---|---|---|---|---|---|---|---|
| cos2 | false_negative | 0.1 | 20 | 4 | 4096 | 0.0354 | 0.0318 | -0.0037 [-0.0043, -0.0031] | 3.37e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 20 | 4 | 16384 | 0.0350 | 0.0319 | -0.0031 [-0.0037, -0.0025] | 6.26e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 20 | 8 | 4096 | 0.0343 | 0.0301 | -0.0042 [-0.0051, -0.0032] | 6.8e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 20 | 8 | 16384 | 0.0347 | 0.0294 | -0.0052 [-0.0056, -0.0049] | 1.37e-07 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 64 | 4 | 4096 | 0.0222 | 0.0170 | -0.0052 [-0.0066, -0.0038] | 9.85e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 64 | 4 | 16384 | 0.0229 | 0.0165 | -0.0064 [-0.0067, -0.0060] | 4.4e-08 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 64 | 8 | 4096 | 0.0325 | 0.0213 | -0.0112 [-0.0120, -0.0105] | 2.38e-07 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.1 | 64 | 8 | 16384 | 0.0337 | 0.0212 | -0.0125 [-0.0130, -0.0120] | 1.06e-08 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 20 | 4 | 4096 | 0.0478 | 0.0445 | -0.0034 [-0.0045, -0.0022] | 0.000208 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 20 | 4 | 16384 | 0.0475 | 0.0443 | -0.0032 [-0.0037, -0.0027] | 3.16e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 20 | 8 | 4096 | 0.0460 | 0.0419 | -0.0041 [-0.0051, -0.0031] | 7.01e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 20 | 8 | 16384 | 0.0465 | 0.0418 | -0.0047 [-0.0052, -0.0041] | 6.15e-06 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 64 | 4 | 4096 | 0.0455 | 0.0407 | -0.0048 [-0.0053, -0.0043] | 2.36e-06 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 64 | 4 | 16384 | 0.0453 | 0.0406 | -0.0047 [-0.0055, -0.0039] | 3.37e-05 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 64 | 8 | 4096 | 0.0558 | 0.0496 | -0.0062 [-0.0068, -0.0055] | 2.19e-06 | < 0 (VCS more accurate) |
| cos2 | false_negative | 0.3 | 64 | 8 | 16384 | 0.0565 | 0.0497 | -0.0068 [-0.0072, -0.0065] | 5.02e-08 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 20 | 4 | 4096 | 0.0450 | 0.0335 | -0.0115 [-0.0134, -0.0096] | 3.56e-05 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 20 | 4 | 16384 | 0.0455 | 0.0336 | -0.0119 [-0.0131, -0.0106] | 2.53e-06 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 20 | 8 | 4096 | 0.0627 | 0.0343 | -0.0284 [-0.0290, -0.0278] | 8.9e-11 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 20 | 8 | 16384 | 0.0621 | 0.0336 | -0.0285 [-0.0295, -0.0276] | 2.06e-09 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 64 | 4 | 4096 | 0.0261 | 0.0173 | -0.0088 [-0.0106, -0.0069] | 6.26e-05 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 64 | 4 | 16384 | 0.0257 | 0.0169 | -0.0088 [-0.0098, -0.0079] | 3.31e-06 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 64 | 8 | 4096 | 0.0410 | 0.0219 | -0.0191 [-0.0199, -0.0183] | 9.67e-09 | < 0 (VCS more accurate) |
| cos2 | independent | 0.1 | 64 | 8 | 16384 | 0.0420 | 0.0219 | -0.0201 [-0.0206, -0.0197] | 1.58e-10 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 20 | 4 | 4096 | 0.0674 | 0.0605 | -0.0069 [-0.0083, -0.0055] | 6.26e-05 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 20 | 4 | 16384 | 0.0676 | 0.0602 | -0.0074 [-0.0080, -0.0068] | 7.46e-07 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 20 | 8 | 4096 | 0.0914 | 0.0802 | -0.0112 [-0.0132, -0.0093] | 3.66e-05 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 20 | 8 | 16384 | 0.0922 | 0.0798 | -0.0124 [-0.0129, -0.0119] | 4.38e-09 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 64 | 4 | 4096 | 0.0508 | 0.0452 | -0.0056 [-0.0066, -0.0046] | 5.24e-05 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 64 | 4 | 16384 | 0.0518 | 0.0452 | -0.0066 [-0.0071, -0.0061] | 3.57e-07 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 64 | 8 | 4096 | 0.0704 | 0.0615 | -0.0089 [-0.0098, -0.0081] | 1.57e-06 | < 0 (VCS more accurate) |
| cos2 | independent | 0.3 | 64 | 8 | 16384 | 0.0714 | 0.0614 | -0.0100 [-0.0107, -0.0094] | 9.44e-08 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 20 | 4 | 4096 | 0.0149 | 0.0086 | -0.0063 [-0.0076, -0.0050] | 6.26e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 20 | 4 | 16384 | 0.0145 | 0.0084 | -0.0061 [-0.0067, -0.0055] | 2.62e-06 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 20 | 8 | 4096 | 0.0191 | 0.0129 | -0.0063 [-0.0071, -0.0054] | 1.65e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 20 | 8 | 16384 | 0.0195 | 0.0126 | -0.0069 [-0.0073, -0.0066] | 4.22e-08 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 64 | 4 | 4096 | 0.0161 | 0.0090 | -0.0072 [-0.0084, -0.0059] | 3.66e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 64 | 4 | 16384 | 0.0162 | 0.0087 | -0.0075 [-0.0079, -0.0070] | 7.79e-08 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 64 | 8 | 4096 | 0.0268 | 0.0143 | -0.0125 [-0.0143, -0.0106] | 2.37e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.1 | 64 | 8 | 16384 | 0.0272 | 0.0139 | -0.0134 [-0.0142, -0.0126] | 8.64e-08 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 20 | 4 | 4096 | 0.0355 | 0.0313 | -0.0043 [-0.0051, -0.0034] | 6.26e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 20 | 4 | 16384 | 0.0356 | 0.0310 | -0.0045 [-0.0053, -0.0038] | 3.66e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 20 | 8 | 4096 | 0.0340 | 0.0305 | -0.0036 [-0.0041, -0.0030] | 3.37e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 20 | 8 | 16384 | 0.0341 | 0.0304 | -0.0037 [-0.0041, -0.0033] | 3.14e-06 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 64 | 4 | 4096 | 0.0415 | 0.0363 | -0.0052 [-0.0059, -0.0044] | 1.9e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 64 | 4 | 16384 | 0.0415 | 0.0360 | -0.0055 [-0.0063, -0.0047] | 1.93e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 64 | 8 | 4096 | 0.0503 | 0.0449 | -0.0054 [-0.0064, -0.0043] | 5.87e-05 | < 0 (VCS more accurate) |
| quad3 | false_negative | 0.3 | 64 | 8 | 16384 | 0.0509 | 0.0445 | -0.0064 [-0.0067, -0.0060] | 4.94e-08 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 20 | 4 | 4096 | 0.0334 | 0.0118 | -0.0216 [-0.0247, -0.0185] | 1.93e-05 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 20 | 4 | 16384 | 0.0351 | 0.0117 | -0.0233 [-0.0241, -0.0225] | 1.94e-09 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 20 | 8 | 4096 | 0.0639 | 0.0177 | -0.0462 [-0.0481, -0.0442] | 8.69e-09 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 20 | 8 | 16384 | 0.0657 | 0.0169 | -0.0487 [-0.0494, -0.0481] | 3.45e-12 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 64 | 4 | 4096 | 0.0215 | 0.0105 | -0.0110 [-0.0134, -0.0085] | 6.8e-05 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 64 | 4 | 16384 | 0.0213 | 0.0098 | -0.0115 [-0.0124, -0.0106] | 6.28e-07 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 64 | 8 | 4096 | 0.0412 | 0.0159 | -0.0254 [-0.0274, -0.0233] | 5.86e-07 | < 0 (VCS more accurate) |
| quad3 | independent | 0.1 | 64 | 8 | 16384 | 0.0422 | 0.0155 | -0.0268 [-0.0273, -0.0262] | 9.99e-11 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 20 | 4 | 4096 | 0.0703 | 0.0583 | -0.0120 [-0.0131, -0.0110] | 1e-06 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 20 | 4 | 16384 | 0.0702 | 0.0575 | -0.0127 [-0.0136, -0.0118] | 2.38e-07 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 20 | 8 | 4096 | 0.1005 | 0.0848 | -0.0157 [-0.0182, -0.0133] | 2.96e-05 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 20 | 8 | 16384 | 0.1008 | 0.0845 | -0.0163 [-0.0175, -0.0151] | 3.18e-07 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 64 | 4 | 4096 | 0.0514 | 0.0446 | -0.0068 [-0.0079, -0.0058] | 2.62e-05 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 64 | 4 | 16384 | 0.0527 | 0.0446 | -0.0081 [-0.0088, -0.0074] | 8.6e-07 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 64 | 8 | 4096 | 0.0737 | 0.0635 | -0.0102 [-0.0113, -0.0091] | 2.99e-06 | < 0 (VCS more accurate) |
| quad3 | independent | 0.3 | 64 | 8 | 16384 | 0.0752 | 0.0633 | -0.0119 [-0.0126, -0.0112] | 9.01e-08 | < 0 (VCS more accurate) |

### Population block S2P (N 131,072, 8,000 updates, seed 200; descriptive)

| class | type | ε | d | I | e_JS | e_VCS | Δe | lr js / vcs |
|---|---|---|---|---|---|---|---|---|
| cos2 | false_negative | 0.1 | 20 | 4 | 0.0350 | 0.0317 | -0.0033 | 0.003 / 0.01 |
| cos2 | false_negative | 0.1 | 20 | 8 | 0.0351 | 0.0299 | -0.0053 | 0.03 / 0.03 |
| cos2 | false_negative | 0.1 | 64 | 4 | 0.0226 | 0.0163 | -0.0063 | 0.1 / 0.03 |
| cos2 | false_negative | 0.1 | 64 | 8 | 0.0324 | 0.0206 | -0.0118 | 0.03 / 0.1 |
| cos2 | false_negative | 0.3 | 20 | 4 | 0.0472 | 0.0442 | -0.0030 | 0.003 / 0.01 |
| cos2 | false_negative | 0.3 | 20 | 8 | 0.0461 | 0.0417 | -0.0044 | 0.1 / 0.1 |
| cos2 | false_negative | 0.3 | 64 | 4 | 0.0453 | 0.0407 | -0.0047 | 0.01 / 0.01 |
| cos2 | false_negative | 0.3 | 64 | 8 | 0.0553 | 0.0492 | -0.0062 | 0.003 / 0.003 |
| cos2 | independent | 0.1 | 20 | 4 | 0.0450 | 0.0335 | -0.0115 | 0.01 / 0.1 |
| cos2 | independent | 0.1 | 20 | 8 | 0.0627 | 0.0336 | -0.0291 | 0.01 / 0.1 |
| cos2 | independent | 0.1 | 64 | 4 | 0.0253 | 0.0166 | -0.0086 | 0.01 / 0.1 |
| cos2 | independent | 0.1 | 64 | 8 | 0.0419 | 0.0216 | -0.0202 | 0.1 / 0.3 |
| cos2 | independent | 0.3 | 20 | 4 | 0.0672 | 0.0601 | -0.0071 | 0.003 / 0.003 |
| cos2 | independent | 0.3 | 20 | 8 | 0.0915 | 0.0794 | -0.0121 | 0.03 / 0.01 |
| cos2 | independent | 0.3 | 64 | 4 | 0.0514 | 0.0450 | -0.0064 | 0.003 / 0.01 |
| cos2 | independent | 0.3 | 64 | 8 | 0.0703 | 0.0612 | -0.0092 | 0.03 / 0.003 |
| quad3 | false_negative | 0.1 | 20 | 4 | 0.0143 | 0.0082 | -0.0061 | 0.01 / 0.01 |
| quad3 | false_negative | 0.1 | 20 | 8 | 0.0192 | 0.0125 | -0.0067 | 0.3 / 0.03 |
| quad3 | false_negative | 0.1 | 64 | 4 | 0.0165 | 0.0086 | -0.0079 | 0.01 / 0.01 |
| quad3 | false_negative | 0.1 | 64 | 8 | 0.0271 | 0.0136 | -0.0135 | 0.03 / 0.03 |
| quad3 | false_negative | 0.3 | 20 | 4 | 0.0347 | 0.0310 | -0.0037 | 0.003 / 0.01 |
| quad3 | false_negative | 0.3 | 20 | 8 | 0.0338 | 0.0302 | -0.0036 | 0.01 / 0.003 |
| quad3 | false_negative | 0.3 | 64 | 4 | 0.0413 | 0.0362 | -0.0052 | 0.03 / 0.003 |
| quad3 | false_negative | 0.3 | 64 | 8 | 0.0505 | 0.0443 | -0.0061 | 0.03 / 0.1 |
| quad3 | independent | 0.1 | 20 | 4 | 0.0347 | 0.0114 | -0.0234 | 0.003 / 0.01 |
| quad3 | independent | 0.1 | 20 | 8 | 0.0654 | 0.0169 | -0.0485 | 0.01 / 0.03 |
| quad3 | independent | 0.1 | 64 | 4 | 0.0214 | 0.0098 | -0.0117 | 0.01 / 0.01 |
| quad3 | independent | 0.1 | 64 | 8 | 0.0417 | 0.0154 | -0.0263 | 0.3 / 0.03 |
| quad3 | independent | 0.3 | 20 | 4 | 0.0705 | 0.0572 | -0.0133 | 0.03 / 0.03 |
| quad3 | independent | 0.3 | 20 | 8 | 0.1005 | 0.0839 | -0.0166 | 0.01 / 0.003 |
| quad3 | independent | 0.3 | 64 | 4 | 0.0522 | 0.0444 | -0.0078 | 0.03 / 0.003 |
| quad3 | independent | 0.3 | 64 | 8 | 0.0730 | 0.0632 | -0.0098 | 0.3 / 0.3 |

### QC (selected rows): top-lr (1.0) / low-lr / not converged

- cos2:js: {'n': 256, 'top_lr': 49, 'low_lr': 0, 'not_converged': 18}
- cos2:vcs: {'n': 256, 'top_lr': 36, 'low_lr': 0, 'not_converged': 19}
- quad3:js: {'n': 256, 'top_lr': 47, 'low_lr': 0, 'not_converged': 13}
- quad3:vcs: {'n': 256, 'top_lr': 26, 'low_lr': 0, 'not_converged': 17}

## Selection: selected_commonJ — outcome: **CONFIRMED (persists at 4x N)** (64 cells)

| test | mean [95 % t] | p | p Holm | cells | complete | reading |
|---|---|---|---|---|---|---|
| Q1_N4096 | -0.1404 [-0.1462, -0.1345] | 1.41e-10 | 2.83e-10 | 32 | True | < 0 (VCS more accurate) |
| Q2_N16384 | -0.1458 [-0.1508, -0.1408] | 3.43e-11 | 1.03e-10 | 32 | True | < 0 (VCS more accurate) |
| Q3_cos2_bothN | -0.0893 [-0.0940, -0.0847] | 6.69e-10 | 6.69e-10 | 32 | True | < 0 (VCS more accurate) |

