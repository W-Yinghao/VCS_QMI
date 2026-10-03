# P122 — v6 V6-EVIDENCE results (frozen encoders, common measurement augmentation; seed-0 encoders)

Increments are additional score recovered in these function classes and budgets (1000 updates, 3 lrs × 3 inits, TUNE selection), not true terms.
Unit = pair index (one P and one Q per anchor); 95 % bootstrap intervals over EVAL units.

## cifar10 — measurement law: common-standard

| encoder (train aug) | loss | J train | J s | J Z | J H | s − train | Z − s | H − Z | selected s / Z residual / H residual | min |
|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | vcs | 0.8771 | 0.9699 | 0.9740 | 0.9749 | +0.0928 [+0.0886, +0.0968] | +0.0041 [+0.0022, +0.0059] | +0.0009 [-0.0007, +0.0026] | affine / True / True | 3.2 |
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | js | 0.8771 | 0.9700 | 0.9745 | 0.9750 | +0.0930 [+0.0888, +0.0970] | +0.0045 [+0.0028, +0.0063] | +0.0004 [-0.0013, +0.0023] | mlp1d / True / True | 3.2 |
| P107_AP3_augstrong_views4_800ep_seed1 (crop≥0.08) | vcs | 0.8768 | 0.9689 | 0.9732 | 0.9740 | +0.0921 [+0.0878, +0.0965] | +0.0043 [+0.0008, +0.0080] | +0.0009 [-0.0015, +0.0032] | mlp1d / True / True | 4.7 |
| P107_AP3_augstrong_views4_800ep_seed1 (crop≥0.08) | js | 0.8768 | 0.9692 | 0.9739 | 0.9748 | +0.0925 [+0.0884, +0.0967] | +0.0047 [+0.0028, +0.0066] | +0.0009 [-0.0010, +0.0031] | mlp1d / True / True | 4.7 |
| P107_AP3_augstrong_views4_800ep_seed2 (crop≥0.08) | vcs | 0.8768 | 0.9685 | 0.9732 | 0.9738 | +0.0917 [+0.0875, +0.0959] | +0.0046 [+0.0024, +0.0069] | +0.0007 [-0.0007, +0.0021] | mlp1d / True / True | 4.7 |
| P107_AP3_augstrong_views4_800ep_seed2 (crop≥0.08) | js | 0.8768 | 0.9685 | 0.9732 | 0.9739 | +0.0916 [+0.0874, +0.0958] | +0.0048 [+0.0028, +0.0068] | +0.0007 [-0.0013, +0.0028] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | vcs | 0.9028 | 0.9751 | 0.9754 | 0.9758 | +0.0724 [+0.0692, +0.0755] | +0.0003 [-0.0004, +0.0010] | +0.0003 [-0.0007, +0.0015] | affine / True / True | 3.2 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | js | 0.9028 | 0.9754 | 0.9762 | 0.9766 | +0.0726 [+0.0696, +0.0756] | +0.0008 [+0.0001, +0.0016] | +0.0005 [+0.0001, +0.0010] | mlp1d / True / True | 3.2 |
| P107_AP3_views4_800ep_seed1 (crop≥0.2) | vcs | 0.9016 | 0.9726 | 0.9728 | 0.9721 | +0.0710 [+0.0682, +0.0739] | +0.0003 [+0.0000, +0.0005] | -0.0008 [-0.0027, +0.0012] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed1 (crop≥0.2) | js | 0.9016 | 0.9725 | 0.9734 | 0.9728 | +0.0709 [+0.0678, +0.0741] | +0.0009 [+0.0001, +0.0018] | -0.0006 [-0.0023, +0.0011] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed2 (crop≥0.2) | vcs | 0.9025 | 0.9752 | 0.9755 | 0.9747 | +0.0727 [+0.0699, +0.0755] | +0.0003 [+0.0001, +0.0005] | -0.0008 [-0.0026, +0.0008] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed2 (crop≥0.2) | js | 0.9025 | 0.9752 | 0.9759 | 0.9761 | +0.0727 [+0.0697, +0.0755] | +0.0007 [+0.0004, +0.0011] | +0.0002 [+0.0001, +0.0004] | mlp1d / True / True | 4.7 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | vcs | 0.9866 | 0.9939 | 0.9939 | 0.9934 | +0.0074 [+0.0059, +0.0089] | -0.0000 [-0.0003, +0.0002] | -0.0005 [-0.0008, -0.0002] | affine / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | js | 0.9866 | 0.9940 | 0.9939 | 0.9938 | +0.0074 [+0.0062, +0.0088] | -0.0001 [-0.0004, +0.0003] | -0.0002 [-0.0007, +0.0003] | affine / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed1 (crop≥0.2) | vcs | 0.9862 | 0.9940 | 0.9940 | 0.9940 | +0.0078 [+0.0066, +0.0093] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | affine / True / False | 4.7 |
| P35_vcs_a5_views4_800ep_seed1 (crop≥0.2) | js | 0.9862 | 0.9939 | 0.9939 | 0.9939 | +0.0078 [+0.0066, +0.0091] | -0.0000 [-0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | affine / True / False | 4.7 |
| P35_vcs_a5_views4_800ep_seed2 (crop≥0.2) | vcs | 0.9850 | 0.9918 | 0.9918 | 0.9921 | +0.0068 [+0.0056, +0.0079] | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0001, +0.0006] | mlp1d / False / True | 4.7 |
| P35_vcs_a5_views4_800ep_seed2 (crop≥0.2) | js | 0.9850 | 0.9920 | 0.9920 | 0.9920 | +0.0069 [+0.0056, +0.0082] | -0.0000 [-0.0000, +0.0000] | +0.0001 [-0.0001, +0.0003] | affine / True / True | 4.7 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | vcs | — | 0.9796 | 0.9796 | 0.9802 | — | +0.0000 [-0.0000, +0.0000] | +0.0006 [+0.0000, +0.0012] | mlp1d / True / True | 3.2 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | js | — | 0.9797 | 0.9797 | 0.9805 | — | +0.0000 [-0.0000, +0.0000] | +0.0008 [+0.0003, +0.0015] | mlp1d / True / True | 3.2 |
| P41_simclr_views4_800ep_seed1 (crop≥0.2) | vcs | — | 0.9765 | 0.9765 | 0.9759 | — | -0.0000 [-0.0009, +0.0008] | -0.0006 [-0.0023, +0.0010] | affine / True / True | 4.7 |
| P41_simclr_views4_800ep_seed1 (crop≥0.2) | js | — | 0.9759 | 0.9761 | 0.9744 | — | +0.0002 [-0.0004, +0.0009] | -0.0017 [-0.0037, +0.0004] | mlp1d / True / True | 4.7 |
| P41_simclr_views4_800ep_seed2 (crop≥0.2) | vcs | — | 0.9788 | 0.9768 | 0.9772 | — | -0.0020 [-0.0034, -0.0006] | +0.0004 [-0.0000, +0.0008] | mlp1d / True / True | 4.7 |
| P41_simclr_views4_800ep_seed2 (crop≥0.2) | js | — | 0.9786 | 0.9772 | 0.9773 | — | -0.0015 [-0.0030, +0.0000] | +0.0002 [+0.0000, +0.0003] | mlp1d / True / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | vcs | — | 0.9745 | 0.9755 | 0.9745 | — | +0.0010 [+0.0005, +0.0015] | -0.0010 [-0.0028, +0.0006] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | js | — | 0.9745 | 0.9764 | 0.9755 | — | +0.0019 [+0.0008, +0.0030] | -0.0008 [-0.0027, +0.0010] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed1 (crop≥0.08) | vcs | — | 0.9730 | 0.9730 | 0.9744 | — | +0.0000 [+0.0000, +0.0000] | +0.0013 [-0.0002, +0.0029] | mlp1d / False / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed1 (crop≥0.08) | js | — | 0.9729 | 0.9742 | 0.9753 | — | +0.0013 [+0.0007, +0.0020] | +0.0011 [-0.0003, +0.0024] | mlp1d / True / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed2 (crop≥0.08) | vcs | — | 0.9723 | 0.9634 | 0.9634 | — | -0.0089 [-0.0140, -0.0038] | +0.0000 [+0.0000, +0.0000] | mlp1d / True / False | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed2 (crop≥0.08) | js | — | 0.9722 | 0.9726 | 0.9727 | — | +0.0004 [-0.0011, +0.0020] | +0.0001 [-0.0014, +0.0017] | mlp1d / True / True | 4.7 |

## cifar10 — measurement law: common-strong

| encoder (train aug) | loss | J train | J s | J Z | J H | s − train | Z − s | H − Z | selected s / Z residual / H residual | min |
|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | vcs | 0.8218 | 0.8974 | 0.9005 | 0.9155 | +0.0757 [+0.0715, +0.0801] | +0.0031 [+0.0016, +0.0045] | +0.0149 [+0.0104, +0.0192] | mlp1d / True / True | 3.2 |
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | js | 0.8218 | 0.8973 | 0.9020 | 0.9207 | +0.0755 [+0.0713, +0.0800] | +0.0046 [+0.0021, +0.0071] | +0.0187 [+0.0130, +0.0244] | mlp1d / True / True | 3.2 |
| P107_AP3_augstrong_views4_800ep_seed1 (crop≥0.08) | vcs | 0.8237 | 0.9005 | 0.9042 | 0.9145 | +0.0768 [+0.0723, +0.0812] | +0.0038 [+0.0008, +0.0064] | +0.0103 [+0.0064, +0.0146] | mlp1d / True / True | 4.8 |
| P107_AP3_augstrong_views4_800ep_seed1 (crop≥0.08) | js | 0.8237 | 0.9005 | 0.9045 | 0.9200 | +0.0768 [+0.0725, +0.0812] | +0.0039 [+0.0011, +0.0066] | +0.0156 [+0.0111, +0.0198] | mlp1d / True / True | 4.8 |
| P107_AP3_augstrong_views4_800ep_seed2 (crop≥0.08) | vcs | 0.8216 | 0.8961 | 0.8995 | 0.9121 | +0.0745 [+0.0700, +0.0792] | +0.0034 [+0.0020, +0.0047] | +0.0126 [+0.0089, +0.0160] | mlp1d / True / True | 4.7 |
| P107_AP3_augstrong_views4_800ep_seed2 (crop≥0.08) | js | 0.8216 | 0.8959 | 0.9006 | 0.9161 | +0.0743 [+0.0698, +0.0791] | +0.0047 [+0.0021, +0.0071] | +0.0155 [+0.0106, +0.0203] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | vcs | 0.7097 | 0.7817 | 0.7908 | 0.8057 | +0.0720 [+0.0658, +0.0785] | +0.0091 [+0.0048, +0.0136] | +0.0149 [+0.0091, +0.0209] | mlp1d / True / True | 3.2 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | js | 0.7097 | 0.7822 | 0.7912 | 0.8058 | +0.0725 [+0.0662, +0.0788] | +0.0090 [+0.0037, +0.0143] | +0.0146 [+0.0091, +0.0199] | mlp1d / True / True | 3.2 |
| P107_AP3_views4_800ep_seed1 (crop≥0.2) | vcs | 0.7084 | 0.7827 | 0.7863 | 0.8023 | +0.0743 [+0.0681, +0.0800] | +0.0036 [-0.0014, +0.0083] | +0.0160 [+0.0092, +0.0230] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed1 (crop≥0.2) | js | 0.7084 | 0.7827 | 0.7879 | 0.8066 | +0.0744 [+0.0682, +0.0800] | +0.0051 [-0.0006, +0.0105] | +0.0187 [+0.0135, +0.0237] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed2 (crop≥0.2) | vcs | 0.7067 | 0.7797 | 0.7868 | 0.7960 | +0.0731 [+0.0668, +0.0792] | +0.0071 [+0.0035, +0.0110] | +0.0092 [+0.0022, +0.0162] | mlp1d / True / True | 4.7 |
| P107_AP3_views4_800ep_seed2 (crop≥0.2) | js | 0.7067 | 0.7799 | 0.7879 | 0.8009 | +0.0732 [+0.0670, +0.0793] | +0.0080 [+0.0040, +0.0122] | +0.0131 [+0.0078, +0.0180] | mlp1d / True / True | 4.7 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | vcs | 0.7868 | 0.8180 | 0.8318 | 0.8487 | +0.0312 [+0.0259, +0.0365] | +0.0138 [+0.0055, +0.0213] | +0.0169 [+0.0110, +0.0228] | bins32 / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | js | 0.7868 | 0.8194 | 0.8403 | 0.8444 | +0.0326 [+0.0269, +0.0380] | +0.0208 [+0.0113, +0.0299] | +0.0042 [+0.0018, +0.0065] | mlp1d / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed1 (crop≥0.2) | vcs | 0.7814 | 0.8157 | 0.8340 | 0.8392 | +0.0343 [+0.0286, +0.0396] | +0.0182 [+0.0109, +0.0257] | +0.0053 [-0.0053, +0.0162] | mlp1d / True / True | 4.7 |
| P35_vcs_a5_views4_800ep_seed1 (crop≥0.2) | js | 0.7814 | 0.8161 | 0.8364 | 0.8468 | +0.0347 [+0.0291, +0.0399] | +0.0202 [+0.0095, +0.0303] | +0.0105 [+0.0071, +0.0139] | mlp1d / True / True | 4.7 |
| P35_vcs_a5_views4_800ep_seed2 (crop≥0.2) | vcs | 0.7874 | 0.8185 | 0.8285 | 0.8402 | +0.0312 [+0.0258, +0.0366] | +0.0100 [+0.0011, +0.0185] | +0.0117 [+0.0065, +0.0167] | bins32 / True / True | 4.7 |
| P35_vcs_a5_views4_800ep_seed2 (crop≥0.2) | js | 0.7874 | 0.8204 | 0.8445 | 0.8514 | +0.0330 [+0.0276, +0.0383] | +0.0241 [+0.0150, +0.0332] | +0.0069 [+0.0025, +0.0114] | mlp1d / True / True | 4.7 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | vcs | — | 0.7764 | 0.7837 | 0.7815 | — | +0.0073 [+0.0041, +0.0105] | -0.0022 [-0.0066, +0.0017] | bins32 / True / True | 3.2 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | js | — | 0.7764 | 0.7854 | 0.7844 | — | +0.0090 [+0.0056, +0.0124] | -0.0010 [-0.0042, +0.0021] | bins32 / True / True | 3.2 |
| P41_simclr_views4_800ep_seed1 (crop≥0.2) | vcs | — | 0.7792 | 0.7828 | 0.7833 | — | +0.0037 [-0.0010, +0.0083] | +0.0005 [-0.0023, +0.0032] | mlp1d / True / True | 4.7 |
| P41_simclr_views4_800ep_seed1 (crop≥0.2) | js | — | 0.7796 | 0.7874 | 0.7870 | — | +0.0078 [+0.0034, +0.0126] | -0.0004 [-0.0030, +0.0022] | mlp1d / True / True | 4.7 |
| P41_simclr_views4_800ep_seed2 (crop≥0.2) | vcs | — | 0.7830 | 0.7904 | 0.7920 | — | +0.0075 [+0.0045, +0.0103] | +0.0016 [-0.0002, +0.0033] | mlp1d / True / True | 4.7 |
| P41_simclr_views4_800ep_seed2 (crop≥0.2) | js | — | 0.7856 | 0.7931 | 0.7924 | — | +0.0075 [+0.0042, +0.0106] | -0.0007 [-0.0030, +0.0017] | mlp1d / True / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | vcs | — | 0.9026 | 0.9030 | 0.9108 | — | +0.0004 [-0.0000, +0.0009] | +0.0078 [+0.0037, +0.0117] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | js | — | 0.9024 | 0.9038 | 0.9128 | — | +0.0014 [+0.0002, +0.0027] | +0.0090 [+0.0051, +0.0130] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed1 (crop≥0.08) | vcs | — | 0.9006 | 0.9014 | 0.9044 | — | +0.0008 [+0.0000, +0.0015] | +0.0030 [+0.0017, +0.0044] | affine / True / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed1 (crop≥0.08) | js | — | 0.9010 | 0.9020 | 0.9066 | — | +0.0011 [-0.0001, +0.0022] | +0.0046 [+0.0020, +0.0070] | mlp1d / True / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed2 (crop≥0.08) | vcs | — | 0.9041 | 0.9048 | 0.9120 | — | +0.0007 [+0.0003, +0.0012] | +0.0072 [+0.0042, +0.0102] | mlp1d / True / True | 4.7 |
| P89_simclr_views4_800ep_augstrong_seed2 (crop≥0.08) | js | — | 0.9039 | 0.9051 | 0.9125 | — | +0.0012 [+0.0004, +0.0020] | +0.0074 [+0.0042, +0.0104] | mlp1d / True / True | 4.7 |

## cifar100 — measurement law: common-standard

| encoder (train aug) | loss | J train | J s | J Z | J H | s − train | Z − s | H − Z | selected s / Z residual / H residual | min |
|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_c100_views4_800ep_seed0 (crop≥0.2) | vcs | 0.9016 | 0.9777 | 0.9734 | 0.9747 | +0.0761 [+0.0730, +0.0788] | -0.0044 [-0.0077, -0.0012] | +0.0014 [-0.0020, +0.0049] | mlp1d / True / True | 2.5 |
| P107_AP3_c100_views4_800ep_seed0 (crop≥0.2) | js | 0.9016 | 0.9778 | 0.9778 | 0.9781 | +0.0762 [+0.0731, +0.0790] | -0.0000 [-0.0013, +0.0013] | +0.0003 [-0.0009, +0.0016] | mlp1d / True / True | 2.5 |
| P91_c100_simclr_views4_800ep_seed0 (crop≥0.2) | vcs | — | 0.9752 | 0.9753 | 0.9746 | — | +0.0000 [-0.0000, +0.0000] | -0.0007 [-0.0011, -0.0002] | affine / True / True | 2.5 |
| P91_c100_simclr_views4_800ep_seed0 (crop≥0.2) | js | — | 0.9755 | 0.9755 | 0.9741 | — | -0.0000 [-0.0000, +0.0000] | -0.0014 [-0.0023, -0.0004] | mlp1d / True / True | 2.5 |
| P91_c100_vcs_a5_views4_800ep_seed0 (crop≥0.2) | vcs | 0.9865 | 0.9930 | 0.9928 | 0.9928 | +0.0065 [+0.0045, +0.0085] | -0.0002 [-0.0005, +0.0000] | +0.0000 [+0.0000, +0.0000] | affine / True / False | 2.5 |
| P91_c100_vcs_a5_views4_800ep_seed0 (crop≥0.2) | js | 0.9865 | 0.9934 | 0.9922 | 0.9922 | +0.0069 [+0.0051, +0.0088] | -0.0013 [-0.0026, -0.0002] | +0.0000 [+0.0000, +0.0000] | affine / True / False | 2.5 |

## Density-ratio consistency (necessary-condition diagnostic only; f = 0 also gives 0)

| dataset | law | encoder | loss | level | global log E_Q e^{2f} | per-anchor mean (sd) | per-anchor ESS |
|---|---|---|---|---|---|---|---|
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | vcs | train | -1.533 | -1.553 (0.078) | 158.3 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | vcs | s | -1.293 | -2.644 (1.334) | 4.4 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | vcs | Z | -2.108 | -2.907 (1.144) | 4.2 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | vcs | H | -1.537 | -3.050 (1.401) | 3.7 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | js | train | -1.533 | -1.553 (0.078) | 158.3 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | js | s | -0.250 | -2.202 (1.618) | 4.2 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | js | Z | -1.214 | -2.460 (1.379) | 3.9 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed0 | js | H | -0.767 | -2.579 (1.741) | 3.3 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | vcs | train | -1.543 | -1.549 (0.081) | 157.8 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | vcs | s | -1.392 | -2.440 (1.487) | 3.9 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | vcs | Z | -1.705 | -2.287 (1.705) | 3.3 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | vcs | H | +13.558 | -1.725 (2.595) | 2.6 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | js | train | -1.543 | -1.549 (0.081) | 157.8 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | js | s | -0.539 | -2.135 (1.591) | 4.4 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | js | Z | -1.393 | -2.211 (1.404) | 4.1 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed1 | js | H | -0.926 | -2.507 (1.862) | 3.1 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | vcs | train | -1.534 | -1.551 (0.078) | 158.1 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | vcs | s | -1.399 | -2.479 (1.307) | 5.5 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | vcs | Z | -2.102 | -2.758 (1.160) | 5.0 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | vcs | H | -2.165 | -3.050 (1.434) | 4.3 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | js | train | -1.534 | -1.551 (0.078) | 158.1 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | js | s | -0.622 | -2.171 (1.610) | 4.5 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | js | Z | -1.351 | -2.378 (1.410) | 4.2 |
| cifar10 | standard | P107_AP3_augstrong_views4_800ep_seed2 | js | H | -1.423 | -2.725 (1.755) | 3.5 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | train | -1.746 | -1.766 (0.066) | 183.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | s | -0.784 | -2.449 (1.708) | 2.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | Z | -0.918 | -2.449 (1.700) | 2.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | H | -1.938 | -3.138 (1.628) | 2.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | train | -1.746 | -1.766 (0.066) | 183.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | s | -0.504 | -2.234 (1.687) | 3.6 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | Z | -0.546 | -2.233 (1.679) | 3.5 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | H | -1.032 | -2.535 (1.623) | 3.5 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | vcs | train | -1.736 | -1.763 (0.067) | 182.0 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | vcs | s | -1.488 | -2.366 (1.124) | 5.4 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | vcs | Z | -1.558 | -2.367 (1.113) | 5.3 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | vcs | H | -1.115 | -2.555 (1.437) | 4.1 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | js | train | -1.736 | -1.763 (0.067) | 182.0 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | js | s | -0.720 | -2.075 (1.543) | 3.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | js | Z | -0.963 | -2.094 (1.503) | 3.6 |
| cifar10 | standard | P107_AP3_views4_800ep_seed1 | js | H | -1.043 | -2.509 (1.806) | 3.1 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | vcs | train | -1.739 | -1.765 (0.064) | 183.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | vcs | s | -1.689 | -2.522 (1.175) | 5.2 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | vcs | Z | -1.776 | -2.526 (1.161) | 5.1 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | vcs | H | -2.597 | -3.059 (1.162) | 4.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | js | train | -1.739 | -1.765 (0.064) | 183.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | js | s | -1.179 | -2.328 (1.431) | 4.0 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | js | Z | -1.291 | -2.373 (1.424) | 3.8 |
| cifar10 | standard | P107_AP3_views4_800ep_seed2 | js | H | -1.427 | -2.445 (1.405) | 3.8 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | train | -3.484 | -3.761 (0.628) | 18.7 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | s | -0.765 | -5.034 (2.707) | 2.2 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | Z | -0.544 | -4.862 (2.709) | 2.2 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | H | -0.272 | -4.878 (2.941) | 2.0 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | train | -3.484 | -3.761 (0.628) | 18.7 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | s | -1.465 | -4.762 (2.278) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | Z | -1.273 | -4.646 (2.281) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | H | -1.688 | -4.925 (2.361) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | vcs | train | -3.459 | -3.744 (0.611) | 18.9 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | vcs | s | -0.649 | -4.788 (2.165) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | vcs | Z | -0.640 | -4.782 (2.165) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | vcs | H | -0.640 | -4.782 (2.165) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | js | train | -3.459 | -3.744 (0.611) | 18.9 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | js | s | -0.443 | -4.600 (2.171) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | js | Z | -0.439 | -4.596 (2.171) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed1 | js | H | -0.439 | -4.596 (2.171) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | vcs | train | -3.379 | -3.761 (0.605) | 18.3 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | vcs | s | -2.019 | -4.369 (1.800) | 2.9 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | vcs | Z | -2.019 | -4.369 (1.800) | 2.9 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | vcs | H | -2.336 | -4.620 (1.829) | 2.8 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | js | train | -3.379 | -3.761 (0.605) | 18.3 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | js | s | -0.917 | -4.473 (2.212) | 2.4 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | js | Z | -0.917 | -4.473 (2.212) | 2.4 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed2 | js | H | -1.035 | -4.596 (2.245) | 2.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | vcs | s | -2.608 | -3.091 (1.027) | 8.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | vcs | Z | -2.604 | -3.087 (1.027) | 8.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | vcs | H | -3.121 | -3.419 (0.956) | 9.0 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | js | s | -1.654 | -2.808 (1.514) | 4.6 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | js | Z | -1.653 | -2.807 (1.514) | 4.6 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | js | H | -2.562 | -3.181 (1.449) | 4.8 |
| cifar10 | standard | P41_simclr_views4_800ep_seed1 | vcs | s | -2.500 | -2.902 (1.026) | 5.9 |
| cifar10 | standard | P41_simclr_views4_800ep_seed1 | vcs | Z | -2.823 | -2.970 (1.049) | 5.9 |
| cifar10 | standard | P41_simclr_views4_800ep_seed1 | vcs | H | -2.677 | -2.931 (1.444) | 4.3 |
| cifar10 | standard | P41_simclr_views4_800ep_seed1 | js | s | -1.812 | -2.496 (1.388) | 3.6 |
| cifar10 | standard | P41_simclr_views4_800ep_seed1 | js | Z | -2.030 | -2.518 (1.423) | 3.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed1 | js | H | -1.082 | -2.112 (2.208) | 2.3 |
| cifar10 | standard | P41_simclr_views4_800ep_seed2 | vcs | s | -2.259 | -2.940 (1.084) | 8.1 |
| cifar10 | standard | P41_simclr_views4_800ep_seed2 | vcs | Z | -2.143 | -2.893 (1.269) | 7.9 |
| cifar10 | standard | P41_simclr_views4_800ep_seed2 | vcs | H | -2.616 | -3.138 (1.233) | 8.2 |
| cifar10 | standard | P41_simclr_views4_800ep_seed2 | js | s | -1.558 | -2.705 (1.490) | 4.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed2 | js | Z | -0.714 | -2.643 (1.732) | 3.9 |
| cifar10 | standard | P41_simclr_views4_800ep_seed2 | js | H | -0.982 | -2.735 (1.712) | 3.9 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | vcs | s | -2.092 | -2.727 (0.902) | 7.5 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | vcs | Z | -2.248 | -2.755 (0.906) | 7.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | vcs | H | -2.971 | -3.350 (0.991) | 6.4 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | js | s | -1.183 | -2.458 (1.449) | 4.3 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | js | Z | -1.558 | -2.507 (1.455) | 4.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | js | H | -2.452 | -3.154 (1.669) | 3.6 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed1 | vcs | s | -1.899 | -2.763 (1.061) | 6.5 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed1 | vcs | Z | -1.899 | -2.763 (1.061) | 6.5 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed1 | vcs | H | -3.007 | -3.497 (1.081) | 6.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed1 | js | s | -0.709 | -2.484 (1.542) | 4.3 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed1 | js | Z | -0.973 | -2.513 (1.533) | 4.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed1 | js | H | -2.403 | -3.291 (1.674) | 3.7 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed2 | vcs | s | -1.710 | -2.695 (1.295) | 5.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed2 | vcs | Z | +7.975 | -0.429 (4.717) | 1.7 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed2 | vcs | H | +7.975 | -0.429 (4.717) | 1.7 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed2 | js | s | -0.966 | -2.475 (1.613) | 4.1 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed2 | js | Z | -1.587 | -2.576 (1.714) | 3.8 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed2 | js | H | -1.078 | -2.646 (2.197) | 2.8 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | train | -1.508 | -1.538 (0.093) | 160.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | s | -0.813 | -1.400 (0.798) | 10.2 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | Z | -0.994 | -1.467 (0.746) | 9.9 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | H | -0.372 | -1.673 (1.446) | 6.7 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | train | -1.508 | -1.538 (0.093) | 160.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | s | -0.393 | -1.244 (0.971) | 9.0 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | Z | -0.752 | -1.332 (0.924) | 8.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | H | +1.733 | -1.013 (1.912) | 4.7 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | vcs | train | -1.507 | -1.537 (0.091) | 160.0 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | vcs | s | -0.832 | -1.372 (0.792) | 9.7 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | vcs | Z | -1.228 | -1.521 (0.740) | 9.2 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | vcs | H | -0.972 | -1.760 (1.357) | 6.0 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | js | train | -1.507 | -1.537 (0.091) | 160.0 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | js | s | -0.471 | -1.237 (0.943) | 8.9 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | js | Z | -0.991 | -1.346 (0.857) | 8.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed1 | js | H | -0.764 | -1.378 (1.355) | 5.8 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | vcs | train | -1.513 | -1.538 (0.089) | 160.7 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | vcs | s | -1.050 | -1.467 (0.693) | 11.6 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | vcs | Z | -1.184 | -1.524 (0.667) | 11.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | vcs | H | -1.292 | -1.891 (1.068) | 8.9 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | js | train | -1.513 | -1.538 (0.089) | 160.7 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | js | s | -0.580 | -1.236 (0.917) | 9.2 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | js | Z | -0.887 | -1.349 (0.895) | 8.6 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed2 | js | H | -0.168 | -1.250 (1.525) | 5.8 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | train | -1.725 | -1.762 (0.082) | 194.2 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | s | -0.879 | -1.073 (0.418) | 18.9 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | Z | -0.735 | -1.150 (0.719) | 18.1 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | H | -0.482 | -1.038 (1.069) | 12.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | train | -1.725 | -1.762 (0.082) | 194.2 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | s | -0.552 | -0.877 (0.678) | 17.6 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | Z | -0.496 | -0.960 (0.925) | 15.6 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | H | -0.471 | -0.884 (1.268) | 10.3 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | vcs | train | -1.725 | -1.761 (0.081) | 193.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | vcs | s | -0.920 | -1.117 (0.419) | 18.9 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | vcs | Z | -0.719 | -1.131 (0.773) | 18.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | vcs | H | +0.100 | -0.706 (1.309) | 9.4 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | js | train | -1.725 | -1.761 (0.081) | 193.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | js | s | -0.593 | -0.882 (0.594) | 15.5 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | js | Z | -0.461 | -0.921 (0.957) | 13.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed1 | js | H | -0.061 | -0.894 (1.185) | 9.6 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | vcs | train | -1.724 | -1.759 (0.084) | 194.0 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | vcs | s | -0.891 | -1.089 (0.460) | 20.4 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | vcs | Z | -0.780 | -1.141 (0.693) | 20.1 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | vcs | H | -0.481 | -1.254 (1.388) | 11.0 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | js | train | -1.724 | -1.759 (0.084) | 194.0 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | js | s | -0.376 | -0.848 (0.730) | 18.4 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | js | Z | -0.445 | -0.916 (0.906) | 17.2 |
| cifar10 | strong | P107_AP3_views4_800ep_seed2 | js | H | -0.531 | -1.090 (1.209) | 12.0 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | train | -2.297 | -2.995 (0.990) | 27.4 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | s | +7.290 | -0.610 (3.224) | 16.1 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | Z | +9.031 | -0.559 (3.693) | 9.3 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | H | +9.139 | +0.385 (3.564) | 7.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | train | -2.297 | -2.995 (0.990) | 27.4 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | s | -0.496 | -1.586 (1.163) | 19.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | Z | +1.147 | -1.308 (1.740) | 8.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | H | +1.508 | -0.876 (1.686) | 8.0 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | vcs | train | -2.327 | -3.019 (0.962) | 27.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | vcs | s | -0.993 | -1.840 (1.031) | 22.4 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | vcs | Z | +1.355 | -0.896 (1.792) | 7.3 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | vcs | H | +4.927 | +1.074 (2.281) | 3.9 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | js | train | -2.327 | -3.019 (0.962) | 27.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | js | s | -0.387 | -1.588 (1.167) | 19.0 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | js | Z | +1.284 | -1.101 (1.771) | 8.1 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed1 | js | H | +1.835 | -0.493 (1.872) | 6.6 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | vcs | train | -2.328 | -3.003 (0.974) | 26.9 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | vcs | s | -0.255 | -1.258 (2.129) | 17.7 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | vcs | Z | +1.288 | -0.634 (3.068) | 5.5 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | vcs | H | +4.091 | +0.927 (3.621) | 4.3 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | js | train | -2.328 | -3.003 (0.974) | 26.9 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | js | s | -0.428 | -1.581 (1.168) | 18.0 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | js | Z | +0.446 | -1.181 (1.770) | 7.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed2 | js | H | +1.847 | -0.041 (2.087) | 5.4 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | vcs | s | -1.078 | -0.784 (2.122) | 23.1 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | vcs | Z | -0.984 | -0.750 (2.177) | 22.4 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | vcs | H | -1.435 | -1.361 (2.093) | 20.3 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | js | s | -1.078 | -0.784 (2.122) | 23.1 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | js | Z | -1.034 | -0.804 (2.191) | 21.7 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | js | H | -1.478 | -1.379 (2.130) | 19.2 |
| cifar10 | strong | P41_simclr_views4_800ep_seed1 | vcs | s | -1.307 | -1.340 (0.363) | 21.4 |
| cifar10 | strong | P41_simclr_views4_800ep_seed1 | vcs | Z | -0.985 | -1.358 (0.740) | 20.7 |
| cifar10 | strong | P41_simclr_views4_800ep_seed1 | vcs | H | -1.349 | -1.779 (0.777) | 19.5 |
| cifar10 | strong | P41_simclr_views4_800ep_seed1 | js | s | -1.144 | -1.158 (0.578) | 19.6 |
| cifar10 | strong | P41_simclr_views4_800ep_seed1 | js | Z | -0.989 | -1.234 (0.856) | 18.6 |
| cifar10 | strong | P41_simclr_views4_800ep_seed1 | js | H | -1.332 | -1.691 (0.929) | 16.0 |
| cifar10 | strong | P41_simclr_views4_800ep_seed2 | vcs | s | -1.088 | -1.154 (0.365) | 18.4 |
| cifar10 | strong | P41_simclr_views4_800ep_seed2 | vcs | Z | -0.978 | -1.205 (0.564) | 17.3 |
| cifar10 | strong | P41_simclr_views4_800ep_seed2 | vcs | H | -1.310 | -1.544 (0.637) | 17.3 |
| cifar10 | strong | P41_simclr_views4_800ep_seed2 | js | s | -0.921 | -1.052 (0.675) | 18.0 |
| cifar10 | strong | P41_simclr_views4_800ep_seed2 | js | Z | -0.852 | -1.126 (0.828) | 17.2 |
| cifar10 | strong | P41_simclr_views4_800ep_seed2 | js | H | -1.380 | -1.634 (0.858) | 17.0 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | vcs | s | -1.450 | -1.681 (0.581) | 10.4 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | vcs | Z | -1.484 | -1.699 (0.587) | 10.4 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | vcs | H | -1.230 | -1.712 (1.106) | 7.8 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | js | s | -1.137 | -1.436 (0.873) | 8.6 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | js | Z | -1.214 | -1.465 (0.898) | 8.5 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | js | H | -0.575 | -1.352 (1.493) | 5.7 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed1 | vcs | s | -0.955 | -1.226 (0.761) | 9.0 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed1 | vcs | Z | -0.999 | -1.259 (0.769) | 8.9 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed1 | vcs | H | -1.406 | -1.628 (0.771) | 9.1 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed1 | js | s | -1.088 | -1.374 (0.850) | 8.0 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed1 | js | Z | -1.142 | -1.398 (0.874) | 7.9 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed1 | js | H | -1.312 | -1.712 (1.140) | 6.5 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed2 | vcs | s | -1.481 | -1.729 (0.568) | 11.4 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed2 | vcs | Z | -1.518 | -1.741 (0.572) | 11.3 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed2 | vcs | H | -1.716 | -1.892 (0.887) | 10.2 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed2 | js | s | -1.103 | -1.464 (0.885) | 8.7 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed2 | js | Z | -1.212 | -1.494 (0.890) | 8.6 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed2 | js | H | -1.346 | -1.615 (1.213) | 7.0 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | vcs | train | -1.787 | -1.764 (0.061) | 195.5 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | vcs | s | -1.121 | -2.371 (1.526) | 5.4 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | vcs | Z | +13.589 | +3.592 (7.051) | 1.6 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | vcs | H | +41.072 | +5.390 (11.673) | 1.5 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | js | train | -1.787 | -1.764 (0.061) | 195.5 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | js | s | +0.260 | -2.012 (1.999) | 3.7 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | js | Z | +0.296 | -2.046 (2.064) | 3.5 |
| cifar100 | standard | P107_AP3_c100_views4_800ep_seed0 | js | H | +0.598 | -2.268 (2.385) | 3.1 |
| cifar100 | standard | P91_c100_simclr_views4_800ep_seed0 | vcs | s | -1.630 | -2.995 (1.061) | 7.1 |
| cifar100 | standard | P91_c100_simclr_views4_800ep_seed0 | vcs | Z | -1.635 | -2.995 (1.061) | 7.1 |
| cifar100 | standard | P91_c100_simclr_views4_800ep_seed0 | vcs | H | -1.985 | -3.329 (0.969) | 7.6 |
| cifar100 | standard | P91_c100_simclr_views4_800ep_seed0 | js | s | +0.248 | -2.622 (1.688) | 4.0 |
| cifar100 | standard | P91_c100_simclr_views4_800ep_seed0 | js | Z | +0.237 | -2.623 (1.688) | 4.0 |
| cifar100 | standard | P91_c100_simclr_views4_800ep_seed0 | js | H | -0.343 | -3.282 (1.590) | 4.1 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | vcs | train | -3.624 | -3.652 (0.583) | 21.9 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | vcs | s | -2.170 | -5.244 (2.632) | 2.3 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | vcs | Z | -1.960 | -5.484 (2.920) | 2.1 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | vcs | H | -1.960 | -5.484 (2.920) | 2.1 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | js | train | -3.624 | -3.652 (0.583) | 21.9 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | js | s | -1.605 | -5.005 (2.802) | 2.2 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | js | Z | +0.132 | -4.963 (3.380) | 2.1 |
| cifar100 | standard | P91_c100_vcs_a5_views4_800ep_seed0 | js | H | +0.132 | -4.963 (3.380) | 2.1 |
