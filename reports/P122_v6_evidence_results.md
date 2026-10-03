# P122 — v6 V6-EVIDENCE results (frozen encoders, common measurement augmentation; seed-0 encoders)

Increments are additional score recovered in these function classes and budgets (1000 updates, 3 lrs × 3 inits, TUNE selection), not true terms.
Unit = pair index (one P and one Q per anchor); 95 % bootstrap intervals over EVAL units.

## cifar10 — measurement law: common-standard

| encoder (train aug) | loss | J train | J s | J Z | J H | s − train | Z − s | H − Z | selected s / Z residual / H residual | min |
|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | vcs | 0.8771 | 0.9699 | 0.9740 | 0.9749 | +0.0928 [+0.0886, +0.0968] | +0.0041 [+0.0022, +0.0059] | +0.0009 [-0.0007, +0.0026] | affine / True / True | 3.2 |
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | js | 0.8771 | 0.9700 | 0.9745 | 0.9750 | +0.0930 [+0.0888, +0.0970] | +0.0045 [+0.0028, +0.0063] | +0.0004 [-0.0013, +0.0023] | mlp1d / True / True | 3.2 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | vcs | 0.9028 | 0.9751 | 0.9754 | 0.9758 | +0.0724 [+0.0692, +0.0755] | +0.0003 [-0.0004, +0.0010] | +0.0003 [-0.0007, +0.0015] | affine / True / True | 3.2 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | js | 0.9028 | 0.9754 | 0.9762 | 0.9766 | +0.0726 [+0.0696, +0.0756] | +0.0008 [+0.0001, +0.0016] | +0.0005 [+0.0001, +0.0010] | mlp1d / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | vcs | 0.9866 | 0.9939 | 0.9939 | 0.9934 | +0.0074 [+0.0059, +0.0089] | -0.0000 [-0.0003, +0.0002] | -0.0005 [-0.0008, -0.0002] | affine / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | js | 0.9866 | 0.9940 | 0.9939 | 0.9938 | +0.0074 [+0.0062, +0.0088] | -0.0001 [-0.0004, +0.0003] | -0.0002 [-0.0007, +0.0003] | affine / True / True | 3.2 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | vcs | — | 0.9796 | 0.9796 | 0.9802 | — | +0.0000 [-0.0000, +0.0000] | +0.0006 [+0.0000, +0.0012] | mlp1d / True / True | 3.2 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | js | — | 0.9797 | 0.9797 | 0.9805 | — | +0.0000 [-0.0000, +0.0000] | +0.0008 [+0.0003, +0.0015] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | vcs | — | 0.9745 | 0.9755 | 0.9745 | — | +0.0010 [+0.0005, +0.0015] | -0.0010 [-0.0028, +0.0006] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | js | — | 0.9745 | 0.9764 | 0.9755 | — | +0.0019 [+0.0008, +0.0030] | -0.0008 [-0.0027, +0.0010] | mlp1d / True / True | 3.2 |

## cifar10 — measurement law: common-strong

| encoder (train aug) | loss | J train | J s | J Z | J H | s − train | Z − s | H − Z | selected s / Z residual / H residual | min |
|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | vcs | 0.8218 | 0.8974 | 0.9005 | 0.9155 | +0.0757 [+0.0715, +0.0801] | +0.0031 [+0.0016, +0.0045] | +0.0149 [+0.0104, +0.0192] | mlp1d / True / True | 3.2 |
| P107_AP3_augstrong_views4_800ep_seed0 (crop≥0.08) | js | 0.8218 | 0.8973 | 0.9020 | 0.9207 | +0.0755 [+0.0713, +0.0800] | +0.0046 [+0.0021, +0.0071] | +0.0187 [+0.0130, +0.0244] | mlp1d / True / True | 3.2 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | vcs | 0.7097 | 0.7817 | 0.7908 | 0.8057 | +0.0720 [+0.0658, +0.0785] | +0.0091 [+0.0048, +0.0136] | +0.0149 [+0.0091, +0.0209] | mlp1d / True / True | 3.2 |
| P107_AP3_views4_800ep_seed0 (crop≥0.2) | js | 0.7097 | 0.7822 | 0.7912 | 0.8058 | +0.0725 [+0.0662, +0.0788] | +0.0090 [+0.0037, +0.0143] | +0.0146 [+0.0091, +0.0199] | mlp1d / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | vcs | 0.7868 | 0.8180 | 0.8318 | 0.8487 | +0.0312 [+0.0259, +0.0365] | +0.0138 [+0.0055, +0.0213] | +0.0169 [+0.0110, +0.0228] | bins32 / True / True | 3.2 |
| P35_vcs_a5_views4_800ep_seed0 (crop≥0.2) | js | 0.7868 | 0.8194 | 0.8403 | 0.8444 | +0.0326 [+0.0269, +0.0380] | +0.0208 [+0.0113, +0.0299] | +0.0042 [+0.0018, +0.0065] | mlp1d / True / True | 3.2 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | vcs | — | 0.7764 | 0.7837 | 0.7815 | — | +0.0073 [+0.0041, +0.0105] | -0.0022 [-0.0066, +0.0017] | bins32 / True / True | 3.2 |
| P41_simclr_views4_800ep_seed0 (crop≥0.2) | js | — | 0.7764 | 0.7854 | 0.7844 | — | +0.0090 [+0.0056, +0.0124] | -0.0010 [-0.0042, +0.0021] | bins32 / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | vcs | — | 0.9026 | 0.9030 | 0.9108 | — | +0.0004 [-0.0000, +0.0009] | +0.0078 [+0.0037, +0.0117] | mlp1d / True / True | 3.2 |
| P89_simclr_views4_800ep_augstrong_seed0 (crop≥0.08) | js | — | 0.9024 | 0.9038 | 0.9128 | — | +0.0014 [+0.0002, +0.0027] | +0.0090 [+0.0051, +0.0130] | mlp1d / True / True | 3.2 |

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
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | train | -1.746 | -1.766 (0.066) | 183.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | s | -0.784 | -2.449 (1.708) | 2.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | Z | -0.918 | -2.449 (1.700) | 2.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | vcs | H | -1.938 | -3.138 (1.628) | 2.9 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | train | -1.746 | -1.766 (0.066) | 183.7 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | s | -0.504 | -2.234 (1.687) | 3.6 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | Z | -0.546 | -2.233 (1.679) | 3.5 |
| cifar10 | standard | P107_AP3_views4_800ep_seed0 | js | H | -1.032 | -2.535 (1.623) | 3.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | train | -3.484 | -3.761 (0.628) | 18.7 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | s | -0.765 | -5.034 (2.707) | 2.2 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | Z | -0.544 | -4.862 (2.709) | 2.2 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | vcs | H | -0.272 | -4.878 (2.941) | 2.0 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | train | -3.484 | -3.761 (0.628) | 18.7 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | s | -1.465 | -4.762 (2.278) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | Z | -1.273 | -4.646 (2.281) | 2.5 |
| cifar10 | standard | P35_vcs_a5_views4_800ep_seed0 | js | H | -1.688 | -4.925 (2.361) | 2.5 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | vcs | s | -2.608 | -3.091 (1.027) | 8.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | vcs | Z | -2.604 | -3.087 (1.027) | 8.4 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | vcs | H | -3.121 | -3.419 (0.956) | 9.0 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | js | s | -1.654 | -2.808 (1.514) | 4.6 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | js | Z | -1.653 | -2.807 (1.514) | 4.6 |
| cifar10 | standard | P41_simclr_views4_800ep_seed0 | js | H | -2.562 | -3.181 (1.449) | 4.8 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | vcs | s | -2.092 | -2.727 (0.902) | 7.5 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | vcs | Z | -2.248 | -2.755 (0.906) | 7.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | vcs | H | -2.971 | -3.350 (0.991) | 6.4 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | js | s | -1.183 | -2.458 (1.449) | 4.3 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | js | Z | -1.558 | -2.507 (1.455) | 4.2 |
| cifar10 | standard | P89_simclr_views4_800ep_augstrong_seed0 | js | H | -2.452 | -3.154 (1.669) | 3.6 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | train | -1.508 | -1.538 (0.093) | 160.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | s | -0.813 | -1.400 (0.798) | 10.2 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | Z | -0.994 | -1.467 (0.746) | 9.9 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | vcs | H | -0.372 | -1.673 (1.446) | 6.7 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | train | -1.508 | -1.538 (0.093) | 160.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | s | -0.393 | -1.244 (0.971) | 9.0 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | Z | -0.752 | -1.332 (0.924) | 8.3 |
| cifar10 | strong | P107_AP3_augstrong_views4_800ep_seed0 | js | H | +1.733 | -1.013 (1.912) | 4.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | train | -1.725 | -1.762 (0.082) | 194.2 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | s | -0.879 | -1.073 (0.418) | 18.9 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | Z | -0.735 | -1.150 (0.719) | 18.1 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | vcs | H | -0.482 | -1.038 (1.069) | 12.7 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | train | -1.725 | -1.762 (0.082) | 194.2 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | s | -0.552 | -0.877 (0.678) | 17.6 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | Z | -0.496 | -0.960 (0.925) | 15.6 |
| cifar10 | strong | P107_AP3_views4_800ep_seed0 | js | H | -0.471 | -0.884 (1.268) | 10.3 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | train | -2.297 | -2.995 (0.990) | 27.4 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | s | +7.290 | -0.610 (3.224) | 16.1 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | Z | +9.031 | -0.559 (3.693) | 9.3 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | vcs | H | +9.139 | +0.385 (3.564) | 7.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | train | -2.297 | -2.995 (0.990) | 27.4 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | s | -0.496 | -1.586 (1.163) | 19.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | Z | +1.147 | -1.308 (1.740) | 8.2 |
| cifar10 | strong | P35_vcs_a5_views4_800ep_seed0 | js | H | +1.508 | -0.876 (1.686) | 8.0 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | vcs | s | -1.078 | -0.784 (2.122) | 23.1 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | vcs | Z | -0.984 | -0.750 (2.177) | 22.4 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | vcs | H | -1.435 | -1.361 (2.093) | 20.3 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | js | s | -1.078 | -0.784 (2.122) | 23.1 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | js | Z | -1.034 | -0.804 (2.191) | 21.7 |
| cifar10 | strong | P41_simclr_views4_800ep_seed0 | js | H | -1.478 | -1.379 (2.130) | 19.2 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | vcs | s | -1.450 | -1.681 (0.581) | 10.4 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | vcs | Z | -1.484 | -1.699 (0.587) | 10.4 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | vcs | H | -1.230 | -1.712 (1.106) | 7.8 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | js | s | -1.137 | -1.436 (0.873) | 8.6 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | js | Z | -1.214 | -1.465 (0.898) | 8.5 |
| cifar10 | strong | P89_simclr_views4_800ep_augstrong_seed0 | js | H | -0.575 | -1.352 (1.493) | 5.7 |
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
