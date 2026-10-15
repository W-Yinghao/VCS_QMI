# P159 (SEP-3) aggregate — clean-EVAL readouts on frozen natural features

Effects (positive = VCS better): dJ_rel = (J_VCS − J_JS) / mean J_JS of the cell; dA_rel = (AUROC_VCS − AUROC_JS) / mean (1 − AUROC_JS); paired t over seeds.

## Selection: selected — outcome: **FIT-ONLY** (40 cells; missing 0)

| test | mean [95 % t] | p | p Holm | cells | complete | reading |
|---|---|---|---|---|---|---|
| Q1_dJ_eps_gt_0 | +0.0562 [+0.0548, +0.0576] | 4.47e-12 | 1.34e-11 | 32 | True | > 0 (VCS better) |
| Q2_dJ_eps_0 | +0.0020 [+0.0014, +0.0026] | 7.61e-05 | 7.61e-05 | 8 | True | > 0 (VCS better) |
| Q3_dA_eps_gt_0 | +0.0485 [+0.0455, +0.0515] | 2.09e-09 | 4.18e-09 | 32 | True | > 0 (VCS better) |

### Cells: dJ_rel (Holm over cells)

| source | class | type | ε | N | JS | VCS | effect [95 % t] | p Holm | reading |
|---|---|---|---|---|---|---|---|---|---|
| cifar_ap3 | cos2 | clean | 0 | 4096 | 0.8661 | 0.8672 | +0.0014 [-0.0003, +0.0030] | 0.272 | n.s. |
| cifar_ap3 | cos2 | clean | 0 | 8192 | 0.8656 | 0.8671 | +0.0017 [+0.0012, +0.0022] | 0.00103 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.1 | 4096 | 0.8145 | 0.8616 | +0.0579 [+0.0542, +0.0616] | 8.12e-08 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.1 | 8192 | 0.8181 | 0.8625 | +0.0543 [+0.0506, +0.0579] | 1.24e-07 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.3 | 4096 | 0.7036 | 0.7450 | +0.0588 [+0.0529, +0.0646] | 1.43e-06 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.3 | 8192 | 0.7039 | 0.7455 | +0.0591 [+0.0555, +0.0627] | 6.68e-08 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.1 | 4096 | 0.8494 | 0.8569 | +0.0088 [+0.0059, +0.0117] | 0.00143 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.1 | 8192 | 0.8494 | 0.8562 | +0.0080 [+0.0065, +0.0094] | 5.22e-05 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.3 | 4096 | 0.7625 | 0.7713 | +0.0115 [+0.0064, +0.0166] | 0.00647 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.3 | 8192 | 0.7635 | 0.7717 | +0.0108 [+0.0068, +0.0148] | 0.00263 | > 0 (VCS better) |
| cifar_ap3 | quad3 | clean | 0 | 4096 | 0.8333 | 0.8392 | +0.0070 [+0.0049, +0.0092] | 0.00103 | > 0 (VCS better) |
| cifar_ap3 | quad3 | clean | 0 | 8192 | 0.8342 | 0.8392 | +0.0060 [+0.0049, +0.0071] | 5.22e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.1 | 4096 | 0.7439 | 0.8347 | +0.1220 [+0.1080, +0.1360] | 3.03e-06 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.1 | 8192 | 0.7489 | 0.8338 | +0.1133 [+0.1049, +0.1218] | 2.28e-07 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.3 | 4096 | 0.6218 | 0.7245 | +0.1652 [+0.1398, +0.1906] | 1.9e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.3 | 8192 | 0.6173 | 0.7243 | +0.1734 [+0.1590, +0.1877] | 4.33e-07 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.1 | 4096 | 0.7867 | 0.8315 | +0.0570 [+0.0490, +0.0650] | 1.05e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.1 | 8192 | 0.7866 | 0.8321 | +0.0579 [+0.0538, +0.0620] | 1.76e-07 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.3 | 4096 | 0.6773 | 0.7282 | +0.0752 [+0.0592, +0.0912] | 0.000127 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.3 | 8192 | 0.6813 | 0.7287 | +0.0697 [+0.0623, +0.0771] | 2.02e-06 | > 0 (VCS better) |
| coco_clip | cos2 | clean | 0 | 4096 | 0.9293 | 0.9296 | +0.0003 [-0.0004, +0.0010] | 0.693 | n.s. |
| coco_clip | cos2 | clean | 0 | 16384 | 0.9296 | 0.9298 | +0.0002 [-0.0000, +0.0003] | 0.216 | n.s. |
| coco_clip | cos2 | false_negative | 0.1 | 4096 | 0.8931 | 0.9181 | +0.0280 [+0.0252, +0.0308] | 1.4e-06 | > 0 (VCS better) |
| coco_clip | cos2 | false_negative | 0.1 | 16384 | 0.8952 | 0.9196 | +0.0272 [+0.0260, +0.0284] | 6.26e-09 | > 0 (VCS better) |
| coco_clip | cos2 | false_negative | 0.3 | 4096 | 0.7835 | 0.8135 | +0.0383 [+0.0342, +0.0425] | 2.2e-06 | > 0 (VCS better) |
| coco_clip | cos2 | false_negative | 0.3 | 16384 | 0.7883 | 0.8170 | +0.0365 [+0.0346, +0.0383] | 1.81e-08 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.1 | 4096 | 0.8789 | 0.9249 | +0.0524 [+0.0494, +0.0553] | 3.48e-08 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.1 | 16384 | 0.8796 | 0.9264 | +0.0532 [+0.0511, +0.0554] | 4.01e-09 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.3 | 4096 | 0.7570 | 0.7987 | +0.0550 [+0.0496, +0.0605] | 1.43e-06 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.3 | 16384 | 0.7603 | 0.8037 | +0.0570 [+0.0549, +0.0592] | 2.69e-09 | > 0 (VCS better) |
| coco_clip | quad3 | clean | 0 | 4096 | 0.9290 | 0.9292 | +0.0002 [-0.0002, +0.0006] | 0.693 | n.s. |
| coco_clip | quad3 | clean | 0 | 16384 | 0.9292 | 0.9287 | -0.0005 [-0.0009, -0.0001] | 0.0846 | n.s. |
| coco_clip | quad3 | false_negative | 0.1 | 4096 | 0.8939 | 0.9167 | +0.0255 [+0.0214, +0.0296] | 2.29e-05 | > 0 (VCS better) |
| coco_clip | quad3 | false_negative | 0.1 | 16384 | 0.8949 | 0.9189 | +0.0268 [+0.0248, +0.0288] | 2.41e-07 | > 0 (VCS better) |
| coco_clip | quad3 | false_negative | 0.3 | 4096 | 0.7860 | 0.8159 | +0.0381 [+0.0278, +0.0483] | 0.000548 | > 0 (VCS better) |
| coco_clip | quad3 | false_negative | 0.3 | 16384 | 0.7886 | 0.8176 | +0.0368 [+0.0323, +0.0413] | 4.48e-06 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.1 | 4096 | 0.8780 | 0.9233 | +0.0516 [+0.0496, +0.0536] | 3.43e-09 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.1 | 16384 | 0.8798 | 0.9250 | +0.0513 [+0.0489, +0.0537] | 1.01e-08 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.3 | 4096 | 0.7549 | 0.7993 | +0.0588 [+0.0524, +0.0652] | 2.24e-06 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.3 | 16384 | 0.7592 | 0.8038 | +0.0587 [+0.0542, +0.0633] | 2.72e-07 | > 0 (VCS better) |

### Cells: dA_rel (Holm over cells)

| source | class | type | ε | N | JS | VCS | effect [95 % t] | p Holm | reading |
|---|---|---|---|---|---|---|---|---|---|
| cifar_ap3 | cos2 | clean | 0 | 4096 | 0.9919 | 0.9919 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | clean | 0 | 8192 | 0.9917 | 0.9917 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | false_negative | 0.1 | 4096 | 0.9919 | 0.9919 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | false_negative | 0.1 | 8192 | 0.9917 | 0.9917 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | false_negative | 0.3 | 4096 | 0.9919 | 0.9919 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | false_negative | 0.3 | 8192 | 0.9917 | 0.9917 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | independent | 0.1 | 4096 | 0.9919 | 0.9919 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | independent | 0.1 | 8192 | 0.9917 | 0.9917 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | independent | 0.3 | 4096 | 0.9919 | 0.9919 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | cos2 | independent | 0.3 | 8192 | 0.9917 | 0.9917 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| cifar_ap3 | quad3 | clean | 0 | 4096 | 0.9848 | 0.9849 | +0.0106 [+0.0048, +0.0164] | 0.112 | n.s. |
| cifar_ap3 | quad3 | clean | 0 | 8192 | 0.9844 | 0.9845 | +0.0050 [-0.0004, +0.0104] | 1 | n.s. |
| cifar_ap3 | quad3 | false_negative | 0.1 | 4096 | 0.9791 | 0.9849 | +0.2798 [+0.2605, +0.2991] | 1.86e-07 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.1 | 8192 | 0.9786 | 0.9844 | +0.2719 [+0.2296, +0.3141] | 4.85e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.3 | 4096 | 0.9703 | 0.9817 | +0.3846 [+0.3127, +0.4565] | 0.00016 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.3 | 8192 | 0.9702 | 0.9813 | +0.3733 [+0.3349, +0.4117] | 2.92e-06 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.1 | 4096 | 0.9845 | 0.9849 | +0.0276 [+0.0149, +0.0403] | 0.0449 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.1 | 8192 | 0.9839 | 0.9845 | +0.0351 [+0.0095, +0.0606] | 0.455 | n.s. |
| cifar_ap3 | quad3 | independent | 0.3 | 4096 | 0.9822 | 0.9837 | +0.0854 [+0.0552, +0.1156] | 0.00985 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.3 | 8192 | 0.9816 | 0.9831 | +0.0844 [+0.0703, +0.0985] | 7.74e-05 | > 0 (VCS better) |
| coco_clip | cos2 | clean | 0 | 4096 | 0.9964 | 0.9964 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | clean | 0 | 16384 | 0.9962 | 0.9962 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | false_negative | 0.1 | 4096 | 0.9964 | 0.9964 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | false_negative | 0.1 | 16384 | 0.9962 | 0.9962 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | false_negative | 0.3 | 4096 | 0.9964 | 0.9964 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | false_negative | 0.3 | 16384 | 0.9962 | 0.9962 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | independent | 0.1 | 4096 | 0.9964 | 0.9964 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | independent | 0.1 | 16384 | 0.9962 | 0.9962 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | independent | 0.3 | 4096 | 0.9964 | 0.9964 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | cos2 | independent | 0.3 | 16384 | 0.9962 | 0.9962 | +0.0000 [+0.0000, +0.0000] | 1 | n.s. |
| coco_clip | quad3 | clean | 0 | 4096 | 0.9962 | 0.9961 | -0.0016 [-0.0073, +0.0041] | 1 | n.s. |
| coco_clip | quad3 | clean | 0 | 16384 | 0.9959 | 0.9959 | -0.0006 [-0.0023, +0.0010] | 1 | n.s. |
| coco_clip | quad3 | false_negative | 0.1 | 4096 | 0.9962 | 0.9962 | -0.0050 [-0.0099, -0.0001] | 1 | n.s. |
| coco_clip | quad3 | false_negative | 0.1 | 16384 | 0.9959 | 0.9959 | +0.0006 [-0.0031, +0.0044] | 1 | n.s. |
| coco_clip | quad3 | false_negative | 0.3 | 4096 | 0.9961 | 0.9961 | -0.0009 [-0.0063, +0.0046] | 1 | n.s. |
| coco_clip | quad3 | false_negative | 0.3 | 16384 | 0.9959 | 0.9959 | +0.0024 [-0.0003, +0.0051] | 1 | n.s. |
| coco_clip | quad3 | independent | 0.1 | 4096 | 0.9962 | 0.9962 | +0.0033 [-0.0032, +0.0098] | 1 | n.s. |
| coco_clip | quad3 | independent | 0.1 | 16384 | 0.9959 | 0.9959 | +0.0004 [-0.0012, +0.0020] | 1 | n.s. |
| coco_clip | quad3 | independent | 0.3 | 4096 | 0.9961 | 0.9962 | +0.0050 [-0.0020, +0.0120] | 1 | n.s. |
| coco_clip | quad3 | independent | 0.3 | 16384 | 0.9959 | 0.9959 | +0.0044 [-0.0004, +0.0091] | 1 | n.s. |

### Cells: ECE_JS − ECE_VCS (Holm over cells)

| source | class | type | ε | N | JS | VCS | effect [95 % t] | p Holm | reading |
|---|---|---|---|---|---|---|---|---|---|
| cifar_ap3 | cos2 | clean | 0 | 4096 | 0.0132 | 0.0120 | +0.0012 [-0.0009, +0.0034] | 1 | n.s. |
| cifar_ap3 | cos2 | clean | 0 | 8192 | 0.0141 | 0.0131 | +0.0011 [-0.0003, +0.0025] | 0.714 | n.s. |
| cifar_ap3 | cos2 | false_negative | 0.1 | 4096 | 0.0833 | 0.0288 | +0.0545 [+0.0457, +0.0633] | 3.9e-05 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.1 | 8192 | 0.0812 | 0.0274 | +0.0539 [+0.0511, +0.0566] | 2.07e-08 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.3 | 4096 | 0.1675 | 0.1539 | +0.0136 [+0.0114, +0.0159] | 4.52e-05 | > 0 (VCS better) |
| cifar_ap3 | cos2 | false_negative | 0.3 | 8192 | 0.1688 | 0.1552 | +0.0135 [+0.0120, +0.0151] | 3.72e-06 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.1 | 4096 | 0.0510 | 0.0392 | +0.0118 [+0.0076, +0.0161] | 0.00336 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.1 | 8192 | 0.0486 | 0.0390 | +0.0096 [+0.0075, +0.0116] | 0.000174 | > 0 (VCS better) |
| cifar_ap3 | cos2 | independent | 0.3 | 4096 | 0.1273 | 0.1272 | +0.0001 [-0.0029, +0.0032] | 1 | n.s. |
| cifar_ap3 | cos2 | independent | 0.3 | 8192 | 0.1234 | 0.1244 | -0.0011 [-0.0032, +0.0010] | 1 | n.s. |
| cifar_ap3 | quad3 | clean | 0 | 4096 | 0.0257 | 0.0183 | +0.0074 [+0.0047, +0.0100] | 0.00336 | > 0 (VCS better) |
| cifar_ap3 | quad3 | clean | 0 | 8192 | 0.0230 | 0.0166 | +0.0064 [+0.0045, +0.0083] | 0.00101 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.1 | 4096 | 0.0889 | 0.0246 | +0.0642 [+0.0540, +0.0745] | 3.7e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.1 | 8192 | 0.0842 | 0.0243 | +0.0599 [+0.0561, +0.0637] | 9.56e-08 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.3 | 4096 | 0.1428 | 0.1130 | +0.0298 [+0.0233, +0.0364] | 0.000201 | > 0 (VCS better) |
| cifar_ap3 | quad3 | false_negative | 0.3 | 8192 | 0.1455 | 0.1131 | +0.0324 [+0.0265, +0.0383] | 7.63e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.1 | 4096 | 0.0786 | 0.0298 | +0.0488 [+0.0421, +0.0555] | 1.46e-05 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.1 | 8192 | 0.0779 | 0.0277 | +0.0502 [+0.0446, +0.0558] | 3.72e-06 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.3 | 4096 | 0.1473 | 0.1190 | +0.0283 [+0.0205, +0.0362] | 0.000769 | > 0 (VCS better) |
| cifar_ap3 | quad3 | independent | 0.3 | 8192 | 0.1440 | 0.1178 | +0.0262 [+0.0234, +0.0289] | 2.72e-06 | > 0 (VCS better) |
| coco_clip | cos2 | clean | 0 | 4096 | 0.0058 | 0.0050 | +0.0009 [-0.0013, +0.0031] | 1 | n.s. |
| coco_clip | cos2 | clean | 0 | 16384 | 0.0040 | 0.0035 | +0.0005 [-0.0001, +0.0011] | 0.714 | n.s. |
| coco_clip | cos2 | false_negative | 0.1 | 4096 | 0.0632 | 0.0328 | +0.0304 [+0.0261, +0.0348] | 1.83e-05 | > 0 (VCS better) |
| coco_clip | cos2 | false_negative | 0.1 | 16384 | 0.0606 | 0.0295 | +0.0311 [+0.0291, +0.0330] | 9.41e-08 | > 0 (VCS better) |
| coco_clip | cos2 | false_negative | 0.3 | 4096 | 0.1438 | 0.1353 | +0.0085 [+0.0065, +0.0105] | 0.000314 | > 0 (VCS better) |
| coco_clip | cos2 | false_negative | 0.3 | 16384 | 0.1410 | 0.1328 | +0.0082 [+0.0073, +0.0091] | 2.68e-06 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.1 | 4096 | 0.0736 | 0.0170 | +0.0567 [+0.0505, +0.0628] | 3.35e-06 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.1 | 16384 | 0.0712 | 0.0130 | +0.0582 [+0.0570, +0.0594] | 3.37e-11 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.3 | 4096 | 0.1598 | 0.1459 | +0.0138 [+0.0110, +0.0166] | 0.000129 | > 0 (VCS better) |
| coco_clip | cos2 | independent | 0.3 | 16384 | 0.1574 | 0.1421 | +0.0153 [+0.0140, +0.0165] | 6.03e-07 | > 0 (VCS better) |
| coco_clip | quad3 | clean | 0 | 4096 | 0.0072 | 0.0065 | +0.0006 [-0.0007, +0.0020] | 1 | n.s. |
| coco_clip | quad3 | clean | 0 | 16384 | 0.0046 | 0.0061 | -0.0015 [-0.0032, +0.0003] | 0.714 | n.s. |
| coco_clip | quad3 | false_negative | 0.1 | 4096 | 0.0619 | 0.0343 | +0.0276 [+0.0220, +0.0331] | 0.000129 | > 0 (VCS better) |
| coco_clip | quad3 | false_negative | 0.1 | 16384 | 0.0602 | 0.0298 | +0.0304 [+0.0281, +0.0327] | 3.51e-07 | > 0 (VCS better) |
| coco_clip | quad3 | false_negative | 0.3 | 4096 | 0.1420 | 0.1337 | +0.0083 [+0.0034, +0.0131] | 0.0439 | > 0 (VCS better) |
| coco_clip | quad3 | false_negative | 0.3 | 16384 | 0.1403 | 0.1317 | +0.0086 [+0.0069, +0.0102] | 0.000108 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.1 | 4096 | 0.0736 | 0.0199 | +0.0537 [+0.0485, +0.0589] | 1.62e-06 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.1 | 16384 | 0.0707 | 0.0156 | +0.0552 [+0.0542, +0.0561] | 1.11e-11 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.3 | 4096 | 0.1602 | 0.1448 | +0.0154 [+0.0124, +0.0183] | 0.000107 | > 0 (VCS better) |
| coco_clip | quad3 | independent | 0.3 | 16384 | 0.1572 | 0.1414 | +0.0158 [+0.0133, +0.0183] | 3.27e-05 | > 0 (VCS better) |

### Degradation ratios J(ε) / J(0) per rule (same seed's clean cell) beside the absolute effect (R4-E0 reading: both, never one alone)

| source | class | type | ε | N | ratio JS | ratio VCS | ratio VCS − JS [95 % t] | dJ_rel [95 % t] |
|---|---|---|---|---|---|---|---|---|
| cifar_ap3 | cos2 | false_negative | 0.1 | 4096 | 0.9404 | 0.9935 | +0.0531 [+0.0495, +0.0566] | +0.0579 [+0.0542, +0.0616] |
| cifar_ap3 | cos2 | false_negative | 0.1 | 8192 | 0.9451 | 0.9947 | +0.0496 [+0.0463, +0.0529] | +0.0543 [+0.0506, +0.0579] |
| cifar_ap3 | cos2 | false_negative | 0.3 | 4096 | 0.8124 | 0.8590 | +0.0466 [+0.0420, +0.0512] | +0.0588 [+0.0529, +0.0646] |
| cifar_ap3 | cos2 | false_negative | 0.3 | 8192 | 0.8132 | 0.8598 | +0.0466 [+0.0439, +0.0493] | +0.0591 [+0.0555, +0.0627] |
| cifar_ap3 | cos2 | independent | 0.1 | 4096 | 0.9808 | 0.9881 | +0.0073 [+0.0036, +0.0110] | +0.0088 [+0.0059, +0.0117] |
| cifar_ap3 | cos2 | independent | 0.1 | 8192 | 0.9813 | 0.9875 | +0.0062 [+0.0044, +0.0080] | +0.0080 [+0.0065, +0.0094] |
| cifar_ap3 | cos2 | independent | 0.3 | 4096 | 0.8804 | 0.8893 | +0.0089 [+0.0040, +0.0139] | +0.0115 [+0.0064, +0.0166] |
| cifar_ap3 | cos2 | independent | 0.3 | 8192 | 0.8820 | 0.8900 | +0.0080 [+0.0041, +0.0119] | +0.0108 [+0.0068, +0.0148] |
| cifar_ap3 | quad3 | false_negative | 0.1 | 4096 | 0.8928 | 0.9946 | +0.1018 [+0.0900, +0.1137] | +0.1220 [+0.1080, +0.1360] |
| cifar_ap3 | quad3 | false_negative | 0.1 | 8192 | 0.8977 | 0.9935 | +0.0958 [+0.0889, +0.1027] | +0.1133 [+0.1049, +0.1218] |
| cifar_ap3 | quad3 | false_negative | 0.3 | 4096 | 0.7461 | 0.8633 | +0.1172 [+0.0978, +0.1365] | +0.1652 [+0.1398, +0.1906] |
| cifar_ap3 | quad3 | false_negative | 0.3 | 8192 | 0.7399 | 0.8630 | +0.1231 [+0.1127, +0.1335] | +0.1734 [+0.1590, +0.1877] |
| cifar_ap3 | quad3 | independent | 0.1 | 4096 | 0.9440 | 0.9909 | +0.0469 [+0.0399, +0.0538] | +0.0570 [+0.0490, +0.0650] |
| cifar_ap3 | quad3 | independent | 0.1 | 8192 | 0.9429 | 0.9915 | +0.0487 [+0.0447, +0.0526] | +0.0579 [+0.0538, +0.0620] |
| cifar_ap3 | quad3 | independent | 0.3 | 4096 | 0.8127 | 0.8678 | +0.0551 [+0.0432, +0.0670] | +0.0752 [+0.0592, +0.0912] |
| cifar_ap3 | quad3 | independent | 0.3 | 8192 | 0.8166 | 0.8684 | +0.0517 [+0.0461, +0.0574] | +0.0697 [+0.0623, +0.0771] |
| coco_clip | cos2 | false_negative | 0.1 | 16384 | 0.9630 | 0.9890 | +0.0260 [+0.0249, +0.0272] | +0.0272 [+0.0260, +0.0284] |
| coco_clip | cos2 | false_negative | 0.1 | 4096 | 0.9610 | 0.9877 | +0.0266 [+0.0242, +0.0291] | +0.0280 [+0.0252, +0.0308] |
| coco_clip | cos2 | false_negative | 0.3 | 16384 | 0.8479 | 0.8787 | +0.0308 [+0.0292, +0.0324] | +0.0365 [+0.0346, +0.0383] |
| coco_clip | cos2 | false_negative | 0.3 | 4096 | 0.8431 | 0.8751 | +0.0321 [+0.0286, +0.0355] | +0.0383 [+0.0342, +0.0425] |
| coco_clip | cos2 | independent | 0.1 | 16384 | 0.9461 | 0.9963 | +0.0502 [+0.0481, +0.0523] | +0.0532 [+0.0511, +0.0554] |
| coco_clip | cos2 | independent | 0.1 | 4096 | 0.9458 | 0.9950 | +0.0492 [+0.0462, +0.0522] | +0.0524 [+0.0494, +0.0553] |
| coco_clip | cos2 | independent | 0.3 | 16384 | 0.8178 | 0.8643 | +0.0465 [+0.0448, +0.0482] | +0.0570 [+0.0549, +0.0592] |
| coco_clip | cos2 | independent | 0.3 | 4096 | 0.8146 | 0.8592 | +0.0446 [+0.0400, +0.0491] | +0.0550 [+0.0496, +0.0605] |
| coco_clip | quad3 | false_negative | 0.1 | 16384 | 0.9631 | 0.9894 | +0.0263 [+0.0243, +0.0283] | +0.0268 [+0.0248, +0.0288] |
| coco_clip | quad3 | false_negative | 0.1 | 4096 | 0.9622 | 0.9865 | +0.0243 [+0.0204, +0.0283] | +0.0255 [+0.0214, +0.0296] |
| coco_clip | quad3 | false_negative | 0.3 | 16384 | 0.8487 | 0.8803 | +0.0317 [+0.0279, +0.0354] | +0.0368 [+0.0323, +0.0413] |
| coco_clip | quad3 | false_negative | 0.3 | 4096 | 0.8460 | 0.8781 | +0.0320 [+0.0233, +0.0408] | +0.0381 [+0.0278, +0.0483] |
| coco_clip | quad3 | independent | 0.1 | 16384 | 0.9469 | 0.9960 | +0.0491 [+0.0469, +0.0512] | +0.0513 [+0.0489, +0.0537] |
| coco_clip | quad3 | independent | 0.1 | 4096 | 0.9451 | 0.9937 | +0.0486 [+0.0468, +0.0504] | +0.0516 [+0.0496, +0.0536] |
| coco_clip | quad3 | independent | 0.3 | 16384 | 0.8170 | 0.8655 | +0.0484 [+0.0446, +0.0523] | +0.0587 [+0.0542, +0.0633] |
| coco_clip | quad3 | independent | 0.3 | 4096 | 0.8127 | 0.8603 | +0.0476 [+0.0422, +0.0530] | +0.0588 [+0.0524, +0.0652] |

### dJ_rel at ε > 0 by factor (descriptive)

- N=16384: +0.0435
- N=4096: +0.0565
- N=8192: +0.0683
- cls=cos2: +0.0386
- cls=quad3: +0.0738
- ctype=false_negative: +0.0663
- ctype=independent: +0.0461
- eps=0.1: +0.0497
- eps=0.3: +0.0627
- source=cifar_ap3: +0.0689
- source=coco_clip: +0.0435

### QC (selected rows): top-lr (1.0) / low-lr / not converged

- cifar_ap3:cos2:js: {'n': 80, 'top_lr': 10, 'low_lr': 0, 'not_converged': 3}
- cifar_ap3:cos2:vcs: {'n': 80, 'top_lr': 10, 'low_lr': 0, 'not_converged': 9}
- cifar_ap3:quad3:js: {'n': 80, 'top_lr': 65, 'low_lr': 0, 'not_converged': 15}
- cifar_ap3:quad3:vcs: {'n': 80, 'top_lr': 65, 'low_lr': 0, 'not_converged': 18}
- coco_clip:cos2:js: {'n': 80, 'top_lr': 27, 'low_lr': 0, 'not_converged': 4}
- coco_clip:cos2:vcs: {'n': 80, 'top_lr': 26, 'low_lr': 0, 'not_converged': 6}
- coco_clip:quad3:js: {'n': 80, 'top_lr': 60, 'low_lr': 0, 'not_converged': 13}
- coco_clip:quad3:vcs: {'n': 80, 'top_lr': 51, 'low_lr': 0, 'not_converged': 11}

## Selection: selected_commonJ — outcome: **FIT-ONLY** (40 cells; missing 0)

| test | mean [95 % t] | p | p Holm | cells | complete | reading |
|---|---|---|---|---|---|---|
| Q1_dJ_eps_gt_0 | +0.0281 [+0.0262, +0.0299] | 3.21e-09 | 7.18e-09 | 32 | True | > 0 (VCS better) |
| Q2_dJ_eps_0 | +0.0014 [+0.0013, +0.0015] | 1.83e-08 | 1.83e-08 | 8 | True | > 0 (VCS better) |
| Q3_dA_eps_gt_0 | +0.0425 [+0.0398, +0.0451] | 2.39e-09 | 7.18e-09 | 32 | True | > 0 (VCS better) |

