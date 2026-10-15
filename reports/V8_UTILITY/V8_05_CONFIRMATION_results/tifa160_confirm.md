# V8-05 §B — TIFA160 secondary external evaluation (generated from tifa160_confirm.json, 2026-10-15T19:33:24Z)

Secondary evaluation added 2026-10-15 (not the registered confirmation source).  Images scored by every method: 800.  Bootstrap: 5000 resamples over the 160 text inputs, seed 2718.  Comparators present: ['B0_FGA_released_O', 'B1_VQAScore', 'B4T_TIFA']; absent: ['B4D_DSG']; exposed: ['B4D_DSG'].  Intervals only — no claim language.

| label source | images with a rating | ratings per image (count of images) | parse record |
|---|---|---|---|
| TIFA 2-rater mean (human_annotations.json) | 800 | {2: 800} | layout dict; located by {'key:image': 800}; rating fields {'tier1:list:human_scores.[*]': 800}; unparsed 0 |
| DSG 5-rater mean (tifa160-likert-anns.csv) | 800 | {3: 1, 4: 3, 5: 796} | raters per image {3: 1, 4: 3, 5: 796}; row counts {'rows': 3995, 'rows_used': 3995, 'text_id_from_source_id': 3995} |

## labels = tifa_2rater

Images 800 over 160 text inputs (dropped without a rating: 0).

### tifa_primary

| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |
|---|---|---|---|---|---|---|
| U-CS − B0_FGA_released_O | spearman | +0.018 [-0.002, +0.039] | 0.0888 | 0.0888 | 0.767 / 0.749 |  |
| U-CS − B0_FGA_released_O | pearson | +0.022 [+0.004, +0.042] | 0.0224 | 0.0224 | 0.735 / 0.713 |  |
| U-CS − B0_FGA_released_O | kendall_tau_b | +0.017 [-0.003, +0.037] | 0.1036 | 0.1036 | 0.600 / 0.583 |  |
| U-CS − B1_VQAScore | spearman | +0.071 [+0.018, +0.124] | 0.0076 | 0.0152 | 0.767 / 0.696 |  |
| U-CS − B1_VQAScore | pearson | +0.077 [+0.023, +0.133] | 0.0048 | 0.0096 | 0.735 / 0.658 |  |
| U-CS − B1_VQAScore | kendall_tau_b | +0.068 [+0.020, +0.115] | 0.0040 | 0.0080 | 0.600 / 0.532 |  |
| U-CS − B4T_TIFA | spearman | +0.241 [+0.165, +0.323] | 0.0002 | 0.0006 | 0.767 / 0.525 |  |
| U-CS − B4T_TIFA | pearson | +0.227 [+0.139, +0.333] | 0.0002 | 0.0006 | 0.735 / 0.508 |  |
| U-CS − B4T_TIFA | kendall_tau_b | +0.179 [+0.115, +0.248] | 0.0002 | 0.0006 | 0.600 / 0.421 |  |

### tifa_attribution

| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |
|---|---|---|---|---|---|---|
| U-CS − U-BCE | spearman | -0.009 [-0.017, -0.001] | 0.0268 | 0.0268 | 0.767 / 0.776 |  |
| U-CS − U-BCE | pearson | -0.002 [-0.010, +0.006] | 0.6124 | 0.6124 | 0.735 / 0.737 |  |
| U-CS − U-BCE | kendall_tau_b | -0.009 [-0.017, -0.001] | 0.0268 | 0.0268 | 0.600 / 0.609 |  |

### tifa_same_data_descriptive

| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |
|---|---|---|---|---|---|---|
| U-CS − U-RPS | spearman | -0.007 [-0.017, +0.003] | 0.1916 | nan | 0.767 / 0.774 |  |
| U-CS − U-RPS | pearson | +0.001 [-0.010, +0.011] | 0.8840 | nan | 0.735 / 0.734 |  |
| U-CS − U-RPS | kendall_tau_b | -0.006 [-0.016, +0.004] | 0.2668 | nan | 0.600 / 0.606 |  |
| U-CS − U-CE | spearman | -0.015 [-0.026, -0.004] | 0.0040 | nan | 0.767 / 0.782 |  |
| U-CS − U-CE | pearson | -0.002 [-0.013, +0.009] | 0.7232 | nan | 0.735 / 0.737 |  |
| U-CS − U-CE | kendall_tau_b | -0.016 [-0.027, -0.004] | 0.0072 | nan | 0.600 / 0.616 |  |
| U-CS − U-MSE | spearman | -0.006 [-0.016, +0.004] | 0.2764 | nan | 0.767 / 0.773 |  |
| U-CS − U-MSE | pearson | -0.005 [-0.015, +0.005] | 0.3300 | nan | 0.735 / 0.740 |  |
| U-CS − U-MSE | kendall_tau_b | -0.005 [-0.014, +0.005] | 0.3816 | nan | 0.600 / 0.605 |  |
| U-CS − U-L1 | spearman | +0.007 [-0.001, +0.016] | 0.0896 | nan | 0.767 / 0.760 |  |
| U-CS − U-L1 | pearson | +0.010 [+0.002, +0.019] | 0.0172 | nan | 0.735 / 0.725 |  |
| U-CS − U-L1 | kendall_tau_b | +0.007 [-0.001, +0.015] | 0.0940 | nan | 0.600 / 0.593 |  |
| U-CS − U-RANK | spearman | +0.014 [-0.006, +0.033] | 0.1704 | nan | 0.767 / 0.753 |  |
| U-CS − U-RANK | pearson | +0.008 [-0.008, +0.025] | 0.3152 | nan | 0.735 / 0.727 |  |
| U-CS − U-RANK | kendall_tau_b | +0.016 [-0.004, +0.035] | 0.1008 | nan | 0.600 / 0.584 |  |
| U-CS − U-LIST | spearman | +0.004 [-0.012, +0.020] | 0.6308 | nan | 0.767 / 0.763 |  |
| U-CS − U-LIST | pearson | +0.008 [-0.007, +0.023] | 0.2844 | nan | 0.735 / 0.727 |  |
| U-CS − U-LIST | kendall_tau_b | +0.006 [-0.009, +0.022] | 0.4116 | nan | 0.600 / 0.594 |  |

### Levels (Spearman / Pearson / Kendall tau-b; Spearman per generator)

| method | Spearman | Pearson | Kendall | mini_dalle | stable_diffusion_v1_1 | stable_diffusion_v1_5 | stable_diffusion_v2_1 | vq_diffusion |
|---|---|---|---|---|---|---|---|---|
| U-CS | 0.767 | 0.735 | 0.600 | 0.766 | 0.755 | 0.753 | 0.692 | 0.722 |
| U-BCE | 0.776 | 0.737 | 0.609 | 0.786 | 0.759 | 0.759 | 0.699 | 0.729 |
| U-RPS | 0.774 | 0.734 | 0.606 | 0.777 | 0.745 | 0.752 | 0.704 | 0.727 |
| U-CE | 0.782 | 0.737 | 0.616 | 0.801 | 0.747 | 0.761 | 0.715 | 0.734 |
| U-MSE | 0.773 | 0.740 | 0.605 | 0.772 | 0.750 | 0.751 | 0.698 | 0.732 |
| U-L1 | 0.760 | 0.725 | 0.593 | 0.761 | 0.740 | 0.741 | 0.679 | 0.715 |
| U-RANK | 0.753 | 0.727 | 0.584 | 0.770 | 0.724 | 0.724 | 0.675 | 0.728 |
| U-LIST | 0.763 | 0.727 | 0.594 | 0.767 | 0.728 | 0.747 | 0.683 | 0.735 |
| B0_FGA_released_O | 0.749 | 0.713 | 0.583 | 0.735 | 0.718 | 0.771 | 0.672 | 0.683 |
| B1_VQAScore | 0.696 | 0.658 | 0.532 | 0.707 | 0.652 | 0.731 | 0.688 | 0.602 |
| B4T_TIFA | 0.525 | 0.508 | 0.421 | 0.560 | 0.487 | 0.587 | 0.532 | 0.437 |

## labels = dsg_5rater

Images 800 over 160 text inputs (dropped without a rating: 0).

### tifa_primary

| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |
|---|---|---|---|---|---|---|
| U-CS − B0_FGA_released_O | spearman | +0.008 [-0.014, +0.029] | 0.4816 | 0.4816 | 0.773 / 0.765 |  |
| U-CS − B0_FGA_released_O | pearson | +0.017 [-0.002, +0.037] | 0.0800 | 0.0800 | 0.743 / 0.726 |  |
| U-CS − B0_FGA_released_O | kendall_tau_b | +0.006 [-0.014, +0.027] | 0.5484 | 0.5484 | 0.589 / 0.583 |  |
| U-CS − B1_VQAScore | spearman | +0.068 [+0.016, +0.122] | 0.0084 | 0.0168 | 0.773 / 0.704 |  |
| U-CS − B1_VQAScore | pearson | +0.078 [+0.028, +0.129] | 0.0020 | 0.0040 | 0.743 / 0.665 |  |
| U-CS − B1_VQAScore | kendall_tau_b | +0.062 [+0.014, +0.111] | 0.0124 | 0.0248 | 0.589 / 0.527 |  |
| U-CS − B4T_TIFA | spearman | +0.242 [+0.163, +0.330] | 0.0002 | 0.0006 | 0.773 / 0.530 |  |
| U-CS − B4T_TIFA | pearson | +0.260 [+0.167, +0.373] | 0.0002 | 0.0006 | 0.743 / 0.483 |  |
| U-CS − B4T_TIFA | kendall_tau_b | +0.181 [+0.115, +0.253] | 0.0002 | 0.0006 | 0.589 / 0.408 |  |

### tifa_attribution

| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |
|---|---|---|---|---|---|---|
| U-CS − U-BCE | spearman | -0.013 [-0.025, -0.004] | 0.0040 | 0.0040 | 0.773 / 0.786 |  |
| U-CS − U-BCE | pearson | -0.006 [-0.016, +0.004] | 0.2428 | 0.2428 | 0.743 / 0.749 |  |
| U-CS − U-BCE | kendall_tau_b | -0.012 [-0.022, -0.003] | 0.0060 | 0.0060 | 0.589 / 0.601 |  |

### tifa_same_data_descriptive

| contrast | correlation | Δ [95 %] | p | p Holm | ours / comp | exposed |
|---|---|---|---|---|---|---|
| U-CS − U-RPS | spearman | -0.006 [-0.015, +0.004] | 0.2452 | nan | 0.773 / 0.778 |  |
| U-CS − U-RPS | pearson | +0.001 [-0.008, +0.010] | 0.7936 | nan | 0.743 / 0.742 |  |
| U-CS − U-RPS | kendall_tau_b | -0.004 [-0.013, +0.005] | 0.4124 | nan | 0.589 / 0.593 |  |
| U-CS − U-CE | spearman | -0.017 [-0.028, -0.006] | 0.0016 | nan | 0.773 / 0.790 |  |
| U-CS − U-CE | pearson | -0.002 [-0.014, +0.008] | 0.6864 | nan | 0.743 / 0.745 |  |
| U-CS − U-CE | kendall_tau_b | -0.017 [-0.028, -0.006] | 0.0036 | nan | 0.589 / 0.606 |  |
| U-CS − U-MSE | spearman | -0.006 [-0.015, +0.002] | 0.1496 | nan | 0.773 / 0.779 |  |
| U-CS − U-MSE | pearson | -0.005 [-0.013, +0.003] | 0.2492 | nan | 0.743 / 0.748 |  |
| U-CS − U-MSE | kendall_tau_b | -0.004 [-0.012, +0.004] | 0.3720 | nan | 0.589 / 0.593 |  |
| U-CS − U-L1 | spearman | +0.001 [-0.010, +0.011] | 0.7932 | nan | 0.773 / 0.771 |  |
| U-CS − U-L1 | pearson | +0.003 [-0.007, +0.013] | 0.5124 | nan | 0.743 / 0.740 |  |
| U-CS − U-L1 | kendall_tau_b | +0.003 [-0.007, +0.013] | 0.4852 | nan | 0.589 / 0.586 |  |
| U-CS − U-RANK | spearman | +0.028 [+0.007, +0.049] | 0.0076 | nan | 0.773 / 0.744 |  |
| U-CS − U-RANK | pearson | +0.029 [+0.011, +0.047] | 0.0016 | nan | 0.743 / 0.714 |  |
| U-CS − U-RANK | kendall_tau_b | +0.028 [+0.009, +0.047] | 0.0060 | nan | 0.589 / 0.561 |  |
| U-CS − U-LIST | spearman | +0.005 [-0.011, +0.021] | 0.5232 | nan | 0.773 / 0.767 |  |
| U-CS − U-LIST | pearson | +0.012 [-0.003, +0.027] | 0.1212 | nan | 0.743 / 0.731 |  |
| U-CS − U-LIST | kendall_tau_b | +0.006 [-0.009, +0.021] | 0.4228 | nan | 0.589 / 0.583 |  |

### Levels (Spearman / Pearson / Kendall tau-b; Spearman per generator)

| method | Spearman | Pearson | Kendall | mini_dalle | stable_diffusion_v1_1 | stable_diffusion_v1_5 | stable_diffusion_v2_1 | vq_diffusion |
|---|---|---|---|---|---|---|---|---|
| U-CS | 0.773 | 0.743 | 0.589 | 0.756 | 0.744 | 0.760 | 0.719 | 0.784 |
| U-BCE | 0.786 | 0.749 | 0.601 | 0.776 | 0.754 | 0.780 | 0.728 | 0.792 |
| U-RPS | 0.778 | 0.742 | 0.593 | 0.766 | 0.744 | 0.757 | 0.727 | 0.780 |
| U-CE | 0.790 | 0.745 | 0.606 | 0.788 | 0.748 | 0.774 | 0.738 | 0.796 |
| U-MSE | 0.779 | 0.748 | 0.593 | 0.769 | 0.752 | 0.755 | 0.728 | 0.781 |
| U-L1 | 0.771 | 0.740 | 0.586 | 0.750 | 0.745 | 0.766 | 0.720 | 0.769 |
| U-RANK | 0.744 | 0.714 | 0.561 | 0.762 | 0.720 | 0.703 | 0.662 | 0.778 |
| U-LIST | 0.767 | 0.731 | 0.583 | 0.772 | 0.740 | 0.748 | 0.694 | 0.770 |
| B0_FGA_released_O | 0.765 | 0.726 | 0.583 | 0.779 | 0.734 | 0.777 | 0.677 | 0.759 |
| B1_VQAScore | 0.704 | 0.665 | 0.527 | 0.704 | 0.686 | 0.746 | 0.673 | 0.611 |
| B4T_TIFA | 0.530 | 0.483 | 0.408 | 0.505 | 0.488 | 0.608 | 0.535 | 0.474 |

