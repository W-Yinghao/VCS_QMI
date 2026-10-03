# P118 — nested critics (v5 NEXT-I-NEST): results

## 6.1 Truth compression (mean over repeats; increment errors vs oracle; posterior MSE vs exact eta)

| cond | true M→A | obj | variant | readout | err B→M | err M→A (RMSE) | err B→A | post B | post M | post A | R_orth | violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null | 0.0000 | vcs | indep_sampled | sampled | -0.0002 | +0.0009 (0.0023) | +0.0007 | 0.0012 | 0.0018 | 0.0021 | +0.0023 | 1.0 |
| null | 0.0000 | vcs | indep_sampled | exact | +0.0004 | +0.0008 (0.0026) | +0.0011 | 0.0012 | 0.0018 | 0.0021 | +0.0023 | 1.0 |
| null | 0.0000 | vcs | indep_exact | sampled | +0.0000 | +0.0009 (0.0028) | +0.0009 | 0.0010 | 0.0010 | 0.0012 | +0.0016 | 0.8 |
| null | 0.0000 | vcs | indep_exact | exact | +0.0000 | +0.0008 (0.0025) | +0.0008 | 0.0009 | 0.0010 | 0.0012 | +0.0016 | 0.8 |
| null | 0.0000 | vcs | nested_sampled | sampled | +0.0000 | +0.0006 (0.0014) | +0.0007 | 0.0021 | 0.0021 | 0.0021 | +0.0001 | 0.0 |
| null | 0.0000 | vcs | nested_sampled | exact | -0.0000 | +0.0004 (0.0012) | +0.0004 | 0.0021 | 0.0021 | 0.0021 | +0.0001 | 0.4 |
| null | 0.0000 | vcs | nested_exact | sampled | +0.0001 | +0.0007 (0.0014) | +0.0008 | 0.0007 | 0.0008 | 0.0012 | +0.0000 | 0.0 |
| null | 0.0000 | vcs | nested_exact | exact | +0.0001 | +0.0007 (0.0015) | +0.0008 | 0.0007 | 0.0008 | 0.0012 | +0.0000 | 0.0 |
| null | 0.0000 | vcs | zero | sampled | +0.0000 | +0.0000 (0.0000) | +0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.0 |
| null | 0.0000 | vcs | zero | exact | +0.0000 | +0.0000 (0.0000) | +0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.0 |
| null | 0.0000 | js | indep_sampled | sampled | +0.0001 | +0.0002 (0.0022) | +0.0003 | 0.0011 | 0.0019 | 0.0015 | +0.0027 | 1.0 |
| null | 0.0000 | js | indep_sampled | exact | +0.0006 | -0.0004 (0.0022) | +0.0002 | 0.0011 | 0.0019 | 0.0015 | +0.0027 | 1.2 |
| null | 0.0000 | js | indep_exact | sampled | -0.0006 | +0.0006 (0.0008) | -0.0000 | 0.0010 | 0.0007 | 0.0009 | +0.0009 | 1.0 |
| null | 0.0000 | js | indep_exact | exact | -0.0005 | +0.0004 (0.0006) | -0.0000 | 0.0009 | 0.0007 | 0.0009 | +0.0009 | 1.0 |
| null | 0.0000 | js | nested_sampled | sampled | +0.0000 | +0.0000 (0.0001) | +0.0000 | 0.0025 | 0.0025 | 0.0015 | +0.0000 | 0.2 |
| null | 0.0000 | js | nested_sampled | exact | +0.0000 | -0.0007 (0.0014) | -0.0007 | 0.0025 | 0.0025 | 0.0015 | +0.0000 | 0.4 |
| null | 0.0000 | js | nested_exact | sampled | +0.0001 | +0.0001 (0.0001) | +0.0001 | 0.0007 | 0.0008 | 0.0009 | +0.0000 | 0.0 |
| null | 0.0000 | js | nested_exact | exact | +0.0001 | +0.0001 (0.0001) | +0.0001 | 0.0007 | 0.0008 | 0.0009 | +0.0000 | 0.0 |
| null | 0.0000 | js | zero | sampled | +0.0000 | +0.0000 (0.0000) | +0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.0 |
| null | 0.0000 | js | zero | exact | +0.0000 | +0.0000 (0.0000) | +0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.0 |
| weak | 0.0169 | vcs | indep_sampled | sampled | -0.0086 | -0.0080 (0.0085) | -0.0165 | 0.0255 | 0.0181 | 0.0104 | +0.0302 | 1.0 |
| weak | 0.0169 | vcs | indep_sampled | exact | -0.0069 | -0.0090 (0.0092) | -0.0160 | 0.0254 | 0.0183 | 0.0104 | +0.0301 | 1.0 |
| weak | 0.0169 | vcs | indep_exact | sampled | -0.0100 | -0.0041 (0.0055) | -0.0141 | 0.0198 | 0.0114 | 0.0067 | +0.0265 | 1.0 |
| weak | 0.0169 | vcs | indep_exact | exact | -0.0082 | -0.0053 (0.0059) | -0.0136 | 0.0196 | 0.0114 | 0.0067 | +0.0267 | 1.0 |
| weak | 0.0169 | vcs | nested_sampled | sampled | +0.0005 | -0.0071 (0.0077) | -0.0066 | 0.0161 | 0.0167 | 0.0104 | +0.0013 | 0.0 |
| weak | 0.0169 | vcs | nested_sampled | exact | +0.0006 | -0.0073 (0.0076) | -0.0067 | 0.0162 | 0.0168 | 0.0104 | +0.0013 | 0.0 |
| weak | 0.0169 | vcs | nested_exact | sampled | -0.0001 | -0.0050 (0.0063) | -0.0051 | 0.0115 | 0.0112 | 0.0067 | +0.0004 | 0.2 |
| weak | 0.0169 | vcs | nested_exact | exact | -0.0001 | -0.0051 (0.0064) | -0.0051 | 0.0115 | 0.0112 | 0.0067 | +0.0004 | 0.2 |
| weak | 0.0169 | vcs | zero | sampled | +0.0000 | -0.0169 (0.0169) | -0.0169 | 0.0359 | 0.0359 | 0.0192 | +0.0000 | 0.0 |
| weak | 0.0169 | vcs | zero | exact | +0.0000 | -0.0169 (0.0169) | -0.0169 | 0.0355 | 0.0355 | 0.0188 | +0.0000 | 0.0 |
| weak | 0.0169 | js | indep_sampled | sampled | -0.0084 | -0.0086 (0.0092) | -0.0170 | 0.0260 | 0.0193 | 0.0108 | +0.0352 | 1.0 |
| weak | 0.0169 | js | indep_sampled | exact | -0.0062 | -0.0099 (0.0101) | -0.0162 | 0.0259 | 0.0195 | 0.0108 | +0.0350 | 1.0 |
| weak | 0.0169 | js | indep_exact | sampled | -0.0094 | -0.0045 (0.0060) | -0.0139 | 0.0202 | 0.0120 | 0.0072 | +0.0255 | 1.0 |
| weak | 0.0169 | js | indep_exact | exact | -0.0077 | -0.0054 (0.0060) | -0.0131 | 0.0201 | 0.0120 | 0.0072 | +0.0256 | 1.0 |
| weak | 0.0169 | js | nested_sampled | sampled | +0.0006 | -0.0072 (0.0078) | -0.0065 | 0.0168 | 0.0175 | 0.0108 | +0.0014 | 0.0 |
| weak | 0.0169 | js | nested_sampled | exact | +0.0007 | -0.0078 (0.0080) | -0.0071 | 0.0169 | 0.0176 | 0.0108 | +0.0014 | 0.0 |
| weak | 0.0169 | js | nested_exact | sampled | +0.0004 | -0.0050 (0.0065) | -0.0045 | 0.0115 | 0.0121 | 0.0072 | +0.0015 | 0.2 |
| weak | 0.0169 | js | nested_exact | exact | +0.0005 | -0.0054 (0.0069) | -0.0048 | 0.0116 | 0.0121 | 0.0072 | +0.0015 | 0.0 |
| weak | 0.0169 | js | zero | sampled | +0.0000 | -0.0169 (0.0169) | -0.0169 | 0.0359 | 0.0359 | 0.0192 | +0.0000 | 0.0 |
| weak | 0.0169 | js | zero | exact | +0.0000 | -0.0169 (0.0169) | -0.0169 | 0.0355 | 0.0355 | 0.0188 | +0.0000 | 0.0 |
| moderate | 0.0441 | vcs | indep_sampled | sampled | -0.0204 | -0.0103 (0.0105) | -0.0306 | 0.0433 | 0.0224 | 0.0123 | +0.0421 | 1.0 |
| moderate | 0.0441 | vcs | indep_sampled | exact | -0.0198 | -0.0109 (0.0114) | -0.0307 | 0.0431 | 0.0223 | 0.0123 | +0.0425 | 1.0 |
| moderate | 0.0441 | vcs | indep_exact | sampled | -0.0181 | -0.0065 (0.0067) | -0.0246 | 0.0322 | 0.0143 | 0.0073 | +0.0283 | 1.0 |
| moderate | 0.0441 | vcs | indep_exact | exact | -0.0177 | -0.0081 (0.0086) | -0.0259 | 0.0320 | 0.0142 | 0.0073 | +0.0286 | 1.0 |
| moderate | 0.0441 | vcs | nested_sampled | sampled | +0.0009 | -0.0170 (0.0182) | -0.0161 | 0.0287 | 0.0293 | 0.0123 | +0.0012 | 0.0 |
| moderate | 0.0441 | vcs | nested_sampled | exact | +0.0007 | -0.0174 (0.0182) | -0.0167 | 0.0288 | 0.0295 | 0.0123 | +0.0011 | 0.0 |
| moderate | 0.0441 | vcs | nested_exact | sampled | -0.0001 | -0.0069 (0.0070) | -0.0069 | 0.0157 | 0.0157 | 0.0073 | +0.0002 | 0.2 |
| moderate | 0.0441 | vcs | nested_exact | exact | -0.0000 | -0.0083 (0.0087) | -0.0084 | 0.0156 | 0.0155 | 0.0073 | +0.0002 | 0.2 |
| moderate | 0.0441 | vcs | zero | sampled | +0.0000 | -0.0441 (0.0441) | -0.0441 | 0.1118 | 0.1118 | 0.0665 | +0.0000 | 0.0 |
| moderate | 0.0441 | vcs | zero | exact | +0.0000 | -0.0441 (0.0441) | -0.0441 | 0.1107 | 0.1107 | 0.0657 | +0.0000 | 0.0 |
| moderate | 0.0441 | js | indep_sampled | sampled | -0.0192 | -0.0109 (0.0109) | -0.0301 | 0.0442 | 0.0240 | 0.0132 | +0.0508 | 1.0 |
| moderate | 0.0441 | js | indep_sampled | exact | -0.0188 | -0.0116 (0.0120) | -0.0304 | 0.0439 | 0.0239 | 0.0132 | +0.0512 | 1.0 |
| moderate | 0.0441 | js | indep_exact | sampled | -0.0175 | -0.0080 (0.0082) | -0.0255 | 0.0336 | 0.0160 | 0.0078 | +0.0376 | 1.0 |
| moderate | 0.0441 | js | indep_exact | exact | -0.0167 | -0.0094 (0.0097) | -0.0261 | 0.0334 | 0.0158 | 0.0079 | +0.0377 | 1.0 |
| moderate | 0.0441 | js | nested_sampled | sampled | +0.0004 | -0.0169 (0.0179) | -0.0165 | 0.0300 | 0.0304 | 0.0132 | +0.0026 | 0.2 |
| moderate | 0.0441 | js | nested_sampled | exact | +0.0006 | -0.0173 (0.0181) | -0.0167 | 0.0301 | 0.0306 | 0.0132 | +0.0026 | 0.2 |
| moderate | 0.0441 | js | nested_exact | sampled | +0.0000 | -0.0085 (0.0087) | -0.0085 | 0.0171 | 0.0171 | 0.0078 | +0.0000 | 0.0 |
| moderate | 0.0441 | js | nested_exact | exact | +0.0000 | -0.0094 (0.0098) | -0.0094 | 0.0170 | 0.0170 | 0.0079 | +0.0000 | 0.0 |
| moderate | 0.0441 | js | zero | sampled | +0.0000 | -0.0441 (0.0441) | -0.0441 | 0.1118 | 0.1118 | 0.0665 | +0.0000 | 0.0 |
| moderate | 0.0441 | js | zero | exact | +0.0000 | -0.0441 (0.0441) | -0.0441 | 0.1107 | 0.1107 | 0.0657 | +0.0000 | 0.0 |

### Frozen rule vs indep_sampled / sampled readout (both non-null conditions)

| obj | variant | readout | verdict | weak (c1 c2 c3 c4) | moderate |
|---|---|---|---|---|---|
| vcs | indep_sampled | exact | shrinkage | ✗ ✓ ✓ ✗ | ✗ ✓ ✓ ✓ |
| vcs | indep_exact | sampled | no clear reduction | ✓ ✗ ✓ ✓ | ✓ ✓ ✓ ✓ |
| vcs | indep_exact | exact | no clear reduction | ✓ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| vcs | nested_sampled | sampled | no clear reduction | ✗ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| vcs | nested_sampled | exact | no clear reduction | ✗ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| vcs | nested_exact | sampled | error reduced | ✓ ✓ ✓ ✓ | ✓ ✓ ✓ ✓ |
| vcs | nested_exact | exact | no clear reduction | ✗ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| vcs | zero | sampled | shrinkage | ✗ ✓ ✗ ✗ | ✗ ✓ ✗ ✗ |
| vcs | zero | exact | shrinkage | ✗ ✓ ✗ ✗ | ✗ ✓ ✗ ✗ |
| js | indep_sampled | exact | shrinkage | ✗ ✓ ✓ ✗ | ✗ ✓ ✓ ✓ |
| js | indep_exact | sampled | no clear reduction | ✓ ✗ ✓ ✓ | ✓ ✓ ✓ ✓ |
| js | indep_exact | exact | no clear reduction | ✓ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| js | nested_sampled | sampled | no clear reduction | ✗ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| js | nested_sampled | exact | no clear reduction | ✗ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| js | nested_exact | sampled | no clear reduction | ✓ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| js | nested_exact | exact | no clear reduction | ✓ ✓ ✓ ✓ | ✗ ✓ ✓ ✓ |
| js | zero | sampled | shrinkage | ✗ ✓ ✗ ✗ | ✗ ✓ ✗ ✗ |
| js | zero | exact | shrinkage | ✗ ✓ ✗ ✗ | ✗ ✓ ✗ ✗ |

Null floor (null condition): |mean increment error| B→M / M→A per variant (exact readout, VCS): indep_sampled 0.0004 / 0.0008; indep_exact 0.0000 / 0.0008; nested_sampled 0.0000 / 0.0004; nested_exact 0.0001 / 0.0007; zero 0.0000 / 0.0000

## 6.2 Visual nested increments (per encoder seed; main pair h vs logit_h; mean ± refit sd over repeats)

| run | cell | obj | variant | readout | ΔJ main | R_orth | violations | step h→pca64 | step pca64→pca16 | step pca16→logit16 |
|---|---|---|---|---|---|---|---|---|---|---|
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | indep_sampled | sampled | +0.0017 ± 0.0006 | +0.0111 | 1.4 | +0.0011 | +0.0004 | -0.0019 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | indep_sampled | exact | +0.0022 ± 0.0004 | +0.0111 | 1.4 | +0.0017 | -0.0004 | -0.0012 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | indep_exact | sampled | +0.0018 ± 0.0005 | +0.0061 | 1.4 | +0.0005 | +0.0000 | -0.0006 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | indep_exact | exact | +0.0020 ± 0.0004 | +0.0061 | 1.0 | +0.0008 | -0.0000 | -0.0006 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | nested_sampled | sampled | +0.0001 ± 0.0003 | +0.0002 | 0.4 | +0.0000 | +0.0001 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | nested_sampled | exact | +0.0001 ± 0.0003 | +0.0002 | 0.6 | +0.0000 | +0.0001 | -0.0002 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0002 | 0.4 | -0.0004 | +0.0002 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | nested_exact | exact | +0.0000 ± 0.0000 | +0.0002 | 0.4 | -0.0003 | +0.0002 | +0.0002 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | indep_sampled | sampled | +0.0020 ± 0.0007 | +0.0098 | 1.4 | +0.0014 | -0.0005 | -0.0013 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | indep_sampled | exact | +0.0025 ± 0.0006 | +0.0098 | 1.4 | +0.0019 | -0.0010 | -0.0009 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | indep_exact | sampled | +0.0019 ± 0.0004 | +0.0057 | 1.6 | +0.0005 | -0.0000 | -0.0008 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | indep_exact | exact | +0.0022 ± 0.0007 | +0.0057 | 1.4 | +0.0008 | -0.0001 | -0.0007 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0002 | 0.2 | +0.0000 | +0.0001 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0001 | 0.6 | -0.0000 | +0.0001 | -0.0002 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.4 | -0.0004 | +0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0001 | 0.4 | -0.0003 | +0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | indep_sampled | sampled | +0.0020 ± 0.0005 | +0.0072 | 1.6 | +0.0009 | -0.0002 | -0.0007 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | indep_sampled | exact | +0.0025 ± 0.0005 | +0.0072 | 1.0 | +0.0011 | -0.0001 | +0.0003 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | indep_exact | sampled | +0.0019 ± 0.0004 | +0.0101 | 1.6 | +0.0013 | +0.0004 | -0.0017 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | indep_exact | exact | +0.0020 ± 0.0003 | +0.0101 | 1.6 | +0.0011 | +0.0004 | -0.0018 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.4 | +0.0000 | +0.0000 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0001 | 0.2 | +0.0000 | -0.0000 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0004 | 0.6 | -0.0007 | +0.0001 | -0.0007 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | nested_exact | exact | +0.0000 ± 0.0000 | +0.0004 | 0.8 | -0.0006 | +0.0001 | -0.0005 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | indep_sampled | sampled | +0.0020 ± 0.0006 | +0.0066 | 1.4 | +0.0009 | -0.0005 | -0.0007 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | indep_sampled | exact | +0.0025 ± 0.0007 | +0.0066 | 1.0 | +0.0012 | -0.0003 | -0.0001 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | indep_exact | sampled | +0.0021 ± 0.0005 | +0.0077 | 1.4 | +0.0007 | +0.0004 | -0.0008 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | indep_exact | exact | +0.0022 ± 0.0005 | +0.0077 | 1.6 | +0.0007 | +0.0003 | -0.0011 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0004 | 0.4 | +0.0000 | -0.0000 | -0.0003 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0004 | 0.4 | +0.0000 | -0.0002 | -0.0007 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0004 | 0.8 | -0.0005 | +0.0001 | -0.0004 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0004 | 1.2 | -0.0005 | +0.0001 | -0.0004 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | null | vcs | indep_sampled | sampled | +0.0026 ± 0.0024 | +0.0092 | 1.6 | +0.0023 | -0.0017 | +0.0005 |
| P107_AP3_views4_800ep_seed1 | null | vcs | indep_sampled | exact | +0.0038 ± 0.0032 | +0.0092 | 1.0 | +0.0026 | -0.0014 | +0.0007 |
| P107_AP3_views4_800ep_seed1 | null | vcs | indep_exact | sampled | +0.0021 ± 0.0003 | +0.0071 | 1.4 | +0.0016 | -0.0006 | -0.0005 |
| P107_AP3_views4_800ep_seed1 | null | vcs | indep_exact | exact | +0.0019 ± 0.0004 | +0.0071 | 1.2 | +0.0016 | -0.0007 | -0.0003 |
| P107_AP3_views4_800ep_seed1 | null | vcs | nested_sampled | sampled | +0.0001 ± 0.0003 | +0.0002 | 0.0 | +0.0000 | +0.0000 | +0.0005 |
| P107_AP3_views4_800ep_seed1 | null | vcs | nested_sampled | exact | +0.0002 ± 0.0003 | +0.0002 | 0.0 | +0.0000 | +0.0000 | +0.0004 |
| P107_AP3_views4_800ep_seed1 | null | vcs | nested_exact | sampled | +0.0001 ± 0.0002 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | null | vcs | nested_exact | exact | +0.0001 ± 0.0002 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | null | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | null | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | null | js | indep_sampled | sampled | +0.0026 ± 0.0012 | +0.0122 | 1.8 | +0.0039 | -0.0034 | -0.0006 |
| P107_AP3_views4_800ep_seed1 | null | js | indep_sampled | exact | +0.0032 ± 0.0021 | +0.0122 | 1.2 | +0.0047 | -0.0036 | -0.0006 |
| P107_AP3_views4_800ep_seed1 | null | js | indep_exact | sampled | +0.0022 ± 0.0005 | +0.0068 | 1.6 | +0.0019 | -0.0010 | -0.0008 |
| P107_AP3_views4_800ep_seed1 | null | js | indep_exact | exact | +0.0020 ± 0.0004 | +0.0068 | 1.6 | +0.0018 | -0.0012 | -0.0005 |
| P107_AP3_views4_800ep_seed1 | null | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.4 | +0.0000 | -0.0000 | +0.0001 |
| P107_AP3_views4_800ep_seed1 | null | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0001 | 0.4 | +0.0000 | -0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed1 | null | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0002 | 0.4 | +0.0000 | +0.0001 | -0.0003 |
| P107_AP3_views4_800ep_seed1 | null | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0002 | 0.4 | +0.0000 | +0.0000 | -0.0003 |
| P107_AP3_views4_800ep_seed1 | null | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed1 | null | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | indep_sampled | sampled | +0.0027 ± 0.0009 | +0.0083 | 1.4 | +0.0010 | +0.0001 | -0.0007 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | indep_sampled | exact | +0.0030 ± 0.0012 | +0.0083 | 1.2 | +0.0013 | +0.0002 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | indep_exact | sampled | +0.0022 ± 0.0006 | +0.0085 | 1.6 | +0.0004 | +0.0009 | -0.0012 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | indep_exact | exact | +0.0020 ± 0.0005 | +0.0085 | 1.4 | +0.0007 | +0.0006 | -0.0013 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | nested_sampled | sampled | +0.0004 ± 0.0005 | +0.0004 | 0.2 | +0.0000 | +0.0003 | +0.0005 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | nested_sampled | exact | +0.0004 ± 0.0005 | +0.0004 | 0.4 | +0.0000 | +0.0003 | -0.0002 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | nested_exact | sampled | +0.0001 ± 0.0003 | +0.0003 | 0.8 | +0.0000 | +0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | nested_exact | exact | +0.0001 ± 0.0003 | +0.0003 | 0.8 | +0.0000 | +0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | indep_sampled | sampled | +0.0021 ± 0.0004 | +0.0094 | 1.4 | +0.0007 | +0.0009 | -0.0018 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | indep_sampled | exact | +0.0024 ± 0.0005 | +0.0094 | 1.8 | +0.0009 | +0.0009 | -0.0017 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | indep_exact | sampled | +0.0020 ± 0.0005 | +0.0071 | 1.8 | +0.0005 | +0.0004 | -0.0008 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | indep_exact | exact | +0.0019 ± 0.0006 | +0.0071 | 1.8 | +0.0008 | -0.0000 | -0.0007 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0005 | 0.2 | +0.0000 | +0.0002 | +0.0001 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0005 | 0.4 | +0.0000 | +0.0002 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.6 | -0.0003 | +0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0001 | 0.4 | -0.0004 | +0.0001 | -0.0001 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | indep_sampled | sampled | +0.0019 ± 0.0018 | +0.0069 | 1.4 | +0.0006 | -0.0001 | +0.0006 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | indep_sampled | exact | +0.0035 ± 0.0027 | +0.0069 | 1.2 | +0.0009 | +0.0001 | +0.0017 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | indep_exact | sampled | +0.0019 ± 0.0007 | +0.0060 | 1.6 | +0.0006 | -0.0000 | -0.0005 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | indep_exact | exact | +0.0021 ± 0.0004 | +0.0060 | 1.6 | +0.0008 | -0.0003 | -0.0004 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | nested_sampled | sampled | +0.0001 ± 0.0003 | +0.0005 | 0.6 | -0.0002 | +0.0002 | -0.0002 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | nested_sampled | exact | +0.0001 ± 0.0003 | +0.0005 | 0.8 | -0.0002 | +0.0002 | +0.0003 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | nested_exact | sampled | +0.0000 ± 0.0001 | +0.0005 | 1.0 | -0.0004 | +0.0002 | -0.0010 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | nested_exact | exact | +0.0000 ± 0.0001 | +0.0005 | 1.0 | -0.0004 | +0.0002 | -0.0008 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | indep_sampled | sampled | +0.0020 ± 0.0015 | +0.0068 | 1.6 | +0.0007 | -0.0003 | +0.0002 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | indep_sampled | exact | +0.0028 ± 0.0018 | +0.0068 | 1.2 | +0.0010 | -0.0004 | +0.0006 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | indep_exact | sampled | +0.0022 ± 0.0007 | +0.0064 | 1.8 | +0.0008 | -0.0000 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | indep_exact | exact | +0.0020 ± 0.0003 | +0.0064 | 1.8 | +0.0010 | -0.0004 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0004 | 0.4 | +0.0001 | -0.0001 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0004 | 0.4 | +0.0001 | -0.0002 | -0.0004 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0006 | 1.0 | -0.0002 | +0.0002 | -0.0009 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0006 | 1.0 | -0.0002 | +0.0002 | -0.0008 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | null | vcs | indep_sampled | sampled | +0.0020 ± 0.0014 | +0.0106 | 1.6 | +0.0030 | -0.0021 | -0.0011 |
| P107_AP3_views4_800ep_seed2 | null | vcs | indep_sampled | exact | +0.0028 ± 0.0020 | +0.0106 | 1.8 | +0.0035 | -0.0021 | -0.0012 |
| P107_AP3_views4_800ep_seed2 | null | vcs | indep_exact | sampled | +0.0019 ± 0.0003 | +0.0083 | 1.6 | +0.0030 | -0.0023 | -0.0004 |
| P107_AP3_views4_800ep_seed2 | null | vcs | indep_exact | exact | +0.0019 ± 0.0004 | +0.0083 | 1.4 | +0.0028 | -0.0023 | -0.0003 |
| P107_AP3_views4_800ep_seed2 | null | vcs | nested_sampled | sampled | +0.0001 ± 0.0002 | +0.0005 | 0.6 | +0.0001 | +0.0001 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | null | vcs | nested_sampled | exact | +0.0001 ± 0.0002 | +0.0005 | 0.6 | +0.0000 | +0.0001 | -0.0012 |
| P107_AP3_views4_800ep_seed2 | null | vcs | nested_exact | sampled | +0.0001 ± 0.0001 | +0.0001 | 0.0 | +0.0000 | +0.0000 | +0.0002 |
| P107_AP3_views4_800ep_seed2 | null | vcs | nested_exact | exact | +0.0001 ± 0.0002 | +0.0001 | 0.0 | +0.0000 | +0.0000 | +0.0002 |
| P107_AP3_views4_800ep_seed2 | null | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | null | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | null | js | indep_sampled | sampled | +0.0028 ± 0.0017 | +0.0106 | 1.6 | +0.0028 | -0.0018 | -0.0013 |
| P107_AP3_views4_800ep_seed2 | null | js | indep_sampled | exact | +0.0034 ± 0.0024 | +0.0106 | 1.6 | +0.0033 | -0.0017 | -0.0016 |
| P107_AP3_views4_800ep_seed2 | null | js | indep_exact | sampled | +0.0021 ± 0.0007 | +0.0056 | 1.2 | +0.0009 | -0.0001 | -0.0007 |
| P107_AP3_views4_800ep_seed2 | null | js | indep_exact | exact | +0.0020 ± 0.0005 | +0.0056 | 1.8 | +0.0009 | -0.0004 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | null | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0005 | 0.6 | +0.0000 | +0.0000 | -0.0006 |
| P107_AP3_views4_800ep_seed2 | null | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0005 | 0.6 | +0.0000 | -0.0001 | -0.0012 |
| P107_AP3_views4_800ep_seed2 | null | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.4 | +0.0000 | -0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | null | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0001 | 0.6 | +0.0000 | -0.0000 | -0.0001 |
| P107_AP3_views4_800ep_seed2 | null | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P107_AP3_views4_800ep_seed2 | null | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | indep_sampled | sampled | +0.0021 ± 0.0006 | +0.0079 | 1.2 | +0.0006 | +0.0000 | -0.0007 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | indep_sampled | exact | +0.0024 ± 0.0006 | +0.0079 | 1.4 | +0.0011 | -0.0002 | -0.0006 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | indep_exact | sampled | +0.0020 ± 0.0007 | +0.0078 | 1.6 | +0.0010 | -0.0004 | -0.0006 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | indep_exact | exact | +0.0024 ± 0.0012 | +0.0078 | 2.0 | +0.0008 | -0.0004 | -0.0007 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | nested_sampled | sampled | +0.0001 ± 0.0003 | +0.0000 | 0.4 | +0.0000 | +0.0000 | -0.0001 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | nested_sampled | exact | +0.0001 ± 0.0003 | +0.0000 | 0.2 | +0.0000 | +0.0000 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | nested_exact | sampled | +0.0001 ± 0.0002 | +0.0000 | 0.4 | -0.0005 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | nested_exact | exact | +0.0001 ± 0.0002 | +0.0000 | 0.4 | -0.0004 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | indep_sampled | sampled | +0.0030 ± 0.0018 | +0.0101 | 1.4 | +0.0019 | -0.0016 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | indep_sampled | exact | +0.0030 ± 0.0012 | +0.0101 | 1.6 | +0.0019 | -0.0017 | -0.0002 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | indep_exact | sampled | +0.0016 ± 0.0005 | +0.0070 | 1.8 | +0.0008 | -0.0005 | -0.0003 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | indep_exact | exact | +0.0018 ± 0.0006 | +0.0070 | 2.0 | +0.0005 | -0.0002 | -0.0006 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | nested_sampled | sampled | +0.0002 ± 0.0004 | +0.0001 | 0.4 | +0.0001 | -0.0000 | -0.0002 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | nested_sampled | exact | +0.0002 ± 0.0004 | +0.0001 | 0.2 | +0.0001 | +0.0000 | -0.0003 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | nested_exact | sampled | +0.0000 ± 0.0000 | -0.0000 | 0.4 | -0.0005 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | nested_exact | exact | +0.0000 ± 0.0000 | -0.0000 | 0.2 | -0.0005 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | indep_sampled | sampled | +0.0021 ± 0.0004 | +0.0071 | 1.4 | +0.0006 | +0.0004 | -0.0010 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | indep_sampled | exact | +0.0032 ± 0.0010 | +0.0071 | 1.2 | +0.0008 | +0.0008 | -0.0008 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | indep_exact | sampled | +0.0018 ± 0.0005 | +0.0050 | 1.6 | +0.0005 | -0.0000 | -0.0006 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | indep_exact | exact | +0.0023 ± 0.0006 | +0.0050 | 1.6 | +0.0005 | -0.0002 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0003 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0000 | 0.2 | +0.0000 | +0.0000 | +0.0001 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | nested_exact | sampled | +0.0001 ± 0.0001 | +0.0002 | 0.4 | -0.0001 | +0.0001 | +0.0003 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | nested_exact | exact | +0.0001 ± 0.0001 | +0.0002 | 0.6 | -0.0002 | +0.0000 | -0.0000 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | indep_sampled | sampled | +0.0021 ± 0.0003 | +0.0066 | 1.4 | +0.0005 | +0.0001 | -0.0005 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | indep_sampled | exact | +0.0031 ± 0.0008 | +0.0066 | 1.2 | +0.0007 | +0.0006 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | indep_exact | sampled | +0.0018 ± 0.0006 | +0.0049 | 1.6 | +0.0005 | -0.0001 | -0.0006 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | indep_exact | exact | +0.0021 ± 0.0004 | +0.0049 | 1.8 | +0.0004 | -0.0003 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.2 | +0.0000 | +0.0000 | +0.0003 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0001 | 0.4 | +0.0000 | +0.0000 | +0.0001 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.4 | -0.0001 | +0.0000 | +0.0002 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0001 | 0.6 | -0.0001 | +0.0000 | -0.0001 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | vcs | indep_sampled | sampled | +0.0027 ± 0.0014 | +0.0104 | 1.8 | +0.0031 | -0.0031 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | null | vcs | indep_sampled | exact | +0.0031 ± 0.0021 | +0.0104 | 1.6 | +0.0038 | -0.0034 | -0.0002 |
| P41_simclr_views4_800ep_seed1 | null | vcs | indep_exact | sampled | +0.0022 ± 0.0006 | +0.0047 | 1.4 | +0.0006 | -0.0005 | -0.0001 |
| P41_simclr_views4_800ep_seed1 | null | vcs | indep_exact | exact | +0.0020 ± 0.0004 | +0.0047 | 1.8 | +0.0005 | -0.0005 | -0.0001 |
| P41_simclr_views4_800ep_seed1 | null | vcs | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0003 | 0.4 | +0.0000 | -0.0000 | -0.0002 |
| P41_simclr_views4_800ep_seed1 | null | vcs | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0003 | 0.4 | +0.0000 | -0.0001 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | null | vcs | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0001 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | vcs | nested_exact | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | js | indep_sampled | sampled | +0.0030 ± 0.0025 | +0.0076 | 2.0 | +0.0018 | -0.0020 | -0.0003 |
| P41_simclr_views4_800ep_seed1 | null | js | indep_sampled | exact | +0.0035 ± 0.0037 | +0.0076 | 1.6 | +0.0016 | -0.0017 | -0.0000 |
| P41_simclr_views4_800ep_seed1 | null | js | indep_exact | sampled | +0.0022 ± 0.0005 | +0.0049 | 1.4 | +0.0006 | -0.0005 | -0.0000 |
| P41_simclr_views4_800ep_seed1 | null | js | indep_exact | exact | +0.0020 ± 0.0005 | +0.0049 | 1.6 | +0.0005 | -0.0005 | -0.0001 |
| P41_simclr_views4_800ep_seed1 | null | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0003 | 0.4 | +0.0000 | -0.0000 | -0.0002 |
| P41_simclr_views4_800ep_seed1 | null | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0003 | 0.8 | +0.0000 | -0.0001 | -0.0004 |
| P41_simclr_views4_800ep_seed1 | null | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.2 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0000 | 0.4 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed1 | null | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | indep_sampled | sampled | +0.0020 ± 0.0006 | +0.0117 | 1.6 | +0.0032 | -0.0030 | -0.0003 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | indep_sampled | exact | +0.0022 ± 0.0005 | +0.0117 | 1.6 | +0.0040 | -0.0039 | +0.0006 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | indep_exact | sampled | +0.0021 ± 0.0006 | +0.0047 | 1.2 | +0.0004 | -0.0001 | -0.0001 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | indep_exact | exact | +0.0020 ± 0.0003 | +0.0047 | 1.8 | +0.0005 | -0.0004 | -0.0001 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0006 | 0.4 | +0.0001 | +0.0004 | -0.0002 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0007 | 0.4 | +0.0000 | +0.0004 | -0.0004 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | nested_exact | sampled | +0.0001 ± 0.0002 | +0.0001 | 0.4 | -0.0001 | +0.0000 | -0.0002 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | nested_exact | exact | +0.0001 ± 0.0002 | +0.0001 | 0.6 | -0.0001 | +0.0000 | -0.0001 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | indep_sampled | sampled | +0.0021 ± 0.0005 | +0.0117 | 1.8 | +0.0031 | -0.0029 | -0.0008 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | indep_sampled | exact | +0.0024 ± 0.0005 | +0.0117 | 1.8 | +0.0041 | -0.0040 | -0.0002 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | indep_exact | sampled | +0.0020 ± 0.0005 | +0.0046 | 1.4 | +0.0005 | -0.0002 | +0.0001 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | indep_exact | exact | +0.0020 ± 0.0004 | +0.0046 | 1.6 | +0.0006 | -0.0005 | +0.0003 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0003 | 0.6 | -0.0000 | +0.0002 | -0.0003 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0003 | 0.6 | -0.0001 | +0.0002 | -0.0006 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0001 | 0.2 | +0.0000 | +0.0001 | -0.0002 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0001 | 0.4 | +0.0000 | +0.0000 | -0.0001 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | indep_sampled | sampled | +0.0017 ± 0.0008 | +0.0076 | 1.6 | +0.0001 | +0.0007 | -0.0011 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | indep_sampled | exact | +0.0026 ± 0.0008 | +0.0076 | 1.2 | +0.0005 | +0.0014 | -0.0010 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | indep_exact | sampled | +0.0024 ± 0.0017 | +0.0052 | 1.4 | +0.0004 | +0.0003 | -0.0007 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | indep_exact | exact | +0.0024 ± 0.0014 | +0.0052 | 1.6 | +0.0004 | -0.0001 | -0.0007 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | nested_exact | sampled | +0.0003 ± 0.0006 | +0.0002 | 0.8 | +0.0000 | +0.0001 | -0.0004 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | nested_exact | exact | +0.0003 ± 0.0006 | +0.0002 | 1.0 | +0.0000 | +0.0001 | -0.0004 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | indep_sampled | sampled | +0.0020 ± 0.0006 | +0.0081 | 1.6 | +0.0003 | +0.0007 | -0.0012 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | indep_sampled | exact | +0.0028 ± 0.0007 | +0.0081 | 1.0 | +0.0007 | +0.0013 | -0.0012 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | indep_exact | sampled | +0.0018 ± 0.0004 | +0.0053 | 1.2 | +0.0003 | +0.0004 | -0.0008 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | indep_exact | exact | +0.0018 ± 0.0003 | +0.0053 | 1.6 | +0.0003 | +0.0000 | -0.0007 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0003 | 0.6 | +0.0000 | +0.0001 | -0.0004 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0002 | 0.8 | +0.0000 | +0.0001 | -0.0004 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | null | vcs | indep_sampled | sampled | +0.0023 ± 0.0010 | +0.0112 | 1.6 | +0.0021 | -0.0002 | -0.0006 |
| P41_simclr_views4_800ep_seed2 | null | vcs | indep_sampled | exact | +0.0027 ± 0.0016 | +0.0112 | 1.2 | +0.0023 | +0.0004 | -0.0013 |
| P41_simclr_views4_800ep_seed2 | null | vcs | indep_exact | sampled | +0.0026 ± 0.0010 | +0.0072 | 1.6 | +0.0018 | -0.0014 | +0.0001 |
| P41_simclr_views4_800ep_seed2 | null | vcs | indep_exact | exact | +0.0022 ± 0.0009 | +0.0072 | 1.6 | +0.0019 | -0.0017 | -0.0000 |
| P41_simclr_views4_800ep_seed2 | null | vcs | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0005 | 0.2 | +0.0000 | +0.0003 | +0.0006 |
| P41_simclr_views4_800ep_seed2 | null | vcs | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0006 | 0.2 | +0.0000 | +0.0003 | -0.0004 |
| P41_simclr_views4_800ep_seed2 | null | vcs | nested_exact | sampled | +0.0001 ± 0.0002 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0001 |
| P41_simclr_views4_800ep_seed2 | null | vcs | nested_exact | exact | +0.0001 ± 0.0002 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0001 |
| P41_simclr_views4_800ep_seed2 | null | vcs | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | null | vcs | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | null | js | indep_sampled | sampled | +0.0027 ± 0.0016 | +0.0131 | 1.6 | +0.0029 | -0.0013 | -0.0010 |
| P41_simclr_views4_800ep_seed2 | null | js | indep_sampled | exact | +0.0034 ± 0.0022 | +0.0132 | 1.4 | +0.0032 | -0.0009 | -0.0017 |
| P41_simclr_views4_800ep_seed2 | null | js | indep_exact | sampled | +0.0021 ± 0.0004 | +0.0063 | 1.4 | +0.0012 | -0.0009 | +0.0004 |
| P41_simclr_views4_800ep_seed2 | null | js | indep_exact | exact | +0.0020 ± 0.0005 | +0.0063 | 1.4 | +0.0011 | -0.0009 | +0.0002 |
| P41_simclr_views4_800ep_seed2 | null | js | nested_sampled | sampled | +0.0000 ± 0.0000 | +0.0003 | 0.2 | +0.0000 | +0.0003 | +0.0005 |
| P41_simclr_views4_800ep_seed2 | null | js | nested_sampled | exact | +0.0000 ± 0.0000 | +0.0004 | 0.2 | +0.0000 | +0.0003 | -0.0006 |
| P41_simclr_views4_800ep_seed2 | null | js | nested_exact | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0001 | +0.0002 |
| P41_simclr_views4_800ep_seed2 | null | js | nested_exact | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0001 |
| P41_simclr_views4_800ep_seed2 | null | js | zero | sampled | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |
| P41_simclr_views4_800ep_seed2 | null | js | zero | exact | +0.0000 ± 0.0000 | +0.0000 | 0.0 | +0.0000 | +0.0000 | +0.0000 |

## 6.2 I1 audits, seeds 1–2 (rejection rate; level flag > 0.09 at R = 200)

| run | family | cell | R | h vcs / js | logits vcs / js | any-layer maxT vcs / js | flag |
|---|---|---|---|---|---|---|---|
| P107_AP3_views4_800ep_seed1 | blur | null_label_only:0:2000 | 200 | 0.04 / 0.06 | 0.07 / 0.06 | 0.06 / 0.05 | — |
| P107_AP3_views4_800ep_seed1 | blur | null_all_planted:0:2000 | 200 | 0.04 / 0.05 | 0.06 / 0.05 | 0.07 / 0.04 | — |
| P107_AP3_views4_800ep_seed1 | blur | planted:0.25:2000 | 100 | 0.13 / 0.15 | 0.09 / 0.10 | 0.75 / 0.76 | — |
| P107_AP3_views4_800ep_seed1 | colour | null_label_only:0:2000 | 200 | 0.04 / 0.04 | 0.07 / 0.10 | 0.05 / 0.09 | FLAG |
| P107_AP3_views4_800ep_seed1 | colour | null_all_planted:0:2000 | 200 | 0.07 / 0.05 | 0.05 / 0.07 | 0.06 / 0.05 | — |
| P107_AP3_views4_800ep_seed1 | colour | planted:0.1:2000 | 100 | 0.60 / 0.58 | 0.13 / 0.11 | 1.00 / 1.00 | — |
| P107_AP3_views4_800ep_seed2 | blur | null_label_only:0:2000 | 200 | 0.03 / 0.03 | 0.04 / 0.04 | 0.04 / 0.02 | — |
| P107_AP3_views4_800ep_seed2 | blur | null_all_planted:0:2000 | 200 | 0.07 / 0.05 | 0.05 / 0.04 | 0.04 / 0.04 | — |
| P107_AP3_views4_800ep_seed2 | blur | planted:0.25:2000 | 100 | 0.14 / 0.15 | 0.08 / 0.04 | 0.73 / 0.75 | — |
| P107_AP3_views4_800ep_seed2 | colour | null_label_only:0:2000 | 200 | 0.04 / 0.05 | 0.04 / 0.06 | 0.04 / 0.07 | — |
| P107_AP3_views4_800ep_seed2 | colour | null_all_planted:0:2000 | 200 | 0.05 / 0.06 | 0.03 / 0.03 | 0.04 / 0.05 | — |
| P107_AP3_views4_800ep_seed2 | colour | planted:0.1:2000 | 100 | 0.58 / 0.44 | 0.14 / 0.10 | 1.00 / 1.00 | — |
| P41_simclr_views4_800ep_seed1 | blur | null_label_only:0:2000 | 200 | 0.06 / 0.04 | 0.06 / 0.07 | 0.07 / 0.06 | — |
| P41_simclr_views4_800ep_seed1 | blur | null_all_planted:0:2000 | 200 | 0.07 / 0.07 | 0.06 / 0.07 | 0.07 / 0.07 | — |
| P41_simclr_views4_800ep_seed1 | blur | planted:0.25:2000 | 100 | 0.10 / 0.08 | 0.02 / 0.04 | 0.60 / 0.57 | — |
| P41_simclr_views4_800ep_seed1 | colour | null_label_only:0:2000 | 200 | 0.07 / 0.07 | 0.05 / 0.04 | 0.06 / 0.04 | — |
| P41_simclr_views4_800ep_seed1 | colour | null_all_planted:0:2000 | 200 | 0.07 / 0.06 | 0.05 / 0.07 | 0.06 / 0.07 | FLAG |
| P41_simclr_views4_800ep_seed1 | colour | planted:0.1:2000 | 100 | 0.08 / 0.10 | 0.11 / 0.06 | 1.00 / 1.00 | — |
| P41_simclr_views4_800ep_seed2 | blur | null_label_only:0:2000 | 200 | 0.05 / 0.04 | 0.07 / 0.07 | 0.07 / 0.06 | — |
| P41_simclr_views4_800ep_seed2 | blur | null_all_planted:0:2000 | 200 | 0.04 / 0.06 | 0.06 / 0.06 | 0.05 / 0.04 | — |
| P41_simclr_views4_800ep_seed2 | blur | planted:0.25:2000 | 100 | 0.06 / 0.04 | 0.05 / 0.08 | 0.73 / 0.74 | — |
| P41_simclr_views4_800ep_seed2 | colour | null_label_only:0:2000 | 200 | 0.05 / 0.05 | 0.08 / 0.07 | 0.08 / 0.07 | — |
| P41_simclr_views4_800ep_seed2 | colour | null_all_planted:0:2000 | 200 | 0.04 / 0.04 | 0.07 / 0.04 | 0.06 / 0.04 | — |
| P41_simclr_views4_800ep_seed2 | colour | planted:0.1:2000 | 100 | 0.11 / 0.08 | 0.05 / 0.08 | 1.00 / 1.00 | — |

## Paired prediction effects (existing P109 effect files, seeds 1–2; no new computation)

| run | version | Δ p(true) [CI] | Δ acc | flip rate |
|---|---|---|---|---|
| P107_AP3_views4_800ep_seed1 | colour_s0.1 | -0.0027 [-0.0043, -0.0011] | -0.0018 | 0.028 |
| P107_AP3_views4_800ep_seed1 | blur_s0.25 | +0.0001 [-0.0008, +0.0009] | +0.0014 | 0.015 |
| P107_AP3_views4_800ep_seed2 | colour_s0.1 | -0.0038 [-0.0054, -0.0022] | -0.0040 | 0.029 |
| P107_AP3_views4_800ep_seed2 | blur_s0.25 | +0.0005 [-0.0004, +0.0014] | +0.0003 | 0.014 |
| P41_simclr_views4_800ep_seed1 | colour_s0.1 | -0.0029 [-0.0043, -0.0015] | -0.0025 | 0.022 |
| P41_simclr_views4_800ep_seed1 | blur_s0.25 | +0.0003 [-0.0004, +0.0010] | +0.0005 | 0.011 |
| P41_simclr_views4_800ep_seed2 | colour_s0.1 | -0.0027 [-0.0041, -0.0014] | -0.0051 | 0.024 |
| P41_simclr_views4_800ep_seed2 | blur_s0.25 | -0.0001 [-0.0009, +0.0006] | -0.0015 | 0.013 |
