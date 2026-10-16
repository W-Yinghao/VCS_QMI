# V9-N2 precheck — f30k, rho 0.4, seed 903 — formal block ALLOWED

## 1. Teacher scores on A-critic-select (P clean n=2175, Q re-paired n=2175): AUROC 0.9112 (thresholds > 0.999 / < 0.55 refuse), overlap 0.328, share of P below Q median 0.022

| set | q5 | q25 | q50 | q75 | q95 | mean |
|---|---|---|---|---|---|---|
| P | 0.2114 | 0.5653 | 0.7772 | 0.9062 | 0.9580 | 0.7090 |
| Q | 0.0168 | 0.0432 | 0.1352 | 0.3475 | 0.7321 | 0.2256 |

## 2. Weights on U per method

| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |
|---|---|---|---|---|---|---|---|---|---|
| BCE_Weight | 130500 | 0.116 | 0.758 | 0.968 | 0.620 | 101115.5 | 80929.4 | 0.000 | 0.000 |
| Nystrom_Weight | 130500 | 0.021 | 0.754 | 0.982 | 0.613 | 98136.9 | 79989.4 | 0.000 | 0.000 |
| PlainFilter | 130500 | 0.000 | 1.000 | 1.000 | 0.724 | 94513.0 | 94513.0 | 0.276 | 0.724 |
| RankMass | 130500 | 0.000 | 1.000 | 1.000 | 0.623 | 81344.6 | 81344.3 | 0.377 | 0.623 |
| RuLSIF_Weight | 130500 | 0.029 | 0.751 | 0.983 | 0.612 | 98167.2 | 79845.7 | 0.000 | 0.000 |
| Unweighted_A | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| VCS_Weight | 130500 | 0.118 | 0.763 | 0.969 | 0.623 | 101416.5 | 81344.3 | 0.000 | 0.000 |

| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |
|---|---|---|---|---|
| BCE_Weight|Nystrom_Weight | 0.0328 | 1.000 | 0.000 | 0.0989 |
| BCE_Weight|PlainFilter | 0.1918 | 0.774 | 0.000 | 0.7168 |
| BCE_Weight|RankMass | 0.1687 | 0.839 | 0.000 | 0.5003 |
| BCE_Weight|RuLSIF_Weight | 0.0329 | 0.987 | 0.000 | 0.0912 |
| BCE_Weight|Unweighted_A | 0.3799 | NA (constant) | 0.000 | 0.8924 |
| BCE_Weight|VCS_Weight | 0.0032 | 1.000 | 0.000 | 0.0064 |
| Nystrom_Weight|PlainFilter | 0.1709 | 0.774 | 0.000 | 0.6638 |
| Nystrom_Weight|RankMass | 0.1604 | 0.839 | 0.000 | 0.5512 |
| Nystrom_Weight|RuLSIF_Weight | 0.0070 | 0.987 | 0.000 | 0.0629 |
| Nystrom_Weight|Unweighted_A | 0.3871 | NA (constant) | 0.000 | 0.9914 |
| Nystrom_Weight|VCS_Weight | 0.0326 | 1.000 | 0.000 | 0.1008 |
| PlainFilter|RankMass | 0.1009 | 0.794 | 0.899 | 1.0000 |
| PlainFilter|RuLSIF_Weight | 0.1721 | 0.774 | 0.000 | 0.6772 |
| PlainFilter|Unweighted_A | 0.2758 | NA (constant) | 0.724 | 1.0000 |
| PlainFilter|VCS_Weight | 0.1902 | 0.774 | 0.000 | 0.7122 |
| RankMass|RuLSIF_Weight | 0.1608 | 0.839 | 0.000 | 0.5506 |
| RankMass|Unweighted_A | 0.3767 | NA (constant) | 0.623 | 1.0000 |
| RankMass|VCS_Weight | 0.1682 | 0.839 | 0.000 | 0.5060 |
| RuLSIF_Weight|Unweighted_A | 0.3882 | NA (constant) | 0.000 | 0.9837 |
| RuLSIF_Weight|VCS_Weight | 0.0329 | 0.987 | 0.000 | 0.0932 |
| Unweighted_A|VCS_Weight | 0.3767 | NA (constant) | 0.000 | 0.8905 |

## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)

| method | n pool | mean | min | max | sha256 |
|---|---|---|---|---|---|
| BCE_Weight | 145000 | 1.000000 | 0.163426 | 1.519448 | ee2d65e53adbd286 |
| Nystrom_Weight | 145000 | 1.000000 | 0.013269 | 1.534564 | 9a8c0b3d26af1619 |
| PlainFilter | 145000 | 1.000000 | 0.000000 | 1.330117 | 91c77d70bc5fc8ae |
| RankMass | 145000 | 1.000000 | 0.000000 | 1.512870 | 0809d425ec620e1c |
| RuLSIF_Weight | 145000 | 1.000000 | 0.025094 | 1.536901 | 5f745078251cf027 |
| Unweighted_A | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| VCS_Weight | 145000 | 1.000000 | 0.165633 | 1.512870 | 165cfc5de5f1a4d0 |

Identical groups: [['BCE_Weight'], ['Nystrom_Weight'], ['PlainFilter'], ['RankMass'], ['RuLSIF_Weight'], ['Unweighted_A'], ['VCS_Weight']]; merged (run once): none — rule: merge only if identical coefficients AND same seed-derived order/RNG; otherwise both methods run

## 4. Roles and isolation

counts {'teacher_warmup': 1160, 'critic_fit': 870, 'critic_select': 435, 'reference_moments': 435}, U 26100, anchor fraction 0.1, split [0.4, 0.3, 0.15, 0.15]

- teacher_warmup: teacher training only
- critic_fit: critic fitting (P and within-role Q)
- critic_select: lr / start selection of the critic + the precheck separability table
- reference_moments: a = E_P T, b = E_Q T
- U: rho point estimate (80 % of U images) + independent interval batch (20 %); student training with weights; anchors join the student with v = 1

- train_ids: {'n': 29000, 'roles_within_train': True, 'ids_are_per_split_indices': False}
- dev_ids: {'n': 5070, 'overlap_with_roles': 0, 'disjoint': True}
- test_ids: {'n': 5000, 'overlap_with_roles': 0, 'disjoint': True}
- note: dev / test are separate official split files; role images are indices into the train split only
- dev_caps_sha256: 5ef92f119d117690
- test_caps_sha256: eca5cf082363989a

Noise: design rho 0.4, realized 0.4000, mapping sha256 9a7cae760ecad377, A mismatched 0
