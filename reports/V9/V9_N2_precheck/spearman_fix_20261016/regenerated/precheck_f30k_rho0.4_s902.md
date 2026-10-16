# V9-N2 precheck — f30k, rho 0.4, seed 902 — formal block ALLOWED

## 1. Teacher scores on A-critic-select (P clean n=2175, Q re-paired n=2175): AUROC 0.9032 (thresholds > 0.999 / < 0.55 refuse), overlap 0.348, share of P below Q median 0.019

| set | q5 | q25 | q50 | q75 | q95 | mean |
|---|---|---|---|---|---|---|
| P | 0.2036 | 0.5828 | 0.7985 | 0.9108 | 0.9627 | 0.7159 |
| Q | 0.0233 | 0.0422 | 0.1193 | 0.3582 | 0.7894 | 0.2332 |

## 2. Weights on U per method

| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |
|---|---|---|---|---|---|---|---|---|---|
| BCE_Weight | 130500 | 0.126 | 0.705 | 0.959 | 0.598 | 100013.5 | 77989.9 | 0.000 | 0.000 |
| Nystrom_Weight | 130500 | 0.012 | 0.737 | 0.973 | 0.593 | 97196.6 | 77368.6 | 0.021 | 0.000 |
| PlainFilter | 130500 | 0.000 | 1.000 | 1.000 | 0.710 | 92654.0 | 92654.0 | 0.290 | 0.710 |
| RankMass | 130500 | 0.000 | 1.000 | 1.000 | 0.607 | 79192.4 | 79192.3 | 0.393 | 0.607 |
| RuLSIF_Weight | 130500 | 0.022 | 0.740 | 0.973 | 0.593 | 97242.1 | 77350.9 | 0.000 | 0.000 |
| Unweighted_A | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| VCS_Weight | 130500 | 0.132 | 0.721 | 0.963 | 0.607 | 100853.9 | 79192.3 | 0.000 | 0.000 |

| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |
|---|---|---|---|---|
| BCE_Weight|Nystrom_Weight | 0.0397 | 1.000 | 0.000 | 0.1226 |
| BCE_Weight|PlainFilter | 0.2068 | 0.786 | 0.000 | 0.7270 |
| BCE_Weight|RankMass | 0.1796 | 0.846 | 0.000 | 0.5234 |
| BCE_Weight|RuLSIF_Weight | 0.0405 | 0.996 | 0.000 | 0.1110 |
| BCE_Weight|Unweighted_A | 0.4024 | NA (constant) | 0.000 | 0.8826 |
| BCE_Weight|VCS_Weight | 0.0092 | 1.000 | 0.000 | 0.0175 |
| Nystrom_Weight|PlainFilter | 0.1838 | 0.786 | 0.021 | 0.6595 |
| Nystrom_Weight|RankMass | 0.1696 | 0.846 | 0.021 | 0.5315 |
| Nystrom_Weight|RuLSIF_Weight | 0.0062 | 0.996 | 0.001 | 0.0412 |
| Nystrom_Weight|Unweighted_A | 0.4071 | NA (constant) | 0.000 | 1.0000 |
| Nystrom_Weight|VCS_Weight | 0.0383 | 1.000 | 0.000 | 0.1284 |
| PlainFilter|RankMass | 0.1032 | 0.794 | 0.897 | 1.0000 |
| PlainFilter|RuLSIF_Weight | 0.1841 | 0.786 | 0.000 | 0.6530 |
| PlainFilter|Unweighted_A | 0.2900 | NA (constant) | 0.710 | 1.0000 |
| PlainFilter|VCS_Weight | 0.2020 | 0.786 | 0.000 | 0.7147 |
| RankMass|RuLSIF_Weight | 0.1696 | 0.846 | 0.000 | 0.5307 |
| RankMass|Unweighted_A | 0.3932 | NA (constant) | 0.607 | 1.0000 |
| RankMass|VCS_Weight | 0.1779 | 0.846 | 0.000 | 0.5063 |
| RuLSIF_Weight|Unweighted_A | 0.4073 | NA (constant) | 0.000 | 0.9936 |
| RuLSIF_Weight|VCS_Weight | 0.0393 | 0.996 | 0.000 | 0.1166 |
| Unweighted_A|VCS_Weight | 0.3932 | NA (constant) | 0.000 | 0.8770 |

## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)

| method | n pool | mean | min | max | sha256 |
|---|---|---|---|---|---|
| BCE_Weight | 145000 | 1.000000 | 0.184091 | 1.567738 | b1d89f2fa0c143e5 |
| Nystrom_Weight | 145000 | 1.000000 | 0.000000 | 1.578341 | 70ec125cbc9b7d28 |
| PlainFilter | 145000 | 1.000000 | 0.000000 | 1.353193 | ef3924ae7c5d1985 |
| RankMass | 145000 | 1.000000 | 0.000000 | 1.547620 | 962d9bf4fbae8f51 |
| RuLSIF_Weight | 145000 | 1.000000 | 0.010094 | 1.578645 | 666bb742f96c16bb |
| Unweighted_A | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| VCS_Weight | 145000 | 1.000000 | 0.190292 | 1.547620 | 602d55f57f26ee9d |

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

Noise: design rho 0.4, realized 0.4000, mapping sha256 5bce6b176453c876, A mismatched 0
