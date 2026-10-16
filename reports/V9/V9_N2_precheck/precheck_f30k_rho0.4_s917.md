# V9-N2 precheck — f30k, rho 0.4, seed 917 — formal block ALLOWED

## 1. Teacher scores on A-critic-select (P clean n=2175, Q re-paired n=2175): AUROC 0.9035 (thresholds > 0.999 / < 0.55 refuse), overlap 0.346, share of P below Q median 0.024

| set | q5 | q25 | q50 | q75 | q95 | mean |
|---|---|---|---|---|---|---|
| P | 0.2673 | 0.6459 | 0.8349 | 0.9304 | 0.9651 | 0.7557 |
| Q | 0.0208 | 0.0489 | 0.1694 | 0.4486 | 0.8131 | 0.2719 |

## 2. Weights on U per method

| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |
|---|---|---|---|---|---|---|---|---|---|
| BCE_Weight | 130500 | 0.088 | 0.747 | 0.943 | 0.598 | 99027.0 | 78048.0 | 0.000 | 0.000 |
| Nystrom_Weight | 130500 | 0.021 | 0.742 | 0.966 | 0.595 | 97366.9 | 77593.4 | 0.000 | 0.000 |
| PlainFilter | 130500 | 0.000 | 1.000 | 1.000 | 0.719 | 93782.0 | 93782.0 | 0.281 | 0.719 |
| RankMass | 130500 | 0.000 | 1.000 | 1.000 | 0.596 | 77833.5 | 77833.3 | 0.404 | 0.596 |
| RuLSIF_Weight | 130500 | 0.021 | 0.740 | 0.967 | 0.594 | 97173.5 | 77569.4 | 0.000 | 0.000 |
| Unweighted_A | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| VCS_Weight | 130500 | 0.087 | 0.745 | 0.943 | 0.596 | 98792.7 | 77833.3 | 0.000 | 0.000 |

| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |
|---|---|---|---|---|
| BCE_Weight|Nystrom_Weight | 0.0265 | 1.000 | 0.000 | 0.0675 |
| BCE_Weight|PlainFilter | 0.1949 | 0.595 | 0.000 | 0.7356 |
| BCE_Weight|RankMass | 0.1715 | 0.718 | 0.000 | 0.5555 |
| BCE_Weight|RuLSIF_Weight | 0.0271 | 0.988 | 0.000 | 0.0759 |
| BCE_Weight|Unweighted_A | 0.4019 | -0.008 | 0.000 | 0.9190 |
| BCE_Weight|VCS_Weight | 0.0016 | 1.000 | 0.000 | 0.0036 |
| Nystrom_Weight|PlainFilter | 0.1870 | 0.595 | 0.000 | 0.6910 |
| Nystrom_Weight|RankMass | 0.1695 | 0.718 | 0.000 | 0.5429 |
| Nystrom_Weight|RuLSIF_Weight | 0.0043 | 0.988 | 0.000 | 0.0252 |
| Nystrom_Weight|Unweighted_A | 0.4054 | -0.008 | 0.000 | 0.9829 |
| Nystrom_Weight|VCS_Weight | 0.0263 | 1.000 | 0.000 | 0.0659 |
| PlainFilter|RankMass | 0.1222 | 0.824 | 0.878 | 1.0000 |
| PlainFilter|RuLSIF_Weight | 0.1862 | 0.595 | 0.000 | 0.6947 |
| PlainFilter|Unweighted_A | 0.2814 | 0.602 | 0.719 | 1.0000 |
| PlainFilter|VCS_Weight | 0.1954 | 0.595 | 0.000 | 0.7388 |
| RankMass|RuLSIF_Weight | 0.1684 | 0.718 | 0.000 | 0.5419 |
| RankMass|Unweighted_A | 0.4036 | 0.515 | 0.596 | 1.0000 |
| RankMass|VCS_Weight | 0.1711 | 0.718 | 0.000 | 0.5522 |
| RuLSIF_Weight|Unweighted_A | 0.4056 | -0.007 | 0.000 | 0.9948 |
| RuLSIF_Weight|VCS_Weight | 0.0269 | 0.988 | 0.000 | 0.0743 |
| Unweighted_A|VCS_Weight | 0.4036 | -0.008 | 0.000 | 0.9205 |

## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)

| method | n pool | mean | min | max | sha256 |
|---|---|---|---|---|---|
| BCE_Weight | 145000 | 1.000000 | 0.126948 | 1.566755 | 9d603c2c6bb208f6 |
| Nystrom_Weight | 145000 | 1.000000 | 0.026893 | 1.574489 | 407c93783839b3d4 |
| PlainFilter | 145000 | 1.000000 | 0.000000 | 1.339096 | e031a2758261eb8c |
| RankMass | 145000 | 1.000000 | 0.000000 | 1.570398 | e6e3d134ed0ac1b8 |
| RuLSIF_Weight | 145000 | 1.000000 | 0.008123 | 1.574900 | 4079182bda084001 |
| Unweighted_A | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| VCS_Weight | 145000 | 1.000000 | 0.124835 | 1.570398 | 9e926d528afbdb52 |

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

Noise: design rho 0.4, realized 0.4000, mapping sha256 900041f60307b1f3, A mismatched 0
