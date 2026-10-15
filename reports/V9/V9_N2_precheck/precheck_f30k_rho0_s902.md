# V9-N2 precheck — f30k, rho 0, seed 902 — formal block ALLOWED

## 1. Teacher scores on A-critic-select (P clean n=2175, Q re-paired n=2175): AUROC 0.9032 (thresholds > 0.999 / < 0.55 refuse), overlap 0.348, share of P below Q median 0.019

| set | q5 | q25 | q50 | q75 | q95 | mean |
|---|---|---|---|---|---|---|
| P | 0.2036 | 0.5828 | 0.7985 | 0.9108 | 0.9627 | 0.7159 |
| Q | 0.0233 | 0.0422 | 0.1193 | 0.3582 | 0.7894 | 0.2332 |

## 2. Weights on U per method

| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |
|---|---|---|---|---|---|---|---|---|---|
| BCE_Weight | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| Nystrom_Weight | 130500 | 0.996 | 1.000 | 1.000 | 0.999 | 130488.4 | 130350.1 | 0.000 | 0.000 |
| PlainFilter | 130500 | 0.000 | 1.000 | 1.000 | 0.943 | 123072.0 | 123072.0 | 0.057 | 0.943 |
| RankMass | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| RuLSIF_Weight | 130500 | 0.995 | 1.000 | 1.000 | 0.998 | 130495.7 | 130292.0 | 0.000 | 0.000 |
| Unweighted_A | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| VCS_Weight | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |

| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |
|---|---|---|---|---|
| BCE_Weight|Nystrom_Weight | 0.0011 | -0.049 | 0.000 | 1.0000 |
| BCE_Weight|PlainFilter | 0.0569 | 0.885 | 0.943 | 1.0000 |
| BCE_Weight|RankMass | 0.0000 | 1.000 | 1.000 | 0.0000 |
| BCE_Weight|RuLSIF_Weight | 0.0016 | -0.048 | 0.000 | 0.1889 |
| BCE_Weight|Unweighted_A | 0.0000 | 1.000 | 1.000 | 0.0000 |
| BCE_Weight|VCS_Weight | 0.0000 | 1.000 | 1.000 | 0.0000 |
| Nystrom_Weight|PlainFilter | 0.0566 | 0.123 | 0.000 | 0.9967 |
| Nystrom_Weight|RankMass | 0.0011 | -0.049 | 0.000 | 1.0000 |
| Nystrom_Weight|RuLSIF_Weight | 0.0006 | 0.984 | 0.000 | 0.8570 |
| Nystrom_Weight|Unweighted_A | 0.0011 | -0.049 | 0.000 | 1.0000 |
| Nystrom_Weight|VCS_Weight | 0.0011 | -0.049 | 0.000 | 1.0000 |
| PlainFilter|RankMass | 0.0569 | 0.885 | 0.943 | 1.0000 |
| PlainFilter|RuLSIF_Weight | 0.0567 | 0.124 | 0.000 | 0.9952 |
| PlainFilter|Unweighted_A | 0.0569 | 0.885 | 0.943 | 1.0000 |
| PlainFilter|VCS_Weight | 0.0569 | 0.885 | 0.943 | 1.0000 |
| RankMass|RuLSIF_Weight | 0.0016 | -0.048 | 0.000 | 0.1889 |
| RankMass|Unweighted_A | 0.0000 | 1.000 | 1.000 | 0.0000 |
| RankMass|VCS_Weight | 0.0000 | 1.000 | 1.000 | 0.0000 |
| RuLSIF_Weight|Unweighted_A | 0.0016 | -0.048 | 0.000 | 0.1889 |
| RuLSIF_Weight|VCS_Weight | 0.0016 | -0.048 | 0.000 | 0.1889 |
| Unweighted_A|VCS_Weight | 0.0000 | 1.000 | 1.000 | 0.0000 |

## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)

| method | n pool | mean | min | max | sha256 |
|---|---|---|---|---|---|
| BCE_Weight | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| Nystrom_Weight | 145000 | 1.000000 | 0.000000 | 1.001035 | c5d52ca3704e7ddd |
| PlainFilter | 145000 | 1.000000 | 0.000000 | 1.053994 | 4ca794100a4aa9c4 |
| RankMass | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| RuLSIF_Weight | 145000 | 1.000000 | 0.812283 | 1.001436 | ac960a048efd0566 |
| Unweighted_A | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| VCS_Weight | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |

Identical groups: [['BCE_Weight', 'RankMass', 'Unweighted_A', 'VCS_Weight'], ['Nystrom_Weight'], ['PlainFilter'], ['RuLSIF_Weight']]; merged (run once): ['RankMass', 'Unweighted_A', 'VCS_Weight'] — rule: merge only if identical coefficients AND same seed-derived order/RNG; otherwise both methods run

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

Noise: design rho 0.0, realized 0.0000, mapping sha256 3bdd4adc58fcb1b1, A mismatched 0
