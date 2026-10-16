# V9-N2 precheck — f30k, rho 0, seed 917 — formal block ALLOWED

## 1. Teacher scores on A-critic-select (P clean n=2175, Q re-paired n=2175): AUROC 0.9035 (thresholds > 0.999 / < 0.55 refuse), overlap 0.346, share of P below Q median 0.024

| set | q5 | q25 | q50 | q75 | q95 | mean |
|---|---|---|---|---|---|---|
| P | 0.2673 | 0.6459 | 0.8349 | 0.9304 | 0.9651 | 0.7557 |
| Q | 0.0208 | 0.0489 | 0.1694 | 0.4486 | 0.8131 | 0.2719 |

## 2. Weights on U per method

| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |
|---|---|---|---|---|---|---|---|---|---|
| BCE_Weight | 130500 | 0.958 | 0.998 | 0.999 | 0.992 | 130458.2 | 129445.4 | 0.000 | 0.000 |
| Nystrom_Weight | 130500 | 0.966 | 0.998 | 0.999 | 0.992 | 130410.4 | 129392.2 | 0.000 | 0.000 |
| PlainFilter | 130500 | 0.000 | 1.000 | 1.000 | 0.948 | 123762.0 | 123762.0 | 0.052 | 0.948 |
| RankMass | 130500 | 1.000 | 1.000 | 1.000 | 0.992 | 129429.8 | 129429.6 | 0.008 | 0.992 |
| RuLSIF_Weight | 130500 | 0.966 | 0.998 | 0.999 | 0.992 | 130415.3 | 129421.9 | 0.000 | 0.000 |
| Unweighted_A | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| VCS_Weight | 130500 | 0.957 | 0.998 | 0.999 | 0.992 | 130456.7 | 129429.6 | 0.000 | 0.000 |

| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |
|---|---|---|---|---|
| BCE_Weight|Nystrom_Weight | 0.0020 | 1.000 | 0.001 | 0.3244 |
| BCE_Weight|PlainFilter | 0.0521 | 0.119 | 0.000 | 0.9587 |
| BCE_Weight|RankMass | 0.0143 | -0.008 | 0.000 | 0.8959 |
| BCE_Weight|RuLSIF_Weight | 0.0019 | 0.948 | 0.001 | 0.4870 |
| BCE_Weight|Unweighted_A | 0.0081 | -0.034 | 0.000 | 0.1464 |
| BCE_Weight|VCS_Weight | 0.0001 | 1.000 | 0.000 | 0.0027 |
| Nystrom_Weight|PlainFilter | 0.0513 | 0.119 | 0.000 | 0.9666 |
| Nystrom_Weight|RankMass | 0.0128 | -0.008 | 0.000 | 0.8772 |
| Nystrom_Weight|RuLSIF_Weight | 0.0003 | 0.948 | 0.003 | 0.1626 |
| Nystrom_Weight|Unweighted_A | 0.0085 | -0.034 | 0.000 | 0.4708 |
| Nystrom_Weight|VCS_Weight | 0.0020 | 1.000 | 0.002 | 0.3218 |
| PlainFilter|RankMass | 0.0434 | 0.914 | 0.957 | 1.0000 |
| PlainFilter|RuLSIF_Weight | 0.0513 | 0.119 | 0.000 | 0.9671 |
| PlainFilter|Unweighted_A | 0.0516 | 0.898 | 0.948 | 1.0000 |
| PlainFilter|VCS_Weight | 0.0521 | 0.119 | 0.000 | 0.9581 |
| RankMass|RuLSIF_Weight | 0.0127 | -0.008 | 0.000 | 0.8760 |
| RankMass|Unweighted_A | 0.0082 | 0.983 | 0.992 | 1.0000 |
| RankMass|VCS_Weight | 0.0144 | -0.008 | 0.000 | 0.8940 |
| RuLSIF_Weight|Unweighted_A | 0.0083 | -0.033 | 0.000 | 0.6334 |
| RuLSIF_Weight|VCS_Weight | 0.0020 | 0.948 | 0.001 | 0.4843 |
| Unweighted_A|VCS_Weight | 0.0082 | -0.034 | 0.000 | 0.1491 |

## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)

| method | n pool | mean | min | max | sha256 |
|---|---|---|---|---|---|
| BCE_Weight | 145000 | 1.000000 | 0.859852 | 1.007326 | d259afa9665cc259 |
| Nystrom_Weight | 145000 | 1.000000 | 0.533235 | 1.007699 | cf3dab225cc8d704 |
| PlainFilter | 145000 | 1.000000 | 0.000000 | 1.048734 | a1f5e2107beb3053 |
| RankMass | 145000 | 1.000000 | 0.000000 | 1.007437 | 4b8f3aa2b03485ef |
| RuLSIF_Weight | 145000 | 1.000000 | 0.369350 | 1.007491 | 1f7631c1ba0c03c2 |
| Unweighted_A | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| VCS_Weight | 145000 | 1.000000 | 0.857265 | 1.007437 | 7edf502db185366b |

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

Noise: design rho 0.0, realized 0.0000, mapping sha256 3bdd4adc58fcb1b1, A mismatched 0
