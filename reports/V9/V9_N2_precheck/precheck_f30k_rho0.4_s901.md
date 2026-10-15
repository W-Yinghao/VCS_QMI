# V9-N2 precheck — f30k, rho 0.4, seed 901 — formal block ALLOWED

## 1. Teacher scores on A-critic-select (P clean n=2175, Q re-paired n=2175): AUROC 0.8837 (thresholds > 0.999 / < 0.55 refuse), overlap 0.382, share of P below Q median 0.037

| set | q5 | q25 | q50 | q75 | q95 | mean |
|---|---|---|---|---|---|---|
| P | 0.1144 | 0.4466 | 0.7129 | 0.8803 | 0.9491 | 0.6406 |
| Q | 0.0137 | 0.0312 | 0.0888 | 0.2845 | 0.7401 | 0.1984 |

## 2. Weights on U per method

| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |
|---|---|---|---|---|---|---|---|---|---|
| BCE_Weight | 130500 | 0.172 | 0.674 | 0.955 | 0.603 | 103554.9 | 78658.2 | 0.000 | 0.000 |
| Nystrom_Weight | 130500 | 0.033 | 0.705 | 0.972 | 0.598 | 101276.7 | 78015.4 | 0.006 | 0.000 |
| PlainFilter | 130500 | 0.000 | 1.000 | 1.000 | 0.757 | 98848.0 | 98848.0 | 0.243 | 0.757 |
| RankMass | 130500 | 0.000 | 1.000 | 1.000 | 0.612 | 79849.0 | 79848.9 | 0.388 | 0.612 |
| RuLSIF_Weight | 130500 | 0.047 | 0.702 | 0.967 | 0.598 | 101411.4 | 78026.2 | 0.000 | 0.000 |
| Unweighted_A | 130500 | 1.000 | 1.000 | 1.000 | 1.000 | 130500.0 | 130500.0 | 0.000 | 1.000 |
| VCS_Weight | 130500 | 0.188 | 0.684 | 0.955 | 0.612 | 105233.1 | 79848.9 | 0.000 | 0.000 |

| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |
|---|---|---|---|---|
| BCE_Weight|Nystrom_Weight | 0.0504 | 1.000 | 0.000 | 0.1652 |
| BCE_Weight|PlainFilter | 0.2481 | 0.537 | 0.000 | 0.7493 |
| BCE_Weight|RankMass | 0.2026 | 0.709 | 0.000 | 0.5372 |
| BCE_Weight|RuLSIF_Weight | 0.0492 | 0.998 | 0.000 | 0.1518 |
| BCE_Weight|Unweighted_A | 0.3973 | -0.008 | 0.000 | 0.8380 |
| BCE_Weight|VCS_Weight | 0.0094 | 1.000 | 0.000 | 0.0192 |
| Nystrom_Weight|PlainFilter | 0.2187 | 0.537 | 0.006 | 0.6801 |
| Nystrom_Weight|RankMass | 0.1997 | 0.709 | 0.006 | 0.5561 |
| Nystrom_Weight|RuLSIF_Weight | 0.0085 | 0.998 | 0.000 | 0.0481 |
| Nystrom_Weight|Unweighted_A | 0.4022 | -0.008 | 0.000 | 1.0000 |
| Nystrom_Weight|VCS_Weight | 0.0486 | 1.000 | 0.000 | 0.1810 |
| PlainFilter|RankMass | 0.1456 | 0.790 | 0.854 | 1.0000 |
| PlainFilter|RuLSIF_Weight | 0.2180 | 0.537 | 0.000 | 0.6905 |
| PlainFilter|Unweighted_A | 0.2425 | 0.641 | 0.757 | 1.0000 |
| PlainFilter|VCS_Weight | 0.2471 | 0.537 | 0.000 | 0.7308 |
| RankMass|RuLSIF_Weight | 0.1998 | 0.709 | 0.000 | 0.5438 |
| RankMass|Unweighted_A | 0.3881 | 0.520 | 0.612 | 1.0000 |
| RankMass|VCS_Weight | 0.2071 | 0.709 | 0.000 | 0.5198 |
| RuLSIF_Weight|Unweighted_A | 0.4021 | -0.008 | 0.000 | 0.9898 |
| RuLSIF_Weight|VCS_Weight | 0.0480 | 0.998 | 0.000 | 0.1675 |
| Unweighted_A|VCS_Weight | 0.3881 | -0.008 | 0.000 | 0.8223 |

## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)

| method | n pool | mean | min | max | sha256 |
|---|---|---|---|---|---|
| BCE_Weight | 145000 | 1.000000 | 0.252183 | 1.556492 | 5ed70304c368a34d |
| Nystrom_Weight | 145000 | 1.000000 | 0.000000 | 1.567306 | e9d9f3e1ec861d66 |
| PlainFilter | 145000 | 1.000000 | 0.000000 | 1.279246 | 29cd42fab6dfcf10 |
| RankMass | 145000 | 1.000000 | 0.000000 | 1.536849 | bdb4ce44291e763a |
| RuLSIF_Weight | 145000 | 1.000000 | 0.016002 | 1.567124 | 50dff6b61e10619a |
| Unweighted_A | 145000 | 1.000000 | 1.000000 | 1.000000 | 5437dcbbb5d0b435 |
| VCS_Weight | 145000 | 1.000000 | 0.273084 | 1.536849 | f4fc3026721dc72c |

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

Noise: design rho 0.4, realized 0.4000, mapping sha256 783ae9337cd27317, A mismatched 0
