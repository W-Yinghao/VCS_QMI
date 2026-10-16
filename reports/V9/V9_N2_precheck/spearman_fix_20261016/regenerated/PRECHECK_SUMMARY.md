# V9-N2 precheck summary — f30k — REGENERATED 2026-10-16 with the tie-averaged Spearman (a rebuild, NOT the gate record; the record is reports/V9/V9_N2_precheck/PRECHECK_SUMMARY.md)

Job rebuild 2026-10-16 from the saved weights of precheck job 1042133 (not the gate record), code commit 480beff (tie-averaged Spearman; record code 127dd18), host login node, CPU (record: node58); thresholds: refuse if AUROC > 0.999 or < 0.55 (teacher clean vs within-role re-paired on A-critic-select); merge rule: loss coefficients identical at 1e-12 AND same data order / RNG state.

| rho | seed | allowed | AUROC | overlap | P med | Q med | rho-hat (U_est) | rho-hat independent [95 % Hoeffding] | merges | teacher s/epoch | peak GPU MB | teacher shared |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.4 | 901 | yes | 0.8837 | 0.382 | 0.713 | 0.089 | 0.3922 (point_estimate) | 0.4090 [0.195, 0.584] | none | 16.3 | 9503 | trained here |
| 0 | 901 | yes | 0.8837 | 0.382 | 0.713 | 0.089 | 0.0000 (outside_parameter_space) | 0.0000 [0.000, 0.208] | 5 | 16.3 | 9503 | rho0.4 |
| 0.4 | 902 | yes | 0.9032 | 0.348 | 0.798 | 0.119 | 0.4052 (point_estimate) | 0.4087 [0.227, 0.561] | none | 13.4 | 9401 | trained here |
| 0 | 902 | yes | 0.9032 | 0.348 | 0.798 | 0.119 | 0.0000 (outside_parameter_space) | 0.0037 [0.000, 0.188] | 3 | 13.4 | 9401 | rho0.4 |
| 0.4 | 903 | yes | 0.9112 | 0.328 | 0.777 | 0.135 | 0.3708 (point_estimate) | 0.3692 [0.185, 0.525] | none | 12.8 | 9601 | trained here |
| 0 | 903 | yes | 0.9112 | 0.328 | 0.777 | 0.135 | 0.0000 (outside_parameter_space) | 0.0000 [0.000, 0.143] | 5 | 12.8 | 9601 | rho0.4 |

## Weight summary per method (mean w, ESS, total mass, share at 0 / 1)

**rho 0.4, seed 901** (n_U pairs = 130500)

| method | mean w | ESS | mass | at 0 | at 1 |
|---|---|---|---|---|---|
| BCE_Weight | 0.6027 | 103555 | 78658 | 0.0000 | 0.0000 |
| Nystrom_Weight | 0.5978 | 101277 | 78015 | 0.0061 | 0.0000 |
| PlainFilter | 0.7575 | 98848 | 98848 | 0.2425 | 0.7575 |
| RankMass | 0.6119 | 79849 | 79849 | 0.3881 | 0.6119 |
| RuLSIF_Weight | 0.5979 | 101411 | 78026 | 0.0000 | 0.0000 |
| Unweighted_A | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| VCS_Weight | 0.6119 | 105233 | 79849 | 0.0000 | 0.0000 |

near-monotone pairs (Spearman > 0.99): BCE_Weight|Nystrom_Weight: mean|dw| 0.05044, rho_s 1.0, identical 2.3e-05 -> distinct runs; BCE_Weight|RuLSIF_Weight: mean|dw| 0.04922, rho_s 0.998273, identical 0.0 -> distinct runs; BCE_Weight|VCS_Weight: mean|dw| 0.00941, rho_s 1.0, identical 0.00023 -> distinct runs; Nystrom_Weight|RuLSIF_Weight: mean|dw| 0.00853, rho_s 0.998273, identical 4.6e-05 -> distinct runs; Nystrom_Weight|VCS_Weight: mean|dw| 0.04863, rho_s 1.0, identical 2.3e-05 -> distinct runs; RuLSIF_Weight|VCS_Weight: mean|dw| 0.04796, rho_s 0.998273, identical 8e-06 -> distinct runs

Spearman undefined (NA): BCE_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Nystrom_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; PlainFilter|Unweighted_A: constant_input, identical 0.757456 -> distinct runs; RankMass|Unweighted_A: constant_input, identical 0.611862 -> distinct runs; RuLSIF_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Unweighted_A|VCS_Weight: constant_input, identical 0.0 -> distinct runs

**rho 0, seed 901** (n_U pairs = 130500)

| method | mean w | ESS | mass | at 0 | at 1 |
|---|---|---|---|---|---|
| BCE_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| Nystrom_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| PlainFilter | 0.9549 | 124618 | 124618 | 0.0451 | 0.9549 |
| RankMass | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| RuLSIF_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| Unweighted_A | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| VCS_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |

near-monotone pairs (Spearman > 0.99): none

Spearman undefined (NA): BCE_Weight|Nystrom_Weight: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|PlainFilter: constant_input, identical 0.954927 -> distinct runs; BCE_Weight|RankMass: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|RuLSIF_Weight: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|PlainFilter: constant_input, identical 0.954927 -> distinct runs; Nystrom_Weight|RankMass: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|RuLSIF_Weight: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); PlainFilter|RankMass: constant_input, identical 0.954927 -> distinct runs; PlainFilter|RuLSIF_Weight: constant_input, identical 0.954927 -> distinct runs; PlainFilter|Unweighted_A: constant_input, identical 0.954927 -> distinct runs; PlainFilter|VCS_Weight: constant_input, identical 0.954927 -> distinct runs; RankMass|RuLSIF_Weight: constant_input, identical 1.0 -> MERGED (one run); RankMass|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); RankMass|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); RuLSIF_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); RuLSIF_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); Unweighted_A|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run)

**rho 0.4, seed 902** (n_U pairs = 130500)

| method | mean w | ESS | mass | at 0 | at 1 |
|---|---|---|---|---|---|
| BCE_Weight | 0.5976 | 100014 | 77990 | 0.0000 | 0.0000 |
| Nystrom_Weight | 0.5929 | 97197 | 77369 | 0.0214 | 0.0000 |
| PlainFilter | 0.7100 | 92654 | 92654 | 0.2900 | 0.7100 |
| RankMass | 0.6068 | 79192 | 79192 | 0.3932 | 0.6068 |
| RuLSIF_Weight | 0.5927 | 97242 | 77351 | 0.0000 | 0.0000 |
| Unweighted_A | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| VCS_Weight | 0.6068 | 100854 | 79192 | 0.0000 | 0.0000 |

near-monotone pairs (Spearman > 0.99): BCE_Weight|Nystrom_Weight: mean|dw| 0.03968, rho_s 0.999995, identical 2.3e-05 -> distinct runs; BCE_Weight|RuLSIF_Weight: mean|dw| 0.04053, rho_s 0.996423, identical 8e-06 -> distinct runs; BCE_Weight|VCS_Weight: mean|dw| 0.00921, rho_s 1.0, identical 0.0 -> distinct runs; Nystrom_Weight|RuLSIF_Weight: mean|dw| 0.00621, rho_s 0.996419, identical 0.000736 -> distinct runs; Nystrom_Weight|VCS_Weight: mean|dw| 0.03828, rho_s 0.999995, identical 2.3e-05 -> distinct runs; RuLSIF_Weight|VCS_Weight: mean|dw| 0.0393, rho_s 0.996423, identical 8e-06 -> distinct runs

Spearman undefined (NA): BCE_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Nystrom_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; PlainFilter|Unweighted_A: constant_input, identical 0.709992 -> distinct runs; RankMass|Unweighted_A: constant_input, identical 0.606835 -> distinct runs; RuLSIF_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Unweighted_A|VCS_Weight: constant_input, identical 0.0 -> distinct runs

**rho 0, seed 902** (n_U pairs = 130500)

| method | mean w | ESS | mass | at 0 | at 1 |
|---|---|---|---|---|---|
| BCE_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| Nystrom_Weight | 0.9989 | 130488 | 130350 | 0.0000 | 0.0000 |
| PlainFilter | 0.9431 | 123072 | 123072 | 0.0569 | 0.9431 |
| RankMass | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| RuLSIF_Weight | 0.9984 | 130496 | 130292 | 0.0000 | 0.0000 |
| Unweighted_A | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| VCS_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |

near-monotone pairs (Spearman > 0.99): none

Spearman undefined (NA): BCE_Weight|Nystrom_Weight: constant_input, identical 0.0 -> distinct runs; BCE_Weight|PlainFilter: constant_input, identical 0.94308 -> distinct runs; BCE_Weight|RankMass: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|RuLSIF_Weight: constant_input, identical 0.0 -> distinct runs; BCE_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|RankMass: constant_input, identical 0.0 -> distinct runs; Nystrom_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Nystrom_Weight|VCS_Weight: constant_input, identical 0.0 -> distinct runs; PlainFilter|RankMass: constant_input, identical 0.94308 -> distinct runs; PlainFilter|Unweighted_A: constant_input, identical 0.94308 -> distinct runs; PlainFilter|VCS_Weight: constant_input, identical 0.94308 -> distinct runs; RankMass|RuLSIF_Weight: constant_input, identical 0.0 -> distinct runs; RankMass|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); RankMass|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); RuLSIF_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; RuLSIF_Weight|VCS_Weight: constant_input, identical 0.0 -> distinct runs; Unweighted_A|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run)

**rho 0.4, seed 903** (n_U pairs = 130500)

| method | mean w | ESS | mass | at 0 | at 1 |
|---|---|---|---|---|---|
| BCE_Weight | 0.6201 | 101116 | 80929 | 0.0000 | 0.0000 |
| Nystrom_Weight | 0.6129 | 98137 | 79989 | 0.0000 | 0.0000 |
| PlainFilter | 0.7242 | 94513 | 94513 | 0.2758 | 0.7242 |
| RankMass | 0.6233 | 81345 | 81344 | 0.3767 | 0.6233 |
| RuLSIF_Weight | 0.6118 | 98167 | 79846 | 0.0000 | 0.0000 |
| Unweighted_A | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| VCS_Weight | 0.6233 | 101416 | 81344 | 0.0000 | 0.0000 |

near-monotone pairs (Spearman > 0.99): BCE_Weight|Nystrom_Weight: mean|dw| 0.03275, rho_s 1.0, identical 8e-06 -> distinct runs; BCE_Weight|VCS_Weight: mean|dw| 0.00318, rho_s 1.0, identical 0.0 -> distinct runs; Nystrom_Weight|VCS_Weight: mean|dw| 0.03258, rho_s 1.0, identical 2.3e-05 -> distinct runs

Spearman undefined (NA): BCE_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Nystrom_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; PlainFilter|Unweighted_A: constant_input, identical 0.724238 -> distinct runs; RankMass|Unweighted_A: constant_input, identical 0.623326 -> distinct runs; RuLSIF_Weight|Unweighted_A: constant_input, identical 0.0 -> distinct runs; Unweighted_A|VCS_Weight: constant_input, identical 0.0 -> distinct runs

**rho 0, seed 903** (n_U pairs = 130500)

| method | mean w | ESS | mass | at 0 | at 1 |
|---|---|---|---|---|---|
| BCE_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| Nystrom_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| PlainFilter | 0.9542 | 124525 | 124525 | 0.0458 | 0.9542 |
| RankMass | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| RuLSIF_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| Unweighted_A | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |
| VCS_Weight | 1.0000 | 130500 | 130500 | 0.0000 | 1.0000 |

near-monotone pairs (Spearman > 0.99): none

Spearman undefined (NA): BCE_Weight|Nystrom_Weight: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|PlainFilter: constant_input, identical 0.954215 -> distinct runs; BCE_Weight|RankMass: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|RuLSIF_Weight: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); BCE_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|PlainFilter: constant_input, identical 0.954215 -> distinct runs; Nystrom_Weight|RankMass: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|RuLSIF_Weight: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); Nystrom_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); PlainFilter|RankMass: constant_input, identical 0.954215 -> distinct runs; PlainFilter|RuLSIF_Weight: constant_input, identical 0.954215 -> distinct runs; PlainFilter|Unweighted_A: constant_input, identical 0.954215 -> distinct runs; PlainFilter|VCS_Weight: constant_input, identical 0.954215 -> distinct runs; RankMass|RuLSIF_Weight: constant_input, identical 1.0 -> MERGED (one run); RankMass|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); RankMass|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); RuLSIF_Weight|Unweighted_A: constant_input, identical 1.0 -> MERGED (one run); RuLSIF_Weight|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run); Unweighted_A|VCS_Weight: constant_input, identical 1.0 -> MERGED (one run)

## Reading

- The teacher is one per (dataset, seed): the rho 0 cell reuses the rho 0.4 teacher (identical roles hashes, clean anchor pairs), so AUROC is identical across rho within a seed by construction.
- AUROC far from both refusal thresholds in every cell: the teacher separates clean from re-paired anchor pairs but is not a label oracle.
- rho-hat on the held-out 20 % of U images (split by image) is a held-out-image diagnostic interval, not an independent-sample guarantee (overlap audit 2026-10-16); the U_est point estimate is what the weights use.
- Merged by the coefficient rule (one training per seed, reported for every member): rho 0 s901: {BCE_Weight, Nystrom_Weight, RankMass, RuLSIF_Weight, Unweighted_A, VCS_Weight}; rho 0 s902: {BCE_Weight, RankMass, Unweighted_A, VCS_Weight}; rho 0 s903: {BCE_Weight, Nystrom_Weight, RankMass, RuLSIF_Weight, Unweighted_A, VCS_Weight}; no merge in rho 0.4 s901, rho 0.4 s902, rho 0.4 s903.
- Teacher cost is the first real GPU number: 5 warm-up epochs on the A-teacher-warmup pairs (4 % of the training pairs).
