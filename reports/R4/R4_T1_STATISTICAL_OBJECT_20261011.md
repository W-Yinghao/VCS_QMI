# R4-T1 — statistical object and assumptions, frozen before R4-T2 — 2026-10-11

Owner go-ahead, 2026-10-11: "都开始做，做完整" (start R4-02, R4-T1/T2 and R4-E1 and do them completely).  Package r4 docs/02 §R4-T1, docs/03 §1, §4–6.
Everything below is a **standard** derivation instantiated for this problem; none of it is claimed as new (claim type: standard instantiation check).

## 1. Object
- **Fixed critic.**  Representation, critic and preprocessing are fixed outside EVAL.  Target: J(T) = E_P a₊(T) + E_Q a₋(T), with
  a₊(t) = t − t²/2 and a₋(t) = −t − t²/2.  The optimum is S = E_M η² with M = (P + Q)/2 and η = (p − q)/(p + q).  J(T) ≤ S, and
  S − J(T) = E_M (T − η)².
- **Cross-pair regime (docs/03 §4.2).**  W_i = (X_i, Y_i) iid ~ P; negatives (X_i, Y_j), i ≠ j, follow Q = P_X ⊗ P_Y.  Ĵ is the second-order
  U-statistic with the symmetric kernel k_T.
- **Image-local regime (§4.1).**  Images G iid; P_G / Q_G exact within each image (here: uniform on matched / all index pairs).  Target
  μ_G(T) = E_G J_G(T); the image-level optimum E_G S_G bounds it from above.
- **Four sources of variability, kept apart:**
  1. independent base-observation / image sampling (EVAL);
  2. negative-pair sampling given the base observations (sparse pairs);
  3. critic initialisation and learning-data variation (FIT and optimiser seeds; stage 3 only);
  4. the bias J(T) vs S (a gap, not a variance).
- **One independent FIT / CAL vs EVAL split.**  No K-fold aggregation treated as iid.

## 2. Hypotheses checked by R4-T2
| id | statement | type |
|---|---|---|
| H1 | Var(Ĵ) = 4ζ₁/n + 2E h₂²/(n(n − 1)) matches the Monte Carlo variance of Ĵ (implementation check) | theory → implementation |
| H2 | the pair-iid SE (all n(n − 1) negatives treated as independent, positives independent of negatives) mis-covers J(T) when shared base observations matter | negative control |
| H3 | the delete-one jackknife of the U-statistic covers J(T) near nominal for non-degenerate cells, improving with n; degenerate cells are flagged, normal approximation not assumed | exploratory prediction |
| H4 | bounded-difference √(32 log(2/δ)/n) and block / image Hoeffding radii cover at ≥ nominal; their width is reported | theory |
| H5 | sparse cyclic shifts: conditional variance (1 − K/(n − 1)) s_v²/K is exact; total variance = Var(Ĵ_all) + E conditional variance; cost is reported | theory → implementation |
| H6 | image-local: the image-cluster interval covers μ_G near nominal; pooled pair-iid mis-covers; the image Hoeffding radius is valid | exploratory prediction / theory |
| H7 | a lower confidence bound for J(T) is a valid one-sided lower bound for S (J ≤ S); an interval for J(T) is not a two-sided interval for S | theory |
| H8 | (stage 3) fixed-critic intervals condition on the trained critic; across training seeds Var J(T_s) adds a component they do not cover | exploratory prediction |

## 3. Assumption — derivation — implementation — literature
| assumption | derivation | implementation (scripts/r4_t2_sampling.py) | literature |
|---|---|---|---|
| W_i iid; T fixed and independent of EVAL | Ĵ = C(n, 2)⁻¹ Σ_{i<j} k_T(W_i, W_j) | `cross_readouts` (full kernel); unit test of the identity | Hoeffding (1948) |
| same | Var Ĵ = 4ζ₁/n + 2E h₂²/(n(n − 1)) | `exact_hoeffding` (finite-state enumeration) vs Monte Carlo; unit test by exhaustive enumeration at n = 3 | Hoeffding (1948); Serfling (1980) §5.2; Lee (1990) |
| non-degenerate kernel (ζ₁ > 0) | delete-one jackknife is consistent for Var Ĵ | `cross_readouts` (O(n) deletions from row / column sums); brute-force unit test | Arvesen (1969); Efron & Stein (1981) |
| ζ₁ = 0 (independence with a constant critic) | n(Ĵ − μ) has a weighted-χ² limit, or Ĵ is constant; the normal approximation is not assumed | degenerate cells flagged | Serfling (1980) §5.5 |
| k_T ∈ [−3, 1] | one W_i changes Ĵ by ≤ 8/n → McDiarmid radius √(32 log(2/δ)/n) | `r_bd`; unit test of the 8/n constant | McDiarmid (1989) |
| disjoint pairs (W₂ₜ₋₁, W₂ₜ) independent | iid bounded means: t-interval; Hoeffding √(8 log(2/δ)/m), m = n/2 | `blocks` | Hoeffding (1963) |
| images iid; J_G ∈ [−3, 1] | Hoeffding √(8 log(2/δ)/N); image-cluster t / bootstrap | `image_local` | Hoeffding (1963); Davison & Hinkley (1997) |
| shifts uniform without replacement | finite-population variance (1 − K/(n − 1)) s_v²/K; law of total variance | `sparse` (formula vs resampling; incomplete-design jackknife, exploratory) | Cochran (1977); incomplete U-statistics: Blom (1976), Janson (1984); repo P138 |
| any T in [−1, 1] | S − J(T) = E_M (T − η)² (completed square; RPC with α = β = γ = 1) | unit test | Tsai et al. (2021, RPC); package docs/03 §1–2 |

## 4. Not claimed
- No new interval method.
- No two-sided interval for S.
- No statement that a fixed-critic interval covers the training pipeline.
- No K-fold scores treated as iid.
