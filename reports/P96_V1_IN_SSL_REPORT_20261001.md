# P96 — v1 estimator improvements inside full CIFAR-10 SSL at 8× (P95) — 2026-10-01

Pre-registration `P95_V1_IN_SSL_PREREG_FROZEN_20260929.md`; stage-B selection addenda part 1 (43c87cc) and part 2 (9b79bf4); results-only commit
(`P95_v1_in_ssl_table.md`, 18 runs).  Owner go 2026-09-29 ("我建议都放进ssl里试试，这种probe还是不能代替全量实验。就在cifar10尝试").  Protocol: the 8× recipe
(4 views, B 256, 800 epochs), selection split, frozen-h linear + kNN at epoch 800.  Recipe 87.01 ± 0.53 (86.42 / 87.16 / 87.44), kNN 85.46 ± 0.12.

## 1. Stage A (seed 0, all cells)
| family | cells: linear / kNN | selected |
|---|---|---|
| residual (1−λ)T_cos + λ tanh(MLP) | λ 0.25: 86.02 / 80.90; λ 0.5: 85.74 / 83.54 | λ 0.25 |
| dictionary (cosine, concat-MLP, bilinear; simplex weights) | 86.56 / 83.20 | — |
| observation noise at the critic input | τ 0.1: 86.52 / 85.42; τ 0.3: 87.32 / 84.92 | τ 0.3 |
| periodic refresh of the cosine scalars | R 50: 84.96 / 82.14; R 200: 85.94 / 84.38 | R 200 |
| matched-JS control (same critic, balanced logistic loss) | 86.20 / — | always seeded |

## 2. Stage B and the pre-stated verdicts (3 seeds; pooled SE = √(sd²/3 + 0.53²/3))
| family | seeds 0 / 1 / 2 (linear) | mean ± sd | Δ vs recipe | 2 × pooled SE | verdict | kNN mean ± sd |
|---|---|---|---|---|---|---|
| residual λ 0.25 | 86.02 / 86.06 / 86.32 | 86.13 ± 0.16 | −0.87 | 0.64 | **hurts** | 81.01 ± 0.22 |
| dictionary | 86.56 / 84.60 / 86.20 | 85.79 ± 1.04 | −1.22 | 1.35 | **on par** (seed spread) | 83.17 ± 0.52 |
| noise τ 0.3 | 87.32 / 87.12 / 87.02 | 87.15 ± 0.15 | **+0.15** | 0.63 | **on par** | 84.91 ± 0.08 |
| refresh R 200 | 85.94 / 85.78 / 86.42 | 86.05 ± 0.33 | −0.96 | 0.72 | **hurts** | 84.43 ± 0.05 |
| JS control | 86.20 / 86.62 / 86.68 | 86.50 ± 0.26 | −0.51 | 0.68 | on par | 84.89 ± 0.25 |

**No v1 improvement improves the recipe in full training.**  Observation noise τ 0.3 is the only family at or above the recipe (+0.15, three seeds within
0.3 of each other, the lowest seed 87.02 above the recipe's worst 86.42) but well inside the "on par" band; its kNN is 0.55 below.  The residual and refresh
families hurt; the dictionary is on par only because of its seed spread (one seed at 84.60).  The matched-JS control is 0.51 below the recipe (on par) —
in full SSL the quadratic J is not worse than the matched logistic loss on the same critic, and not measurably better.

## 3. Mechanism notes (trainer logs, descriptive)
- Residual and dictionary critics add a flexible MLP read-out; like S-Kernel in P88 they let the critic separate P from Q without the encoder spreading h
  (kNN −4.5 and −2.3), the pattern already seen with flexible critics in the recipe search.
- Refresh raises the batch J at each refit but the encoder then trains against a steeper critic (P84 addendum: converged cosine has a 3–7× smaller
  representation gradient); both R 50 and R 200 end below the recipe.
- Noise τ 0.3 changes the estimand to S_σ and keeps positives off saturation; it matches the recipe's linear accuracy with the smallest seed spread of all
  families.  Whether averaging the noisy loss over draws helps is the v3 N1 question (P104), not answered here.

## 4. Consequences
- The conditional follow-up P103 (CIFAR-100 confirmation of an improving family) is **not triggered**: no family meets "improves".  The orchestrator row for
  P103 stays inert (no prereg will be frozen).
- For the paper: the v1 package's estimator-side ideas (bounded residual, dictionary, online refresh) do not transfer into better SSL representations at
  this budget; observation noise at τ 0.3 is neutral.  These are negative results to report as such.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: full_ssl
protocol_id: P95_v1_in_ssl
source_commit: 97c062b (frozen), 43c87cc / 9b79bf4 (selection); results this commit's parent
estimator: vcs_neural (residual | dictionary | noisy | refresh) | js_matched (control)
estimand: S (S_sigma for noise)
evaluation_readout: frozen-h linear + kNN, selection split, epoch 800
n_independent_units: 45000 fit images; stage A 8 cells x 1 seed, stage B 5 families x 2 more seeds
split_manifest_hash: cifar10_dev45k_val5k (f819026a...)
status: complete
```
