# P84 addendum 1 — report: actual gate, dictionary disagreement bounds, converged cosine, ridge member and Jacobians on the P84 checkpoints — 2026-09-28

Pre-registration `P83_ADDENDUM1_P84_SUPPLEMENT_FROZEN_20260928.md`; results `P84_addendum1_supplement.md` (job 1013257, commit 256a5d9) on the ten
P84 checkpoints with P84's roles and seeds (members re-fitted because P84 stored no TUNE outputs; the refits reproduce P84's EVAL J to the reported digits).

## 1. The actual gate is 4–6× what 1 − J suggests
| VCS 4-view recipe, epoch | training-critic J | actual gate E_M(1 − T²) | 1 − J | ratio |
|---|---|---|---|---|
| 100 | 0.948 | 0.204 | 0.052 | 3.9 |
| 400 | 0.981 | 0.112 | 0.019 | 5.8 |
| 800 | 0.986 | 0.085 | 0.014 | 6.1 |
| P5 original recipe, 200 | 0.901 | 0.095 | 0.099 | 1.0 |

The average learning gate of the recipe's critic is 0.085 at the end of training, not 0.014: unsaturated pairs (mostly the hard negatives and the harder
positives, cf. P94 D2) keep the gradient alive.  The supplement's "4.7× gate change" computed from 1 − S is therefore not a measurement of gates; the
technical note reports the measured column.

## 2. Dictionary disagreement: the bound is loose, the realised gain is nil
| representation | max_w D(w) on TUNE (P84 dictionary) | U_dict | D at P84's weights | extended-dictionary mix: EVAL gain over best single member |
|---|---|---|---|---|
| VCS ep 100 / 400 / 800 | 0.046 / 0.049 / 0.050 | 0.054 / 0.063 / 0.065 | 0.0029 / 0.0007 / 0.0000 (vertex) | +0.0011 / −0.0000 / −0.0000 |
| SimCLR ep 100 / 400 / 800 | 0.028 / 0.025 / 0.025 | 0.035 / 0.033 / 0.032 | 0.0073 / 0 / 0 | +0.0008 / −0.0003 / −0.0007 |
| P5 ep 200 | 0.065 | 0.077 | 0.0065 | +0.0007 |

The pre-stated closure criterion ("max D ≤ 2 block SE ≈ 0.005") is **not** met: the members disagree enough that a weight vector could exceed the weighted
member average by 0.03–0.06 J.  The bound is loose because the weights that maximise D put mass on the weaker members; the exact simplex QP on TUNE, which
already accounts for that, chose a vertex at epochs 400 and 800 and gains ≤ 0.0011 on EVAL with the extended dictionary (converged cosine and tanh-ridge added).
No member satisfies "worth a seed" (EVAL gain > 2 block SE and D_TUNE ≥ gain).  Reading: the combination family is closed for these representations by the
realised solutions (P84 + this supplement), not by the dictionary bound; the auxiliary C line asks for no compute.

## 3. Converged cosine: steeper, with a smaller representation gradient
| VCS epoch | training a, b | C0_conv a_eff, b_eff | J: training / C0_conv / tanh-ridge | ‖∂T/∂z‖: training / C0_conv / ridge |
|---|---|---|---|---|
| 100 | 9.9, −8.0 | 27.2, −22.5 | 0.948 / 0.969 / 0.946 | 2.35 / 0.90 / 0.39 |
| 400 | 21.6, −19.4 | 65.4, −59.2 | 0.981 / 0.989 / 0.966 | 3.28 / 0.54 / 0.26 |
| 800 | 24.4, −22.0 | 81.4, −74.3 | 0.986 / 0.9935 / 0.968 | 2.84 / 0.41 / 0.16 |

The converged two-parameter cosine is 2.7–3.3× steeper than the training critic and raises EVAL J by 0.008–0.021, but its mean representation gradient
is 0.14–0.38× the training critic's: the extra steepness saturates more pairs.  The pre-stated "steeper, not better" label does not trigger by its letter
(it required a *larger* Jacobian); the caution it was meant to carry is stronger in this form — refitting the critic to convergence at a fixed representation
would cut the encoder's learning signal by 3–7× — and goes into the technical note as the reason not to refresh the critic online.  The tanh-wrapped ridge
member is below the training critic in J on every trained checkpoint (raw |T| > 1 on 23–25 % of pairs before wrapping).  SimCLR representations converge to a
far shallower cosine (a_eff 5.4–5.9), consistent with their wider positive-pair cosine distribution.

## Delivery (Spec v2 §13.2)
```yaml
experiment_family: diagnostic
protocol_id: P83_addendum1_P84_supplement
source_commit: 36771f5 (code, frozen); results 256a5d9
source_basis: supplement_af7172c_plus_v2
estimator: vcs_neural members (C0, C1, C2, converged C0, tanh-ridge; raw ridge listed separately)
estimand: J on frozen representations (no oracle S)
evaluation_readout: J_common on EVAL blocks; D(w), max D, U_dict on TUNE; Jacobians on EVAL rows
n_independent_units: 10 checkpoints x (27k / 6k / 6k / 6k identity roles, seed 20260928)
status: complete
```
