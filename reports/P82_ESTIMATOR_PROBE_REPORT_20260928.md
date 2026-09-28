# P82 report — estimator package v1, first batch (stages 0–2), Gaussian d = 20, seed 0 (2026-09-28)

Pre-registration `P81_ESTIMATOR_PROBE_PREREG_FROZEN_20260928.md`; stage-0 gate CPU 1013081 (13/13 checks + 29 reference tests pass); probe GPU
1013086 (A100, exit 0, **60 s per condition**, 3 min total); table `P82_estim_probe.md`, raw JSON `P82_estim_probe_I{0.5,4,8}_d20_s0.json`
(results-only commit faed605).  Single seed: this is a probe, every reading below is provisional until the 3-seed replication.
The §8 checkpoint diagnostics are a separate unit (in preparation).

QC (all pass): oracle J on EVAL within 2.1 se of S at every I; oracle J on TRUTH within 1.1 se of S; oracle dS/dρ identical across δ ∈ {1e-2, 3e-3,
1e-3} and within 1.6 se of the envelope form; simplex KKT ≤ 1.4e-16, JS-mixture KKT ≤ 2e-11.

## Key numbers (EVAL posterior MSE = E_M (T − η)²; lower is better; J_eval se ≈ 0.002–0.003)
| I (S) | VCS C0 | VCS C1 | VCS C2 (SELECT-best single at I = 4, 8) | C2 best-of-3 restarts | **VCS mix** | JS mix | **VCS residual** | CQ (class ∋ oracle) VCS / JS |
|---|---|---|---|---|---|---|---|---|
| 0.5 (0.206) | **0.008** (best single) | 0.082 | 0.111 | 0.111 | 0.008 (w = C0) | 0.008 | 0.008 (λ = 0) | 0.0002 / 0.0003 |
| 4 (0.803) | 0.143 | 0.222 | 0.095 | 0.092 | **0.078** | 0.078 | **0.071** (λ = 0.58) | 0.021 / 0.014 |
| 8 (0.963) | 0.202 | 0.265 | 0.050 | 0.049 | 0.049 (w ≈ C2) | 0.046 | **0.039** (λ = 0.50) | 0.034 / 0.022 |

## 1. Where the error sits (spec §16 Q1)
- **Generalisation, for the flexible class.**  C2 over-fits FIT at every I: FIT J 0.395 / 0.904 / 0.997 vs S 0.206 / 0.803 / 0.963 (the FIT value
  *exceeds* the truth), SELECT picks it early (updates 200 / 300 / 500 of 2000), and its EVAL posterior MSE is the largest of all classes at I = 0.5.
  With 4 096 FIT pairs, a 76 k-parameter MLP is sample-limited, not optimisation-limited.  C1 (rank-16 bilinear) over-fits too (FIT J > S at
  I = 0.5) and is the worst class at I = 4 / 8 — it lacks the ‖x‖², ‖y‖² terms and fits noise in its 641 parameters.
- **C0: function class and budget, not separable here.**  C0 is near-oracle at weak dependence (MSE 0.008).  At I = 4 / 8 its class lacks the
  ‖x‖², ‖y‖² terms, but it also starts at a = 0 and is still improving at update 2 000 (selected at the last update), so part of its 0.14 / 0.20
  may be budget.  (Found while building §8: on unit-norm features the same start makes C0 unreadable; the frozen stage starts C0 at the SSL
  recipe's a0 = 5.)  The lr-grid item in §5 separates the two.
- **Optimisation, at strong dependence.**  CQ contains the oracle exactly (stage-0 test) yet keeps MSE 0.021 / 0.034 at I = 4 / 8, with
  SELECT still improving at update 2000: the budget (lr 5e-4 × 2 000 updates) is too short to reach the oracle's large weights.  JS gets there
  faster (0.014 / 0.022): near saturation the VCS logit gradient (1 − T)(1 − T²) decays faster than JS's 2σ(−2f).  The same holds for C0.
- **Saturation.**  At I = 8 the oracle's own gates are 0.045 / 0.030 (most pairs saturated), and excess-to-Bayes-risk is > 1 for every learned
  model (1 − S = 0.037 is a small denominator) — the regime where the absolute MSE, not the ratio, should be read.

## 2. What reduced the error (Q2)
- **Residual candidate: yes, at I = 4 and 8.**  EVAL MSE −25 % (0.095 → 0.071) and −22 % (0.050 → 0.039); J_eval +0.023 (≈ 8 se) and +0.011
  (≈ 6 se); better than the best-of-3 restart (0.092 / 0.049), so it is not a restart effect.  At I = 0.5 the TUNE step correctly returns λ = 0
  (Â < 0: the residual would hurt) and EVAL equals the base.  **Not yet separated from capacity**: the residual adds a second 76 k-parameter
  network (2× parameters and fitting cost of C2); the spec's §7.3 capacity / equal-cost control and a JS residual are required before this
  is attributed to the quadratic structure.
- **VCS dictionary mix: yes at I = 4 only** (0.095 → 0.078, J +0.016 ≈ 6 se; weights 0.38 C0 + 0.62 C2); at I = 0.5 it selects C0 alone
  (= the best single), at I = 8 it collapses onto C2.  The TUNE objective never falls below the best vertex (it cannot, by construction).
- **Mix gain is not VCS-specific**: the JS dictionary control gives the same EVAL MSE (0.078 at I = 4) and slightly better at I = 8 (0.046).

## 3. JS control with the same capacity (Q3)
Single-model VCS and matched-JS are within ≈ 1 se of each other on C0 / C1 / C2 at every I; JS is better on the classes limited by optimisation
(CQ, C0 at I = 4 / 8).  On this probe there is no estimator-side advantage of the quadratic (VCS) fit over JS; the only VCS-only mechanism
tested — the residual step with a closed-form λ — has no JS counterpart yet.

## 4. Channel-parameter gradients (§6.4)
With the critic fixed, dJ/dρ through the generator is biased for every learned class, in both directions: at I = 4, oracle 1.343 vs C2 1.698 (+26 %),
C0 0.792 (−41 %), CQ 1.188 (−12 %); at I = 8, oracle 0.539 vs C2 0.728 (+35 %), CQ 0.557 (+3 %).  The over-fitted C2 over-states the dependence
gradient; for any use of the estimator's gradient (SSL, channel optimisation) the posterior error matters beyond the value error.

## 5. What goes forward (Q4) — proposal, not launched
1. **Replicate + controls for the residual**: seeds 0–2; add a JS residual (same λ-free structure is not available for JS → JS residual trained on the
   mixture's log loss with λ fitted on TUNE by 1-D line search), a capacity control (single C2 of width 362 ≈ 2× parameters) and an equal-cost control
   (C2 with 4 000 updates, and the average of two C2 restarts).
2. **Budget sensitivity** (spec's pre-set grid lr ∈ {1e-4, 5e-4, 2e-3}) for every method equally — CQ/C0 are budget-limited at I = 4 / 8.
3. **Second-batch generators** (spec §6.3): d ∈ {2, 20, 50}, cubic transform, sign-mixture; FIT size as an explicit axis (the C2 error is sample-limited).
Cost (measured 4–7 s per fit on an A100): items 1–3 over 3 seeds ≈ 81 conditions × ~2 min ≈ **3 GPU-h**; observation-scale stage adds ≈ 1 GPU-h per σ grid.
