# Pre-registration — P116: v5 NEXT-E-SOLVE — ridge→tanh calibration, continued bounded-J optimisation and matched JS on one shared split, 2026-10-03 — FROZEN 2026-10-03T12:08:50Z before the full run (CPU gate 1019945; pilot 1019946 disclosed)

Status: FROZEN 2026-10-03T12:08:50Z (main session; owner 2026-10-03: execute the v5 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the full-run jobs.
Source: `VCS_Server_Next_Round_v5_CN.md` §4, `VCS_Results_Review_and_Next_Plan_v5_CN.md` §4.3.  Code: `src/vcs_measure/solve.py`, `scripts/p116_solve.py`,
`slurm/p116_{gate,unit}.sbatch`, `tests/test_p116.py`; backward-compatible edits to `scripts/precheck_d_tests.py` and `scripts/cond_test_t1_ablation.py`.

## 1. Definitions corrected (no historical number changes)
- `precheck_d_tests.closed_form_critic` is a **two-stage** estimator, canonical name **`ridge_tanh_calibrated`** (alias added): the exact part is the
  ridge solution of the **raw linear J** (quadratic sub-problem); the output scale c is then picked on VAL from 25 values and T = tanh(c φᵀw).  It is
  not an exact optimiser of the tanh-wrapped bounded J.  Returned objects now carry `solver` metadata; numerics unchanged.
- `cond_test_t1_ablation.exact_js_critic` is an **L-BFGS numerical solve** (strong Wolfe, max_iter 200 per λ, tolerance_grad 1e-10 on the max-abs
  gradient, tolerance_change 1e-12).  It now records per λ: iterations, closure evaluations, termination reason, final max-abs / L2 gradient and the
  last objective change.  Fitted weights unchanged.
- Every fitter in `precheck_d_tests` (`fit_vcs_critic`, `closed_form_critic`, `fit_c2st`) and `exact_js_critic` accepts an explicit `split=(ti, vi)`;
  `split=None` reproduces the historical seed-based split bit for bit.  (`cond_test_t1.fit_js_critic` is outside this unit's file ownership and is
  not used here.)
- **Evidence:** `tests/test_p116.py` — new vs pre-edit code (fetched from git: precheck df6345d, ablation d98452a) give **bit-identical** P105
  instances on the same node; the historical P105 smoke instance (job 1016884, nodecpu05) is reproduced with identical p-values for all six critics,
  the ridge / L-BFGS statistics to float precision, and the float32 MLP statistics to ~3e-6 relative (cross-node CPU drift, disclosed).

## 2. Question
On the T1 linear class φ(z, n) = [z(2n − 1), 1] and identical data, how do cost, fit risk and permutation power compare across
(A1) ridge→tanh→calibration, (A2) continued optimisation of the original bounded J from A1, and (A3) the matched JS numerical solve?
Classifier fit and permutation power are separate endpoints (a better-fitting critic need not give a more powerful permutation statistic).

## 3. Design
- **Data:** T1 construction unchanged (P45 features; fresh N | Y per repeat; disjoint FIT / EVAL / POOL of n; within-class POOL negatives; B = 200
  shared within-class permutations of N on EVAL; δ = 0.05).
- **One shared split:** FIT → 80 / 20 (ti, vi) from a dedicated Generator (data-split role, seed·1000 + repeat); every algorithm uses the same rows.
  RNG roles: numpy `rng` = draws + POOL + permutations; torch Generator = split; no initialisation randomness (A1 deterministic, A2 warm-starts at A1,
  A3 / JS_P105 start at zero).
- **Algorithms** (all in φ; ridge grid λ ∈ {1e-3, 1e-2, 1e-1, 1}·sc, sc = mean diagonal of the second-moment matrix; calibration grid c = logspace(−1, 1.5, 25)):
  - **A1 `ridge_tanh_calibrated`** — identical to the historical "vcs_closed" (asserted).  Cost: 4 linear solves + 100 VAL evaluations.
  - **A2@B** — from A1's v0 = c w, L-BFGS (strong Wolfe) on −J_FIT(tanh φv) + λ sc |v|²/2 for every λ (warm start A1), B/4 closures per λ, model =
    (λ, checkpoint every ≤ 10 closures) with the best VAL J, **A1 itself a candidate** (so VAL J(A2) ≥ VAL J(A1) by construction).
  - **A2L@B** — the same runs without early stopping: last checkpoint within B, λ chosen on VAL (decomposes "continued optimisation" from "VAL selection").
  - **A3@B** — matched JS (Deep-InfoMax logistic on f = φv: E_P softplus(−f) + E_Q softplus(f) + λ sc |v|²/2), L-BFGS from zero, B/4 closures per λ,
    last checkpoint per λ, λ on VAL JS; common squared-score map T = tanh(f/2).
  - **JS_P105** — the historical `exact_js_critic` (max_iter 200 per λ) on the shared split: reference.
  - Budgets B ∈ {50, 200, 800} closures (objective + gradient evaluations on FIT) from one table; A2's cost includes A1's.  One run per algorithm to
    800, budget-B models read from checkpoints (identical to separate budget-B runs up to the last chunk boundary).
- **Endpoints per repeat:** own-objective FIT / VAL risk (unpenalised); common bounded-J score J(T) on FIT / VAL / EVAL; permutation p-value and
  rejection for the **own statistic** (VCS: J(tanh f); JS: JS value of f) and the **common statistic** J(T); optimisation residuals (max-abs / L2
  gradient, termination, closures used); seconds (fit, calibration / continuation, permutation, total).  Finite-oracle posterior MSE is **not
  available** for these image features (no oracle); not reported.
- **Never selected on EVAL:** solver, λ, c, checkpoint and budget choices use FIT / VAL only.

## 4. Pilot (method validation; disclosed — run after the gate, before the freeze)
Three conditions **named before running** (from P106's nine unsaturated T1 cells), 25 repeats each, seeds 601–603:
VCS-encoder colour s 0.05 n 1000 (P106: vcs_closed 0.59, js_exact 0.72); SimCLR colour s 0.2 n 2000 (0.99 / 0.96); SimCLR blur σ 0.5 n 1000 (0.68 / 0.72).
Nothing in the design was changed after the pilot (the A2 penalty grid was introduced after the
**gate** timing smoke 1019942/1019934, from VAL-only evidence: the unpenalised continuation over-fitted FIT — J_FIT 0.03 → 0.23 — and never beat A1
on VAL; no EVAL statistic was used).

## 5. Full run (after the freeze)
All nine P106 unsaturated cells at R = 100 (seeds 611–614 by encoder × family) + level nulls (both colour nulls at n 2000, both encoders, R = 200; seeds
615–616).  Three CPU bundle jobs (`slurm/p116_lines.txt`).

## 6. Pre-stated reading (descriptive; cost–risk–power per algorithm and budget)
- Per cell, paired over repeats (same draws): differences vs A1 in (a) EVAL common score J(T), (b) own-statistic rejection, (c) common-statistic
  rejection, each with mean and 95 % normal interval of the paired difference; plus mean seconds and closures.
- **"Fits better"** (classifier / score endpoint) for X vs A1 in a cell: paired 95 % interval of ΔJ_EVAL(T) excludes 0 on the positive side.
- **"Permutation power higher"** for X vs A1: paired 95 % interval of Δrejection excludes 0 on the positive side.  The two are reported **separately**;
  neither implies the other.
- Level: every statistic ≤ 0.09 at R = 200 on the null cells; a statistic above 0.09 is flagged and its power cells are not read.
- A2: the fraction of repeats in which VAL early stopping returns A1 (`chose_A1`) is reported per cell and budget.
- Not claimed: global optimality of A2 (non-convex); anything outside φ; a symbolic exact JS solution; nothing about SSL training.

## 7. Gate and cost
CPU gate 1019942 (nodecpu): `tests/test_p116.py` 8 passed (incl. bit-identity vs pre-edit code and the historical smoke reproduction), P109 9 passed,
P102 6 passed; runner timing on the three pilot cells (2 repeats, full budgets, 200 permutations): 5.3 / 5.5 / 6.3 s per repeat on 8 CPU threads.
GPU timing (job 1019938) did not start (user GPU quota full with the SSL runs) and was cancelled; the problems are small dense linear algebra and
the CPU partition does not use the GPU quota, so CPU is used.

## 8. Pilot results (job 1019946, nodecpu05, exit 0; `reports/P116/pilot_*.json`, aggregate `reports/P116_pilot_results.md`)
25 repeats per condition, 200 permutations, budgets 50 / 200 / 800.  Power = own / common statistic (identical for VCS critics).

| condition | A1 | A2@800 (chose A1) | A2L@800 | A3@50 | A3@800 | JS_P105 | ΔJ(T)_EVAL vs A1: A2L@800 / A3@800 |
|---|---|---|---|---|---|---|---|
| VCS colour s 0.05, n 1000 | 0.68 | 0.68 (1.00) | 0.64 | 0.72 | 0.72 | 0.72 | −0.030 [−0.033, −0.028] / −0.009 [−0.011, −0.008] |
| SimCLR colour s 0.2, n 2000 | 1.00 | 1.00 (0.96) | 1.00 | 0.92 | 0.92 | 0.92 | −0.010 [−0.011, −0.008] / −0.003 [−0.005, −0.002] |
| SimCLR blur σ 0.5, n 1000 | 0.88 | 0.88 (1.00) | 0.84 | 0.84 / 0.80 | 0.84 | 0.84 | −0.021 [−0.024, −0.018] / −0.006 [−0.009, −0.004] |

- **A2 (VAL early stopping) returns A1 in 100 / 96 / 100 % of repeats**: continuing to optimise the bounded J from the ridge→tanh→calibrated solution
  never improved VAL J in these cells (with or without the ridge-grid penalty).  **A2L** (forced continuation, no early stopping) fits FIT better but
  has a clearly *worse* EVAL J(T) (−0.010 to −0.030, all intervals below 0) and equal or slightly lower power (Δ −0.04 to 0, intervals include 0):
  on n ≈ 800 FIT rows for 513 features, further optimisation of the bounded J over-fits.
- **A3 (matched JS L-BFGS)** has a lower EVAL common score J(T = tanh(f/2)) than A1 in all three cells (−0.003 to −0.009, intervals below 0), but its
  permutation power is not distinguishable from A1's at R = 25 (Δ +0.04 / −0.08 / −0.04, all intervals include 0).  A3@50 ≈ A3@800 ≈ JS_P105: the
  convex JS problem is essentially solved within ≈ 12 closures per λ; at 800 most λ end by the change tolerance.
- **Cost (mean seconds per repeat-fit, 8 CPU threads):** A1 0.03–0.04 s; A3@50 0.05–0.06 s; A3@800 1.45–1.58 s; JS_P105 1.58–1.72 s; A2@800 2.1–2.3 s
  (A1 included).  Vectorised permutation tests 1–2 ms per critic.  Whole repeat ≈ 5.5–5.8 s.
- These are pilot (method-validation) numbers on 3 cells at R = 25; the pre-stated reading (§6) is applied to the full run only.

## 9. Full-run cost (measured)
≈ 5.5–5.8 s per repeat at n ≤ 2000 on 8 CPU threads → bundle a (4 cells × 100) ≈ 40 min, bundle b (5 cells × 100) ≈ 50 min, bundle c (4 null cells × 200)
≈ 80 min; ≈ 2.8 CPU-node-hours in 3 jobs, no GPU.

## Decisions at the freeze (main session)
1. Full run as drafted on the CPU partition (3 bundle jobs; the per-fit cost is CPU-bound and small; GPU not re-measured).
2. The pre-pilot design change (ridge-grid penalty for A2, the A2L decomposition) was made on gate VAL evidence only and is disclosed above; no
   EVAL statistic chose any solver, λ, c or iteration count, in the pilot or the full run.
3. `cond_test_t1.fit_js_critic` (the T1 iterative JS fitter) is not part of P116 and keeps its historical split; P116 compares A1 / A2 / A2L / A3 /
   JS_P105 on one shared split per repeat.
