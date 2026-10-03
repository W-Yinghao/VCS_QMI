# P116 — solver decomposition on the T1 linear class (v5 NEXT-E-SOLVE) — report — 2026-10-03

Pre-registration `P116_V5_SOLVE_PREREG_FROZEN_20261003.md`; jobs 1019966–1019968 (CPU) exit 0; results-only commit `6ed5a01` (`P116_results.{md,json}`, unit JSONs in
`reports/P116/`).  Same features φ = [z(2n−1), 1] as T1; every algorithm on one shared FIT / VAL split per repeat, same EVAL draws and permutations.
Algorithms: **A1** ridge_tanh_calibrated (exact ridge of the raw linear J → VAL-chosen output scale → tanh); **A2@B** penalised continuation of the bounded J
from A1 with VAL early stopping (A1 itself a candidate); **A2L@B** the same without early stopping (decomposition only); **A3@B** matched JS by L-BFGS, λ on VAL;
**JS_P105** the historical solve on the same split.  B = 50 / 200 / 800 closures.

## 1. Level
All four null cells (both colour nulls, both encoders, n 2000, R 200): every statistic 0.04–0.07 (≤ 0.09) → no flag; all power cells are read.

## 2. Frozen readings over the 9 unsaturated cells (paired vs A1, 95 % intervals)

| algorithm | "fits better" (ΔJ_EVAL(T) > 0) | fits worse | "permutation power higher" (own statistic) | power lower | seconds per fit |
|---|---|---|---|---|---|
| A2@50/200/800 | 0 / 9 | 0 / 9 | 0 / 9 | 0 / 9 | 0.1–1.9 (VAL returns A1 in 100 % of repeats, every cell) |
| A2L@800 | 0 / 9 | **9 / 9** | 1 / 9 | 0 / 9 | ≈ 1.8 |
| A3@50 | 0 / 9 | **9 / 9** | 2 / 9 | 0 / 9 | ≈ 0.05 |
| A3@800 / JS_P105 | 0 / 9 | **9 / 9** | **2 / 9** | 0 / 9 | 1.2–1.4 |
| A1 (reference) | — | — | — | — | **0.04** |

A3@800 power vs A1 per cell: +0.07, +0.01, −0.03, +0.04, +0.04 [+0.00, +0.08], **+0.12 [+0.04, +0.20]**, 0.00, **+0.08 [+0.03, +0.13]**, +0.02.

## 3. Reading
- **Continuing to optimise the bounded J from the ridge-tanh solution adds nothing**: VAL early stopping keeps A1 in every repeat of every cell; forcing the
  continuation (A2L) over-fits — the EVAL common score drops in all 9 cells — without a power gain (1 / 9 higher).  Within this linear class the two-stage
  ridge → tanh solution is already at the VAL optimum of the bounded J, at ≈ 1/40 of the iterative cost.
- **The two endpoints dissociate** (v5 asked for them separately): the matched JS solution fits the common squared score worse than A1 in all 9 cells
  (ΔJ −0.002 to −0.027), yet its permutation power is never lower and is higher in 2 of 9 cells (+0.08, +0.12; mean over cells ≈ +0.04).  So "fits the
  squared score better" does not imply "tests better", and the JS solve's extra cost buys a little power in some weak-signal cells.
- **Attribution update for P74 / P106:** with shared splits and the same linear class, the ridge-tanh (VCS) solution is not more powerful than the
  numerically solved JS — at best level, sometimes slightly behind.  Its distinct merit is cost: ≈ 0.04 s per fit vs 1.2–1.4 s (≈ 30–35×), with essentially
  the same power.  JS converges in ≈ 50 closures (A3@50 ≈ A3@800).
- Not claimed: global optimality of A2 (non-convex), anything outside φ, a symbolic JS solution, anything about SSL.
