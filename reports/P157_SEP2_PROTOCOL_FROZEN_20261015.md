# Pre-registration — P157 (SEP-2): confirmatory test of P156's secondary finding — the quadratic rule in parametric critic classes under contamination — FROZEN 2026-10-15T07:21:18Z (before any formal P157 cell)

## 0. Why this unit, and who decided
- P156 (`reports/P156_SEP_REPORT_20261015.md` §3.2) found one region where the quadratic rule (VCS, γ = 1) is the more accurate posterior estimator than matched logistic (JS): parametric critic classes (cos-2 g = a·cos(x, y) + b, the application form; quad-3) under independent-pair contamination.
  - VCS had the lower error in all 16 such cells; cos-2 in all 8 cells, Holm p ≤ 3.5e-6.
  - It was a secondary result. P156's P3 falsifier deleted every "better than JS" sentence, and the report proposed a confirmatory unit.
- **Autonomous unit** under the owner's standing rules (2026-10-11: "when GPUs are free, judge necessity and submit useful experiments without asking"; 2026-10-14: "不要让空着，也不要让没有排队的实验"). On 2026-10-15 an L40S node was idle, and every ready job was H100- or ≥ 80 GB-only, or blocked by V8-03's tuning-first rule. The owner is informed in the session report and can stop the unit at any stage.
- P156's own data are not reused for inference: fresh seeds, and a new primary question fixed before any run.

## 1. Questions
- **Q1.** At N 4,096 / 2,000 updates (P156's budget), is VCS's posterior error lower than JS's in parametric classes under contamination?
- **Q2.** Does it persist at N 16,384 (4 × the data)? P156 block C showed that the contamination narrowing in the expressive class reverses at 4 × N. A population projection effect should persist; a finite-sample effect should shrink.
- **Q3.** Does cos-2 alone (the application critic form, both N) carry it?

## 2. Design (frozen P156 trainer and readouts, `sep.run_cell`, unchanged)
- **Classes:** cos-2 and quad-3, starting from a = b = c = 0 for every arm, as in P156.
- **Contamination:** (a) independent pairs and (b) false negatives, both with product negatives, at ε 0.1 and 0.3. Type (b) was not run in parametric classes in P156.
- **Grid:** d ∈ {20, 64}, I ∈ {4, 8} nats.
- **Block S2:** N ∈ {4,096, 16,384}, 2,000 updates, batch 256. 2 × 2 × 2 × 2 × 2 × 2 = **64 cells per seed**.
- **Arms:** JS (γ 0) and VCS (γ 1).
- **lr grid:** {1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0}. This is P156's parametric grid widened at the top, because P156's edge rule fired there and lr 0.3 was still chosen often after its addendum.
  - Selection: native risk on SELECT (primary) and common J (sensitivity).
  - A non-finite score is never selected (`p157.reselect_finite`; identical to `run_cell` when every score is finite).
- **Seeds:** 200–207 (8 fresh seeds).
- **Block S2P (descriptive, one seed 200):** the same 32 configurations at N 131,072 and 8,000 updates. It is a large-sample proxy for the population projections of the two rules.

## 3. Readouts and tests
- **Endpoint:** posterior MSE e against the contaminated posterior η_ε under M_ε on EVAL (P156 §4).
- **Effect per cell and seed:** (e_VCS − e_JS) / ē_JS, where ē_JS is the cell's JS error averaged over seeds; negative means VCS is more accurate.
  - The relative scale replaces P156's clean-JS standardisation: quad-3's clean error (≈ 5e-4) made that scale explode (P156 T4).
- **Primary family, Holm over 3.** For each test, take per seed the mean of the effect over the test's cells, then a paired t over the 8 seeds:
  - **Q1:** the 32 cells at N 4,096;
  - **Q2:** the 32 cells at N 16,384;
  - **Q3:** the 32 cos-2 cells, both N.
- **Outcome rule (fixed now):**
  - **CONFIRMED** if Q1 < 0 and Q2 < 0 (Holm). Permitted sentence: "In parametric critic classes under mixture contamination, the quadratic (Brier) rule estimates the posterior more accurately than matched logistic, also at four times the data". It must be reported as the L² projection property of the Brier score that VCS inherits, not as a property of the CS divergence, with P156's P3 null at ε = 0 stated next to it.
  - **FINITE-SAMPLE ONLY** if Q1 < 0 and Q2 is not < 0.
  - **NOT CONFIRMED** if Q1 is not < 0. P156's secondary finding is then withdrawn.
- **Secondary:**
  - absolute Δe per cell, Holm over the 64;
  - Q1–Q3 under common-J selection.
- **Descriptive:**
  - the S2P large-sample table;
  - lr-edge shares (top lr 1.0) and not-converged counts. If the top lr is still chosen in more than half of a class's selections, it is reported as a limitation; there is no further extension.

## 4. Not claimed
- Anything about SSL, VL or natural features.
- Expressive classes (P156 covers them).
- Robustness.
- Any statement at ε = 0 (P156 P3).
- **Prior art cited with any result:** Buja, Stuetzle & Shen 2005 (Beta family); the L² projection of the Brier score; the noisy-label literature listed in P156 §6.

## 5. Gate, cost, priority
- **Gate before launch:**
  - CPU tests: `tests/test_p157.py` (cell lists; re-selection; a tiny end-to-end cell of each class × contamination type; the aggregate on synthetic cells with known effects, giving outcome FINITE-SAMPLE ONLY), together with the P156 suite.
  - GPU smoke: 2 full S2 cells (types a and b) and 1 full S2P cell, timed.
- **Cost:** about 30 s per S2 cell, so 512 cells ≈ 4.5 GPU-h, plus S2P (32 cells at 4 × the updates) ≈ 1.5 GPU-h.
  - Submission: 4 jobs of 2 seeds plus 1 S2P job.
  - Partitions RTX6000PRO, H100 and L40S, excluding node51 and node60; nice 400, so every V7 / V8 / REV job starts first.
- **Aggregation:** `scripts/p157_aggregate.py --dir outputs/P157_sep --out reports/P157_SEP2_results/p157_aggregate`. Results-only commit, then the report.

## 6. Gate record
- **CPU tests:** `tests/test_p157.py` + `tests/test_p156.py`, 54 / 54 (job 1041058).
- **GPU smoke** (job 1041064, L40S node50; outputs/P157_smoke, not used):
  - 2 full S2 cells (cos-2, ε 0.3, d 64, I 8, N 16,384: types (a) and (b)), 23 s each;
  - 1 full S2P cell (quad-3, type (b), ε 0.1, d 64, I 8), 106 s;
  - 0 non-finite steps.
  - Cost: 512 × 23 s ≈ 3.3 GPU-h plus 32 × 106 s ≈ 0.9 GPU-h.
- **Disclosure.** The smoke cells use seed 200, one of the eight formal seeds, and their selected errors were printed (VCS lower in all three). Sections 0–5 were written before the smoke and are unchanged. The formal run recomputes these cells in `outputs/P157_sep`, deterministically, from the same code.
