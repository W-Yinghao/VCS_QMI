# VL1-14 — does the VL critic need an SSL-style search? — report — 2026-10-09

Protocol `VL1_14_CRITIC_SCREEN_FROZEN_20261009.md`; results-only commit `40e501e` (`reports/VL1/VL1_14_results.json`, `scripts/vl1_14_aggregate.py`,
job 1031756).  VCS, N all, seeds 0–2, CLIP crops and SigLIP 2 crops; every cell selects its lr and checkpoint by CAL common J only; C0 = the
frozen F2r fits (lr ≤ 2e-3, 3 000 updates).  Differences paired by seed, 95 % t intervals; J × 100.

## 1. Result
| cell | CLIP: CAL J (Δ vs C0) | CLIP: DEV Top-1 (Δ) | SigLIP 2: CAL J (Δ) | SigLIP 2: DEV Top-1 (Δ) | selected lr (update) |
|---|---|---|---|---|---|
| C0 frozen F2r | 7.20 | 68.80 | 11.63 | 76.00 | 2e-3 (cap-bound on CLIP) |
| C1 F2r, lr ≤ 1e-2, 10k updates | 7.41 (+0.20 [−0.07, +0.48]) | 69.26 (+0.46 [−1.45, +2.37]) | 11.92 (**+0.29 [+0.03, +0.55]**) | 76.76 (+0.76 [−1.71, +3.23]) | 5e-3–1e-2 (1 300–2 150) |
| C2 residual MLP 512 × 3 | 7.31 (+0.10 [−0.12, +0.33]) | 69.50 (**+0.70 [+0.36, +1.03]**) | 11.82 (**+0.19 [+0.05, +0.34]**) | 77.30 (**+1.29 [+0.84, +1.74]**) | 5e-3–1e-2 (1 850–2 900) |
| C3 bilinear, rank 64 | 6.85 (**−0.36**) | 70.08 (**+1.28**) | 10.80 (**−0.83**) | 77.75 (**+1.75**) | 2e-3 (1 350–2 350) |
| C4 affine a·cos + b | 3.21 (−3.99) | 65.99 (= raw) | 7.21 (−4.42) | 75.48 (= raw) | 5e-3–1e-2 |

## 2. Frozen decision
**No cell meets the rule** (ΔJ ≥ 0.50 with the interval excluding 0) on either feature set.  The best cell by CAL J is C1 (optimisation only):
+0.20 on CLIP (interval includes 0) and +0.29 on SigLIP 2 (interval excludes 0, below the threshold).  **Verdict: "no critic search needed".**
F2r stays, and the VL tables (VL1-10 … VL1-13) stand.

## 3. What the screen shows (descriptive)
1. **The estimator is robust to the critic within this space.**  The frozen F2r recipe is within 0.3 J (×100) of the best of four alternatives
   (≈ 3 % of J on CLIP, 2.5 % on SigLIP 2), even though its lr sat at the grid edge.  Moving past the edge (C1) finds an interior optimum
   (5e-3) and the 10 000-update cap no longer binds (optima at 1 300–2 900 updates), but J changes little.  The edge was real; its cost was small.
2. **Calibration of cosine alone carries ~45 % (CLIP) / 62 % (SigLIP 2) of the fitted J** (C4 vs C0).  The residual MLP's corrections add the
   rest; ranking-changing capacity is where the remaining J comes from.
3. **Estimation and ranking are separable.**  The bilinear critic ranks better than F2r (+1.3 / +1.75 Top-1) but estimates worse (−0.36 / −0.83
   J).  The larger MLP ranks better (+0.7 / +1.3) at almost the same J.  Selecting by J (the estimator criterion) does not pick the best ranker.
   This is the same message as the task-loss rows (softmax ranks best, has no J) and as VL1-12 (higher J on strong features without a Top-1 gain).
4. **Answer to the owner's question:** unlike SSL, where the scorer and its scalars had to be searched because they shape the learned
   representation, the VL critic estimating dependence between frozen features does not need a search: the estimator reading is stable across
   critic families and optimisation settings to within about 0.3 J.  A search would only pay off for the ranking task, which is the task loss's
   job.  If ranking were the goal, C2 / C3 would be the candidates, selected by a task criterion, outside this estimator-centred design.
