# Pre-registration — P134: R1, estimator-level synthetic contamination on the P85 staircase cell (VCS / JS / InfoNCE / NWJ) — FROZEN 2026-10-04T17:56:13Z

Status: DRAFT (fork D).  The main session freezes it (renamed `*_FROZEN_*`, timestamp, "Decisions at the freeze") before any full-unit GPU job.
Owner approval 2026-10-04: "全部提交" (the list included "R1 synthetic contamination").  Sources: `reports/NEXT_ROUND_EXECUTION_PLAN_DRAFT_20260928.md` §2 R1;
`VCS_QMI_Next_Round_Plan_v2.md` "R1 估计器层面的污染" (claim, metric, falsification); estimator server spec v1 §11.1 (a contamination that changes a
marginal must change Q to the product of the contaminated marginals; report the contaminated-truth error and the change against the clean target
separately — "污染会改变 P，真实目标的变化不能混同为估计不稳定").  P69 (the original R1 draft) was never frozen; its `data.contaminate` is not used.

## Question
When a fraction ε of the joint pairs is replaced by contaminating pairs, how far does each learned estimator's value move, measured in that
estimator's own resolution on the clean dependence staircase?  The spec's claim: VCS moves at most ≈ 1 resolution unit per 0.1 of ε in every
contamination type (bounded influence: a₊, a₋ ∈ [−1.5, 0.5]), while JS / InfoNCE / NWJ move further.

## Design (code: `src/vcs_estim/r1.py`, `scripts/p134_r1.py`, `scripts/p134_aggregate.py`, `slurm/p134_r1.sbatch`, `slurm/p134_units/seed{0,1,2}.txt`)
**Base cell = the P85 staircase cell** `gaussian, d = 20, I = 4 nats (ρ = 0.5742, S = 0.802), N = 4096, B = 256, 2000 updates`, seeds 0, 1, 2 (P85
convention: 3 seeds, each seed = its own roles and its own critic initialisation).  Roles exactly as P85 (`benchmark.build_roles`, same seeded
streams): FIT = 4096 per distribution, TUNE = SELECT = 1024, EVAL = 32 768 per side, TRUTH = 200 000 per side (GRAD unused).
**Estimators** (`benchmark.neural_rows`, unchanged): VCS, JS (Deep-InfoMax form), InfoNCE, NWJ — every P85 negative construction (VCS, JS: product /
cyclic8 / in-batch; InfoNCE: in-batch / cyclic8; NWJ: product / in-batch) — JointMLP concat → 256 → 256 → 1, Adam, batch 256, 2000 updates,
lr ∈ {1e-4, 5e-4, 2e-3}, the state and the lr selected by the native risk on the (contaminated) SELECT role every 100 updates.  Same budget for all.
**Contamination** (every role except GRAD; exactly round(ε n) pairs per role, seeded permutation, indices recorded with a SHA-256 per role; fresh
noise from its own stream): ε ∈ {0.01, 0.05, 0.1, 0.2} and the clean cell ε = 0 (bit-for-bit the P85 roles), joint P_ε = (1 − ε) P + ε C:

| type | contaminating pair C | marginals of C | Q (product negatives) |
|---|---|---|---|
| **independent** | (x, y′), y′ ~ N(0, I) fresh | unchanged | unchanged (P_ε = (1 − ε) P + ε Q, the spec's form; oracle exact) |
| **outlier** | x_o = x + 5 (5σ on every coordinate); y_o = ρ_h x_o + √(1 − ρ_h²) e: y from the conditional of the *other end of the staircase* (I = 10 nats, ρ_h = 0.795) given the shifted x | N(5, I), N(5ρ_h, I) | x-side: ε-fraction + 5; y-side (independent ε-fraction) + 5ρ_h |
| **heavy** | y_c = y + η, η multivariate Student-t₂ (z / √W, z ~ N(0, I₂₀), W ~ Exp(1), one scale per pair) | x unchanged; y: N(0, I) ⊕ t₂ | y-side: ε-fraction + independent t₂ |
| outlier_indep (descriptive) | the P69-draft reading: x + 5, y an independent N(0, I) draw | N(5, I), unchanged | x-side + 5 |

Q is always the product of the *actual* contaminated marginals (spec v1 §11.1); in-batch / cyclic negatives form it from the contaminated P sample.
**Truth of the contaminated problem:** log r_ε = log p_ε − log p_ε,X − log p_ε,Y in closed form (Gaussian mixtures) or with a log-space trapezoid over
the t₂ scale (3 001 nodes on log W ∈ [−40, 6]; converged to 1e-8 against 12 001 nodes); S_ε = E_M tanh(log r_ε / 2)², JS2_ε, MI_ε = E_{P_ε} log r_ε
by Monte Carlo on the contaminated TRUTH role (with s.e.); the posterior η_ε on EVAL for the VCS / JS posterior read-outs.  ε = 0 uses P85's truths.
**Cells:** 3 clean + 4 types × 4 ε × 3 seeds = 51 cells; 30 fits per cell.

## Metric (pre-stated)
- **Resolution unit u** of an estimator-construction = the mean adjacent-step gap of its selected native value around I = 4 on the **P85 clean
  staircase** (existing cells I = 2 / 4 / 6, seeds 0–2, mean over seeds): u = (V̄(6) − V̄(2)) / 2; the one-sided gaps u_down = V̄(4) − V̄(2),
  u_up = V̄(6) − V̄(4) define a directional variant (secondary).  A construction whose clean staircase is not ordered around I = 4 (u_down ≤ 0 or
  u_up ≤ 0) has **no resolution unit**; its shifts are reported in native units only.  Known before any contaminated run: NWJ-product is not ordered
  (P85 V̄(6) = −60.9, rare-event variance), so **NWJ's primary row is its in-batch construction**.  Units from P85 (selected rows): VCS-product 0.195,
  JS-product 0.297, InfoNCE-in-batch 1.614, NWJ-in-batch 1.804 (all seed sd at I = 4 ≤ 0.02 u).
- **Shift** Δ(type, ε) = V(ε) − V(0) per seed (paired with the P134 clean cell of the same seed), mean ± sd over 3 seeds; **normalized shift**
  = |Δ̄| / u / (ε / 0.1) (resolution units per 0.1 of ε); per type m_E(type) = max over ε of the normalized shift.
- **Decomposition (reported for every row):** Δ = [truth_ε − truth_0] + [error change], with each estimator's own target (S, JS2, MI); the target
  change in resolution units is printed next to the shift, so a moved target is not read as estimator instability (spec v1).

## Reading (pre-stated; primary rows VCS-product and JS-product; primary types independent, outlier, heavy)
- **supported**: m_VCS(type) ≤ 1 in every primary type, and JS is not "alike";
- **refuted (JS alike)** — the spec's falsification condition: m_VCS ≤ 1 in every type *and* m_JS(type) ≤ 1 and m_JS(type) ≤ 1.2 × m_VCS(type) in every
  type (learned JS bounded alike, within 20 %);
- **not supported**: m_VCS(type) > 1 in some primary type.
- InfoNCE-in-batch and NWJ-in-batch: the same m statistics, descriptive (cross-target comparison only through each estimator's own resolution unit,
  P85 rule 5; nothing is converted between targets).  All other constructions, the directional unit and outlier_indep: descriptive.
- The error against the contaminated truth and the target change are reported for every cell; no claim about estimating the *clean* target
  (known-ε cleaning formulas are not used — spec v1 §11.1).

## QC (per cell, in the aggregate)
ε = 0 cells reproduce P85's I = 4 rows (same roles, seeds, trainer; difference reported per row, GPU non-determinism only); exact contaminated counts
per role and side; sampler–density agreement by the bounded identity E_M[η_ε] = 0 on TRUTH (|value| ≤ 4 s.e.; the importance identities E_Q r = 1,
E_P 1/r = 1 are reported but not used — r is heavy-tailed under the outlier law); every selected value finite; non-finite training steps counted.

## Not claimed
Anything about SSL training, image data or the official test set; robustness of the estimators in general (one Gaussian setting, d = 20, I = 4, one
critic class and budget); recovery of the clean target; any conversion between S, JS and MI.

## Gate (CPU) — filled from the jobs
- 1021150: 30 / 31 tests; the failure was the first version's importance identity E_Q r = 1 under the outlier law (MC mean 0.44 ± 0.09 with 5 000
  samples — rare-event dominated, not a sampler error); replaced by the bounded identity E_M η = 0 (+ a power test that a wrong shift is detected).
  CPU smoke (N = 256, 40 updates, 3 cells: clean, outlier 0.1, heavy 0.2): all rows finite, 121–130 s per tiny cell on 8 CPU threads; aggregator ran.
- 1021163 / 1021180: the bounded identity passes for all 4 types × ε ∈ {0.05, 0.2} on the full 200 000-pair TRUTH role; the first power test
  (wrong shift in outlier_indep) was a bad design — that contaminating pair is independent, so its log r ≈ 0 for any shift — replaced by a wrong
  conditional (density with ρ at I = 4 instead of I = 10 for samples drawn at I = 10).
- **1021185 (final): 36 / 36 tests pass** (ε = 0 bit-for-bit for all types; exact counts and recorded indices for 4 types × 4 ε; determinism; closed
  form of the independent log-ratio; ε = 0 → PMI; t₂ quadrature converged and normalised; bounded identity for 4 types × 2 ε; the identity rejects a
  wrong density by the same 4-s.e. criterion; S_ε decreasing in ε for the independent type; reading helpers on toy numbers; CPU cell smoke);
  CPU smoke cells (3) and the aggregator rc = 0.

## Cost and submission
**GPU timing (job 1021151, RTX6000PRO node58, full-size cells, written to `outputs/P134_TIMING/`, not reused):** 130 s for the clean cell and 128 s
for heavy ε = 0.2 (the costliest truth: t₂ quadrature) → 51 cells ≈ 1.9 GPU-h on a healthy RTX6000PRO, ≈ 2× on L40S; 3 bundled jobs (one per seed,
17 cells, ≈ 40 min each on RTX), resumable.  **Disclosure:** that job's log prints the selected native values of its two cells (seed 0, clean and
heavy ε = 0.2); they were seen by the fork before the freeze.  The clean cell reproduces P85's I = 4 seed-0 selected values (8 / 10 constructions
identical to 4 decimals; InfoNCE-in-batch +1e-4, NWJ-in-batch −0.004: GPU non-determinism in the large in-batch matrices).  The final jobs recompute
both cells into `outputs/P134_r1/`.  Partitions RTX6000PRO / H100 / L40S, node51 and node60 excluded.
```
for s in 0 1 2; do sbatch --job-name=p134_s$s --export=ALL,UNITS=slurm/p134_units/seed$s.txt,OUT=/home/infres/yinwang/CS_QMI/outputs/P134_r1 slurm/p134_r1.sbatch; done
python scripts/p134_aggregate.py --p134 /home/infres/yinwang/CS_QMI/outputs/P134_r1 --p85 /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark --out reports/P134_r1_aggregate
```

## Open decisions for the freeze
1. **Outlier law.** The spec's "y from the other end of the staircase" is read as the I = 10 conditional of the shifted x (primary); the P69 draft's
   reading (independent y) is kept as the descriptive `outlier_indep`.  Alternative: swap the two, or drop `outlier_indep` (−12 cells).
2. **t₂ noise**: multivariate (one scale per pair, as the P69 draft) vs independent per coordinate.  Draft: multivariate.
3. **NWJ primary = in-batch** (product staircase not ordered in P85).
4. Resolution unit = mean adjacent gap (primary) vs directional gap (secondary).

## Decisions at the freeze (main session)
Owner 2026-10-04 "全部提交" (R1 approved).  Gate: jobs 1021150 / 1021163 / 1021180 / 1021185 (final 36 / 36, CPU smoke and aggregator rc 0); GPU timing
1021151 (two seed-0 cells seen before the freeze, disclosed above, not reused).
1. Outlier law as drafted: primary = x + 5σ with y from the I = 10 conditional of the shifted x; `outlier_indep` (P69 reading) kept, descriptive.
2. t₂ noise: multivariate (one scale per pair), as drafted.
3. NWJ primary construction = in-batch (NWJ-product's P85 staircase is not ordered), as drafted.
4. Resolution unit: mean adjacent gap primary, directional gap secondary, as drafted.
5. Submit the three seed jobs now (normal QOS, RTX6000PRO / H100 / L40S, node51 + node60 excluded); short jobs, so the long training jobs pending
   behind them get Nice=500 until they start (EligibleTime unchanged).
