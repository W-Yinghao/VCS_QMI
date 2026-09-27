# P65 addendum 1 (DRAFT) — why the subject-swapped null is anti-conservative at n = 240, and the fix (2026-09-28) — FROZEN 2026-09-27T23:09:01Z before the re-run (diagnosis CPU jobs 1012103 / 1012104; P66 results untouched; P66b = same prep, same seeds, null = exact circular orbit; the whole grid (i)–(v) is re-read on P66b)

Status: DRAFT for the main session to freeze; the frozen P66 tables (`P66_task_D_fmri.{md,json}`) are untouched.  Diagnosis = CPU-only re-analysis
of the existing prep outputs (`outputs/P65_task_D_fmri/prep`, 60 subjects, α = 0, lag 0, subject-swapped pairs z_i / N_{i+1}); jobs 1012103 (3-subject
smoke) and 1012104 (full), logs `slurm_logs/fmri_nulldiag_{smoke,full}_*.out`; script `scripts/task_d_fmri_null_diag.py`; tables
`P65_null_diagnosis.{md,json}`.  No GPU, no fleet re-run.

## 1. Observation (frozen P66, exact null by construction)
| n | L_eval | vcs_shift | hsic_shift | qcfc_corr |
|---|---|---|---|---|
| 120 | 36 | 0.08 / 0.07 / 0.05 | 0.08 / 0.10 / 0.10 | 0.10 / 0.07 / 0.05 |
| **240** | **72** | **0.22 / 0.30 / 0.20** | **0.15 / 0.12 / 0.18** | **0.12 / 0.13 / 0.17** |
| all (475) | 142 | 0.08 / 0.07 / 0.08 | 0.05 / 0.10 / 0.05 | 0.12 / 0.10 / 0.07 |
(α = 0 / 0.5 / 1; R = 60 subjects, so 0.05 ± 0.03.)  In the frozen JSON every rejection at n = 240 has p = 1/201 exactly (7 / 60 qcfc, 9 / 60 HSIC,
13 / 60 VCS), and the next smallest p-value is 0.06: the null distribution has a gap between "observed above every draw" and everything else.

## 2. Cause: the admissible-shift rule leaves 13 distinct shifts at L_eval = 72
The frozen construction draws the 200 null shifts of the evaluation block from [min_shift, L − min_shift] with min_shift = 30, falling back to
the full group {1, …, L − 1} only when L − 30 ≤ 30.

| L_eval (n) | distinct admissible shifts | note |
|---|---|---|
| 36 (n = 120) | 35 = the full group | fallback → an exact Monte-Carlo test |
| 60 (n = 200) | 59 = the full group | fallback → exact |
| **72 (n = 240)** | **13 (shifts 30–42)** | 200 draws of 13 values ≈ 15 copies each |
| 90 (n = 300) | 31 (30–60) | |
| 142 (n = all) | 83 (30–112) | |

With 13 distinct null values the p-value can only be 1/201 (observed above all 13) or ≥ 16/201 ≈ 0.08 (below at least one value with its ~15
copies), so "reject" means "observed exceeds the maximum of 13 restricted shifts".  Even under perfect exchangeability that event has
probability 1/14 = 0.071, not 0.05; and shift 0 is *not* exchangeable with a subset of mid-range shifts — a Monte-Carlo permutation test is
exact only when the null draws come uniformly from the whole group (here all L − 1 circular shifts, or a superset that contains the identity's
orbit).  Restricting to "far" shifts removes exactly the shifts whose statistics resemble the observed one, so the null is too narrow.  The
seam of a circular shift adds a little on top (mean |r| over the orbit 0.122 vs 0.133 at shift 0; HSIC 0.0053 vs 0.0054; J −0.0139 vs −0.0146):
small next to the restriction effect.  The VCS test is hit hardest because its statistic is also the least smooth function of the shift (fitted
critic, K = 4 product shifts drawn from the same restricted set), so the observed value more often sits above all 13 orbit points (18 / 60
subjects also sit *below* all of them, p = 1.0 — the bimodal p-histogram of the frozen JSON).
n = 240 is "special" only because L_eval = 72 is the first length above the fallback threshold: the rule gives ≤ 40 distinct shifts for
61 ≤ L ≤ 100 and recovers slowly beyond that (83 at L = 142 → the milder 0.07–0.12 at n = all).

## 3. The other hypotheses, checked
- *Near-identity effective offsets in the null draws:* none at n = 240 (all draws are 30–42 away from the identity on a 72-block; the wrap
  point is inside the block by construction of a circular shift of the block — that is the seam, measured above).  The near-identity shifts
  are present at n = 120 and n = 200 through the fallback — and those cells are the well-calibrated ones, because the fallback *is* the full group.
- *Truncation / cosine-phase coupling between z_i and the swapped N_{i+1}:* the two series come from different subjects, each high-passed on its
  own run; truncating both to the first n TRs cannot create dependence.  The observed statistics across the 60 subjects behave as an
  autocorrelated-but-independent null should at every n: |r| mean 0.158 / 0.133 / 0.103 and 95th percentile 0.376 / 0.358 / 0.247 for L_eval
  = 36 / 72 / 142 (i.i.d. sd would be 0.167 / 0.118 / 0.084; the excess is the autocorrelation, monotone in L, nothing peculiar at 240).  The
  anomaly is in the null construction only.

## 4. Re-analysis of the swapped null at n = 200 / 240 / 300 (α = 0, lag 0, 200 draws, same critics, CPU job 1012104)
| n (L_eval) | null construction | qcfc | hsic | vcs |
|---|---|---|---|---|
| 240 (72) | **frozen rule** (13 distinct shifts) | **0.12** | **0.17** | **0.20** |
| 240 (72) | shifts uniform on the full group {1..71} | 0.07 | 0.08 | 0.07 |
| 240 (72) | exact orbit test (all 71 shifts once, p = (1 + #≥)/72) | 0.07 | 0.07 | 0.05 |
| 240 (72) | shift over the whole 240-window, then take the block | 0.08 | 0.07 | 0.07 |
| 240 (72) | moving-block bootstrap (block 12 TRs) | 0.08 | 0.07 | 0.07 |
| 240 (72) | phase-randomised surrogate | 0.05 | 0.00 | 0.05 |
| 200 (60) | frozen rule (= full group by fallback) | 0.08 | 0.08 | 0.08 |
| 200 (60) | full group / exact orbit / window / block-boot / phase | 0.05 / 0.05 / 0.07 / 0.05 / 0.05 | 0.08 / 0.08 / 0.10 / 0.12 / 0.05 | 0.08 / 0.08 / 0.10 / 0.08 / 0.08 |
| 300 (90) | **frozen rule** (31 distinct shifts) | 0.10 | 0.07 | **0.13** |
| 300 (90) | full group / exact orbit | 0.05 / 0.05 | 0.02 / 0.02 | 0.05 / 0.05 |
| 300 (90) | window / block-boot / phase | 0.07 / 0.08 / 0.05 | 0.10 / 0.05 / 0.02 | 0.05 / 0.07 / 0.05 |

Every construction that uses the whole group (or a model-based surrogate) is at the nominal level within the R = 60 noise (0.05 ± 0.03) for
all three tests; the restricted rule alone reproduces the anomaly, and it reproduces it wherever the distinct-shift count is small (n = 300 too,
which P66 did not run).  The (3a) variant asked for — "shifts restricted to [min_shift, L − min_shift] on the evaluation block" — *is* the frozen
rule; it is the cause, not a fix.

## 5. Proposed fix (a new run; nothing in P66 is edited)
1. **Null:** the exact circular-orbit test on the evaluation block — all L − 1 shifts once, p = (1 + #{null ≥ obs}) / L — for `vcs_shift`,
   `hsic_shift`, `c2st` and `qcfc_corr`.  It is exact for the cyclic group (up to the seam, shown to be ≤ 0.01–0.02), cheaper than 200 draws
   (L − 1 ≤ 141 evaluations), and its granularity 1/L is fine for δ = 0.05 at every L used (minimum p 1/36 = 0.028 at n = 120).
   `min_shift` stays only where it belongs: the K = 4 product-of-marginals shifts of the Q-term (there a near-identity shift would make the
   "negative" pairs dependent), drawn as now; the null shifts must not be filtered.
2. **Report (i) on the re-run**, not on P66: the P66 null cells at n = 240 are invalid by construction (and n = all mildly so); the frozen power
   cells at n = 240 used the same 13-shift null and are biased upward by the same mechanism, so the whole grid (i)–(v) is re-read on the re-run.
   Expected effect on the reading: rule (ii) at n = 240 will lose some of its rejections; the qualitative picture at n = all (83 shifts) should
   move little.
3. **Cost:** identical prep (already on disk); tests ≈ 1 GPU-h (fewer null evaluations than P66).  Output prefix `reports/P66b_task_D_fmri`.
4. **Disclosure:** this addendum, the diagnosis tables and the fact that the frozen rule was chosen before its discreteness at L = 72 was noticed
   are all stated in the P66b report; the P66 tables remain as the record of the first run.

## 6. Not changed by this addendum
The reading rules (i)–(v), the preprocessing, the contiguous split, the critics, the subject set, the lag grid, δ.
