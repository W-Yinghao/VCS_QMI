# Task pre-check D-fMRI — motion leakage in cleaned resting-state BOLD (P65 / P66): report

Prereg: `P65_TASK_D_FMRI_PREREG_FROZEN_20260927.md`.  Table: `P66_task_D_fmri.md` (results-only commit ada3866).  Data: AOMIC-PIOP1 (ds002785, CC0),
60 subjects, resting-state fMRIPrep output in T1w space (TR 0.75 s, 475 volumes after dropping 5), 24-parameter motion model applied at
α ∈ {0, 0.5, 1}, PCA-32 block-mean features, nuisance = framewise displacement; tests with circular-shift nulls (200 shifts ≥ 30 TRs), δ = 0.05,
R = 60 subjects per cell.  Evidence category: completed (CPU prep 1011936, GPU tests 1011937, 1 h).  QC: TR 0.75 s, 475 volumes, FD mean 0.08 mm
(a low-motion cohort), PCA explained variance 0.36–0.47 (≥ 0.30 sentinel met).

## 1. Results (rejection rate over 60 subjects; τ = lag in TRs)
| α (motion cleaning) | n | vcs_shift τ = 0 / 1 / 2 / 4 / 8 | hsic_shift τ = 0 / 1 / 2 / 4 / 8 | qcfc_corr τ = 0 | vcs_hoeff | mean Ĵ (τ = 0) |
|---|---|---|---|---|---|---|
| 0 (none) | 120 | .17 / .15 / .13 / .20 / .18 | .07 / .08 / .13 / .05 / .07 | .13 | 0 | −0.013 |
| 0 | 240 | .30 / .38 / .28 / .40 / .37 | .43 / .48 / .35 / .33 / .27 | .37 | 0 | −0.016 |
| 0 | all (475) | **.45 / .47 / .47 / .38 / .22** | **.52 / .43 / .43 / .32 / .30** | .40 | 0 | +0.004 |
| 0.5 | all | .23 / .38 / .42 / .37 / .22 | .30 / .40 / .30 / .27 / .18 | .22 | 0 | −0.006 |
| 1 (standard) | all | **.00** / .22 / .17 / .13 / .03 | .15 / .27 / .25 / .25 / .12 | .02 | 0 | −0.012 |

Exact null (subject i's BOLD with subject i + 1's FD, τ = 0): n = 120 → vcs .05–.08, hsic .08–.10, qcfc .05–.10; **n = 240 → vcs .20–.30, hsic .12–.18,
qcfc .12–.17**; n = all → vcs .07–.08, hsic .05–.10, qcfc .07–.12.  Ĵ strictly decreasing in α (τ = 0, n = all): 23 / 60 subjects.

## 2. Reading against the frozen rules
- **(i) false alarm ≤ 0.09 on the swapped null:** holds for `vcs_shift` at n = 120 and n = all (≤ 0.08); `hsic_shift` touches 0.10 at n = 120 and n = all;
  at **n = 240 every test — including the field's own QC-FC correlation, which has no fitted component — rejects 0.12–0.30**.  The tests are
  therefore *invalid at n = 240* by the rule, and because the baseline is affected too, the cause is the null construction or that window, not
  the critics.  Diagnosis is running (addendum 1 draft to follow); the n = 240 rows are not read.
- **(ii) power at α = 0, n = 240, τ = 0 ≥ 0.8:** 0.30 (invalid cell); the valid neighbour n = all gives 0.45 for VCS, 0.52 for HSIC, 0.40 for QC-FC.  Below the
  0.5 floor of the *does-not-hold* clause.
- **(iii) Ĵ strictly decreasing in α in ≥ 90 % of subjects:** 38 %.  The means do decrease (+0.004 → −0.006 → −0.012) but per subject the ordering
  is noise: held-out Ĵ is ≈ 0 ± 0.03 everywhere.
- **(iv) lag profile (reported):** at α = 0 the rejection rate is flat over τ = 0–2 (0.45–0.47) and decays by τ = 8 (6 s) to 0.22 — the expected shape.
- **(v) residual leakage after full cleaning (finding):** at α = 1 the *same-time* dependence is gone (VCS 0.00, QC-FC 0.02 — the linear model removes
  it by construction) but lagged dependence survives: VCS 0.22 / 0.17 / 0.13 at τ = 1 / 2 / 4, HSIC 0.27 / 0.25 / 0.25, against a null of ≤ 0.08.  The
  standard 24-parameter regression leaves motion effects at 1–3 s lags in roughly a fifth to a quarter of these subjects.

**Verdict: does not hold** (rule (ii) below 0.5; (iii) far below 90 %; (i) invalid at one window).  What the data say in substance: in a
low-motion cohort the leakage of head motion into the cleaned BOLD state is weak (held-out Ĵ ≈ 0), all three detectors — VCS, HSIC and the
field's QC-FC correlation — find it in 40–50 % of subjects at the full run length, and none is better than the others; the VCS test adds nothing
here except that it produces the same lag profile and the same residual-lag finding as HSIC with a bounded statistic.  The guaranteed test
never fires (Ĵ ≪ τ_{142,142} = 0.35).  This is a legitimate negative outcome of the pre-check, not a failure of the pipeline — except for the
n = 240 null, which is a defect to be explained before any of this is cited.

## 3. Not claimed / follow-ups
Only one dataset, resting state, one cleaning model, a low-motion cohort; no clinical statement.  Not run: the within-block conditional version
(block-wise circular shifts are not exact).  Follow-ups that would change the picture, all *proposed*: a high-motion cohort or task runs;
aCompCor / ICA-AROMA cleaning as the α = 1 reference; a block-bootstrap null if the diagnosis attributes the n = 240 excess to circular wrap.

## 4. P66b — the same tests with the exact circular-orbit null (addendum 1; GPU job 1012119; same prep, same seeds)
Addendum 1 traced the n = 240 anomaly to the frozen admissible-shift rule: on the 72-TR evaluation block only 13 distinct shifts satisfied
"≥ 30 TRs from either end", so the 200 null draws were ~15 copies of each and the p-value could only be 1/201 or ≥ 0.08 — every n = 240 rejection
in P66 was exactly 1/201, for the fit-free QC-FC correlation as much as for VCS.  P66b uses all L − 1 circular shifts once.

| cell | vcs_shift | hsic_shift | qcfc_corr | note |
|---|---|---|---|---|
| swapped null, all α, n = 120 / 240 / all | 0.00–0.10 | 0.02–0.08 | 0.03–0.10 | calibrated; two cells at 6/60 = 0.10 (VCS α = 1 n = 240; QC-FC α = 0.5 n = 240) |
| α = 0, n = 240, τ = 0 (rule ii) | **0.10** | 0.28 | 0.23 | |
| α = 0, n = all, τ = 0 / 1 / 2 / 4 / 8 | 0.43 / 0.45 / 0.37 / 0.37 / 0.13 | 0.47 / 0.37 / 0.32 / 0.17 / 0.08 | 0.30 / 0.15 / 0.13 / 0.17 / 0.05 | lag profile |
| α = 1, n = all, τ = 0 / 1 / 2 / 4 | 0.03 / 0.15 / 0.08 / 0.13 | 0.13 / 0.20 / 0.18 / 0.17 | 0.00 / 0.05 / 0.07 / 0.08 | residual after cleaning |
| Ĵ strictly decreasing in α (n = all) | 26 / 60 = 0.43 | | | rule iii |

**Re-read of the frozen grid on P66b.**  (i) holds up to two 6/60 cells (by the letter those two cells are invalid; the construction is now
sound).  (ii) 0.10 at n = 240 — the fitted critic has 72 evaluation TRs and no power there; HSIC 0.28, QC-FC 0.23; at full length VCS 0.43 vs
HSIC 0.47.  (iii) 43 %.  **Verdict unchanged: does not hold.**  What the corrected run adds: with a clean null the picture is the same but
sharper — in this low-motion cohort the leakage is weak, HSIC is the better detector at short windows because it fits nothing, the three
detectors agree at full length, the lag profile is flat over 0–3 s and gone by 6 s, and the residual lagged dependence after 24-parameter
cleaning is 0.08–0.15 (VCS) / 0.17–0.20 (HSIC) against a null of ≤ 0.08.  The family reading for row D is not changed by this unit: the
bounded statistic and the exact within-class shuffle were not the limiting factor here; the signal was.
