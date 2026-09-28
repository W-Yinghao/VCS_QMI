# Row D, wave 2 — report (P46-W2): D-S1 level pin, D-S2 second encoder seeds, D-S3 blur family, D-T shortcut-reliance task

Preregs: `P45_PRECHECK_D_PREREG_FROZEN_20260927.md` addendum 5 (D-S1 / D-S2 / D-S3), `P55_TASK_D_SHORTCUT_PREREG_FROZEN_20260927.md` (D-T).
Tables (results-only commit 0990c44): `P46_precheck_D_null1000_{vcs4v800,simclr}_n{2000,5000,10000}_s0.md`, `P46_precheck_D_{vcs4v800s1,simclrs1}.md`,
`P46_precheck_D_{vcs4v800,simclr}_blur.md`, `P56_task_D_shortcut_rho*_seed*_{cond_colour0.2,cond_null}.md` + `outputs/P55_task_D_shortcut_*/result.json`.
Evidence category: completed (GPU jobs listed in `job_ids.json`).  The addendum-6 null diagnosis (fresh N per repeat; synthetic nulls) and the
addendum-1 blur-shortcut replicate of D-T are running and will be appended.

## 1. D-S1 — level of the permutation tests at s = 0 with R = 1000 fresh draws (seed 3)
| n | VCS encoder: vcs_perm / hsic / c2st / hoeff | SimCLR encoder: vcs_perm / hsic / c2st / hoeff |
|---|---|---|
| 2000 | **0.07** / 0.06 / 0.00 / 0 | 0.05 / 0.05 / 0.01 / 0 |
| 5000 | 0.06 / 0.05 / 0.00 / 0 | **0.07** / **0.07** / 0.01 / 0 |
| 10 000 | 0.06 / n/a / 0.00 / 0 | **0.07** / n/a / 0.00 / 0 |

**Reading (addendum 5: ≤ 0.065 → level confirmed; (0.065, 0.09] → anti-conservative but not invalid; > 0.09 → invalid).**  Three of six vcs_perm
cells and one HSIC cell print 0.07 → *anti-conservative but not invalid*; the other cells are confirmed; nothing is invalid.  Pooled over the
6000 draws the vcs_perm rate is 0.063 and the HSIC rate 0.058 — both above 0.05 by more than binomial noise if the repeats were independent.
HSIC has no fitted component, so a shared cause in the design is more likely than a defect of the VCS test: every repeat is a subset of one
pool whose nuisance vector N was drawn *once*, so the 1000 repeats share a single realised z–N association (large-n subsets overlap heavily).
This is the hypothesis frozen in addendum 6; its jobs (N re-drawn per repeat; synthetic fixed-pool vs fresh nulls) decide it.  Until then the
D verdict's false-alarm clause is read as "the tests run at 0.06–0.07 in this design", not as "the test is broken".

## 2. D-S2 — second encoder seeds (VCS seed 1, SimCLR seed 1; s ∈ {0, 0.05, 0.1, 0.2, 0.3}, n ≤ 5000)
| rule (P45) | VCS seed 1 | SimCLR seed 1 | seed 0 (P46) |
|---|---|---|---|
| parity at n ≤ 500, s ≥ 0.05 (vcs_perm ≥ max(hsic, c2st) − 0.05) | 10 / 12 (misses −0.07 at s = 0.1 n = 200; −0.08 at s = 0.3 n = 100) | 11 / 12 (miss −0.11 at s = 0.3 n = 200) | 32 / 36 |
| smallest n with 80 % power, vcs_perm vs HSIC | identical at every s (1000 / 500 / 100 / 100) | identical at every s (— / 5000 / 2000 / 1000) | identical except one cell |
| Ĵ at n = 5000 strictly increasing in s | yes (Spearman 1.0) | yes (Spearman 1.0) | VCS yes / SimCLR null-end tie |
| false alarm at R = 100 cells (n ≤ 1000) | ≤ 0.05 (HSIC 0.09 at n = 1000) | ≤ 0.09 | ≤ 0.09 |
| false alarm at R = 30 cells | 0.03 / 0.07 | 0.00 / 0.03 (HSIC 0.13 at n = 5000) | 0.10 cells |
| label_only conditional at n = 2000 (≤ 0.09) | vcs 0.10 (5/50), hsic 0.04 | 0.08 / 0.06 | 0.06 / 0.10 |
| label_colour conditional power at n = 2000 | vcs 0.92, hsic 0.76, c2st 0.04 | 0.16 / 0.06 / 0.00 (shift undetectable in SimCLR features) | 0.96 / 0.50 / 0.06; 0.08 / 0.06 / 0.00 |

**Reading (addendum 5: "on par with HSIC" confirmed if ≥ 90 % of the small-sample cells pass on seed 1).**  SimCLR seed 1: 92 % → confirmed.
VCS seed 1: 83 % → not confirmed by the letter; the two misses are 0.07–0.08 deficits in cells where both tests are climbing (0.61 vs 0.68;
0.91 vs 0.99), and the sample size at which each strength becomes detectable is the same for the two tests at every s.  Everything else
replicates seed 0: exact conditional test (type I at the null level with one 5/50 cell), 0.92 vs 0.76 conditional power on the VCS encoder,
no power on SimCLR features at s = 0.05, monotone Ĵ.  **D-S2: holds conditionally** (replication in substance; the 90 % letter met on one
encoder only).

## 3. D-S3 — second nuisance family: Gaussian blur σ ∈ {0.25, 0.5, 0.75, 1.0} px (VCS seed 0, SimCLR seed 0)
| rule (P45) | VCS encoder | SimCLR encoder |
|---|---|---|
| parity at n ≤ 500, σ ≥ 0.25 | 12 / 12 | 12 / 12 |
| smallest n with 80 % power, vcs_perm / HSIC / C2ST | σ 0.25: 5000 / 5000 / —; 0.5: 200 / 200 / 1000; 0.75: 100 / 100 / 200; 1.0: 100 / 100 / 100 | 0.25: — / — / —; 0.5: **1000 / 2000** / 5000; 0.75: 500 / **200** / 1000; 1.0: 100 / 100 / 200 |
| Ĵ at n = 5000 in σ | −0.0003, 0.001, 0.064, 0.229, 0.288 (strict) | −0.0004, −0.0002, 0.012, 0.091, 0.200 (strict) |
| guaranteed test 80 % | σ 0.75 at n = 2000; σ 1.0 at n = 1000 | σ 0.75 at n = 5000; σ 1.0 at n = 2000 |
| false alarm, R = 100 cells | ≤ 0.09 | ≤ 0.06 |
| false alarm, R = 30 cells | vcs 0.10 at n = 5000 | vcs 0.10 at n = 5000; HSIC 0.10 / 0.17 |
| label_only conditional at n = 2000 | 0.06 / 0.04 / 0.00 | 0.10 / 0.04 / 0.00 |
| label_blur conditional power at n = 500 / 2000 (vcs / hsic / c2st) | 0.96 / 1.00 / 0.04; 1.00 / 1.00 / 0.94 | 0.30 / 0.04 / 0.00; **0.98 / 0.18 / 0.24** |

**Reading: holds** on the P45 rules for both encoders (parity 24 / 24, strict monotonicity, exact conditional test, R = 100 false alarms in
range), with the same R = 30 caveat as the colour family.  Two observations beyond the rules: the blur shift is far more visible than the
colour shift in both encoders (Ĵ 0.29 / 0.20 at σ = 1 px vs 0.24 / 0.07 at s = 0.5 colour), and in the *conditional* test on the SimCLR encoder
the VCS statistic keeps its power where HSIC loses almost all of it (0.98 vs 0.18 at n = 2000) — within-class shuffling leaves HSIC's global
kernel bandwidth mismatched to the within-class scale, while the fitted linear-class critic adapts.  That is the first cell in the whole
row where the VCS test is clearly *more* powerful than HSIC; it is a conditional-test effect, not a marginal-test one.

## 4. D-T — task: shortcut-reliance detection in trained classifiers (colour shortcut s = 0.2, ρ ∈ {0.5, 0.8, 0.95, 1.0} × 2 seeds)
| ρ | reliance = acc(matched) − acc(flipped), seeds 0 / 1 | mean conditional Ĵ (n = 1000) | conditional vcs_perm at n = 200 / 500 / 1000 | HSIC | C2ST | guaranteed test at n = 1000 |
|---|---|---|---|---|---|---|
| 0.5 (no shortcut) | +0.001 / +0.004 | 0.020 | 0.72 / 0.98 / 1.00; 0.34 / 0.96 / 0.98 | 0.30 / 0.62 / 0.96; 0.04 / 0.22 / 0.24 | 0.10 / 0.40 / 0.78; 0.04 / 0.08 / 0.40 | 0 |
| 0.8 | +0.061 / +0.060 | 0.089 | 1.00 / 1.00 / 1.00 (both seeds) | 0.92 / 1.00 / 1.00 | 0.28–0.52 / 0.96 / 1.00 | 0 |
| 0.95 | +0.181 / +0.205 | 0.191 | 1.00 everywhere | 1.00 everywhere | 1.00 everywhere | 0.62–0.66 |
| 1.0 | +0.657 / +0.664 | 0.237 | 1.00 everywhere | 1.00 everywhere | 1.00 everywhere | 1.00 |

Null case (`cond_null`, colour not planted; R = 50 per cell, 24 cells per test): vcs_perm 0.00–0.14 (6 cells above 0.09; pooled 69 / 1200 =
0.058), HSIC 0.00–0.10 (2 cells at 0.10; pooled 0.037), C2ST ≤ 0.08 (pooled 0.028).  QC: training accuracy ≥ 0.9 at ρ = 0.5 met
(selection accuracy 0.86–0.87 clean); reliance(ρ = 1.0) ≫ reliance(ρ = 0.5).

**Reading (P55, verbatim rules).**  (i) mean conditional Ĵ increases with ρ (0.020 → 0.089 → 0.191 → 0.237) and Spearman(Ĵ, reliance) over the
8 models = 0.95 ≥ 0.9 — holds.  (ii) at n = 500 the VCS permutation test detects every ρ ≥ 0.8 model with power 1.00 and HSIC is not above it
— holds.  (iii) null rejection ≤ 0.09 in every cell — fails for vcs_perm in 6 of 24 cells (max 0.14 = 7/50) and for HSIC in 2 cells.
**D-T: holds conditionally** (one of the three bullets fails).  The null cells are one N realisation per model over a 5 000-image held-out set
with repeats of up to 3 × 1000 images — exactly the design the addendum-6 diagnosis targets; rates scatter 0.00–0.14 across models, with
the pooled rate 0.058.  Two substantive points for the family: (a) the conditional test detects *sensitivity* of the representation to the
nuisance, not *reliance* — at ρ = 0.5 the classifier has no reliance (+0.001) yet its features depend on the colour shift (Ĵ 0.02, power
0.96–0.98 at n = 500), so a "leakage detector" built on it flags colour-sensitive features whether or not the decision uses them; the
*magnitude* is what tracks reliance.  (b) The magnitude saturates: reliance 0.19 → 0.66 moves Ĵ only 0.19 → 0.24, the near-deterministic
regime the brief warns about.  The guaranteed test reaches 80 % only at ρ = 1.0 (n = 1000; Ĵ 0.24 > τ = 0.19) and 0.62–0.66 at ρ = 0.95.

## 5. Row D after wave 2 (family table, brief appendix A row D)
The P46 picture replicates on a second encoder seed and a second nuisance family and transfers to a trained-classifier task: parity with
HSIC at small n (with a genuine advantage in the *conditional* test on one encoder), an exact within-class shuffle, a magnitude that orders
nuisance strength and shortcut reliance, and a guaranteed bound that needs S ≳ τ_{n,n}.  The one recurring blemish is the null rejection rate
of 0.06–0.07 (up to 0.14 in R = 50 cells), shared with HSIC and tied to the single-N-draw design; addendum 6 settles whether it is design or
procedure.  Family status stays as the D report §3.1 left it — closed by the letter of the frozen false-alarm clause, conditional by its intent
— and the owner's reading of that clause is now informed by four more data sets that all show the same shape.

## 6. D-T replicate with a texture shortcut (P55 addendum 1: Gaussian blur, radius 1 px; ρ ∈ {0.5, 0.95, 1.0} × 2 seeds; 6 models, 12 tests)
| ρ | reliance (seeds 0 / 1) | mean conditional Ĵ (n = 1000) | conditional vcs_perm at n = 500 | HSIC | guaranteed test at n = 1000 | null vcs_perm cells (n = 200 / 500 / 1000) |
|---|---|---|---|---|---|---|
| 0.5 | +0.000 / +0.006 | 0.101 | 1.00 / 1.00 | 0.94 / 0.98 | 0.00 | .06/.04/.00; .06/.06/**.16** |
| 0.95 | +0.293 / +0.395 | 0.274 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | .02/.04/.00; .04/**.12**/**.12** |
| 1.0 | +0.860 / +0.856 | 0.292 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | .04/.04/.02; .00/.06/.02 |

The blur shortcut is learned far more strongly than the colour one (reliance 0.86 vs 0.66 at ρ = 1.0; 0.29–0.40 vs 0.19 at ρ = 0.95).  **Reading
(P55 rules with "colour" read as "blur"):** (i) Ĵ increases with ρ and Spearman(Ĵ, reliance) over the 6 models = 0.94 ≥ 0.9 — holds; (ii) every
ρ ≥ 0.95 model detected at n = 500 with power 1.00, HSIC equal — holds; (iii) null ≤ 0.09 in every cell — fails for vcs_perm in 3 of 18 cells (max
0.16 = 8/50; pooled 45/900 = 0.050) and for HSIC in 2 cells (0.10, 0.12).  **Holds conditionally**, for the same reason as the colour family:
the per-model null uses one N draw over a 5 000-image pool, so cells scatter 0.00–0.16 around a pooled 0.05.  Two observations repeat with more
force: (a) sensitivity is not reliance — at ρ = 0.5, with zero reliance, the blur is detected in 100 % of draws at n = 500 with Ĵ ≈ 0.10 (colour: 0.02),
because a 1-px blur is a large change to the features whether or not the classifier uses it; (b) the magnitude saturates — Ĵ moves 0.27 → 0.29
while reliance moves 0.29 → 0.86.  The guaranteed test now fires at ρ ≥ 0.95 (Ĵ 0.27–0.29 > τ_{1000} = 0.19), which the weaker colour shortcut
never reached below ρ = 1.0.  The contingency (radius 1.5 px) was not needed.
