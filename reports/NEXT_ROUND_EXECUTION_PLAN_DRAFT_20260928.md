# Next round (estimator-centred) — execution plan DRAFT (2026-09-28), for the owner's review before any prereg is frozen

Source: `CS_QMI/VCS_QMI_Next_Round_Plan_v2.md` (exploration brief, not a run authorisation).  This draft turns its four lines into concrete
units with the data, grids and threshold numbers the brief leaves to the executor, and lists what has to be built.  Nothing here is frozen or
launched.  State of the repository at writing: main e384ccf (the brief cites 4729a1e); **S1 is already complete** (P67 frozen → P68 report, official
test set evaluated once) — its numbers are in `P68_FINAL_OFFICIAL_TEST_REPORT_20260928.md`; the gaps the brief quotes for P41 (3.5 / 3.3 / 2.4 / 1.3)
are the selection-split values, the test-set values are 4.0 / 3.5 / 2.3 / 1.5.  The brief's `oracle_analysis/RESULTS_20260928.md` is **not on the
server** (no such directory under `CS_QMI/`); either copy it over or I re-derive the oracle table as the first unit of E (cheap) and we compare.

## 0. Disciplines carried into every unit
Same critic class, optimiser budget and lr grid for every estimator; the negative construction named in every table (single shuffled negative /
K cyclic shifts / all in-batch); heavy-tailed estimators (NWJ, DV/MINE) report 1 % / 99 % quantiles, max deviation and divergence counts; **JS is in
every control set** (Deep-InfoMax form: E_P[−softplus(−f)] − E_Q[softplus(f)], same equal-weight mixture as VCS, statistic bounded by log 2);
cross-objective comparisons only on target-free axes: staircase separability (brief appendix B), test power, tail risk, learned-vs-oracle gap.

## 1. E line — controlled estimator benchmark (new module `src/vcs_estim/`; synthetic data; ≈ 40 GPU-h or CPU partition)
**Settings (truth analytic or high-precision MC).**  (a) 20-d Gaussian, per-coordinate correlation ρ, MI staircase 2 → 4 → 6 → 8 → 10 nats (Song & Ermon);
(b) the same through a coordinate-wise cubic map (MI invariant); (c) non-Gaussian: 10 independent 2-d pairs, each a two-component mixture with
opposite correlations (XOR-like), MI from a 2-d numerical integral × 10, staircase by mixture separation.  Sample sizes / batches N ∈ {64, 256, 1024}.
**Estimators.**  VCS (J, tanh critic; three negative constructions), JS (same negatives), InfoNCE (in-batch), NWJ, DV with EMA baseline (MINE),
SMILE (τ = 5).  Critic: joint MLP 2 × 256 ReLU (the standard benchmark critic) for all; Adam, lr ∈ {1e-4, 5e-4, 2e-3}; 20 k steps per staircase
(4 k per step); 5 seeds.  Oracle critics on the same batches where analytic (Gaussian: T* = tanh(PMI/2), JS*, InfoNCE with the true PMI).
**Metrics per cell.**  Own-target error; staircase separability (appendix B; IQR version for NWJ/DV); tail: quantiles, max |jump| between consecutive
logged values, divergence count (NaN or |estimate| > 3 × truth); learned / oracle ratio.
**Readings to freeze (proposed numbers).**  E1: VCS shows 0 divergences and 0 spikes (jump > 5 nats) in every cell; the claim narrows to "vs the
KL family" if JS and InfoNCE also show 0 *and* their separability is within 10 % of VCS's.  E2: at N = 64 and MI ≥ 6.5 nats, VCS separability ≥ 1.2 ×
InfoNCE's, ≥ SMILE's and ≥ JS's (the oracle ratio was 1.20); refuted if the ordering reverses in the learned-critic runs.  E3: |learned/oracle − 1| of
VCS < that of JS and of InfoNCE at every lr, and its range over lr < theirs; refuted if JS's gap ≤ VCS's.  E4: closed-form VCS and RuLSIF (α = ½,
same random-Fourier features) agree to < 1 % after the constant factor — otherwise stop and debug; neural VCS error on S < kernel RuLSIF's for
d ∈ {20, 50}; refuted if RuLSIF is level at every d ∈ {2, 5, 10, 20, 50}.
**To build.**  Data generators with truth; six estimators sharing one critic class; oracle critics; staircase runner with logging; RuLSIF
(closed-form kernel least squares with CV over σ, λ); tables.  Probe: one Gaussian staircase at N = 64 for all estimators before the grid.

## 2. R line — robustness (R1 synthetic; R2 / R3 on the COCO + CLIP adapter infrastructure of pre-check A)
**R1.**  20-d Gaussian at MI 4 nats; contaminate a fraction ε ∈ {0, 0.01, 0.05, 0.1, 0.2} of joint samples by (a) independent pairs, (b) outlier
pairs (x shifted by 5σ; y from the other end of the staircase), (c) heavy-tailed noise on y (t₂).  Estimators VCS / JS / InfoNCE / NWJ, learned
critics, same budget.  Metric: shift of the estimate divided by that estimator's own adjacent-step gap from the clean staircase ("resolution
units").  Reading: VCS shift ≤ 1 resolution unit per 0.1 of ε in every contamination type; refuted if learned JS is bounded alike and within 20 %.
**R2.**  Topic-pairing setting of P57 (animal source, indoor source): inject a mismatch fraction m ∈ {0, 0.1, 0.2, 0.4, 0.6} of random wrong captions
into the *training* pairs; methods VCS, InfoNCE, logistic (native and with the log K intercept correction), balanced logistic = JS; the P49 grid for
each; 3 seeds; metric: R@1 / R@5 on clean SRC-EVAL and TGT-EVAL vs m.  Reading: VCS's retrieval loss from m = 0 to 0.4 smaller than each competitor's
by ≥ 1 point; refuted if the curves coincide within 1 point or VCS is worse.
**R3.**  After R2 training at m = 0.2 and 0.4, rank the *training* pairs by each method's native score (VCS (1+T)/2, InfoNCE cosine, logistic logit,
JS score); AUROC against the injected labels, no clean calibration set.  Reading: VCS AUROC ≥ each competitor's + 0.02; refuted if within 0.02.
**To build.**  JS and log-K-corrected logistic in the adapter trainer; mismatch injection with recorded indices; the R1 contamination generators.

## 3. T line — conditional dependence test with fair controls (extends the D machinery; ≈ 10 GPU-h)
**T1.**  Colour and blur nuisance families, both encoders, conditional (within-class) test, **N re-drawn per repeat**, n ∈ {200, 500, 1000, 2000}, R = 100,
same fit / selection / evaluation splits and 200 permutations for every test.  Controls: HSIC with per-class median bandwidth; HSIC with a learned
deep kernel (features trained on the fit split to maximise the test statistic's power, Liu et al. 2020 style, selected on the validation split);
C2ST with the mean logit as statistic; a JS statistic from the same critic class as VCS (fit on FIT, early-stopped on VAL, permutation null on EVAL).
Reading: VCS power ≥ every control − 0.05 at each (family, n) with s at the smallest detectable strength of the family; refuted if the JS statistic is
within 0.05 everywhere (then the effect is "learned critic", not VCS).  Level check: swapped-N and s = 0 nulls ≤ 0.065 at R = 1000 for every test.

## 4. S line
**S1 — done** (P68).  **S2 — standard protocol.**  Recommended: port VCS into solo-learn (MIT) and run its CIFAR-10 ResNet-18 / 1000-epoch / LARS recipe
for VCS and SimCLR (the control confirms the recipe), 3 seeds each ≈ 6 × 12 h; the claim (gap < 1.3 points) is read on the protocol's own metric.
**Discipline conflict to resolve before freezing:** the solo-learn protocol reports official-test accuracy, but the brief's common discipline exempts
only S1; either grant S2 an explicit exemption (one evaluation per run, at the end, pre-registered) or run S2 on the selection split.  **S3** — the
"no more objective tuning" conclusion is added to the technical note as stated.

## 5. Order and budget
1. E probe (Gaussian staircase, N = 64, all estimators) and the oracle re-derivation → prereg E → grid (≈ 40 GPU-h).  2. T1 (reuses D code; ≈ 10 GPU-h).
3. R1 (with E's generators), then R2 / R3 (≈ 15 GPU-h).  4. S2 after the owner's decision on the test-set exemption (≈ 70 GPU-h).  Every unit: frozen
prereg (P-series numbering continues at P69), probe, fleet, results-only commit, report.

## 6. Questions for the owner before the first freeze
1. Copy `oracle_analysis/` to the server, or accept a re-derivation as E's first unit?  2. Are the proposed threshold numbers in §1–3 acceptable
(they follow the brief's falsification conditions; the margins 1.2× / 0.05 / 0.02 / 1 point are mine)?  3. S2: exemption for the official test set,
or selection split?  4. "Learned-kernel HSIC" = deep-kernel HSIC selected on a validation split — acceptable as the control?  5. Anything from the
proposed-only list of the final summary (momentum queue, 16× control) to fold in, or is that closed?
