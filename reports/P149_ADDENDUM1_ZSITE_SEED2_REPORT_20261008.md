# P149 addendum 1 — projector site z and a second encoder seed — report — 2026-10-08

Pre-registration `P149_ADDENDUM1_ZSITE_SEED2_PREREG_FROZEN_20261008.md`; results-only commit `4ee35c5` (`reports/P149A1_results_{z_seed1,h_seed2,z_seed2}.json`,
`slurm/p149a1_aggregate.sbatch` = the frozen P149 aggregator).  Same protocol as P149 (4 settings × 3 estimators × 3 init seeds, shared EVAL pools,
paired bootstrap 1 000); four CPU jobs, 28–36 min each.  Real images: J is a fit-limited lower reading, approximation bias never covered.

## 1. Common J at t 0 (VCS-MLP / JS-MLP / RFF; mean over init seeds)
| encoder family | h, seed 1 (P149) | h, seed 2 | z, seed 1 | z, seed 2 |
|---|---|---|---|---|
| C10 VCS (A-P3) | 0.828 / 0.824 / 0.321 | 0.846 / 0.841 / 0.342 | 0.967 / 0.968 / 0.730 | 0.966 / 0.969 / 0.742 |
| C10 SimCLR | 0.809 / 0.808 / 0.205 | 0.815 / 0.811 / 0.186 | 0.972 / 0.972 / 0.363 | 0.972 / 0.971 / 0.364 |
| C100 VCS (A-P3) | 0.867 / 0.868 / 0.213 | 0.872 / 0.874 / 0.244 | 0.967 / 0.969 / 0.738 | 0.969 / 0.972 / 0.727 |
| C100 SimCLR | 0.806 / 0.808 / 0.115 | 0.794 / 0.798 / 0.143 | 0.968 / 0.968 / 0.381 | 0.963 / 0.965 / 0.368 |

Saturation (|T| > 0.95 on P) at t 0: h 0.62–0.81, z 0.90–0.97.

## 2. Frozen reading
- **Resolution.** 47 of the 48 new fixture × estimator × site cells are numerically resolvable along t 0 → 0.25 → 1 (ordering probability ≥ 0.998
  everywhere).  The exception is VCS-MLP on the C10 VCS encoder at z, seed 1: ordering probability 0.998 but standardised gap 1.9 sd < 2 at the
  first step (ΔJ 0.010 near the top of the scale); JS-MLP on the same fixture passes (2.4 sd).  At z the first channel step moves the MLP readings by only
  0.010–0.040 because they are near saturation; the second step is clearly resolved (ΔJ 0.13–0.44, ≥ 10 sd).  RFF moves 0.17–0.30 at the first step.
- **Coordinate stability.** Orthogonal-refit drift intervals contain 0 in 34 of the 36 new cells (z seeds 1–2, h seed 2); the two exceptions are both
  routes on the seed-2 C10 SimCLR encoder at h (−0.022 [−0.047, −0.002] and −0.022 [−0.037, −0.006], ≈ 2 SE).  Across P149 + this addendum J is
  coordinate-stable to about 2 SE.
- **Model observation (P149 §4) is seed-stable at h:** VCS encoders read higher at t 0 (C10 0.846 vs 0.815, C100 0.872 vs 0.794 at seed 2; same
  sign at seed 1) and SimCLR encoders lose more under the channel (total ΔJ t 0 → 1 at seed 2: C10 0.374 vs 0.576, C100 0.498 vs 0.692; seed 1:
  0.390 vs 0.520, 0.514 vs 0.729) — same sign at both seeds and for both MLP routes.  At z the t 0 levels are equal (0.96–0.97, saturated), and
  the SimCLR representation again loses more under the channel (ΔJ t 0 → 1 = 0.43–0.47 vs 0.14–0.17 for the VCS encoders).  The caveat from P149 stands: the channel
  acts on scalar-standardised coordinates, so this is a property of the representation under this channel, not a ranking of SSL methods.

## 3. z vs h — what the comparison does and does not show
J at z exceeds J at h by 0.10–0.17 for every encoder and seed, although z = g(h) is a deterministic function of h and the data-processing
inequality gives S(z₁, z₂) ≤ S(h₁, h₂).  **This is not a violation**: the inequality holds for S, and J is a fit-limited lower reading — at h
(512-d per view) the critic fitted on 2 048 pairs is further below S than at z (128-d), where the SSL objective has already aligned the two
views.  Comparing sites by fitted J therefore compares estimation difficulty as much as dependence.  RFF shows the same direction more strongly
(0.12–0.34 at h vs 0.36–0.74 at z) and, at z only, separates the encoder families (VCS 0.73–0.74 vs SimCLR 0.36–0.38) where the MLP routes
saturate.
**Writing item W11:** cross-site (or cross-dimension) comparisons of fitted J must not be read as data-processing statements; only the
within-site channel ladder (same coordinates, same fit design) supports a DPI-direction reading.
