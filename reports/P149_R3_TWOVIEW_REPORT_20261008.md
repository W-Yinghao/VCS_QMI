# P149 (r3 R3-V0 + V1) — two-view dependence of frozen encoders: repeatability under invertible coordinates, resolution along a common channel — report — 2026-10-08

Pre-registration `P149_R3_TWOVIEW_PREREG_FROZEN_20261008.md`; results-only commit `f9aec1d` (`reports/P149_results.json`, `scripts/p149_aggregate.py`
over `outputs/P149_twoview/measure_<run>.{json,npz}`).  Four frozen seed-1 encoders (CIFAR-10 / CIFAR-100 × final VCS A-P3 / SimCLR), 20 000 base
images, two standard-law views each (law signature `22f3b58bfde64c92` asserted), representation h (512), FIT 2 048 + 2 048 pairs, TUNE 500 + 500, one
fixed EVAL of 2 000 P pairs and 2 000 independent Q pairs shared by every setting / estimator / seed; 3 init seeds; 1 000 paired-bootstrap reps
on the shared EVAL indices.  Four CPU jobs, 21–43 min each.  Real images: no oracle, the approximation bias of J is never covered.

## 1. Implementation checks
Transport check (identity-setting seed-0 critics, first layer / RFF frequencies transported onto rotated inputs): max |Δf| ≤ 3.3e-6 (MLP) and ≤ 2.6e-7
(RFF) on every fixture.  Permutation p (derangement null, B = 200) = 1 / 201 (the floor) in all 144 cells; the derangement null J is −0.02 to −0.87.

## 2. Common J (mean over 3 init seeds; refit sd; two-pool SE ≈ 0.008–0.017; Hoeffding radius 0.086 at δ 0.05)
| encoder | estimator | t 0 | orthogonal refit (drift, paired 95 % CI) | t 0.25 | t 1 | sat |T| > .95 at t 0 |
|---|---|---|---|---|---|---|
| C10 VCS | VCS-MLP | 0.828 (sd .008) | 0.818 (−0.010 [−0.023, +0.006]) | 0.739 | 0.438 | 0.69 |
| | JS-MLP | 0.824 (.006) | 0.809 (−0.015 [−0.028, −0.002]) | 0.734 | 0.428 | 0.62 |
| | RFF ridge-tanh | 0.321 (.010) | 0.312 (−0.009 [−0.047, +0.029]) | 0.181 | 0.037 | 0.06 |
| C10 SimCLR | VCS-MLP | 0.809 (.013) | 0.803 (−0.007 [−0.033, +0.013]) | 0.671 | 0.289 | 0.74 |
| | JS-MLP | 0.808 (.010) | 0.797 (−0.011 [−0.034, +0.005]) | 0.666 | 0.266 | 0.64 |
| | RFF | 0.205 (.023) | 0.222 (+0.017 [−0.037, +0.080]) | 0.106 | 0.031 | 0.01 |
| C100 VCS | VCS-MLP | 0.867 (.002) | 0.864 (−0.002 [−0.014, +0.009]) | 0.738 | 0.352 | 0.70 |
| | JS-MLP | 0.868 (.005) | 0.865 (−0.003 [−0.016, +0.012]) | 0.737 | 0.330 | 0.77 |
| | RFF | 0.213 (.026) | 0.218 (+0.005 [−0.032, +0.046]) | 0.109 | 0.026 | 0.01 |
| C100 SimCLR | VCS-MLP | 0.806 (.006) | 0.800 (−0.006 [−0.030, +0.014]) | 0.565 | 0.077 | 0.62 |
| | JS-MLP | 0.808 (.006) | 0.801 (−0.008 [−0.032, +0.016]) | 0.550 | 0.034 | 0.73 |
| | RFF | 0.115 (.014) | 0.104 (−0.011 [−0.071, +0.043]) | 0.053 | 0.011 | 0.00 |

## 3. Frozen reading (§4)
- **Repeatability.** Refit spread over init seeds at t 0 is 0.002–0.013 for the two MLP routes (≤ 1.3 SE) and 0.010–0.026 for RFF.  The invertible
  change of coordinates (independent Haar rotations of the two views, critics refitted) moves J by −0.015 to +0.017; the paired interval contains 0 in
  11 of 12 fixture × estimator cells (JS-MLP on C10 VCS: −0.015 [−0.028, −0.002], 1.5 SE).  J is coordinate-stable at this n to about one SE.
- **Resolution.** Every fixture × estimator is **numerically resolvable** along t 0 → 0.25 → 1: ordering probability 1.000 at every step (0.999 for
  RFF on C100 SimCLR at the second step), standardised gaps 3.6–30 pooled sd; no cell is estimation-sensitive (at every step the larger of the refit spread and the re-pairing sd is below the adjacent gap |ΔJ|,
  0.043–0.516).  The channel cannot increase S, and every estimator orders the three levels correctly.
- **Agreement of routes.** The direct VCS fit and the logistic fit read as common J agree within 0.005 at t 0 on all four encoders and within 0.043 at
  t 1 (largest on C100 SimCLR, 0.077 vs 0.034, where J is nearly exhausted); RFF ridge-tanh (D 1 024 on 1 024-d inputs) reads 0.11–0.32 at t 0, i.e. it is function-class-limited on this relation (its J is a valid but much
  looser lower reading), while it still orders the channel levels.
- **Context, not estimates.** Saturation is high at t 0 (62–77 % of P-pool outputs at |T| > 0.95): the t 0 reading is near the top of the scale, as in
  P122 (0.97–0.99 there with a different fit design).  The conservative lower reading J − Hoeffding radius is 0.72–0.78 for the MLP routes at t 0.
  **S_plugin is not a usable estimate here**: under the channel it separates from J (t 1: C100 VCS J 0.35 vs S_plugin 0.80; C100 SimCLR 0.08 vs 0.42) —
  the plug-in assumes a calibrated T, which a fitted critic on real images does not provide; P108's preference for the plug-in holds only in its
  calibrated synthetic cells.

## 4. Model observation (descriptive; no encoder was required to win)
At t 0 the VCS encoders' views are slightly more dependent than SimCLR's (C10 0.828 vs 0.809, C100 0.867 vs 0.806 with the VCS-MLP route), and under
the common channel the SimCLR representation loses dependence faster (C100: J at t 1 0.077 vs 0.352; C10: 0.289 vs 0.438).  Caveat: the channel acts on
scalar-standardised coordinates, so its effect depends on how each representation spreads its variance over the 512 dimensions; this is a property of
the representation under this channel, not a ranking of SSL methods, and rests on one encoder seed each.

## 5. Not claimed
Values of S itself (no oracle; J is fit-limited); z-site; other encoder seeds; the FIT sample was fixed (`refit_spread_resampled_pool` not run); any
link between these numbers and downstream accuracy.
