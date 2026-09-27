# Pre-check B1 — closed-form linear-class critic vs trained tanh critic (P47/P48), final

Pre-registration: `P47_PRECHECK_B1_PREREG_FROZEN_20260927.md`.  Table: `P48_precheck_B1.md/.json` (CPU job 1010589; attempt 1010587 ran only the
first checkpoint because of an sbatch list-splitting bug, kept as `*_attempt1_partial`).  5 000 selection images, two train-distribution views,
2 500 fit / 2 500 eval pairs, K = 8 cyclic-shift negatives per split.  All J values on the eval split (lower bounds on S up to sampling).

| checkpoint (trained critic) | neural J_eval (same pairs) | best closed-form J_eval, no tanh (class, λ) | best J_eval of tanh(c·w*ᵀφ) (class, λ, c) | gap = neural − tanh-closed |
|---|---|---|---|---|
| P5_vcs_seed0 (concat-MLP, K = 1) | 0.899 | 0.808 (P3, 1e-4) | 0.887 (P3, 1e-4, c 3.5) | **+0.012** |
| P18_vcs_crit_cosine_seed0 (cosine) | 0.969 | 0.875 (P2/P3, 1e-4) | 0.972 (P2/P3, 1e-2) | **−0.003** |
| P24_vcs_cos_negdetach_seed0 (cosine + detach) | 0.927 | 0.862 (P3, 1e-4) | 0.952 (P3, 1e-4) | **−0.025** |
| P35_vcs_a5_views4_800ep_seed0 (final recipe) | 0.971 | 0.892 (P3, 1e-4) | 0.978 (P3, 1e-4/1e-2) | **−0.006** |
| P5_simclr_seed0 (no critic) | — | 0.917 (P2, 1e-6) | 0.944 (P3, 1e-4) | — |
Per class (P35): P1 0.957, P2 0.977, P3 0.978, H1 0.964 (tanh-wrapped, λ = 1e-4); ridge: P1 is ill-conditioned for P35 (closed-form J falls from
0.863 to 0.683 at λ = 1e-2 while the tanh-wrapped value stays at 0.947); P2/P3/H1 vary by < 0.01 across λ.

## Reading (pre-committed grid)
1. **Property holds.** For every VCS checkpoint the within-class gap is ≤ 0.02 — and for the three cosine-family checkpoints the closed-form
   linear critic wrapped in tanh with a single fitted scalar *exceeds* the trained critic's held-out J (by 0.003–0.025).  The between-family
   spread of neural J (0.90 for the MLP critic vs 0.97 for cosine critics on their own features) is 4–8× the within-class gap: the linear
   class is not the bottleneck; the trained critics are (nearly) at the linear-class optimum of their own features.
2. **Without tanh the raw linear closed form is 0.07–0.10 lower** (|T| > 1 on part of the pairs, clipped by the objective's quadratic), so as a
   *value* the frozen tanh output form matters; as a *lower bound on S* the raw closed form is still informative (≥ 0.81 on every VCS checkpoint,
   0.89 on the recipe; the frozen threshold was 0.5).
3. **Detach-trained critics are not J-optimal for their own features.** For P24 and P35 the closed form finds a better critic on held-out pairs
   (+0.025 / +0.006); detach changes the training operating point (threshold 0.82–0.90, positives unsaturated), which is what helped the
   *encoder*, at the cost of a slightly sub-optimal critic value — consistent with SSL §3.2 / P29 §4.
4. **Feature-class ordering:** P2/P3 (adding |z1 − z2|) ≥ P1 (z1⊙z2) > H1 (encoder features): the projector output is the more separable code,
   as the per-layer probes showed from the other side (z probes worse than h for classes, better for pair separability).
5. **A SimCLR embedding has a higher linear-class S lower bound (0.92–0.94) than every VCS embedding on its own two-view pairs** — the pair
   problem is easier on SimCLR's z; the VCS-trained encoders reach 0.86–0.89 closed-form / 0.95–0.98 with tanh.  This is an observation about
   separability, not about downstream quality.

## Consequence for the brief (appendix A, row B)
The closed-form critic is usable as a measurement device on frozen features (question 1 answered positively); question 2 (registration energy
surfaces) proceeds with the constructed-modality pairs (separate pre-registration).  Families opened: registration / extrinsic calibration /
cross-spectral cost / label-free embedding probes — subject to B2.

## Not claimed
Single split (5k selection images, one RNG draw of views); K = 8 cyclic shifts as the product sample (as in training); no statement about
tasks.  Evidence category: completed.
