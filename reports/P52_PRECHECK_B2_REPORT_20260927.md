# Pre-check B2 — closed-form J* as a rigid-registration energy vs histogram MI / NMI (P51/P52), final

Pre-registration: `P51_PRECHECK_B2_PREREG_FROZEN_20260927.md`.  Table: `P52_precheck_B2.md/.json` (GPU job 1010624; smoke `P52_precheck_B2_smoke.md`
disclosed).  60 COCO val2017 images → constructed tent-map modality pairs (truth ω* = identity), 20 000 pixel pairs per evaluation with the same
overlap mask for all measures; translation grid ±24 px, rotation ±30°; Nelder–Mead ≤ 150 evaluations from 10 random initial offsets per radius.

| measure | peak at truth (transl. / rot.) | mean # local maxima on the translation grid | median basin half-width px / ° | success R = 5 / 10 / 20 / 30 | divergence |
|---|---|---|---|---|---|
| VCS closed-form J* (49-d Fourier class) | 1.00 / 1.00 | **10.2** | **20 / 15** | 0.99 / 0.96 / **0.77 / 0.54** | 0.00 |
| MI (32×32 histogram) | 1.00 / 1.00 | 6.5 | 24 / 21 | 1.00 / 0.99 / 0.88 / 0.70 | 0.00 |
| NMI | 1.00 / 1.00 | 6.5 | 24 / 21 | 1.00 / 0.99 / 0.88 / 0.70 | 0.00 |

## Reading (pre-committed grid)
**Does not hold.**  The closed-form J* surface has the truth as its global maximum in every image and never diverges, but it is *rougher* than the
histogram measures: 57 % more local maxima on the translation grid, a narrower basin in both translation and rotation, and a lower success rate
from large initial offsets (−0.11 at R = 20, −0.16 at R = 30; the frozen rule asked for +0.10).  With the sample size and feature class fixed,
the quadratic lower bound is a sharper but bumpier energy than MI/NMI on these pairs.

## What this does and does not say
- It is one feature class (tensor Fourier features, k, l ≤ 3, ridge 1e-3) and one constructed modality relation; a smoother class (fewer
  frequencies, larger ridge) or coarser pixel sampling would smooth the surface — but the brief's claim was about the estimator's structure,
  and the direct comparison at equal samples and mask goes to MI/NMI.  A follow-up could sweep the class smoothness, but that is tuning the
  energy, not testing a property.
- The closed form is cheap and exact (a 49×49 solve per evaluation) and J* at the truth is a valid S lower bound for the constructed relation;
  as a *measurement* on frozen features (B1) the property holds — as a *registration energy* it does not beat histogram MI.
- Family table (brief appendix A, row B): registration / extrinsic calibration / cross-spectral cost are **not** opened by this pre-check;
  label-free embedding probes (B1) remain.

Evidence category: completed.  Single constructed modality; rigid transforms only; no real cross-modal images.
