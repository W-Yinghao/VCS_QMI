# Pre-check B2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy (60 constructed modality pairs) — variant smooth4 — 2026-09-27T16:08:50Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10 random initial offsets per radius; success = < 1 px and < 1°.

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 0.83 | 0.95 | 5.0 | 23 / 17 | 0.74 / 0.73 / 0.61 / 0.47 | 0.00 |
| mi | 0.98 | 0.98 | 3.1 | 24 / 23 | 0.98 / 0.97 / 0.91 / 0.73 | 0.00 |
| nmi | 0.98 | 0.97 | 3.2 | 24 / 23 | 0.97 / 0.97 / 0.91 / 0.73 | 0.00 |
