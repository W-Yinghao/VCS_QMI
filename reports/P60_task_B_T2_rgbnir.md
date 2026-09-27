# Task pre-check B-T2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on 60 real RGB–NIR pairs (EPFL IVRL scene dataset) — 2026-09-27T16:13:15Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10 random initial offsets per radius; success = < 1 px and < 1°; truth = the authors' registration.

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 0.93 | 0.98 | 16.4 | 15 / 12 | 0.78 / 0.68 / 0.50 / 0.32 | 0.00 |
| mi | 0.98 | 0.98 | 8.1 | 20 / 21 | 0.95 / 0.89 / 0.71 / 0.50 | 0.00 |
| nmi | 0.98 | 0.98 | 8.3 | 20 / 21 | 0.95 / 0.89 / 0.70 / 0.52 | 0.00 |
