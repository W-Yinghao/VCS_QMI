# Task pre-check B-T2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on 3 real RGB–NIR pairs (EPFL IVRL scene dataset) — 2026-09-27T15:19:03Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 2 random initial offsets per radius; success = < 1 px and < 1°; truth = the authors' registration.

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 1.00 | 1.00 | 12.0 | 12 / 16 | 0.67 / 0.83 / 0.17 / 0.17 | 0.00 |
| mi | 1.00 | 1.00 | 5.0 | 18 / 27 | 1.00 / 1.00 / 0.67 / 0.17 | 0.00 |
| nmi | 1.00 | 1.00 | 5.3 | 18 / 27 | 1.00 / 1.00 / 0.83 / 0.17 | 0.00 |
