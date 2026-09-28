# Task pre-check B-T2 [patch_ctf] — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on 60 real RGB–NIR pairs (EPFL IVRL scene dataset) — 2026-09-28T07:38:00Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10 random initial offsets per radius; success = < 1 px and < 1°; truth = the authors' registration.

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 0.93 | 0.95 | 11.6 | 16 / 14 | 0.75 / 0.77 / 0.67 / 0.48 | 0.02 |
| mi | 0.98 | 0.98 | 8.1 | 20 / 21 | 0.96 / 0.93 / 0.82 / 0.65 | 0.02 |
| nmi | 0.98 | 0.98 | 8.3 | 20 / 21 | 0.97 / 0.94 / 0.82 / 0.64 | 0.02 |
