# Task pre-check B-T2 [patch_ctf] — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy on 60 real IXI PD/T2 mid-axial slice pairs — 2026-09-28T10:13:03Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10 random initial offsets per radius; success = < 1 px and < 1°; truth = identity (same acquisition).

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 1.00 | 1.00 | 1.1 | 24 / 14 | 1.00 / 1.00 / 1.00 / 1.00 | 0.00 |
| mi | 1.00 | 1.00 | 1.0 | 24 / 30 | 1.00 / 1.00 / 1.00 / 1.00 | 0.00 |
| nmi | 1.00 | 1.00 | 1.0 | 24 / 30 | 1.00 / 1.00 / 1.00 / 1.00 | 0.00 |
