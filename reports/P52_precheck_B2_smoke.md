# Pre-check B2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy (3 constructed modality pairs) — 2026-09-27T08:56:30Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 2 random initial offsets per radius; success = < 1 px and < 1°.

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 1.00 | 1.00 | 2.7 | 22 / 11 | 1.00 / 1.00 / 0.67 / 0.83 | 0.00 |
| mi | 1.00 | 1.00 | 2.7 | 24 / 24 | 1.00 / 1.00 / 1.00 / 0.83 | 0.00 |
| nmi | 1.00 | 1.00 | 2.3 | 24 / 24 | 1.00 / 1.00 / 1.00 / 0.83 | 0.00 |
