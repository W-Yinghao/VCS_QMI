# Pre-check B2 — closed-form VCS J* vs histogram MI/NMI as a rigid-registration energy (60 constructed modality pairs) — variant patch — 2026-09-27T16:41:39Z

20000 pixel pairs per evaluation, identical overlap mask and samples for all measures; translation grid ±24 px step 2 (θ = 0); rotation ±30° step 1 (t = 0); Nelder–Mead ≤ 150 evaluations from 10 random initial offsets per radius; success = < 1 px and < 1°.

| measure | peak at truth (transl.) | peak at truth (rot.) | mean # local maxima (transl. grid) | median basin half-width px / deg | success R=5 / 10 / 20 / 30 | divergence R=30 |
|---|---|---|---|---|---|---|
| vcs | 1.00 | 1.00 | 6.6 | 22 / 16 | 0.99 / 0.99 / 0.85 / 0.64 | 0.00 |
| mi | 1.00 | 1.00 | 6.5 | 24 / 21 | 1.00 / 0.99 / 0.88 / 0.70 | 0.00 |
| nmi | 1.00 | 1.00 | 6.5 | 24 / 21 | 1.00 / 0.99 / 0.88 / 0.70 | 0.00 |
