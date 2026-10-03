# P112 — fixed-scale sensitivity screen (seed 0) — report — 2026-10-03

Pre-registration `P112_FIXED_SCALE_SENSITIVITY_PREREG_FROZEN_20261002.md`; results `P112_table.md` (56e14c3); follow-up `P112_ADDENDUM1_FOLLOWUP_FROZEN_20261003.md`.
Linear-val / kNN, selection split, 800 epochs, seed 0; each cell = that setting's G2 config with only (a, b) changed (κ = −b/a).

| setting | G2 (2, 0.5) | (1.5, 0.5) | (3, 0.5) | (2, 0.25) | (2, 0.75) |
|---|---|---|---|---|---|
| CIFAR-10 standard | 88.66 / 86.96 | 88.06 / 86.64 | 88.48 / 86.34 | **88.72** / 86.88 | 88.64 / 86.82 |
| CIFAR-10 strong aug | 86.86 / 84.90 | — | 86.92 / 84.64 | 87.88 / 84.56 | **88.22** / 85.74 |
| CIFAR-100 | 59.46 / 54.12 | — | 59.66 / 54.24 | **59.84** / 55.64 | 58.34 / 51.24 |

**Frozen reading.**  "The best fixed scale shifts" needs the best strong-aug / CIFAR-100 cell to be a different (a, κ) than the standard best **and** to beat its
setting's G2 seed-0 reference by ≥ 0.5 linear.
- CIFAR-10 standard: flat — every κ ∈ {0.25, 0.5, 0.75} at a = 2 within 0.08 of G2; a = 1.5 −0.60, a = 3 −0.18.  Best (2, 0.25) 88.72.
- **Strong augmentation: met** — best (2, 0.75) 88.22, +1.36 over G2-strong seed 0, and a different cell than the standard best.  (2, 0.25) is also +1.02;
  a = 3 is +0.06.  Both κ directions help at a = 2, so this is not a clean "threshold must move up" pattern.
- CIFAR-100: not met — best (2, 0.25) 59.84, +0.38 (< 0.5); (2, 0.75) −1.12, a = 3 +0.20.  No CIFAR-100 follow-up.
**Caveat (important):** the references are G2's *seed-0* values; G2-strong seed 0 (86.86) is its lowest seed (seeds 1–2: 87.70, 87.80; mean 87.45).
Against that mean, strong (2, 0.75) is +0.77 and (2, 0.25) +0.43.  The follow-up seeds 1–2 (addendum 1, running) give the paired comparison that decides it.
