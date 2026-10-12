# R6 REG-P3 — secondary readouts (descriptive)

## Window-macro DEV success (%; per window over its 10 perturbations, mean over windows; mean ± sd over seeds)

| method | seeds | n windows | < 10.5 px | < 2.25 px |
|---|---|---|---|---|
| vcs | [0, 1, 2] | 74 | 99.3 ± 0.2 | 87.3 ± 1.6 |
| logistic | [0, 1, 2] | 74 | 99.2 ± 0.3 | 87.0 ± 1.4 |
| infonce | [0, 1, 2] | 74 | 99.4 ± 0.7 | 89.6 ± 1.1 |
| comir | [0, 1, 2] | 74 | 96.1 ± 1.3 | 85.0 ± 0.9 |
| mi | [0] | 74 | 93.9 | 82.7 |
| nmi | [0] | 74 | 94.1 | 82.7 |
| kernel | [0] | 74 | 87.4 | 75.3 |

## Energy profiles around the truth (first DEV window of 5 categories)

Offset of the profile maximum from the truth and the width of the >= 0.5 region of the min–max-normalised profile (θ in degrees; x in the profile's own 512-px-window pixels).  Seeded rows: all seeds.

| method | windows × seeds | θ: max offset 0 (count) | θ: median width (°) | x: max offset 0 (count) | x: median width (px) |
|---|---|---|---|---|---|
| vcs | 15 | 15 / 15 | 6.0 | 12 / 15 | 8.0 |
| logistic | 15 | 15 / 15 | 4.0 | 12 / 15 | 8.0 |
| infonce | 15 | 14 / 15 | 2.0 | 12 / 15 | 4.0 |
| comir | 15 | 15 / 15 | 4.0 | 12 / 15 | 6.0 |
| mi | 5 | 5 / 5 | 2.0 | 4 / 5 | 4.0 |
| nmi | 5 | 5 / 5 | 2.0 | 4 / 5 | 4.0 |
| kernel | 5 | 4 / 5 | 4.0 | 4 / 5 | 8.0 |
