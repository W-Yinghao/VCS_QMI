# P131 — CIFAR-10-C-style, regenerated locally: results

21 runs; missing: none.  Primary = mean accuracy over the 12 implemented standard corruptions x 5 severities (omitted: motion_blur, snow, frost, spatter (extra set)).

| method | seeds | clean dev-val | mCA (primary) | drop | noise | blur | weather | digital | extra |
|---|---|---|---|---|---|---|---|---|---|
| A-P3 | 5 | 89.02 | 70.58 | 18.44 | 51.89 | 70.40 | 83.14 | 78.46 | 76.71 |
| SimCLR | 5 | 88.31 | 70.63 | 17.68 | 52.96 | 70.41 | 81.90 | 78.40 | 76.27 |
| G2 | 5 | 88.69 | 71.31 | 17.38 | 54.38 | 71.27 | 83.03 | 78.17 | 77.46 |
| JS-AP3 | 3 | 88.81 | 71.94 | 16.87 | 54.92 | 71.83 | 83.21 | 79.16 | 77.70 |
| recipe VCS | 3 | 87.01 | 66.23 | 20.78 | 46.34 | 65.25 | 80.24 | 74.88 | 71.85 |

## Paired contrasts (by seed; labels as P114: close |Δ| < 0.3, clear |Δ| ≥ 0.3 with the 95 % t interval excluding 0, else inconclusive)

| contrast | metric | n | mean Δ | 95 % CI | label |
|---|---|---|---|---|---|
| A-P3 − SimCLR | mca | 5 | -0.04 | [-0.80, +0.71] | close |
| A-P3 − SimCLR | drop | 5 | +0.76 | [+0.16, +1.36] | clear |
| A-P3 − SimCLR | family:noise | 5 | -1.08 | [-2.98, +0.83] | inconclusive |
| A-P3 − SimCLR | family:blur | 5 | -0.01 | [-0.67, +0.64] | close |
| A-P3 − SimCLR | family:weather | 5 | +1.24 | [+0.80, +1.69] | clear |
| A-P3 − SimCLR | family:digital | 5 | +0.06 | [-0.37, +0.50] | close |
| A-P3 − JS-AP3 | mca | 3 | -1.36 | [-2.65, -0.07] | clear |
| A-P3 − JS-AP3 | drop | 3 | +1.57 | [-0.00, +3.14] | inconclusive |
| A-P3 − JS-AP3 | family:noise | 3 | -3.18 | [-6.22, -0.14] | clear |
| A-P3 − JS-AP3 | family:blur | 3 | -1.33 | [-2.83, +0.18] | inconclusive |
| A-P3 − JS-AP3 | family:weather | 3 | -0.03 | [-0.16, +0.09] | close |
| A-P3 − JS-AP3 | family:digital | 3 | -0.67 | [-1.90, +0.56] | inconclusive |
| G2 − SimCLR | mca | 5 | +0.68 | [-0.17, +1.53] | inconclusive |
| G2 − SimCLR | drop | 5 | -0.30 | [-0.99, +0.39] | close |
| G2 − SimCLR | family:noise | 5 | +1.42 | [-1.44, +4.28] | inconclusive |
| G2 − SimCLR | family:blur | 5 | +0.85 | [+0.00, +1.70] | clear |
| G2 − SimCLR | family:weather | 5 | +1.13 | [+0.68, +1.58] | clear |
| G2 − SimCLR | family:digital | 5 | -0.23 | [-0.53, +0.07] | close |
| A-P3 − recipe VCS | mca | 3 | +4.35 | [+2.25, +6.46] | clear |
| A-P3 − recipe VCS | drop | 3 | -2.33 | [-5.57, +0.90] | inconclusive |
| A-P3 − recipe VCS | family:noise | 3 | +5.40 | [+0.42, +10.37] | clear |
| A-P3 − recipe VCS | family:blur | 3 | +5.26 | [+3.03, +7.49] | clear |
| A-P3 − recipe VCS | family:weather | 3 | +2.94 | [+1.51, +4.37] | clear |
| A-P3 − recipe VCS | family:digital | 3 | +3.60 | [+2.72, +4.48] | clear |
