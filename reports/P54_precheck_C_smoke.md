# Pre-check C — batch decoupling (pairing = topic; frozen CLIP features; identity adapters) — 2026-09-27T10:50:12Z

Fixed updates: 30 steps at every B; fixed exposure: 1 epochs at every B.  lr selected per (method, B, axis) on SRC-CAL from [0.001]; 1 seeds. Balanced error at the SRC-CAL quantile threshold (FNR 5 %).  Retention = value(B) / value(256) for R@1 and for accuracy = 1 − balanced error.

## fixed_updates

| method | B | steps | SRC R@1 (ret.) | SRC 1−bal.err (ret.) | TGT R@1 (ret.) | TGT 1−bal.err (ret.) | TGT ECE(Platt) | J (SRC/TGT) |
|---|---|---|---|---|---|---|---|---|
| vcs | 16 | 30 | 0.371 (1.03) | 0.723 (0.90) | 0.370 (1.00) | 0.566 (0.93) | 0.0873 | 0.356/0.079 |
| vcs | 256 | 30 | 0.361 (1.00) | 0.806 (1.00) | 0.369 (1.00) | 0.608 (1.00) | 0.0718 | 0.473/0.136 |
| infonce | 16 | 30 | 0.411 (1.03) | 0.738 (0.91) | 0.394 (1.02) | 0.586 (0.93) | 0.0859 | — |
| infonce | 256 | 30 | 0.398 (1.00) | 0.813 (1.00) | 0.387 (1.00) | 0.633 (1.00) | 0.0659 | — |
| logistic | 16 | 30 | 0.295 (0.87) | 0.544 (0.72) | 0.262 (0.78) | 0.541 (0.88) | 0.0987 | — |
| logistic | 256 | 30 | 0.340 (1.00) | 0.755 (1.00) | 0.336 (1.00) | 0.618 (1.00) | 0.0992 | — |

## fixed_exposure

| method | B | steps | SRC R@1 (ret.) | SRC 1−bal.err (ret.) | TGT R@1 (ret.) | TGT 1−bal.err (ret.) | TGT ECE(Platt) | J (SRC/TGT) |
|---|---|---|---|---|---|---|---|---|
| vcs | 16 | 125 | 0.313 (0.77) | 0.801 (1.13) | 0.302 (0.76) | 0.638 (1.15) | 0.0915 | 0.483/0.142 |
| vcs | 256 | 7 | 0.409 (1.00) | 0.712 (1.00) | 0.396 (1.00) | 0.557 (1.00) | 0.1106 | 0.323/0.089 |
| infonce | 16 | 125 | 0.345 (0.74) | 0.811 (1.10) | 0.347 (0.80) | 0.638 (1.08) | 0.0641 | — |
| infonce | 256 | 7 | 0.465 (1.00) | 0.737 (1.00) | 0.433 (1.00) | 0.592 (1.00) | 0.0885 | — |
| logistic | 16 | 125 | 0.260 (0.68) | 0.748 (1.24) | 0.218 (0.59) | 0.608 (1.04) | 0.0842 | — |
| logistic | 256 | 7 | 0.380 (1.00) | 0.602 (1.00) | 0.368 (1.00) | 0.584 (1.00) | 0.1045 | — |

