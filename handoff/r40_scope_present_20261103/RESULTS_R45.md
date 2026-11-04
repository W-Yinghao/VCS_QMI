Publication copy (sample names aliased; numbers and conclusions unchanged).

# R45 results (numbers only)

> **Readout note (2026-11-03, R44 audit):** numbers in this file use the native IN²R per-batch centring over the sequential native layout (true image-caption pairs share a batch mean); JOINT students since R39 were also trained with that layout. Layout-free re-evaluations of the same checkpoints: CORRECTED_READOUTS.md; audit: R44_AUDIT.md; layout-free retraining: RESULTS_R47.md.

COCO IN²R-native parents, N-JOINT 16000 updates (cosine over 16000, from the original H + fresh zero-output head, training seed 1001) vs the R43-A 8000-update N-JOINT and N-frozen cells on the same parent. DEV = native 1K split (1000 images × 5000 captions), two-stage (frozen-H top-100 reranked). TEST not read.

| parent | H | obj | N-frozen (R43-A) | N-JOINT 8k (R43-A) | 16k @8000 | 16k @12000 | 16k @16000 | 16k@16000 − 8k | 16k@16000 − frozen |
|---|---|---|---|---|---|---|---|---|---|
| C60_IN2R_H1001 | 538.94 | J | 547.00 | 558.16 | 564.02 | 567.16 | 567.88 | 9.72 | 20.88 |
| C60_IN2R_H1001 | 538.94 | BCE | 547.04 | 556.94 | 564.22 | 567.38 | 568.24 | 11.30 | 21.20 |
| C60_IN2R_H1001 | 538.94 | NCE | 547.12 | 560.32 | 564.48 | 567.62 | 567.78 | 7.46 | 20.66 |
| C80_IN2R_H1001 | 526.52 | J | 536.72 | 545.08 | 549.48 | 552.78 | 553.74 | 8.66 | 17.02 |
| C80_IN2R_H1001 | 526.52 | BCE | 536.46 | 544.26 | 549.00 | 551.14 | 552.56 | 8.30 | 16.10 |
| C80_IN2R_H1001 | 526.52 | NCE | 536.82 | 543.42 | 548.42 | 550.78 | 551.90 | 8.48 | 15.08 |

J−BCE / J−NCE at 16000:

| parent | J−BCE | J−NCE |
|---|---|---|
| C60_IN2R_H1001 | -0.36 | 0.10 |
| C80_IN2R_H1001 | 1.18 | 1.84 |
