# P65 addendum 1 — null-construction diagnosis on the subject-swapped null (alpha = 0, lag 0, 60 subjects, 200 draws, delta = 0.05)

Admissible circular shifts of the evaluation block under the frozen rule (min_shift 30; fallback to all shifts when L - 30 <= 30):

| L_eval | distinct shifts | range | fallback to full group |
|---|---|---|---|
| 36 | 35 | 1–35 | True |
| 60 | 59 | 1–59 | True |
| 72 | 13 | 30–42 | False |
| 90 | 31 | 30–60 | False |
| 142 | 83 | 30–112 | False |

Rejection rate over subjects (true independence by construction; nominal 0.05):

| n | L_eval | distinct in 200 draws | null construction | qcfc | hsic | vcs |
|---|---|---|---|---|---|---|
| 200 | 60 | 57.0 | eval_restricted | 0.08 | 0.08 | 0.08 |
| 200 | 60 | 57.0 | eval_full | 0.05 | 0.08 | 0.08 |
| 200 | 60 | 57.0 | orbit_exact | 0.05 | 0.08 | 0.08 |
| 200 | 60 | 57.0 | window | 0.07 | 0.10 | 0.10 |
| 200 | 60 | 57.0 | block_boot | 0.05 | 0.12 | 0.08 |
| 200 | 60 | 57.0 | phase | 0.05 | 0.05 | 0.08 |
| 240 | 72 | 13.0 | eval_restricted | 0.12 | 0.17 | 0.20 |
| 240 | 72 | 13.0 | eval_full | 0.07 | 0.08 | 0.07 |
| 240 | 72 | 13.0 | orbit_exact | 0.07 | 0.07 | 0.05 |
| 240 | 72 | 13.0 | window | 0.08 | 0.07 | 0.07 |
| 240 | 72 | 13.0 | block_boot | 0.08 | 0.07 | 0.07 |
| 240 | 72 | 13.0 | phase | 0.05 | 0.00 | 0.05 |
| 300 | 90 | 30.9 | eval_restricted | 0.10 | 0.07 | 0.13 |
| 300 | 90 | 30.9 | eval_full | 0.05 | 0.02 | 0.05 |
| 300 | 90 | 30.9 | orbit_exact | 0.05 | 0.02 | 0.05 |
| 300 | 90 | 30.9 | window | 0.07 | 0.10 | 0.05 |
| 300 | 90 | 30.9 | block_boot | 0.08 | 0.05 | 0.07 |
| 300 | 90 | 30.9 | phase | 0.05 | 0.02 | 0.05 |

Seam diagnostic (mean over subjects): statistic at shift 0 vs mean over the full orbit of shifts 1..L-1 of the evaluation block

| n | |r| obs | |r| orbit | HSIC obs | HSIC orbit | J obs | J orbit |
|---|---|---|---|---|---|---|
| 200 | 0.1198 | 0.1273 | 0.00625 | 0.00614 | -0.0126 | -0.0163 |
| 240 | 0.1330 | 0.1215 | 0.00543 | 0.00530 | -0.0146 | -0.0139 |
| 300 | 0.1231 | 0.1132 | 0.00431 | 0.00427 | -0.0136 | -0.0133 |
