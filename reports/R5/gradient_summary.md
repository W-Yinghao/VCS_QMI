# r5 R5-20A — hard-distractor reference mass and score-level gradients (exploratory, DEV)

Score-level derivatives of each model's own training loss at its CAL-selected state; not encoder gradients.  Image-macro over queries with >= 1 valid and >= 1 invalid candidate; seed mean (per-seed values in gradient_diagnostics.jsonl).  T-based fields of softmax models are not identified (b unconstrained).

## refcocog

| model | shared top-1 q share | shared top-1 |g| share | self top-1 q share | self top-1 |g| share | shared top-5 |g| share | invalid T ≥ 0.95 | invalid T + 1 |
|---|---|---|---|---|---|---|---|
| VCS | 0.057 | 0.213 | 0.057 | 0.481 | 0.591 | 0.001 | 0.137 |
| JS | 0.057 | 0.206 | 0.057 | 0.502 | 0.578 | 0.001 | 0.110 |
| set softmax | 0.057 | 0.247 | 0.057 | 0.604 | 0.636 | 0.012 | 0.188 |
| VCS(nu) | 0.066 | 0.213 | 0.117 | 0.523 | 0.593 | 0.000 | 0.168 |
| JS(nu) | 0.066 | 0.197 | 0.119 | 0.494 | 0.570 | 0.000 | 0.175 |
| distribution softmax | 0.057 | 0.251 | 0.057 | 0.523 | 0.641 | 0.002 | 0.169 |
| VCS(qnorm) | 0.057 | 0.234 | 0.057 | 0.546 | 0.619 | 0.001 | 0.137 |
| JS(qnorm) | 0.057 | 0.229 | 0.057 | 0.541 | 0.609 | 0.001 | 0.131 |

## refcocoplus

| model | shared top-1 q share | shared top-1 |g| share | self top-1 q share | self top-1 |g| share | shared top-5 |g| share | invalid T ≥ 0.95 | invalid T + 1 |
|---|---|---|---|---|---|---|---|
| VCS | 0.042 | 0.140 | 0.042 | 0.458 | 0.445 | 0.001 | 0.125 |
| JS | 0.042 | 0.139 | 0.042 | 0.453 | 0.441 | 0.001 | 0.123 |
| set softmax | 0.042 | 0.160 | 0.042 | 0.668 | 0.471 | 0.003 | 0.062 |
| VCS(nu) | 0.045 | 0.135 | 0.094 | 0.484 | 0.445 | 0.000 | 0.166 |
| JS(nu) | 0.045 | 0.131 | 0.095 | 0.457 | 0.434 | 0.000 | 0.175 |
| distribution softmax | 0.042 | 0.178 | 0.042 | 0.583 | 0.511 | 0.003 | 0.103 |
| VCS(qnorm) | 0.042 | 0.167 | 0.042 | 0.495 | 0.493 | 0.001 | 0.134 |
| JS(qnorm) | 0.042 | 0.163 | 0.042 | 0.497 | 0.486 | 0.001 | 0.126 |

