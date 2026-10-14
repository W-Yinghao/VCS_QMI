# V8-01 — raw overall-vote audit (EvalMuse FIT / CAL / DEV; V7 data manifest) — 2026-10-14

Source `/home/infres/yinwang/CS_QMI/outputs/V7/data/manifest_images.jsonl` (sha256 `53584c8d021d100d…`); script `scripts/v8/v8_labels.py audit`.  Every image keeps its valid 1–5 votes; histograms sum to 1, tails are non-increasing and 1 + Σ tails equals the vote mean in every role (max error below).

| role | images | prompts | 3 / 6 votes | natural histogram (1..5) | natural tails P(H≥2..5) | max \|1+Σtail−mean\| |
|---|---|---|---|---|---|---|
| FIT | 26177 | 2385 | 20018 / 6159 | [0.11191, 0.32556, 0.27459, 0.16954, 0.1184] | [0.88809, 0.56252, 0.28794, 0.1184] | 8.9e-16 |
| CAL_SELECT | 1653 | 145 | 1277 / 376 | [0.10811, 0.35107, 0.28066, 0.15335, 0.10681] | [0.89189, 0.54082, 0.26016, 0.10681] | 8.9e-16 |
| CAL_PROB | 1630 | 154 | 1320 / 310 | [0.09567, 0.3334, 0.29259, 0.17438, 0.10397] | [0.90433, 0.57093, 0.27835, 0.10397] | 8.9e-16 |
| DEV | 3257 | 307 | 2456 / 801 | [0.10316, 0.30284, 0.27895, 0.18039, 0.13466] | [0.89684, 0.594, 0.31505, 0.13466] | 8.9e-16 |

All roles: {'3': 25071, '6': 7646} (images by vote count).  FIT priors π = [0.8881, 0.5625, 0.2879, 0.1184] for H ≥ 2, 3, 4, 5; no degenerate threshold (none).

Natural sampling: uniform prompt → uniform valid image → uniform valid vote (3- and 6-vote images weigh the same).  No image is excluded for element-level flags; every image in the four roles has at least one valid overall vote.
