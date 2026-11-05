Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR4 results (numbers only)

protocol `CR4_FIXED_GRAPH_POSITIVE_EXCLUSIVE_V1`; frozen sha256 `b9d66d1a74f767e023c31842f2829dea072e070fc20ef07db3e2fcbf10fbf7b9`; group GPUs: CC152K_RECOVERED_20261104=NVIDIA L40S (595.84), Flickr00=NVIDIA L40S (595.84), Flickr60=NVIDIA L40S (595.84); reference decisions: CC152K_RECOVERED_20261104=RERUN_CR3_REFERENCES_IN_THIS_GROUP, Flickr00=RERUN_CR3_REFERENCES_IN_THIS_GROUP, Flickr60=RERUN_CR3_REFERENCES_IN_THIS_GROUP; DEV only; seed 1001; step 8000; TEST not read.

| condition | relation | plan / status | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | PC−COMPLETE_CE | PC−RECT_CE | −OBS(PC) | −J_MASS_OBS(PC) | −BASE/BCE/NCE TOP1 (PC) | −RECT OBS | i2t resc/lost | t2i resc/lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | OBS | TRAIN_NEW / COMPLETE | 93.10 | 99.21 | 99.51 | 81.24 | 95.03 | 97.42 | 565.50 | +7.06 | +13.39 | +0.26 |  | +0.39 | // | +0.26 | 37/15 | 228/108 |
| Flickr00 | J_MASS_OBS | TRAIN_NEW / COMPLETE | 93.10 | 99.11 | 99.51 | 81.08 | 94.93 | 97.38 | 565.11 | +6.67 | +12.50 | +0.00 | -0.39 |  | // | -0.14 | 36/14 | 219/107 |
| Flickr60 | OBS | TRAIN_NEW / COMPLETE | 91.42 | 98.72 | 99.31 | 79.92 | 94.60 | 97.02 | 560.99 | +2.54 | +1.97 | +0.45 |  | +0.02 | +0.24/-0.14/+0.06 | +0.45 | 30/25 | 220/167 |
| Flickr60 | J_MASS_OBS | TRAIN_NEW / COMPLETE | 91.32 | 98.82 | 99.41 | 79.78 | 94.65 | 96.98 | 560.97 | +2.52 | +1.85 | +0.16 | -0.02 |  | +0.22/-0.16/+0.04 | +0.43 | 31/27 | 215/169 |
| Flickr60 | BASE_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 91.42 | 98.82 | 99.41 | 79.68 | 94.50 | 96.92 | 560.75 | +2.31 | +0.39 | +0.71 | -0.24 | -0.22 | /-0.37/-0.18 | +0.22 | 31/26 | 185/144 |
| Flickr60 | J_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 91.91 | 99.01 | 99.31 | 79.94 | 94.52 | 97.00 | 561.70 | +3.25 | +1.36 | +2.17 | +0.71 | +0.73 | +0.95/+0.57/+0.77 | +1.16 | 36/26 | 217/163 |
| Flickr60 | BCE_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 91.72 | 98.92 | 99.31 | 79.86 | 94.48 | 96.84 | 561.12 | +2.68 | +0.75 | +1.50 | +0.14 | +0.16 | +0.37//+0.20 | +0.59 | 31/23 | 208/158 |
| Flickr60 | NCE_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 91.81 | 98.72 | 99.31 | 79.59 | 94.64 | 96.86 | 560.93 | +2.49 | +0.18 | +1.97 | -0.06 | -0.04 | +0.18/-0.20/ | +0.39 | 35/26 | 215/179 |
| Flickr60 | KEEP_ORACLE | TRAIN_NEW / COMPLETE | 91.52 | 98.82 | 99.51 | 79.98 | 94.75 | 96.90 | 561.48 | +3.04 | +3.53 | +1.62 | +0.49 | +0.51 | +0.73/+0.36/+0.55 | +0.95 | 33/27 | 220/164 |
| Flickr60 | FULL_TRUE_ORACLE | TRAIN_NEW / COMPLETE | 91.81 | 99.01 | 99.51 | 80.22 | 94.69 | 97.08 | 562.33 | +3.89 | +5.25 | -0.20 | +1.34 | +1.36 | +1.58/+1.20/+1.40 | +1.79 | 34/25 | 226/158 |
| CC152K_RECOVERED_20261104 | OBS | IDENTITY_CHECK_THEN_REUSE_CR3_COMPLETE / COMPLETE | 78.93 | 93.36 | 95.96 | 76.62 | 91.92 | 95.53 | 532.32 | +13.71 | +0.00 | +0.00 |  | +0.00 | +0.00/-0.72/-0.14 | +0.00 | 33/14 | 40/19 |
| CC152K_RECOVERED_20261104 | J_MASS_OBS | IDENTITY_CHECK_THEN_REUSE_CR3_COMPLETE / COMPLETE | 78.64 | 93.36 | 95.96 | 76.77 | 92.35 | 95.24 | 532.32 | +13.71 | +0.00 | +0.00 | +0.00 |  | +0.00/-0.72/-0.14 | +0.00 | 32/15 | 41/19 |
| CC152K_RECOVERED_20261104 | BASE_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 78.35 | 93.51 | 96.39 | 76.48 | 91.92 | 95.67 | 532.32 | +13.71 | +5.05 | -0.58 | +0.00 | +0.00 | /-0.72/-0.14 | +0.00 | 29/14 | 38/18 |
| CC152K_RECOVERED_20261104 | J_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 78.50 | 93.80 | 95.96 | 76.77 | 91.92 | 95.53 | 532.47 | +13.85 | +4.47 | -0.14 | +0.14 | +0.14 | +0.14/-0.58/+0.00 | +0.14 | 29/13 | 38/16 |
| CC152K_RECOVERED_20261104 | BCE_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 79.22 | 93.36 | 96.10 | 77.06 | 92.06 | 95.24 | 533.04 | +14.43 | +5.77 | -0.29 | +0.72 | +0.72 | +0.72//+0.58 | +0.72 | 34/13 | 40/16 |
| CC152K_RECOVERED_20261104 | NCE_KEEP_TOP1 | TRAIN_NEW / COMPLETE | 78.64 | 93.51 | 96.10 | 76.91 | 91.92 | 95.38 | 532.47 | +13.85 | +5.05 | -0.43 | +0.14 | +0.14 | +0.14/-0.58/ | +0.14 | 31/14 | 40/17 |

Resources: GPU jobs 4.749 GPU·h (RESOURCE_LEDGER.csv). Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, CONSUMER_INCREMENT.csv, SOURCE_INCREMENT.csv, REFERENCE_RECEIPTS.json, SAME_DRAW_PAIR_TARGET_RECEIPT.csv, GRAD_COMPONENT_AUDIT_*.json, SOURCE_GRAPH_HASHES.json, GPU_ENV.json, PROTOCOL_FROZEN_CR4.json.
