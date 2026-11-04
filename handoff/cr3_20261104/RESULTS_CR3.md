Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR3 results (numbers only)

protocol `CR3_FIXED_GRAPH_SUPPORT_COMPLETE_V1`; frozen sha256 `4b1f9694476a69321a230afc8f5d937aa5b0e38ad8be0c89c49c7b129e605c3c`; amendment CR3_A1 (per-condition GPU lock; groups: Flickr00=NVIDIA L40S; Flickr60=NVIDIA L40S; CC152K_RECOVERED_20261104=NVIDIA L40S), torch 2.9.0+cu128, CUDA 12.8; DEV only; seed 1001; step 8000; TEST not read.

| condition | relation | consumer | plan | i2t R1 | R5 | R10 | t2i R1 | R5 | R10 | rSum | Δbase | COMPLETE−RECT | −OBS (same consumer) | −J_MASS_OBS | −BASE/BCE/NCE TOP1 | i2t resc/lost | t2i resc/lost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | OBS | RECT | RERUN_ON_LOCK | 93.20 | 99.01 | 99.51 | 81.10 | 94.99 | 97.44 | 565.25 | +6.80 |  |  | +0.14 | // | 37/14 | 221/108 |
| Flickr00 | OBS | COMPLETE | RUN_NEW | 85.21 | 98.03 | 99.41 | 77.51 | 94.69 | 97.26 | 552.11 | -6.33 | -13.14 |  | -0.49 | // | 44/102 | 211/280 |
| Flickr00 | J_MASS_OBS | RECT | RUN_NEW | 93.20 | 99.01 | 99.41 | 81.10 | 95.01 | 97.38 | 565.11 | +6.67 |  | -0.14 |  | // | 37/14 | 219/106 |
| Flickr00 | J_MASS_OBS | COMPLETE | RUN_NEW | 85.40 | 98.22 | 99.41 | 77.69 | 94.64 | 97.24 | 552.60 | -5.84 | -12.50 | +0.49 |  | // | 44/100 | 216/276 |
| Flickr60 | OBS | RECT | RERUN_ON_LOCK | 91.22 | 98.72 | 99.31 | 79.78 | 94.54 | 96.96 | 560.53 | +2.09 |  |  | -0.28 | +0.49/+0.91/+1.58 | 30/27 | 215/169 |
| Flickr60 | OBS | COMPLETE | RUN_NEW | 89.35 | 98.52 | 99.51 | 80.16 | 94.60 | 96.88 | 559.01 | +0.57 | -1.52 |  | -0.10 | -1.34/-1.36/-1.74 | 30/46 | 229/164 |
| Flickr60 | J_MASS_OBS | RECT | RUN_NEW | 91.12 | 98.92 | 99.31 | 79.84 | 94.64 | 96.98 | 560.81 | +2.37 |  | +0.28 |  | +0.77/+1.18/+1.85 | 32/30 | 217/168 |
| Flickr60 | J_MASS_OBS | COMPLETE | RUN_NEW | 89.64 | 98.52 | 99.51 | 80.00 | 94.56 | 96.88 | 559.11 | +0.67 | -1.70 | +0.10 |  | -1.24/-1.26/-1.64 | 32/45 | 227/170 |
| Flickr60 | BASE_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 91.42 | 98.72 | 99.41 | 79.27 | 94.44 | 96.79 | 560.04 | +1.60 |  | -0.49 | -0.77 | /+0.41/+1.08 | 32/27 | 186/166 |
| Flickr60 | BASE_KEEP_TOP1 | COMPLETE | RUN_NEW | 90.43 | 98.82 | 99.21 | 80.45 | 94.58 | 96.86 | 560.36 | +1.91 | +0.32 | +1.34 | +1.24 | /-0.02/-0.39 | 31/36 | 204/124 |
| Flickr60 | J_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 91.03 | 98.82 | 99.21 | 79.37 | 94.34 | 96.77 | 559.53 | +1.08 |  | -1.01 | -1.28 | -0.51/-0.10/+0.57 | 32/31 | 207/182 |
| Flickr60 | J_KEEP_TOP1 | COMPLETE | RUN_NEW | 90.04 | 98.82 | 99.41 | 80.39 | 94.64 | 97.04 | 560.34 | +1.89 | +0.81 | +1.32 | +1.22 | -0.02/-0.04/-0.41 | 30/39 | 203/126 |
| Flickr60 | BCE_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 91.12 | 99.01 | 99.31 | 79.07 | 94.30 | 96.80 | 559.63 | +1.18 |  | -0.91 | -1.18 | -0.41//+0.67 | 32/30 | 202/192 |
| Flickr60 | BCE_KEEP_TOP1 | COMPLETE | RUN_NEW | 89.94 | 98.82 | 99.41 | 80.53 | 94.65 | 97.02 | 560.37 | +1.93 | +0.75 | +1.36 | +1.26 | +0.02//-0.37 | 29/39 | 205/121 |
| Flickr60 | NCE_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 90.73 | 98.72 | 99.21 | 79.15 | 94.36 | 96.79 | 558.95 | +0.51 |  | -1.58 | -1.85 | -1.08/-0.67/ | 31/33 | 203/189 |
| Flickr60 | NCE_KEEP_TOP1 | COMPLETE | RUN_NEW | 90.43 | 98.82 | 99.41 | 80.39 | 94.60 | 97.10 | 560.75 | +2.31 | +1.79 | +1.74 | +1.64 | +0.39/+0.37/ | 30/35 | 204/127 |
| Flickr60 | KEEP_ORACLE | RECT | RERUN_ON_LOCK | 91.03 | 98.82 | 99.31 | 79.45 | 94.50 | 96.77 | 559.86 | +1.42 |  | -0.67 | -0.95 | -0.18/+0.24/+0.91 | 32/31 | 205/176 |
| Flickr60 | KEEP_ORACLE | COMPLETE | RUN_NEW | 88.76 | 98.03 | 99.41 | 80.22 | 94.62 | 96.92 | 557.95 | -0.49 | -1.91 | -1.07 | -1.16 | -2.41/-2.43/-2.80 | 26/48 | 219/151 |
| Flickr60 | FULL_TRUE_ORACLE | RECT | RUN_NEW | 91.91 | 99.01 | 99.51 | 80.34 | 94.71 | 97.04 | 562.52 | +4.08 |  | +1.99 | +1.72 | +2.49/+2.90/+3.57 | 33/23 | 232/158 |
| Flickr60 | FULL_TRUE_ORACLE | COMPLETE | RUN_NEW | 87.97 | 97.73 | 99.11 | 80.55 | 94.65 | 97.06 | 557.08 | -1.36 | -5.44 | -1.93 | -2.03 | -3.27/-3.29/-3.67 | 25/55 | 203/118 |
| CC152K_RECOVERED_20261104 | OBS | RECT | RERUN_ON_LOCK | 78.93 | 93.36 | 95.96 | 76.62 | 91.92 | 95.53 | 532.32 | +13.71 |  |  | +0.00 | -0.58/-1.01/-0.58 | 33/14 | 40/19 |
| CC152K_RECOVERED_20261104 | OBS | COMPLETE | RUN_NEW | 78.93 | 93.36 | 95.96 | 76.62 | 91.92 | 95.53 | 532.32 | +13.71 | +0.00 |  | +0.00 | +5.05/+5.05/+4.91 | 33/14 | 40/19 |
| CC152K_RECOVERED_20261104 | J_MASS_OBS | RECT | RUN_NEW | 78.64 | 93.36 | 95.96 | 76.77 | 92.35 | 95.24 | 532.32 | +13.71 |  | +0.00 |  | -0.58/-1.01/-0.58 | 32/15 | 41/19 |
| CC152K_RECOVERED_20261104 | J_MASS_OBS | COMPLETE | RUN_NEW | 78.64 | 93.36 | 95.96 | 76.77 | 92.35 | 95.24 | 532.32 | +13.71 | +0.00 | +0.00 |  | +5.05/+5.05/+4.91 | 32/15 | 41/19 |
| CC152K_RECOVERED_20261104 | BASE_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 78.50 | 93.51 | 96.25 | 76.77 | 92.35 | 95.53 | 532.90 | +14.29 |  | +0.58 | +0.58 | /-0.43/+0.00 | 30/14 | 38/16 |
| CC152K_RECOVERED_20261104 | BASE_KEEP_TOP1 | COMPLETE | RUN_NEW | 77.49 | 92.93 | 95.96 | 74.60 | 91.63 | 94.66 | 527.27 | +8.66 | -5.63 | -5.05 | -5.05 | /+0.00/-0.14 | 29/20 | 38/31 |
| CC152K_RECOVERED_20261104 | J_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 78.93 | 93.65 | 95.96 | 76.77 | 91.92 | 95.38 | 532.61 | +14.00 |  | +0.29 | +0.29 | -0.29/-0.72/-0.29 | 31/12 | 37/15 |
| CC152K_RECOVERED_20261104 | J_KEEP_TOP1 | COMPLETE | RUN_NEW | 77.49 | 92.78 | 95.82 | 75.90 | 91.92 | 94.08 | 527.99 | +9.38 | -4.62 | -4.33 | -4.33 | +0.72/+0.72/+0.58 | 32/23 | 42/26 |
| CC152K_RECOVERED_20261104 | BCE_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 79.37 | 93.36 | 95.96 | 77.34 | 91.92 | 95.38 | 533.33 | +14.72 |  | +1.01 | +1.01 | +0.43//+0.43 | 34/12 | 41/15 |
| CC152K_RECOVERED_20261104 | BCE_KEEP_TOP1 | COMPLETE | RUN_NEW | 77.78 | 92.64 | 95.67 | 75.18 | 91.77 | 94.23 | 527.27 | +8.66 | -6.06 | -5.05 | -5.05 | +0.00//-0.14 | 32/21 | 41/30 |
| CC152K_RECOVERED_20261104 | NCE_KEEP_TOP1 | RECT | RERUN_ON_LOCK | 78.93 | 93.36 | 95.96 | 77.06 | 92.06 | 95.53 | 532.90 | +14.29 |  | +0.58 | +0.58 | +0.00/-0.43/ | 31/12 | 39/15 |
| CC152K_RECOVERED_20261104 | NCE_KEEP_TOP1 | COMPLETE | RUN_NEW | 77.49 | 92.78 | 95.96 | 75.32 | 91.63 | 94.23 | 527.42 | +8.80 | -5.48 | -4.91 | -4.91 | +0.14/+0.14/ | 30/21 | 39/27 |

## COMPLETE / RECT cost (per update, 64 query events)

| condition | relation | consumer | pairs/update | pairs/query event | cache reads/update | max pairs | peak GPU MB | train s | same draws |
|---|---|---|---|---|---|---|---|---|---|
| Flickr00 | OBS | RECT | 4090.3 | 63.91 | 127.9 | 4096 | 681.6 | 29.6 |  |
| Flickr00 | OBS | COMPLETE | 4346.0 | 67.91 | 383.6 | 4352 | 691.7 | 319.3 | True |
| Flickr00 | J_MASS_OBS | RECT | 4090.5 | 63.91 | 127.9 | 4096 | 691.0 | 29.8 |  |
| Flickr00 | J_MASS_OBS | COMPLETE | 4346.2 | 67.91 | 383.6 | 4352 | 694.8 | 318.8 | True |
| Flickr60 | OBS | RECT | 4090.4 | 63.91 | 127.9 | 4096 | 681.6 | 30.1 |  |
| Flickr60 | OBS | COMPLETE | 4346.1 | 67.91 | 383.6 | 4352 | 691.7 | 325.7 | True |
| Flickr60 | J_MASS_OBS | RECT | 4090.6 | 63.92 | 127.9 | 4096 | 691.0 | 30.7 |  |
| Flickr60 | J_MASS_OBS | COMPLETE | 4346.3 | 67.91 | 383.6 | 4352 | 694.8 | 323.8 | True |
| Flickr60 | BASE_KEEP_TOP1 | RECT | 4090.0 | 63.91 | 127.9 | 4096 | 704.5 | 30.7 |  |
| Flickr60 | BASE_KEEP_TOP1 | COMPLETE | 4661.9 | 72.84 | 699.3 | 4815 | 704.2 | 320.8 | True |
| Flickr60 | J_KEEP_TOP1 | RECT | 4089.9 | 63.90 | 127.9 | 4096 | 713.9 | 30.2 |  |
| Flickr60 | J_KEEP_TOP1 | COMPLETE | 4692.6 | 73.32 | 730.1 | 4888 | 707.3 | 327.7 | True |
| Flickr60 | BCE_KEEP_TOP1 | RECT | 4089.8 | 63.90 | 127.9 | 4096 | 693.0 | 30.3 |  |
| Flickr60 | BCE_KEEP_TOP1 | COMPLETE | 4700.7 | 73.45 | 738.2 | 4990 | 691.7 | 326.6 | True |
| Flickr60 | NCE_KEEP_TOP1 | RECT | 4090.2 | 63.91 | 127.9 | 4096 | 701.4 | 30.3 |  |
| Flickr60 | NCE_KEEP_TOP1 | COMPLETE | 4642.4 | 72.54 | 679.6 | 4738 | 702.0 | 327.4 | True |
| Flickr60 | KEEP_ORACLE | RECT | 4090.4 | 63.91 | 127.9 | 4096 | 713.8 | 29.7 |  |
| Flickr60 | KEEP_ORACLE | COMPLETE | 4547.4 | 71.05 | 584.7 | 4597 | 712.4 | 327.0 | True |
| Flickr60 | FULL_TRUE_ORACLE | RECT | 4090.5 | 63.91 | 127.9 | 4096 | 723.2 | 29.9 |  |
| Flickr60 | FULL_TRUE_ORACLE | COMPLETE | 4346.1 | 67.91 | 383.6 | 4352 | 721.8 | 326.5 | True |
| CC152K_RECOVERED_20261104 | OBS | RECT | 4093.2 | 63.96 | 128.0 | 4096 | 746.9 | 30.3 |  |
| CC152K_RECOVERED_20261104 | OBS | COMPLETE | 4093.2 | 63.96 | 128.0 | 4096 | 755.9 | 328.3 | True |
| CC152K_RECOVERED_20261104 | J_MASS_OBS | RECT | 4093.0 | 63.95 | 128.0 | 4096 | 759.4 | 29.9 |  |
| CC152K_RECOVERED_20261104 | J_MASS_OBS | COMPLETE | 4093.0 | 63.95 | 128.0 | 4096 | 759.0 | 326.3 | True |
| CC152K_RECOVERED_20261104 | BASE_KEEP_TOP1 | RECT | 4093.1 | 63.95 | 128.0 | 4096 | 765.7 | 30.2 |  |
| CC152K_RECOVERED_20261104 | BASE_KEEP_TOP1 | COMPLETE | 4157.9 | 64.97 | 192.7 | 4424 | 765.3 | 332.9 | True |
| CC152K_RECOVERED_20261104 | J_KEEP_TOP1 | RECT | 4092.9 | 63.95 | 128.0 | 4096 | 775.0 | 30.2 |  |
| CC152K_RECOVERED_20261104 | J_KEEP_TOP1 | COMPLETE | 4156.1 | 64.94 | 191.1 | 4428 | 768.4 | 332.7 | True |
| CC152K_RECOVERED_20261104 | BCE_KEEP_TOP1 | RECT | 4093.0 | 63.95 | 128.0 | 4096 | 756.3 | 30.3 |  |
| CC152K_RECOVERED_20261104 | BCE_KEEP_TOP1 | COMPLETE | 4155.9 | 64.94 | 190.8 | 4434 | 755.9 | 330.8 | True |
| CC152K_RECOVERED_20261104 | NCE_KEEP_TOP1 | RECT | 4092.9 | 63.95 | 128.0 | 4096 | 765.7 | 29.5 |  |
| CC152K_RECOVERED_20261104 | NCE_KEEP_TOP1 | COMPLETE | 4155.3 | 64.93 | 190.3 | 4430 | 765.3 | 335.4 | True |

Resources: GPU jobs 2.684 GPU·h; CPU jobs (excl. feeder) 0.541 h (RESOURCE_LEDGER.csv).

Files: TASK_ENDPOINTS.csv, TASK_CURVES.csv, TRANSFER_STATS.csv, REUSE_OR_RERUN_RECEIPTS.json, RESOURCE_LEDGER.csv, cr2_audit_export/, PROTOCOL_FROZEN_CR3.json.

## Records (appended once after code/site/summarize_cr3.py)
- Amendment `AMENDMENT_CR3_A1.json` (sha256 2154e673…, read-only; PROTOCOL_FROZEN_CR3.json unchanged): GPU lock per matched group (one Slurm job per condition, partition RTX6000PRO,L40S,H100, exclude node51/53/60). All three groups ran on NVIDIA L40S, driver 595.84, torch 2.9.0+cu128, CUDA 12.8 (node52) — every comparison in the table, including cross-condition, is same-GPU-type.
- Acceptance: RTX PRO 6000 1088291 ALL_PASS (A1–A7 + reuse receipts); L40S 1088522 ALL_PASS (A1–A7). Package tests 26/26 (1088293).
- Reuse: all 12 historical RECT cells failed the 250-step bitwise receipt (max |Δparam| 0.022–0.026) -> rerun; CR3 path is bitwise reproducible across jobs (probe 1088303). Pre-A1 RTX run of Flickr00 OBS RECT (1088304, 565.25) kept under students/ on record; the table uses the L40S rerun.
- Cancelled before running: 1088305–1088315, 1088334 (RTX-only, superseded by A1). Group jobs: 1088549 (Flickr60, 16 cells), 1088550 (CC, 12), 1088551 (Flickr00, 4); 32/32 complete, 0 failed.
- All COMPLETE cells used exactly the same edge draws as their RECT pair (edge-exposure hash equal: "same draws = True").
