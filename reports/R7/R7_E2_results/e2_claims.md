# r7 E2 claims (Qwen2.5-VL-7B, 10k FIT vs FIT_FULL) — 2026-10-15T00:10:08Z

E2-8 (DF-8 analogue): a Qwen D1 claim is made only where the two Qwen scales (10k FIT: R7-40c; FIT_FULL: E2) agree in sign and both pass their own Holm-corrected primary tests (alpha 0.05); the full-scale numbers are the headline; any disagreement is reported as scale-dependent; the cross-host reading (LLaVA D1-FULL vs Qwen E2) stays descriptive: no pooling, no cross-host test

| bench | contrast | 10k Δ [95% CI] p_holm | full Δ [95% CI] p_holm | verdict |
|---|---|---|---|---|
| gqa_dev | S1-S2@f0.25 | -0.61 [-1.11, -0.13] 0.026 | -0.38 [-0.74, -0.01] 0.094 | not claimed: significant at the 10k scale only (scale-dependent) |
| gqa_dev | S1-S2@f0.5 | -0.28 [-0.70, +0.11] 0.35 | -0.54 [-0.87, -0.18] 0.012 | not claimed: significant at the full scale only (scale-dependent) |
| gqa_dev | S1-S2@f0.75 | -0.27 [-0.62, +0.07] 0.42 | -0.65 [-0.90, -0.40] 0.009 | not claimed: significant at the full scale only (scale-dependent) |
| gqa_dev | S1-S3@f0.25 | -1.24 [-1.74, -0.76] 0.009 | -3.89 [-4.51, -3.28] 0.009 | claim: S1 below |
| gqa_dev | S1-S3@f0.5 | -1.47 [-1.89, -1.05] 0.009 | -2.84 [-3.35, -2.31] 0.009 | claim: S1 below |
| gqa_dev | S1-S3@f0.75 | -1.16 [-1.51, -0.81] 0.009 | -1.47 [-1.85, -1.10] 0.009 | claim: S1 below |
| gqa_dev | S1-grid@f0.25 | -4.47 [-5.10, -3.81] 0.009 | -4.72 [-5.42, -4.07] 0.009 | claim: S1 below |
| gqa_dev | S1-grid@f0.5 | -3.43 [-4.04, -2.83] 0.009 | -3.07 [-3.67, -2.51] 0.009 | claim: S1 below |
| gqa_dev | S1-grid@f0.75 | -2.00 [-2.45, -1.55] 0.009 | -1.68 [-2.10, -1.26] 0.009 | claim: S1 below |
| textvqa_val | S1-S2@f0.25 | +14.81 [+13.84, +15.74] 0.009 | -2.40 [-2.95, -1.81] 0.009 | scale-dependent: significant with opposite signs |
| textvqa_val | S1-S2@f0.5 | +3.98 [+3.32, +4.61] 0.009 | -0.44 [-0.81, -0.10] 0.03 | scale-dependent: significant with opposite signs |
| textvqa_val | S1-S2@f0.75 | +0.02 [-0.37, +0.41] 0.886 | -0.19 [-0.47, +0.09] 0.193 | not claimed: neither scale significant |
| textvqa_val | S1-S3@f0.25 | +8.55 [+7.68, +9.45] 0.009 | +29.80 [+28.60, +30.96] 0.009 | claim: S1 above |
| textvqa_val | S1-S3@f0.5 | +2.87 [+2.24, +3.49] 0.009 | +14.82 [+13.93, +15.75] 0.009 | claim: S1 above |
| textvqa_val | S1-S3@f0.75 | +0.37 [-0.04, +0.81] 0.312 | +3.07 [+2.52, +3.60] 0.009 | not claimed: significant at the full scale only (scale-dependent) |
| textvqa_val | S1-grid@f0.25 | -7.17 [-8.25, -6.15] 0.009 | -1.83 [-2.88, -0.75] 0.009 | claim: S1 below |
| textvqa_val | S1-grid@f0.5 | -2.42 [-3.20, -1.66] 0.009 | +0.26 [-0.53, +1.04] 0.511 | not claimed: significant at the 10k scale only (scale-dependent) |
| textvqa_val | S1-grid@f0.75 | -1.10 [-1.63, -0.60] 0.009 | -0.49 [-1.04, +0.04] 0.158 | not claimed: significant at the 10k scale only (scale-dependent) |
| chartqa_val | S1-S2@f0.25 | -12.60 [-14.19, -11.05] 0.009 | -3.58 [-4.82, -2.29] 0.009 | claim: S1 below |
| chartqa_val | S1-S2@f0.5 | -8.37 [-9.67, -7.01] 0.009 | -3.14 [-4.20, -2.11] 0.009 | claim: S1 below |
| chartqa_val | S1-S2@f0.75 | -2.22 [-3.22, -1.25] 0.009 | -0.89 [-1.62, -0.19] 0.048 | claim: S1 below |
| chartqa_val | S1-S3@f0.25 | +0.66 [-0.80, +2.20] 0.4 | +1.79 [-0.04, +3.67] 0.094 | not claimed: neither scale significant |
| chartqa_val | S1-S3@f0.5 | -0.99 [-2.46, +0.48] 0.35 | +3.16 [+1.31, +4.94] 0.009 | not claimed: significant at the full scale only (scale-dependent) |
| chartqa_val | S1-S3@f0.75 | -0.42 [-1.45, +0.56] 0.786 | +1.51 [+0.38, +2.71] 0.048 | not claimed: significant at the full scale only (scale-dependent) |
| chartqa_val | S1-grid@f0.25 | -17.26 [-19.35, -15.16] 0.009 | -15.47 [-17.63, -13.26] 0.009 | claim: S1 below |
| chartqa_val | S1-grid@f0.5 | -13.80 [-15.46, -12.12] 0.009 | -11.72 [-13.42, -10.00] 0.009 | claim: S1 below |
| chartqa_val | S1-grid@f0.75 | -4.62 [-5.85, -3.44] 0.009 | -3.09 [-4.30, -1.88] 0.009 | claim: S1 below |

Scale contrast full − 10k (descriptive):

| bench | family@fraction | 10k | full | Δ [95% CI] |
|---|---|---|---|---|
| gqa_dev | S1@f0.125 | 76.78 | 76.80 | +0.03 [-0.46, +0.48] |
| gqa_dev | S1@f0.25 | 79.89 | 79.64 | -0.25 [-0.71, +0.18] |
| gqa_dev | S1@f0.5 | 82.80 | 83.15 | +0.35 [-0.01, +0.75] |
| gqa_dev | S1@f0.75 | 85.35 | 85.67 | +0.32 [-0.03, +0.67] |
| gqa_dev | S2@f0.125 | 77.81 | 77.03 | -0.78 [-1.29, -0.27] |
| gqa_dev | S2@f0.25 | 80.50 | 80.02 | -0.49 [-0.96, -0.03] |
| gqa_dev | S2@f0.5 | 83.08 | 83.69 | +0.61 [+0.20, +0.99] |
| gqa_dev | S2@f0.75 | 85.62 | 86.32 | +0.70 [+0.36, +1.04] |
| gqa_dev | S3@f0.125 | 78.17 | 79.95 | +1.78 [+1.24, +2.29] |
| gqa_dev | S3@f0.25 | 81.13 | 83.53 | +2.41 [+1.93, +2.92] |
| gqa_dev | S3@f0.5 | 84.27 | 85.99 | +1.72 [+1.32, +2.15] |
| gqa_dev | S3@f0.75 | 86.52 | 87.14 | +0.62 [+0.34, +0.89] |
| textvqa_val | S1@f0.125 | 51.36 | 54.26 | +2.90 [+2.03, +3.82] |
| textvqa_val | S1@f0.25 | 66.01 | 71.34 | +5.34 [+4.62, +6.09] |
| textvqa_val | S1@f0.5 | 78.44 | 81.12 | +2.68 [+2.18, +3.19] |
| textvqa_val | S1@f0.75 | 82.99 | 83.59 | +0.60 [+0.24, +0.99] |
| textvqa_val | S2@f0.125 | 32.11 | 60.27 | +28.16 [+27.07, +29.18] |
| textvqa_val | S2@f0.25 | 51.19 | 73.74 | +22.55 [+21.60, +23.53] |
| textvqa_val | S2@f0.5 | 74.46 | 81.56 | +7.10 [+6.51, +7.71] |
| textvqa_val | S2@f0.75 | 82.96 | 83.78 | +0.81 [+0.44, +1.21] |
| textvqa_val | S3@f0.125 | 39.64 | 26.59 | -13.05 [-13.95, -12.10] |
| textvqa_val | S3@f0.25 | 57.46 | 41.54 | -15.91 [-16.85, -14.87] |
| textvqa_val | S3@f0.5 | 75.57 | 66.29 | -9.27 [-10.05, -8.41] |
| textvqa_val | S3@f0.75 | 82.62 | 80.52 | -2.10 [-2.58, -1.60] |
| chartqa_val | S1@f0.125 | 26.41 | 27.74 | +1.34 [+0.07, +2.67] |
| chartqa_val | S1@f0.25 | 39.51 | 41.30 | +1.79 [+0.35, +3.34] |
| chartqa_val | S1@f0.5 | 60.57 | 62.66 | +2.08 [+0.70, +3.37] |
| chartqa_val | S1@f0.75 | 75.12 | 76.65 | +1.53 [+0.54, +2.53] |
| chartqa_val | S2@f0.125 | 35.24 | 30.50 | -4.74 [-6.17, -3.35] |
| chartqa_val | S2@f0.25 | 52.12 | 44.88 | -7.24 [-8.69, -5.82] |
| chartqa_val | S2@f0.5 | 68.94 | 65.80 | -3.14 [-4.39, -1.86] |
| chartqa_val | S2@f0.75 | 77.34 | 77.53 | +0.19 [-0.64, +1.08] |
| chartqa_val | S3@f0.125 | 27.59 | 27.19 | -0.40 [-1.65, +0.83] |
| chartqa_val | S3@f0.25 | 38.85 | 39.51 | +0.66 [-0.81, +2.03] |
| chartqa_val | S3@f0.5 | 61.56 | 59.50 | -2.07 [-3.46, -0.79] |
| chartqa_val | S3@f0.75 | 75.54 | 75.14 | -0.40 [-1.37, +0.50] |
