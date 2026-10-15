# V7C-04 — GenAI-Rank terminal scoring of the frozen V7 families: report — 2026-10-15

**Provenance:**
- Scoring request `V7C_04_SCORING_REQUEST.md` and ledger row, both committed before the read (`0e6159e`).
- Results-only commit `879986a` (`V7C_04_results/`), from job 1040759.
- Frozen scorer `scripts/v7/v7_score.py rank` (sha256 `e86af47e…`) on freeze v3 (methods.json `548c5293…`).
- Sealed label file sha256 `0eb955b4…` (equal to the seal).
- Endpoint: the human score (1–5 mean) of the selected candidate. Official N = 3 groups, N = 9 pools and first-index ties; prompt-cluster bootstrap 5,000 (seed 2718), prompts shared across generators.

## 1. Reading (fixed wording of amendment v2: a system comparison, not a loss attribution)
1. **Against the released FGA-BLIP2 (B0) — level.**
   - Effects: DALLE_3 −0.023 [−0.057, +0.011]; SDXL_Base +0.001 [−0.029, +0.030].
   - Both intervals contain 0.
2. **Against VQAScore CLIP-FlanT5-XXL (B1) — behind on both generators.**
   - Effects: −0.119 [−0.165, −0.072] and −0.085 [−0.128, −0.042]; Holm p 0.003, negative on every seed.
   - The modern-host VQAScore (B2, Qwen2.5-VL-7B; secondary row) is the strongest system on both generators (N9 4.314 / 3.605).
3. **Against TIFA and DSG — ahead on SDXL_Base, level on DALLE_3.**
   - TIFA: +0.091 [+0.044, +0.138] and DSG: +0.108 [+0.065, +0.153], both Holm p 0.003 on SDXL_Base.
   - On DALLE_3: +0.045 (TIFA) and +0.028 (DSG), not significant.
   - Both TIFA and DSG are the frozen local routes, with fidelity documented in V7C-03.
4. **Attribution (VCS vs logistic): no difference.**
   - M2-E − M1-E: +0.003 [−0.015, +0.022] and −0.001 [−0.017, +0.014], Holm p 1.0.
   - This agrees with Table A and with every matched comparison in the programme.
5. **Robustness.** The common-defined subset (without the 41 prompts where a primary or attribution method has a flagged failure) reproduces every reading; its significant cells are the same.
6. **Secondary readouts (descriptive).**
   - The native overall O of the same units selects slightly better than E (M2-O N9 4.161 / 3.495 vs M2-E 4.140 / 3.477).
   - The frozen H lies between the two (M2-H 4.157 / 3.498).
   - The N0 reproduction of the native FGA trainer is the best of the FGA-lineage rows (4.177 / 3.514).
   - CLIPScore selects below random on DALLE_3 (4.011 vs 4.031).
7. **Common features (X21, parsed prompts; descriptive Holm over its 12 contrasts).**
   - VCS-F vs BCE-F: not significant.
   - VCS-F vs the kernels: ahead of KDE (SDXL_Base +0.074, Holm p 0.005; DALLE_3 +0.061, Holm p 0.052, not significant after correction); level with Nyström and RuLSIF.
   - VCS-F vs the KL-bridge critics: ahead of DV on DALLE_3 (+0.036, Holm p 0.022).

**In the closeout package's terms (docs/05 §3).**
- The full V7 system with the element readout is competitive with the released FGA-BLIP2.
- It beats the question-answering pipelines (TIFA, DSG) on one generator.
- It does not reach VQAScore.
- The quadratic (VCS) objective adds nothing over logistic.
- Allowed: "the visual-evidence system has application value against FGA / TIFA / DSG, but not against VQAScore".
- Not allowed: any statement that the quadratic fit gives an exclusive gain.

## 2. Tables (generated from `V7C_04_results/*.json`)
### Primary family (amendment v2): M2-E vs the four external systems, 8 contrasts, Holm

| contrast | Δ (selected human score, N = 9) | 95 % interval | p Holm | per seed |
|---|---|---|---|---|
| M2_E - B0_FGA_released_O @ DALLE_3 | -0.023 | [-0.057, +0.011] | 0.570 | -0.003 / -0.022 / -0.045 |
| M2_E - B0_FGA_released_O @ SDXL_Base | +0.001 | [-0.029, +0.030] | 0.972 | -0.029 / +0.037 / -0.006 |
| M2_E - B1_VQAScore @ DALLE_3 | -0.119 | [-0.165, -0.072] | 0.003 | -0.099 / -0.117 / -0.141 |
| M2_E - B1_VQAScore @ SDXL_Base | -0.085 | [-0.128, -0.042] | 0.003 | -0.114 / -0.049 / -0.092 |
| M2_E - B4T_TIFA @ DALLE_3 | +0.045 | [-0.002, +0.092] | 0.254 | +0.065 / +0.047 / +0.023 |
| M2_E - B4T_TIFA @ SDXL_Base | +0.091 | [+0.044, +0.138] | 0.003 | +0.062 / +0.127 / +0.084 |
| M2_E - B4D_DSG @ DALLE_3 | +0.028 | [-0.020, +0.074] | 0.570 | +0.048 / +0.029 / +0.006 |
| M2_E - B4D_DSG @ SDXL_Base | +0.108 | [+0.065, +0.153] | 0.003 | +0.079 / +0.144 / +0.101 |

### Attribution family: M2-E vs M1-E (VCS vs logistic), 2 contrasts

| contrast | Δ (selected human score, N = 9) | 95 % interval | p Holm | per seed |
|---|---|---|---|---|
| M2_E - M1_E @ DALLE_3 | +0.003 | [-0.015, +0.022] | 1.000 | +0.042 / -0.003 / -0.030 |
| M2_E - M1_E @ SDXL_Base | -0.001 | [-0.017, +0.014] | 1.000 | -0.025 / +0.025 / -0.005 |

### v1-registered family (V7-03): M2-E vs M1-E, B0, B1, 6 contrasts

| contrast | Δ (selected human score, N = 9) | 95 % interval | p Holm | per seed |
|---|---|---|---|---|
| M2_E - M1_E @ DALLE_3 | +0.003 | [-0.015, +0.022] | 1.000 | +0.042 / -0.003 / -0.030 |
| M2_E - M1_E @ SDXL_Base | -0.001 | [-0.017, +0.014] | 1.000 | -0.025 / +0.025 / -0.005 |
| M2_E - B0_FGA_released_O @ DALLE_3 | -0.023 | [-0.057, +0.011] | 0.760 | -0.003 / -0.022 / -0.045 |
| M2_E - B0_FGA_released_O @ SDXL_Base | +0.001 | [-0.029, +0.030] | 1.000 | -0.029 / +0.037 / -0.006 |
| M2_E - B1_VQAScore @ DALLE_3 | -0.119 | [-0.165, -0.072] | 0.002 | -0.099 / -0.117 / -0.141 |
| M2_E - B1_VQAScore @ SDXL_Base | -0.085 | [-0.128, -0.042] | 0.002 | -0.114 / -0.049 / -0.092 |

### Sensitivity: primary family without the 41 flagged-failure prompts

| contrast | Δ (selected human score, N = 9) | 95 % interval | p Holm | per seed |
|---|---|---|---|---|
| M2_E - B0_FGA_released_O @ DALLE_3 | -0.022 | [-0.058, +0.014] | 0.561 | -0.002 / -0.020 / -0.045 |
| M2_E - B0_FGA_released_O @ SDXL_Base | -0.002 | [-0.032, +0.028] | 0.902 | -0.031 / +0.035 / -0.010 |
| M2_E - B1_VQAScore @ DALLE_3 | -0.116 | [-0.165, -0.067] | 0.003 | -0.095 / -0.113 / -0.139 |
| M2_E - B1_VQAScore @ SDXL_Base | -0.090 | [-0.133, -0.046] | 0.003 | -0.119 / -0.053 / -0.098 |
| M2_E - B4T_TIFA @ DALLE_3 | +0.045 | [-0.006, +0.095] | 0.315 | +0.065 / +0.047 / +0.022 |
| M2_E - B4T_TIFA @ SDXL_Base | +0.087 | [+0.038, +0.134] | 0.004 | +0.058 / +0.124 / +0.079 |
| M2_E - B4D_DSG @ DALLE_3 | +0.033 | [-0.017, +0.082] | 0.561 | +0.054 / +0.036 / +0.010 |
| M2_E - B4D_DSG @ SDXL_Base | +0.106 | [+0.060, +0.152] | 0.003 | +0.077 / +0.143 / +0.097 |

### Sensitivity: attribution without the 41 flagged-failure prompts

| contrast | Δ (selected human score, N = 9) | 95 % interval | p Holm | per seed |
|---|---|---|---|---|
| M2_E - M1_E @ DALLE_3 | +0.007 | [-0.011, +0.026] | 0.907 | +0.048 / -0.001 / -0.026 |
| M2_E - M1_E @ SDXL_Base | -0.001 | [-0.018, +0.015] | 0.907 | -0.022 / +0.025 / -0.007 |

### X21 common-feature rows (parsed prompts; 8 prompts excluded), VCS-F vs the others, Holm over 12 (descriptive family)

| contrast | Δ (selected human score, N = 9) | 95 % interval | p Holm | per seed |
|---|---|---|---|---|
| X21_N-VCS-F_E - X21_N-BCE-F_E @ DALLE_3 | +0.004 | [-0.009, +0.016] | 1.000 | +0.013 / +0.003 / -0.004 |
| X21_N-VCS-F_E - X21_N-BCE-F_E @ SDXL_Base | -0.006 | [-0.017, +0.005] | 1.000 | -0.020 / -0.011 / +0.013 |
| X21_N-VCS-F_E - X21_N-DV_E @ DALLE_3 | +0.036 | [+0.013, +0.058] | 0.022 | +0.040 / +0.027 / +0.040 |
| X21_N-VCS-F_E - X21_N-DV_E @ SDXL_Base | +0.011 | [-0.008, +0.030] | 1.000 | -0.013 / +0.017 / +0.029 |
| X21_N-VCS-F_E - X21_N-NWJ_E @ DALLE_3 | +0.030 | [+0.009, +0.052] | 0.065 | +0.028 / +0.031 / +0.032 |
| X21_N-VCS-F_E - X21_N-NWJ_E @ SDXL_Base | +0.012 | [-0.006, +0.029] | 1.000 | -0.003 / +0.006 / +0.033 |
| X21_N-VCS-F_E - X21_K-KDE-S_E @ DALLE_3 | +0.061 | [+0.019, +0.104] | 0.052 | +0.066 / +0.051 / +0.065 |
| X21_N-VCS-F_E - X21_K-KDE-S_E @ SDXL_Base | +0.074 | [+0.033, +0.115] | 0.005 | +0.069 / +0.073 / +0.082 |
| X21_N-VCS-F_E - X21_K-NYSTROM_E @ DALLE_3 | +0.006 | [-0.021, +0.032] | 1.000 | +0.011 / -0.004 / +0.010 |
| X21_N-VCS-F_E - X21_K-NYSTROM_E @ SDXL_Base | -0.008 | [-0.031, +0.015] | 1.000 | -0.013 / -0.009 / +0.000 |
| X21_N-VCS-F_E - X21_K-RULSIF_E @ DALLE_3 | +0.002 | [-0.023, +0.027] | 1.000 | +0.008 / -0.008 / +0.006 |
| X21_N-VCS-F_E - X21_K-RULSIF_E @ SDXL_Base | -0.005 | [-0.029, +0.020] | 1.000 | -0.011 / -0.007 / +0.003 |

### Per-method table (mean over seed files; N = 9 and N = 3 selected human score, N = 9 regret; N = 1 = random expectation)

| method | DALLE_3 N9 | N3 | regret N9 | SDXL_Base N9 | N3 | regret N9 |
|---|---|---|---|---|---|---|
| B2_VQAScore_Qwen25VL7B | 4.314 | 4.210 | 0.300 | 3.605 | 3.469 | 0.347 |
| B1_VQAScore | 4.259 | 4.154 | 0.355 | 3.562 | 3.433 | 0.391 |
| B5_HPSv3 | 4.191 | 4.124 | 0.423 | 3.486 | 3.413 | 0.467 |
| N0_O | 4.177 | 4.135 | 0.437 | 3.514 | 3.433 | 0.439 |
| M1_O | 4.164 | 4.113 | 0.449 | 3.504 | 3.434 | 0.449 |
| B0_FGA_released_O | 4.163 | 4.119 | 0.451 | 3.476 | 3.424 | 0.477 |
| M2_O | 4.161 | 4.116 | 0.453 | 3.495 | 3.429 | 0.458 |
| M2_H | 4.157 | 4.107 | 0.457 | 3.498 | 3.427 | 0.455 |
| M1_H | 4.146 | 4.109 | 0.468 | 3.486 | 3.426 | 0.467 |
| X21_N-VCS-F_E_neutralfill | 4.144 | 4.108 | 0.469 | 3.475 | 3.420 | 0.478 |
| X21_K-RULSIF_E_neutralfill | 4.143 | 4.104 | 0.471 | 3.480 | 3.418 | 0.473 |
| M2_Ecal | 4.142 | 4.102 | 0.472 | 3.476 | 3.417 | 0.477 |
| M0_O | 4.141 | 4.114 | 0.472 | 3.499 | 3.436 | 0.454 |
| X21_N-BCE-F_E_neutralfill | 4.140 | 4.108 | 0.473 | 3.481 | 3.419 | 0.472 |
| M2_E | 4.140 | 4.101 | 0.474 | 3.477 | 3.416 | 0.476 |
| X21_K-NYSTROM_E_neutralfill | 4.139 | 4.105 | 0.475 | 3.482 | 3.425 | 0.471 |
| M1_Ecal | 4.137 | 4.107 | 0.476 | 3.478 | 3.419 | 0.475 |
| M1_E | 4.137 | 4.106 | 0.477 | 3.478 | 3.419 | 0.475 |
| M0_H | 4.136 | 4.109 | 0.478 | 3.506 | 3.429 | 0.447 |
| M0_E | 4.115 | 4.097 | 0.498 | 3.478 | 3.417 | 0.475 |
| X21_N-NWJ_E_neutralfill | 4.114 | 4.097 | 0.499 | 3.463 | 3.409 | 0.490 |
| M0_Ecal | 4.113 | 4.101 | 0.501 | 3.475 | 3.421 | 0.478 |
| B4D_DSG | 4.112 | 4.062 | 0.502 | 3.369 | 3.340 | 0.584 |
| X21_N-DV_E_neutralfill | 4.109 | 4.092 | 0.505 | 3.464 | 3.402 | 0.489 |
| B3_PNVQA_own | 4.105 | 4.090 | 0.509 | 3.383 | 3.370 | 0.570 |
| B4T_TIFA | 4.095 | 4.061 | 0.519 | 3.386 | 3.352 | 0.567 |
| B3P_PNVQA_parser | 4.094 | 4.081 | 0.520 | 3.364 | 3.352 | 0.589 |
| X21_K-KDE-S_E_neutralfill | 4.084 | 4.068 | 0.530 | 3.401 | 3.389 | 0.552 |
| CLIPScore_sanity | 4.011 | 4.053 | 0.603 | 3.368 | 3.344 | 0.585 |
| random (N = 1) | 4.031 | | | 3.295 | | |
## 3. Limits
- One benchmark: 800 prompts × 2 generators × 9 candidates. The 1–5 means are a benchmark convention, not a ratio scale.
- Seeds are training repeats, not independent prompts.
- TIFA and DSG are local routes (DSG with a substituted LLM); paid-API routes were not run.
- The X21 rows use untrained common features and exclude 8 parser-failed prompts. Their neutral-fill full-denominator variants are in the per-method table.
- Nothing was changed after the read.
- V8's later use of GenAI-Rank is a pre-specified follow-up on the same benchmark (V8-03 frozen before this read).
