# R7-30 D1-10 — frozen CAL readout (2026-10-12)

**Provenance**
- Protocol: `reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml`, read with the coordinator notes `rank_criterion` and `yes_no_mass` (both added before this readout) and the manifest addendum.
- Manifest: sha256 `32c72126…`, unchanged.
- Jobs: 14 teacher bundles, b000–b013, on H100 NVL and L40S, all exit 0. Completeness and readout ran in job 1035060 (`scripts/r7_d1_complete.py`, then `scripts/r7_d1_analyze.py --roles CAL FIT`).
- Completeness: 56 / 56 shards; FIT 39,960 / 39,960 and CAL 7,989 / 7,989 records; 0 duplicates; 0 skipped images. DEV teacher records exist (7,986) and were not read.
- Raw outputs: `outputs/R7_D1/teacher/d1_10_results.{json,md}` and `d1_10_completeness.json`.

Scale: C ∈ {−1, +1}, T = 2p̃ − 1, h = (C − a)² − (C − b)², d = (b − a)², balanced class weights from FIT, image-clustered bootstrap with 2,000 reps.

## Primary: CAL (every item, as frozen)

## CAL: 7989 questions, 2000 images (balanced weights) — primary: every item

Teacher: balanced Brier full 0.4103 [0.3929797477766452, 0.42875655397303775], text-only 0.9857; accuracy full 0.856, text-only 0.609.

QC yes / no mass: full min 0.322, below 0.9 3 / 7989; masked min 0.331, below 0.9 31 / 95868.

| mechanism | budget | mean h [95 %] | mean d [95 %] | mean h − d [95 %] | P(h<0) | P(h>d) | Pearson / Spearman (h,d) | acc |
|---|---|---|---|---|---|---|---|---|
| random | 0.125 | 0.1616 [0.1452, 0.1773] | 0.1808 [0.1711, 0.1906] | -0.0192 [-0.0333, -0.0063] | 0.28 | 0.72 | 0.54 / 0.46 | 0.791 |
| random | 0.25 | 0.0901 [0.0781, 0.1025] | 0.1017 [0.0949, 0.1089] | -0.0116 [-0.0224, -0.0007] | 0.32 | 0.68 | 0.45 / 0.38 | 0.818 |
| random | 0.5 | 0.0436 [0.0350, 0.0522] | 0.0475 [0.0435, 0.0518] | -0.0039 [-0.0119, 0.0041] | 0.40 | 0.60 | 0.37 / 0.27 | 0.839 |
| random | 0.75 | 0.0158 [0.0105, 0.0212] | 0.0189 [0.0169, 0.0210] | -0.0031 [-0.0082, 0.0019] | 0.45 | 0.55 | 0.25 / 0.17 | 0.849 |
| grid | 0.125 | 0.1579 [0.1423, 0.1731] | 0.1672 [0.1580, 0.1763] | -0.0093 [-0.0220, 0.0032] | 0.28 | 0.72 | 0.54 / 0.47 | 0.791 |
| grid | 0.25 | 0.0956 [0.0832, 0.1080] | 0.0993 [0.0924, 0.1067] | -0.0037 [-0.0141, 0.0067] | 0.32 | 0.68 | 0.51 / 0.4 | 0.816 |
| grid | 0.5 | 0.0393 [0.0310, 0.0470] | 0.0430 [0.0393, 0.0469] | -0.0037 [-0.0112, 0.0034] | 0.41 | 0.59 | 0.38 / 0.25 | 0.839 |
| grid | 0.75 | 0.0210 [0.0151, 0.0264] | 0.0205 [0.0186, 0.0227] | 0.0004 [-0.0048, 0.0055] | 0.45 | 0.55 | 0.3 / 0.19 | 0.849 |
| qsim | 0.125 | 0.1410 [0.1253, 0.1565] | 0.1697 [0.1598, 0.1798] | -0.0287 [-0.0421, -0.0158] | 0.33 | 0.67 | 0.54 / 0.4 | 0.803 |
| qsim | 0.25 | 0.0854 [0.0726, 0.0981] | 0.1069 [0.0998, 0.1149] | -0.0215 [-0.0335, -0.0104] | 0.37 | 0.63 | 0.48 / 0.32 | 0.826 |
| qsim | 0.5 | 0.0440 [0.0355, 0.0529] | 0.0503 [0.0461, 0.0545] | -0.0064 [-0.0140, 0.0014] | 0.44 | 0.56 | 0.39 / 0.21 | 0.838 |
| qsim | 0.75 | 0.0141 [0.0088, 0.0195] | 0.0193 [0.0175, 0.0213] | -0.0052 [-0.0103, -0.0000] | 0.50 | 0.50 | 0.23 / 0.11 | 0.850 |

Per-item rank criterion (coordinator note 2026-10-12): observed Spearman(h, d) [image-clustered 95 %] against the calibrated-teacher ceiling band (C* ~ Bernoulli(p̃_full), 200 draws); the 0.5 line is superseded.

| mechanism | budget | observed [95 %] | ceiling band (median) | verdict | 0.5 line (superseded) |
|---|---|---|---|---|---|
| random | 0.125 | 0.455 [0.429, 0.479] | [0.461, 0.502] (0.484) | not_below_calibrated_ceiling | below 0.5 |
| random | 0.25 | 0.384 [0.358, 0.409] | [0.378, 0.423] (0.398) | not_below_calibrated_ceiling | below 0.5 |
| random | 0.5 | 0.268 [0.241, 0.295] | [0.244, 0.296] (0.271) | not_below_calibrated_ceiling | below 0.5 |
| random | 0.75 | 0.172 [0.144, 0.198] | [0.154, 0.201] (0.178) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.125 | 0.472 [0.448, 0.496] | [0.463, 0.500] (0.481) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.25 | 0.402 [0.376, 0.428] | [0.380, 0.422] (0.402) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.5 | 0.249 [0.219, 0.278] | [0.231, 0.274] (0.249) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.75 | 0.194 [0.166, 0.222] | [0.173, 0.217] (0.194) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.125 | 0.396 [0.372, 0.421] | [0.419, 0.461] (0.442) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.25 | 0.317 [0.289, 0.343] | [0.336, 0.373] (0.357) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.5 | 0.214 [0.187, 0.242] | [0.208, 0.256] (0.229) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.75 | 0.107 [0.077, 0.135] | [0.104, 0.151] (0.127) | not_below_calibrated_ceiling | below 0.5 |

## Pre-declared secondary: yes / no mass ≥ 0.9

## Secondary: excludes yes/no mass < 0.9 (coordinator_note_20261012_yes_no_mass, pre-declared before the CAL readout)

Excluded items per role (full-pass first-token yes / no mass < 0.9): CAL: 3 of 7989; FIT: 22 of 39960.

### CAL without low-mass items: 7986 questions, 2000 images

Teacher: balanced Brier full 0.4100 [0.39266469122514625, 0.42856650815329717], text-only 0.9856; accuracy full 0.856, text-only 0.609.

QC yes / no mass: full min 0.917, below 0.9 0 / 7986; masked min 0.914, below 0.9 0 / 95832.

| mechanism | budget | mean h [95 %] | mean d [95 %] | mean h − d [95 %] | P(h<0) | P(h>d) | Pearson / Spearman (h,d) | acc |
|---|---|---|---|---|---|---|---|---|
| random | 0.125 | 0.1616 [0.1452, 0.1772] | 0.1809 [0.1711, 0.1906] | -0.0193 [-0.0334, -0.0064] | 0.28 | 0.72 | 0.54 / 0.46 | 0.791 |
| random | 0.25 | 0.0901 [0.0780, 0.1024] | 0.1018 [0.0949, 0.1090] | -0.0117 [-0.0225, -0.0007] | 0.32 | 0.68 | 0.45 / 0.38 | 0.818 |
| random | 0.5 | 0.0436 [0.0350, 0.0522] | 0.0475 [0.0436, 0.0518] | -0.0039 [-0.0119, 0.0040] | 0.40 | 0.60 | 0.37 / 0.27 | 0.839 |
| random | 0.75 | 0.0157 [0.0105, 0.0211] | 0.0189 [0.0169, 0.0210] | -0.0032 [-0.0082, 0.0018] | 0.45 | 0.55 | 0.25 / 0.17 | 0.849 |
| grid | 0.125 | 0.1578 [0.1423, 0.1730] | 0.1672 [0.1580, 0.1763] | -0.0094 [-0.0221, 0.0031] | 0.28 | 0.72 | 0.54 / 0.47 | 0.791 |
| grid | 0.25 | 0.0956 [0.0831, 0.1080] | 0.0994 [0.0924, 0.1067] | -0.0038 [-0.0142, 0.0066] | 0.32 | 0.68 | 0.51 / 0.4 | 0.816 |
| grid | 0.5 | 0.0392 [0.0310, 0.0469] | 0.0430 [0.0393, 0.0469] | -0.0038 [-0.0112, 0.0034] | 0.41 | 0.59 | 0.38 / 0.25 | 0.839 |
| grid | 0.75 | 0.0209 [0.0150, 0.0264] | 0.0206 [0.0186, 0.0227] | 0.0004 [-0.0048, 0.0055] | 0.45 | 0.55 | 0.3 / 0.19 | 0.849 |
| qsim | 0.125 | 0.1410 [0.1253, 0.1565] | 0.1698 [0.1599, 0.1798] | -0.0288 [-0.0422, -0.0159] | 0.33 | 0.67 | 0.54 / 0.4 | 0.803 |
| qsim | 0.25 | 0.0854 [0.0726, 0.0981] | 0.1070 [0.0999, 0.1149] | -0.0216 [-0.0336, -0.0103] | 0.37 | 0.63 | 0.48 / 0.32 | 0.826 |
| qsim | 0.5 | 0.0440 [0.0355, 0.0529] | 0.0504 [0.0461, 0.0546] | -0.0064 [-0.0140, 0.0014] | 0.44 | 0.56 | 0.39 / 0.21 | 0.838 |
| qsim | 0.75 | 0.0142 [0.0088, 0.0195] | 0.0193 [0.0175, 0.0213] | -0.0052 [-0.0103, -0.0000] | 0.50 | 0.50 | 0.23 / 0.11 | 0.850 |

Per-item rank criterion (coordinator note 2026-10-12): observed Spearman(h, d) [image-clustered 95 %] against the calibrated-teacher ceiling band (C* ~ Bernoulli(p̃_full), 200 draws); the 0.5 line is superseded.

| mechanism | budget | observed [95 %] | ceiling band (median) | verdict | 0.5 line (superseded) |
|---|---|---|---|---|---|
| random | 0.125 | 0.455 [0.431, 0.479] | [0.461, 0.501] (0.484) | not_below_calibrated_ceiling | below 0.5 |
| random | 0.25 | 0.384 [0.359, 0.408] | [0.379, 0.422] (0.399) | not_below_calibrated_ceiling | below 0.5 |
| random | 0.5 | 0.268 [0.241, 0.294] | [0.250, 0.298] (0.271) | not_below_calibrated_ceiling | below 0.5 |
| random | 0.75 | 0.172 [0.144, 0.200] | [0.160, 0.200] (0.179) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.125 | 0.472 [0.448, 0.495] | [0.460, 0.502] (0.482) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.25 | 0.401 [0.376, 0.429] | [0.381, 0.426] (0.403) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.5 | 0.249 [0.222, 0.277] | [0.230, 0.272] (0.253) | not_below_calibrated_ceiling | below 0.5 |
| grid | 0.75 | 0.194 [0.166, 0.222] | [0.168, 0.216] (0.192) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.125 | 0.396 [0.370, 0.422] | [0.416, 0.461] (0.441) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.25 | 0.317 [0.290, 0.343] | [0.337, 0.379] (0.357) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.5 | 0.214 [0.188, 0.243] | [0.206, 0.253] (0.232) | not_below_calibrated_ceiling | below 0.5 |
| qsim | 0.75 | 0.108 [0.079, 0.134] | [0.104, 0.146] (0.126) | not_below_calibrated_ceiling | below 0.5 |

## Descriptive strata (frozen readout field): GQA structural type at 12.5 %

| cell | strata |
|---|---|
| random:72 | compare n=193: h 0.113 / d 0.048 | logical n=2215: h 0.067 / d 0.119 | verify n=5581: h 0.201 / d 0.210 |
| grid:72 | compare n=193: h 0.097 / d 0.037 | logical n=2215: h 0.067 / d 0.102 | verify n=5581: h 0.196 / d 0.198 |
| qsim:72 | compare n=193: h 0.054 / d 0.027 | logical n=2215: h 0.074 / d 0.128 | verify n=5581: h 0.171 / d 0.191 |

## FIT (descriptive, not the readout): cells where the rank rule fires

| cell | observed [95 %] | ceiling band | verdict |
|---|---|---|---|
| qsim:72 | 0.385 [0.372, 0.397] | [0.423, 0.440] | d_does_not_rank_per_question_losses |
| qsim:144 | 0.308 [0.295, 0.320] | [0.338, 0.358] | d_does_not_rank_per_question_losses |

## Verdict lines (frozen interpretation grid, rank line superseded by the coordinator note)

1. **Grid line 1 — d tracks h at population level (|h − d| ≤ 0.1 · h with the CI covering 0, at every budget).**
   - **Holds for grid masks** at all four budgets. The residuals are −0.009, −0.004, −0.004 and +0.0004, and every CI covers 0.
   - **Does not hold for random masks.** At 12.5 % and 25 % the residuals are −0.019 [−0.033, −0.006] and −0.012 [−0.022, −0.001], about 12 % of h.
   - **Does not hold for qsim masks.** At 12.5 % and 25 % the residuals are −0.029 [−0.042, −0.016] and −0.022 [−0.034, −0.010], 20–25 % of h; at 75 % the CI's upper end is 0.000.
2. **Grid line 2 — consistent-sign residual: teacher miscalibration along the compression direction.**
   - **Applies to random and qsim masks.** The sign is negative at every budget, and the CI excludes 0 at the small budgets.
   - **Meaning:** d over-states the realised risk change. By the D1-L1 reference, a negative residual means an over-confident full teacher along (b − a).
   - **Consequence (frozen wording):** D1-20 is judged on realised h, and any d-based read-out is calibrated on CAL.
3. **Rank criterion — calibrated-teacher ceiling** (coordinator note; it supersedes the 0.5 line).
   - **On CAL the rule fires in no cell (0 / 12).** For random and grid masks, the observed per-item Spearman(h, d) lies inside or at the calibrated-teacher band.
   - **qsim on CAL is borderline but not fired.** At 12.5 % the observed value is 0.396 [0.372, 0.421] against a band of [0.419, 0.461]; at 25 % it is 0.317 [0.289, 0.343] against [0.336, 0.373].
   - **On FIT (descriptive) the rule fires for qsim at 12.5 % and 25 %.** With the larger n, the interval is tighter.
   - **Reading, under-claimed:** for random and grid masks, d ranks per-question losses about as well as a calibrated teacher could. For question-similarity masks there is a sign, borderline on CAL and clear on FIT, that d ranks losses worse than that ceiling.
   - **Superseded 0.5 line:** every cell is below 0.5, as D1-L1 predicted for any teacher.
4. **Text-only line.** Balanced text-only accuracy is 0.609 on CAL (0.616 on FIT), so question priors are present. The visual-evidence share, Brier_text − Brier_full, is 0.986 − 0.410 = 0.576 on the ±1 scale.
5. **Package §F.** The residual is not "large and unpredictable": its magnitude is at most about 25 % of h and its sign is stable. So the squared-gap read-out is not invalid, but it is biased upwards at small budgets.

## Surprises, disclosed before any D1-20 decision

- **qsim keeps the most task information at small budgets.** At 12.5 %, mean h is 0.141 for qsim against 0.162 for random and 0.158 for grid. qsim also has the largest negative residual and the weakest rank agreement.
- **Strata pull in opposite directions** (CAL, random 12.5 %, descriptive).
  - Compare questions (n = 193): h 0.113 vs d 0.048, so d under-states.
  - Logical questions (n = 2,215): h 0.067 vs d 0.119, so d over-states.
  - Verify questions (n = 5,581): h 0.201 vs d 0.210, close.
- **Masked accuracy at 12.5 %** is 0.79–0.80, against 0.856 for the full teacher. At 75 % it is 0.849–0.850.
- **The sensitivity row changes nothing.** The yes / no-mass exclusion removes 3 CAL items and 22 FIT items, and the tables move only in the 4th decimal.

> **Clarification (2026-10-12, owner-adopted review).**  The "calibrated-teacher ceiling" band is a reference generated under the full
> teacher's own probabilities (labels redrawn as C* ~ Bernoulli(p̃_full)); it is what a calibrated teacher would produce on these items, not a
> theoretical upper bound for all methods on natural data.  D1-10 shows only that the host's squared output gap d relates to the realised risk
> change h, with systematic bias in some conditions; it is not yet evidence about the trained selectors (D1-20 / D1-30).
