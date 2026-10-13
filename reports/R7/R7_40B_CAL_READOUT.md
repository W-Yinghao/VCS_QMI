# r7 D1-20 — CAL-only readout (DEV sealed until D1-30)

Every number is from CAL (2,000 images; 95,868 question × mask pairs); the frozen selection rule (max CAL Spearman between the
selector's risk score r(K) and the realised teacher h) already chose each unit's checkpoint.  The native criterion is a pre-declared
secondary, never used for selection.

## 1. Learning-rate search (seed 0) and the frozen choice

| objective | lr | selected step | CAL Spearman(r, h) | native-selected step | its Spearman |
|---|---|---|---|---|---|
| vcs | 0.0001 | 5000 | 0.1364 | 4000 | 0.1355 |
| vcs | 0.0003 | 6000 | 0.1363 | 4000 | 0.1076 |
| logistic | 0.0001 | 6000 | 0.1387 | 4000 | 0.1288 |
| logistic | 0.0003 | 6000 | 0.1371 | 4000 | 0.1134 |
| direct | 0.0001 | 1000 | 0.1856 | 3000 | 0.1838 |
| direct | 0.0003 | 2000 | 0.2272 | 2000 | 0.2272 |

Frozen choice (lr_choice.json; the frozen planner's rule): vcs 0.0003 (CAL Spearman 0.0001: 0.1364, 0.0003: 0.1363); logistic 0.0001 (CAL Spearman 0.0001: 0.1387, 0.0003: 0.1371); direct 0.0003 (CAL Spearman 0.0001: 0.1856, 0.0003: 0.2272)

## 2. Main units at the chosen rates (seeds 0–2)

| unit | selected step | CAL Spearman(r, h) | native-selected Spearman | best eval Spearman | dynamic mean kept (0.25 / 0.5 / 0.75) | soft − hard gap at budgets 72 / 144 / 288 / 432 | h |
|---|---|---|---|---|---|---|---|
| vcs_lr0.0003_s0 | 6000 | 0.1363 | 0.1076 | 0.1363 | 0.25 / 0.5 / 0.75 | -0.035 / -0.028 / -0.016 / -0.004 | 2.99 |
| vcs_lr0.0003_s1 | 6000 | 0.1251 | 0.1010 | 0.1251 | 0.25 / 0.5 / 0.75 | -0.029 / -0.023 / -0.013 / -0.004 | 1.35 |
| vcs_lr0.0003_s2 | 6000 | 0.1186 | 0.1164 | 0.1186 | 0.25 / 0.5 / 0.75 | -0.038 / -0.029 / -0.018 / -0.006 | 1.34 |
| logistic_lr0.0001_s0 | 6000 | 0.1387 | 0.1288 | 0.1387 | 0.25 / 0.5 / 0.75 | -0.001 / -0.000 / +0.000 / -0.000 | 1.06 |
| logistic_lr0.0001_s1 | 6000 | 0.1454 | 0.1276 | 0.1454 | 0.25 / 0.5 / 0.75 | -0.007 / -0.004 / -0.000 / -0.000 | 1.32 |
| logistic_lr0.0001_s2 | 6000 | 0.1360 | 0.1031 | 0.1360 | 0.25 / 0.5 / 0.75 | -0.002 / -0.001 / -0.000 / -0.000 | 1.06 |
| direct_lr0.0003_s0 | 2000 | 0.2272 | 0.2272 | 0.2272 | 0.25 / 0.5 / 0.75 | -0.053 / -0.016 / -0.001 / -0.000 | 1.06 |
| direct_lr0.0003_s1 | 3000 | 0.2021 | 0.1952 | 0.2021 | 0.25 / 0.5 / 0.75 | -0.061 / -0.034 / -0.016 / -0.008 | 1.06 |
| direct_lr0.0003_s2 | 2000 | 0.2125 | 0.2125 | 0.2125 | 0.25 / 0.5 / 0.75 | +0.000 / +0.000 / +0.000 / -0.000 | 1.05 |

## 3. Ablations (seeds 0–2)

| unit | selected step | CAL Spearman(r, h) | native-selected Spearman | best eval Spearman | dynamic mean kept | soft − hard gap at budgets 72 / 144 / 288 / 432 | h |
|---|---|---|---|---|---|---|---|

## 4. Summary (CAL Spearman of the selected checkpoint, mean ± sd over seeds)

| arm | n | mean | sd | values |
|---|---|---|---|---|
| vcs | 3 | 0.1267 | 0.0089 | 0.1363, 0.1251, 0.1186 |
| logistic | 3 | 0.1400 | 0.0048 | 0.1387, 0.1454, 0.1360 |
| direct | 3 | 0.2140 | 0.0126 | 0.2272, 0.2021, 0.2125 |

## 5. Reading (coordinator, CAL only; DEV sealed — the R7-40c primary contrasts decide, not this table)
- Qwen2.5-VL selectors, frozen D1-20 recipe re-run on Qwen (rates by the frozen rule: vcs 3e-4, logistic 1e-4, direct 3e-4).
- **Same ordering as LLaVA (D1-20):** direct regression of h ranks the realised risk best on this criterion (0.214 ± 0.013), above logistic
  (0.140 ± 0.005) and VCS (0.127 ± 0.009).  The criterion is S3's own training target (frozen selection-asymmetry disclosure), and it measures
  which masks hurt, not answer quality.
- On Qwen, logistic is slightly above VCS (+0.013, about 1.5 seed sd); on LLaVA they were equal (+0.005).
- Every Qwen value is higher than its LLaVA counterpart (VCS 0.127 vs 0.088, logistic 0.140 vs 0.093, direct 0.214 vs 0.171).
- Execution: notes 40b_1–40b_3 (partition widening, L40S out-of-memory for the direct objective, stage-2 restart on RTX6000PRO / H100); no
  analysis change.  Unit records: reports/R7/R7_40B_units/.
