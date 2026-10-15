# REV9-FULL — the same paired phase-B interventions on the frozen D1-FULL selectors: report — 2026-10-15

**Provenance.**
- Protocol: REV9 addendum `addendum_full_20261014`, execution notes 3–5.
- Phase B: job 1039944. Guards exact for VCS and logistic.
- The direct objective's guard failed (cross-hardware). Its primary readout is stopped (note 3), and its H100 secondary rerun is pending (jobs 1040704 / 1040705).
- Evaluations: jobs 1040486 and 1040487.
- Scoring: job 1041006, frozen `rev9_score.py`, SparseVLM environment, tags VCS and logistic only.
- Results-only commit `be9a829`.
- Tables: `scripts/rev9_report_tables.py` (job 1041038). The uniform-grid columns are descriptive context from the frozen D1-30 rows on the same item sets.

## 1. Result
In D1-FULL both objectives were CAL-chosen at the **same** phase-A estimator learning rate, 1e-4. **There the two objectives respond alike.**

| objective (lr) | I1: critic in eval mode | I2: I1 + straight-through hard top-k |
|---|---|---|
| **VCS** (1e-4) | small gains: 3 of 12 Holm-positive (+0.51 to +0.63), none negative | **hurts: 7 of 12 negative**, −1.96 to −10.52 (TextVQA, budget 144); none positive |
| **logistic** (1e-4) | gains: 5 of 12 positive (+0.54 to +1.86, TextVQA), none negative | **hurts: 8 of 12 negative**, −1.16 to −7.48; none positive |

**Mechanism.** The gate's soft − hard gap is small for both objectives at lr 1e-4:
- VCS-FULL −0.0024 at budget 72, logistic-FULL −0.0013;
- for comparison, 10k VCS at lr 3e-4 had −0.057.

So there is no relaxation gap for the hard top-k to repair, and forcing it costs accuracy, as it did for 10k logistic.

**Reading against REV9 (10k), descriptive.**
- At 10k, VCS (lr 3e-4) gained from I2 and logistic (lr 1e-4) lost.
- At FULL, VCS at lr 1e-4 behaves like logistic at lr 1e-4.
- This is **consistent with the 10k pattern being driven by the learning rate (through the relaxation gap), not by the objective.** It is not a test of that, because the data scale also differs (FULL vs 10k).
- **REV10 is the test.** It is frozen in `REV10_LR_FACTORIAL_PROTOCOL_FROZEN_20261015.yaml` and has a pre-fixed decision rule: VCS at 1e-4 and logistic at 3e-4 at the 10k scale.

**Against the grid (descriptive).**
- The frozen D1-FULL selectors are below the uniform grid at small budgets: TextVQA −6.81 / −8.61 (VCS) and −7.79 / −10.62 (logistic) at 72 / 144; GQA about −3.
- The eval-mode critic closes only a little of this, at most +1.86. The training-mechanics fixes that closed most of the gap at 10k do not close it at FULL.

## 2. Tables (generated)
### Guard (I0, seed 0 of each objective)

| objective | estimator max abs diff | gate max abs diff | passed |
|---|---|---|---|
| vcs_lr0.0001 | 0.0 | 0.0 | True |
| logistic_lr0.0001 | 0.0 | 0.0 | True |

Prediction-file integrity: 216 files, 0 dropped / blank / duplicate lines.

### vcs_lr0.0001: accuracy (%), mean over seeds; differences are seed means of per-item paired differences (image-grouped bootstrap, Holm over 24)

| bench | budget | frozen | I1 − frozen | p Holm | I2 − frozen | p Holm | I2 − I1 | uniform grid | frozen − grid | I1 − grid | I2 − grid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gqa_dev | 72 | 75.92 | +0.32 [-0.02, +0.67] | 0.610 | -4.94 [-5.67, -4.22] | 0.024 | -5.26 [-5.98, -4.52] | 79.18 | -3.26 | -2.94 | -8.20 |
| gqa_dev | 144 | 78.36 | +0.51 [+0.20, +0.81] | 0.024 | -5.30 [-6.04, -4.58] | 0.024 | -5.81 [-6.57, -5.04] | 81.12 | -2.76 | -2.25 | -8.06 |
| gqa_dev | 288 | 81.83 | +0.15 [-0.11, +0.40] | 0.960 | -0.58 [-1.09, -0.08] | 0.324 | -0.72 [-1.25, -0.20] | 82.56 | -0.72 | -0.58 | -1.30 |
| gqa_dev | 432 | 83.83 | +0.10 [-0.05, +0.24] | 0.955 | -0.30 [-0.65, +0.01] | 0.610 | -0.40 [-0.71, -0.09] | 83.50 | +0.33 | +0.43 | +0.03 |
| textvqa_val | 72 | 18.41 | -0.35 [-0.72, +0.01] | 0.610 | -7.65 [-8.34, -6.98] | 0.024 | -7.29 [-7.98, -6.60] | 25.22 | -6.81 | -7.16 | -14.46 |
| textvqa_val | 144 | 23.94 | -0.19 [-0.58, +0.19] | 0.960 | -10.52 [-11.37, -9.68] | 0.024 | -10.34 [-11.17, -9.47] | 32.55 | -8.61 | -8.80 | -19.13 |
| textvqa_val | 288 | 34.06 | +0.60 [+0.26, +0.97] | 0.045 | -3.70 [-4.53, -2.79] | 0.024 | -4.30 [-5.12, -3.39] | 39.34 | -5.28 | -4.68 | -8.98 |
| textvqa_val | 432 | 44.32 | +0.63 [+0.39, +0.88] | 0.024 | -4.26 [-4.88, -3.68] | 0.024 | -4.90 [-5.54, -4.29] | 43.18 | +1.14 | +1.77 | -3.12 |
| chartqa_val | 72 | 14.38 | +0.38 [+0.00, +0.79] | 0.605 | -1.13 [-1.95, -0.34] | 0.056 | -1.51 [-2.31, -0.72] | 15.83 | -1.46 | -1.08 | -2.59 |
| chartqa_val | 144 | 15.05 | +0.33 [-0.07, +0.71] | 0.826 | -1.96 [-2.87, -1.10] | 0.024 | -2.29 [-3.18, -1.38] | 16.98 | -1.93 | -1.60 | -3.89 |
| chartqa_val | 288 | 17.07 | +0.36 [-0.07, +0.84] | 0.826 | +0.31 [-0.55, +1.18] | 0.960 | -0.05 [-1.03, +0.93] | 17.29 | -0.23 | +0.14 | +0.09 |
| chartqa_val | 432 | 19.17 | +0.10 [-0.22, +0.44] | 0.960 | -0.89 [-1.52, -0.26] | 0.065 | -0.99 [-1.66, -0.31] | 18.80 | +0.36 | +0.47 | -0.52 |

### logistic_lr0.0001: accuracy (%), mean over seeds; differences are seed means of per-item paired differences (image-grouped bootstrap, Holm over 24)

| bench | budget | frozen | I1 − frozen | p Holm | I2 − frozen | p Holm | I2 − I1 | uniform grid | frozen − grid | I1 − grid | I2 − grid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gqa_dev | 72 | 76.21 | +0.22 [-0.18, +0.61] | 1.000 | -5.92 [-6.66, -5.16] | 0.024 | -6.14 [-6.87, -5.35] | 79.18 | -2.96 | -2.75 | -8.88 |
| gqa_dev | 144 | 78.35 | +0.58 [+0.21, +0.96] | 0.026 | -5.89 [-6.68, -5.11] | 0.024 | -6.47 [-7.29, -5.67] | 81.12 | -2.77 | -2.19 | -8.66 |
| gqa_dev | 288 | 81.83 | +0.05 [-0.25, +0.32] | 1.000 | -0.66 [-1.15, -0.16] | 0.066 | -0.71 [-1.22, -0.19] | 82.56 | -0.72 | -0.68 | -1.39 |
| gqa_dev | 432 | 83.88 | +0.07 [-0.10, +0.23] | 1.000 | -0.37 [-0.71, -0.05] | 0.230 | -0.44 [-0.77, -0.12] | 83.50 | +0.38 | +0.45 | +0.01 |
| textvqa_val | 72 | 17.43 | +0.74 [+0.31, +1.18] | 0.024 | -5.70 [-6.34, -5.08] | 0.024 | -6.44 [-7.10, -5.78] | 25.22 | -7.79 | -7.05 | -13.49 |
| textvqa_val | 144 | 21.93 | +1.86 [+1.38, +2.32] | 0.024 | -7.48 [-8.20, -6.74] | 0.024 | -9.34 [-10.08, -8.50] | 32.55 | -10.62 | -8.76 | -18.09 |
| textvqa_val | 288 | 31.86 | +1.70 [+1.24, +2.15] | 0.024 | -3.33 [-4.10, -2.52] | 0.024 | -5.03 [-5.84, -4.20] | 39.34 | -7.48 | -5.77 | -10.81 |
| textvqa_val | 432 | 44.43 | +0.54 [+0.28, +0.78] | 0.024 | -5.69 [-6.30, -5.03] | 0.024 | -6.23 [-6.83, -5.58] | 43.18 | +1.25 | +1.79 | -4.44 |
| chartqa_val | 72 | 14.34 | +0.43 [+0.03, +0.85] | 0.279 | -1.16 [-1.93, -0.43] | 0.036 | -1.60 [-2.37, -0.83] | 15.83 | -1.49 | -1.06 | -2.66 |
| chartqa_val | 144 | 15.28 | +0.31 [-0.09, +0.74] | 0.945 | -2.40 [-3.24, -1.56] | 0.024 | -2.71 [-3.60, -1.85] | 16.98 | -1.70 | -1.39 | -4.10 |
| chartqa_val | 288 | 16.79 | +0.14 [-0.35, +0.63] | 1.000 | +0.10 [-0.72, +0.91] | 1.000 | -0.03 [-0.85, +0.79] | 17.29 | -0.50 | -0.36 | -0.40 |
| chartqa_val | 432 | 19.31 | -0.17 [-0.50, +0.14] | 1.000 | -0.85 [-1.63, -0.05] | 0.279 | -0.68 [-1.39, +0.09] | 18.80 | +0.50 | +0.33 | -0.35 |

### Holm-significant contrasts (p Holm < 0.05): (positive, negative) of 12 per intervention

- vcs_lr0.0001: I1 (3, 0), I2 (0, 7)
- logistic_lr0.0001: I1 (5, 0), I2 (0, 8)

### Soft-hard gap of the gate (mean over the units that have it; soft − hard selection loss, logged at the end of phase B)

| objective | arm | budget 72 | budget 144 | budget 288 | budget 432 |
|---|---|---|---|---|---|
| logistic_lr0.0001 | I1 | -0.0024 (n 3) | -0.0012 (n 3) | -0.0005 (n 3) | -0.0002 (n 3) |
| logistic_lr0.0001 | I2 | -0.0023 (n 3) | -0.0020 (n 3) | -0.0010 (n 3) | -0.0002 (n 3) |
| logistic_lr0.0001 | frozen | -0.0013 (n 3) | -0.0008 (n 3) | -0.0002 (n 3) | -0.0000 (n 3) |
| vcs_lr0.0001 | I1 | -0.0034 (n 3) | -0.0021 (n 3) | -0.0014 (n 3) | -0.0006 (n 3) |
| vcs_lr0.0001 | I2 | -0.0034 (n 3) | -0.0040 (n 3) | -0.0005 (n 3) | -0.0004 (n 3) |
| vcs_lr0.0001 | frozen | -0.0024 (n 3) | -0.0014 (n 3) | -0.0010 (n 3) | -0.0004 (n 3) |

## 3. Limits
- D1-FULL only, 3 seeds per objective, DEV / val benchmarks, fixed budgets.
- Direct is not read here; it awaits the H100 secondary.
- The cross-scale comparison with REV9 is descriptive.
