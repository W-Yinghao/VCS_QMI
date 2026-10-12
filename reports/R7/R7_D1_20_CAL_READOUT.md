# r7 D1-20 — CAL-only readout (DEV sealed until D1-30)

Every number is from CAL (2,000 images; 95,868 question × mask pairs); the frozen selection rule (max CAL Spearman between the
selector's risk score r(K) and the realised teacher h) already chose each unit's checkpoint.  The native criterion is a pre-declared
secondary, never used for selection.

## 1. Learning-rate search (seed 0) and the frozen choice

| objective | lr | selected step | CAL Spearman(r, h) | native-selected step | its Spearman |
|---|---|---|---|---|---|
| vcs | 0.0001 | 9000 | 0.0874 | 4000 | 0.0669 |
| vcs | 0.0003 | 8000 | 0.0920 | 4000 | 0.0592 |
| logistic | 0.0001 | 11000 | 0.0961 | 4000 | 0.0740 |
| logistic | 0.0003 | 13000 | 0.0922 | 4000 | 0.0510 |
| direct | 0.0001 | 4000 | 0.1685 | 4000 | 0.1685 |
| direct | 0.0003 | 3000 | 0.1660 | 3000 | 0.1660 |

Frozen choice (lr_choice.json; the frozen planner's rule): vcs 0.0003 (CAL Spearman 0.0001: 0.0874, 0.0003: 0.0920); logistic 0.0001 (CAL Spearman 0.0001: 0.0961, 0.0003: 0.0922); direct 0.0001 (CAL Spearman 0.0001: 0.1685, 0.0003: 0.1660)

## 2. Main units at the chosen rates (seeds 0–2)

| unit | selected step | CAL Spearman(r, h) | native-selected Spearman | best eval Spearman | dynamic mean kept (0.25 / 0.5 / 0.75) | soft − hard gap at budgets 72 / 144 / 288 / 432 | h |
|---|---|---|---|---|---|---|---|
| vcs_lr0.0003_s0 | 8000 | 0.0920 | 0.0592 | 0.0920 | 0.25 / 0.5 / 0.75 | -0.064 / -0.037 / -0.014 / -0.002 | 1.94 |
| vcs_lr0.0003_s1 | 7000 | 0.0848 | 0.0504 | 0.0848 | 0.25 / 0.5 / 0.75 | -0.053 / -0.037 / -0.016 / -0.003 | 1.95 |
| vcs_lr0.0003_s2 | 10000 | 0.0859 | 0.0359 | 0.0859 | 0.25 / 0.5 / 0.75 | -0.053 / -0.033 / -0.012 / -0.002 | 1.95 |
| logistic_lr0.0001_s0 | 11000 | 0.0961 | 0.0740 | 0.0961 | 0.25 / 0.5 / 0.75 | -0.003 / -0.001 / +0.001 / +0.000 | 1.95 |
| logistic_lr0.0001_s1 | 7000 | 0.0940 | 0.0631 | 0.0940 | 0.25 / 0.5 / 0.75 | -0.004 / -0.002 / -0.002 / -0.001 | 1.95 |
| logistic_lr0.0001_s2 | 11000 | 0.0885 | 0.0664 | 0.0885 | 0.25 / 0.5 / 0.75 | -0.004 / -0.007 / -0.008 / -0.002 | 1.95 |
| direct_lr0.0001_s0 | 4000 | 0.1685 | 0.1685 | 0.1685 | 0.25 / 0.5 / 0.75 | +0.000 / +0.000 / -0.000 / +0.000 | 1.95 |
| direct_lr0.0001_s1 | 1000 | 0.1701 | 0.1621 | 0.1701 | 0.25 / 0.5 / 0.75 | +0.000 / -0.000 / -0.000 / -0.000 | 1.95 |
| direct_lr0.0001_s2 | 1000 | 0.1742 | 0.1397 | 0.1742 | 0.25 / 0.5 / 0.75 | -0.000 / -0.000 / -0.000 / -0.000 | 1.95 |

## 3. Ablations (seeds 0–2)

| unit | selected step | CAL Spearman(r, h) | native-selected Spearman | best eval Spearman | dynamic mean kept | soft − hard gap at budgets 72 / 144 / 288 / 432 | h |
|---|---|---|---|---|---|---|---|
| vcs_lr0.0003_s0_noq | 6000 | 0.0597 | 0.0483 | 0.0597 | 0.25 / 0.5 / 0.75 | -0.004 / -0.001 / +0.000 / -0.000 | 1.95 |
| vcs_lr0.0003_s1_noq | 5000 | 0.0620 | 0.0422 | 0.0620 | 0.25 / 0.5 / 0.75 | -0.069 / -0.038 / -0.044 / -0.015 | 1.95 |
| vcs_lr0.0003_s2_noq | 8000 | 0.0499 | 0.0499 | 0.0499 | 0.25 / 0.5 / 0.75 | -0.046 / -0.018 / -0.007 / -0.001 | 4.83 |
| direct_lr0.0001_s0_dtarget | 3000 | 0.1852 | 0.1852 | 0.1852 | 0.25 / 0.5 / 0.75 | -0.000 / -0.000 / +0.000 / +0.000 | 1.95 |
| direct_lr0.0001_s1_dtarget | 3000 | 0.1795 | 0.1783 | 0.1795 | 0.25 / 0.5 / 0.75 | -0.000 / -0.000 / -0.000 / +0.000 | 1.95 |
| direct_lr0.0001_s2_dtarget | 7000 | 0.1886 | 0.1769 | 0.1886 | 0.25 / 0.5 / 0.75 | -0.001 / -0.001 / -0.000 / -0.000 | 1.95 |

## 4. Summary (CAL Spearman of the selected checkpoint, mean ± sd over seeds)

| arm | n | mean | sd | values |
|---|---|---|---|---|
| vcs | 3 | 0.0875 | 0.0039 | 0.0920, 0.0848, 0.0859 |
| logistic | 3 | 0.0929 | 0.0039 | 0.0961, 0.0940, 0.0885 |
| direct | 3 | 0.1709 | 0.0029 | 0.1685, 0.1701, 0.1742 |
| A1 no question (vcs) | 3 | 0.0572 | 0.0064 | 0.0597, 0.0620, 0.0499 |
| A2 target d (direct) | 3 | 0.1844 | 0.0046 | 0.1852, 0.1795, 0.1886 |

## 5. Reading (coordinator, CAL only; DEV sealed — the D1-30 primary contrasts decide, not this table)
- **S3 (direct regression of h) ranks the realised risk about twice as well as the posterior selectors on CAL:** Spearman 0.171 ± 0.003 against
  0.088 ± 0.004 (S1 VCS) and 0.093 ± 0.004 (S2 logistic).  This criterion is S3's own training target, so it favours S3 by construction (frozen
  `selection_asymmetry_disclosure`); it is a ranking of which masks hurt, not answer quality.
- **VCS ≈ logistic** on this criterion (difference 0.005, within about one seed sd).
- **The question matters for VCS:** without it (A1) the Spearman drops from 0.088 to 0.057.
- **Training on the squared output gap d (A2) ranks h slightly better than training on h itself (0.184 vs 0.171)**, consistent with d being a
  smoother proxy of the noisier realised h (D1-10).
- **Absolute levels are low (0.06–0.18, pooled over masks and budgets).**  For orientation only, not like-for-like: the teacher's own d ranked h
  with Spearman 0.11–0.47 per mechanism × budget cell on CAL (D1-10), and the selectors see only cheap features, not the teacher's outputs.
- The native criterion would have chosen earlier checkpoints for S1 / S2 with lower CAL Spearman (0.036–0.074); for S3 it coincides or nearly so.
- The soft − hard gap is −0.05 to −0.06 at 72 tokens for S1, about −0.003 for S2 and about 0 for S3 (descriptive).
- Unit records (CAL only): `reports/R7/D1_20_units/*.json`; frozen choice: vcs 3e-4, logistic 1e-4, direct 1e-4 (execution note E4).
