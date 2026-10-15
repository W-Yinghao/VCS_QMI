# V7C — DEV secondary system table (EvalMuse DEV; descriptive, unadjusted) — 2026-10-15

**Provenance.**
- Frozen scorer `scripts/v7x/x40_dev_external.py` (V7X-40), CPU job 1041052.
- Inputs: every DEV score file, all present (`missing: []`).
- The external DEV rows were completed in jobs 1040990–1040992 (DSG generation and scoring; TIFA questions and scoring; VQAScore B1). Each step ran on the GPU type where that step had run before.
- DEV labels were already exposed in V7; this table is descriptive.
- B0, the released FGA-BLIP2, was trained on all of EvalMuse TRAIN, which contains DEV, so its rows are exposed.

## Reading
- **In distribution (EvalMuse DEV), the EvalMuse-trained systems are far ahead of every external system.**
  - Overall: SRCC 0.76–0.77 for N0 / M0 / M1 / M2, against 0.47–0.48 for VQAScore (B1, B2) and HPSv3, 0.40 for DSG, 0.27 for TIFA.
  - Within-prompt pairwise accuracy: about 0.70 against 0.56–0.64.
- **On GenAI-Rank, outside EvalMuse, VQAScore is ahead of M2-E** (V7C-04: −0.119 / −0.085). The ordering reverses between the in-distribution table and the external benchmark. That is the generalisation gap the closeout records, not a contradiction: the DEV table measures fit to EvalMuse's own annotation.
- **Element level, common set: M1 (logistic) and M2 (VCS) are indistinguishable.**
  - Natural Brier 0.174–0.175 for both; M0 (L1) 0.183–0.185.
  - PN-VQA, the only external system with element outputs, is at 0.239.

## Tables (generated from `V7C_DEV_results/x40_dev_external.json`)
| system | images | SRCC [95 %] | PLCC | within-prompt pairwise accuracy [95 %] |
|---|---|---|---|---|
| B0_FGA_released_O (exposed) | 3257 | 0.905 [0.892, 0.916] | 0.900 | 0.794 [0.780, 0.807] |
| N0_s0_O | 3257 | 0.764 [0.728, 0.794] | 0.759 | 0.702 [0.685, 0.718] |
| N0_s1_O | 3257 | 0.766 [0.732, 0.794] | 0.760 | 0.702 [0.685, 0.718] |
| N0_s2_O | 3257 | 0.764 [0.729, 0.794] | 0.759 | 0.702 [0.684, 0.718] |
| M0_semantic_l1-s0_O | 3257 | 0.768 [0.731, 0.798] | 0.762 | 0.707 [0.691, 0.723] |
| M0_semantic_l1-s0_E | 2591 | 0.658 [0.598, 0.709] | 0.638 | 0.694 [0.673, 0.715] |
| M0_semantic_l1-s1_O | 3257 | 0.765 [0.731, 0.795] | 0.761 | 0.703 [0.687, 0.719] |
| M0_semantic_l1-s1_E | 2591 | 0.656 [0.594, 0.709] | 0.642 | 0.695 [0.673, 0.715] |
| M0_semantic_l1-s2_O | 3257 | 0.758 [0.721, 0.791] | 0.753 | 0.705 [0.689, 0.722] |
| M0_semantic_l1-s2_E | 2591 | 0.635 [0.571, 0.689] | 0.621 | 0.705 [0.685, 0.724] |
| M1_semantic_logistic-s0_O | 3257 | 0.764 [0.727, 0.795] | 0.759 | 0.709 [0.693, 0.724] |
| M1_semantic_logistic-s0_E | 2591 | 0.663 [0.598, 0.717] | 0.645 | 0.706 [0.687, 0.725] |
| M1_semantic_logistic-s1_O | 3257 | 0.767 [0.732, 0.796] | 0.762 | 0.704 [0.688, 0.721] |
| M1_semantic_logistic-s1_E | 2591 | 0.673 [0.612, 0.724] | 0.656 | 0.702 [0.682, 0.722] |
| M1_semantic_logistic-s2_O | 3257 | 0.772 [0.738, 0.802] | 0.767 | 0.703 [0.686, 0.718] |
| M1_semantic_logistic-s2_E | 2591 | 0.667 [0.603, 0.721] | 0.647 | 0.701 [0.682, 0.720] |
| M2_semantic_vcs-s0_O | 3257 | 0.764 [0.726, 0.796] | 0.757 | 0.706 [0.690, 0.722] |
| M2_semantic_vcs-s0_E | 2591 | 0.664 [0.601, 0.716] | 0.644 | 0.703 [0.684, 0.722] |
| M2_semantic_vcs-s1_O | 3257 | 0.763 [0.729, 0.793] | 0.759 | 0.702 [0.686, 0.718] |
| M2_semantic_vcs-s1_E | 2591 | 0.667 [0.607, 0.718] | 0.652 | 0.698 [0.678, 0.718] |
| M2_semantic_vcs-s2_O | 3257 | 0.764 [0.728, 0.795] | 0.758 | 0.702 [0.686, 0.718] |
| M2_semantic_vcs-s2_E | 2591 | 0.669 [0.606, 0.721] | 0.651 | 0.707 [0.688, 0.726] |
| B1_VQAScore | 3257 | 0.470 [0.412, 0.524] | 0.447 | 0.624 [0.605, 0.643] |
| B2_VQAScore_Qwen25VL7B | 3257 | 0.485 [0.429, 0.538] | 0.470 | 0.632 [0.612, 0.651] |
| B5_HPSv3 | 3257 | 0.483 [0.430, 0.533] | 0.479 | 0.639 [0.621, 0.656] |
| B4T_TIFA | 3257 | 0.274 [0.202, 0.344] | 0.293 | 0.558 [0.544, 0.572] |
| B4D_DSG | 3257 | 0.399 [0.326, 0.466] | 0.418 | 0.573 [0.559, 0.587] |
| B3_PNVQA_own | 3257 | 0.344 [0.273, 0.412] | 0.385 | 0.593 [0.573, 0.613] |
| B3_PNVQA_evalmuse_elements | 3257 | 0.353 [0.279, 0.420] | 0.388 | 0.594 [0.574, 0.613] |

Element level, common set of 11997 elements:

| system | natural Brier (macro) | AUROC (prompt macro) | majority accuracy |
|---|---|---|---|
| B0_FGA_released_O (exposed) | 0.1740 | 0.935 | 0.828 |
| N0_s0_O | 0.1834 | 0.918 | 0.801 |
| N0_s1_O | 0.1830 | 0.919 | 0.802 |
| N0_s2_O | 0.1843 | 0.919 | 0.801 |
| M0_semantic_l1-s0 | 0.1826 | 0.940 | 0.814 |
| M0_semantic_l1-s1 | 0.1851 | 0.934 | 0.813 |
| M0_semantic_l1-s2 | 0.1842 | 0.936 | 0.804 |
| M1_semantic_logistic-s0 | 0.1742 | 0.940 | 0.814 |
| M1_semantic_logistic-s1 | 0.1746 | 0.945 | 0.813 |
| M1_semantic_logistic-s2 | 0.1745 | 0.943 | 0.812 |
| M2_semantic_vcs-s0 | 0.1747 | 0.939 | 0.811 |
| M2_semantic_vcs-s1 | 0.1751 | 0.942 | 0.811 |
| M2_semantic_vcs-s2 | 0.1748 | 0.940 | 0.811 |
| B3_PNVQA_evalmuse_elements | 0.2385 | 0.678 | 0.696 |
