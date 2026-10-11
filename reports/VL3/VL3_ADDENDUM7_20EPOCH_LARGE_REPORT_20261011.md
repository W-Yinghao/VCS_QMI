# VL3 addendum 7 — the 20-epoch schedule on the six Large cells, three seeds — report — 2026-10-11

- **Protocol:** `VL3_ADDENDUM7_20EPOCH_LARGE_FROZEN_20261010.md` (`8693614`; aggregators extended to Large before results).
- **Results-only:** `9bd479e` (`VL3_results_e20.json`, `VL3_add2_results.json`; aggregation 1034114).
- **Runs:** 54 / 54 at 20 epochs.  One run (RefCOCO × CLIP-L softmax s1) had never been submitted, a feeding error.  It was found at the
  11-cell aggregation and run as 1034113, unchanged.
- **Gate 2:** as VL3.
- **10 epochs remains the primary VL3 schedule.**  Both schedules are reported in full.

## Pre-registered readings (12 cells = add. 4's B/16 + add. 7's Large; 95 % t, paired by seed)
| quantity | pooled over 12 cells | B/16 (6 cells) | Large (6 cells) |
|---|---|---|---|
| VCS − softmax at 20 epochs | +0.35 [+0.02, +0.68] | +0.24 [−0.16, +0.64] | +0.46 [−0.22, +1.15] |
| VCS − softmax at 10 epochs | +0.01 [−0.35, +0.36] | | |
| **schedule effect (VCS − softmax)₂₀ − (VCS − softmax)₁₀** | **+0.35 [+0.12, +0.57]** | +0.37 [−0.11, +0.85] | +0.32 [+0.07, +0.57] |
| VCS − JS at 20 / 10 epochs | +0.26 [+0.05, +0.46] / +0.07 [−0.01, +0.15] | | |
| schedule effect on VCS − JS | +0.18 [−0.02, +0.39] | | |
| 20 − 10 per objective | VCS +0.19 [+0.02, +0.35]; JS +0.00 [−0.14, +0.14]; softmax −0.16 [−0.32, −0.00] | | |

The aggregator log prints "over 12 cells" on the B/16 and Large rows.  The pooled values equal add. 4's 6-cell B/16 result and a 6-cell Large
pool, so this is a label error in the log; the numbers are not affected.

## Pre-fixed wording (reading 4)
- **The 12-cell pooled schedule effect lies above 0, so: "a longer shared schedule favours the estimator objectives by 0.35 points".**  The
  10-epoch tie (+0.01 [−0.35, +0.36]) stays the primary VL3 result.
- This supersedes add. 4's reading 3, "the comparison does not depend on the schedule" (B/16 only), as that protocol stated.
- **Where it comes from:** softmax loses slightly with the longer shared schedule (−0.16), VCS gains slightly (+0.19) and JS is unchanged.
  Every run selects its epoch on CAL within its own schedule; the two cosine schedules are not nested.
- Per the protocol (reading 5), no further schedules, seeds or recipe changes follow without a new pre-registration.

## Caveats
- DEV is reused (exploratory).
- Given annotated boxes only (VL3 setting): this does not speak to the detector-proposal task of R4 / r5.
- Three seeds; pooled over cells that share backbones and datasets.
