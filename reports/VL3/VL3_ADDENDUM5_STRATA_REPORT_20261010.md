# VL3 addendum 5 — per-query strata of the complete 12-cell table — report — 2026-10-10

Protocol `VL3_ADDENDUM5_STRATA_FROZEN_20261010.md`; results-only commit `f172da6` (`reports/VL3/VL3_add5_results.json`, `scripts/vl3_add5_aggregate.py`,
job 1033468).  Evaluation of the 108 saved CAL-Top-1 checkpoints on DEV (given boxes), per query.  **Gate S (reloaded fp16 checkpoint reproduces
the run's DEV Top-1 within 0.3 point): 108 / 108 pass** (largest gap 0.11).

## Strata (fixed in advance)
- Length tertile cuts (words): RefCOCOg ≤ 6 / 7–9 / > 9; RefCOCO and RefCOCO+ ≤ 2 / 3–4 / > 4.
- Location-word share: RefCOCOg 1 953 of 6 442 queries; RefCOCO 7 462 of 11 698.
- Referred objects per image: 2 / 3 / ≥ 4.

## 1. Primary (pre-registered): expression length
**S1 long − short of (VCS − softmax), pooled over 12 cells: −0.02 [−0.36, +0.33].**  Pre-fixed wording: **"expression length does not explain the
dataset pattern."**  The per-cell values are evenly split (−0.77 to +0.73; 6 positive, 6 negative).  Within each length tertile VCS − softmax is ≈ 0
(short −0.01, long −0.03).  The dataset pattern in the complete report (RefCOCOg +0.60, RefCOCO+ −0.47) therefore does not come from longer
expressions as such.

## 2. Secondary (descriptive; six secondary contrasts, no multiplicity correction)
| contrast (pooled over cells, 95 % t) | VCS − softmax | VCS − JS |
|---|---|---|
| S2 location word: with − without (8 cells) | **−0.62 [−1.15, −0.10]** | **+0.35 [+0.14, +0.56]** |
| — within "without location words" | **+0.63 [+0.27, +1.00]** | −0.03 [−0.07, +0.01] |
| — within "with location words" | +0.01 [−0.62, +0.64] | +0.32 [+0.12, +0.52] |
| S3 ≥ 4 − 2 referred objects (12 cells) | +0.21 [−0.25, +0.67] | +0.07 [−0.31, +0.45] |
| S1 long − short (12 cells) | −0.02 [−0.36, +0.33] | +0.08 [−0.06, +0.22] |

- **Location words.**  On RefCOCO and RefCOCOg expressions *without* a location word, VCS ranks better than softmax (+0.63).  On expressions
  *with* one, they tie.  The with − without contrast is negative in 7 of 8 cells (all four RefCOCO cells, −0.45 to −1.47).  Relative to JS, VCS
  gains on location expressions (+0.32).
- This does not match a simple "RefCOCO+ has no location words, so softmax wins there" story.  Within RefCOCO / RefCOCOg the non-location
  expressions favour VCS, yet on RefCOCO+ (all non-location) softmax leads.  So the RefCOCO+ deficit is tied to that dataset, not to the absence of
  location words.  What differs on RefCOCO+ is left open.
- **Number of candidates:** no clear effect.

## 3. Reading
The dataset pattern of the complete table is not explained by expression length (pre-registered, null) or by the number of candidates.
Location wording interacts with the objective within RefCOCO / RefCOCOg in a direction that the RefCOCO+ result does not follow.  None of this
changes the pooled VL3 statement ("VCS matches the task loss after full fine-tuning").  The secondary findings stay descriptive, and nothing here
calls for new training.
