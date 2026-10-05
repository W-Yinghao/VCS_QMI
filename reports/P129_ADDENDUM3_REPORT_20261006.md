# P129 addendum 3 — CIFAR-100 tuned VCS vs tuned JS (3 seeds) — report — 2026-10-06

Addenda 2 / 3 (`P129_ADDENDUM2_…`, `P129_ADDENDUM3_JS_C100_A3K05_SEEDS12_FROZEN_20261005.md`); results-only commit `7bdb392` (`P129_tuned_c100_paired.{txt,json}`).
Selected cells by the frozen rules: VCS (P127) = (2, 0.5) (its best cell (3, 0.5) +0.44 was below the 0.50 threshold); JS (P129) = (3, 0.5) (+1.96).

| seed | VCS A-P3 (2, 0.5) | JS (3, 0.5) | VCS − JS linear | VCS − JS kNN |
|---|---|---|---|---|
| 0 | 60.20 / 55.92 | 60.72 / 56.18 | −0.52 | −0.26 |
| 1 | 60.08 / 55.96 | 60.44 / 56.28 | −0.36 | −0.32 |
| 2 | 59.72 / 55.52 | 60.30 / 56.48 | −0.58 | −0.96 |
| mean | 60.00 / 55.80 | 60.49 / 56.31 | **−0.49 [−0.77, −0.20] — clear (JS better)** | −0.51 [−1.48, +0.45] — inconclusive |

Non-selected extra JS (2, 0.25), seeds 0–2: 59.58 / 56.42, 60.18 / 56.02, 59.18 / 55.52 (mean 59.65 / 55.99) — reported, not used for selection.

## Reading (frozen P129 rule)
- **With each loss at its rule-selected scorer, matched JS is clearly ahead of VCS on CIFAR-100 linear (−0.49, all three seeds); kNN inconclusive.**
  At the shared default scorer the same comparison was clearly in VCS's favour (+0.80 at 5 seeds, P120 addendum 1), and on CIFAR-10 the tuned
  comparison is close (+0.05).  So the CIFAR-100 VCS advantage is a property of the shared scorer; JS gains more from scorer selection
  (+1.96 vs VCS's sub-threshold +0.44 at seed 0; VCS's best tested cell (3, 0.5) 60.64 ≈ JS's 60.72 at seed 0 — descriptive only).
- **Required before a general statement:** the lr check for both losses at the selected cells.  VCS's selected cell is (2, 0.5), already covered by
  P135 (lr-insensitive); JS's selected cell (3, 0.5) is not → **addendum 4 (frozen now): JS C100 (3, 0.5) at lr 5e-4 and 2e-3, seed 0** (P135 found
  JS gains +0.60 at lr 2e-3 at (2, 0.5), so this can only widen the gap or leave it).  Until then the statement is limited to "at lr 1e-3".
