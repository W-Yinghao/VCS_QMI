# P155 — official CIFAR-10 test of the final recipe A-P3 — protocol — FROZEN 2026-10-14 (before any official-test read of these runs)

**Owner, 2026-10-14:** "我们现在已经决定A-P3是最终配方，所以直接用测试集测啊" (A-P3 is the final recipe; evaluate it on the test set).  This closes the
gap noted in `reports/R4/R4_02_EVIDENCE_LEDGER_20261011.md` §6.1 ("No v3–v7 recipe was ever confirmed on held-out data").

## Units (`slurm/p155_official_test_units.txt`; every checkpoint is the one the selection-split decisions were made on, `epoch_800.pt`)
| cell | runs | test status before P155 |
|---|---|---|
| A-P3 (fixed tanh(2s − 1), all-view tokens, full gradient; 4 views, 800 ep) | `P107_AP3_views4_800ep_seed0–4` | never read |
| SimCLR at the same view / epoch budget (4 views, 800 ep) | `P41_simclr_views4_800ep_seed0–2`, `P111_simclr_views4_800ep_seed3–4` | seeds 0–2 read once in P68 (88.13 ± 0.07, files re-read, not re-run); seeds 3–4 never read |
| JS-AP3 (matched JS on the A-P3 recipe, P114) | `P114_JSAP3_views4_800ep_seed0–2` | never read |

## Evaluation (unchanged P67 / P68 machinery)
`scripts/final_official_test_eval.py` → `vcs_ssl.evaluate` protocol `final_official_test`: frozen-h linear head and kNN bank on the 45 000-image fit split,
the frozen pilot probe hyper-parameters, clean transform, scored on the 10 000 official test images; double unlock (`VCS_FINAL_ROUND=1`), no
overwrite.  Job `slurm/p155_official_test.sbatch` (RTX6000PRO / H100 / L40S, node51 and node60 excluded).
**Gate (inside the job, before the unlock):** the stand-in run of unit 1 (A-P3 seed 0 on the selection split) must reproduce its selection-split linear
accuracy (89.06) within 0.1 points; otherwise the job exits without opening the test partition.

## Pre-stated contrasts (`scripts/p155_contrasts.py`; paired by seed; 95 % paired t interval over seeds)
- **Primary:** A-P3 − SimCLR, official-test linear accuracy, seeds 0–4 (n = 5).  Reading: "A-P3 ahead of SimCLR on the official test" if the interval's
  lower end > 0; "behind" if its upper end < 0; otherwise "no detectable difference".  (Selection split, P107: +0.72 [+0.48, +0.95]; the script
  reproduces this exactly on those numbers.)
- **Secondary:** A-P3 − SimCLR kNN (seeds 0–4; selection split −0.45 [−0.92, +0.02]); A-P3 − JS-AP3 linear and kNN (seeds 0–2, n = 3; selection split
  +0.21 [−1.03, +1.46] linear, P114).  Same reading words; n = 3 intervals are descriptive.
- Reported for every run: test − selection (agreement of the development metric with the test).  The v2 recipe's P68 test value (86.65 ± 0.26) is
  quoted for reference only.

## Disclosures
- A-P3 was chosen on the selection split after the v3–v7 screens (seed 0 was the screen-selection seed; seeds 3–4 were fresh, P107); the test read
  confirms a selected recipe, it does not select one.  No recipe or checkpoint is changed after the read.
- Part of the comparator is already known: SimCLR seeds 0–2 on the test (P68).  The official partition was read once before (P68, 2026-09-28) for the
  v1 / v2 cells and the P41 controls; this is its second, owner-approved use, for the final recipe only.
- Not in P155: the strong-augmentation and CIFAR-100 A-P3 runs (other test partitions / variants need their own approval), any new training.
