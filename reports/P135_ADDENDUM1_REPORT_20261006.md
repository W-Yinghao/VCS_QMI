# P135 addendum 1 — JS-AP3 CIFAR-100 at lr 2e-3, seeds 0–2 — report — 2026-10-06

Frozen addendum `P135_ADDENDUM1_JS_C100_LR2X_FROZEN_20261005.md`; results-only commit `291d7f1` (`reports/P135/addendum1_*`).  Shared scorer (2, 0.5), all-view
tokens, 4 views / B 256 / 800 epochs, development split; seed 0 from P135, seeds 1–2 new (59.50 / 56.24, 58.60 / 55.06).

| contrast (paired by seed, n = 3) | linear Δ [95 % CI] | per seed | label | kNN Δ [95 % CI] | label |
|---|---|---|---|---|---|
| JS lr 2e-3 − JS lr 1e-3 (P120) | −0.05 [−1.52, +1.42] | +0.60, −0.18, −0.56 | close | +0.45 [−1.33, +2.23] | inconclusive |
| VCS A-P3 (lr 1e-3) − JS lr 2e-3 | +0.85 [+0.18, +1.52] | +0.84, +0.58, +1.12 | **clear** | +0.07 [−0.86, +0.99] | close |

Reading (descriptive, as frozen): the seed-0 gain of JS at lr 2e-3 (+0.60, the trigger) does not hold over three seeds (mean 59.15 vs 59.20).
The shared-scorer CIFAR-100 result — VCS ahead of matched JS on linear, kNN about equal — keeps its sign and stays clear when JS uses the higher
learning rate (P120 add. 1 at lr 1e-3: +0.80 [+0.06, +1.54], five seeds).  The selected-scorer comparison (JS (3, 0.5)) is a separate protocol;
its lr check is P129 addendum 4 (lr × 2 cell pending).
