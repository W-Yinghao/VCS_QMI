# P129 addendum 4 — lr check at the selected JS CIFAR-100 cell (3, 0.5) — report — 2026-10-07

Frozen addendum `P129_ADDENDUM4_JS_C100_LRCHECK_FROZEN_20261006.md`; results-only commit `5918335` (`reports/P129/addendum4_lr_gaps.json`).  Seed 0,
CIFAR-100, development split; VCS (2, 0.5) lrs from P135, JS (3, 0.5) lrs from this addendum; only optimizer.lr changes.

| lr | VCS (2, 0.5) linear / kNN | JS (3, 0.5) linear / kNN | VCS − JS linear | VCS − JS kNN |
|---|---|---|---|---|
| 5e-4 (× 0.5) | 60.38 / 55.20 | 60.22 / 55.02 | +0.16 | +0.18 |
| 1e-3 (registered) | 60.20 / 55.92 | 60.72 / 56.18 | −0.52 | −0.26 |
| 2e-3 (× 2) | 60.52 / 56.68 | 60.26 / 56.62 | +0.26 | +0.06 |

**Reading (frozen): "lr-dependent"** — the seed-0 gap changes sign across the three learning rates.  The 3-seed result at the registered lr 1e-3
(addendum 3: VCS − JS = −0.49 [−0.77, −0.20], clear) stays as reported, but it is a statement about lr 1e-3, not about the two losses in general.
JS (3, 0.5) is best at the registered lr (60.72) and loses ~0.5 at both neighbours; VCS (2, 0.5) varies within 0.32 over the three lrs.  Single
seed per lr; no tuned-optimum claim for either loss.
