# P129 addendum 1 — CIFAR-10 tuned VCS vs tuned JS (3 seeds) — report — 2026-10-05

Addendum `P129_ADDENDUM1_JS_C10_SEEDS12_FROZEN_20261005.md`; results-only commit `718d777` (`P129_tuned_c10_paired.{txt,json}`).  Selected cells:
VCS (P127) = (2, 0.5) = A-P3 seeds 0–2 (P107); JS (P129) = (2, 0.25) — confirmed as the CIFAR-10 selection after the last grid cell ((3, 0.5) 88.60).

| seed | VCS A-P3 (2, 0.5) | JS (2, 0.25) | VCS − JS linear | VCS − JS kNN |
|---|---|---|---|---|
| 0 | 89.06 / 87.30 | 89.26 / 87.68 | −0.20 | −0.38 |
| 1 | 88.96 / 87.48 | 88.60 / 87.48 | +0.36 | 0.00 |
| 2 | 89.06 / 87.36 | 89.06 / 87.02 | 0.00 | +0.34 |
| mean | 89.03 / 87.38 | 88.97 / 87.39 | **+0.05 [−0.65, +0.76] — close** | **−0.01 [−0.91, +0.88] — close** |

## Reading (frozen P129 rule)
- **Tuned VCS and tuned JS are close on CIFAR-10** (|Δ| < 0.3 on both readouts).  The seed-0 lead of tuned JS (+0.20) did not hold; JS's
  seed spread at (2, 0.25) is 0.66 vs VCS's 0.10.  At the shared default the CIFAR-10 contrast was also close (P114, +0.21), so on CIFAR-10 the two
  losses reach the same accuracy with and without per-loss scorer tuning; VCS gets there at the default scorer, JS needs a lower threshold.
- No lr check is triggered (it applies to a clear difference).  CIFAR-100 tuned comparison: JS (3, 0.5) seeds 1–2 (addendum 3) are queued.
