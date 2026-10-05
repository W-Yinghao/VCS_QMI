# P129 — v6 §8.2 per-dataset (a, κ) grid for matched JS on the A-P3 structure, CIFAR-10 and CIFAR-100, seed 0 — report — 2026-10-05

Pre-registration `P129_V6_JS_AFFINE_GRID_PREREG_FROZEN_20261004.md` (owner: "JS tuning grid还是要的"); results-only commit `166d621` (`P129_results.txt`,
`reports/P129/`).  Each cell = that dataset's JS-AP3 seed-0 config with only (a, b = −aκ) changed; (2, 0.5) = JS-AP3 seed 0 (P114 / P120).
Development validation split; official test closed.  Addenda (frozen while the grid ran, submitted early): 1 — C10 (2, 0.25) seeds 1–2;
2 — C100 (2, 0.25) seeds 1–2 (now a non-selected extra); 3 — C100 (3, 0.5) seeds 1–2.

## 1. Results (linear / kNN %, seed 0)

| (a, κ) | CIFAR-10 JS | vs JS (2, .5) | vs VCS A-P3 | CIFAR-100 JS | vs JS (2, .5) | vs VCS A-P3 |
|---|---|---|---|---|---|---|
| (2, 0.5) JS-AP3 | 88.72 / 87.50 | — | −0.34 / +0.20 | 58.76 / 54.80 | — | −1.44 / −1.12 |
| (1.5, 0.5) | 88.50 / 87.26 | −0.22 / −0.24 | −0.56 / −0.04 | 58.86 / 54.48 | +0.10 / −0.32 | −1.34 / −1.44 |
| (2, 0.25) | **89.26** / 87.68 | **+0.54** / +0.18 | +0.20 / +0.38 | 59.58 / 56.42 | +0.82 / +1.62 | −0.62 / +0.50 |
| (2, 0.75) | 88.52 / 87.06 | −0.20 / −0.44 | −0.54 / −0.24 | 58.24 / 52.20 | −0.52 / −2.60 | −1.96 / −3.72 |
| (3, 0.5) | 88.60 / 87.06 | −0.12 / −0.44 | −0.46 / −0.24 | **60.72** / 56.18 | **+1.96** / +1.38 | +0.52 / +0.26 |
| (3, 0.25) | 88.28 / 87.12 | −0.44 / −0.38 | −0.78 / −0.18 | 60.24 / **57.12** | +1.48 / +2.32 | +0.04 / +1.20 |

## 2. Reading (frozen rule)
- **Dataset-selected JS cells replace (2, 0.5) on both datasets:** CIFAR-10 (2, 0.25) +0.54 (≥ 0.30); CIFAR-100 (3, 0.5) +1.96 (≥ 0.50) — seeds 1–2
  running (addenda 1, 3).  The VCS grid (P127) kept (2, 0.5) on both datasets.
- **JS is much more sensitive to the scorer than VCS:** the JS grid spans 0.98 (C10) / 2.48 (C100) linear and up to 4.9 kNN on CIFAR-100; its best
  cells sit at a lower threshold-to-scale ratio (κ 0.25 at a = 2; a = 3 with κ 0.5 / 0.25), where VCS was flat or slightly worse.
- **Seed-0 position of tuned JS vs tuned VCS:** CIFAR-10 89.26 vs 89.06 (+0.20); CIFAR-100 60.72 vs 60.20 (+0.52), kNN 56.18 vs 55.92.  At the shared
  default (2, 0.5) VCS led (CIFAR-100: clear at 5 seeds, +0.80).  So the CIFAR-100 VCS advantage appears tied to the shared default scorer; whether
  tuned VCS and tuned JS differ is decided only by the pre-stated 3-seed paired comparison (P129 reading), followed by the P135 lr check.
- Also notable: JS (3, 0.25) CIFAR-100 kNN 57.12 — the highest kNN of any VCS / JS run on CIFAR-100 (SimCLR recipe 57.21).  One seed.
- Not claimed: anything from seed-0 differences alone; tuned-vs-tuned against SimCLR (its recipe is fixed by owner decision).
