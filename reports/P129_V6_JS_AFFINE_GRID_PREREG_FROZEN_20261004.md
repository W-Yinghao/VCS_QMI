# Pre-registration — P129: v6 §8.2 per-dataset affine grid for matched JS on the A-P3 structure (CIFAR-10 and CIFAR-100, seed 0) — FROZEN 2026-10-04T17:13:25Z

Owner 2026-10-04: "全部提交，但是JS tuning grid还是要的。simclr取消是因为这个是个很成熟的方法，最优参数都是固定的…我们的方法还是全新的" — the
matched-JS control belongs to our method family and gets the same (a, κ) budget as VCS (P127); SimCLR keeps its recipe (P128 withdrawn).
Grid fixed by v6 §8.2 (identical to P127): (a, κ) ∈ {(1.5, .5), (2, .25), (2, .5), (2, .75), (3, .5), (3, .25)}; (2, .5) is the existing JS-AP3 seed 0
(CIFAR-10 P114 88.72 / 87.50; CIFAR-100 P120 58.76 / 54.80) — 5 new cells per dataset, 10 units.  Each config differs from that dataset's JS-AP3 seed-0
config only in cosine_scale_init = a and cosine_bias_init = −aκ (verified by diff; the all-view matched-JS loss reads f = a·C + b from these, the
same logits as VCS).  `configs/make_p129_configs.py`, `configs/P129_SHA256.json`, `slurm/p129_lines.txt`.

## Pre-stated reading
- Per dataset, final frozen-h linear (primary) and kNN; selection only on the development validation labels; official test closed.
- **Dataset-selected JS cell** = highest linear; it replaces (2, .5) only if it beats JS (2, .5) by ≥ 0.30 (CIFAR-10) / ≥ 0.50 (CIFAR-100) (P127's
  thresholds); a replacing cell gets seeds 1–2 (addendum, submitted as soon as the rule triggers).
- **Tuned VCS vs tuned JS** (both families searched over the same grid, the same budget): once P127 and P129 are complete and each dataset's
  selected VCS and JS cells have 3 seeds, paired by seed: selected-VCS − selected-JS, mean and 95 % t interval, P114 labels (close |Δ| < 0.3;
  clear |Δ| ≥ 0.3 with the interval excluding 0; otherwise inconclusive).  A clear difference still needs the pre-specified lr / optimiser-budget
  check for both losses before a general statement (the learning rate was searched for neither).  SimCLR stays at its recipe and is not part of the
  tuned comparison.
- No early stopping on any monitor.
## Compute
10 × 800 epochs; RTX6000PRO / H100 / L40S (node51, node60 excluded); normal QOS.
