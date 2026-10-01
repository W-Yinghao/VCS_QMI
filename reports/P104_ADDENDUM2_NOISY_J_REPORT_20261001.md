# P104 addendum 2 — N1 vs P95 τ 0.3 under the same noisy evaluation — report — 2026-10-01

Pre-registration: `P104_ADDENDUM2_NOISY_J_EVAL_FROZEN_20261001.md`. Jobs 1016900 (the three P95 runs) and 1016901 (N1), both exit 0.
Results-only commit: `1b81ee2`, files `P104_noisyJ_*.json`.
Evaluation only: the selection split, `critic_holdout` exactly as in training, τ 0.3, R_eval = 16.

**Parity check passes.** On N1, the script reproduces the noisy J that N1's training evaluation logged (`knn_epoch_*.json`):

| epoch | 100 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|
| script, noisy J | 0.9312 | 0.9462 | 0.9564 | 0.9604 | 0.9624 |
| training log, noisy J | 0.9326 | 0.9478 | 0.9573 | 0.9613 | 0.9632 |
| \|Δ\| | 0.0014 | 0.0016 | 0.0009 | 0.0009 | 0.0008 |

- Every \|Δ\| is under the 0.005 limit.
- The script reads about 0.001 lower everywhere, for clean J as well. That fits a small offset from the data-loader stream.
- Every run in the comparison below is scored by the same script, so the offset cancels out.

## Results at epoch 800 (P95: mean ± sd over seeds 0–2)

| run | training R | clean J | noisy J | clean gate | noisy gate | linear | kNN |
|---|---|---|---|---|---|---|---|
| P95 τ 0.3, seed 0 | 1 | 0.9779 | 0.9611 | 0.0471 | 0.0664 | 87.32 | 84.92 |
| P95 τ 0.3, seed 1 | 1 | 0.9778 | 0.9611 | 0.0472 | 0.0660 | 87.12 | 84.98 |
| P95 τ 0.3, seed 2 | 1 | 0.9785 | 0.9618 | 0.0471 | 0.0662 | 87.02 | 84.82 |
| P95 τ 0.3, mean | 1 | 0.9781 ± 0.0004 | 0.9613 ± 0.0004 | 0.0471 | 0.0662 | 87.15 ± 0.15 | 84.91 |
| **N1, seed 0** | **4** | **0.9791** | **0.9624** | **0.0442** | **0.0629** | **87.30** | **85.32** |

## Reading (descriptive, as pre-registered)
- **Averaging the loss over R = 4 draws raises the noisy estimator value the representation reaches, by a small amount.**
  - N1's noisy J is 0.0011 above the P95 mean, against a seed SD of 0.0004.
  - N1's clean J is 0.0010 above.
  - N1's noisy gate is 0.003 lower, against 0.066.
  - The same ordering holds at every epoch from 200 on.
  - This is consistent with R = 4 lowering the variance of the training gradient, though gradient variance was not measured here.
- **The gain does not show up in accuracy.**
  - N1's linear accuracy, 87.30, is inside the P95 range (87.02–87.32). It equals P95 seed 0 at the same seed: 87.30 vs 87.32.
  - On kNN, N1 is 0.4 above P95's three seeds (84.82–84.98). That is one seed, so no claim is made.
- **No accuracy difference between N1 and P95 is claimed.** N1 was not selected in P104 addendum 3, and §10.1 gives it no further seeds.
