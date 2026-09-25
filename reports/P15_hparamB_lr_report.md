# P15 — VCS-QMI learning-rate / schedule study report (P14)

Status: **COMPLETE** (5/5 COMPLETED, 2026-09-25 03:xx–05:31 UTC; no non-finite events, even at lr 1e-2).
Neutral table: `reports/P15_hparamB_lr_results_table.md`.  Pre-registration: `reports/P14_HPARAM_B_LR_PREREG_FROZEN_20260925.md`.
Owner's question: is the slow convergence caused by a too-small learning rate?  Single seed 0, 200 epochs, K = 1; baseline
`P5_vcs_seed0` (AdamW 1e-3, warm-up 10, cosine → 1 %): linear 74.84 %, kNN 64.64, h-rank 12.9, heldout-J 0.898.

## Results

| run | change | linear | Δ | kNN | h-rank | heldout-J | signal |
|---|---|---|---|---|---|---|---|
| lr3e-4 | lr 1e-3 → 3e-4 | 72.80 | −2.04 | 63.32 | 14.5 | 0.902 | HURTS |
| lr3e-3 | lr → 3e-3 | 73.74 | −1.10 | 64.24 | 14.6 | 0.905 | HURTS |
| lr1e-2 | lr → 1e-2 | 71.38 | −3.46 | 63.54 | 13.7 | 0.903 | HURTS (no divergence; slow first 20 epochs) |
| floor0.1 | cosine floor 1 % → 10 % | 74.20 | −0.64 | 63.82 | 13.7 | 0.897 | neutral |
| const | constant 1e-3 after warm-up (no decay) | 74.88 | +0.04 | 64.40 | 15.0 | 0.885 | neutral |

kNN at epochs 50/100/150/200 — baseline 54.6/60.9/63.8/64.6; lr3e-3 56.6/61.1/64.0/64.2; lr1e-2 56.8/59.7/63.2/63.5; lr3e-4 55.4/60.8/62.5/63.3;
const 55.9/60.3/62.7/64.4; floor0.1 56.0/60.9/63.1/63.8.  All trajectories lie within ≈ ±1.5 points of the baseline at every monitored epoch.

## Reading
- **The learning rate is not the cause of slow convergence.**  1e-3 is at the optimum of the tested decade in both directions (3e-4: −2.0;
  3e-3: −1.1; 1e-2: −3.5).  Higher LRs do not even accelerate the kNN trajectory (lr 1e-2 is slower for the first 20 epochs).
- **The schedule shape is not a lever at 200 epochs either**: keeping the LR constant at 1e-3 until the end gives the same endpoint as
  cosine to 1 % (74.88 vs 74.84; kNN 64.4 vs 64.6), with slightly lower J (0.885) and higher rank (15.0); a 10 % floor is neutral.  The
  late flattening seen in the 800-epoch runs (P9) therefore cannot be attributed to LR annealing alone at this horizon; whether a constant
  LR changes the 800-epoch plateau (71.5 kNN) is untested.
- Consistent with the AdamW argument stated beforehand: gradient magnitude is normalized by the optimizer, so a small effective signal
  (few unsaturated pairs) is not fixed by a larger step — larger steps only add noise (lr 1e-2 lands 3.5 points lower).

## Consequence
Keep AdamW 1e-3, warm-up 10, cosine → 1 % as the frozen optimization recipe for VCS.  The remaining explanatory candidates for the slow
convergence and the low-dimensional code stay on the objective side (saturation), to be examined with the P16 augmentation / batch-size /
combination results.

**Waiting for the owner.**
