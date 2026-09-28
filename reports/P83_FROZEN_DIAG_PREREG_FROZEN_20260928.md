# Pre-registration — P83: estimator package v1 §8, estimator diagnostics on frozen SSL checkpoints (2026-09-28) — FROZEN before GPU compute

Source: package spec §4.3, §4.4, §8, §15; owner go 2026-09-28 (package first).  Code: `src/vcs_estim/frozen.py` (+ candidates/fitting additions),
`slurm/estim_frozen_diag.sbatch`.  Results → `P84_*` (per-checkpoint JSON in `outputs/P83_estim_frozen_diag/`, table `reports/P84_frozen_diag.md`).

## Checkpoints (inventory checked; epoch 20 was never saved)
4-view VCS recipe `P35_vcs_a5_views4_800ep_seed0`: initial (untrained), epoch 100, 400, 800.  SimCLR same budget `P41_simclr_views4_800ep_seed0`:
same four (no training critic → measurement critics only).  Original early recipe `P5_vcs_seed0` (2 views, MLP critic, no detach): initial and
epoch 200 (the only saved points).  Ten checkpoints.

## Protocol
Identity roles FIT / TUNE / SELECT / EVAL = 27k / 6k / 6k / 6k of the 45k fit images (seed 20260928, disjoint, hash in JSON); the run's own
training augmentation, models in eval() on deep copies; critic input = L2-normalised projector output z.  FIT pairs: two views of one image vs
K = 8 cyclic-shift partners; TUNE / SELECT / EVAL: independent (anchor, partner) blocks, SE by block.  (1) Training critic (VCS runs): J, residuals,
gates, saturation, and a separate train-mode gradient diagnostic (grad_left / grad_right; detach kept as is).  (2) Refitted measurement critics
C0 / C1 / C2 with VCS and matched JS (Adam 5e-4, 2000 updates, batch 256, SELECT every 100), VCS {C0, C1, C2} simplex mix on TUNE, one VCS
residual (C2, λ₀ = 0.5, λ on TUNE), JS dictionary.  (3) EVAL once: J per model, ΔJ vs the training critic, cost.  S and posterior MSE absent (no oracle).

## Changes after the fork's smoke, disclosed
C0's inner product is standardised by FIT (mean, sd) — an affine reparametrisation of the same class — because on the narrow-cone z both a = 0
(budget cannot reach the needed scale; smoke J 0.011) and a = 5, b = 0 (every pair saturates; smoke J −0.997) are unreadable.  A converged C0
(full-batch L-BFGS, 2 parameters) is added as a labelled diagnostic, outside the dictionary, to separate C0's budget from its class
(smoke 1013107: fixed-budget C0 0.118 vs converged 0.998 on a 1k-image subset with 100 updates).

## Reading (descriptive; no thresholds)
Per checkpoint: does any refitted critic, mix or residual exceed the training critic's EVAL J beyond 2 block SE; how the gap between the
training critic and the best refit evolves over training; where VCS and SimCLR representations differ in refit J; gates and gradients of the
training critic over training.  Nothing is re-selected after EVAL.
