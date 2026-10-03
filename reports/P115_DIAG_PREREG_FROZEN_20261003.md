# Pre-registration — P115 diagnostics: augmentation and gradient organisation of standard vs strong-augmentation runs (v5 NEXT-A-AUG §3 "对应诊断"), 2026-10-03 — FROZEN 2026-10-03T12:30:24Z (job 1 on existing runs; CPU gate 1019935, CPU timing 1019989)

Status: FROZEN 2026-10-03T12:30:24Z (main session; owner 2026-10-03: execute the v5 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the full jobs.
Source: `CS_QMI/VCS_Server_Next_Round_v5_CN.md` §3 (diagnostics paired with the crop-only / jitter-only ablation) and
`VCS_Results_Review_and_Next_Plan_v5_CN.md` §2.2 (the P113 epoch-800 check covers only the final distribution).
Code: `src/vcs_diag/augdiag.py`, `scripts/p115_diag.py`, `slurm/p115_diag.sbatch`; tests `tests/test_p115_diag.py` (15).

**Read-only.**  Checkpoints are loaded into fresh model copies; run directories are never written.  Nothing filters, groups or reweights
training pairs — the run's own objective is only re-evaluated on fixed diagnostic batches.  The official test set is never read.

## Runs
- Job 1 (existing, seed 0): A-P3 std / strong (`P107_AP3_views4_800ep_seed0`, `P107_AP3_augstrong_views4_800ep_seed0`), G2 std / strong (`P104_G2_…_seed0`,
  `P111_G2_augstrong_…_seed0`), tuned SimCLR std / strong (`P41_simclr_views4_800ep_seed0`, `P89_simclr_views4_800ep_augstrong_seed0`), recipe VCS
  std / strong (`P35_vcs_a5_views4_800ep_seed0`, `P43_vcs_a5_views4_800ep_augstrong_seed0`).
- Job 2: the four P115 crop-only-strong / jitter-only-strong runs (A-P3 and SimCLR, seed 0) once all four are COMPLETED (run ids filled in from the
  frozen P115 ablation prereg).

## Checkpoints
Requested epochs 20, 100, 400, 800.  A requested epoch that was not saved maps to the nearest **trained** checkpoint (initial.pt is never a
substitute) and a substitute already used is measured once; every mapping is recorded in the output (`checkpoint_map`).  In practice:
A-P3 / G2 runs have epochs 20 / 100 / 400 / 800; SimCLR and recipe-VCS runs have no epoch-20 checkpoint (→ 100, measured once), so their earliest
point is epoch 100 — early-training comparisons at epoch 20 are therefore only available for the fixed-scorer runs (disclosed, not repaired).

## Fixed inputs (identical for every run and checkpoint)
- Base images for gradients / distributions / crop groups: the first `grad_batches × B` FIT uids (B = the run's batch, 256; grad_batches = 2) after a
  permutation seeded 20261003 — the same uids for all runs (shared dev45k/val5k manifest; `base_uids_sha256` recorded).
- Views: the run's configured training augmentation, with the RandomResizedCrop box drawn by torchvision `get_params` and recorded; the remaining
  ops (flip, colour jitter, grayscale, normalisation) are the run's own transform objects.  Draws come from a forked, seeded torch RNG (seed + 100 + batch)
  — independent of training randomness and identical across checkpoints of a run.  Standard and strong runs differ in their augmentation by design,
  so their boxes differ; the base images and the RNG stream are the same.
- Geometry images: the first 2 000 SELECTION uids after a permutation seeded 20261004, clean transform.
- Negatives: the run's own pairing (cross-view K = 8 cyclic shifts drawn from a fixed pair seed; all-view tokens; SimCLR in-batch).
- BN in training mode on throw-away copies (as in a training step; buffers discarded); the critic is scored clean.

## Recorded per checkpoint (per batch and mean over the 2 batches)
1. **Distributions** of positive / negative s; for VCS also T, residuals 1 − T (P) and −1 − T (Q), 1 − T² (P and Q), the critic's a, b, κ = −b/a,
   the fraction of positives with s < κ and of negatives with s > κ.  SimCLR: s only (no T).
2. **P / Q gradient split of the run's own training loss** (L = L_P + L_Q exactly; tested against the training implementation):
   VCS: L_P = Σ_P (−T + T²/2)/N_P-weights, L_Q = Σ_Q (T + T²/2)/N_Q-weights with the run's pairing scope and negative routing (detached right side where
   the run detaches); SimCLR: L_P = mean(−logit_pos), L_Q = mean(logsumexp over non-self entries, positive included).  Norms of ∂L_P and ∂L_Q and their
   cosine w.r.t. z, h, encoder parameters and projector parameters.
3. **Crop-overlap groups of positive pairs** by the true IoU of the two views' boxes and by the intersection area / image area (fixed bins
   [0, 0.1), [0.1, 0.25), [0.25, 0.5), [0.5, 1]).  Per group: count, mean s (and T, fraction below κ), loss share, encoder-gradient norm and its
   **share of the total positive-term encoder gradient** ⟨g_group, g_P⟩ / ‖g_P‖² (shares sum to 1), and cosine with g_P.
4. **Multi-view organisation.**  Per unordered view pair: the positive-term encoder gradient; pairwise cosines, mean off-diagonal cosine, and
   ‖Σ g‖ / Σ ‖g‖.  Per token (z-space, analytic pulls verified against autograd to < 1e-6 relative error): cancellation ‖Σ pulls‖ / Σ ‖pulls‖, overall and
   split by whether the token has a partner with IoU < 0.1 (strong-crop pairs).
5. **Geometry** of h and z on the clean held-out images: norm distribution, mean-vector norm, top-16 normalised spectrum, effective rank (diagnostic only;
   never a selection criterion).

## Reading (descriptive only; no mechanism is pre-written)
- Each quantity is reported per run and checkpoint, side by side for standard vs strong (and, with job 2, crop-only vs jitter-only), per method.
  One seed per cell: differences are described, not tested.
- The epoch-800 check (P113) found < 5 % of positives below κ in both G2 conditions, so "positives cross the fixed threshold" is **not** the default
  mechanism; and an epoch-800 checkpoint cannot exclude effects earlier in training — hence the 20 / 100 / 400 / 800 trajectory.
- Crop-overlap groups and per-group gradient shares are diagnostic decompositions; they do not imply that reweighting or filtering would help, and no
  such change is made or proposed here.
- Not claimed: causal statements from these correlational readouts; anything about seeds other than 0.

## Gate and smoke (disclosed)
- Tests: 15 pass (crop IoU on known boxes; L_P + L_Q equals the training loss and its gradient for VCS cross-view (detach / full), VCS all-view tokens
  (detach / full) and SimCLR; analytic pulls = autograd for fixed / learned cosine critics, all-view and SimCLR; RNG determinism and isolation;
  checkpoint mapping; CPU batch smoke with shares summing to 1).
- CPU gate job 1019935 (exit 0): tests + smoke (B 16, 1 batch, 64 geometry images) on A-P3 seed 0 (epochs 20, 800) and SimCLR seed 0 (20 → 100, 800);
  autograd check 7.6e-8 / 2.6e-7; IoU-group shares sum to 1.000.  Smoke numbers are code-path evidence only.
- GPU smoke: job 1019941 failed before measuring (torch.cuda.reset_peak_memory_stats before CUDA init — "Invalid device argument", as in P85; fixed by a
  zero-size allocation + guard); re-submitted as job 1019956 at full settings (B 256, 2 batches, 2 000 geometry images), epoch 800 of the same two runs.
  It was still queued behind the 8-GPU quota when this draft was written (its output will land in `reports/P115_DIAG_GATE_1019956/`).
- Full-setting CPU timing, job 1019989 (exit 0): A-P3 seed 0, epoch 800 (all-view tokens, the heaviest objective: 3 072 positives and 1 044 480
  negatives per batch), B 256, 2 batches, 2 000 geometry images, 16 CPU threads: **438 s per checkpoint** (augmentation 1.1 s).  Autograd check of the
  analytic pulls < 1e-4 (rounded 0.0); IoU-group shares sum to 1.  Values are code-path / cost evidence only.

## Cost
Job 1 = 8 runs: A-P3 and G2 (std, strong) × 4 checkpoints + SimCLR and recipe (std, strong) × 3 checkpoints (epoch 20 → 100, measured once) =
**28 checkpoints**.  Measured on CPU: 438 s per checkpoint (A-P3, the heaviest) → ≤ 3.4 h on the CPU partition as an upper bound; on one GPU the
forward / backward passes dominate and are expected to take a small fraction of that (GPU smoke 1019956 gives the number).  Job 2 = 4 runs × 3–4
checkpoints.  Memory: < 48 GB host; the all-view 1 024 × 1 024 score matrix is the largest tensor.

## Decisions at the freeze (main session)
1. Job 1 (eight existing std / strong runs) runs now on the CPU partition (measured 438 s per checkpoint, ≤ 3.4 h for 28 checkpoints): the 8-GPU quota
   is saturated by SSL runs.  The queued GPU smoke 1019956 was cancelled (the CPU path is measured; the GPU-only fix — CUDA init before
   reset_peak_memory_stats — is exercised whenever this runs on a GPU).
2. Job 2 (the four P115 crop-only / jitter-only runs) is registered with the orchestrator once the P115 run ids are frozen, condition: all four
   COMPLETED.  Same code, same base images, same diagnostic RNG.
3. Reading stays descriptive (no pre-written mechanism).
