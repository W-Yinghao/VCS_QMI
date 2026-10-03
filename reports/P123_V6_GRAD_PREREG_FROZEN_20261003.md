# Pre-registration — P123: v6 V6-GRAD — actual vs counterfactual-readonly scorer gradients on fixed checkpoints, CIFAR-10 and CIFAR-100, 2026-10-03 — FROZEN 2026-10-03T19:36:55Z before the full jobs (CPU gate 1020285, GPU smoke 1020287)

Status: FROZEN 2026-10-03T19:36:55Z (main session; owner 2026-10-03: organise and submit the v6 plan).  Source:
`VCS_Server_Tasks_and_Theory_v6_20261003.zip` → `VCS_Server_Tasks_v6_CN.md` §5 (and §2.3 for the gradient formulas).  Code: P123 section appended to
`src/vcs_diag/augdiag.py` (the frozen P115 functions are untouched — test), `scripts/p123_grad.py`, `slurm/p123_{unit,gate,gpu_smoke}.sbatch`,
`tests/test_p123.py`.  Read-only on checkpoints; no training, no parameter update; official test set never read; labels never filter or weight anything.

## Question
On the same checkpoint, images and pairs, how does the shape of the scorer f organise the instantaneous update — which similarity ranges and which pairs
(by true crop overlap) carry the gradient — and how does that differ between the checkpoint's own scorer (actual) and replaced scorers (counterfactual)?
This chooses which full-training interactions to test next; it does not redefine accuracy and has no success criterion (no "negatives orthogonal",
"uniform positive gradient" or "larger rank" target).

## Two tables, kept apart
- **actual** — each checkpoint's own loss, pairing, gradient routing and scale (VCS −J with its pair scope / negative detach and its critic (a, b);
  SimCLR NT-Xent).  For VCS the actual block is computed through the same code path as the counterfactual with f = the checkpoint's own affine scorer;
  it is checked against the P115 decomposition of the training loss in every batch (`check_vs_training_loss`).
- **counterfactual-readonly** — same checkpoint, images, pairs and routing; only f replaced, the loss is −J with T = tanh f; instantaneous derivatives
  only; **never a training result**.  Scorers: affine f = a(s − κ), a ∈ {1, 2, 3} × κ ∈ {0.25, 0.5, 0.75} (9 cells) and the v6 §7 curvatures
  f = a[(s − κ) + λ s(1 − s)], λ = ±0.25 at (a, κ) = (2, 0.5) — 11 read-only scorers, not 11 trainings.  SimCLR checkpoints have no VCS pairing: their
  counterfactual uses all view tokens with full gradient (A-P3's structure), recorded as `simclr_counterfactual_pairing`.

## Recorded per scorer and batch (positives and negatives separately)
- Distributions (n, mean, sd, quantiles) of s, f, T, c − T, f′(s) and |∂A_c/∂s| = |(c − T)(1 − T²) f′(s)|.
- Scalar peaks: affine closed form s± = κ ∓ log 2 / (2a) with its in-range flag; for every scorer the in-range argmax of |∂A_c/∂s| on [−1, 1] (grid)
  and its value (affine maximum 32a/27); the actual zero of f (numerical; ≠ κ once λ ≠ 0); data mass within |s − s_peak| ≤ 0.05 (fixed window).
- Three distinct quantities: the scalar total absolute action Σ w |∂A_c/∂s| (w = 1/n_P or 1/n_Q); vector norms of the P, Q and total gradients w.r.t.
  z, r (projector output), h, encoder parameters and projector parameters, with cos(P, Q); projection contributions.
- Positives grouped by true crop IoU (P115 crop boxes; bins [0, 0.1), [0.1, 0.25), [0.25, 0.5), [0.5, 1]): projection contribution
  ⟨g_b, g_P⟩ / ‖g_P‖² at the encoder-parameter and h levels (may be negative; sums to 1 — tested) with ‖g_b‖, the group's share of the scalar action,
  mean s and T.  Negatives stay the random population.
- Isolated semantic view (labels read only here, never used to filter or weight): fraction of same-class negatives, their share of the negative scalar
  action and their h-level projection contribution / norm.
- Geometry of h and z in eval mode on 2 000 clean selection images (kept separate from the train-mode BN gradient copies).

## Runs, checkpoints, batches
- CIFAR-10 (P115 job-1 set, seed 0): A-P3 std / strong (P107_AP3_views4_800ep_seed0, P107_AP3_augstrong_views4_800ep_seed0), G2 std / strong
  (P104_G2_views4_800ep_seed0, P111_G2_augstrong_views4_800ep_seed0), SimCLR std / strong (P41_simclr_views4_800ep_seed0,
  P89_simclr_views4_800ep_augstrong_seed0), recipe VCS std / strong (P35_vcs_a5_views4_800ep_seed0, P43_vcs_a5_views4_800ep_augstrong_seed0);
  epochs 100 / 400 / 800 (all saved).
- CIFAR-100: recipe VCS (P91_c100_vcs_a5_views4_800ep_seed0), A-P3 (P107_AP3_c100_views4_800ep_seed0), SimCLR (P91_c100_simclr_views4_800ep_seed0) at
  epochs 100 / 400 / 800.  **The two P91 runs keep only epoch 800** (intermediate checkpoints were removed by the outputs clean-up): their epoch-100 and
  epoch-400 requests are recorded as mapping to the true epoch 800 (`checkpoint_map`, duplicate flag, measured once) — never relabelled.  A-P3 C100 has
  all three.  29 distinct checkpoints in total.
- 4 independent batches per checkpoint (B = the run's batch, 256 images × 4 views): first 4 × 256 FIT uids of the P115 seeded permutation of each
  dataset's manifest; augmentation views from the P115 forked RNG (seed + 100 + batch); pair RNG 99 + batch.  CIFAR-10 batches 0–1 are the P115 job-1
  batches.  An 8-batch formal mechanism report, if wanted, is a separate addendum (`--grad-batches 8`, finished runs resumable).

## Pre-stated reading (descriptive; labels per v6 §11.2: observed result / identity verified by definition / hypothesis to test)
- Report per run × checkpoint × scorer the quantities above, mean over the 4 batches with the batch spread; actual and counterfactual in separate tables.
- Identities checked by definition (tests): counterfactual with the actual scorer = the training loss and gradients; ∂A_c/∂s formula = autograd for
  affine and curvature scorers; projection contributions sum to 1; curvature endpoints f(0) = −aκ, f(1) = a(1 − κ), a/4 ≤ f′ ≤ 7a/4.
- Observed results are stated per run and checkpoint (one seed each); any statement that a scorer shape "would" change training is a **hypothesis to
  test** in full training (v6 §7 CURVE), never inferred from read-only derivatives, J values or gradient norms.  CIFAR-10 vs CIFAR-100 differences are
  not reduced to a single similarity-threshold story.

## Gate and smoke (disclosed; numbers are not results)
- **CPU gate, job 1020285 (gate_rc 0):** `tests/test_p123.py` 16 / 16 and `tests/test_p115_diag.py` 15 / 15 pass (31).  The P115 functions are byte-identical to
  the frozen commit c84b504 and the P115 section is a verbatim prefix of the extended module (test).  Smoke (B 16, 1 batch, epoch 800, all 11 scorers)
  ran on A-P3 C10, SimCLR C100 and recipe VCS C10.  **P115 reproduction** (batch 0 at full size, epoch 800, actual only): encoder |g_P| / |g_Q| equal the
  stored P115 job-1 values to relative 1.2e-5 / 2.6e-6 (A-P3 C10) and 2.6e-5 / 3.6e-6 (SimCLR C10) — float32 accumulation across CPU nodes.
  The actual block's Lp / Lq equal the P115 training-loss decomposition to relative 0.0 in every smoke batch.
- **GPU smoke, job 1020287 (A100, exit 0):** full size (B 256 × 4 views, 1 batch, all 11 scorers + actual + eval geometry): 15.3 s per checkpoint (A-P3
  C10; batch 13.7 s) and 14.2 s (A-P3 C100); peak 6.3 GB.  CPU full-size timing (gate): 497 s per checkpoint-batch → GPU ≈ 33× faster → full run on GPU.

## Cost
≈ 13.5 s per batch on an A100 + ≈ 2 s geometry per checkpoint.  Job 1 (CIFAR-10): 8 runs × 3 checkpoints × 4 batches ≈ 24 × 56 s ≈ 25 min; job 2
(CIFAR-100): 5 distinct checkpoints ≈ 5 min; plus model loading.  ≈ 0.6 GPU-h in total, 2 bundle jobs (`slurm/p123_lines.txt`), normal QOS,
A100 / L40S / RTX6000PRO / H100, node60 allowed.  Output ≈ 1–2 MB per run JSON (`outputs/P123_grad_results/`).

## Decisions at the freeze (main session)
1. 4 batches per checkpoint (first-pass report, v6 §5.2); an 8-batch formal mechanism report is a later resumable addendum if the first pass motivates it.
2. P91 CIFAR-100 runs keep only epoch 800: their 100 / 400 requests are recorded as mapping to the true epoch 800 and measured once (never relabelled);
   A-P3 CIFAR-100 has 100 / 400 / 800.
3. Launch both GPU jobs of `slurm/p123_lines.txt` now.  The P115-section code of augdiag.py is unchanged (only appended), so P115 diagnostics job 2 is unaffected.
