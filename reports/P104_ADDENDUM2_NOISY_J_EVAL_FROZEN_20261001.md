# P104 addendum 2 — N1 vs P95 τ 0.3: clean and noisy held-out J with the same evaluation (spec §7.3) — FROZEN 2026-10-01T13:27:41Z before the evaluation jobs

Owner go 2026-10-01 ("其他两项也一起做了吧").  Decision 4 of the P104 freeze ("an evaluation-only addendum computes noisy-J with R_eval = 16 on the
P95 τ 0.3 checkpoints").  Evaluation only, no training, official test set untouched.

**What.**  `scripts/noisy_j_eval.py` calls `vcs_ssl.diagnostics.critic_holdout` exactly as the training evaluation does (selection images,
two-view train transform, 4 repeats, rng_seed 20260926, K = 8, all modules in eval mode, `torch.manual_seed(0)` model build), with
noise_tau = 0.3 from each run's config and **R_eval = 16** fresh noise draws per pair on a dedicated generator.  Checkpoints epochs 100 / 200 / 400 /
600 / 800 of `P95_noise_tau0.3_views4_800ep_seed{0,1,2}` (trained with R = 1) and of `P104_N1_views4_800ep_seed0` (trained with R = 4; run after its
training completes).  **Parity check:** on N1 the script must reproduce the noisy J that N1's own training evaluation logged at each epoch (same
code path and seeds); a mismatch beyond GPU run-to-run noise (|Δ| > 0.005) is reported and the comparison is not read.

**Reading (descriptive, as in §7.3; no threshold).**  Per epoch: clean J, noisy J, clean / noisy gate for N1 seed 0 next to the P95 τ 0.3 seeds 0–2
(mean ± sd), with the final linear / kNN accuracies alongside.  Statements are limited to whether averaging the loss over R = 4 draws changes the
noisy estimator value S_σ the representation reaches, and whether that tracks the accuracy difference (N1 seed 0 vs P95 τ 0.3 seed 0: same seed).
One N1 seed: no claim about N1 vs P95 accuracy beyond what P104's selection rule allows.
