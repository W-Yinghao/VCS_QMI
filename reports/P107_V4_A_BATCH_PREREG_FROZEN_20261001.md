# Pre-registration — P107: package v4 module A next batch (scorer × loss, pairing × gradient around P104 G2), five full 800-epoch CIFAR-10 seed-0 units, 2026-10-01 — FROZEN 2026-10-01T18:46:22Z before 800-epoch compute (CPU gate 1017240: all P107 checks green; launch gated on the GPU smoke 1017269 by scripts/p107_smoke_ok.py)

Status: FROZEN 2026-10-01T18:46:22Z (main session; owner shared the v4 package 2026-10-01; standing go "除了imagenet的先不提交，后续都可以提交").  Source:
`VCS_Next_Experiments_v4_Package.zip` → `VCS_Next_Experiment_Plan_v4_CN.md` §A1–A2, §4 and `experiments_v4_logical.yaml` (basis commit 4c86d83).
Owner: v4 package supplied 2026-10-01; standing instruction "除了imagenet的先不提交，后续都可以提交".

## Question
P104's first batch put the fixed angular scorer G2 (f = 2s − 1, T = tanh f, s = ⟨z1, z2⟩ of the L2-normalised projector outputs) first
(88.66 linear / 86.96 kNN, seed 0).  Two questions are left open by that batch, and this unit answers them with five direct cells:
1. **Scorer × loss (v4 §A1).**  Is the gain a property of the scorer structure (then it should also appear for the matched JS / logistic loss), or of
   its interaction with the squared VCS objective?  The 2 × 2 is {fixed f = 2s − 1, f = as + b learned from (2, −1)} × {VCS −J, matched JS}.
2. **Pairing × negative gradient at G2's scale (v4 §A2).**  P104's G3 / U2 used (a, b) = (1, 0); G4's failure (learned (5, 0) + full negative gradient)
   does not say what full negative gradients do at the fixed G2 scale.  The 2 × 2 is {K = 8 cross-view cyclic shifts, all view tokens} ×
   {negative right detach, plain autodiff through the negatives}.
No interpretation is written in advance (v4 §A1: "不要将任何一种结果提前写成结论").

## Cells (stage `P107_v4_A_batch`; `configs/make_p107_configs.py --write`; unit file `slurm/p107_units.txt`)
All cells are P104's G2 (or G2F) YAML — the frozen 8× recipe `cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml` plus the G2 / G2F critic fields, imported
from `make_p104_configs.apply` — with only the listed fields changed.  4 views, B 256, 800 epochs (140 000 encoder updates), AdamW 1e-3, standard
augmentation, seed 0, checkpoints 20 / 100 / 200 / 400 / 600 / 800.

| unit | v4 id | scorer | loss | pairing | negatives | reused counterpart | config (sha256 prefix) |
|---|---|---|---|---|---|---|---|
| P107_AL1 | A-L1 | fixed (2, −1) | matched JS: mean_P softplus(−2f) + mean_Q softplus(2f) | K8 cross-view | right detach | G2 (VCS) | `cifar10_hpY_AL1_views4_800ep_seed0.yaml` (44684630…) |
| P107_AL2 | A-L2 | learned from (2, −1) | matched JS | K8 cross-view | right detach | G2F (VCS) | `cifar10_hpY_AL2_views4_800ep_seed0.yaml` (57bee96b…) |
| P107_AP1 | A-P1 | fixed (2, −1) | −J | K8 cross-view | full autodiff | G2 | `cifar10_hpY_AP1_views4_800ep_seed0.yaml` (6fbe9e6f…) |
| P107_AP2 | A-P2 | fixed (2, −1) | −J | all view tokens (chunk 256) | right detach | G2 | `cifar10_hpY_AP2_views4_800ep_seed0.yaml` (c916be8f…) |
| P107_AP3 | A-P3 | fixed (2, −1) | −J | all view tokens | full autodiff | G2 | `cifar10_hpY_AP3_views4_800ep_seed0.yaml` (63efcb70…) |

Each YAML differs from its P104 parent (G2 for AL1 / AP*, G2F for AL2) only in `objective.loss` (+ the explicit marker `objective.js_fixed_scorer: true`
for AL1), `pairing.negative_detach`, `pairing.pair_scope` / `all_view_chunk`, and `run.stage` (verified by diff and by
`tests/test_vcs_ssl_p107.py::test_generated_configs_hashes_and_routing`).  All-view counts: n_P = V·B·(V−1), n_Q = V·B·V·(B−1), P and Q averaged
separately; positives always carry gradient on both sides; full negative gradient = ordinary backpropagation (no extra coefficient, no custom
gradient).  JS uses the same scorer, the same cyclic-shift index draws and the same detach as the VCS counterpart; T = tanh f, J is logged from the
detached scores.

## Code (this unit)
`src/vcs_ssl/config.py` only: (i) optional `objective.js_fixed_scorer` (not filled when absent → no existing config changes its resolved dict or
config_hash); (ii) the matched JS loss is allowed on the fixed scorer **only** with that explicit marker (P104's refusal of fixed + JS without it is
kept, as is every other P95 / P104 refusal: JS + all-view tokens, noise on the fixed scorer, marker on a learned scorer or without the JS loss).  No
change to `objectives.py`: the JS multi-view path reads `critic.scale` / `critic.bias` (buffers of `FixedCosineCritic`) and the all-view path already
routes full gradients through the keys when `negative_detach` is false.  Tests `tests/test_vcs_ssl_p107.py` (17).

## Reading (pre-stated; one seed per cell, descriptive screen)
- Screen metric: final-h linear-val at epoch 800, kNN adjacent; cost (step time, peak memory, pair evaluations) reported per cell.  No early stopping on
  J, rank or kNN (v4 §A2); only non-finite values, implementation errors or infrastructure failures stop a run (spec v3 §10.3 rule kept).
- References (same protocol, selection split): G2 88.66 / 86.96 (seed 0; seeds 1–2 and G2F seeds 0–2 from P104 addendum 1 / 3 when complete);
  U2 87.96 / 86.10; G3 87.64; G4 81.70; recipe 87.01 ± 0.53; tuned SimCLR 88.32 ± 0.30.
- A1 contrasts reported (seed-0 differences, no verdict on a single seed): AL1 − G2 (loss at fixed scorer), AL2 − G2F (loss at learned scorer),
  G2 − G2F and AL1 − AL2 (fixed vs learned within each loss).  A2 contrasts: AP1 − G2 (full gradient, K8), AP2 − G2 (pairing, detach), AP3 − AP2
  (full gradient, all-view), AP3 − G2 (both).  Additivity is not assumed.
- Selection: at most two cells (by final linear-val; < 0.1 → cheaper / simpler) proceed to seeds 1–2 together with their method controls (v4 §4:
  "新 SSL 提升候选最多选 1–2 个补 seeds 1–2，方法归因对照同样补齐"); positive final claims use pre-fixed new seeds 3–4; no re-selection from confirmation
  outcomes.  All results, including negative ones, are reported.
- Not claimed: anything about JS vs VCS beyond this matched configuration (v4 §A1: identical hyper-parameters are not a sufficient-tuning guarantee
  for either loss; a pre-specified small optimiser / learning-rate sensitivity control is required before any winner claim).

## Gate (CPU) and GPU smoke — filled in from jobs below
**CPU gate, job 1017240** (`slurm/p107_cpu_gate.sbatch`; `reports/P107_GATE_1017240/`; gate_rc = 1 because of one pre-existing test-data failure, see below):
- Suites: test_ssl_core 29 passed; test_integration 40 passed; test_vcs_ssl_v2 14 passed; test_vcs_ssl_p95 15 passed; test_vcs_ssl_p100 8 passed;
  **test_vcs_ssl_p107 17 passed**; package v3 `validate_core.py` 25 / 25.
- **test_vcs_ssl_p104: 29 passed, 1 failed — not caused by P107.**  `test_generated_configs_hashes_and_routing` iterates over
  `configs/P104_SHA256.json`, which since P104 addenda 1 / 3 also lists the confirmation controls G2F / U2F; the test's expected-critic table has no
  entry for them (`KeyError: 'G2F'`).  It fails identically at HEAD (the JSON is committed); the fix is two entries (`"G2F": "CosineCritic",
  "U2F": "CosineCritic"`) in `tests/test_vcs_ssl_p104.py`, outside this fork's file ownership — left to the main session.
- Frozen recipe first-step gradients (CPU) vs the P35 GPU run: max relative difference 8.5e-5.
- Config-hash invariance: 318 configs (317 loadable at HEAD); config_hash changed by the P107 code: **0**.
- Smokes (6 steps, pseudo-epochs of 3, 4 views, B 256, CPU): G2 parent and all five cells COMPLETED; stop-at-2 / resume-to-4 COMPLETED for all five.
  CPU step-time ratio vs G2: AL1 1.01, AL2 0.99, AP1 1.08, AP2 1.10, AP3 1.11.  J after 2 steps (same init): G2 0.134, AL1 0.132, AL2 0.132,
  AP1 0.276, AP2 0.132, AP3 0.272 (CPU smoke values; code-path evidence only).

**GPU smoke, job 1017269** (`slurm/p107_gpu_smoke.sbatch`, VARIANTS="AL1 AL2 AP1 AP2 AP3", RTX6000PRO / H100, normal QOS; 100 real-CIFAR steps per
cell plus G2 as reference, stop-at-50 / resume-to-100 per cell): **submitted, pending (all 8 GPUs held by the P104 confirmation runs) — results in
`reports/P107_GATE_SMOKE_1017269/summary.json`; must be read (finite gradients, step time, peak memory, resume) before this prereg is frozen.**

## Cost
Measured 800-epoch step times of the parent / sibling cells on RTX6000PRO (P104 summaries, steady state): G2 0.1129 s/step, G3 (fixed scale,
full negative routing) 0.1138, U2 (fixed scale, all view tokens) 0.1135, N2 (matched JS) 0.1153; peak allocated 5.27 GB / reserved 8.14 GB for all.
Each P107 cell is one of these code paths with a different scale or loss, so the projection is ≈ 0.113–0.116 s/step × 140 000 steps ≈ **4.4–4.5 h per
cell on RTX6000PRO** (CPU-gate ratios vs G2 ≤ 1.11 → ≤ 4.9 h); 5 cells ≈ 22–25 GPU-h; well under the 23 h wall → single run_unit jobs, no chaining.
To be confirmed by GPU smoke 1017269.

| cell | critic pair evaluations per step | projected 800 ep, RTX6000PRO |
|---|---|---|
| AL1, AL2, AP1 (K8 cross-view) | 13 824 | ≈ 4.4–4.5 h |
| AP2, AP3 (all view tokens) | 1 047 552 (76×) | ≈ 4.4–4.5 h (encoder-dominated, as U1 / U2) |

## Launch (after freezing)
`slurm/p107_lines.txt` — five single `run_unit.sbatch` jobs, RTX6000PRO,H100, normal QOS (never runfill), FINAL_CKPT epoch_800.pt.

## Delivery (v4 §5)
```yaml
experiment_family: full_ssl (screen)
protocol_id: P107_v4_A_batch
source_commit: <freeze commit>
estimator: vcs (J) | js_matched_logistic, scorer f = 2s-1 fixed or a s + b learned from (2, -1)
evaluation_readout: frozen-h linear + kNN, selection split, epoch 800; cost per cell
n_independent_units: 45000 fit images; 1 seed per cell (screen)
status: draft
```

## Decisions at the freeze (main session)
1. **Launch condition (mechanical, frozen now):** the orchestrator submits `slurm/p107_lines.txt` only when `scripts/p107_smoke_ok.py` exits 0 — GPU
   smoke 1017269 summary present; AL1, AL2, AP1, AP2, AP3 each COMPLETED 100 real-CIFAR steps with finite gradients; stop/resume COMPLETED with
   resumed-vs-uninterrupted |ΔJ| ≤ 0.05.  If it fails, nothing is launched and the failure is reported; no design change without a new addendum.
2. The CPU gate's only failing test (P104 hash/routing KeyError on G2F / U2F) was a stale expectation table in `tests/test_vcs_ssl_p104.py`
   after P104 addenda 1 / 3, fixed in the main session (test now passes); it is not a P107 effect.
3. The v4 §A1 optimiser / learning-rate sensitivity control for any JS-vs-VCS winner claim is deferred to its own unit, designed only if A-L1 or
   A-L2 is selected.  Selection after this screen follows v4 §4 (≤ 2 candidates; seeds 1–2 with method controls; fixed new seeds 3–4 later).
