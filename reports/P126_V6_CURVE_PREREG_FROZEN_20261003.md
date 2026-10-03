# Pre-registration — P126: v6 V6-CURVE — fixed anchored-quadratic scorer on A-P3 (CIFAR-10 / CIFAR-100, λ = ±0.25, seed 0), 2026-10-03 — FROZEN 2026-10-03T20:31:00Z before 800-epoch compute (CPU gate 1020302, GPU smoke 1020303)

Status: FROZEN 2026-10-03T20:31:00Z (main session; owner batch approval 2026-10-03: "同意进行item1").  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title).  **Start condition (v6 §7.1): unit checks
(this gate) + the stage-B review (V6-EVIDENCE / V6-GRAD first reports) + a separate batch approval** — the four 800-epoch units are NOT submitted by this
draft.  Source: `VCS_Server_Tasks_and_Theory_v6_20261003.zip` → `VCS_Server_Tasks_v6_CN.md` §7, §11.1; `VCS_Theory_to_Experiments_v6_CN.md` §7;
reference core `support/geometry_evidence_core.py`.  This is the v6 package's own new proposal (not run before; not derived as an optimal form).

## 1. Question
Holding the evidence scale at the two ends fixed and changing only the curvature of the score curve in between — is that better for representation
learning than the pure affine scorer of A-P3?  The objective J, the P / Q weights, the pairing (all-view tokens), the gradient routing (full), the encoder,
projector, augmentation and schedule are unchanged.

## 2. Scorer (new, separately named class `FixedCurvedCosineCritic`; FixedCosineCritic and MonoSplineCritic untouched)
    f_{a,κ,λ}(s) = a[(s − κ) + λ s(1 − s)] = a·s + b + a·λ·s(1 − s),   b = −aκ,   T = tanh f,   a, κ, λ fixed buffers for the whole run.
- f(0) = −aκ and f(1) = a(1 − κ) for every λ; f′(s) = a[1 + λ(1 − 2s)].  With a > 0 and |λ| ≤ 1/4: a/4 ≤ f′ ≤ 7a/4 on s ∈ [−1, 1] → strictly
  monotone (the config policy refuses |λ| > 1/4, a ≤ 0, an implicit κ, and curvature_lambda with any affine_mode other than `fixed_curved`).
- κ is the **affine anchor**, not the zero of f when λ ≠ 0.  The actual zero s0 is solved numerically (bisection, float64) and written to every run
  manifest (`hparams.p126_curve`): at (a, κ) = (2, 0.5), λ = +0.25 → s0 = (5 − √17)/2 = 0.4384; λ = −0.25 → s0 = (√17 − 3)/2 = 0.5616.
- logits(left, right), forward and score_matrix(C) all apply the same non-linearity; the all-view matrix path calls it (no re-hard-coded a·C + b);
  the matched-JS paths (K8 and all-view) read the same logits (never invert a saturated T).  λ = 0 evaluates exactly A-P3's a·s + b.
- No trainable affine: zero critic parameters, the optimizer has no critic group.

## 3. Cells (stage `P126_v6_curve`; `configs/make_p126_configs.py`; `configs/P126_SHA256.json`)
Each YAML = the A-P3 config of that dataset with ONLY `model.critic.affine_mode` fixed → fixed_curved, `model.critic.curvature_lambda` added, `run.stage`
(verified by diff; seed, data, views, optimizer, train, pairing, objective blocks identical).

| unit | dataset | (a, κ) | λ | λ = 0 baseline (existing run, same seed / config otherwise) |
|---|---|---|---|---|
| P126_CV_AP3_c10_lamm025_views4_800ep_seed0 | CIFAR-10 standard | (2, 0.5) | −0.25 | P107_AP3_views4_800ep_seed0: 89.06 linear / 87.30 kNN |
| P126_CV_AP3_c10_lamp025_views4_800ep_seed0 | CIFAR-10 standard | (2, 0.5) | +0.25 | same |
| P126_CV_AP3_c100_lamm025_views4_800ep_seed0 | CIFAR-100 standard | (2, 0.5) | −0.25 | P107_AP3_c100_views4_800ep_seed0: 60.20 linear / 55.92 kNN |
| P126_CV_AP3_c100_lamp025_views4_800ep_seed0 | CIFAR-100 standard | (2, 0.5) | +0.25 | same |

(a, κ) = (2, 0.5) is A-P3's existing value on both datasets (the mechanism control v6 §7.1 allows); no within-dataset tuning has been completed for the
all-view / full-gradient structure, so the same (a, κ) is used for both curvatures and the λ = 0 run is the full baseline.

## 4. Pre-stated reading (descriptive screen; one seed per cell)
- Primary: final frozen-h linear (selection split, epoch 800); kNN, training cost (step time, peak memory) reported alongside; the official test set is closed.
- Per dataset, λ = ±0.25 vs the λ = 0 baseline (same seed): differences reported; no verdict on one seed.
- **Candidate rule (per dataset, at most one):** a λ cell whose final linear exceeds the λ = 0 baseline by ≥ max(0.3, 2 × A-P3's seed sd on that dataset)
  — CIFAR-10: 2 × 0.04 → **0.30**; CIFAR-100: 2 × 0.25 → **0.50**.  If both λ qualify, the higher linear; a gap < 0.1 between them → the higher kNN.
  A candidate then gets seeds 1–2 (addendum), and only after that a matched-JS of the same shape.  No candidate → reported as "no curvature gain at
  this (a, κ)"; no further λ values are added after seeing the results.
- Trade-offs reported for a candidate only after its seeds 1–2 (evaluation-only, existing runners): strong-augmentation behaviour (P115 / P111 protocol),
  CIFAR-10 → CIFAR-100 frozen transfer (P119 readouts), V1 label efficiency (P110 runner).
- **No selection by read-only J, measured J or gradient norms; no early stopping on low early kNN / J / rank** (only non-finite values or infrastructure
  failures stop a run).  Not claimed: an optimal scorer shape; anything about other (a, κ), backbones or datasets.

## 5. Implementation checks (gate) — filled in from the jobs
- `tests/test_vcs_ssl_p126.py` (36 tests): endpoints f(0), f(1); f′ = autograd and within [a/4, 7a/4], strictly increasing; actual-zero solver vs the
  closed-form roots (and None without a sign change); invalid parameters refused; no trainable params / no optimizer critic group; score_matrix = pairwise
  logits; **λ = 0 bit-identical to A-P3** (FixedCosineCritic) on logits / forward / score_matrix, all-view VCS and JS losses and gradients (dense and chunked:
  chunk 1 / 5 / 24), K8 VCS and K8 matched-JS losses and gradients, and a 3-step trainer run (identical encoder / projector states); λ ≠ 0 all-view VCS / JS
  = brute-force curved reference and ≠ the affine one; K8 JS reads the curved logits; config policy refusals; generated-config diffs; trainer smoke +
  manifest record; stop / resume.  Local run: 36 / 36 pass.
- CPU gate (job 1020302) and GPU smoke (job 1020303): see §7.

## 6. Cost
Projected from the GPU smoke (step time vs A-P3 on the same GPU); A-P3 800 epochs ≈ 4.4 h on an unshared RTX6000PRO.  4 units ≈ 18 GPU-h.
Launch lines: `slurm/p126_lines.txt` (RTX6000PRO / H100, normal QOS, node60 allowed).

## 7. Gate and smoke results — filled in below

### CPU gate — job 1020302 (nodecpu03, exit 0, gate_rc 0; `reports/P126_GATE_1020302/`)
- Test suites: ssl_core 29, integration 40, v2 14, p95 15, p100 8, p104 30, p107 17, p114 15, **p126 36** — all passed; package v3 `validate_core.py` 25 / 25.
- Frozen recipe (hpK seed 0) first-step gradients vs the P35 GPU run: max relative difference 3.7e-5.
- **Config-hash invariance: 378 configs, 377 loadable at HEAD (the one exception is the pre-existing plan file `configs/next_stage_plan.yaml`);
  config_hash changed by the P126 code: 0.**
- 6-step smokes of the four P126 configs + the two A-P3 parents: all COMPLETED; CPU step-time ratio vs A-P3 1.00–1.03; stop / resume COMPLETED for all four;
  each run manifest carries `hparams.p126_curve` (a 2, b −1, κ anchor 0.5, λ, actual zero s0 0.5616 for λ = −0.25 and 0.4384 for λ = +0.25,
  f(0) = −1, f(1) = 1, slope range on [−1, 1]: λ = −0.25 → [0.5, 2.5], λ = +0.25 → [1.5, 3.5]).

### GPU smoke — job 1020303 (A100-40GB node56, exit 0, smoke_rc 0; `reports/P126_GATE_SMOKE_1020303/`)
100 real-CIFAR steps (B 256 × 4 views), CIFAR-10, seed 0, NONDET=1 (second uninterrupted run as the GPU run-to-run baseline):

| cell | s / step | ratio vs A-P3 | peak alloc / reserved MB | J at step 100 | finite grads | resumed vs uninterrupted \|ΔJ\| (step 99) | rerun \|ΔJ\| (step 99) |
|---|---|---|---|---|---|---|---|
| A-P3 (λ = 0 reference) | 0.3642 | 1.000 | 7569 / 12278 | 0.547 | yes | — | — |
| λ = −0.25 | 0.3673 | 1.009 | 7569 / 12294 | 0.533 | yes | 0.0005 | 0.0003 |
| λ = +0.25 | 0.3746 | 1.028 | 7569 / 12294 | 0.567 | yes | 0.0008 | 0.0005 |

Resume drift is the size of GPU run-to-run drift (non-deterministic kernels; CPU resume is exact, test_stop_resume_identical); cost counters identical after
resume; encoder / projector init hashes equal P35 seed 0 (the critic hash differs by construction: fixed buffers vs P35's learned (5, 0)).
**Cost:** step time ≈ A-P3 (+1–3 %), memory identical → 800 epochs ≈ 4.4–4.6 h on an unshared RTX6000PRO (14.2–14.6 h on this A100); 4 units ≈ 18 GPU-h.

## Decisions at the freeze (main session)
1. **Batch approval (v6 §7.1):** owner 2026-10-03 "同意进行item1" after the stage-B summary (P122: the fixed scorer is far from a calibrated posterior of s;
   P123: read-only, curvature ±0.25 changes the low-IoU positive-gradient share far less than κ — so its effect, if any, must act through another route;
   P124 granularity still running and not used for this decision).
2. Candidate rule as drafted: gain over the λ = 0 baseline (A-P3 seed 0) ≥ 0.30 linear on CIFAR-10, ≥ 0.50 on CIFAR-100; at most one candidate per dataset,
   which then gets seeds 1–2 and afterwards a matched-JS run of the same shape.  No selection by read-only J or gradient norm; no early stopping.
3. Launch the four seed-0 units now: normal QOS, RTX6000PRO / H100, node60 allowed.
