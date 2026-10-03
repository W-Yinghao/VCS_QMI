# Pre-registration — P122: v6 V6-EVIDENCE — scalar vs full-pair evidence on frozen encoders under a common measurement augmentation, 2026-10-03 — FROZEN 2026-10-03T19:31:46Z before the full jobs (CPU gate 1020286, GPU smoke 1020288)

Status: FROZEN 2026-10-03T19:31:46Z (main session; owner 2026-10-03: organise and submit the v6 plan).  The main session freezes it (rename to `*_FROZEN_*`, freeze time in the title) before the full GPU jobs.  Source: package
`VCS_Server_Tasks_and_Theory_v6_20261003.zip` → `VCS_Server_Tasks_v6_CN.md` §4 (V6-EVIDENCE), §11; theory appendix §2 (nested observation chain).
Code: `src/vcs_measure/evidence.py` (new), `scripts/p122_evidence.py`, `scripts/p122_aggregate.py`, `slurm/p122_{unit,gate,gpu_smoke}.sbatch`,
`slurm/p122_lines.txt`, `tests/test_p122.py`.  No encoder training; the official test partition is never opened (training-partition loaders only).

## Question
On the same frozen representation, how much additional verifiable score do richer pair functions recover beyond (i) the run's own training scorer and
(ii) the best function of the scalar s = ⟨z1, z2⟩ alone — first from the full projector output Z, then from the pre-projector representation H?  A gain
would point to a pair-expression bottleneck of the scalar similarity.  This is a measurement of "additional score recovered in these function classes
and budgets", **not** an identification of the three true terms of S_W − J(T) = L(W→Z) + L(Z→s) + L(s→T) (theory appendix §2: two finite fits differ
by (S_Z − S_s) − e_Z + e_s).

## Encoders (seed 0, epoch 800, eval mode, frozen BN, float32 raw h and z; COMPLETED runs only)
- CIFAR-10: recipe VCS `P35_vcs_a5_views4_800ep_seed0`; A-P3 `P107_AP3_views4_800ep_seed0`; SimCLR `P41_simclr_views4_800ep_seed0`; A-P3 strong
  `P107_AP3_augstrong_views4_800ep_seed0`; SimCLR strong `P89_simclr_views4_800ep_augstrong_seed0`.  (G2 `P104_G2_views4_800ep_seed0` is supported as
  an optional auxiliary but not in the launch lines.)
- CIFAR-100: recipe VCS `P91_c100_vcs_a5_views4_800ep_seed0`; A-P3 `P107_AP3_c100_views4_800ep_seed0`; SimCLR `P91_c100_simclr_views4_800ep_seed0`.

## Measurement augmentation (common to every encoder — separates "representation differs" from "evaluation pair distribution differs")
- CIFAR-10: **common-standard** (crop scale ≥ 0.2, jitter 0.4/0.4/0.4/0.1, p 0.8, grayscale 0.2, flip 0.5; the block of
  `configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml`) **and common-strong** (crop ≥ 0.08, jitter 0.8/0.8/0.8/0.2; the block of
  `configs/cifar10_hpO_a5_views4_800ep_augstrong_vcs_seed0.yaml`): every CIFAR-10 encoder is evaluated under both (cross-evaluation).
- CIFAR-100: common-standard (block of `configs/cifar100_hpS_vcs_a5_views4_800ep_seed0.yaml`).
- Normalisation constants are each encoder's own (all runs use mean = std = 0.5).  The augmented images are generated once per (dataset, law) from
  fixed per-image seeds (seed 122, tag = view, uid) under `fork_rng`, cached as uint8, and shared by every encoder.

## Pools and pairs (per dataset; manifest FIT uids only; seed 20261003)
Disjoint base-image pools FIT 8192 / TUNE 2048 / EVAL 8192 (all fit in the 45 000 FIT uids; no scaling needed), each half anchors, half independent
partners → FIT 4096 units, TUNE 1024, EVAL 4096.  Unit i: P_i = (view A1 of anchor i, view A2 of anchor i), Q_i = (view A1 of anchor i, view B of
partner i); P_i and Q_i share the anchor view; same augmentation law on both margins.  Evaluation intervals are bootstrapped over units (1 000 reps), never
over token counts.  All fitters share these explicit indices.

## Hierarchical measurement critics (f = pre-tanh logit, T = tanh f; losses on the same logits)
| level | candidates | parent (always a candidate) |
|---|---|---|
| train | the run's own training scorer, unchanged (VCS runs; SimCLR has none → "—") | — |
| s | affine tanh (L-BFGS from a ∈ {1, 5, 20}); 16- and 32-bin equal-frequency squared regression (per-bin (n_P − n_Q)/(n_P + n_Q), the in-bin optimum of both losses); 1-D MLP 32-32 | train scorer |
| Z | f_Z = f_s(s) + q_ψ(z1, z2), selected s branch frozen | the selected s critic ("no residual") |
| H | f_H = f_Z(z) + q_ω(h1, h2), selected Z model frozen; h standardised with FIT mean / sd (fixed map) | the selected Z critic |

q: MLP on [u, v, u⊙v, |u − v|], hidden 256 × 2, output layer zero-initialised (step 0 = parent), symmetrised ½(q(u, v) + q(v, u)).  Losses: **vcs** = −J;
**js** = mean_P softplus(−2f) + mean_Q softplus(2f) (logistic logit 2f, common posterior T = tanh f).  Budget per MLP candidate: 1 000 full-batch AdamW
updates (wd 1e-2), lr ∈ {1e-4, 5e-4, 2e-3} × 3 initialisations, early stopping on TUNE own risk every 10 steps (step 0 included).  Selection of each
level only by TUNE own-objective risk; EVAL is only read for the report.  VCS and JS get the same library, indices and budget.

## Readouts
Per (dataset, law, encoder, loss): EVAL J (common), squared risk (= 1 − J), native JS, per level; the selected candidate per level and every candidate's
TUNE risk / EVAL J / fit seconds; paired increments **Ĵ_s − Ĵ_train**, **Ĵ_Z − Ĵ_s**, **Ĵ_H − Ĵ_Z** (and Ĵ_H − Ĵ_s) with 95 % bootstrap intervals over
EVAL units (negative values kept); cost (feature extraction, fits, total, peak GPU memory).  Optional density-ratio consistency (global log E_Q e^{2f},
per-anchor log E_{Z2 ∼ P_Z2} e^{2f(z1, Z2)} over 512 anchors × 512 independent partner views, log-sum-exp, effective sample size) — a necessary-condition
diagnostic only (f = 0 also passes), never used to select anything.

## Pre-stated reading (descriptive; seed-0 encoders only in this unit)
1. **"Positive structural gain"** for an (encoder, law): the 95 % interval of Ĵ_Z − Ĵ_s, or of Ĵ_H − Ĵ_Z, lies above 0 **for both losses**.  Such
   encoders get encoder seeds 1–2 in a later addendum.  Otherwise: "no additional score recovered at this budget" — never "the similarity is sufficient"
   (a zero difference does not prove sufficiency, and a positive one does not imply better downstream accuracy).
2. Ĵ_s − Ĵ_train is reported as the gap between the training scorer and the best scalar critic of the same s (for fixed-scale runs this includes the
   fact that a fixed (a, κ) is not a calibrated posterior — v6 §2.2); n/a for SimCLR.
3. Cross-evaluation (CIFAR-10): each level's J under common-standard vs common-strong per encoder, side by side, descriptive.
4. Labels per v6 §11.2: **observed result / identity verified by definition / hypothesis to be tested**; a finite Ĵ is not a per-draw lower bound.
Not claimed: true values of the nested terms on real images; anything about other seeds or datasets; any causal link to linear / kNN accuracy.

## Gate and smoke (disclosed; numbers are not results)
- **CPU gate, job 1020286 (gate_rc 0):** `tests/test_p122.py` 8/8 (pools disjoint and deterministic; per-image augmentation seeds order-independent and
  RNG-isolated; squared risk = 1 − J; JS at f = 0; bins closed form; zero-initialised symmetric residual, nested step 0 = parent; parent always a
  candidate and the selected model never worse on TUNE; paired increment = J difference; f = 0 passes the density diagnostic), `tests/test_p109.py` 9/9;
  reduced-size smokes (512 / 256 / 512 base images, 60 steps, one lr / init) for CIFAR-10 (A-P3, SimCLR × standard, strong) and CIFAR-100 (recipe VCS).
- **GPU smoke / timing, job 1020288 (A100, smoke_rc 0):** one FULL-size (encoder, law) — A-P3 CIFAR-10, common-standard: augmentation of 18 432 base
  images × 3 views 15 s (once per dataset × law, cached); features 1.4 s; both losses 191 s in total; peak GPU memory 1.4 GB.  Readings at this size
  (not results): EVAL J train 0.903, s 0.975 (affine for VCS, 1-D MLP for JS), Z 0.975–0.976, H 0.976–0.977; Ĵ_s − Ĵ_train +0.072; Ĵ_Z − Ĵ_s +0.0003 (VCS,
  interval includes 0) / +0.0008 (JS); Ĵ_H − Ĵ_Z +0.0003 / +0.0005.  CIFAR-100 path ran at smoke size on GPU.
- **Observed headroom limitation (disclosed, no design change):** with an independent random partner as the product sample, P and Q are already
  separated to J ≈ 0.975 by the best scalar critic of s on CIFAR-10 A-P3, so structural increments are bounded by the remaining ≈ 0.025; small increments
  are expected and are reported with their intervals.
- The selected residual fits early-stop at 20–90 of 1 000 updates at lr 1e-4 (TUNE risk), recorded per candidate.

## Cost
≈ 3.2 GPU-min per (encoder, law) on an A100 (+ 15 s augmentation per dataset × law): job A (CIFAR-10, 5 encoders × 2 laws) ≈ 35 min, job B (CIFAR-100,
3 encoders × 1 law) ≈ 12 min; total ≈ 0.8 GPU-h.  Cache ≈ 1.2 GB under `outputs/P122_cache` (keys: dataset, pool hash, law signature, aug seed,
checkpoint sha256, float32, normalisation, code version).  Launch lines `slurm/p122_lines.txt` (2 GPU jobs; A100, L40S, RTX6000PRO, H100; normal QOS).

## Delivery (v6 §11.2 fields)
```yaml
logical_id: V6-EVIDENCE
protocol_id: P122
datasets: [cifar10, cifar100]
pretrain_dataset: same as eval dataset (frozen seed-0 encoders listed above)
measurement_augmentation: common-standard (C10, C100); common-strong (C10)
pair_definition: P = two views of an anchor; Q = anchor view vs independent partner view; unit = anchor
observed_objects: s, Z (projector output, L2), H (raw encoder output, FIT-standardised)
function_classes: train scorer | affine-tanh / bins16 / bins32 / 1-D MLP | nested residual MLP on Z | nested residual MLP on H
native_losses: vcs (-J), js (logit 2f)
common_score: J of T = tanh f
selection: TUNE own-objective risk; parent always a candidate
independent_units: EVAL 4096 anchors per (dataset, law)
status: draft
```

## Decisions at the freeze (main session)
1. Encoders as drafted (G2 auxiliary not added; v6 lists it as optional).
2. Seeds 1–2 follow-up rule kept: a structural increment (Ĵ_Z − Ĵ_s or Ĵ_H − Ĵ_Z) counts as positive only if its paired interval excludes 0 for both VCS and JS.
   Because the scalar critic already reaches J ≈ 0.975 (structural headroom ≈ 0.025), every increment is also reported as a share of the remaining
   headroom 1 − Ĵ_s; no effect-size threshold is added after the smoke.
3. Launch: the two GPU jobs of `slurm/p122_lines.txt` now (normal QOS, A100 / L40S / RTX6000PRO / H100).
