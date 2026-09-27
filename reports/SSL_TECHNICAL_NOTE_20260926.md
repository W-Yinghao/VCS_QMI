# VCS-QMI self-supervised learning on CIFAR-10 — technical note (implementation, recipe, evidence)

Version 2026-09-27 06:40 UTC (all SSL runs finished; no further submissions per the owner) (commit trail in this repository; every number below is traceable to a run directory, a results table and a
frozen pre-registration).  Purpose: a complete, defensible reference for the SSL part of the paper.  Sections marked **[pending]** are
filled when the corresponding runs land (ceiling runs: 2026-09-27 morning UTC; control tuning P41: after the owner's go).

---

## 1. What is fixed and what was searched

**Fixed by the collaborator's plan** (`Variational_CS_QMI_Research_Plan (1).pdf`, sha256 0f87809e…884da; see `PLAN_CONSTRAINTS_20260925.md`):
the objective J, the mixture reference M = (P+Q)/2, negatives as product-of-marginals samples, the tanh-bounded critic output, no log transform
and no clipping of scores.  Also fixed by the owner: the ResNet-18 trunk with two-view (multi-view) augmentation, single-GPU FP32 training.

**Searched** (everything else; ~150 runs, waves P5–P43): critic functional form, negative-sampling count and gradient routing, initial critic
scale, number of views, batch size, schedule length, learning rate and schedule shape, weight decay, augmentation strength, projector
width/depth/kind, target branches (EMA / stop-gradient / predictor), extra critic steps, symmetric pairing.  Rule for every wave: frozen
pre-registration before launch, one factor per unit, single seed for exploration, three seeds for the base/final recipe, results table
committed before interpretation.

---

## 2. Method as implemented

### 2.1 Objective (from the plan; `reference/ssl_core.py::vcs_from_scores`)
For a critic score T ∈ (−1, 1) on pairs, with P the joint (two views of the same image) and Q the product of marginals (views of different
images),

    J(T) = E_P[T] − E_Q[T] − ½ E_P[T²] − ½ E_Q[T²],        loss = −J.

Each expectation is the mean over its own sample set (positives and negatives are averaged separately even when their counts differ).  The
population optimum is T* = tanh(PMI/2) ⇒ J* is the CS-QMI-type quantity of the plan; the diagnostics also log R_binary = 1 − J (the
squared-loss risk with targets ±1), the mean and second moment of positive / negative scores, and the fraction of |T| > 0.95 ("saturated").

### 2.2 Critic (searched → fixed)
    T(z₁, z₂) = tanh( a · ⟨z₁, z₂⟩ + b ),      z = p / ‖p‖₂,   a, b learnable scalars,   a₀ = 5 (optional; 1 also works at ≥ 2× compute), b₀ = 0.
z is the L2-normalised output of the projector (§3).  `src/vcs_ssl/models/critic.py::CosineCritic`.  Two parameters; the concat-MLP critic of
the original recipe (MLP([z₁, z₂]) → ℝ, 512-512 hidden, tanh) is what the plan's generic "critic network" would suggest and is the first
thing the search replaced (§5.2).

### 2.3 Negatives (from the plan; `cyclic_negative_indices`)
For a batch of B images with views (z₁ᵢ, z₂ᵢ): positives are (z₁ᵢ, z₂ᵢ); negatives are (z₁ᵢ, z₂_{π_k(i)}) for K distinct non-zero cyclic
shifts π_k drawn per step from a dedicated CPU generator (K = 8 ⇒ 8 negatives per image per pair of views; B·K negatives per step).  This
is the plan's product-of-marginals construction from off-diagonal pairs.  K ∈ {1, 8, 64, 255 (= all pairs, also as a vectorised matrix
sampler)} were tested; K = 8 is the recipe (§5.2).

### 2.4 Negative detach (searched → fixed; `vcs_pair_loss_negdetach`)
In the negative pairs the shifted partner is detached: T(z₁ᵢ, stopgrad(z₂_{π(i)})).  Positives are unchanged.  The objective value is
identical; only the gradient path through the *second* view of negative pairs into the encoder is removed.  Effect: +2.4 linear on three
seeds with the cosine critic; with MLP-type critics the same change collapses training (P20, P27), so it is a property of the
similarity-type critic, not a general trick.  Mechanism evidence in §6.

### 2.5 Multi-view averaging (searched → fixed; `compute_objective_views`)
With n views per image (n = 4 in the recipe; n = 8 under study), one concatenated forward of all n·B images (BN statistics shared), then the
same J is computed for every unordered view pair (a, b), a < b (n(n−1)/2 = 6 pairs for n = 4), each pair with its own K shifts, and the
losses are averaged.  This is not a new loss: it is more Monte-Carlo coverage of the same P and Q (positives per image per step: 1 → 6).
A step costs n/2 encoder forwards per image relative to two views.

### 2.6 One training step (recipe)
    x_1..x_n ← n augmentations of the B images;  h = f_θ(x) (ResNet-18);  p = g_φ(h) (projector);  z = p/‖p‖
    loss = mean over pairs (a<b) of  −J( T(z_a, z_b) , T(z_a, stopgrad(z_b)[π_k]) , k = 1..K )
    AdamW step on θ, φ and (a, b) jointly (critic lr multiplier 1, no critic weight decay).
No EMA, no predictor, no stop-gradient branch, no extra critic steps (all tested, all neutral or worse; §5.4).

---

## 3. Architecture, data and optimisation recipe

| component | value (final recipe) | supporting ablation |
|---|---|---|
| encoder | torchvision ResNet-18, CIFAR stem (3×3 conv, stride 1, no max-pool), h ∈ ℝ⁵¹² after global pooling | fixed by owner |
| projector | Linear(512→512, no bias) – BN – ReLU – Linear(512→128, bias) | output 64–512, hidden 512–2048, depth 1–3, BN-only, output-BN: all neutral or worse (P13/P25/P29/P40) |
| critic input | z = L2(p) (128-d) | critic on L2(h): −5.6 (P25) |
| critic | tanh(a⟨z₁,z₂⟩+b), a₀ = 5, b₀ = 0 | §5.2 |
| negatives | K = 8 non-zero cyclic shifts per view pair, shifted partner detached | §5.2 |
| views | 4 per image (6 pairs) | §5.3; 8 views under study (§5.5) |
| augmentation | RandomResizedCrop(32, scale [0.2, 1], ratio [3/4, 4/3], bilinear, antialias) → HFlip 0.5 → ColorJitter(0.4, 0.4, 0.4, 0.1) p 0.8 → Grayscale p 0.2; no blur, no solarize; normalise with CIFAR mean/std | weak −7, stronger −0.8…−1.5 at 200 ep on the old recipe (P17); at 800 ep with 4 views, crop 0.08 + jitter 0.8 gives 87.78 vs 87.01 ± 0.53 (single seed, +0.8; heldout-J 0.877 vs 0.974, threshold 0.95) — stronger augmentation pays only on long schedules |
| batch | 256 images (B = 128 for the 1×-compute recipe) | P17, P40 |
| optimiser | AdamW, lr 1e-3, betas (0.9, 0.999), eps 1e-8, wd 1e-4 on matrix weights (0 on bias/norm), critic lr ×1, critic wd 0 | lr 3e-4…1e-2, wd 1e-5…5e-4, critic lr ×0.1…×10 (P13/P15/P17/P40) |
| schedule | linear warm-up 10 epochs → cosine to 1 % of peak, per step | schedule shape neutral (P15); warm-up 5 hurts on short runs (P40) |
| epochs | 200 (2×), 100 with B 128 (1×), 400 / 800 / 1600 (ceiling) | §5.1, §5.5 |
| precision | FP32, TF32 off, cudnn.benchmark off, no compile, one GPU | fixed |
| data | CIFAR-10 official train (50 000): stratified split, seed 20260924, 500/class → **selection** (5 000), rest → **fit** (45 000); SSL never sees labels; official test untouched so far | fixed |

Initialisation: encoder and projector from `torch.manual_seed(seed)` (bit-identical across methods for a given seed); critic from a separate
stream.  Determinism: dedicated generators for data order, augmentation workers and negative shifts; checkpoints carry all RNG states;
CPU resume is bit-exact (test_7); GPU resume is exact at the epoch boundary up to cuDNN nondeterminism.

---

## 4. Evaluation protocol (identical for every method and run)

- **Linear probe (primary):** frozen encoder, features h (512-d, before the projector) of clean images (ToTensor + normalise only); a
  linear layer with bias trained on the 45 k *fit* features with SGD (lr 0.1, momentum 0.9, wd 0, batch 256, 100 epochs, cosine to 0.1 %,
  seed 20260925), reported at the last probe epoch on the 5 k *selection* features.  No feature normalisation (a standardised-feature probe
  was checked on 60+ runs: ±0.5, Spearman 0.95; `PROBE_STANDARDIZED.md`).
- **kNN:** k = 200, temperature 0.1, cosine on L2-normalised h, fit → selection.
- **Effective rank:** exp(entropy of the normalised covariance eigen-spectrum) of h (and of p, z) on the first 4 096 selection UIDs.
- **Held-out J:** the training objective evaluated with the training wiring (same critic, K, detach, views) on two fresh views of the
  selection images, 4 repeats.
- In-training monitors at fixed epochs (kNN / rank / J on a fresh model copy under forked RNG; training RNG untouched, verified by
  fingerprint).  All numbers in this note are on the selection split of the official training set; **the official test set has not been
  used** and will be evaluated once for the frozen recipes.

---

## 5. Results

### 5.1 Final recipe by compute budget (1× = 2 views × 200 epochs = 8.96 M image-forwards)
| compute | recipe | linear-val (%) | kNN (%) | h eff-rank | seeds | table |
|---|---|---|---|---|---|---|
| 1× | 4 views, B 128, 100 ep | **83.19 ± 0.40** | 78.43 ± 0.11 | 56 | 3 | P40 |
| 2× | 4 views, B 256, 200 ep | **84.54 ± 0.16** | 81.40 ± 0.22 | 76 | 3 | P36 |
| 2× | same with a₀ = 1 | 84.41 ± 0.06 | 80.93 ± 0.16 | 58 | 3 | P29 |
| 4× | 4 views, 400 ep | 85.90 | 83.58 | 105 | 1 | P36 |
| 4× | 8 views, 200 ep | 85.95 ± 0.35 (86.28 / 85.58 / 85.98) | 83.47 ± 0.38 | 103 | 3 | P38 |
| 4× | 2 views, 800 ep | 85.30 ± 0.21 | 82.33 ± 0.25 | 86 | 3 | P36 |
| 8× | 4 views, 800 ep | **87.01 ± 0.53** (86.42 / 87.16 / 87.44) | 85.46 ± 0.12 | 134 | 3 | P36 |
| 8× | 8 views, 400 ep | 86.76 | 84.94 | 137 | 1 | P44 |
| 8× | 16 views, 200 ep | 86.74 | 84.48 | 126 | 1 | P44 |
| 8× | 4 views, 800 ep, B 128 | 86.70 | 84.54 | 127 | 1 | P44 |
| 8× | 4 views, 800 ep, strong aug (crop 0.08, jitter 0.8) | **87.78** | 85.58 | 107 | 1 | P44 |
| 16× | 4 views, 1600 ep | 87.50 | 85.84 | 157 | 1 | P44 |
| 16× | 8 views, 800 ep | 87.14 | 86.32 | 157 | 1 | P44 |

Reference points on the same split and protocol (P5, 3 seeds, 200 epochs, 2 views, **untuned** frozen recipes): SimCLR-matched 86.09 ± 0.38 /
kNN 83.98 / rank 90; VICReg-matched-128 85.46 ± 0.26 / 81.92 / 76.  Original VCS recipe (concat-MLP critic, K = 1): 74.34 ± 0.46 / 63.93 / 13.
**Comparison claims wait for the equal-budget control tuning (P41, draft pre-registration written, not launched).**

### 5.2 How the recipe was found — the factors that mattered (200 epochs unless stated)
| step | change | linear | Δ | kNN | rank | seeds | stage |
|---|---|---|---|---|---|---|---|
| 0 | original: concat-MLP critic, K = 1 | 74.34 ± 0.46 | — | 63.93 | 13 | 3 | P5 |
| 1 | K = 8 | 76.74 ± 0.78 | +2.4 | 67.09 | 16 | 3 | P10 |
| 2 | cosine critic | 78.21 ± 0.13 | +1.5 | 73.09 | 58 | 3 | P18/P30 |
| 3 | negative detach | 80.59 ± 0.12 | +2.4 | 74.41 | 30 | 3 | P24/P26 |
| 4 | a₀ = 5 | 81.56 ± 0.44 | +1.0 | 76.94 | 46 | 3 | P26/P31 |
| 5 | 4 views (2× compute) | 84.54 ± 0.16 | +3.0 | 81.40 | 76 | 3 | P28/P35 |
| 5' | 4 views at 1× compute (B 128, 100 ep) | 83.19 ± 0.40 | +1.6 | 78.43 | 56 | 3 | P39 |
| 6 | 800 epochs (2 views) | 85.30 ± 0.21 | +3.7 vs step 4 | 82.33 | 86 | 3 | P35 |
| 6' | 800 epochs (4 views) | 87.01 ± 0.53 | +2.5 vs step 5 | 85.46 | 134 | 3 | P35 |
Interactions: a₀ and views overlap (with 4 views a₀ = 1 gives 84.41); K > 8 stops helping once detach is on; the batch/steps effect needs the
views (2 views, B 128: +0.6; 4 views, B 128 at 1×: +1.6); detach is necessary with 4 views too (−2.2 without).

### 5.3 Views (P38 + addenda)
Fixed schedule (200 ep, B 256, 35 k steps): 2 → 4 → 8 → 16 views = 81.56 → 84.54 → 85.95 (3 seeds) → 86.74 (+3.0, +1.4, +0.8: flattening; 16 views cost 4× the step time of 4 views).  Fixed compute: at 1× 4 views/100 ep (81.88) ≈ 2 views
(81.56) > 8 views/50 ep (80.60); at 2× 4 views/200 ep (84.54) = 8 views/100 ep/B 128 (84.60) > 8 views/100 ep/B 256 (83.40); at 4× 8 views/
200 ep (85.95 ± 0.35) ≈ 4 views/400 ep (85.90) > 2 views/800 ep (85.30).  Reading: more pairs per image help as long as the update count is not
reduced to pay for them; from 4× on, views buy more than epochs.  8 views × 800 ep: 87.14 / kNN 86.32 (16×).

### 5.4 Ablations on the final recipe (single seed; P40 unless stated; comparator 83.19 ± 0.40 at 1× or 84.54 ± 0.16 at 2×)
| knob | tested | result |
|---|---|---|
| negative detach off | 4 v, 100 ep | 79.66 (−2.2) |
| K = 1 / K = 127 (all pairs) | 4 v, B 128, 100 ep | 81.14 (−0.7, kNN −2.1) / 82.86 (−0.3, kNN +0.5) |
| lr 2e-3 / 5e-4 | 2 v, 200 ep | 81.64 (+0.1) / 80.56 (−1.0) |
| wd 5e-5 / 5e-4 | 1× recipe | 82.96 / 82.54 (neutral) |
| projector out 512 / hidden 2048 | 1× recipe | 82.88 / 82.82 (neutral) |
| a₀ = 1 | 1× recipe / 2× recipe | 82.12 (−1.1) / 84.41 (−0.1) |
| warm-up 5 | 1× recipe | 82.00 (−1.2) |
| B 64 (1×) / B 128 at 200 ep (2×) | 4 v | 82.78 (−0.4 vs B 128) / 84.74 (+0.2 vs B 256) |
| 8 views at equal compute | see §5.3 | ties at 2×, wins at 4× |
Earlier waves (on the intermediate bases; all neutral or worse): critic capacity (width 128–2048, depth 1–3) and gain; critic lr ×0.1–×10;
critic weight decay; projector depth 3, BN-only projector (−2.8), output BN (−2.1), critic on h (−5.6); EMA 0.99/0.996, stop-gradient,
predictor (−0.4…−1.0); 2–5 extra critic steps; symmetric pairing; spline / diagonal-metric / shared-metric / interaction-only / bilinear
critics (neutral, neutral, collapse, collapse, neutral); bias calibration; LR 3e-4…1e-2; cosine floor 10 % and constant LR; batch 128–1024 with
2 views; weak / strong augmentation, blur; K = 64 / 255 without detach (+0.4 / +0.9), with detach (+0.2).  Full list: `ALL_RUNS.md`,
`SYNTHESIS_20260925.md` §2, P13–P40 reports.

### 5.5 Ceiling runs (owner: compute unconstrained) — final (P44; single seed each)
| compute | run | linear | kNN | h-rank | kNN curve (last points) |
|---|---|---|---|---|---|
| 8× | 4 v × 800 ep (3 seeds) | 87.01 ± 0.53 | 85.46 ± 0.12 | 134 | 85.2 → 85.5 (600 → 800) |
| 8× | 4 v × 800 ep, strong aug | **87.78** | 85.58 | 107 | 85.3 → 85.6 |
| 8× | 8 v × 400 ep | 86.76 | 84.94 | 137 | 85.1 → 84.9 (flat) |
| 8× | 16 v × 200 ep | 86.74 | 84.48 | 126 | 84.7 → 84.5 (flat) |
| 8× | 4 v × 800 ep, B 128 | 86.70 | 84.54 | 127 | 83.9 → 84.5 |
| 16× | 4 v × 1600 ep | 87.50 | 85.84 | 157 | see P44 |
| 16× | 8 v × 800 ep | 87.14 | **86.32** | 157 | see P44 |
Reading: at 8× compute every allocation (more epochs, more views, more updates) lands at 86.7–87.0; doubling again to 16× adds ≈ +0.3
(87.1–87.5).  The linear ceiling of the recipe on this split is therefore ≈ 87.5 (+0.5 per doubling and flattening); kNN keeps rising with
views and schedule (86.3 at 8 views × 800 ep).  The one lever with a gain at fixed compute is stronger augmentation on the long schedule
(+0.8, single seed).  The "best absolute" configuration for the paper is 4 views × 800 epochs (3 seeds, 87.01 ± 0.53), with strong
augmentation (87.78) and 4 views × 1600 epochs (87.50) as single-seed upper points; the owner stopped further SSL submissions on
2026-09-26, so these are not seeded.

---

## 6. Mechanism diagnostics (read-only; `GEOMETRY_DIAG*.md`, `PROBE_LAYERS*.md`, `SYNTHESIS_20260925.md` §3–4d)

1. **Critic form decides the encoder's geometry, not critic capacity.**  Every concat-MLP critic (any width/depth/lr/K) leaves h at effective
   rank 13–21 and the projector output at 7–11 principal directions; the last ResNet stage learns almost nothing (layer-3 → h: +0.5…+0.8 linear
   vs +4.7 for SimCLR).  Similarity-type critics restore both (rank 58, +3…+4 for the last stage).  Any *extra* freedom given to the critic
   (learnable shared metric, interaction MLP) is spent on collapsing the code (rank 1–3), not on enriching h.
2. **The objective is satisfied by separability, not spread.**  Learned critics converge to a threshold cos* = −b/a ≈ 0.8–0.9; negatives are
   pushed just below it and no further.  Under negative detach the positives stay unsaturated for the whole run (0 % with |T| > 0.95 vs 83 %
   without detach): detach turns "push negatives below a low threshold" into "pull positives above a high threshold".
3. **h-uniformity (Wang–Isola) orders every VCS run by linear accuracy and separates them from the controls** (−1.5…−1.8 for the 2-view
   recipes, −2.0 at 4 views, −2.3 at 800 epochs; SimCLR −2.8, VICReg −2.5).  z-uniformity does not: under detach z lives in a narrow cone
   (negatives at cosine 0.76 in the 800-ep run) while h spreads — where the spreading happens is the mechanism, not how much.
4. **Under the final recipe the last ResNet stage is fully productive** (+4.9 layer-3 → h, SimCLR +4.7) and the remaining ≈ 2-point deficit to
   the untuned SimCLR is uniform across depth (2.8 / 2.1 / 1.9 at layer-2 / layer-3 / h) instead of growing with depth (3.8 / 7.6 / 11.6 for the
   original recipe).  The projector still keeps only ≈ 20–30 effective dimensions (SimCLR 77); giving it more pairs (views) or more updates
   widens what reaches h.
5. **Initial critic scale acts through the first ≈ 50 epochs only** (all a₀ ≤ 10 converge to a ≈ 10, b ≈ −8); a₀ = 20 starts saturated and
   never recovers; the effect vanishes once 4 views supply the early signal.
6. Statements the data do *not* support: that saturation fraction alone limits learning (harder augmentation lowers J without helping); that
   LR/schedule/batch are bottlenecks; that the probe protocol explains the gap (standardised probe ±0.5).

---

## 7. Compute and engineering

- Per-epoch time (B 256): 2 views ≈ 31 s (A100), 4 views ≈ 65–68 s (A100) / 33 s (H100), 8 views ≈ 120–140 s (A100, node-dependent) / 70 s (H100),
  16 views ≈ 257 s (A100).  Peak GPU memory: 3.7 / 7.6 / 14.9 / 29.6 GB for 2 / 4 / 8 / 16 views.
- A 200-epoch 2-view run ≈ 1.7 h; 4-view 800-ep ≈ 15 h (A100); runs longer than the 23 h walltime are submitted as dependent chains
  (`slurm/submit_chain.sh`, resume from `last.pt` at an epoch boundary; verified end to end), on H100/RTX6000PRO first.
- All compute through SLURM (8-GPU quota, 30-job cap, babysitter feeder); CPU test gate before every run on new code; watcher status board;
  results tables committed before interpretation; job ids in `reports/job_ids.json`; config sha256 manifests per wave (`configs/HPARAM_*_SHA256.json`).
- Code: `src/vcs_ssl/` (config schema + policy checks, data/splits/transforms, models, objectives, trainer with resume, evaluation,
  summaries), `reference/ssl_core.py` (plan objective and pairing, imported verbatim), 35 integration + 29 reference tests.

---

## 8. Caveats and open items

- All numbers are on the 5 k selection split of the official training set, chosen repeatedly during the search; the official test set is
  evaluated once at the end for the frozen recipes.  Single-seed cells are marked; the recipe's 1× and 2× numbers and the 2-view 800-ep number
  are 3-seed means.
- Controls are the frozen P5 recipes (2 views, 200 epochs, untuned).  Equal-budget tuning of SimCLR / VICReg (temperature / weights, 4 views,
  batch, 800 epochs; code ready, `run.control_tuning`) is pre-registered as a draft (P41) and **not launched**; no "beats SimCLR" claim is made
  before it.  With 4 views the VCS runs use 2× the encoder compute of the 2-view controls at equal epochs (stated wherever compared).
- Mechanism statements are observational (diagnostics on frozen checkpoints) except where an intervention exists (detach on/off, critic form,
  views, a₀).
- Theory-side questions raised by the experiments (for the paper's discussion, not implemented): the fixed 1 : 1 P/Q weight implied by
  M = (P+Q)/2 (negatives never dominate the gradient however large K is); the degenerate optimum in the high-dependence regime (T* ≈ ±1 for
  augmentation pairs, so the encoder geometry is chosen by the critic family and the dynamics, not by the objective).

---

## 9. File index
Pre-registrations: `reports/P*_PREREG_FROZEN_*.md` (+ addenda).  Results tables (observed values only): `reports/P{6,9,11,13,15,17,19,21,23,25,27,29,30,32,34,36,38,40,44}_*results_table.md`.
Reports: `P{4,7,9,11,13,15,17,19,25,27,29,30,32,34,36,38,40}_*REPORT*.md`.  Syntheses: `SYNTHESIS_20260925.md` (§1–7), `PLAN_CONSTRAINTS_20260925.md`.
Diagnostics: `GEOMETRY_DIAG.md`, `GEOMETRY_DIAG_v2.md`, `PROBE_LAYERS.md`, `PROBE_LAYERS_v2.md`, `PROBE_STANDARDIZED.md`, `ALL_RUNS.md`.
Configs: `configs/cifar10_hp{A..O}_*.yaml` with `configs/HPARAM_*_SHA256.json`; final recipe configs `cifar10_hpK_a5_views4_vcs_seed{0,1,2}.yaml`
(2×), `cifar10_hpM_a5_views4_b128_100ep_vcs_seed{0,1,2}.yaml` (1×), `cifar10_hpK_a5_views4_800ep_vcs_seed{0,1,2}.yaml` (8×).
Run directories: `/home/infres/yinwang/CS_QMI/outputs/<run_id>/` (config.resolved.yaml, manifest, checkpoints, logs, evaluations).
