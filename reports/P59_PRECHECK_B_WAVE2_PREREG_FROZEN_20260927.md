# Pre-registration DRAFT — row B, wave 2: B-S1 (closed form on cross-modal features), B-S2 (registration-energy variants), B-T1 (unlabelled embedding probe) — 2026-09-27 — FROZEN 2026-09-27T14:59:28Z before compute (CPU probes 1011089 identity, 1011097 B-S1, 1011100 B-T1 disclosed inside)

Status: DRAFT written by the implementation fork; the main session freezes it (renames to `*_FROZEN_*`) before any of the three units runs at
full size.  Designs follow `SECOND_APP_WAVE2_PLAN_20260927.md` (row B); deviations from the plan are marked **[deviation]**.  Code:
`scripts/precheck_b1_clip.py`, `scripts/precheck_b2_registration.py` (wave-2 flags; default path byte-identical to P52, see the identity check),
`scripts/task_b_layer_probe.py`; sbatch `slurm/precheck_b_wave2_{b1clip,b2var,layerprobe,b2identity}.sbatch`.  Nothing here touches the frozen items
(tanh output form, reference measure M = (P+Q)/2, product-of-marginals negatives); nothing is trained beyond the P50 adapters, linear solves and a scalar.

## B-S1 — closed-form linear-class critic vs the trained cosine critic on cross-modal CLIP adapter features
**Question.** P48 showed on CIFAR two-view features that a closed-form linear-class critic (with the frozen tanh output and one fitted scalar)
recovers the trained critic's held-out J within 0.02.  Does the same hold on cross-modal features (image tower vs text tower, frozen CLIP + the
P50 setting-2 adapters), where the pair problem is in the mid-dependence regime (J ≈ 0.57 source / 0.16 target)?
**Data.** P49 features (`outputs/P49_precheck_A`, COCO train2017 no-animal → animal split, identity = image id, split hashes in `splits.json`),
topic pairing (setting 2, `topic_partners`, seed 20260927).  Adapters: the P50 setting-2 *selected* VCS configuration — lr 1e-3, **15 epochs**
(**[deviation]** the plan text said 40 epochs, which is the setting-1 selection; setting 2 selected 15 — `P50_precheck_A2.json`), seeds 0 / 1 / 2.
**Pairs.** Positives (u_i, v_j(i)) with j(i) the fixed topic partner's caption 4; K = 8 product samples per image from random other images'
caption 4 (generator fixed per split); the SAME pairs feed the neural critic tanh(a·cos + b) and every closed form.
**Closed form.** Classes P1 = u⊙v (512), P2 = [u⊙v, |u−v|] (1024), P3 = [u⊙v, |u−v|, u, v] (2048), each + intercept; w* = ½(A_M + λ·mean diag·I)⁻¹d,
λ ∈ {1e-6, 1e-4, 1e-2}; J(w) without tanh and tanh(c·w*ᵀφ); (λ, c) chosen on a validation part (20 % of the fit pairs) by the tanh-wrapped J.
*In-domain* (P47 mirror, primary): each eval split's pairs are halved by a fixed permutation; fit half (80/20 fit/val) → eval half; the neural J
is computed on the same eval half.  *Transfer* (reported, not read): closed form fitted on SRC-CAL pairs, evaluated on all SRC-EVAL / TGT-EVAL pairs.
**Pre-committed reading** (gap = neural J_eval − max over classes of the tanh-closed J_eval, mean over seeds, per split):
- **holds** if gap ≤ 0.02 on both SRC-EVAL and TGT-EVAL (negative gaps — the closed form above the trained critic — count as holding, as in P48);
- **holds conditionally** if ≤ 0.02 on one split only, or only for one class family (e.g. only P1, which contains the cosine pre-activation);
- **does not hold** if gap > 0.05 on both splits.  Ridge sensitivity is stated when the best and worst λ differ by > 0.01 (P47 rule).
Not claimed: anything about calibration or tasks (that is A-S1 / A-T); the transfer rows are descriptive.
**Probe (disclosed).** CPU smoke, job 1011097 (2 000 images per split, 1 epoch, seed 0): in-domain gaps −0.061 (SRC-EVAL, neural 0.321 vs P1-tanh
0.383) and −0.031 (TGT-EVAL, 0.090 vs 0.121); P3 over-fits at this size (2 049-d on ≈ 1 600 positives).  Barely-trained adapters — not evidence.
**Compute.** CPU partition, ≈ 3 × 15 epochs of a 512×512 adapter pair on 20 000 images + linear solves: minutes to tens of minutes.

## B-S2 — registration-energy variants (is the roughness of J* intrinsic or a feature-class artefact?)
**Question.** P52 found the closed-form J* surface rougher than histogram MI/NMI (10.2 vs 6.5 local maxima; basin 20/15 vs 24/21; success at
R = 20/30 0.77/0.54 vs 0.88/0.70).  Each variant changes ONE thing relative to the P52 protocol (same 60 val2017 images, same seed, same pixel
samples and overlap mask; the P52 numbers are the baseline, not re-run):
| variant | flag | what changes | applies to |
|---|---|---|---|
| Fourier 121 | `--n-fourier 121` | k, l ≤ 5 per side (11 × 11 = 121-d) instead of ≤ 3 (49-d) | J* only |
| patch features | `--patch-features` | J* features add the 8×8 block-mean intensity channel per side ([1, cos/sin 2πk·a (k ≤ 3), cos/sin 2πk·ā (k ≤ 2)], 11 per side → 121-d, same dimension as Fourier 121) | J* only |
| smoothing σ = 2, σ = 4 | `--smooth 2`, `--smooth 4` | Gaussian pre-smoothing of both images before sampling, identically for J*, MI, NMI | all measures |
| coarse-to-fine | `--coarse-to-fine` | Nelder–Mead at 64 px (≤ 50 evaluations) → 128 px (≤ 50) → 256 px (≤ 50), same total budget 150; surfaces unchanged | optimiser, all measures |
**Pre-committed reading.** B2 is **re-opened** only if some single variant brings J* to *both* (i) mean local maxima ≤ that of NMI in the same
run and (ii) success rates at R = 20 and R = 30 within 0.05 of NMI's in the same run.  If a variant that changes all measures (smoothing,
coarse-to-fine) improves all of them alike and the gap stays, the roughness is called **intrinsic to the energy**; if only the J*-only variants
(121 / patch) close the gap, the P52 verdict becomes **holds conditionally** (feature class matters and a richer class suffices), with the caveat
that the class was chosen after seeing P52 — this is tuning the energy, disclosed as such.  Otherwise the P52 verdict stands.
**Byte-identity.** The refactored script's default path is checked against the committed P52 script on the CPU smoke (job 1011089,
`slurm/precheck_b_wave2_b2identity.sbatch`): **IDENTICAL** — per-image metrics, optimisation records and aggregates of the old and new scripts agree exactly (log `slurm_logs/b2identity_1011089.out`).
**Compute.** 5 GPU runs (121, patch, σ = 2, σ = 4, coarse-to-fine) × ≈ 1.7 h on A100 (P52 took 1 h 42 min) ≈ 8.5 GPU-h; runnable in parallel.
**[deviation]** the plan estimated ≈ 2 GPU-h for B-S2; the P52 protocol at 60 images costs 1.7 GPU-h per variant.

## B-T1 — task: the closed-form J* as an unlabelled embedding probe (layer correspondence vs CKA)
**Task.** Given two networks, identify for each stage of network A the corresponding stage of network B from representations alone (the
CKA sanity test): VCS seed 0 vs VCS seed 1 (`P35_vcs_a5_views4_800ep_seed{0,1}`, epoch 800) and VCS seed 0 vs SimCLR seed 0 (`P5_simclr_seed0`,
epoch 200).  Stages: stem (conv1-bn-relu, avg-pooled, 64), layer1 (64), layer2 (128), layer3 (256), h (avg-pooled layer4, 512), z (projector output,
L2, 128) — **[deviation]** the plan listed "layer1–4 and h", but avg-pooled layer4 *is* h, so the sixth stage is z.
**Data.** The 5 000 selection images (clean transform), fixed permutation (seed 20260927): 2 500 fit (1 875 fit-proper / 625 val) / 2 500 eval.
**Measure under test.** Per side PCA to p = 64 whitened components (fitted on fit-proper; the whitening is part of the measure), bilinear class
φ(u, v) = vec(u vᵀ) + intercept (4 097-d); positives = same image, product samples = K = 8 cyclic shifts of v; w* = ½(A_M + λ·mean diag·I)⁻¹d with
λ ∈ {1e-4, 1e-2, 1} and the tanh scalar c chosen on val by the tanh-wrapped J; reported J = tanh-wrapped closed form on eval (raw J* kept).
**Competitors.** Linear CKA and RBF-kernel CKA (median heuristic) on the raw stage features of the same 2 500 eval images.
**Outputs.** Two 6 × 6 matrices per measure; identification = number of stages of A whose row-wise argmax over B's stages is the same stage
(column-wise also reported); invariance checks on the (h, h) pair of seed 0 vs seed 1 with B's raw features transformed: random orthogonal
rotation, isotropic scaling × 10, per-feature scaling (log-uniform in [0.1, 10]); Δ of every measure.
**Pre-committed reading.**
- **holds** if, on both network pairs, the J probe's row-wise identification count ≥ the better of the two CKAs' counts, and |Δ| < 10⁻³ under the
  orthogonal and isotropic transforms;
- **holds conditionally** if it is one stage short of the better CKA on one or both pairs (or ties one pair and loses one stage on the other), with
  the invariances holding;
- **does not hold** if two or more stages short on either pair, or an invariance fails (|Δ| ≥ 10⁻³).
Per-feature scaling is reported, not read (neither the J probe nor linear CKA is invariant to it by construction).  Not claimed: anything about
downstream quality; one image split; three checkpoints.
**Probe (disclosed before freezing).** CPU smoke, job 1011100 (600 images: 225/75/300; PCA-16; K = 4): identification vcs_tanh 5/5 vs CKA 6/6 (seed 0
vs seed 1) and 4/4 vs 5/5 (vs SimCLR); the miss is the h row, whose argmax is B's z (0.772 vs 0.666 at h).  Invariance: orthogonal |Δ| ≤ 4.5e-8,
isotropic ≤ 6.3e-10 (CKA-RBF ≤ 1.1e-6); per-feature scaling changes all three measures by 0.12–0.16.  Under the rule above this probe would read
"holds conditionally"; the full-size run (PCA-64, 2 500 eval images) decides.
**Compute.** CPU partition: feature extraction of 5 000 images × 3 networks + 76 closed-form problems of dimension 4 097 (≈ 20–40 min); GPU optional.

**Addendum 1 (frozen 2026-09-27T19:24:24Z, owner: "都补充上").**  B-S2 combination variant: `--patch-features --coarse-to-fine` together (the two variants
that each closed half of the gap in the report), same 60 images and seed.  Reading unchanged: the family is re-opened only if J* matches
MI/NMI on local maxima (≤ NMI's) *and* success at R = 20 / 30 within 0.05 — and even then the report states that parity, not advantage, is
what was shown.  One GPU job (≈ 1.7 h).
