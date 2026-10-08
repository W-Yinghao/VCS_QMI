# VL1-10 / VL1-11 frozen protocol — Table A (same features) and prior-corrected ranking — FROZEN 2026-10-09

Design (owner 2026-10-08, "就按这个设计来"): the estimator / critic is the subject; matched JS is the same-posterior control, expected ≈ VCS.
Data / features / roles as audited in VL1-00 (`VL1_DATA_AUDIT.json`, `vl1_raw_clip_trainside.json`).

## Data and features
RefCOCOg (UMD), COCO-2014 images; roles by image from the UMD train split (seed 20261008): FIT 17 519 / CAL 2 190 / DEV 2 190 images; eligible = ≥ 2
referred objects (FIT 10 674, CAL 1 399, DEV 1 356).  FIT N ∈ {1 000, 4 000, all = 10 674} nested prefixes (no 16 k level exists; no duplication).
Official UMD val / test closed (val only for external reproduction checks and the final validation of frozen configurations).
Features: frozen open_clip ViT-B-16-quickgelu (OpenAI), tight crop → official preprocessing; fp16 cache (fp32 argmax agreement 99.8 %).

## Laws
Main: region-uniform p_G(r, w) = (1/m) A_rw / Σ_j A_rj; Q_G = product of the image's marginals (matching cells kept); images weighted equally.
VL1-11: phrase-uniform law p_G(r) ∝ number of expressions of r (same A), used for training; readout adds the prior-corrected ranking 2f + log p_G(r).

## Scorer class (identical for every learned row)
Primary F2r: f(u, v) = a·cos(u, v) + b + g([u, v, u⊙v]), a = softplus(α) > 0, (a, b) initialised at the A-P3 scorer (2, −1), g = MLP 3d→256→256→1
(GELU) with zero-initialised last layer → every route starts exactly at the raw-CLIP ranking (unit test).  Reason (gate 1029643): the package's
pure concat F2 starts from no similarity structure and sat at 45 % DEV Top-1 after the smoke budget vs raw CLIP 70 % on the same images.
Sensitivity row: the package's concat F2 for VCS and JS at N = all, seed 0.
Kernel route input: φ = [cos(u, v), RFF([u, v, u⊙v]) (D 1 024, bandwidth ∈ {0.5, 1, 2} × FIT median distance), 1].

## Routes
raw (cosine, no training); **VCS** (1 − mean_G J_G); **matched JS** (mean_P softplus(−2f) + mean_Q softplus(2f)); **candidate softmax**
(multi-positive CE over the image's referred candidates; task panel only); **SigLIP-style control** (per-pair sigmoid on u = 2f, every cell of the
image equally weighted, π_G = #pos / #cells recorded; common J read with T = tanh(f − ½ logit π_G)); **RFF ridge-tanh** (weighted ridge on ±1
targets under M = (P + Q)/2, λ ∈ {1e-4, 1e-2, 1} × scale, tanh scale c from a 25-point grid on CAL; two-stage fit, not the exact bounded-J optimum).

## Fitting and selection
AdamW, lr ∈ {1e-4, 5e-4, 2e-3}, weight decay 1e-4, 32 images per batch, ≤ 50 epochs or 3 000 updates, evaluation every 50 updates, patience 10
evaluations (both criteria); 3 init seeds per (route, N).  Two saved checkpoints per trajectory: **estimator_selected** = CAL max common J (VCS, JS,
SigLIP-style, RFF); **task_selected** = DEV max image-macro Top-1 (earliest tie); lr chosen per criterion; every candidate recorded.

## Units (CPU)
Table A: {VCS, JS, softmax, SigLIP-style} × N {1k, 4k, all} × 3 seeds × 3 lr (one job per route), RFF × N × 3 seeds (one job), raw (one job).
Sensitivity: concat F2 × {VCS, JS} × N all × seed 0.  VL1-11: {VCS, JS, softmax} × phrase-uniform law × N all × 3 seeds.

## Pre-stated reading (descriptive; no "VCS wins" target)
1. **Same posterior:** VCS − JS on DEV Top-1 and on CAL common J per N, paired by seed (95 % t interval); expected close (|Δ| < 0.5 point Top-1).
2. **Sample efficiency / learning value:** each learned route vs raw CLIP per N (paired by seed); "adds value" if the interval excludes 0 above;
   the package's practical rule "not worse than raw CLIP by > 1 point" is checked for every route.
3. **Estimator panel:** common J on CAL for VCS / JS / SigLIP-style (de-biased) / RFF at each N — the same target, fit-limited lower estimates.
4. **VL1-11 (theory prediction, O1 §3.4):** under the phrase-uniform law, prior-corrected Top-1 ≥ raw-f Top-1 for VCS and JS (paired by seed);
   reading "holds" if the mean difference ≥ 0 with the interval excluding a loss of > 0.2 point, else "not shown"; under the region-uniform law the
   correction is the identity (checked in every evaluation).
5. Secondary: all-objects Top-1 (referred + distractors) and same-category / long-expression strata, descriptive.
Not claimed: natural-image oracle or posterior MSE; official val / test numbers before the final pass.

## Decisions at the freeze
- Gate 1029643 (package concat F2): every learned route stayed at 45 % DEV Top-1 after the smoke budget vs raw 70 % → scorer made residual (F2r).
- Gate 1029645 (F2r): tests 30 / 30 (incl. "F2r at init = 2·cos − 1", prior-correction identity / recovery); VCS, JS, softmax, SigLIP-style and the
  phrase-uniform law all start at raw-CLIP ranking (70.3 % smoke DEV) with positive calibrated common J (VCS 0.019, JS 0.019); concat F2 runs.
- Smoke 1029647 (kernel route with [cos, RFF([u, v, u⊙v]), 1]): 69.4 % smoke DEV, CAL common J 0.036.
- Jobs: `slurm/vl1_10_run.sbatch <args>` (CPU), outputs/VL1_10; 8 jobs (4 neural routes, RFF, raw, concat sensitivity, VL1-11).
