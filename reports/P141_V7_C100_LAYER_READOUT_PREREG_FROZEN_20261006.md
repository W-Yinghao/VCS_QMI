# Pre-registration — P141: v7 V7-C100-LAYER-READOUT — CIFAR-100 coarse / fine / conditional readouts at four feature sites (layer3 / h / r / z) of frozen SSL encoders — FROZEN 2026-10-06T00:58:23Z 

Owner 2026-10-05: "查看这个压缩包和那个md文档，把这批次的分析也开始做" (v7 plan `VCS_SSL_Server_Plan_v7_CN.md` §4.4, batch A, analysis only).  Evaluation
only: no encoder is trained, no training data or label enters SSL, the official CIFAR-100 test file is never opened.

## 1. Question
P124 found that A-P3's CIFAR-100 gain over the recipe VCS is a coarse-specific gain at h (coarse +3.05, fine ≈ 0, conditional −0.27), and P127 found no
(a, κ) that improves the within-superclass readout without losing coarse accuracy.  Where along the frozen chain layer3 → h → r → z does each method
carry coarse, fine and within-superclass (conditional) structure, and do VCS (A-P3), matched JS, SimCLR, the recipe VCS and G2 differ in that profile?
The plan's cross-unit question is recorded but answered only descriptively once the inputs exist: "if ResNet-50's h improves the fine-grained readout
while the wider ResNet-18 projector (P137 V7-HEAD) does not, that is evidence for a larger backbone rather than a larger scorer".

## 2. Sites (code `src/vcs_vtask/layer_readout.py`; one forward pass per image, run's own clean normalisation, eval mode)
| site | definition | ResNet-18 / ResNet-50 dim |
|---|---|---|
| layer3 | torchvision layer3 output, fixed global average pooling | 256 / 1024 |
| h | encoder output after avgpool — identical to the h of every earlier readout (QC: max |h − P124-cached h|) | 512 / 2048 |
| r | the projector's unnormalised output p = g(h) | 128 |
| z | z = r / max(‖r‖, eps), the training's L2 map with the run's eps | 128 |
r → z is a deterministic compression (z is a function of r).  L2-normalised h is not a parent of r or z in the nested chain (r = g(h) reads the
unnormalised h) and is not a site.

## 3. Readouts (identical P124 rules at every site and for every method; no site- or method-dependent argument — tested)
Same FIT (45k) / selection (5k) images (P91 manifest, asserted against each encoder's manifest); same CIFAR-100 coarse / fine labels from the local
`train` file; tasks coarse 20-way, fine 100-way, conditional 5-way given the coarse label (one probe per coarse group; macro primary, overall
reported; never an unconditional 100-way accuracy); 100-way head decomposition and masked-fine-head conditional (from the recipe head) as in P124.
Default readouts: **recipe_raw (primary — P124's primary frozen linear rule)**, raw_std (standardised features, lr / wd chosen on the inner 80 / 20
split of the FIT labels only; scale-robust, needed because the sites differ in scale — z is unit norm), kNN (k 200, T 0.1).  `--readouts all`
reproduces P124's five readouts (≈ 2.5× the cost; open decision).  **Each site gets its own heads; the best site per model never replaces the h
(backbone) metric.**

## 4. Encoders (final epoch-800 checkpoints; COMPLETED only — others are skipped and read by a later resubmission of the same resumable job)
A-P3 C100 (P107 seeds 0–2, P120 addendum-1 seeds 3–4); JS-AP3 C100 (P120 seeds 0–4); SimCLR C100 (P91 seeds 0–2); recipe VCS C100 (P91 vcs_a5
seeds 0–2); G2 C100 (P111 seeds 0–2); tuned JS (3, 0.5) C100 (P129 seeds 0–2, seed 2 running at the draft); ResNet-50 A-P3 / JS-AP3 / SimCLR C100
(P130, seed 0, queued at the draft) — 25 encoders in total.  Float32 features cached under `outputs/P141_features/<run>/` (smoke caches separate).

## 5. Pre-stated reading (descriptive; P114 labels for the two frozen contrasts)
- Per method × site × task: mean ± sample sd over seeds for recipe_raw (primary), raw_std, kNN; seeds listed.
- **Seed-paired contrasts at each site and task (recipe_raw):** A-P3 − JS-AP3 (seeds 0–4) and A-P3 − SimCLR (seeds 0–2), mean and 95 % t interval;
  labels close |Δ| < 0.3, clear |Δ| ≥ 0.3 with the interval excluding 0, otherwise inconclusive.  raw_std and kNN contrasts reported alongside.
- Descriptive only: A-P3 − tuned JS (3, 0.5) (seeds 0–2); ResNet-50 − ResNet-18 per method at seed 0 (single seed, "single seed" label); the site
  profile layer3 → h → r → z per method (where coarse / fine / conditional accuracy rises or falls); with P137 (when available) the cross-unit
  backbone-vs-projector reading of §1.
- Not claimed: that a lower readout at a later site means information was deleted (a linear readout on a compressed site can fall for other
  reasons); any causal mechanism; anything about other datasets or about the official test set.

## 6. Gate (CPU) — filled from the job
Job 1023083 (`slurm/p141_gate.sbatch`): `tests/test_p141.py` (site shapes for ResNet-18 / ResNet-50, h = encoder forward, z = L2(r), r → z determinism,
task-loop symmetry and structure on toy data, aggregator labels / seed pairing, runner skips non-COMPLETED runs) + `tests/test_p124.py` regression;
CPU smoke of the runner on two encoders (subset FIT / selection, all four sites, default readouts) + aggregator.
**Result (exit 0):** test_p141 5 / 5, test_p124 7 / 7; smoke A-P3 seed 0 and SimCLR seed 0 (1500 FIT / 500 selection images, CPU): features 16 s,
≈ 100 s per encoder for 4 sites × 3 readouts; aggregator ran; QC max |h − P124-cached h| 3.3e-6 / 4.3e-6 (same h as every earlier readout).
**Observation for the freeze:** at r and especially z the fixed-lr recipe probe under-fits (smoke SimCLR z fine: recipe_raw 33.0 vs raw_std 47.2;
h: 46.4 vs 43.0) — the recipe rule was set for raw h and is scale-sensitive on unit-norm z, so cross-site comparisons should read raw_std
(standardised, inner-split-selected) as co-primary, or report recipe_raw at r / z with that caveat (open decision).

## 7. Cost and submission
One GPU bundle job (`slurm/p141_unit.sbatch`, RTX6000PRO / H100 / L40S, node51 / node52 / node60 excluded, 12 h limit, resumable).  Estimated from
P124 (443 s per encoder on L40S for five readouts at one site): ≈ 10–12 min per encoder at four sites with the default three readouts on L40S,
≈ 4–5 h for the 25 encoders (less on RTX6000PRO / H100); the P129 seed-2 and P130 encoders are read by a resubmission after they finish.

## Decisions at the freeze (main session)
Owner 2026-10-05 v7 plan ("把这批次的分析也开始做"), §4.4.  Gate 1023083: tests 5 / 5 + P124 regression 7 / 7; CPU smoke ran; h identical to P124's cache
(max |Δ| 4.3e-6).
1. Readout per site: recipe_raw stays primary at h (the backbone metric); because P124's fixed-lr recipe probe underfits unit-norm z (smoke: SimCLR fine
   z 33.0 recipe_raw vs 47.2 raw_std), **raw_std is co-primary for all cross-site comparisons** (layer3 / h / r / z), recipe_raw reported alongside.
2. Readout set: the default three (recipe_raw, raw_std, kNN); P124's full five not run.
3. One bundle GPU job now (22 available encoders incl. P129 JS (3, 0.5) seeds 0–2); the same line is re-run after P130 to add the ResNet-50 encoders.
