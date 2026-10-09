# VL1-12 addendum 2 — Table A on SigLIP 2 Base crop features (estimator as a probe) — DRAFT 2026-10-09

Planned in `EXPERIMENT_ORDER_20261008.md` B3 (CLIP vs FG-CLIP 2 vs SigLIP 2).  Same protocol as `VL1_FROZEN_PROTOCOL.md` and the VL1-12 addendum
(roles, region-uniform law, scorer F2r with the feature width 768, routes, lr grid, selection, readings); only the feature cache changes.

## Features
SigLIP 2 Base, `google/siglip2-base-patch16-224` @ 75de2d5 (Apache-2.0; provenance file next to the weights).  No region interface exists, so the
region input is the tight clipped box crop (as the CLIP cache) through the model's own image processor (squash resize 224, mean / std 0.5) —
labelled an adaptation.  Text: lower-cased expression, padding "max_length" 64 (model card).  `scripts/vl1_12_siglip2_features.py cache` →
`features_siglip2_base/train_side_features.pt` (fp16, CLIP-cache key layout).
Implementation check (gate): ImageNet-1k val zero-shot top-1 on 10 images per class, prompt "this is a photo of {name}." with the open_clip class
names; the SigLIP 2 paper reports 78.2 for B/16 at 224 with its own names / prompts → pass if within 3 points; otherwise stop and report.

## Units (CPU, `slurm/vl1_10_run.sbatch` with VL1_OUT=VL1_12_siglip2)
raw, VCS, JS, softmax, RFF at N = all × 3 seeds.

## Reading (frozen at freeze; descriptive)
1. Per route, SigLIP 2 − CLIP on CAL common J and DEV image-macro Top-1, paired by seed (as VL1-12).
2. FG-CLIP 2 − SigLIP 2 per route, paired by seed: FG-CLIP 2 is initialised from SigLIP 2, so this contrast carries FG-CLIP 2's fine-grained
   training **and** its RoIAlign region interface (vs a crop) — confounded by construction; stated, not separated.
3. VCS − JS on SigLIP 2 features (same-posterior check).
4. Table B row: raw SigLIP 2 cosine on the referred candidates (ranking by sigmoid(s·cos + b) equals ranking by cosine).
Caveat carried (W11): fitted J across feature sets compares estimation difficulty as well as relation strength; the RFF route already disagreed
with the neural routes on the CLIP → FG-CLIP 2 direction.
