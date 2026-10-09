# VL1-15 addendum 1 (So400m tier) and VL2 addendum 1 (Large tier on RefCOCO / RefCOCO+) — FROZEN 2026-10-10

Owner 2026-10-09: keep 5–10 GPU jobs pending, complete experiments.  Both extend VL1-15 (frozen) without changing its protocol.

## VL1-15 add. 1 — So400m tier on RefCOCOg
SigLIP 2 So400m/14-384 (`google/siglip2-so400m-patch14-384` @ e8e4872) and FG-CLIP 2 So400m (`qihoo360/fg-clip2-so400m` @ d57d30f); same
extractors and checks.  Gate: the So400m check ≥ the Base check of the family (SigLIP 2 ImageNet 75.63; FG-CLIP 2 COCO box 64.77).  Units after the gate:
raw, VCS, JS, softmax, RFF at N all × 3 seeds.  Reading: Base → Large → So400m ladder per family (where gates pass), CAL J and DEV Top-1.

## Gate record so far
- FG-CLIP 2 Large (job 1032157): COCO box top-1 **55.06** < Base 64.77 → **FAIL** → no FG-CLIP 2 Large fits (cache exists, unused).
  Reported, not tuned around (no processor / resolution search to make it pass).

## VL2 add. 1 — Large tier on RefCOCO / RefCOCO+
Feature caches (GPU jobs 1032296–301): CLIP ViT-L/14@336, SigLIP 2 L/16-256, FG-CLIP v1 Large on both datasets (same extractors, dataset switch).
Fits (Table A units as VL2) only for the models whose VL1-15 gate passes.  Reading: Large − Base per dataset, family and route; the VL2 probe
consistency (J order vs learned Top-1) over Base + Large feature sets.
