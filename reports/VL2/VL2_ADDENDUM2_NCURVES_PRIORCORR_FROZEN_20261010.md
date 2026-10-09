# VL2 addendum 2 — sample-efficiency curves for every Base feature set, and the VL1-11 prior-corrected ranking on RefCOCO / RefCOCO+ — FROZEN 2026-10-10

Owner 2026-10-09: complete experiments, keep the GPUs busy.  With the GPU fitter (identical results, 25× faster; `vl_gpu_fit_check.json`) these
are cheap.  Protocol unchanged (`VL1_FROZEN_PROTOCOL.md`, VL2).

## Units (GPU, `slurm/vl_fit_gpu.sbatch`)
1. **N curves:** VCS, JS, softmax at N ∈ {1 000, 4 000} × 3 seeds for every Base feature set not yet covered: RefCOCOg {SigLIP 2, FG-CLIP 2,
   FG-CLIP v1, ReCLIP features} and RefCOCO / RefCOCO+ {CLIP, SigLIP 2, FG-CLIP 2, FG-CLIP v1, ReCLIP features}.  CLIP on RefCOCOg exists (VL1-10).
   The nested FIT prefixes are those of each dataset's roles (RefCOCO / RefCOCO+ prefixes from `RG.fit_prefixes`).
2. **VL1-11 on RefCOCO / RefCOCO+:** phrase-uniform law (p(r) ∝ #expressions), VCS / JS / softmax at N all × 3 seeds, CLIP and SigLIP 2 features;
   reading = prior-corrected − raw-f Top-1 (O1 §3.4), the VL1-11 rule.

## Readings (descriptive)
- Per feature set: CAL J and DEV Top-1 vs N for VCS / JS; learned − raw per N; VCS − JS per N (same posterior).
- Whether the feature-set order by J (VL1-12 consistency) holds at every N.
- VL1-11 on two more datasets: holds / not shown, by the frozen VL1-11 rule.
