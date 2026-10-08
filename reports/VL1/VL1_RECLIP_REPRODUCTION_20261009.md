# VL1 Table B — ReCLIP reproduction on RefCOCOg (UMD) val, ground-truth boxes — 2026-10-09

| configuration | ours, IoU > 0.5 (ReCLIP's rule) | paper (App. Table 7) | ours, exact target |
|---|---|---|---|
| ReCLIP (parse: isolated proposal scoring + relations) | **68.20** | 68.08 | 65.99 |
| ReCLIP IPS-only (baseline) | **65.26** | 65.32 | 62.97 |

4 896 val expressions (= paper count); independent reproduction in the repo's issue #6: 67.91.  **The reproduction holds within 0.12 points.**
Setup (`vl1_reclip_val_reproduction.json`): official code allenai/reclip @ 3ed4f47 with one import guard (ALBEF, unused for CLIP runs;
`external/reclip_patched/PATCH.md`), inputs rebuilt from the REFER files because the official preprocessed data is no longer available, COCO-2014
names symlinked to the local COCO-2017 files, documented flags (RN50x16 + ViT-B/32 ensemble, crop + blur), spaCy 3.8 instead of 3.0.
Comparability notes for Table B: ReCLIP's candidate set is **all** COCO objects of the image (our secondary "all-objects" set) and it counts
IoU > 0.5 as correct (a non-target box overlapping the target counts); our Table A uses the exact target, so both rules are reported.  ReCLIP's
σ, prompt and relation heuristics were tuned on RefCOCOg val by its authors, so its val number is not held-out; our comparison uses the DEV role
(runs queued) and, at the final pass, the UMD test split.
