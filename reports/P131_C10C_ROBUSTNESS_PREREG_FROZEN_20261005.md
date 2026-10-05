# Pre-registration — P131: CIFAR-10-C-style corruption robustness of frozen SSL representations (regenerated locally; reporting only) — FROZEN 2026-10-05T03:04:41Z 

**Authorisation.** Owner 2026-10-04 "全部提交" in reply to the list that stated: "CIFAR-10-C … built from official test images, so it needs your
authorisation (reporting only)".  Scope of that authorisation as implemented: the official CIFAR-10 test images are read **only as inputs to
the corruption functions**; clean test images are never written to disk and never scored; no model, probe, hyper-parameter or checkpoint is
selected with these numbers.  The official clean test evaluation (P67 protocol) stays closed.

**Label.** "CIFAR-10-C-style, regenerated locally" — the released CIFAR-10-C files are not used (no downloads).  The corruption functions and the
CIFAR severity parameters are re-implemented from Hendrycks & Dietterich (2019), `hendrycks/robustness`, `ImageNet-C/create_c/make_cifar_c.py`,
with numpy / scipy / PIL only (`src/vcs_diag/c10c.py`).  Parameters were transcribed from that public script without access to it here — the
main session should cross-check them before freezing (listed in the module, one line per corruption).

## 1. Corruption set
- **Primary (12 of the standard 15):** noise — gaussian_noise, shot_noise, impulse_noise; blur — defocus_blur, glass_blur, zoom_blur;
  weather — fog, brightness; digital — contrast, elastic_transform, pixelate, jpeg_compression.  Severities 1–5, 10 000 images each.
- **Omitted (no faithful reproduction possible here):** motion_blur and snow (ImageMagick via Wand), frost (frost texture images of the original
  repository).  Extra set: speckle_noise, gaussian_blur, saturate reported separately; spatter omitted (cv2).
- **Exact re-implementations** of the missing library calls (skimage rgb2hsv / hsv2rgb / gaussian / random_noise s&p, cv2 GaussianBlur / filter2D
  with BORDER_REFLECT_101); **approximation:** elastic_transform's cv2.warpAffine → scipy affine_transform (no 1/32-px coordinate quantisation).
  The original's quirks are kept (glass-blur "swap" = copy; plasma fractal `wibble * uniform(-wibble, wibble)`; `np.uint8` truncation).
- RNG: `RandomState(131·1000 + 10·index + severity)` per (corruption, severity) block — reproducible and order-independent; same distribution
  as the released set, not the same pixels.  Stored under `outputs/P131_c10c_regen/` with a manifest (parameters, seeds, versions, sha256).

## 2. Models and readout (frozen final checkpoints, epoch 800, CIFAR-10 standard augmentation, development split dev45k / val5k)
A-P3 seeds 0–4 (P107), SimCLR seeds 0–4 (P41 seeds 0–2, P111 seeds 3–4), G2 seeds 0–4 (P104 seeds 0–2, P111 seeds 3–4), JS-AP3 seeds 0–2
(P114), recipe VCS seeds 0–2 (P35) — 21 runs.  Readout = the pilot frozen-h linear head (`vcs_ssl.diagnostics.linear_probe`, frozen probe
config, trained on the 45k fit split; `return_head=True` only returns the trained head).  Per run the clean development-validation accuracy is
recomputed and must equal the stored `evaluation_epoch_800.json` value (|Δ| < 0.05); the GPU feature fast path for the corrupted arrays
(uint8 → /255 → normalise, batch 512) must match the pilot feature path on the 5k selection split.  Clean dev-val is the reference for the drop.

## 3. Pre-stated reading
- **Primary:** mean corruption accuracy (mCA) over the 12 primary corruptions × 5 severities, per method (mean over seeds).
- **Paired by seed:** A-P3 − SimCLR (5 seeds) and A-P3 − JS-AP3 (3 seeds); mean, sd, 95 % t interval.  Labels as P114: **close** |Δ| < 0.3;
  **clear** |Δ| ≥ 0.3 and the interval excludes 0; otherwise **inconclusive**.  Secondary contrasts reported with the same labels:
  G2 − SimCLR, A-P3 − recipe VCS.
- **Secondary:** robustness drop = clean dev-val − mCA, same paired contrasts (separates "more accurate overall" from "degrades less");
  per-family breakdown (noise / blur / weather / digital) of both; extra set separately.
- **Not claimed:** CIFAR-10-C numbers comparable to published ones (regenerated, 12/15 corruptions); anything about training-time robustness
  (no model is trained or selected here); causes of any difference.  One pass; no re-run with other corruption parameters after seeing results.

## 4. Compute
Generation: one CPU job (16 cores; ~15–25 min).  Evaluation: one GPU job for all 21 runs, finished runs skipped (~1–2 min per run), then the
aggregate.  Smoke (disclosed): jobs 1021154 (generation, 200 images) / 1021155 (evaluation, 2 runs × 200 images per block) — filled in below.

## 5. Smoke (disclosed; not results)
- 1021154 (CPU, 200 test images, 15 workers): all 15 corruptions generated, 20 s wall; glass_blur 19.4 s / 1 000 images and zoom_blur 9.2 s
  dominate → full set (50 000 images per corruption) ≈ 16 min wall with 15 workers; ~2.3 GB on disk.
- 1021155 (RTX6000PRO, 200 images per block): A-P3 seed 0 — clean dev-val recomputed 89.06 = stored 89.06; fast path max |Δh| 1.6e-5, dev-val
  via fast path 89.06; 8.9 s per run at smoke size (full: ~40 s per run → 21 runs ≈ 15 min).  The second smoke run (SimCLR seed 0) was dropped
  because RUNS was passed inside `--export=ALL,…` and SLURM split it on the comma — the full job uses the default run list (no RUNS export).
- Unit tests `tests/test_p131_c10c.py`: 49 passed (shape / dtype, determinism, severity monotonicity where the parameters are monotone, helper
  equivalences, the glass-blur copy quirk, block-seed uniqueness, generator refuses without the authorisation flag + env).  Regression after the
  `linear_probe(return_head=…)` change: tests/test_ssl_core.py + tests/test_vcs_ssl_p114.py 44 passed.

## Decisions at the freeze (main session)
Owner 2026-10-04 "全部提交" after being told CIFAR-10-C is built from official test images — the authorisation for this one reporting-only pass.
Gate / smoke: tests 49 / 49 (+ 44 regression after the `linear_probe(return_head=False)` change); CPU smoke 1021154; GPU smoke 1021155 (A-P3 seed 0
clean dev-val recomputed 89.06 = stored 89.06).
1. **Severity parameters could not be verified** against the original `make_cifar_c.py` (no local copy; no downloads).  The set is therefore used for
   *internal* comparisons between our encoders only — never compared with published CIFAR-10-C numbers — and every table says "CIFAR-10-C-style,
   regenerated locally, parameters transcribed (unverified)".
2. Primary set = the 12 faithfully reproducible corruptions (15 minus motion_blur, snow, frost); no approximate motion blur.  Extras (speckle,
   gaussian_blur, saturate) reported separately.
3. Readout: frozen-h linear head only (trained on dev45k, as the pilot protocol); no kNN.  Primary contrasts A-P3 − SimCLR and A-P3 − JS-AP3 (paired by
   seed, P114 labels); G2 − SimCLR and A-P3 − recipe VCS descriptive.
4. `linear_probe(return_head=...)` accepted (backward compatible, default unchanged, regression tests pass).
5. Clean official-test accuracy is never computed, written or reported; no model selection or tuning from these results.
